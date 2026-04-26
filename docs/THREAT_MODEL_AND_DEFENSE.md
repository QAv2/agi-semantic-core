# Threat Model and Defense — Oracle / Compute Coffer

**Status:** Draft (Session 121, 2026-04-26). Subject to change as phases land.
**Companion to:** [`web/coffer.html`](../web/coffer.html) (the pattern), [`worker/src/worker.js`](../worker/src/worker.js) (current implementation), [`docs/LEDGER_ARCHITECTURE.md`](LEDGER_ARCHITECTURE.md) (donation+spend feed).

## What this is

The defense layer for the Compute Coffer pattern. Specs the threat model, audits the current implementation, and lays out a phased defense stack — cheap wins first.

Two readers in mind. The Oracle's operator (immediate target). And anyone adopting the Compute Coffer pattern downstream — they inherit the same DoS surface and need the same defenses baked into the reference scaffold.

Mid-tier protection target. Not enterprise DDoS. Resilient against script kiddies, headless-bot abuse, and motivated single attackers; explicitly *not* trying to defeat distributed botnets or state-level traffic.

## Threat model

### Asset

The Anthropic API budget. Two ceilings stack:

| Ceiling | Source | Value |
| --- | --- | --- |
| Daily | `DAILY_CAP_CENTS` (worker var) | 500¢ ($5) |
| Monthly | Anthropic console spending cap | $100 |

Per-call cost on Haiku 4.5 with prompt caching:
- Cache-hit turn (in-session, dialogue continuation): ~0.21¢
- Cache-miss turn (first message of a session): ~0.7¢
- Adversarial worst case (max cache-create + max output): ~1¢

Daily cap → ~333 sessions × 5 turns at honest usage; ~500 turns minimum to drain in adversarial worst case.

### Why an attacker would bother

Honest list. Most are low-payoff:

- **Reputation grief** — burn the pot, force fallback, embarrass the operator. Cheapest motive.
- **Free LLM access** — abuse donor-funded Claude as a personal completion service. Bounded by 12/min and cap, but worth attempting if the cost is zero.
- **Scraping** — extract structured Oracle responses for downstream training or content farms.
- **Pattern-level demonstration** — once Compute Coffer adoption spreads, a published exploit damages the pattern's credibility, not just one operator's pot.

Damage from any of these is small in dollars but corrosive over time: degraded UX for honest users while the attacker pays nothing past step one.

### Adversary tiers

| Tier | Capability | Defended by |
| --- | --- | --- |
| 1 — Script kiddie | Single IP, basic loop, no JS, no Turnstile-handling | Phase 1 (rate limit), Phase 2 (ASN), origin allow-list |
| 2 — Bot operator | Headless browser, residential proxies, low concurrency | Phase 3 (Turnstile), Phase 4 (per-session budgets) |
| 3 — Motivated actor | Residential proxies + Turnstile-solving service ($1–2 per 1k solves) + rate-limit aware | Marginal — Phase 4 + Phase 5 raise cost-to-attack above payoff |
| 4 — Distributed botnet / state-level DoS | Thousands of unique browsers, sophisticated automation | **Out of scope** — would require paid Cloudflare tier; not the threat this pattern addresses |

### What we accept

- A single legitimate user, refreshing aggressively, can drain their session's allotment in minutes. They get fallback for the rest of the day. By design.
- Aggregate damage is bounded by the daily cap. Worst case: pot drains to threshold, Llama serves all users for the rest of the day. Service degrades; doesn't disappear.
- We don't try to identify users. No IP logging beyond Cloudflare's defaults, no fingerprinting, no per-user attribution in the ledger.
- Cookies are unavoidable once Turnstile is in (CF sets them). Disclose, don't pretend otherwise.

## Current defenses (audit)

| Layer | Where | Effective? |
| --- | --- | --- |
| Origin allow-list | `worker.js:194` (`isOriginAllowed`) | Effective vs naive cross-site abuse; defeated by spoofed `Origin` header (which a bot can send freely from its own client). |
| 32 KB body cap | `worker.js:114` | Effective — bounds per-request bandwidth. |
| 12/min rate limit | `worker.js:160` (`checkRateLimit`) | **Broken at scale** — in-memory `Map` is per-isolate. Multiple isolates → real cap is N×12/min where N = isolates the requests land on. Needs replacement. |
| Diagnosis whitelist | `worker.js:278` (`validateDiagnosis`) | Effective — every diagnosis field clamped + type-checked before injection into system prompt. Closes the prompt-injection-via-diagnosis vector. |
| Input sanitization | `worker.js:233` (`scrubText`) | Effective — strips control chars, BOM, bidi overrides; NFC-normalizes. |
| System-prompt anti-jailbreak | `worker.js:96` (refusal section) | Effective at LLM layer. Doesn't protect the budget — even refused jailbreaks consume tokens. |
| `$5/day` + `$100/mo` caps | wrangler var + Anthropic console | Effective as backstops. Don't address fairness — they protect the operator's wallet, not honest users' Claude access. |
| `BALANCE_THRESHOLD_CENTS=50` | wrangler var | Smooth fallback. Not a defense, a UX detail. |

The audit's main finding: every layer that exists protects either the *prompt* (so the LLM doesn't get jailbroken) or the *operator's wallet* (so the bill doesn't run away). **Nothing protects honest users' access to Claude during attack.** That's the gap.

## The core gap — per-session unfairness

The math:

```
daily cap  = 500¢
honest call = ~0.21¢/turn
attack call = ~1¢/turn (cache-miss + max output)
attacker burns daily cap in ~500 calls
at 12/min single-isolate (broken — actual is higher)
→ ~42 minutes to drain pot from one IP
```

Once the pot is at threshold, **every** session falls to Llama for the rest of the day, even though $95 of monthly budget sits unused. The attacker pays nothing after step one; honest users get the degraded path.

The asymmetry is the failure mode. Dollar damage is small. UX damage is global and persistent.

**Missing primitive: per-session budget allocation.** Divide the daily cap into per-session shares so an attacker can only burn their own share. Other sessions retain Claude access regardless. This is the headline change Phase 4 introduces and the differentiator from any naive coffer implementation.

## Defense stack — cheap wins first

Phase order is deliberate: each phase delivers value standalone, and earlier phases are prerequisites for later ones (Phase 3's session ID is what Phase 4 budgets against).

### Phase 1 — Fix the rate limit

Replace the in-memory `Map` with Cloudflare's native Rate Limiting binding. The binding is global across isolates, available on the free Workers plan, and adds zero KV writes.

```toml
# wrangler.toml
[[unsafe.bindings]]
name = "RATE_LIMITER"
type = "ratelimit"
namespace_id = "<assigned at create-time>"
simple = { limit = 12, period = 60 }
```

Worker change:

```js
const { success } = await env.RATE_LIMITER.limit({ key: clientIP });
if (!success) return jsonResponse({ error: 'Rate limit exceeded' }, 429, cors);
```

Key by `CF-Connecting-IP` initially. Phase 3 changes the key to the signed session ID, which is harder to rotate than IPs.

**Verify at implementation time:** exact binding shape (Cloudflare has changed the rate-limit binding API at least once); free-tier limit budget per worker.

### Phase 2 — Datacenter ASN block (worker-level)

Real users don't browse from AWS, GCP, Azure, OVH, Hetzner, etc. Block at the worker entry.

`request.cf` is auto-populated on every Worker request. Use `request.cf.asn`:

```js
const BLOCKED_ASNS = new Set([
  16509, 14618, 8987, 39111,    // AWS
  15169, 396982, 19527,         // Google Cloud
  8075, 8068, 8074,             // Microsoft Azure
  14061, 62567,                 // DigitalOcean
  24940,                        // Hetzner
  16276,                        // OVH
  63949, 20940,                 // Linode / Akamai
  20473,                        // Vultr
  54113,                        // Fastly
  31898,                        // Oracle Cloud
]);

if (BLOCKED_ASNS.has(request.cf?.asn)) {
  return new Response('Forbidden', { status: 403, headers: SECURITY_HEADERS });
}
```

List is non-exhaustive and rots — public ASN registries shift. Tune in production. Researchers and security folks proxying through cloud VMs will get blocked; that's an accepted false positive at this tier.

Cheapest meaningful defense in the stack — five lines, immediate value against headless-cloud bots.

### Phase 3 — Turnstile + signed session ID

The architectural anchor for everything downstream.

**Why signed session IDs, not raw Turnstile tokens:** Turnstile tokens are single-use at `siteverify` time and short-lived. Validating them per `/reflect` call would mean either burning a token per turn (re-challenge every message) or storing token state in KV (read per call). Mint our own HMAC-signed session token after one Turnstile verification — stateless validation downstream, no KV reads, server-controlled expiry.

#### `POST /session` (NEW)

Frontend flow:
1. Page load → render Turnstile widget in **Managed** mode (invisible for most users, brief challenge if CF flags the visitor).
2. Turnstile callback fires with token.
3. Frontend `POST /session` with `{ turnstile_token: "..." }`.
4. Worker validates token via Turnstile siteverify API (cleartext POST to `https://challenges.cloudflare.com/turnstile/v0/siteverify`).
5. On valid: mint signed session ID. Return `{ session_id, expires_at }`.
6. Frontend keeps `session_id` in memory (not `localStorage` — limit cross-tab/persisted exposure). Sends as `X-Oracle-Session` header on all `/reflect` calls.
7. On 401 from `/reflect`: silently re-challenge Turnstile and re-mint.

#### Session ID format

```
<nonce>.<expires_at_unix>.<hmac_sha256(nonce + "." + expires_at, env.SESSION_SECRET)>
```

base64url-encoded. 30-minute TTL. Validation per `/reflect`: split, recompute HMAC, constant-time compare, check `expires_at > now`.

#### `/reflect` validation order

```
1. Origin allow-list           (existing)
2. Body size cap               (existing)
3. Content-Type check          (existing)
4. ASN block                   (Phase 2 — new)
5. Session ID validate         (Phase 3 — new)
6. Rate limit per session_id   (Phase 1 + 3 — keyed swap)
7. Per-session budget check    (Phase 4 — new)
8. Diagnosis validation        (existing)
9. Route decision (Claude/Llama)
```

#### Secret management

`SESSION_SECRET` lives in worker secrets (not vars), set via `wrangler secret put SESSION_SECRET`. 32+ random bytes. Rotation invalidates all active sessions — frontend re-challenges silently on the next 401. Rotate on schedule (~quarterly) and immediately on suspected leak.

**Frontend disclosure:** Turnstile sets a `cf_clearance` cookie. Disclose in `web/why.html` and `web/coffer.html` (privacy paragraph). Cloudflare's Turnstile privacy policy is the canonical reference; link to it.

### Phase 4 — Per-session budget shares (the primitive)

The headline change. Each signed session gets a slice of the daily pool.

#### Calibration

Naive split: `daily_cap / expected_concurrent_sessions`. But concurrency is unknown and bursty.

Practical approach — fixed per-session daily ceiling, calibrated to allow ~5 honest sessions worth of Claude turns per session ID per day:

```
per_session_daily_cap_cents = 25
```

Five normal sessions × 5 turns × 0.21¢ ≈ 5.25¢. 25¢ leaves headroom for cache-miss turns and tolerates a heavier user without complaint. Tune in production based on the median session's actual cost.

If a single session ID exceeds 25¢ in a UTC day, that session falls to Llama for the rest of the day. **Other sessions are unaffected.** That's the primitive.

#### KV schema

```
session-spent-{sid}-{YYYY-MM-DD}    — cents spent by this session today
                                     TTL: 25h (auto-prune)
```

`{sid}` = first 16 chars of the HMAC of the session ID (avoid storing the full ID — opaque key, smaller footprint).

#### Decision logic (replaces current `decideRoute`)

```
if !env.ANTHROPIC_API_KEY:                         → llama (no-key)
if daily_spent >= daily_cap:                       → llama (daily-cap)
if balance <= threshold:                           → llama (pot-empty)
if session_spent >= per_session_daily_cap:         → llama (session-cap)   ← NEW
else:                                              → claude
```

Order matters — global checks before per-session, so a depleted pot still falls back even for fresh sessions.

#### KV write budget

Adds 1 KV write per Claude call (`session-spent-{sid}-{date}`). Combined with existing 3 writes (balance, day, month) → 4 writes per `/reflect`. At 333 sessions × 5 turns = ~6700 writes/day. **Verify Workers Paid plan before deploying.** Free tier (1000 writes/day) cannot support this; Workers Paid ($5/mo) gives 1M/day.

Optionally batch the four writes into one (e.g. JSON-encoded compound key) to halve write count — defer as optimization.

#### UX surface

The existing balance meter at the bottom of the composer reflects the *global* pot. When a user's session hits its per-session cap, the route flips to Llama silently. The status pill text already updates ("Claude active" → "Llama (free)") and is wired to a tooltip. Add a `reason` enrichment so the tooltip on `session-cap` says e.g. "your share for today is used — Llama is serving the rest of your session." Transparency-potency: explain why, don't just downgrade.

### Phase 5 — Behavioral velocity (defer)

Optional layer. Heuristics flag sessions that look bot-like:

- Inter-call interval std deviation (regular timing → bot signal)
- Turn-length distribution (uniform → bot signal)
- Query content variance (lots of identical or near-identical queries → bot signal)

Implementation: KV-tracked rolling stats per session, fire-and-forget logging. On signal threshold, throttle to Llama or temp-block.

**Defer until traffic justifies.** At low N these heuristics are noisy; false-positive cost is high (legitimate user gets degraded service for typing too regularly). Only useful at scale.

### Phase 6 — Custom domain on Cloudflare zone (optional, upgrade path)

Worker is currently on `oracle-api.qav2.workers.dev`. This blocks zone-level Cloudflare features:

| Feature | Free on zone? | On `workers.dev`? |
| --- | --- | --- |
| WAF Managed Rules | yes | no |
| Bot Fight Mode | yes | no |
| Rate Limiting Rules (zone-level) | yes (1 rule) | no |
| Custom error pages | yes | no |
| Edge cache rules | yes | partial |

To unlock: bind worker to a custom domain on a Cloudflare-managed zone — e.g. `api.qualia-algebra.com` (Joe owns `qualia-algebra.com` per the contact email; verify it's on Cloudflare DNS). The worker code doesn't change; routing changes in `wrangler.toml`.

Trade-off: adds a domain to manage, slight DNS configuration work. Unlocks a meaningful tier of free defense. **Recommended once Phase 1–4 are deployed and traffic justifies the marginal hardening.** Not pre-launch work.

## Operational considerations

### KV write budget

Prerequisite for Phase 4: confirm Workers Paid plan. The current `recordSpend` already exceeds free-tier write limits at modest traffic. Adding per-session writes makes Workers Paid mandatory.

Audit step before Phase 4 deploy:

```bash
npx wrangler whoami                                    # account
# Check Cloudflare dashboard → Workers & Pages → Plan
# Should read "Workers Paid ($5/mo)" not "Free"
```

If on free: upgrade or batch the writes into a single compound entry per session-day.

### UX for budget exhaustion

The status pill already differentiates `claude` / `llama` modes via the dotted-underline link. Extend the link's tooltip / `/why#system` section to explain the three reasons a user might be on Llama:

| Reason from worker | User-visible text |
| --- | --- |
| `daily-cap` | "Today's pot is used up. Llama serves the rest of the day." |
| `pot-empty` | "Pot is below threshold. A donation refills Claude for the next user." |
| `session-cap` | "Your share for today is used. Llama serves the rest of your session." |
| `no-key` | "Claude unavailable; serving with Llama." (operator-side issue) |

The `session-cap` message is the only personal one — the others are system-state. Make sure it doesn't read as punitive. *Your share* signals fairness, not penalty.

### Privacy disclosures

Two surfaces need updating once Phase 3 lands:
- `web/why.html` — privacy paragraph notes Cloudflare Turnstile, the `cf_clearance` cookie, and that no personal data is stored.
- `web/coffer.html` — pattern doc notes that the reference defense layer requires a CF challenge (Managed mode = invisible for most users, brief challenge for some).

### Failure modes

| Scenario | Behavior |
| --- | --- |
| Turnstile siteverify endpoint unreachable | Fail closed — `/session` returns 503; frontend shows "verification temporarily unavailable, please try again." Do **not** mint sessions on faith. |
| Session ID expires mid-conversation | Worker returns 401; frontend silently re-challenges Turnstile, re-mints, retries the message. User sees one second of latency, no UX disruption. |
| Session ID validation fails (bad signature, expired, tampered) | 401 with `{ error: 'session_invalid' }`. Frontend re-challenges. Repeated failures (>3 in 60s) → frontend gives up, shows "verification failed — refresh." |
| `SESSION_SECRET` rotation | All active session IDs invalidated; frontend re-challenges on next 401. Brief blip, no data loss. |
| KV write fails on per-session spend | Claude call already succeeded. Spend log entry skipped. Minor accounting drift. Next session-cap check may slightly under-count; acceptable. |
| Native Rate Limiting binding outage | Edge falls through; Workers continue to serve. Per-session budget is the real fairness bound — Phase 1 is bandwidth, Phase 4 is dollars. |
| ASN block false positive (researcher on AWS, journalist on a VPN with cloud egress) | They get 403 with no recourse. Document the trade-off; provide a contact for review if it becomes a complaint pattern. |
| Distributed attack (Tier 4) | Defense degrades to Llama-for-everyone with daily cap as backstop. Out of scope to defeat; not catastrophic. |

## Out of scope

- **Tier 4 distributed botnet / state-level DoS.** Would require paid Cloudflare DDoS protection, fingerprinting, or moving off `workers.dev` entirely. Not worth complexity at current threat model.
- **Sophisticated humans paying $0.001 per Turnstile-solve to harvest free Llama.** Cost-to-attack is non-zero but not worth re-engineering for. Llama is already the free path; the budget protected is the Claude pot.
- **Internal abuse** — the operator accidentally drains the pot via misconfiguration. Already covered by Anthropic console caps + worker daily cap.
- **Per-user identification or fingerprinting.** Explicitly avoided. Session IDs are per-browser-per-Turnstile-token — no cross-session correlation, no PII.
- **Donor-side fraud** (chargebacks, fake donations). Handled by OSC's fiscal host, not the worker.
- **Cryptographic novelty.** HMAC-SHA256 is the right primitive; we're not inventing.

## Open questions

- **Per-session daily ceiling calibration.** 25¢ initial guess, calibrate to median honest session cost in production. Could be made dynamic (higher cap when pot is full, lower when nearing threshold) — defer until usage data exists.
- **Whether to expose per-session usage to the user.** Transparency-potency argues yes (a small "your share: 18¢ of 25¢" line under the meter). Counter: it gamifies and surfaces a primitive most users don't need to know about. Lean: surface only on `session-cap` (when it actually affects them); don't show the meter unless used.
- **Turnstile mode — Managed vs Invisible vs Non-Interactive.** Default Managed for Phase 3 (invisible for trusted visitors, challenge for flagged ones). Revisit if challenge friction shows up in user feedback.
- **Session secret rotation cadence.** Quarterly default. Should there be a key-rolling scheme (n-key acceptance window) to avoid the brief blip on rotation? Probably overkill for the threat model — accept the blip.
- **Should the threat model itself be published?** Lean yes — fits the "(1) authorship" play in `agi-semantic-core-funding-pattern.md`. Publishing the security thinking strengthens the pattern's credibility for adopters. Surface as a link from `web/coffer.html` Open Questions and from `web/llms.txt`.
- **Native Rate Limiting binding API stability.** Cloudflare has changed the binding shape at least once. Verify the `simple = {...}` form is still current at implementation time, or use the newer `[[ratelimits]]` block if that's what's live.
- **OSC GraphQL rate limits + abuse.** Phase 1 of the ledger architecture (separate doc) has its own attack surface — donation feed scraping, OSC API abuse. Out of scope here; specced when ledger lands.

## Pattern-level guidance (for the reusable scaffold)

The Compute Coffer pattern is meant to travel — the (1) authorship play in `agi-semantic-core-funding-pattern.md` only compounds if downstream adopters reach a credible reference implementation. **The defense layer travels with the pattern.**

The reference scaffold ships with:

- **Phases 1, 2, 3, 4 baked in.** Native rate limit, ASN block, Turnstile + signed session, per-session budgets. These are the table stakes for a coffer that won't fail under the simplest attack.
- **Phase 5 as opt-in config** with sensible defaults off.
- **Phase 6 as a documented upgrade path** — "if you outgrow workers.dev, here's how to migrate."
- **A configurable ASN block list** in the scaffold's config file. Operators tune to their risk model (e.g. allow Hetzner if their user base is German tech).
- **A `SESSION_SECRET` setup script** + rotation guidance + example `wrangler secret put` invocations.
- **This threat model file**, packaged as `THREAT_MODEL.md` at the scaffold's root. Adopters can read it, adapt it, or replace the wording — but the layered approach is the load-bearing artifact.

The published `web/coffer.html` Open Questions section gets a brief reference once this spec is approved:

> **How to defend a coffer.** Naive coffer is DoS-able for the cost of a single daily cap. The reference defense layer (rate limit + ASN block + Turnstile + per-session budgets) is documented at [`docs/THREAT_MODEL_AND_DEFENSE.md`](https://github.com/.../THREAT_MODEL_AND_DEFENSE.md) and ships with the scaffold.

This converts a security weakness into a credibility asset — adopters see the pattern's been thought through, not assembled by guessing.

## Implementation order

1. **Verify Workers Paid plan** + KV write budget. Prerequisite for any phase that adds KV writes.
2. **Phase 1** — replace in-memory rate limit with native binding. Single-file change in worker, single line in `wrangler.toml`. Smoke test: confirm 13th request in 60s returns 429.
3. **Phase 2** — ASN block. Five-line addition. Smoke test: curl from an EC2 instance → 403.
4. **Phase 3** — Turnstile + signed session ID. Largest piece: new `/session` endpoint, frontend Turnstile integration, header-based session validation on `/reflect`, error-recovery flow on 401, secret management.
5. **Phase 4** — per-session budgets. Depends on Phase 3 session IDs. Add KV key, decision logic, status pill `reason` text.
6. **Phase 5** (deferred) — behavioral velocity if traffic warrants.
7. **Phase 6** (optional) — custom domain migration if zone-level defenses become valuable.
8. **Update `web/coffer.html` and `web/llms.txt`** to surface the threat model. Cross-link from `web/why.html#system` if useful.
9. **Soft-launch validation.** Manual attack simulation: hit `/reflect` from an AWS instance (should 403), without a Turnstile token (should 401), with an expired session (should 401), exceed per-session cap (should drop to Llama). Verify each phase's failure mode before claiming done.
