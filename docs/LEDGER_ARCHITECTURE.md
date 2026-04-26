# Ledger Architecture — Compute Coffer Public Feed

**Status:** Draft (Session 120, 2026-04-25). Subject to change as the OSC integration lands.
**Companion to:** [`web/coffer.html`](../web/coffer.html) (the pattern), [`worker/src/worker.js`](../worker/src/worker.js) (current implementation).

## Goal

Wire a public, real-time donation+spend feed for the Oracle's Compute Coffer. The feed must:

- Show donations and Claude-call spends as they happen, interleaved by timestamp.
- Expose donor names only when the donor has opted in at the contribution platform — no extra de-anonymization.
- Survive the OSC API being unreachable without breaking the Oracle.
- Add zero new custodianship for the operator: donations remain in the project's collective balance at OSC, never transit the operator's bank.

## Source-of-truth model

Two independent ledgers compose the feed:

| Side | Source of truth | Why |
| --- | --- | --- |
| **Donations** | Open Source Collective (GraphQL API) | OSC holds the funds and the public ledger. The operator has no authority over the donation record — that's the structural integrity of the pattern. |
| **Spends** | Cloudflare KV (`ORACLE_COFFER`) | The worker writes a spend log entry per Claude call. The worker is the only thing that knows what the call cost. |

The current `balance` integer in KV becomes a **derived cache**, not the source of truth. Computed as `cumulative_donations_from_OSC − cumulative_spends_from_KV`, refreshed on a short TTL. If KV gets corrupted or drifts, the next OSC poll re-establishes the authoritative donation total. The operator cannot under- or over-state the balance — it's derived from public OSC data plus locally-tracked spend.

## Worker endpoints

### `GET /balance` (existing — semantics shift)

Currently returns `{ balance, mode, model, daily_cap }` from KV directly.

After integration: same shape, but `balance` is the derived value (cached in KV with ~60s TTL, refreshed from OSC + spend log on miss). `mode` and `model` are unchanged.

### `GET /ledger` (NEW)

Returns recent activity, interleaved by timestamp, newest first:

```json
{
  "events": [
    { "type": "donation", "ts": "2026-04-25T18:30:00Z", "amount_cents": 500, "name": "anonymous" },
    { "type": "spend", "ts": "2026-04-25T18:28:14Z", "amount_cents": 1, "engine": "claude" },
    { "type": "donation", "ts": "2026-04-25T18:21:03Z", "amount_cents": 1000, "name": "Sam K." },
    ...
  ],
  "balance_cents": 1487,
  "donations_total_cents": 2500,
  "spends_total_cents": 1013,
  "as_of": "2026-04-25T18:31:02Z"
}
```

- Default: 30 most-recent events.
- `name` is whatever OSC returns — donor's choice. Spends have no `name`.
- Composition: query OSC for last K donations, scan KV for last K spend entries, merge by `ts`, slice to N.

### `GET /donations` (optional — defer to v2)

Donations-only feed. Probably not needed in v1 if `/ledger` does the job; keep this in reserve in case the frontend wants a donor-focused widget separate from the activity feed.

### `POST /reflect` (existing — gains a side effect)

Per Claude call, after the Anthropic response, the worker now also writes a spend log entry to KV:
- Key: `spend-{ISO8601-ts}-{requestId}`
- Value: `{ amount_cents, engine, ts }`
- Pruned by a periodic worker cron (or on-the-fly during `/ledger` reads): keep last 30 days of spend entries.

## Frontend widget

**Where it lives:** clicking the existing balance amount in the status pill expands a small panel above the composer showing the last ~10 events. Compact rows: `◎ +$5.00 · anonymous · 2m ago` (donation) or `◇ −1¢ · Claude · 14s ago` (spend). Gold for donations (matches existing coffer-icon), text-dim for spends.

**Polling:** when the panel is open AND the tab is visible, frontend polls `/ledger` every 30s. Closed panel = no polling; hidden tab = no polling. This keeps OSC API traffic bounded regardless of how many users have the app open.

**Optional dedicated view:** `/ledger.html` page with the full feed (last 30 days) for users who want the long view. Out of scope for v1 unless the inline panel proves insufficient.

## Privacy defaults

- **Donor name display:** mirror what OSC returns. If donor chose "anonymous" at contribution time, feed shows "anonymous." If they entered a name, show that name. The worker never adds attribution OSC didn't provide, and never strips attribution OSC did provide.
- **Donor messages:** out of scope for v1. If OSC supports public donor messages and we surface them later, treat as opt-in: show only if the message field is non-empty AND the donor's name is non-anonymous.
- **No emails or PII** ever leave OSC's domain — the worker only consumes OSC's public GraphQL fields.
- **Spends are anonymous by design.** No user identifier touches a spend log entry. Just timestamp + amount + engine.

## Caching strategy

- **OSC GraphQL response cached in KV** (`osc-cache:donations`) with 60s TTL. Worker refreshes on cache miss, not per-request. At ~30s frontend poll intervals × N concurrent users, OSC sees ≤1 request per minute.
- **Spend entries are written synchronously** to KV per Claude call (already happens for balance decrement; just adds the log key).
- **Balance derived value** cached 60s alongside the OSC response. Stale balance during cache window is acceptable — it's at most 60s behind reality.
- **`/ledger` response itself** can be cached at the Cloudflare edge for 15-30s, since it's public and identical for all viewers. Use `Cache-Control: public, max-age=30`.

## Failure modes

| Scenario | Behavior |
| --- | --- |
| OSC API unreachable | `/ledger` returns spend-side events only, with `donations_partial: true` flag. Frontend shows banner: "Donation feed temporarily unavailable. Spends still tracked." Balance falls back to last-known cached donation total minus current spend total. |
| OSC API returns malformed data | Treat as unreachable. Log to worker's error tail. |
| KV write fails during spend | Claude call still completes (response already streamed to user). Spend log entry skipped — minor data drift acceptable. Next `/balance` query may briefly show stale balance. |
| OSC and KV diverge | OSC is authoritative for donations, KV for spends. Reconciliation is automatic on next `/ledger` poll — no manual intervention. |
| Donor opted out of public name | Shown as "anonymous" — same as OSC's display. |
| Worker rate-limited by OSC | Cache TTL is the throttle; if rate limit still hit, treat as unreachable. |

## Out of scope (v1)

- **Donor messages / comments.** If OSC surfaces them and feedback warrants, add in v2.
- **Cumulative running-total chart / graph.** Feed only, no time-series viz.
- **Per-call cost breakdown by engine** (cache_read vs cache_write tokens, etc.) in the public feed. Internal accounting only.
- **Server-Sent Events / WebSocket push.** Polling is fine at low volume; push is a v2 optimization if scale warrants.
- **Donor leaderboard / top contributors.** The pattern is "donated forward," not "look how generous you are." Avoid building anything that turns the feed into a status game.
- **Per-user session attribution.** Spends are anonymous. The pattern doesn't need to know who used what.

## Implementation order

1. **Apply Oracle to OSC.** Submit the project for fiscal hosting under a single-project collective. Joe's homework — application processing is dead time.
2. **OSC integration scaffolding.** While waiting for approval: write the GraphQL fetch + KV cache layer in the worker against OSC's public API (a similar collective can stand in for testing). No deploy yet.
3. **Spend log writes.** Add the per-call `spend-{ts}-{id}` KV writes to `/reflect`. Backwards-compatible — `/balance` still works the old way until the new derivation lands.
4. **`/ledger` endpoint.** Composes the feed. Behind a feature flag initially so the frontend can be wired without going live.
5. **Frontend widget.** The expandable panel + 30s polling. Behind the same feature flag.
6. **OSC approval lands.** Wire the BMaC link to redirect to the new OSC contribution URL. Flip the feature flag. Ship.
7. **Update `web/coffer.html` "Donations" paragraph** to drop "application in progress" — replace with the live donate URL.
8. **Soft-launch.** Test with 2-3 small donations end-to-end. Verify the feed renders correctly across themes, the `donations_partial` failure mode degrades cleanly, and donor name privacy is faithfully respected.

## Open questions worth resolving before code

- **Does OSC support outbound webhooks** for new contributions, or is GraphQL polling the only option? Webhooks would let the worker react in seconds rather than within a 60s window. Polling is fine for v1; webhook upgrade is a clean v2 win if available.
- **OSC's GraphQL rate limits** — verify exact numbers during integration. The 60s cache should be well within tolerance, but worth confirming.
- **Currency handling.** OSC supports multiple currencies; the Compute Coffer is denominated in USD. Decide: convert non-USD donations at contribution time (OSC may already provide USD-equivalent), or always show original currency. Probably convert to USD for the feed display to match the coffer's unit.
