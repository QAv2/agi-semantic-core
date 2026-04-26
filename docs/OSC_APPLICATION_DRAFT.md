# Open Source Collective Application — Draft Source Material

**Status:** Draft for Joe to copy when filling out the OSC application form.
**Submit at:** https://www.oscollective.org/apply (or https://opencollective.com/opensource/apply — both flows reach the same review queue).
**Goal:** Single-project collective for the Oracle / AGI Semantic Core, hosted by Open Source Collective, so contributions land in the project's collective balance and Anthropic invoices are paid as expenses from that balance — never transiting the operator's bank account.

---

## Pre-application TODO

These need to be resolved before submission, not discovered mid-application:

- [ ] **Add a `LICENSE` file to the repo root.** Current README declares informal terms ("source-visible, attribution required, no warranty"), which isn't OSI-recognized and won't pass OSC eligibility. Recommended: **MIT** (simplest, matches the transparency stance, donors will recognize it). Alternatives: **Apache-2.0** (adds patent grant, slightly heavier), **AGPL-3.0** (copyleft — only if you want to require derivative deployments to also be open). My read: MIT is right for this project. The pattern's "freely adoptable" claim is best served by the most permissive license.
- [ ] **Pick the public collective name.** Options: `Oracle by AGI Semantic Core`, `AGI Semantic Core`, `Oracle (Compute Coffer reference implementation)`. The most descriptive-yet-short option: **AGI Semantic Core**. The most public-facing: **Oracle**. The slug controls the URL; the name controls the header.
- [ ] **Pick the slug.** `qav2-oracle` matches Netlify and homepage convention. `agi-semantic-core` matches the GitHub repo. Either works; pick whichever you'd rather see at `opencollective.com/{slug}`.
- [ ] **Decide funding goal (optional).** OSC supports an explicit "goal" amount (annual or one-time). For the Compute Coffer model, a meaningful goal might be the annual API spend ceiling (~$1,200 = $100/mo cap × 12). Leaving it unset is also fine — the pattern doesn't depend on a goal.

---

## Field-by-field paste

### Collective name
**AGI Semantic Core** *(or "Oracle by AGI Semantic Core" if the longer form is preferred)*

### Slug
**qav2-oracle** *(matches Netlify; alternatives: `agi-semantic-core`, `oracle-semantic-core`)*

### Tags
`open-source`, `ai`, `llm`, `nlp`, `ontology`, `semantic-web`, `oracle`, `divination`, `claude`, `cloudflare`

### Repository
- Primary: https://github.com/QAv2/agi-semantic-core
- Mirror: https://codeberg.org/QAv2/agi-semantic-core

### License
*(See pre-application TODO — recommend MIT, requires `LICENSE` file in repo)*

### Project URL
https://qav2-oracle.netlify.app

### Short description (~200 chars)

> A consciousness-first semantic dictionary (3,033 hand-encoded concepts in 16D dual-octonion space) and Oracle — a diagnostic engine running geometrically rather than statistically. Free to use; funded by the Compute Coffer pattern.

### Longer description / About

> The AGI Semantic Core is a hand-encoded semantic dictionary where every concept has a deliberate position in 16-dimensional space, opposites are 90° complements (not 180° antiparallel), and meaning composes through quaternion multiplication. Built over a hundred sessions and validated against a purpose-built consistency benchmark (88.8% accuracy, 100% polarity-inversion detection).
>
> The Oracle is the application built on top: a user states a condition, the engine locates it geometrically, finds the 90° complement as "medicine," and projects the reading through three independent traditions — King Wen I Ching, the Toltec I Ching (after William Douglas Horden), and the Tarot Major Arcana. The geometric diagnosis runs in the user's browser via Pyodide; an optional Reflective Principle conversation step (Claude, via a Cloudflare Worker) reflects on the diagnosis with the user.
>
> The project funds its Claude API usage through the **Compute Coffer pattern** — a public running balance, donations that fund the next stranger's reading rather than the operator, and a graceful fallback to a free Llama tier when the coffer empties so the app never breaks. The pattern is documented at [/coffer.html](https://qav2-oracle.netlify.app/coffer.html) as a reusable model for any LLM-fronted project.

---

## Why apply to OSC

> The Oracle implements a funding pattern called the **Compute Coffer** — donations cover the API bill for the next user's reading, the operator never withdraws, and the app gracefully falls back to a free LLM tier when the coffer empties. The pattern's load-bearing claim is that the operator is never the custodian of donations. Open Source Collective is the only fiscal hosting model that makes this structurally true rather than aspirational: contributions land in the project's collective balance, and Anthropic invoices are paid as expenses from that balance — they do not transit the operator's bank account.
>
> The 10% host fee is well within the project's economics (per-session cost ~1.5¢; the absolute fee at current scale is noise). The transparent public ledger built into Open Collective is exactly the donation-side feed the Oracle's frontend will display. OSC's brand alignment with open-source ethics matches the Compute Coffer pattern's "freely adoptable" stance.
>
> The project is open source, public on GitHub and Codeberg, with full encoding contract and architecture documented. The author does not draw revenue from the Oracle and has no plans to. Funds raised are intended exclusively to cover Claude API costs, with surplus retained in the collective for the next user.

---

## Maintainer

- **Name:** Joe Van Horn
- **Email:** bigbuddha827@gmail.com
- **GitHub:** *(your GitHub username)*
- **X / Twitter:** [@jvanhorn_](https://x.com/jvanhorn_)
- **Personal site:** [joeyv23.neocities.org](https://joeyv23.neocities.org)
- **Sole maintainer.** No other contributors at this time. The project is solo-developed but openly documented (the "For AI agents reading this" section in the README is one example of the trust-layer work done in Session 118).

---

## Anticipated Q&A from OSC reviewer

**Q: Is this a single-developer project or a community project?**
A: Solo developer, but the artifact (the dictionary + the geometry) is meant to be a research substrate that other people can study, fork, and extend. The Oracle is one application built on it; the documented pattern (Compute Coffer) is also designed to travel.

**Q: How will funds be used?**
A: Exclusively to cover Anthropic API costs (Claude Haiku 4.5 calls for the Reflective Principle conversation layer). Spending capped at $5/day and $100/month at the Anthropic side; below those caps, the worker switches to a free Cloudflare Workers AI fallback (Llama 3.1 70B) when the coffer's balance drops below 50¢. Funds will not be used for personal compensation, infrastructure beyond the Anthropic bill, or anything else.

**Q: Why fiscal hosting rather than GitHub Sponsors?**
A: GitHub Sponsors funnels money to the maintainer's bank account, which the maintainer then spends on API bills. OSC's fiscal hosting structure means the funds are held by the collective from the moment they arrive — the maintainer submits the API invoice as an expense and the host pays it directly. For the Compute Coffer pattern's claim that "the operator never sees the money" to be structurally true rather than aspirational, the maintainer needs to literally not be in the chain of custody. OSC is the only model that provides that.

**Q: How are donor funds tracked publicly?**
A: OSC's built-in public ledger handles the donation side. The project's frontend will display a real-time feed combining OSC contributions (donation side) with the worker's spend log (Claude API costs). Architecture sketched at [`docs/LEDGER_ARCHITECTURE.md`](./LEDGER_ARCHITECTURE.md).

**Q: Is the project ready for fiscal hosting (i.e., shipped, used, has actual donors)?**
A: Yes. Live at qav2-oracle.netlify.app since Session 119 (2026-04-25). Currently routing donations through Buy Me a Coffee with manual top-up to a Cloudflare KV balance. OSC is the announced direction; this application is the next step toward making the structural claim true.

---

## After submission — what to expect

- **Review timeline:** typically days to a couple weeks for OSC's volunteer review. They check OSS license, project legitimacy, and fit.
- **If approved:** OSC creates the collective at `opencollective.com/{slug}`. The maintainer can then enable a contribute page, set a goal, customize the about/banner, and start receiving contributions.
- **If declined:** OSC will say why. Most common rejection reasons are missing/non-OSS license or the project being too dormant; both are fixable.
- **Once live:** redirect the Buy Me a Coffee link in `web/index.html` to the OSC contribute URL. Migrate the existing $15 BMaC balance to the collective if desired (manual transfer). Begin the worker integration described in `docs/LEDGER_ARCHITECTURE.md`.

---

## Notes for future-Joe / future-self

- The application narrative above is honest about the project's solo-maintainer status and small scale. OSC reviewers see a lot of solo OSS projects and approve them regularly — small + honest + actually-shipped beats inflated claims.
- The Compute Coffer paragraph is the load-bearing pitch. If asked to cut anything, that paragraph stays.
- Don't oversell the dictionary's research value to the OSC reviewer — they're evaluating fiscal-hosting fit, not academic merit. The technical paper at `docs/TECHNICAL_REPORT.md` exists if they want it; don't lead with it.
