# Seed evidence

Receipts for the operator-side seed that backstops the Oracle's compute pool.

## Why this exists

The Compute Coffer pattern claims donations fund the next user's reading, with a graceful Llama fallback when the pot empties. That works only if there is something in the pot to begin with, and only if the operator has somewhere to draw from when the pot is dry but the daily cap isn't reached.

This directory documents both layers:

- **Upstream** — the operator's prepaid credit at `console.anthropic.com`, which pays Anthropic for every Claude turn the worker serves. This is the operator's own money, spent on public compute.
- **Downstream** — the seed balance written to the production KV (`d2859b487d1340538e312e2d3ba25ebe`), which is what `/balance` reports and what `coffer.html` describes as "donated forward." The first $15 of that is operator-paid kickoff capital, not a donation. See the Pot provenance section in cross-session memory.

When the OSC ledger goes live, the operator-paid kickoff is `seed`, not `donation`. The screenshots here are the audit trail.

## What goes in here

- Dated PNGs of the relevant Anthropic console pages (Billing / Credits, API Keys, Usage). Filename convention: `YYYY-MM-DD-<page>.png`.
- One screenshot per significant moment: the initial seed, any top-up, any rotation of the `oracle-prod-worker` key.
- No API keys or secrets in any image — crop or redact before saving. The point is the *balance and the date*, not the credentials.

## Index

- `2026-04-26-anthropic-credit-balance.png` — pre-Oracle credit pool ($41.66) as of the same day the four-layer defense stack and `/stats` endpoint shipped. Establishes what was available to backstop the public $15 coffer at the moment usage telemetry began being collected.
