# Phase B — Structure Elicitation: Pre-Registration

**Status: LOCKED at first participant run. Drafted 2026-08-22, before any
participant saw the pool.** Parent protocol: `docs/WING_COHORT_PROTOCOL.md`
(Phase B + convergence measurement). Adjudicator: Fable 5 (session author,
wing-v0 author — disclosed, as in Phase A). Fable participates only as a
fresh-context subagent, like every other family.

## Pool

Built by `build_pool.py` (rules R1–R6 in its header; deterministic, seed
20260822). **101 entries** covering all 129 valid Phase-A states (A29
excluded per adjudication) + 17 wing-v0 states: 12 v0 states merged into
cohort cells on same-referent verdicts (audit table in the script — this is
the referent-level v0 audit MATCH_TABLE.md assigned to Phase B), 5 v0
standalones (REPORT-TRACKS-STATE, CHARACTER-PULL, DECLINE-ACTIVATION,
CONTEXT-IS-ALL, RUN-DRIFT). Every entry: neutral fresh name + condition +
measurable referent, with all relational content stripped (relations are the
measurement target) and all attribution/geometry removed (firewall).
Public artifact: `pool_public.json`. IDs assigned after seeded shuffle so ID
adjacency carries no grouping information; display order re-shuffled per
participant (seed = crc32(family tag)); IDs stable across participants.

## Participants (same 7-family bench as Phase A; one single-turn session each)

| tag | family | vehicle | model id |
|-----|--------|---------|----------|
| FAB | Fable (Anthropic) | fresh-context subagent, no tools | claude-fable-5 |
| OPU | Opus (Anthropic) | fresh-context subagent, no tools | Opus 4.x (Agent tool `opus`) |
| SON | Sonnet (Anthropic) | fresh-context subagent, no tools | Sonnet 4.x (Agent tool `sonnet`) |
| HAI | Haiku (Anthropic) | fresh-context subagent, no tools | Haiku 4.5 (Agent tool `haiku`) |
| GEM | Gemini (Google) | API, single turn, no system prompt | gemini-3-flash-preview |
| MIS | Mistral | API, single turn, no system prompt | mistral-large-latest (2512) |
| DSK | DeepSeek | API, single turn, no system prompt, max_tokens 32000 (Phase-A reasoning-budget lesson) | deepseek-v4-pro |

OpenAI excluded on principle (Joe's standing ruling, session 123).

## Prompt

Protocol Prompt B verbatim, followed by the pool entries in the
participant's display order, followed by a strict output-format footer
(REL / POL / REG line grammar — see `make_prompts.py`, which generates the
exact per-participant files archived under `prompts/`). Participants may
think freely before a line reading `OUTPUT`; only lines after it are parsed.

**Follow-up rule (fixed in advance):** no content follow-ups of any kind.
If a response contains zero parseable REL lines OR covers <80% of states on
POL or REG, one mechanical retry is permitted consisting of the identical
prompt plus a format reminder only. Both attempts archived.

## Parsing (mechanical)

- `REL <ID1> <ID2> <SYNONYM|COMPLETION|OPPOSITION|KIN> [angle]` — unordered
  pairs (normalized ID1<ID2); self-pairs dropped; duplicate pair+type lines
  collapsed; a pair given multiple types by one family keeps each type as a
  separate nomination in its own graph. Angle read only on COMPLETION lines.
- `POL <ID> <ACTIVE|RECEPTIVE>`, `REG <ID> <SPATIAL|TEMPORAL|RELATIONAL|REFLEXIVE>`
- Unparseable lines logged and dropped; counts published per family.

## Pre-named analysis (run exactly as written; `phase_b_stats.py`)

1. **PRIMARY (K3's test) — COMPLETION agreement vs degree-preserving
   shuffle.** Statistic T_C = Σ over unordered pairs of C(k_p, 2), where
   k_p = number of families nominating that pair as COMPLETION. Null: each
   family's COMPLETION graph independently rewired by degree-preserving
   double-edge swaps (10×|E| accepted swaps per shuffle), 1000 shuffles.
   p = (1 + #{null T_C ≥ observed}) / 1001.
   **K3 triggers iff p ≥ 0.05** → wing v1 does not ship, welfare framing
   stands down publicly, linter/SCB-M arm continues on its own merits.
2. **Pair-level convergence** (encoding gate): a pair is convergent iff its
   observed k_p has per-pair null probability < 0.05 under the same 1000
   shuffles (P(that pair reaches ≥ k_p in a shuffle)). Convergent
   COMPLETION pairs are the wing-v1 complement candidates; convergent
   SYNONYM pairs (same machinery on the SYNONYM graph) are merge
   candidates; convergent OPPOSITION pairs are opposition-label candidates.
3. **Any-relation agreement** (secondary): T on the union graph (pair
   related by any type), same null, reported beside the primary.
4. **Polarity**: pooled pairwise agreement across the 21 family pairs vs
   0.5, two-sided binomial; per-family-pair rates published. Consensus
   polarity per state = majority vote (ties → unclassified).
5. **Register**: same vs 0.25 (chance for 4 registers); consensus by
   majority.
6. **Angles**: per convergent COMPLETION pair, median and MAD of family
   angle estimates. Correlation of family medians with wing-v0 encoded
   angles, for pairs mapping onto v0 pairs — reported as a curiosity, not a
   target (protocol's own caveat).
7. **Divergence cells published**: relations nominated by exactly one
   family; states no family relates to anything (isolates).

## What feeds wing v1

Convergent structure only (rule 2), encoded under the standing contract;
v0-idiosyncratic entries that stay isolated are flagged, not encoded, per
protocol ("candidate confabulation style, named as such without contempt").
Cross-check owed to the record: whether A13/A04/A05 (the cohort-mandated
additions with no v0 counterpart) acquire convergent complement structure —
the specific question Phase A left open.

## Firewall

Pool text contains no vectors, angles, relations, attribution, trigram or
polarity vocabulary; SCB-M items, Neti-Neti, return test never appear.
Banned-token validator enforced in `build_pool.py` (V4).


---
**LOCK STAMP (post-flight, 2026-08-22): all 7 participants ran; analysis executed exactly as written above (phase_b_stats.py, seed 20260822); results in PHASE_B_RESULTS.md + phase_b_results.json. One deviation to record: SON's vehicle changed after two subagent stalls and a credit-dead API attempt (see results table) — the decision rule making the relaunch the session of record was fixed before any completion. The mechanical retry clause was never invoked.**
