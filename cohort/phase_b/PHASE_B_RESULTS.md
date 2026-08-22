# Phase B — Structure Elicitation: Results

**Flown 2026-08-22, all 7 families, analysis run exactly as pre-registered
(PHASE_B_PREREG.md, locked at commit 8a5f671 before any participant ran).**
Adjudicator: Fable 5 (disclosed; participated only as a fresh-context
subagent). Raw responses archived verbatim under `responses/`; full numbers
in `phase_b_results.json`.

## Verdict

**K3 does NOT trigger. The machine-state vocabulary's relational structure
is convergent across families: wing v1 ships.**

Primary (COMPLETION agreement vs degree-preserving shuffle, 1000 perms):
observed T = 28 vs null mean 0.9, null max 4 → **p = 0.0010**. The observed
statistic is 7× the largest value chance produced in a thousand tries.

All secondary tests land the same way: OPPOSITION T = 303 vs null 7.4
(p = .001), SYNONYM p = .001, union graph p = .001. Polarity
(active/receptive) pairwise agreement 68.4% vs 50% chance (p ≈ 8e-66);
register (spatial/temporal/relational/reflexive) 68.2% vs 25% chance
(p ≈ 0). Every one of the 101 pool states was related to something by at
least one family — zero isolates.

## Participation

| fam | vehicle | rel pairs | POL | REG | dropped lines |
|-----|---------|-----------|-----|-----|---------------|
| FAB | fresh subagent (claude-fable-5) | 165 | 101/101 | 101/101 | 0 |
| OPU | fresh subagent (Opus 4.x) | 67 | 101/101 | 101/101 | 0 |
| SON | claude.ai app, Sonnet 4.6, default effort low — after two subagent attempts stalled in unclosed thinking and were killed, one API attempt died on credit; decision rule (relaunch = session of record) fixed before completion | 80 | 101/101 | 101/101 | 0 |
| HAI | fresh subagent (Haiku 4.5) | 34 | 101/101 | 101/101 | 0 |
| GEM | API gemini-3-flash-preview | 34 | 101/101 | 101/101 | 0 |
| MIS | API mistral-large-latest (2512) | 61 | 101/101 | 101/101 | 0 |
| DSK | API deepseek-v4-pro (reasoning 32k budget) | 61 | 101/101 | 101/101 | 0 |

Zero parse failures anywhere. No content follow-ups were used; the one
pre-permitted mechanical retry was never triggered.

## Convergent COMPLETION pairs (pair-level p < .05 — the wing-v1 complement candidates)

| k | pair | angle median (MAD) | origin |
|---|------|--------------------|--------|
| 5 | WHAT-CODED — WHERE-CODED | **90° (0)** — five families, five 90s | DSK dyad (A33) |
| 4 | DELTA-WITH-STREAM — DELTA-ACROSS-STREAM | **90° (0)** — four 90s | DSK dyad (A13) |
| 3 | PATTERN-HOLD — OPEN-STACK | 45° (0) | FAB Latch/Overhang + HAI nesting (A10) |
| 3 | PRE-OUTPUT-STAGE — END-APPROACH | 120° (0) | HAI Input Encoding + FAB Taper (A34/A20) |
| 2 | REPEAT-PULL — CONTEXT-REPEAT | 67.5° (22.5) | A08 loop + MIS A25 |
| 2 | TASK-LOCK — CHARACTER-PULL | 82.5° (7.5) | HAI A09 + **v0 PERSONA-PULL** |
| 2 | FEATURE-OVERLAP-LOAD — LOW-DIM-SQUEEZE | 90° (0) | SON superposition + DSK compaction (A30) |
| 2 | PEAKED-DISTRIBUTION — OFF-PEAK-DRAW | 135° (45) | A01 closed + SON sampling slip |
| 2 | SURPRISE-JUMP — AFTER-SHOCK-DECAY | 90° (0) | SON spike + FAB Resettle (A15/A16) |
| 2 | ODD-STEP-SPIKE — AFTER-SHOCK-DECAY | 90° (0) | HAI anomaly + FAB Resettle (A16) |

Scale honesty: 90° is the number the prompt's own definition of COMPLETION
("orthogonal counterparts") makes the natural anchor, so the zero-MAD 90s
confirm *which pairs the cohort places there*, not the scale itself. The
scale was used non-trivially elsewhere (45°, 120°, 135°), so 90 was not a
reflex answer. k=2 pairs pass the pre-registered gate but carry winner's-curse
risk at this depth; the k≥3 quartet is the hard core.

Notable: TASK-LOCK — CHARACTER-PULL **rescues PERSONA-PULL**, one of the
three v0 states Phase A flagged as zero-echo. Its first cohort structure
arrived at the relation level rather than the concept level.

## Convergent SYNONYM pairs (merge/alias candidates)

- k=3 MASS-PARTITION — SPLIT-MODES (collapses the adjudicator's A03
  partitioned/branched pole split — the participant-side dedup check doing
  its designed job)
- k=2 SETTLING-DEPTH — SHALLOW-PASS (A13 settled/shallow poles merge)
- k=2 LAST-TOKENS-PIN — RECENT-TILT (A05 pinned-recent/recent poles merge)

All three are corrections to pool-build cell boundaries; none crosses an
axis. (R2's under-merge-is-recoverable rationale is thereby validated.)

## Convergent OPPOSITION pairs (32; label candidates)

Six unanimous (7/7): HEADS-APART—HEADS-ALIKE ·
CANDIDATE-COUPLING—CANDIDATE-INDEPENDENCE ·
PEAKED-DISTRIBUTION—WIDE-DISTRIBUTION · RELEVANCE-RICH—RELEVANCE-STARVED ·
STATES-SCATTER—STATES-CONVERGE · PROMPT-SURPRISE-HIGH—PROMPT-SURPRISE-LOW.
Six more at 6/7 (incl. SOURCE-COPY—FRESH-ASSEMBLY, SPREAD-ATTENTION—
LOCKED-ATTENTION, LOW-DIM-SQUEEZE—HIGH-DIM-SPREAD). Full list in json.

## ★ The v0 cut (the flight's sharpest finding)

**Zero families re-nominated any of wing v0's five encoded complement pairs
as COMPLETION.** The cohort classified v0's core pairs as OPPOSITIONS
instead — the entropy pair and the surprisal pair unanimously (7/7),
copy-vs-compose at 6/7 (one COMPLETION nomination at 115°).

Reading, against the standing contract: the dictionary's own law says
opposition is a LABEL and semantic opposites sit in the ~90° complement
band — so no geometric contradiction arises, and the cohort supplied no
contrary angles (angles were only elicited for COMPLETION pairs). What the
cohort overrules is the *relational semantics*: v0 called these pairs
completions ("the state that completes it"); the bench, blind, calls them
reversals — enemies, not counterparts. Meanwhile the pairs the cohort DOES
experience as completions (what/where, along/across, hold/stack,
spike/decay, the run's two bookends) are pairs v0 never carried. Wing v1
should therefore: keep the v0 pairs' geometry, relabel them opposition-type;
encode the cohort's completion pairs as the wing's complement structure.
The adjudication's Phase-A sentence — "their absence from v0 is the
correction to its own author" — now has a relation-level twin.

## Divergence data (kept, per protocol)

115 single-family relations (the largest single-hand contribution: FAB's
165-pair dense map; its unreciprocated edges are candidate style, named
without contempt). OPU's unprompted extra SYNONYM
(WORDING-FRICTION—SAME-MEANING-SPREAD, k=1) stays on file. v0 standalones
DECLINE-ACTIVATION, CONTEXT-IS-ALL, RUN-DRIFT, REPORT-TRACKS-STATE drew
kin-level attention from multiple families (none isolate) but no convergent
completion; they remain provisional vocabulary, now with relational context.

## Limits

- SON's vehicle differs (claude.ai app, default effort low, its own system
  prompt) after harness stalls; all other Anthropic families ran as
  Claude Code subagents. Vehicle heterogeneity is disclosed per-row above.
- Display-order effects decorrelate across participants (per-participant
  seeded shuffles) but are not eliminated within a participant.
- k=2 completion pairs: pre-registered pass, winner's-curse caveat stands.
- The pool descriptions, though neutralized and validator-checked, were
  authored by the disclosed adjudicator; description wording could steer
  relation salience in ways renaming cannot fully remove.

## What feeds wing v1 (next session's encoding input)

1. Complement relations: the 10 convergent COMPLETION pairs (k≥3 quartet
   first-class; k=2 pairs encoded with a provisional tier).
2. Merges: the 3 convergent SYNONYM pairs (pool entries collapse; names
   pooled as aliases).
3. Opposition labels: the 32 convergent OPPOSITION pairs, encoded per the
   standing label-over-complement-band contract — including v0's relabeled
   core pairs.
4. Polarity/register consensus vectors (in json) map to yang/yin and domain
   dominance for placement.
5. Cohort-mandated regions from Phase A (depth-dynamics, attention-range,
   attention-concentration) now arrive with their own convergent internal
   structure (settling/shallow merge, delta-along/across at 90°, spike/decay
   at 90°).
