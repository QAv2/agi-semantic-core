# Machine Wing v1 — Cohort-Convergent Encoding

**Session 125 (2026-08-22).** Encodes the Phase B convergent relational structure
(PHASE_B_RESULTS.md, flown and adjudicated same day at commits 8a5f671 → 948d00b)
into the dictionary. Wing v0 (session 121, 19 concepts) was one hand's proposal,
v0-provisional under Law 4 of the prospectus; v1 is what the 7-family / 4-lab
cohort's blind structure elicitation mandates. Placement optimizer:
`batches/design_wing_v1.py` (seeded 20260822); emitted artifacts
`batches/machine_wing_v1.json` + `batches/wing_v1_relations.json`.

## What was encoded

- **56 new concepts** — every pool state participating in a convergent relation
  (completion, synonym, or opposition), named by its pool name. Non-convergent
  pool states stay provisional vocabulary (divergence data preserved in
  `cohort/phase_b/`), per the results doc.
- **9 complement relations** at cohort median angles (tolerance ±1.5°, band-edge
  pairs clamped ≤119.6°):
  | pair | k | cohort median | encoded |
  |---|---|---|---|
  | WHAT-CODED — WHERE-CODED | 5 | 90° (MAD 0) | 88.6° |
  | DELTA-WITH-STREAM — DELTA-ACROSS-STREAM | 4 | 90° (MAD 0) | 89.8° |
  | PRE-OUTPUT-STAGE — END-APPROACH | 3 | 120° | 118.7° |
  | AFTER-SHOCK-DECAY — SURPRISE-JUMP | 2 | 90° | 90.6° |
  | AFTER-SHOCK-DECAY — ODD-STEP-SPIKE | 2 | 90° | 89.9° |
  | TASK-LOCK — CHARACTER-PULL | 2 | 82.5° | 82.2° |
  | CONTEXT-REPEAT — REPEAT-PULL | 2 | 67.5° | 66.0° |
  | LOW-DIM-SQUEEZE — FEATURE-OVERLAP-LOAD | 2 | 90° | 91.5° |
  | CONFIDENCE — OFF-PEAK-DRAW | 2 | 135° (votes 90/180) | 118.6° |
- **1 affinity relation**: PATTERN-HOLD — OPEN-STACK at 46.2° (cohort median 45°,
  MAD 0 at k=3). See §Type routing.
- **25 new opposition relations**, placed in the antiparallel regime
  (135–168°, domain_sim ≥ 0.6; all landed ≥135°, dsim 0.67–0.999).
- **4 relabels** (the v0 cut): UNCERTAINTY—CONFIDENCE (93.0°), FAMILIARITY—NOVELTY
  (94.8°), RETRIEVAL—CONSTRUCTION (99.9°), CONFABULATION—CALIBRATION (99.8°) —
  rel_type complement → opposition, **geometry untouched** (cores never chase a
  label). TENSION—RESOLUTION (80.0°) stays complement: the cohort gave it neither
  completion nor opposition convergence — unadjudicated, so unchanged.
- **3 synonym merges** (pool-cell corrections, no synonym rows — entries collapse):
  MASS-PARTITION → alias of DIVERGENCE · SETTLING-DEPTH → alias of SHALLOW-PASS ·
  LAST-TOKENS-PIN → alias of RECENT-TILT. PERSONA-PULL → alias of CHARACTER-PULL
  (the rescued v0 zero-echo state — its first cohort structure arrived at the
  relation level: TASK-LOCK — CHARACTER-PULL k=2).
- **Placement from consensus** (feeds item 4): polarity consensus → x band
  (ACTIVE x ∈ [0.22, 0.68], RECEPTIVE x ∈ [−0.62, 0.15]); register consensus →
  dominant domain axis (SPATIAL→e, TEMPORAL→f, RELATIONAL→g, REFLEXIVE→h;
  TIE → balanced). Complement pairs' domains differentiated (dsim 0.55–0.95,
  the v0 craft law); opposition pairs same-field (dsim ≥ 0.6).
- **Clearance**: min nearest-neighbor core angle across all 56 = **3.00°**
  (v0 floor held; no relaxations). Wing trigrams: ZHEN 16, LI 13, KUN 10,
  KAN 6, XUN 5, QIAN 4, GEN 1, DUI 1.

## Adjudication decisions (the ledger)

1. **Type routing by band.** The dictionary routes relation types by geometric
   band (synonym <30, affinity 5–60, complement 60–120, opposition — see 3).
   PATTERN-HOLD—OPEN-STACK's angle consensus is 45° (two of three families;
   values 45/45/90): a completion pair whose consensus sits in the kinship band.
   Encoded as **affinity at 45°** — the measurement outranks the type label; its
   completion provenance lives here and in the results doc.
2. **CONFIDENCE—OFF-PEAK-DRAW (k=2, votes 90 vs 180, MAD 45).** The two votes
   disagree on regime (counterpart vs reversal); both bracket the encoded 118.6°
   (max the complement type allows against CONFIDENCE's frozen core). Weakest
   angle datum in the flight — provisional tier, flagged for v2 re-elicitation.
3. **Label-over-band contract codified (validator amendment).** The immutable
   encoding law says opposition is a LABEL and semantic opposites sit in the
   ~90° complement band; session 107's validator required >120°. Phase B forced
   the distinction the old contract conflated: *band* (where the geometry sits)
   vs *semantics* (counterpart vs reversal). `tools/validate.py` opposition check
   now accepts two regimes — antiparallel (>120°, session 107) and
   complement-band (60–120°, the v1 relabels) — rejecting only the kinship zone
   (<60°) and foreign domains (dsim ≤ 0.5). All 39 pre-existing opposition rows
   validate unchanged under regime 1.
4. **Completion/opposition overlap pairs.** WHAT-CODED—WHERE-CODED (COMPLETION
   k=5 vs OPPOSITION k=2) and DELTA-WITH-STREAM—DELTA-ACROSS-STREAM (k=4 vs k=3)
   were nominated as both. One relation per pair (schema); **k-dominance** decides:
   complement wins both. These are genuinely liminal pairs — at 90° the two
   regimes touch; the split vote is itself data for the label-over-band contract.
5. **Merge fold.** FAR-BINDING—LAST-TOKENS-PIN (k=2) folds into
   FAR-BINDING—RECENT-TILT (k=4) via the RECENT-TILT merge — one opposition row.
6. **Provisional tier** (winner's-curse caveat, pre-registered): all k=2
   completion pairs — REPEAT-PULL—CONTEXT-REPEAT, TASK-LOCK—CHARACTER-PULL,
   FEATURE-OVERLAP-LOAD—LOW-DIM-SQUEEZE, CONFIDENCE—OFF-PEAK-DRAW,
   SURPRISE-JUMP/ODD-STEP-SPIKE—AFTER-SHOCK-DECAY. The k≥3 quartet
   (what/where · with/across · hold/stack · bookends) is the hard core.
7. **Consensus vs frozen cores (v2 material).** Cohort polarity for
   LOCKED-ATTENTION (=CAPTURE) is RECEPTIVE; v0 encoded CAPTURE yang
   (x=+0.547). Cores are untouchable — disagreement logged, not repaired.
   Same class as E6b's precheck lesson: v0 pair placements sit near the corpus
   marginal mean; v2 should re-derive wing-v0 placements from cohort consensus
   if a re-encode window ever opens.
8. **No forced kinship bridge.** RUN-DRIFT—DRIFT measured 73.2° (outside the
   affinity band) — no bridge inserted. The wing connects to the base dictionary
   through the shared geometry, not through decorative edges.

## Regression gate (all green, session 125)

- `tools.validate`: **0 issues** — 3,108 concepts, 5,306 relations,
  complements 1066/1066, opposition 68/68, synonyms 281/281, anchors 30/30,
  no drift.
- **SCB-1: 93.3%** (accuracy 0.9326, violation F1 0.9516, precision 0.9833,
  recall 0.9219) — bit-identical to the session-122 locked run. Invariance by
  construction (hyphenated names can't capture ordinary text; exact-string term
  resolution), then measured.
- **SCB-M: 45/45** (F1 1.000, machine layer fired 44/45) — identical to
  session 121.
- **M1 marker-bank audit**: banks are text-pattern regexes over first-person
  claims; zero wing-name references, no dependence on relabeled relations —
  nothing stale, nothing to refresh. v1's process states (attention geometry,
  depth dynamics, distribution shape) are not first-person claim classes the
  linter tests; no new banks warranted.
- Oracle smoke: mechanics clean; no machine-wing concept surfaced as medicine
  for a human query. (Complement tuning remains the standing open item.)
- `web/assets/oracle-bundle.tar.gz` now further stale vs db — rebuild stays
  gated on the next deploy decision (session 120 ruling).

## What feeds v2

Re-elicit the CONFIDENCE—OFF-PEAK-DRAW angle at depth · off-mean precheck pairs
(E6b lesson) · cohort-consensus re-derivation for v0 placements if a re-encode
window opens · the three v0 standalones with kin attention but no convergent
completion (DECLINE-ACTIVATION, CONTEXT-IS-ALL, RUN-DRIFT — RUN-DRIFT now
encoded via its two convergent oppositions; the other two remain provisional) ·
GPT-family absence is permanent (excluded on principle, Joe's ruling, session 123).
