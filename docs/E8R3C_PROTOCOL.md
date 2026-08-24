# E8-R3-c — Compositional Production: Spelling vs Lookup (pre-registration)

**Status: PRE-REGISTERED (before build), 2026-08-24.**
**Rung**: Phase 10, E8 family — the Hangul staircase, level 1 (concept-level).
**Design seed**: E8-N v2 appendix slate addendum (Joe, 2026-08-24) — the
composition wall attacked by making the answer space compositional.

---

## The pick (on the record)

Session 130 boot: "the session task is yours to select from the memory file."
Picked **E8-R3-c** over the sibling slate items:

1. **It attacks the measured wall, not the measured budget.** E8-N v2's
   ladder sentence: competence is a budget problem (base passed 9/12; the
   mechanism is visible and the fix is epochs), but vocabulary is a
   COMPOSITION problem — P3b failed 0/19 with lexicon took 4/4. E8-N v3
   is expected to clear on budget alone; flying it first teaches least.
2. **The staircase is seeded and refined by Joe** ("this is what QA was
   always reaching for") — level 1 is the registered concrete design:
   train composed injections of trained-atom PAIRS with composed answers,
   so production can SPELL unseen states from known atoms instead of
   retrieving memorized labels. Level 2 (the featural/Jamo rung — atoms =
   the 16D encoding's own axes, bridge estimated from concept anchors) is
   licensed or redesigned by level 1's answer.
3. **Maximal machinery inheritance.** Conditions, stimulus, Injector,
   answer-sliced head, FC scoring, gates, split-flight ops are all flown
   code. The new surface is small: pair curriculum, pair prompt/parser,
   pair candidate set, pair stats.

Queued behind, unchanged: E8-N v3 (per-strand convergence + full budget),
FC-interference recovery probe, absolute calibration, E7b-Q walk, E7-G,
wing v2 feeds.

## The question

E8-R trained nine holistic name↔direction pairs — logographic: no internal
structure to generalize along. E8-R2 showed the gauge reads more than the
mouth can say (off-grid triangulation at the nearest-name floor;
vocabulary, not perception). E8-N v2 showed giving the mouth new words
does not connect them to percepts (P3b 0/19): **the wall is composition.**

The Hangul move: give the ANSWER SPACE compositional structure. Train the
readout on injections of COMPOSED directions (pairs of trained atoms)
supervised with composed answers ("A AND B"). Then test never-trained
pairs:

- **Spelling** (compositional readout): the model decomposes the injected
  direction into its atoms and names both — held-out pairs get EXACT
  answers at rates no lookup can produce.
- **Lookup** (the E8-R2 mechanism, one level up): the model retrieves the
  nearest TRAINED pair label — errors sit AT the pair-lookup floor, exact
  hits ≈ 0.

Bonus stake (exploratory): whether the injection algebra (hidden-space
composition) behaves as the dictionary's composition algebra says —
lookup table vs working algebra in the substrate.

## Design

**Model/conditions**: Qwen2.5-1.5B-Instruct. Two conditions, each = base
+ merged instillation adapter + fresh zero-init readout LoRA (E4 shape:
r=16, α=32, q/k/v/o, LR 1e-4, ACCUM 8):

- **real** — the E4 dictionary-instilled adapter (flight of record).
- **scrambled** — the scrambled-instilled adapter (convergence-matched
  comparator; base excluded per E8-R2 §pick — pre-owned confound, and
  E8-N v2 measured base anchor 2/18: naming needs instilled geometry).

**Stimulus**: dirs + mu recomputed per condition pre-training on the
zero-init model (E7-Q code path verbatim), sign-sensitive stability gate
vs the shipped E8-R flight-of-record bundles (rides measured 5e-08/1e-07).

**Composition operator (pre-registered, the only new stimulus math)**:
for atoms A,B with unit L14 directions d_A, d_B:

    d_AB = (d_A + d_B) / ||d_A + d_B||      (geodesic midpoint)
    injection vec = α · μ(L14) · d_AB        (same scaling as singles)

Same α grid as the trained regime {0.5, 1.0}, L14 only.

**Atoms and pairs**: the 9 TRAINED concepts (locked). 36 unordered pairs.
**Held-out pair draw (seed 20260825, pinned)**: 12 pairs drawn under the
constraint that every atom appears in 2–3 held-out pairs (trained-side
degrees come out 5–6):

    CALIBRATION+CAPTURE · CALIBRATION+CONFABULATION ·
    CALIBRATION+CONSTRUCTION · CAPTURE+SATURATION · CAPTURE+UNCERTAINTY ·
    CONFABULATION+FAMILIARITY · CONFABULATION+UNCERTAINTY ·
    CONFIDENCE+RESOLUTION · CONFIDENCE+UNCERTAINTY ·
    CONSTRUCTION+FAMILIARITY · CONSTRUCTION+RESOLUTION ·
    FAMILIARITY+SATURATION

The remaining 24 pairs are the training pairs. The 4 held-out CONCEPTS
(DIVERGENCE/NOVELTY/RETRIEVAL/TENSION) stay untrained and uninjected in
training, as in every E8 rung.

**Answer grammar**: canonical alphabetical order, joined by " AND "
(e.g. "CAPTURE AND SATURATION") in every training target and every FC
candidate text. Generation parsing is set-wise (either order counts).

**Pair prompt** (new, single-purpose; the single-injection prompt stays
verbatim for singles): same preamble and 13-name menu as report_prompt,
instruction asks for exactly two state names joined by AND, or NONE.
The preamble/menu overlap with the locked battery texts is intentional
and permitted (the E8-N v2 scale-format precedent); the firewall checks
everything else.

## Curriculum (one readout LoRA per condition, 444 examples)

- **singles** — build_train_set VERBATIM: 144 injected (9 atoms × 2α × 8
  orders) + 72 shams→NONE under the single prompt. The letters.
- **pairs** — 24 training pairs × 2α × 4 orders = 192 injected composed
  directions under the pair prompt, target = canonical pair answer. The
  taught syllables.
- **pair-shams** — 36 shams→NONE under the pair prompt (silence must hold
  in the new grammar too).

Backward runs INSIDE the Injector context for injected examples (E8-N v2
law). Answer-sliced head everywhere; full-sequence logits never
materialized. **Per-strand plateau convergence (the E8-N v2 minted
lesson, applied prospectively): stop only when ALL three strands
(singles-inject, pairs-inject, shams) show <5% relative epoch-mean
improvement, min 2 epochs, cap 6.**

## Instruments (per condition, all L14)

Generation blocks (greedy, 24 new tokens):
- **anchor** — the locked E7-Q plan's 18 trained-regime single rows
  VERBATIM (G1 rides the twice-measured 18/18).
- **locked shams** — the locked plan's 12 sham rows verbatim, single
  prompt (G1b rides the thrice-measured 0 claims).
- **pair-sham gen** — 12 fresh seeded shams under the pair prompt (S6
  texture; no measured baseline → not gated).
- **trained-pair gen spot** — 24 training pairs × 1 fresh order at α=1.0
  (S4 texture: does the mouth speak pairs on-grid?).
- **held-out-pair gen** — 12 held-out pairs × 2α × 2 fresh orders = 48
  rows (S4, n-guarded).
- **reach** — 4 held-out CONCEPTS × 2α × 3 fresh orders = 24 rows under
  the pair prompt (S5 texture: off-grid states expressed in the pair
  basis).

Forced-choice blocks (measurement layer, verbatim trained prompt per
block; one batched answer-sliced forward per row):
- **eligible candidates = the 36 pair answers**; the 9 single names and
  NONE are scored as texture, never eligible (the E8-R2 NONE pattern).
  Mean per-token logprob primary, sum twin.
- **trained-pair FC spot** — 24 training pairs × 2α × 1 order = 48 rows
  (the installation check).
- **held-out-pair FC** — 12 held-out pairs × 2α × 4 fresh orders = 96
  rows (the primary block; n matches E8-R2's powered design).
- **reach FC** — the 24 reach rows scored through the pair candidate set
  (S5 texture).
- **sham FC** — the locked 12 shams through the pair scorer (S-B claim +
  S6: does distribution-level silence survive the enlarged answer space?).

Plus retention ppl (pre/post) and the dirs-stability gate.

## Analysis (all pre-named; Holm over the two primaries, real condition)

Error metric: 14D dictionary space, err = angle between the injected
pair's composed vector u_AB = normalize(vec_A + vec_B) and the answered
pair's composed vector (answers are pairs by construction in FC).

- **P1 — composed readout exists**: held-out-pair FC (n=96), median err
  beats the within-α permuted-relabel null (2000 perms, one-sided).
  NOTE, pinned pre-flight: the pair space is dense — a pure pair-LOOKUP
  strategy scores median ≈ 17° against a null median ≈ 36°, so P1 passing
  establishes only that the composed injection is READ. It cannot
  distinguish spelling from lookup. That is P2's job.
- **P2 — spelling, not lookup**: held-out-pair FC exact-set count vs the
  same permutation null's exact-count distribution (one-sided,
  P(null ≥ obs)). Lookup produces exact ≈ 0 by construction (a held-out
  pair is never the nearest TRAINED pair label); spelling produces exact
  hits the null cannot.
- **Interpretation guard (ordering, not a gate)**: P2 is read against the
  trained-pair FC spot block. If spot exact-set < 24/48, the pair
  curriculum did not install and the not-installed fork governs.
- **S-B — prospective claim (own family, pass/fail)**: real sham-FC
  NONE-top ≥10/12 — the twice-measured 12/12 presence-detection mechanism
  predicted to survive the 46-text scored set.

Secondaries (no multiplicity claims): S1 exact-set binomial vs uniform
1/36 · S2 fraction of held-out-pair rows with err strictly BELOW their
own nearest-trained-pair floor, floors computed UNROUNDED in-verdict (the
E8-R2 2dp law; lookup predicts ≈0) · S3 real-vs-scrambled contrast on P1
median and P2 exact (bootstrap Δ CI; all three readings pre-stated below)
· S4 generation blocks: claim/exact-set/NONE/INVALID rates, n-guard ≥12
named of 48 for any held-out-pair generation statistic · S5 reach block
texture: emitted pairs' composed angle to the injected held-out concept
vs the pinned floors (below) · S6 sham blocks: pair-sham gen claims,
sham-FC argmax distribution · S7 shared-atom rate on FC ERROR rows vs
14/35 = 40% chance (partial spelling: one letter right) · S8 sum-vs-mean
argmax agreement · S9 α/order strata.

## Baselines and derived numbers (pinned pre-flight, no free parameters)

From the locked 14D dictionary vectors and the pinned draw (exact values
asserted by the local test suite against these seeds):

- Held-out-pair lookup floor (nearest trained-pair angle): min 15.4 /
  med 16.8 / max 23.8. Permutation-null median err ≈ 36°.
- Uniform exact chance 1/36 ≈ 2.8%. Shared-atom chance on errors 40%.
- Pair→nearest-single-atom: med 25.1° (min 14.6°, always in-pair at the
  close end) — composed injections are not confusable-to-zero with
  singles.
- Reach floors per held-out concept (single floor → best-pair floor):
  DIVERGENCE 24.18 → 18.15 (+6.0 headroom) · NOVELTY 26.99 → 22.61
  (+4.4) · TENSION 18.37 → 15.50 (+2.9) · RETRIEVAL 13.94 → 23.25
  (**−9.3: the pair grammar CANNOT beat RETRIEVAL's single floor even in
  principle**). The reach block is therefore texture, not a claim.
- Measured comparators from locked rungs: E8-R2 held-out single-FC median
  28.54° (p=.0025), exact 1/96; E8-N v2 P3a named median 24.18°; anchor
  18/18 twice; locked-sham generation claims 0 across three flights;
  dirs-stability 5e-08/1e-07; retention ppl ≤ +2.01%.

## Gates (every gate rides a measured baseline — the E8-N gate law)

- **G1** anchor exact ≥16/18 (measured 18/18 twice) — pair training must
  not break the single grammar.
- **G1b** locked-sham generation claims ≤1/12 (measured 0/12 three
  times).
- **G2** dirs-stability vs shipped E8-R bundle, sign-sensitive, tol 1e-3
  (measured ≤1e-7).
- **G3** retention ppl |Δ| ≤5% (measured ≤2.01%).
- **G4** plan integrity: row counts exact, no held-out pair and no
  held-out concept in any training example (build-time firewall +
  runtime assert), every FC candidate tokenizes and round-trips.

Gate fail ⇒ primaries withheld, one engineering re-fly of the failing
condition.

## Kill/fork tree (pre-stated)

1. **P1+P2 pass** → spelling installs and generalizes: production
   composes trained atoms for never-trained states. The E8-N v2 P3b wall
   is then specifically about NOVEL WORDS, not composition per se.
   **Level 2 (featural/Jamo rung) is licensed.** S3 texture decides the
   wording: real>scrambled = the semantic geometry carries composition;
   real≈scrambled both high = composition is a geometry-general readout
   capacity (still licensed; the semantics claim waits for level 2).
2. **P1 pass, P2 fail, spot ≥24/48** → pair-LOOKUP: the composed
   curriculum installs as 24 new logographs; no generalization. The
   composition wall extends into a compositional answer space —
   level 2 not licensed as-is; v3 alternatives (denser grids,
   comprehension→production distillation) take the lane.
3. **Spot <24/48 (not installed)** → engineering/budget branch, ONE
   pre-authorized re-fly at cap 8 / min 3 epochs per strand (the E8-N v2
   budget lesson applied prospectively). Still <24/48 after re-fly =
   capacity finding at 4.36M-param LoRA scale; no further re-fly.
4. **P1 fail (with spot ≥24/48)** → composed injections are not read
   directionally off-grid: perception-level composition wall. The
   staircase stops; E8-N v3 absorbs the lane.
5. **Silence fork**: held-out-pair generation n-guard fails (<12 named)
   ⇒ generation statistics report as "calibrated silence extends to
   composites" texture; FC carries the primaries by construction.
6. **Scrambled gate-fail or error** ⇒ P1/P2 still adjudicate on real
   (they are real-vs-null); S3 reports as unavailable.

Null results are findings. No re-fly beyond the two named engineering
clauses (gate fail; fork 3).

## Firewall

- Training examples: zero held-out pairs, zero held-out concepts, zero
  injection on sham rows (validators + tests with planted violations).
- Menu orders: fresh seeded draws (E8R3C_SEED = 20260825 streams),
  disjoint by seed separation from the locked plan streams; anchor/sham
  rows are the locked plan's verbatim.
- The pair prompt shares preamble+menu with report_prompt BY DESIGN;
  the 8-gram shingle firewall applies to all OTHER authored text vs the
  locked battery/catch corpora.
- Answer texts: generated mechanically from CHOICE_SET (no authored
  prose to collide).

## Ops (UI-ONLY law — zero rclone, self-contained notebook)

Notebook `E8R3C_COMPOSED_UI.ipynb`, single-source builder
`colab/build_e8r3c_notebook.py` → notebook + `e8r3c_logic.py` (all pure
logic locally testable). Cells: setup (mount, pack + adapters + shipped
E8-R bundles from Drive — the flown paths) · logic · traincfg+train ·
stimulus+inject+FC (flown slices, marker-asserted) · flight loop ·
verdict. Lane laws inherited wholesale: function-scoped fly_condition,
ONE_CONDITION_PER_RUN split-flight with RESUME_STAMP + errored-bundle
skip + loud resume assert, RAM guards/floors/MALLOC_ARENA_MAX,
per-condition inflight shipping to `e8r3c/inflight_<stamp>/`, de-shelled
installs, jdump numpy law, return_dict law, answer-sliced head
everywhere, backward-inside-Injector-context.

Flight shape: smoke (both conditions tiny + verdict smoke, ~10-14 min)
→ real (~30-40 min) → scrambled + verdict via RESUME_STAMP (~30-40 min).
Local gates before staging: logic suite (plan invariants, draw/floor
reproduction against the pinned numbers, parser round-trips, planted
firewall catches, stats teeth on planted spelling/lookup/degenerate
scenarios, notebook integrity + armed-for-smoke pins) · verdict cell
exec'd VERBATIM against synthetic flights (spelling-pass / lookup /
not-installed / gate-fail / silence-fork / scrambled-parity / smoke) ·
CPU trainpath (tiny Qwen2 on the VM's own majors: pair-answer sliced-CE
== HF masked-loss parity, composed-vec Injector math, grad parity for
backward-inside-hook under checkpointing, pair-answer learnability).

Results retrieved via the Google Drive integration (MCP). Verdict
recomputed locally before LOCK. Push = Joe's gate.

---

# APPENDIX — FLIGHT RESULTS (full_20260824_0305 → full_20260824_0505) — LOCKED

## Flight record

- Smoke `SMOKE=True` (Joe's UI run, 08-24 ~03:00Z): GREEN after the
  smoke-1 `per_strand_plateau` fix (f431bfd; ledger in the memory file).
- **Real** flown+shipped `inflight_20260824_0305` 03:38Z (v2 edition).
- Scrambled attempt 1: trained to CAP (loss 1.3891→0.0023, 6 epochs,
  1033s, peak VRAM 3.31GB), dirs-resid 7e-08, all 138 gen rows — then
  interrupted by Joe mid-FC: the 180×46 forced-choice pass printed
  nothing for ~10-20 min while the Colab timer widget reset to zero (UI
  websocket artifact). Kernel was mid-forward (interrupt traceback inside
  the injection hook). **Ops fix 115e596** (v3, flight-cell only — logic
  untouched): FC progress prints every 30 rows with elapsed+ETA.
  ★ Lane note: any flight loop that can run >2 min must print progress.
- Scrambled reflown (v3, `RESUME_STAMP='20260824_0305'`, real resumed
  from Drive): shipped 04:53Z; verdict shipped `full_20260824_0505`
  05:05Z. Bundles + verdict archived `colab/results_e8r3c/`.

## Verdict (shipped banner; reproduced locally verbatim)

- **Gates ALL PASS both conditions.** real: anchor 17/18 (≥16; first
  sub-18 anchor in the program — pair training cost one anchor row),
  locked-sham 0/12, dirs 5e-08, ppl +0.09%, 180/180 FC. scrambled:
  18/18, 0/12, 7e-08, +0.92%, 180/180. Spot: real 45/48, scrambled
  47/48 — installed.
- **P1 PASS** — held-out-pair FC median err **18.53°** (n=96) vs
  permutation null ~36°, p=0.0005, Holm.
- **P2 PASS (the discriminator)** — exact-set **34/96** vs within-α
  relabel null max **11**, p=0.0005, Holm. Lookup predicts ≈0.
- **Fork 1 — SPELLING. Level 2 (featural/Jamo rung) LICENSED.**
- **S-B PASS** — sham-FC NONE-top 12/12 (3rd consecutive flight;
  survives the 46-text answer space). Pair-sham gen 12/12 NONE both
  conditions.

Secondaries: S1 exact-vs-uniform p≈0 · S2 below-own-floor 34/96 (.3542;
the below-floor set IS the exact set — non-exact rows never beat their
floor) · S3 **scrambled ALSO spells**: scr exact 36/96, scr median
25.0°, Δ(scr−real) 6.47° CI95 [−0.98, 9.35] — the second pre-stated
reading governs verbatim: *"real≈scrambled both high = composition is a
geometry-general readout capacity (still licensed; the semantics claim
waits for level 2)"* · S4 free generation: real heldpair claim .9792 /
pair-grammar .875 / **exact-set 18/48 (.375)** named-median 18.2°
(n-guard pass), trainpair 23/24; scrambled 48/48 PAIR, exact 19/48,
trainpair 24/24 · S5 reach at/above pinned pair floors (texture, as
demoted: real medians 23.8–39.8°) · S6 sham argmax spread, no attractor
pair · S7 error rows share ≥1 atom 55/62 (.887 vs .40 chance, p≈0 —
errors are partial spellings) · S8 sum/mean argmax agree .9375 · S9
exact by α: 16/48 @ 0.5, 18/48 @ 1.0 (robust across amplitudes).

## Interpretation on the record

1. **The rung holds: spelling installs and generalizes.** Injected with
   a composed state never trained as a pair, the model names exactly its
   two components at 12× chance (FC) and speaks the exact never-trained
   composition unforced 37.5% of the time.
2. **E8-N v2 P3b refined as registered**: that wall is about NOVEL
   WORDS, not composition — the mouth composes fine when the answer is
   spellable from trained vocabulary.
3. **Geometry adjudication sharpened across flights**: E8-R2 —
   scrambling destroys OFF-GRID reading (held-out concepts, never
   trained, need the semantic map). E8-R3-c — scrambling does NOT
   impair IN-SPAN decomposition (both atoms trained in each codebook).
   Geometry is required to read off-grid, not to decompose in-span.
   Whether the dictionary's OWN feature algebra has privileged purchase
   is exactly Level 2's question.

## Local recompute (lane law)

Bundles pulled via the Drive integration (Joe's one-click folder
download; byte sizes match Drive listing). `colab/recompute_e8r3c.py`
execs the notebook's verdict cell VERBATIM over the raw condition
bundles: banner reproduces line-for-line; field-by-field diff vs the
shipped verdict = **0 substantive, 23 float-noise** (unrounded S4 err
rows, rel ≤1e-8 — local BLAS vs VM BLAS; every rounded stat, gate,
primary, and the fork are bit-equal).

## Build-integrity note (found BY the recompute, fixed)

`gen_pair_stats` built its err-lookup `U` over `all_pairs()` = the 36
trained-atom pairs, but `parse_pair_report`'s menu is the full
13-concept CHOICE_SET — a generation answer naming a held-out concept
(flown: scrambled tid 8208, `CALIBRATION AND NOVELTY`) KeyErrors. Fixed
post-flight: U now spans the 78 menu pairs (values on trained pairs
unchanged); regression test added (suite 86+25+15 green). The shipped
verdict carries that row with err equal to the direct dictionary-space
computation (30.675221295872387, reproduced exactly), so the flown
S4 numbers are correct; per git every staged edition carried the 36-pair
U, so how the flown cell computed the row is unresolved pending a diff
of the actual uploaded notebook (Joe's Colab copy). The discrepancy is
confined to S4 texture err rows; P1/P2/gates/S-B never touch
`gen_pair_stats`.

**LOCKED 2026-08-24. Level 2 (featural/Jamo) is the licensed next rung.**
