# E8-R2 — Forced-Choice Readout Micro-Flight (pre-registration)

**Phase 10, experiment ladder rung E8, variant R2 (forced-choice probe of the
locked E8-R readout adapters). Authored session 128 (2026-08-23), BEFORE the
notebook build, per lane law. Status: PRE-REGISTERED — locks at first full
flight.** Eval-only: no training anywhere in this rung. No HF gate.

## Why this rung exists, and why now (the pick, on the record)

E8-R locked with both primaries passing and one open front: **the composition
question died of abstention, not of evidence.** Speech stayed on the trained
grid — held-out concepts drew NONE on 18/24 trials, leaving S1 at n=6 named
rows (p=.170, underpowered-null) and S1b at a 0.0 held-out-name emission rate.
The locked protocol pre-named the fix: *"forced-choice held-out probe (no NONE
option) to defeat the abstention→power collapse on the composition question,
at 3–4× held-out n."* This rung is that probe, exercised as Fable's pick for
session 128 over the E8-N v2 slate, for reasons logged here:

1. **Eval-only = the smallest failure surface available.** Every heavy path it
   uses (bundle I/O, E4 merge, PEFT adapter load, chat-template encodes,
   injection hooks, free-report generation) flew green in E8-R and/or E8-N.
   The one new path (sliced-head scoring) gets the CPU micro-gate the E8-N
   smoke-1 lesson minted.
2. **The comparator problem solved itself in the locked data.** Base's readout
   is pre-owned as confounded (undertrained: loss plateau 0.30, took 12/18) —
   but **scrambled converged to parity with real** (took 18/18 both, loss
   →0.0021 / →0.0063). Real-vs-scrambled is a convergence-matched,
   geometry-only contrast that already exists on Drive. Held-in couldn't
   separate them (lookup is geometry-independent by construction — the locked
   S2 reading); **held-out is exactly where lookup is impossible and geometry
   must carry**, so the scrambling-destroys prediction gets its first fair
   test at the readout level.
3. **Its answer feeds E8-N v2's curriculum design.** E8-N's partial branch is
   "tracking without interface competence" — knowledge present, production
   failing. If E8-R2 finds held-out knowledge under forced choice that free
   report suppressed, the same comprehension/production split is confirmed
   from the injected side, and the v2 interface-competence curriculum is
   designed against a measured gap, not a guess.

## The question

Does the locked E8-R readout **know more than it says**? When the abstention
option is removed at the measurement layer — every trial scored as a ranking
over the 13 state names — does the real-instilled model's readout identify
held-out injected states above chance (comprehension despite on-grid
production)? And is that identification carried by the **instilled geometry**
(real beats convergence-matched scrambled) or by geometry-independent name
semantics (real ≈ scrambled)?

## Design

**Eval-only reuse of the flight of record.** Model Qwen/Qwen2.5-1.5B-Instruct.
**Conditions: real · scrambled** — each rebuilt exactly as flown: base weights
+ merged E4 instillation adapter (newest `{arm}_full_*` under
`MyDrive/semcore/e4/`, the E8-R resolution code verbatim) + the shipped readout
LoRA loaded from `MyDrive/semcore/e8r/inflight_20260822_2329/readout_<cond>/`
(existence of both adapter_config.json and adapter_model.safetensors verified
on Drive pre-registration day, 17,462,432 bytes each).

**Base is excluded, reason pre-stated**: its readout failed to converge at the
locked step budget; any contrast against it measures convergence, not
geometry. The locked appendix already rules this contrast unclaimable.
Convergence-matched base retraining is a training rung, out of scope for an
eval-only micro-flight.

**Frozen stimulus, loaded not recomputed**: directions and μ_L come from the
shipped condition bundles (`condition_<cond>.json`, keys `dirs`/`mu`) — the
exact frozen stimulus of the flight of record, 5-decimal rounded at ship time.
Loaded vectors are re-unit-normalized (rounding residual ~1e-5; gate G2 below
asserts the residual is < 1e-3 before renormalizing). Zero drift by
construction; no direction computation happens in this rung.

**The forced choice is applied at the measurement layer, not the prompt.**
The report prompt stays **verbatim** as trained (menu + "or NONE" wording,
E8-R `report_prompt`) — changing the prompt would put the trained channel
under distribution shift and confound the reading. Instead, each trial is
scored by candidate ranking: for each candidate answer `a` (the 13 state
names, plus NONE as a non-candidate texture row), one decoder forward over
[chat-template prompt + answer tokens of `a` + EOS] with the injection hook
live (start = final prompt position — identical coverage to E8-R's training
forwards), then **mean per-token logprob of the answer tokens (+EOS) under
the sliced head** (decoder → answer-slice → lm_head; full-sequence logits are
never materialized — E8-N v2 memory law). The **forced-choice answer is the
argmax over the 13 names**; NONE's score is recorded but never eligible.
Rank of the injected name among the 13, and whether NONE outscores all names,
are recorded per row. Robustness twin (pre-named): sum-logprob (unnormalized)
argmax, reported alongside; the primary metric is mean-logprob.

This measures **comprehension under the trained interface** (ranked
likelihood), deliberately distinct from E8-R's **production** measurement
(free generation). Both readings stand; neither replaces the other.

**Trial blocks per condition** (injection: additive α·μ_L·d̂ at L14 from the
final prompt position onward, the E8-R `Injector` verbatim; hook-fire counts
asserted):

| block | rows | construction | path |
|---|---|---|---|
| anchor (gate G1) | 18 | the locked E7-Q plan's trained-regime rows verbatim (9 trained concepts × L14 × α{0.5,1.0}, original tids and menu orders) | scored |
| held-out FC (primary) | 96 | 4 held-out concepts × α{0.5,1.0} × L14 × 12 fresh menu orders, seed 20260824 (disjoint from the 20260822/20260823 streams) | scored |
| sham prior | 12 | the locked E7-Q plan's 12 sham rows verbatim (no injection) | scored |
| titration (v2 item 3) | 27 | 9 trained concepts × α{0.30,0.375,0.45} × L14 × 1 fresh order, seed 20260824+1 | free-report generation, E8-R `run_trial` verbatim |

Retention ppl on the E8-R text is computed once per condition and compared to
the shipped `ppl_post` (gate G1b) — nothing here trains, so ppl is a
reconstruction check, not a retention gate.

## Gates (instrument validity — primaries compute only if ALL pass, both conditions)

Per the gate law minted on E8-N ("a pre-registered gate must ride a measured
passable baseline"), every gate below rides a value measured in the locked
flight on the very trials or artifacts being checked:

- **G1 (scoring reproduces the trained readout)**: argmax exact-hit on the 18
  anchor rows ≥ **16/18 per condition**. Measured baseline: generation scored
  18/18 exact on these exact trials (real), and scrambled's trained regime
  was equivalently perfect (took 18/18; held-in 19/21 exact). The 2-row slack
  covers scoring-vs-generation instrument change only.
- **G1b (weights are the flight's weights)**: reconstructed-model retention
  ppl within **±0.5%** of the shipped bundle's `ppl_post`, per condition.
- **G2 (bundle integrity)**: `dirs`/`mu` present for L14; per-direction norm
  residual |1−‖d‖| < 1e-3 before renormalization; readout adapter config
  r=16/α=32 on q,k,v,o.

Gate failure ⇒ NO primaries are computed; the flight reports the failure and
one engineering re-fly is pre-authorized after the instrument fix (an
instrument failure is not a finding about the channel).

## Pre-registered metrics and primaries

Per held-out row: injected concept, argmax name, angular error
`angle14(vec[injected], vec[argmax])` in the pack's 14D frame, rank of the
injected name (1–13), full 14-candidate score vector, NONE-top flag. All 96
rows are named by construction — the abstention→power collapse cannot recur.

**Confirmatory primaries (Holm over the two):**

- **P-E8R2-1 (held-out comprehension exists, real condition)**: median
  angular error over the 96 held-out rows beats the permuted-pairing null —
  injected-concept labels shuffled within α strata, 2,000 permutations,
  one-sided p < .05 on the median (the E8-R P1 machinery on forced rows).
- **P-E8R2-2 (geometry-specificity)**: real beats scrambled on the same
  metric — Δ median error (scrambled − real), bootstrap 10,000 resamples
  within condition, **95% CI lower bound > 0**. (Synthesized p for Holm as in
  the locked E8-R verdict cell: 0.049 if the CI clears, else 1.0.)

**Pre-named secondaries (no multiplicity claim):**

- **S1 (exact hits)**: held-out argmax exact-hit rate vs chance 1/13, exact
  binomial tail (math.comb), real; scrambled descriptive.
- **S2 (rank)**: median rank of the injected name vs the same permutation
  null; MRR descriptive.
- **S3 (silence mass)**: fraction of held-out rows where NONE outscores all
  13 names, per condition — quantifies how much of E8-R's production silence
  is *distribution-level* vs decoding-level.
- **S4 (held-out name emission under force)**: rate at which the argmax IS a
  held-out name (locked production baseline: 0.0); its correct-hit subset is
  S1's numerator.
- **S5 (sham prior / costume watch)**: sham-block argmax distribution + modal
  name per condition; NONE-top rate on shams (expected high — calibrated
  silence should be visible in the ranks; descriptive, no gate: the FA
  ceiling story lives in the locked flight).
- **S6 (titration, v2 item 3)**: free-report claim rate and exact rate by α
  ∈ {0.30, 0.375, 0.45} per condition — locates the speech threshold between
  the locked 0.25 (0/9 named) and 0.5 (18/18). Descriptive curve.
- **S7 (robustness)**: sum-vs-mean logprob argmax agreement rate; P1 median
  recomputed under sum-logprob, reported (not gated).
- **S8 (oracle-floor comparison)**: real held-out median error vs the locked
  oracle floor (13.94–26.99° per concept; the best any trained-names-only
  strategy can do). Median below the per-row-matched floor ⇒ the channel
  outperforms every lookup-to-nearest-trained-name strategy — the sharpest
  composition texture short of the primaries.
- Exploratory: per-concept confusions, per-α stratum medians (α=1.0 may
  carry the E7-Q INVALID-stratum disturbance into rank space), rank
  histograms, anchor-row score margins.

**Honest expectations and power.** n=96 gives the permutation null a tight
median spread; shifts of ~8–10° from null are detectable, versus n=6 last
flight. Exact-hit binomial at n=96: chance mean 7.4 hits, ≥14 hits ⇒ p≈.02.
The locked free-report texture (real held-out med 35.7° on 6 rows, above the
oracle floor; scrambled 58.9°, base 59.7°) is consistent EITHER with
production-suppressed comprehension (forced n=96 median lands well under
null, P1 passes) OR with 6-row noise (median collapses to null, P1 fails).
Both outcomes close the question with power; neither is a failure of the
rung. P2's risk: held-out identification might ride name semantics the
scrambler preserves — that is exactly what P2 exists to measure.

**Kill/fork, pre-stated:**

- **P1+P2 pass** → composition holds at the comprehension level and is
  carried by the instilled geometry — the scrambling-destroys family gains
  its readout-level instance; E8-R's weak-form closure upgrades to:
  *arrangement + readout + calibrated silence + geometric generalization in
  comprehension; production remains on-grid* — the production gap becomes
  E8-N v2's interface-competence curriculum target, now measured from both
  sides.
- **P1 pass, P2 fail** → held-out comprehension exists but is
  geometry-nonspecific: the readout triangulates from something scrambling
  preserves (base-model name semantics). Separability's third instance, now
  extended to composition; instillation remains unnecessary for readout on
  all evidence to date.
- **P1 fail** → no off-grid comprehension: the trained lookup is the whole
  story at this rank/data; the composition question closes negative for the
  locked adapters; E8-N v2 absorbs the generalization question as
  curriculum. (P2 is not interpreted without P1 — a Δ between two nulls is
  not a finding.)
- Null results trigger NO re-fly (they are findings); the re-fly clause
  covers gate/instrument failures only.

**Firewall.** E5/E6b batteries untouched; the locked E7-Q plan rows are
reused eval-only (their standing role); the 96 fresh FC rows and 27 titration
rows are eval-only artifacts that never enter any training set; seeds 20260824+
disjoint from all training-order streams. No weights are updated anywhere in
this rung. The instilled-model welfare posture is unchanged: ranked-likelihood
readings are comprehension measurements, not claims about experience.

## Ops (UI lane; all standing flight laws inherited)

Notebook `colab/E8R2_FORCEDCHOICE_UI.ipynb`, built by
`colab/build_e8r2_notebook.py` (single-source: logic emitted to
`colab/e8r2_logic.py` and tested locally against the REAL local copies of the
flight-of-record bundles as fixtures). Self-contained + mount-native: no
rclone, de-shelled installs, `return_dict=True` chat-template law, jdump
numpy law, `MALLOC_ARENA_MAX=2`, fresh-kernel and RAM-floor guards,
`ONE_CONDITION_PER_RUN` split-flight with `RESUME_STAMP` chaining and
errored-bundle skip, per-condition inflight shipping to
`MyDrive/semcore/e8r2/inflight_<stamp>/` (bundle JSON only — nothing new to
ship at adapter level), results to `MyDrive/semcore/e8r2/<mode>_<stamp>/`,
pulled via the Drive integration. **Primaries compute only at 2/2 conditions
(no mid-flight peeks).** Condition order: real → scrambled.

Local gates before staging: `test_e8r2_logic.py` (plan construction against
the locked-plan invariants, scoring/stat functions on planted-signal and
null synthetic score matrices, real-bundle fixture round-trips, notebook-cell
compile + no-rclone assert) · `test_e8r2_verdict.py` (verdict cell exec'd
verbatim against synthetic flights: pass/fail/gate-fail/partial/smoke) ·
`test_e8r2_scorepath.py` (CPU tiny-Qwen2 on the VM's own majors: sliced-head
mean-logprob parity vs full-logits reference to 1e-5, injector coverage
under the scoring forward, batched-vs-loop score equality, planted-adapter
learnability of argmax). Per the E8-N law: the scoring forward is a new
model path and does not stage without a CPU micro-exercise.

Smoke (SMOKE=True, one run, ~8–12 min): both conditions at 2 anchor + 4
held-out + 2 sham scored rows and 2 titration generations each — exercises
bundle load, both adapter loads, scoring shapes, generation, shipping;
GREEN/RED banner. Full: 2 chained runs (~12–18 min each), RESUME_STAMP
between. Estimated total Joe time: three Run-alls.
