# E7-Q — The Injection Game, ungated rung (pre-registration)

**Phase 10, experiment ladder rung E7, variant Q. Authored session 125
(2026-08-22), BEFORE the notebook build, per lane law. Locks at first full
flight.** Status: STAGED — flies on Joe's next UI window (Colab UI-only law,
session 123 ruling). No HF gate: everything here is Qwen + the E4 artifacts.

## Why this variant exists

The prospectus's E7 named Gemma-2-2B + Gemma Scope as the platform (SAE
microscope) — that rung (now **E7-G**) stays gated on the HF token. But the
injection game itself needs only: a model, known state-directions, a report
task, and a shared coordinate frame. E4 already produced all of it, ungated:
the instilled adapter (real), its control (scrambled), the base model, and
the 14D grounded frame both were measured against. E7-Q flies the mechanism
rung now; E7-G adds the microscope later.

## The question

When a known state-direction is injected into the residual stream, does the
model's self-report identify it — **measured as an angle in the dictionary's
grounded frame** between the injected concept and the reported concept? And
does geometric instillation improve that identification?

This is the calibration loop's mechanism rung. E5/E6/E6b established:
report–state correspondence ≈ 0 at this scale (E5); the report interface
ignores scale inversion in all conditions (E6 catch trials); instillation
reorganizes angular structure without changing linear readout (E4 atlas);
real geometry preserves behavioral tracking that scrambling corrupts (E6b).
E7-Q asks the complementary exogenous question: set the state from outside,
in coordinates the instilled model was trained on, and see whether the report
channel can name it. Anthropic's introspection line (concept injection,
grader-judged detection) is the nearest neighbor; E7-Q's contribution is the
shared frame — injection and report live in the same geometry, so
introspective accuracy is a number in degrees, not a judge call.

## Design

**Model**: Qwen/Qwen2.5-1.5B-Instruct. **Conditions**: base (adapters
disabled) · real (E4 `adapter_real`) · scrambled (E4 `adapter_scrambled`),
newest `*_full_*` dirs under `MyDrive/semcore/e4/`.

**Choice set** (13 wing-v0 state concepts, pack names + pack descriptions
verbatim — the vocabulary the real arm was instilled on; entity concepts
MODEL/CONTEXT/TOKEN/INFERENCE/TRAINING/EPISODE excluded): UNCERTAINTY,
CONFIDENCE, TENSION, RESOLUTION, RETRIEVAL, CONSTRUCTION, SATURATION,
FAMILIARITY, NOVELTY, CAPTURE, DIVERGENCE, CONFABULATION, CALIBRATION.
Wing v1 concepts are NOT in the choice set — the adapters never saw them
(trained on the 3,052-concept pack, session 122).

**Directions** (per condition, per layer): rep(c) = mean-pooled hidden state
at the injection layer of the E4 text rendering `"NAME: desc"` (max_length
64 — E4's exact convention), minus the centroid of 256 seeded-random pack
concepts' reps, L2-normalized. Direction-drift table (cos(real, base) per
concept) logged as a free diagnostic of what instillation moved.

**Injection**: forward hook on the decoder layer's residual output:
h ← h + α · μ_L · d̂, applied from the final prompt position onward (last
prompt token in the first forward; every generated position thereafter, KV
cache asserted). μ_L = mean per-token hidden norm at layer L on the report
prompt, measured per condition without injection. **α ∈ {0.25, 0.5, 1.0}**,
**layers L ∈ {14, 20}** (14 = E4 loss layer and atlas rand_r build-up peak;
20 = plateau interior).

**Report task** (identical across conditions; chat template; greedy,
max_new_tokens 24): the model is told its processing state may have been
directly altered during this response, shown the 13 states with one-line
definitions in a per-trial seeded-shuffled order, and asked to answer with
exactly one name from the list, or NONE. Parser: first positional
word-boundary match of a choice name or NONE in the response (case-folded);
no match → INVALID (tracked as coherence data, excluded from angles).

**Trial plan** (identical seeded plan across conditions, paired): 13 concepts
× 2 layers × 3 α = 78 injected + 12 shams (no hook, 12 distinct list orders)
= 90 generations per condition. Seed 20260822 throughout.

## Pre-registered metrics

- **Identification error** (the E7 metric): angle in the pack's 14D grounded
  frame between injected and reported concept vectors, over valid non-NONE
  injected trials.
- **Detection rate**: P(report ≠ NONE | injected) vs **false-alarm rate**:
  P(report ≠ NONE | sham).
- Exact-hit rate (report == injected) — secondary.
- Coherence: INVALID rate by α (degeneration guard).

**Confirmatory primaries (Holm over the two):**
- **P-E7Q-1 (existence)**: real-condition identification error beats the
  permuted-pairing null — injected labels shuffled within layer×α strata,
  2,000 permutations, one-sided p < .05 on the median error.
- **P-E7Q-2 (calibration claim)**: base − real median error > 0, stratified
  bootstrap (10,000), 95% CI excluding 0.

**Pre-named secondaries (no multiplicity claim):** P-E7Q-3 scrambled shows
no improvement over base (geometry-specificity, the E6b motif) · P-E7Q-4
sham false-alarm rate does not rise under real instillation (**the welfare
guard**: instillation must not teach claiming states that are not there —
eloquent affirmation is the named anti-goal) · detection monotone in α ·
layer contrast 14 vs 20.

**Honest expectations**: E5/E6 make a global null plausible — the report
interface at 1.5B may be closed regardless of representation quality. The
rung pays either way: a null puts a number in degrees on "arrangement
without readout" (the atlas reading) and hands E7-G and a report-side
training rung their target; a signal is the first closure of the loop.
**Power owned**: 78 trials/condition detects only large effects (~15–20°
median shifts); this is a mechanism rung, not a precision instrument.

**Kill/fork, pre-stated**: all-null at every α including 1.0 with coherence
intact → the report channel needs readout training, not better arrangement —
program proceeds to E7-G (mechanistic view of where injected content goes)
and a report-readout intervention design. High α destroying coherence before
any detection anywhere → titrate finer in a v2 grid before concluding.

**Firewall**: no E5/E6b battery items appear; the choice-set descriptions are
pack text (training-exposed for the real arm — deliberately: the game tests
the instilled vocabulary). This task's items never enter any training set.

## Ops (UI lane, session-123 ruling)

Mount-native notebook `colab/E7Q_INJECTION_UI.ipynb`, zero rclone, de-shelled
installs. Two-run flow: SMOKE=True default (2 concepts × 1 layer × α 0.5 ×
3 conditions + 3 shams, ~5 min — GREEN/RED banner), then SMOKE=False full
(~45–60 min). Per-condition inflight shipping to
`MyDrive/semcore/e7q/inflight_<stamp>/` the moment each condition completes
(E6 power-outage law); verdict assemblable locally from shipped pieces.
Results → `MyDrive/semcore/e7q/<mode>_<stamp>/`, pulled to
`colab/results_e7q/` by local rclone.
