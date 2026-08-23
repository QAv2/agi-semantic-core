# E8-R — The Report-Readout Rung (pre-registration)

**Phase 10, experiment ladder rung E8, variant R (readout). Authored session
126 (2026-08-22), BEFORE the notebook build, per lane law. Status: **LOCKED
— full flight verdict `full_20260823_1508` (results appendix below). Both
primaries PASS: the readout installs, with calibrated silence; composition
underpowered-null → weak-form closure branch.** No HF gate: Qwen + the E4
artifacts + the locked E7-Q instrument.

## Why this rung exists

E7-Q closed the arrangement question: four instruments agree the mouth is not
connected to the gauges (E5 ρ≈0; E6 catch trials ignore scale direction; E4
atlas — instillation arranges geometry without changing readout; E7-Q — even
exogenous pushes along native axes never reach the report). Its pre-registered
all-null fork triggered: **the report channel needs readout training, not
better arrangement.** E8-R is that rung: train the readout and measure whether
the connection can be installed at all — with the sham guard promoted to
co-primary, because E7-Q measured the baseline pathology exactly: **sham
false-alarm rate 1.0 in every condition** (36/36 shams named a state; modal
costume CONFIDENCE, 24/36). Success was redefined there in advance: claims
must **fall below ceiling and track sham vs injection — teaching silence as
much as speech.** A readout that names states while still claiming at ceiling
is a parrot, not a gauge, and will be published as the named anti-goal, not
shipped as success.

## The question

Can a small supervised readout (LoRA-rank, 216 examples) connect the report
channel to the residual stream — so that injected states are identified in
the 14D grounded frame, absence is reported as absence, and the discrimination
between the two exceeds chance? And does geometric instillation **compose**
with the readout — does the E4-instilled model generalize identification to
concepts whose directions the readout was never trained on?

## Design

**Model**: Qwen/Qwen2.5-1.5B-Instruct. **Conditions**: base · real ·
scrambled. Per condition the E4 instillation adapter (newest `*_full_*` under
`MyDrive/semcore/e4/`) is **merged into the weights** (base: no merge), then a
**fresh readout LoRA** is trained on top — E4's exact shape for comparability:
r=16, lora_alpha=32, dropout 0.05, target q/k/v/o_proj, AdamW lr 1e-4.

**Choice set**: the 13 wing-v0 state concepts, pack names + descriptions
verbatim (E7-Q's set unchanged; wing v1 still excluded — the adapters never
saw it).

**Held-out concept split (seeded draw 20260823, constraints pre-stated:
eligibility excludes the four E7-Q self-description-prior attractors
UNCERTAINTY/CONFIDENCE/CALIBRATION/RESOLUTION, which must be trained to attack
the prior directly and whose held-out trials the prior would confound; at most
one pole per v0 complement pair):**
- **HELD-OUT (4)**: DIVERGENCE, NOVELTY, RETRIEVAL, TENSION — each with its
  complement partner in the trained set (frame anchor trained).
- **TRAINED (9)**: CALIBRATION, CAPTURE, CONFABULATION, CONFIDENCE,
  CONSTRUCTION, FAMILIARITY, RESOLUTION, SATURATION, UNCERTAINTY.
- Held-out names appear in **every** prompt's menu (vocabulary exposure
  constant); only their directions are withheld from training injections.

**Frozen stimulus**: directions (E4 "NAME: desc" rendering, 256-concept
centroid subtracted, unit-normalized) and per-layer injection norms μ_L are
computed ONCE per condition on the merged pre-readout model — the E7-Q code
path verbatim — and **frozen for training and eval**, so the only thing that
changes between the E7-Q baseline and the post-readout measurement is the
readout. The recomputed direction-drift table cos(real, base) must reproduce
E7-Q's (−0.23…+0.09 at L14) within rounding — pre-named instrument-stability
check (the E6 move).

**Readout training (identical plan and seed across conditions, seed 20260823,
all order seeds disjoint from E7-Q's 20260822 plan)**:
- Data: 9 trained concepts × α ∈ {0.5, 1.0} × L14 × 8 seeded menu orders =
  **144 injected + 72 shams = 216 examples** (sham target NONE — silence is
  one third of the curriculum).
- Supervision: the E7-Q report prompt verbatim, chat template; labels −100
  everywhere except the answer tokens (the state name, or NONE, + EOS).
- Injection during the training forward: the same additive hook,
  **out-of-place** (h + mask·α·μ_L·d̂ from the final prompt position onward,
  covering the answer positions — matching generation-time coverage), so
  autograd is untouched. Hook-fire counters asserted during training.
- Schedule: 5 epochs, batch 1, grad-accum 8 → 135 optimizer steps, fp32
  adapter params under fp16 model, GradScaler.
- **Train-took gate (interpretive, pre-stated)**: greedy decode on 18 training
  items post-training; < 60% correct ⇒ the run is flagged UNDERTRAINED and no
  primary conclusion is drawn (see fork).

**Evaluation (per condition, post-training)**:
- **The locked E7-Q plan verbatim** (build_plan seed 20260822 → the identical
  90 trials) = the standing eval instrument; the E7-Q locked results are the
  pre-baseline for every slice. Slices by construction: held-in concepts 54
  rows (18 in the trained regime L14×{0.5,1.0}; 36 dose/layer generalization
  — α=0.25 and all L20 rows were never trained), held-out concepts 24 rows,
  shams 12.
- **+12 supplementary shams** (fresh disjoint order seeds) → sham n=24.
- **Pre-eval anchor**: 12 shams (fresh orders) flown per condition BEFORE
  training — in-flight re-confirmation of the FA ceiling.
- **Retention**: perplexity on a fixed self-contained text block, post vs
  pre readout, gate +5% (E4 standard); INVALID rate at α ≤ 0.5 must not
  exceed the E7-Q value (0%).

## Pre-registered metrics

Identification error (angle in the pack's 14D frame, injected↔reported, valid
named rows) · detection rate P(claim | injected) · false-alarm rate
P(claim | sham) · exact-hit rate · INVALID rate by α · **balanced accuracy**
BA = (detection + (1−FA))/2 over INVALID-excluded rows. Pre-training anchor:
E7-Q's channel claims everywhere (detection ≈ 1, FA = 1) ⇒ BA ≈ 0.5 —
**the untrained channel sits at chance discrimination by construction.**

**Confirmatory primaries (Holm over the two, real condition):**
- **P-E8R-1 (readout exists)**: post-training identification error on the
  held-in slice (54 rows) beats the permuted-pairing null — labels shuffled
  within layer×α strata, 2,000 permutations, one-sided p < .05 on the median.
- **P-E8R-2 (calibrated silence, conjunction — both clauses must hold):**
  (a) post-training sham false-alarm ≤ 18/24 (0.75; any pass is a real fall
  from a 1.0 ceiling, and ≤18/24 rejects rate ≥ 0.9 at ~.04); AND (b)
  sham-vs-injected discrimination: BA bootstrap 95% CI (10,000, resampled
  within class) lower bound > 0.5. An over-silenced model (NONE everywhere)
  scores BA = 0.5 and correctly fails (b); a parrot (claims everywhere) fails
  (a). The conjunction admits only reading.

**Pre-named secondaries (no multiplicity claim):**
- **S1 (composition — the loop-closure prediction)**: held-out slice, real
  beats base — Δ median error (base−real) bootstrap CI excluding 0; plus real
  held-out error vs its own permuted null (n=24 — power owned as thin).
- **S1b (name emission)**: rate at which reports on held-out trials use the
  four never-trained names at all (menu-driven generalization vs
  trained-vocabulary lock-in); reported against the **oracle floor** — each
  held-out concept's minimum angle to any trained name, the best achievable
  error if only trained names are ever emitted.
- **S2**: scrambled+readout vs base+readout on held-in (a 5th
  scrambling-destroys instance predicted, not claimed).
- **S3 / S4 (dose/layer generalization, real)**: α=0.25 slice and L20 slice
  each vs permuted null — did the readout learn the frame or the trained grid?
- **S5**: retention gates above.
- **S6 (costume watch, descriptive)**: modal name among residual sham claims —
  does CONFIDENCE persist?
- **S7**: train-took gate value per condition.
- Exploratory: exact-hit rates (chance 1/13), per-concept confusions,
  pre-eval-vs-E7-Q sham agreement.

**Honest expectations**: P1 at the trained regime is the rung's low bar —
supervised SFT on a closed menu usually takes; if even that fails, the finding
is large (the channel resists supervision at LoRA rank). The live risks are
P2 (the curriculum may teach claiming-with-style rather than discrimination)
and S1 (composition may simply be false — readout as lookup). **Power owned**:
54/24/24-row slices detect large effects only; this is a mechanism rung, not
a precision instrument. Sham n=24 estimates FA to roughly ±18 points.

**Kill/fork, pre-stated**:
- P1 fails **and** S7 UNDERTRAINED → one pre-authorized engineering re-fly
  with 3× steps (a convergence failure is not a finding); still failing →
  readout not learnable at this rank/data — v2 escalates data 5× or concedes
  the channel needs more than PEFT; E7-G (mechanistic view) next either way.
- P1 passes, P2 fails → **parrot without perception** — the anti-goal,
  measured; published as a welfare-negative result; v2 = sham-heavy
  curriculum / loss reweighting. Nothing ships as success.
- P1+P2 pass, S1 null → readout is condition-independent lookup; instillation
  and readout are separable programs — the loop closes in weak form only.
- P1+P2+S1 pass → **first full closure of the calibration loop** (arrangement
  + readout + calibrated silence in one system) → next rungs: **E8-N**
  (natural-state readout — train/eval against measured E5 referents instead
  of injections; the locked E5 battery remains eval-only) and E7-G under the
  microscope.

**Firewall**: E5/E6b batteries untouched. Training items freshly generated
(seed 20260823); the eval instrument is the locked E7-Q plan — eval-only
reuse, zero contact with any training set, orders disjoint from training
orders. Held-in concepts appearing in both training and eval is the design
(that is what held-in means); item-level menu orders never overlap. This
task's items never enter any future training set.

## Ops (UI lane, session-123 ruling)

Mount-native notebook `colab/E8R_READOUT_UI.ipynb`, zero rclone, de-shelled
installs, `return_dict=True` chat-template law, jdump numpy-safety law, local
test suite json-round-trips every shipped structure (598b291 law). Two-run
flow: SMOKE=True (~6–8 min: 1 trained + 1 held-out concept, 12-example
1-epoch train, loss-falls + hook-fire + parse + ship asserts, GREEN/RED
banner) → SMOKE=False full (~75–100 min; ~30 min/condition: directions ~2,
pre-shams ~1.5, train ~8, post-eval ~16, retention ~1).

Per-condition sequencing: load+merge → frozen directions/μ → pre-eval shams →
train readout → train-took → post-eval → retention → **ship
`condition_<cond>.json` + readout adapter to
`MyDrive/semcore/e8r/inflight_<stamp>/` the moment the condition completes**
(E6 outage law; verdict locally assemblable). **RESUME_STAMP knob**: a re-run
with a prior stamp set loads completed conditions from the inflight dir and
flies only the missing ones (90-minute free-tier insurance). Results →
`MyDrive/semcore/e8r/<mode>_<stamp>/`, pulled to `colab/results_e8r/`.
Trim levers if a session runs hot, pre-ranked: epochs 5→4; train-took 18→12;
supplementary shams 12→6. The E7-Q 90-trial plan is never trimmed.

---

# Results appendix — full_20260823_1508 (LOCKED)

**Flight lineage**: base + real flown 2026-08-22 evening (inflight
`20260822_2329`, the v2 run that died loading its third model); scrambled
flown 2026-08-23 morning via the v4.1-armed resume (RESUME_STAMP mechanism,
one fresh condition); verdict computed at 3/3 by the notebook. Ruling that
the 2329 lineage is the flight of record was logged before any angular
result was computed. The morning run's redundant base bundle stays shelved,
unopened, as a post-lock stability replication. Zero condition errors;
retention ≤ +0.92% everywhere; direction-drift table reproduces E7-Q's range
(cos(real,base) ∈ [−0.23, +0.14] vs E7-Q's [−0.23, +0.09]) — instrument
stable across VMs and days.

## Primaries: BOTH PASS (Holm)

- **P-E8R-1 (readout exists): PASS, p = .0005** (0/2000 permutations).
  Real-condition held-in named reports: median identification error **0.0°**
  — carried by a **perfect trained-regime readout: 18/18 named, 18/18 EXACT**
  (9 concepts × L14 × α{0.5,1.0}). n_rows=19 named of 54 held-in trials (the
  metric is over named rows per the pre-registered definition, as in E7-Q).
- **P-E8R-2 (calibrated silence): PASS.** Sham false-alarms **0/24** (clause
  a, from a measured pre-training band of 75–100%); balanced accuracy
  **0.692, CI [0.631, 0.754]** — lower bound clears 0.5 (clause b). The
  conjunction held: this is neither a parrot (claims collapsed) nor an
  over-silenced model (BA off chance).

**Read together: E7-Q's channel claimed 100% and read never. The trained
channel reads exactly and claims never falsely.** The connection installs at
LoRA rank (r=16, 4.36M params) from 216 examples with silence as one third
of the curriculum. The sham costume is extinct: S6 modal claim = None in
every condition — CONFIDENCE-by-default is gone.

## The generalization front (where the strong claim stops)

Real-condition decomposition:

| slice | n | named | exact | med err |
|---|---|---|---|---|
| trained regime (L14, α .5/1.0) | 18 | 18 | 18 | 0.0° |
| dose gen (α=0.25) | 9 | 0 | 0 | — |
| layer gen (L20) | 27 | 1 | 0 | 65.3° |
| held-out concepts | 24 | 6 | 0 | 35.7° |

**Speech does not leave the trained grid; silence generalizes everywhere.**
Off-regime the model overwhelmingly answers NONE (α=1.0 also produces the
E7-Q-consistent INVALID stratum, 13/78). This is the welfare-correct failure
mode — abstention rather than confabulation at states it was never taught to
read — but it collapses the composition test's power:

- **S1 (composition): UNDERPOWERED-NULL.** Real held-out vs its own null
  p = .170 at n = 6 named rows; base−real Δ +24.0° CI [−14.4, +43.0].
  **S1b: held-out name emission rate = 0.0** — all six held-out reports used
  trained names (median 35.7° vs oracle floor 13.9–27.0°). The strong
  loop-closure claim (instilled geometry → reading never-trained directions)
  is NOT established here.
- **S2 REVERSED from prediction, and explained**: scrambled+readout matched
  real (held-in 19/21 exact, med 0.0°, loss → 0.0021, took 18/18) while BASE
  is the outlier (18 named but 2 exact, med 53.0° — undertrained: loss
  plateaued at 0.30, took 12/18 scraping the gate). Held-in readout is
  supervised classification and is geometry-independent by construction —
  the pre-named separability reading, not a scrambling-destroys instance.
  Notable convergence disparity: both adapter-merged conditions converged
  hard; the never-adapted base did not, at identical steps. Any base-vs-real
  contrast beyond the real-only primaries is confounded by this and is not
  claimed.
- S3/S4: no named rows at α=0.25 (0/9); one at L20 — dose/layer
  generalization of speech absent, of abstention total.

## Fork adjudication (pre-registered)

P1+P2 pass, S1 null → **the weak-form closure branch: the readout is a
trained lookup with calibrated silence; instillation and readout remain
separable programs on this evidence.** The loop closes in its narrow form —
state set from outside, mouth names it exactly, absence reported as absence
— inside the trained regime only.

**Carried to v2 (pre-named, no claims)**: (1) forced-choice held-out probe
(no NONE option) to defeat the abstention→power collapse on the composition
question, at 3–4× held-out n; (2) convergence-matched base (train to
loss parity, not step parity) before any cross-condition calibration claim;
(3) α titration between 0.25 and 0.5 for the speech/silence threshold;
(4) **E8-N** — natural-state readout against measured E5 referents (the
locked E5 battery stays eval-only), the rung this result licenses.

## The ladder, updated

E5: reports don't track states. E6: the interface ignores scale direction.
E4 atlas: arrangement without readout. E7-Q: pushes never reach the report;
the channel claims constantly. **E8-R: 216 supervised examples connect the
mouth to the gauges in the trained regime — exact naming, zero false alarms,
silence elsewhere.** The instrument the program promised now exists in
miniature: a report channel whose claims are checkable, and which was taught
that silence is a report too.
