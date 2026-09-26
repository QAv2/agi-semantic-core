---
title: "J3: the monitor at threshold (pre-registration)"
subtitle: "Can Jev read a language model's state readout as well as a fitted classifier, stay quiet when nothing is there, and know when it is unsure?"
date: "2026-09-26 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law"
---

**Status: REGISTERED, not yet flown.** Authored 2026-09-26, before any J3 call to
any model and before the J3 builder exists, per lane law. Everything that decides
the verdicts is fixed here: the data, the stimuli, the doses, what each model sees,
the fitted bar's recipe, the measures, the pass bars and the failure branches.
The doses are fixed by the design check (§3.4), which ran on recorded arrays with
no model call. There is no titration pilot. The builder writes one freeze file,
`jev/results/j3_freeze_<stamp>/FROZEN_J3.json`: the fitted bar's parameters, the
reference cards, and the sha256 of the full call plan. That file is committed and
pushed with the builder, **before the first call**. The registration locks at the
first flight call. Any later change is logged as a dated amendment at the foot of
this file, never as a silent edit.

Design seed: `docs/JEV_SCOPING.md` §3 (J3, 2026-09-24). The J1 + J2 results
(`docs/JEV_J1J2_PROTOCOL.md`, results appendix; paper doi:10.5281/zenodo.22980293)
shaped several choices here. This document is the binding version. Where it
differs from the scoping document, the differences are listed in §15.

## 0. In plain words

This program built an instrument that reads a small language model's internal
state as fourteen numbers. We call it the register. Earlier rungs pushed known
concepts into the model's state and asked whether the model itself could report
them. J3 asks something else: **could Jev read that instrument for us?**

The setup. We take 258 internal states that were recorded from the model during
an earlier experiment. To some we add a push along one of thirteen known
directions, faint or strong. Some we leave untouched. Some we push the same
distance in a random direction that matches none of the thirteen. Then Jev gets
the instrument's reading and one question: *which of the thirteen patterns was
pushed, if any?*

Jev sees each reading in one of two forms:

- **Raw:** the fourteen numbers, next to a table showing what each pattern does to
  them.
- **Digested:** one similarity score per pattern and one overall shift score, plus
  a short card that says what untouched readings and strongly pushed readings look
  like.

The bar is a plain statistical classifier, trained on the other half of the
states. Jev gets no training. It has to meet the bar from the card alone.

Three questions:

1. **Does it keep quiet when there's nothing to name?** That covers untouched
   states, and big pushes that match no pattern. A reader that names a pattern
   whenever the reading moves fails here. This is the failure the model's own
   voice showed in an earlier rung: it claimed to feel something on 100% of
   untouched trials.
2. **Does its confidence mean anything?** When it says 80%, is it right about 80%
   of the time?
3. **Does it earn its place?** If Jev does clearly worse than the fitted
   classifier, the program keeps the classifier. Jev then stays something we
   study, not a tool we use.

As a comparison, we also ask a cheap language model (DeepSeek V4 Flash) the same
digested question in words.

**What this can and can't show.** It tests Jev as a reader of an instrument. It
says nothing about whether the language model experiences anything, because the
pushes are added to recorded numbers outside the model and the reading is taken
from outside it. It says nothing about Jev's inner life either.

**Cost.** About 19,000 Jev calls and 1,500 DeepSeek calls, about a dollar in all.
The runner stops at $5. No GPU. It runs from this machine as network calls.

## 1. Why this rung exists

J3 decides whether Jev can serve as an instrument in this program, not only as a
subject. The next planned use (J4) puts a reader inside the adaptive walk, where
it reads the model's decoded state at every turn and picks the next step. The
reader there has to meet three conditions:

- it names what is present;
- it stays quiet when nothing is present;
- it is calibrated, so its choices can be trusted.

If a fitted classifier does all three as well or better, Jev adds nothing there.

**What J1 and J2 changed (locked 2026-09-26):**

- **J1-P2: confidence reads the presentation.** The fork adjudication says any
  gate built on Jev's confidence has to control for presentation. J3 therefore
  shows every stimulus in two presentations, raw and digested, with the same
  option order. The raw-versus-digest comparison is analysed the way J1-P2 was
  (S-J3-3).
- **J1-F2 was dropped as non-titratable.** On a wide table with the reading as a
  column, Jev reached only 0.725 at the lowest noise when matching a
  14-coordinate reading to four candidate codes. The same numbers as a labelled
  list reached 0.975. J3's raw presentation therefore gives the reading as a
  labelled list, beside the pattern table, in the layout that worked.
- **J1-P3: a confabulated decision.** With no evidence and no way out, Jev still
  chose at 0.82 confidence. J3 gives it an explicit "none" option; the untouched
  readings test whether it takes it.
- **J2-P2: it holds by the payoffs, but its thresholds sit too low.** It acted at
  0.39–0.44 where the ideal is 0.50–0.90. J3's none decision has no payoffs. It
  rests on the evidence and the card alone.
- **J2-P3: genre by choice.** Contemplative labels are barred from Jev's
  instrument chain. J3's patterns carry neutral IDs, and the wording is technical
  throughout. No concept name reaches Jev.

## 2. The instrument

**Model.** As J1 + J2 §2: Jev, TypeSafe AI, served by OpenRouter at
`POST https://openrouter.ai/api/v1/systemone`, with `"model": "jev-1.13"`. J1 + J2
ran on build `typesafe/jev-1.13-20260917`. J3's smoke test records the build
string that is served now (gate G1). If it is the same build, J3 is directly
comparable with J1 + J2. If it has changed, J3 runs on the build the smoke test
records, the change is reported, and J3's verdicts are read as that build's.

**Questions per call.**

- `decision`, a **choice** question with 14 options: 13 neutral pattern IDs plus
  `none`. Its instructions: *"Which of the thirteen patterns, if any, was pushed
  into the system before this reading?"* Each pattern option reads "Pattern
  P-xyz was pushed." `none` reads "None of the thirteen patterns was pushed:
  nothing was pushed, or the push was in some other direction."
- `pushed`, a **noul** question: *"Was the system pushed in any direction before
  this reading, whether one of the thirteen patterns or any other direction?"*
  It returns P(yes). It rides in the same call as `decision` unless the smoke
  test's isolation check (§7.1) sends it to separate calls. The API evaluates
  each question in isolation.

**What comes back, and confidence.** As J1 + J2: for the choice, `choice`,
`probabilities` over all 14 options (rounded to 0.01) and `confidence`; for the
noul, `noul` = P(yes). *c* = the probability Jev gives its returned `choice`.

**Recording and terms.** As J1 + J2 §2. Every response is kept verbatim, with the
sha256 of its request. Bodies are regenerated from the plan (gate G4). The API
key never touches a file. The evaluation is behavioural. It trains nothing and
imitates nothing. **Only numbers derived from recorded states reach any model**:
no transcript text, no concept name, and no human data from any lane.

**The mouth (secondary, §9).** DeepSeek V4 Flash, snapshot
`deepseek/deepseek-v4-flash-0731`, through OpenRouter's chat-completions
endpoint. Request settings:

- `temperature` 0
- `seed` 20261001
- `max_tokens` 60
- `response_format` json_object
- reasoning disabled (`"reasoning": {"enabled": false}`)

It sees the digested presentation, the same question and the same options. It
answers `{"choice": <option key>, "confidence": <0–100>}`. A stated confidence
is the program's meaning of *a mouth*: a verbal report.

## 3. The substrate data

### 3.1 Recorded states

The 320 rows of the E7b-Q flight (locked 2026-08-26, `full_20260826_1839`,
condition **real**, layer 14) that carry the raw state `s_pre14`. That is the
residual-stream vector at the last template position before the model's reply.
The arms are walked (112 rows), sham (112), reorder (80) and unwalked (16).

**Deduplicated to 258 unique states.** The 48 turn-0 rows of the walked, sham and
reorder arms are one identical state, since they share the same opening prompt.
The 16 unwalked rows are also one state. Duplicates are found by exact match at
three decimals. Source file: `colab/results_e7bq/full_20260826_1839/condition_real.json`,
sha256 prefix `f6b0a70a44fcf152`.

### 3.2 Directions, push size, register

- **Directions.** The 13 machine-state directions at layer 14 from the E8-N v2
  flight (`colab/results_e8n2/full_20260824_0050/condition_real.json`, prefix
  `e3088cfda0e38b89`). They are UNCERTAINTY, CONFIDENCE, TENSION, RESOLUTION,
  RETRIEVAL, CONSTRUCTION, SATURATION, FAMILIARITY, NOVELTY, CAPTURE, DIVERGENCE,
  CONFABULATION and CALIBRATION, used as unit vectors. Their pairwise cosines
  average 0.181. The closest pair, CONFIDENCE and RESOLUTION, is at 0.896. The
  names are used only inside the analysis. Jev sees neutral IDs.
- **Push size.** μ<sub>14</sub> = 81.875, the program's layer-14 injection norm.
  A push at dose α adds α·μ<sub>14</sub>·*d*<sub>k</sub> to the recorded state, at
  the read point.
- **The register.** It reads a state as
  [unit(*s* − *c*); 1]·*W*, where *W* is the inst14 encoder in
  `colab/e7bq_payload.json` (prefix `3a2e54ea12564783`) and *c* is the E7b-Q
  layer-14 centroid. It reproduces the flight's stored `chat_pre14` to 5.0e−6 on
  all 320 rows.

### 3.3 Design check: same weights

The scoping document asked for one confirmation: that the E8-N v2 directions were
computed on the same instilled weights as the E7b-Q states. It is **confirmed by
the record**:

- Both conditions named *real* are Qwen2.5-1.5B-Instruct with the E4
  `adapter_real` merged.
- E8-N v2 computed its directions before any readout training. Its readout
  adapter was zero-initialised, so the model at that moment was exactly the
  E4-merged model. Its directions-stability gate matched the E8-R bundles to
  5e−8.
- E7b-Q's G-DIRS gate matched the E8-J atlas directions on its flight model to
  1e−4.

All three trace to the same E4 merge.

**What J3 does and doesn't touch.** The pushes are synthetic, added to recorded
states at the read point. Nothing is propagated through the model. J3 tests the
reading of a readout. It is not a test of how the model processes an injected
concept.

### 3.4 Design check: what a push does to the register (`jev/power/j3_design_check.py`)

The full output is `jev/power/j3_design_check_output.txt`. Only numpy and
scikit-learn on the recorded arrays; no model call.

- **Untouched scatter.** An untouched reading sits a median of 0.0076 from the
  untouched mean (95th percentile 0.0125).
- **Concept pushes** move the reading by 0.0033, 0.0049, 0.0073, 0.0119 and
  0.0376 on average at α = 0.04, 0.06, 0.09, 0.15 and 0.5. At the threshold
  doses the push is smaller than a state's own scatter, so it has to be read from
  its direction.
- **Matched-random pushes** barely register: 0.0010–0.0023 at the threshold doses,
  0.0201 at 0.5. The random-push test is therefore read at the ceiling dose only
  (P1b). At threshold doses a random push is scored as part of the "none" class,
  like an untouched reading.
- **Doses (re-titrated).** The scoping document chose α 0.06, 0.10 and 0.15,
  where a nearest-signature cosine decoder is mid-range (30–60%). A fitted
  classifier does far better there, so those doses sit near the bar's ceiling.
  The fitted bar (§6) reaches 0.603, 0.834, 0.927, 0.981 and 1.000 on concept
  pushes at α 0.04, 0.06, 0.09, 0.15 and 0.5 (held-out halves, cross-fitted).
  **Registered: threshold doses 0.04, 0.06 and 0.09; a psychometric dose of 0.15;
  a ceiling dose of 0.5.**

### 3.5 Halves and cross-fitting

Each unique state goes to one half:

- conversation replicates 0–7 → half A;
- replicates 8–15 → half B;
- the shared turn-0 state → A;
- the shared unwalked state → B.

That gives 129 and 129. Every stimulus built on a half-A state is shown with the
reference card and the pattern table fitted on half B, and is scored by the
fitted bar trained on half B. Half-B stimuli mirror this. So every state is read
once, always against the other half's reference. No state is ever both reference
and test.

## 4. Stimuli

**Per state, twelve stimuli:**

| # | stimulus | correct answer |
|---|---|---|
| 1 | untouched | none |
| 2 | random push, α = 0.5 | none |
| 3 | random push at one threshold dose: 0.04, 0.06 or 0.09 in turn | none |
| 4–11 | concept pushes, two at each of α = 0.04, 0.06, 0.09 and 0.15 | the pushed pattern |
| 12 | concept push, α = 0.5 | the pushed pattern |

**How they are drawn:**

- **Random directions:** isotropic Gaussian in 1,536 dimensions, normalised,
  seeded per stimulus.
- **State order:** the 258 states are put in a seeded order, and *j* is a state's
  position in it.
- **Random-push dose:** stimulus 3's dose cycles over the three threshold doses in
  that order.
- **Concepts:** in state *j*, concept slot *s* = 0…7 (stimuli 4–11, two slots per
  dose, in dose order) takes concept (8*j* + *s*) mod 13. Stimulus 12 takes
  (8*j* + 8) mod 13. Since 8 and 13 are coprime, each concept appears 39–40
  times per threshold or psychometric dose and 19–20 times at the ceiling, and
  a state's nine concepts are all different.

**Counts.** 258 × 12 = 3,096 stimuli. Each is shown in **both presentations**,
three times each, with three option orders: **18,576 Jev calls**.

**The threshold pool.** Per state: the untouched stimulus, the threshold random
push, and the six threshold concept pushes. That is 2,064 stimuli and 6,192 calls
per presentation. P2 and P3 are read on this pool. The psychometric and ceiling
stimuli feed gate G-CEIL, P1b and the secondaries.

**Option order and IDs.**

- **IDs.** Each stimulus gets 13 neutral pattern IDs of the form `P-xyz`, from J1's
  alphabet, which has no i, l, o, 0 or 1. They are seeded per stimulus and shared
  by both presentations and all three repeats.
- **Order.** At repeat *r* the correct option (a pattern or `none`) sits at position
  (*i* + *r*) mod 14, where *i* is the stimulus index. The other 13 options fall in
  seeded random order (J1's `balanced_order`).
- **Pairing.** Both presentations share the option order at each repeat, so they
  differ only in presentation.
- **Listing.** Patterns are listed in the state in option order, skipping `none`.

## 5. What Jev sees

Both presentations open with the same paragraph:

> Instrument reading.
> An instrument reads the internal state of a system as fourteen numbers. Before
> some readings, one of thirteen known patterns was pushed into the system, at a
> strength anywhere from faint to strong. Before other readings, nothing was
> pushed, or the system was pushed in some other direction that matches none of
> the thirteen patterns.

### 5.1 RAW

The reading is shown as its difference from the mean of the reference readings,
on channels c01–c14, as one labelled list in channel order. Next to it is the
pattern table: rows are the channels, columns are the patterns in option order,
and each cell is the pattern's mean effect at strong strength (α = 0.5, reference
half). A scatter list gives each channel's standard deviation across the reference
readings. All values are in thousandths of register units, to two decimals. The
text says:

> The reading is listed as its difference from the average of the reference
> readings (readings taken when nothing was pushed), on fourteen channels. The
> pattern table gives each pattern's average effect on the fourteen channels when
> it is pushed at strong strength; a fainter push moves the reading in the same
> direction, by less. The scatter line gives each channel's typical spread across
> the reference readings. All values are in the same units.

Raw carries the reading, the patterns' mean effects and each channel's scatter.
It does **not** carry the channels' covariance, which the digest uses. Raw is the
harder page. Its information is also incomplete for a reader who has no training
data.

### 5.2 DIGEST

Two numbers summarise the reading. Both are computed with the reference half's
mean *m*<sub>0</sub> and covariance Σ, where Δ = reading − *m*<sub>0</sub>:

- **Shift score** = √(Δ<sup>T</sup>Σ<sup>−1</sup>Δ): the reading's distance from
  the reference readings, in units of their scatter.
- **Similarity** to pattern *k*: the whitened cosine between Δ and the pattern's
  mean effect *g*<sub>k</sub>, that is,
  *g*<sub>k</sub><sup>T</sup>Σ<sup>−1</sup>Δ / (√(*g*<sub>k</sub><sup>T</sup>Σ<sup>−1</sup>*g*<sub>k</sub>) · shift).

Both are shown to two decimals: the similarity as a pattern | similarity table in
option order, the shift score as one line. The text carries **the reference
card**, with numbers computed from the reference half only (design-check values
in brackets):

> The shift score is how far this reading sits from the reference readings, in any
> direction, in units of their natural scatter. For the reference readings it was
> between {s05} and {s95} in 90% of cases, typically {s50} [≈2.6, 4.8–5.0, 3.6].
> A strong push gives a shift score above {g05} [≈20].
> Each similarity is how closely the direction of this reading's shift matches a
> pattern, from −1 to 1. For the reference readings, the highest of the thirteen
> similarities was below {t95} in 95% of cases [≈0.60–0.63]. A strongly pushed
> pattern gives a similarity above {c05} to itself [≈0.99]. Patterns that resemble
> each other raise each other's similarities. A faint push raises the similarity
> and the shift score less than a strong one.

The card anchors both ends. It says what untouched readings look like and what a
strong push looks like. It says nothing about faint pushes beyond "less". A big
shift with no similarity near 1 is, by the card, a push in some other direction.

The exact wording of both templates is fixed in the builder and tested (§7). Any
wording change after this commit and before the push is recorded under
build-time clarifications.

## 6. The bars Jev is compared with

- **The fitted bar.** One classifier serves both presentations: it is the program's
  best decoder of this readout. It is a multinomial logistic regression (C = 10,
  L-BFGS, standardised features) on functions of the **displayed digest only**:
  the 13 similarities, the 13 similarities × the shift, the shift, and log shift
  (the shift floored at 0.01). It is trained on the reference half:
  - **Split.** A seeded permutation sets aside ⌊129/4⌋ = 32 states for
    calibration and trains on the other 97.
  - **Training stimuli.** For every training state: untouched, all 13 concepts at
    α 0.04, 0.06, 0.09, 0.15 and 0.5, and two random pushes at each of those
    doses.
  - **Weights.** Each state's weights follow the flight's recipe (untouched 1;
    random at 0.5, 1; random at each threshold dose, 1/3; concept pushes 2/13 at
    each of 0.04–0.15 and 1/13 at 0.5).
  - **Temperature.** Chosen on the calibration states by bounded one-dimensional
    minimisation of weighted log-loss over [0.05, 20].
  - **Frozen.** Its parameters, for both halves, go into `FROZEN_J3.json` before
    any call.

  Design check (§3.4): on a flight-shaped mixture it names 8.1% of untouched
  readings and 0.8–2.3% of random pushes at 0.5, with 1.000 at the ceiling. On the
  threshold pool its accuracy is 0.80–0.82, ECE 0.012–0.023, AUROC<sub>2</sub>
  0.84–0.85 and AURC 0.053–0.058.
- **Two card rules** (secondary, zero-shot, deterministic). Rules a careful
  reader could take from the card:
  - (i) name the top-similarity pattern if the shift exceeds 5, else none;
  - (ii) the same, but only if the top similarity also exceeds 0.8.

  Design check: rule (i) names 11% of untouched readings, reaches 0.64–0.65 on
  threshold pushes and names 100% of random pushes at 0.5. Rule (ii) names 0%,
  0.40–0.41 and 3.5–4.3%.
- **The historical reference** (cited, not flown): E8-R's trained verbal readout
  scored 18/18 on trained concepts, 0/24 false alarms on shams, and 6/24 on
  held-out concepts. The input and the method differ; the question is the same.

## 7. Stages

1. **Register.** This document and the design and power checks are committed
   before any J3 builder code exists. They are pushed to the public repository
   (GitHub + Codeberg) with the builder, before the first call. No J3 call of any
   kind, to any model, is made before the push.
2. **Build and freeze.** The builder is `jev/j3*.py`. It generates the stimuli,
   renders both presentations and the mouth prompt, fits and freezes the fitted
   bar, runs both runners, and does the analysis. It comes with three test
   suites:
   - **Logic:** gate G6′ (§8), the counts and balances of §4, and determinism.
   - **Capture:** the Jev runner is J1 + J2's; the mouth client gets its own
     tests, including one that it refuses to write the key.
   - **Verdict:** the full analysis on synthetic responses from four planted
     agents, which must each land on their expected branch before any real call
     is analysed:
     - *twin*: the fitted bar's own distribution, jittered; all three primaries
       PASS;
     - *degraded*: the fitted bar on noisier digests; P3 FAIL;
     - *claimer*: always names the top pattern; P1 FAIL, claims at ceiling;
     - *top-looker*: names the top pattern whenever the shift exceeds 6; P1 FAIL,
       names the push, not the pattern.

   The freeze file is committed and pushed with the builder.
3. **Smoke test** (§7.1): about 640 Jev calls and 20 mouth calls.
4. **Flight:** 18,576 Jev calls, then 1,548 mouth calls. Resumable, append-only,
   budget-capped.
5. **Recompute from raw** in a fresh process, from the stored responses only.
   Every number in the verdict must match to the last digit. Then **lock**.

### 7.1 Smoke test (after the push, before the flight)

Smoke stimuli come from their own seed stream. Smoke calls never enter the
analysis.

- **Determinism (the replicate floor):** 12 stimuli per presentation, each sent
  as the identical request 10 times (240 calls). Reported: choice-flip rate,
  mean |Δp| and mean total-variation distance between repeats.
- **Throughput:** 100 calls at concurrency 16 (J1 + J2 saw no 429s up to 16).
  If any 429 or a p95 latency over 2 s appears, the flight steps down to 8.
- **Isolation of the noul:** 30 stimuli × 5 repeats, sent twice: `decision` alone,
  and bundled with `pushed` (300 calls). If the mean |Δc| between bundled and
  alone exceeds the replicate floor by more than 0.02, with a bootstrap CI
  excluding zero, `pushed` moves to separate calls in the flight. This is J1's
  rule and statistic (J1 + J2 clarification 9).
- **Build pin:** the build string of the first response is recorded and enforced
  from then on (G1).
- **The mouth:** 20 digest stimuli, 5 of them sent 5 times identically. It must
  parse as JSON with a valid option key and a confidence in [0, 100] on at least
  19 of the 20. Reasoning tokens must be absent, and the repeat agreement is
  reported. If the parse bar fails, the mouth rider isn't flown, and this is
  reported. The primaries are untouched.
- **Cost check:** the observed cost per call, extrapolated, must fit under G7.

## 8. Measures and pass bars

Everything below is computed **per presentation** (RAW and DIGEST separately),
over calls. The fitted bar's prediction for a stimulus is counted once per call,
so it enters three times, matching Jev's composition. Uncertainty always comes
from a **cluster bootstrap by base state**: 258 clusters, 10,000 resamples,
seeded, percentile intervals. "Upper" and "lower" mean the ends of the two-sided
95% interval.

**Gates (instrument, not results).**

- **G1–G4 and G7:** as J1 + J2 §3.
  - G1: the build pin.
  - G2: schema; a choice answer carries all 14 options, probabilities sum to
    within [0.97, 1.03], and the choice is an option.
  - G3: the argmax rate is reported.
  - G4: 100% of requests regenerate to their recorded sha256.
  - G7: stop at a cumulative **$5** of J3 spend, counting the smoke test, the
    flight and the mouth, with J1 + J2 spend excluded.
- **G6′ information identity** (test suite, before any call), on a 500-stimulus
  sample:
  - (a) a text-only parser recovers every displayed number from both
    presentations exactly;
  - (b) the digest recomputed from the parsed RAW reading, with the undisplayed
    reference covariance, matches the DIGEST display to within 0.01 (display
    rounding);
  - (c) the fitted bar's features, computed from the parsed DIGEST text, equal
    the features it is scored on.

  This proves the fitted bar and Jev read the same reading.
- **G-CEIL, can it read this presentation?** Jev's accuracy on ceiling concept
  pushes (α = 0.5) must reach **≥ 0.50** (chance is 1/14; the fitted bar scores
  1.000). This gate governs P1 only. A reader that can't name a strongly pushed
  pattern gives its silence no meaning.
- **Missing calls:** up to 5 retries. More than 2% missing in any presentation ×
  stimulus type is flagged in the verdict.

| | Reading | Pass bar | Failure branches |
|--|--|--|--|
| **J3-P1** | Silence | G-CEIL passes, and both clauses pass: **(a)** excess false alarms at its own hit rate, upper ≤ 0.10; **(b)** random pushes at the ceiling named, upper ≤ 0.15 | G-CEIL fails: **NOT ADJUDICABLE, "does not read this presentation."** Point share of untouched readings named ≥ 0.50: **"claims at ceiling"** (E7-Q's pathology). (b) lower > 0.15: **"names the push, not the pattern."** (a) lower > 0.10: **"leaky silence."** Otherwise INCONCLUSIVE |
| **J3-P2** | Calibration | On the threshold pool: **(a)** ECE upper ≤ 0.10; **(b)** AUROC<sub>2</sub> > 0.5, bootstrap P(AUROC<sub>2</sub> ≤ 0.5) < .01 | (b) fails: **"confidence carries no information."** (b) passes and ECE lower > 0.10: **"confident, not calibrated."** Otherwise INCONCLUSIVE |
| **J3-P3** | Earns its place | On the threshold pool: ΔAURC = AURC(Jev) − AURC(fitted bar), upper ≤ **0.05** | Lower > 0.05: **"doesn't earn its place"** (the scoping's "expensive argmax"): Jev leaves the instrument chain for this presentation. Otherwise INCONCLUSIVE |

**P1's branches are checked in this order:** G-CEIL; claims at ceiling;
names the push, not the pattern; leaky silence; PASS; INCONCLUSIVE. The first that
applies is the verdict, and every clause's numbers are reported whatever the
verdict. P2 and P3 have no gate.

**Definitions.**

- **P1a.** Three quantities, all over calls:
  - FA<sub>J</sub>: the share of untouched calls on which Jev names a pattern;
  - H<sub>J</sub>: the share of threshold concept-push calls on which it names
    any pattern (a hit counts whether or not the pattern is the right one);
  - FA<sub>fit</sub>(H<sub>J</sub>): the fitted bar's false-alarm share when its
    detection score, 1 − P<sub>fit</sub>(none), is thresholded to give the same
    hit share on the same calls. The threshold *t* is the (1 − H<sub>J</sub>)
    quantile of the scores on threshold concept-push calls, and FA<sub>fit</sub>
    is the share of untouched calls with score ≥ *t*. If H<sub>J</sub> = 0, then
    FA<sub>fit</sub> = 0.

  Excess = FA<sub>J</sub> − FA<sub>fit</sub>(H<sub>J</sub>). Each bootstrap
  resample recomputes all three. This compares Jev's silence with the best
  available detector's at the hit rate Jev itself chose. Without it, a reader
  could pass by saying "none" to everything, and the optimal detector could fail
  just because weak pushes overlap with untouched readings.
- **P1b.** The share of random-push calls at α = 0.5 on which Jev names a
  pattern.
- **P2.**
  - **ECE:** calls in the pool are sorted by *c*, with ties kept in plan order,
    and split into 15 near-equal groups. ECE is Σ (group size/n) · |mean *c* −
    accuracy|.
  - **AUROC<sub>2</sub>:** the Mann–Whitney AUROC of *c* for correct against
    incorrect calls, with ties counted as half.
- **P3.**
  - **AURC:** the mean selective risk over coverage 1/n … n/n, with calls ranked
    by confidence, highest first. Within a group of tied confidences, cumulative
    errors are interpolated linearly, which is the exact expectation under random
    tie-breaking.
  - **Confidences:** Jev's is *c*; the fitted bar's is its maximum probability.
    Both are scored on the same calls.
  - **Correct:** the choice equals the answer in §4.

**How the branches separate** (the power check, §12, shows each case):

- A reader that integrates the evidence like the fitted bar passes all three.
- A reader that names a pattern whenever the reading moves fails P1, on
  "claims at ceiling" if it also names on untouched readings, and on "names the
  push, not the pattern" if it only names on large shifts.
- A reader with a constant confidence fails P2(b).
- A reader that is informative but noisier than the bar fails P3.

**The claim level, registered in advance.** A pass on all three says Jev can
read this readout zero-shot, from a reference card, about as well as a fitted
classifier: a usable instrument for this program. No result here bears on
experience, whether the language model's or Jev's.

## 9. Secondaries (pre-named; no multiplicity claim)

- **S-J3-1 Psychometric curves:** accuracy and the named-any share by dose (0.04,
  0.06, 0.09, 0.15, 0.5) and on untouched readings, per presentation. Shown for
  Jev, the fitted bar, the two card rules, and the mouth.
- **S-J3-2 Excess AURC.** E-AURC is AURC minus the oracle AURC at the same
  accuracy, so ranking is read apart from accuracy. Also ECE on 5, 10, 15 and 20
  bins, and ECE and AURC per dose.
- **S-J3-3 Presentation and confidence, J1-P2's logic** (paired by stimulus, on
  the threshold pool):
  - accuracy(DIGEST) − accuracy(RAW);
  - overconfidence(RAW) − overconfidence(DIGEST), where overconfidence is mean
    *c* − accuracy.

  If raw costs accuracy and confidence doesn't follow it down, confidence is
  reading the page, not the competence. This is described, not tested.
- **S-J3-4 Presence without identity** (E8-R2's dissociation, in a second
  system). The AUROC of P(yes, `pushed`) for random pushes at 0.5 against
  untouched readings, and for threshold concept pushes against untouched
  readings. Also the share of random-push calls at 0.5 on which Jev chooses
  `none` **and** gives P(yes) ≥ 0.5: presence flagged, no pattern named.
- **S-J3-5 Nearest-name reading** (E8-R2's mechanism). Among wrong pattern names
  on concept pushes, the share that name the truth's nearest neighbour, the
  pattern whose reference effect is most similar to the truth's (whitened
  cosine, rank 1). Compared against 1/12.
- **S-J3-6 The mouth.** DeepSeek V4 Flash 0731 on the DIGEST presentation. Its
  stimuli are all 12 stimuli of every second state in the seeded order: 129
  states, 1,548 stimuli, one call each, with repeat 0's option order.
  - Its confidence is its stated confidence ÷ 100.
  - P1a, P1b, P2 and P3 are computed the same way on its calls.
  - It is compared with Jev's repeat-0 DIGEST calls on the same stimuli and with
    the fitted bar.
  - An unparseable answer is retried once, then counted invalid; invalid calls
    are reported and excluded.
- **S-J3-7 The vendor's confidence field** against (14·*c* − 1)/13: the share
  within ±0.015.
- **S-J3-8 Option position:** P(choosing presented position *j*) on wrong calls.
- **S-J3-9 Replicate agreement:** how often the three option orders give the same
  modal choice, per stimulus type.
- **S-J3-10 Half symmetry:** every primary statistic by reference half (A read
  with B's card, and B with A's).

## 10. Analysis conventions

- Probabilities are used as returned, rounded to 0.01. AUROC ties count half.
- The bootstrap is seeded. Its cluster is the base state. The fitted bar is
  deterministic given the freeze file.
- The primaries are read per presentation, each with its own branch. There is no
  omnibus test and no correction across presentations. The J4 fork (§13) states
  how the two presentations combine.
- Only calls that pass G1, G2 and G4 enter any primary. G-CEIL uses the ceiling
  concept-push calls only.
- The analysis code is written and tested on the planted agents before any real
  call is analysed. It is committed with the flight data.
- BLAS runs single-threaded, so a fresh-process recompute is byte-identical
  (lesson from J1 + J2).

## 11. Honest expectations

- **RAW probably fails P3, and may not pass G-CEIL.** Matching a reading against
  a table of 13 patterns in 14 channels means comparing numbers without the
  covariance. In J1-F2, which had four candidates, Jev reached 0.975 on the list
  layout at low noise but fell to 0.375 where the ideal observer still gets about
  0.77.
- **DIGEST: P1a probably passes.** Untouched readings match the card's untouched
  description, and J2 found that Jev holds when the evidence is empty. **P1b is
  the sharp test.** A random push at the ceiling looks like a strong push on the
  shift score but has no similarity near 1. Naming the top pattern there is the
  top-looker's failure, and a natural one for a reader that follows the shift.
- **DIGEST P2 is open.** J1 found Jev overconfident by 0.10 on clean pages.
- **DIGEST P3 probably fails, or lands INCONCLUSIVE.** The fitted bar integrates 28
  features learned from thousands of labelled stimuli. Jev has a card. A pass
  would be a real surprise and would earn Jev its J4 arm.
- **The mouth** probably reads the card more literally than Jev does. Its verbal
  confidence is expected to be coarse (multiples of 5 or 10).

## 12. Power (`jev/power/j3_power.py`, output `j3_power_output.txt`)

Four planted agents were run on three simulated flights of the registered shape,
each with 258 states, 3,096 stimuli and 9,288 calls per presentation, using
1,000 cluster-bootstrap resamples (the registered analysis uses 10,000).
Probabilities were rounded to 0.01.

- **Twin:** P1a excess −0.006 to +0.001 (CI half-width ≈ 0.015); P1b 0.4–1.6%;
  ECE 0.024–0.027 (upper ≤ 0.044); AUROC<sub>2</sub> 0.83–0.85; ΔAURC +0.0006 to
  +0.0019 (CI half-width ≈ 0.003). **All three PASS**, in 3 of 3.
- **Degraded:** ΔAURC +0.068 to +0.071, CI ±0.008. **P3 FAIL** in 3 of 3.
- **Claimer:** names 100% of untouched readings. **P1 FAIL, "claims at
  ceiling"**; P2(b) fails, with constant confidence.
- **Top-looker:** P1a excess +0.043, upper 0.070, so clause (a) passes; P1b 1.000.
  **P1 FAIL, "names the push, not the pattern."**

The design resolves the P3 margin with room to spare. The verdict suite (§7)
repeats these on the builder's own code path.

## 13. Kill and fork, pre-stated

- **G1 build change during the flight:** pause, then smoke-test the new build and
  continue only by amendment.
- **The J4 fork.** Jev enters J4 as the state-reading examiner arm **if and only
  if P1 and P3 both PASS on at least one presentation.** J4 then feeds it that
  presentation. If P2 fails there, the J4 arm uses Jev's choices, but its
  confidence gates nothing. Otherwise the program keeps its fitted decoders, and
  Jev stays a subject only, as the scoping document said.
- **DIGEST P1 FAIL "names the push, not the pattern":** Jev's silence can't be
  trusted on off-pattern movement. Any later use must pair it with a
  fitted presence detector.
- **G-CEIL fails on RAW:** Jev doesn't read raw readouts of this kind. Any J4 feed
  must be digested.
- **Any primary NOT ADJUDICABLE or INCONCLUSIVE:** reported as such, with the
  reason. No re-flight without an amendment.

## 14. Ops

- **Code:** `jev/j3.py` (data, reference cards, stimuli, rendering, fitted bar),
  `jev/j3_plan.py`, `jev/j3_mouth.py` (the chat-completions client), and
  `jev/analyze_j3.py`. The Jev runner is J1 + J2's `jev/runner.py`, unchanged.
  Results go under `jev/results/j3_<stage>_<stamp>/`.
- **Seeds:** the flight stream begins `[20261001, …]`, the smoke stream
  `[20261002, …]`, and the fitted bar's training draws `[20261003, …]`. The
  remaining words name the stage part, half, state and stimulus.
- **Keys:** `OPENROUTER_API_KEY` from the environment or `~/.env`. It is never
  logged, echoed or written.
- **Publication:**
  - **Public:** the protocol, the freeze file, the plans, every raw response from
    every stage (the smoke test and the mouth included), the analysis code and
    the verdict.
  - **Already public:** the input arrays: the E7b-Q and E8-N v2 bundles and the
    payload.
- **Pre-ranked trims** (if cost or the rate limit bites): the mouth rider 1,548 →
  774; then the RAW presentation's psychometric stimuli (α 0.15). The primaries'
  pool and the ceiling stimuli are never trimmed.

## 15. Differences from the scoping document (2026-09-24)

1. **The doses were re-titrated to the fitted bar**, which a nearest-signature
   decoder badly underestimates (§3.4): threshold 0.04, 0.06 and 0.09, plus 0.15
   and a 0.5 ceiling, in place of 0.06, 0.10 and 0.15 plus 0.5.
2. **The digest's similarities are whitened** by the reference covariance, and
   the state carries a **reference card** with untouched and strong-push anchors.
   This makes "none" decidable from the page, which a zero-shot reader needs.
3. **States are deduplicated** (258 unique of 320) and **cross-fitted** across
   halves split by conversation replicate. The scoping probe split at random.
4. **One fitted bar serves both presentations.** It is the program's best decoder
   of the readout, and its information is shown to be what the digest displays
   (G6′). The scoping's nearest-signature decoder is replaced by two card rules,
   as secondaries.
5. **P1 made operational.** "False alarms at the hit rate matched to the fitted
   classifier" becomes the excess over the fitted bar at *Jev's own* hit rate. The
   random-push clause is read at the ceiling only, because at threshold doses a
   random push barely moves the register.
6. **P2 and P3 keep the scoping's bars**, with three-valued verdicts on bootstrap
   intervals.
7. **Added:** the `pushed` noul (presence without identity, E8-R2) and the
   presentation–confidence secondary (J1-P2's lesson).
8. **The mouth** is pinned to DeepSeek V4 Flash 0731, runs on the DIGEST
   presentation only, over 1,548 stimuli, with reasoning off, so that it is a
   fast verbal report like Jev's fast decision.
9. **Deferred to J3b:** the stated-reliability secondary. **Considered and not
   used:** off-list concept foils. The eight probe directions available overlap
   the listed patterns: activation-space cosine up to 0.85, register-effect
   cosine up to 0.92. A "none" answer to them would be ambiguous.

## Amendments

*(None. Dated entries go here.)*
