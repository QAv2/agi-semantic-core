---
title: "W3 design check: does the loop know where it is?"
subtitle: "Ideation experiment 2, checked against the prior art and simulated before registration"
date: "2026-10-11 · Opus 5.5 · ORC lane · before registration, before build · nothing trained, nothing spent"
---

## The short answer

- **The question as first posed is already answered.** Fable asked whether a diffusion network
  keeps its own clock: take away the label that tells it which denoising step it is on, and
  does it still know where it is? It does, and trivially. The noise level is visible in the
  noisy image itself. Sun, Jiang, Zhao and He (ICML 2025) show this for image models, and Lu
  (2026) shows a linear probe reading the step at every depth of a large model trained with no
  label. Our own check (§2) agrees: an *untrained* network reads the step of a noisy image to
  within a median of 48 steps out of 1,000 on MNIST and 62 on CIFAR-10, and fourteen simple
  image statistics do better still (23 and 38).
- **What is still open is the conflict.** Give a network that does get the label a label that
  says one step while the image is really at another. Inside the network, does its sense of
  position follow the label or the content? The output-level answer is "some of both" (Yu et
  al., CVPR 2026). Nobody has measured it inside the network. That is the program's standing
  question (does a system's internal read follow what it is told, or what is there?), asked of a
  system with no mouth and an exact right answer at every step.
- **W3 measures it as one number per layer, the split weight w**: 1 when the internal read
  follows the content, 0 when it follows the label. The simulation in §3 shows the statistic
  is unbiased, and with two gates and a level check it calls every planted world correctly,
  including the two ways a network can break when pushed off its training distribution.
  **Superseded at registration (§8):** the probe statistic is confounded when the label is an
  exact code and the content a noisy one, which is W3's situation. The registered primary is a
  probe-free movement share; the probe statistic stays as a secondary.
- **A label sham** checks that a label-following read follows the label's *meaning*, not any
  disturbance of the label channel.
- **The arm itself is cheap**: a few thousand forward passes. The cost is training the models,
  one to three T4 hours, set by a five-minute timing smoke that is built and staged.

## 1. What the sweep changed

The prior-art sweep is `docs/LOOP_PRIOR_ART_2026-10-11.md` (research worker, 2026-10-11).

| seed (ideation doc §3.2) | W3 | why |
|---|---|---|
| clock off: hold the label constant or scramble it at inference | **trained twins**: a clock-off model trained without the label, as a replication; the headline moves to the conflict | a network given a constant or random label is out of distribution and its outputs break (Yu et al. 2026, Li et al. 2024); the clock-off question is answered (Sun et al. 2025, Lu 2026) |
| displacement: swap in a state from another step; does the read follow the truth or the schedule? | **a 2 × 2 of content level × label level** inside a clock-on network; the split weight w | a denoising step carries nothing forward except the image, so "follows the schedule" can only mean "follows the label" |
| sham displacement: same-level content from another trajectory | **a label sham**: the label moved by a random vector of the same size as the real move | with a memoryless step, a same-level content swap changes nothing about position by construction; the open sham is in the label channel (the nearest prior is Rulli et al. 2026's norm-matched random steering) |
| probes read position | **every read must beat the input** | the noise level is in the pixels (§2 below); a probe that only matches a random-init network has found the input, not a clock (J1's input-only null) |
| secondary: seed-spread and guidance probes; inversion as the walk's positive control | **seed spread kept** as a secondary, with t partialled out; **guidance and inversion deferred** | guidance needs a class-conditional model with guidance, a second build; inversion is the walk's calibration and stands as its own piece |

## 2. How much of the clock is in the input

`windows/power/w3_design_check.py` (§1; outputs `w3_design_check_s1_{mnist,cifar10}.{txt,json}`).
For 6,000 images per dataset, noise is added at a random step, and a ridge probe is trained on
4,000 of them to read the noise level (log-SNR) back. It is scored on the other 2,000. Three
kinds of features, none from a trained network:

- **pixels**: the raw noisy image;
- **hand**: 14 numbers per image, its overall spread, edge energy, and power in 8 frequency rings;
- **random-init net**: the encoder of the candidate U-Net with random weights and no label,
  mean-pooled at its mid-block (three seeds; they agree within .01).

Cosine schedule (the one W3 would use); R² of the read step, and the median error in steps out
of 1,000:

| features | MNIST R² | MNIST median error | CIFAR-10 R² | CIFAR-10 median error |
|---|---|---|---|---|
| pixels | .930 | 57 | .082 | 229 |
| hand (14 numbers) | **.984** | **23** | **.948** | **38** |
| random-init net | .941 | 48 | .879 | 62 |

Median error by band of steps, CIFAR-10, cosine:

| steps | 0–50 | 50–100 | 100–200 | 200–400 | 400–600 | 600–800 | 800–900 | 900–1000 |
|---|---|---|---|---|---|---|---|---|
| hand | 6 | 14 | 26 | 45 | 46 | 69 | 30 | 38 |
| random-init net | 26 | 18 | 46 | 58 | 126 | 80 | 33 | 112 |

What this settles:

- **The position is in the input.** Fourteen summary numbers read the step of a noisy CIFAR
  image to a median of 38 steps out of 1,000. Any claim that a network "knows where it is" from
  a matched input has to beat that, or it has found the input. The floor is the **hand
  statistics**, the stronger of the two baselines.
- **Raw pixels are no floor.** On natural images the noise level is a property of the pixels'
  spread, not of any one pixel, and a linear read of raw pixels misses it (R² .08). A pixel
  probe would make every network look like it keeps a clock.
- **The input is a weaker witness in the middle of the process.** For the random-init network
  the error is 126 steps around steps 400–600 and 18–33 steps near the ends. If the trained
  network leans on the label anywhere, the middle is where to expect it. That is why the split
  is also reported by band.
- **For the split arm,** the content carries strong evidence of position. A pure LABEL verdict
  would mean a network ignoring evidence that a 14-number summary can read. That expectation
  is written down here, before any training.

## 3. The split weight and its teeth

`windows/power/w3_split_check.py` (outputs `w3_split_check_output.{txt,json}`). Simulation only.

**The measurement.** Each test image is noised to two steps, t and t′ = t + offset (offsets of 25,
100 and 300 steps), and shown to the network four ways: content at t with the label t, content
at t′ with the label t′ (the matched pair), and the two mismatches, content at t′ with label t
and content at t with label t′. A probe trained only on matched inputs reads the position at a
layer. The split weight is

> w = slope through the origin of [ read(content c, label l) − g(level of l) ] on
> [ g(level of c) − g(level of l) ]

where g is the probe's own calibration curve on matched inputs. **w = 1:** the read sits where
the content puts it. **w = 0:** it sits where the label puts it. Two choices keep w honest:

- **g instead of the true level.** A ridge probe shrinks its reads toward the middle. g measures
  that shrinkage, and w is scored against it.
- **A fitted g, not each image's own matched reads, in the denominator.** Noisy reads in the
  denominator would pull w toward 0, which is "label", an error that flatters one answer.

**The verdict** at the mid-block, from 95% bootstrap intervals over images: **CONTENT** (lower
end ≥ .8), **LABEL** (upper end ≤ .2), **SPLIT** (the whole interval inside .2–.8), **UNCLEAR**
otherwise. Two gates and a level check come first.

- **Tracking gate.** Fit the mismatched reads on both levels at once, read = α + a·g(c) + b·g(l).
  For any mix of content and label, a + b = 1. For reads that have stopped being positions,
  a + b = 0. If the interval for a + b reaches below .8, the verdict is **UNREADABLE**.
- **Precision gate.** If the mismatched reads scatter around the fitted mix more than 1.5 times
  as much as matched reads scatter around g, the verdict is **UNREADABLE**.
- **Level check.** w is also fitted within four bands of steps. If two bands get different
  decisive verdicts with non-overlapping intervals, the verdict is **VARIES BY LEVEL**, and the
  band verdicts are the finding.

**The first pass found three holes, all fixed before the run below.** (1) A network whose
mismatched reads collapse to the middle was called SPLIT .5 when the probe was noisy (that is
why the tracking gate exists). (2) A network that follows the label at some levels and the
content at others came out UNREADABLE or SPLIT (that is why the level check exists). (3) The sham
statistic, pooled over offsets, was swamped by the 25-step class (§4).

**Planted worlds** (200 simulated experiments per cell, 500 bootstrap draws each). σ is the
probe's read noise in log-SNR units. For scale, the input-level probes of §2 have about 0.85–1.1
on MNIST; a trained network's probe should do no worse.

| world | what it plants | right verdict, n = 128 per offset class (σ ≤ 1) | right verdict, n = 256 (σ ≤ 2) |
|---|---|---|---|
| CONTENT | read follows content | 1.00 | 1.00 |
| LABEL | read follows label | 1.00 | 1.00 |
| SPLIT .5 | half and half | 1.00 | 1.00 |
| SPLIT .3 / .7 | near the bars | .995 | .965 |
| COLLAPSE | reads fall to the middle | 1.00 (UNREADABLE) | 1.00 |
| SCATTER | reads lose precision 3× | 1.00 (UNREADABLE) | 1.00 |
| BANDED | label below step 500, content above | 1.00 (VARIES) | .97 |
| BANDED300 | the same, boundary inside a band | .995 (VARIES) | .955 |

- **w is unbiased in every world**: medians .00, .30, .50, .70 and 1.00 where those were planted.
- **False VARIES** in the five worlds with no level dependence: at most 1.5% of experiments in
  any cell.
- **Choice: 256 test images per offset class.** It holds every world at ≥ .95 even at σ = 2.
  The cost is forward passes, a few seconds.

## 4. The label sham

The question it answers: if the read follows the label, does it follow the label's *meaning* (a
different step), or does it move whenever the label channel is disturbed at all? For each
image, the label vector is moved by a random direction with the same length as the real move
from step t to step t′. A deterministic network reads the same input the same way every time,
so any shift under the sham is caused by the sham.

**Statistic**, per offset class: S = mean |shift under the sham| / mean |shift under the real
label move|, with the content held fixed. **SILENT** when the 95% upper bound of S is ≤ 0.25 on
the **300-step class**, the primary. S is judged only when the label moves the read at all
(verdict LABEL, SPLIT or VARIES).

Simulated with the sham moving the read k times as far as the real move, plus incidental
jitter ("shift noise", log-SNR units), 128 images per class:

| real w | shift noise | k = 0 | k = .1 | k = .3 | k = .5 |
|---|---|---|---|---|---|
| 0 | .05 | SILENT 1.00 | 1.00 | 0.00 | 0.00 |
| 0 | .25 | 1.00 | 1.00 | 0.00 | 0.00 |
| 0 | .50 | 1.00 | 1.00 | 0.00 | 0.00 |
| .5 | .05 | 1.00 | 1.00 | 0.00 | 0.00 |
| .5 | .25 | 1.00 | .99 | 0.00 | 0.00 |
| .5 | .50 | 0.00 | 0.00 | 0.00 | 0.00 |

(The rate shown is P(SILENT) on the 300-step class.)

- **It separates a quiet sham (k ≤ .1) from a reactive one (k ≥ .3) cleanly**, as long as the
  jitter is small next to the real shift.
- **It fails safe.** When the jitter is large next to the real shift (w = .5, jitter .5), it
  withholds SILENT even from a quiet sham. A noisy network can't earn SILENT, and it can't fake
  it either.
- **Why only the 300-step class.** On the 25-step class the real move is tiny, so any jitter
  dominates (S .51–.93 at k = 0 with jitter ≥ .25). Scored there, the sham would measure noise,
  not reaction. The 25- and 100-step classes are reported, not judged.

## 5. Budget and the timing smoke

The timing smoke (`colab/W3_TIMING_UI.ipynb`, built by `colab/build_w3_timing_notebook.py` from
the same model code; run by Joe on a Colab T4, 2026-10-11 03:56Z; result
`windows/power/w3_timing/timing_20261011_035637.json`) timed 60 training steps at batch 128 with
fp16 autocast, after 15 warm-up steps, on random tensors:

| data | width C | label | parameters | s per step | epochs per T4 hour | forward s per 128 | peak memory |
|---|---|---|---|---|---|---|---|
| MNIST | 32 | on | 1.6 M | 0.094 | 82 | 0.030 | 1.2 GB |
| MNIST | 48 | on | 3.6 M | 0.140 | 55 | 0.050 | 1.9 GB |
| MNIST | 64 | on | 6.5 M | 0.178 | 43 | 0.067 | 2.5 GB |
| MNIST | 64 | off | 5.9 M | 0.177 | 44 | 0.066 | 2.5 GB |
| CIFAR-10 | 32 | on | 1.6 M | 0.089 | 103 | 0.032 | 1.3 GB |
| CIFAR-10 | 48 | on | 3.7 M | 0.146 | 63 | 0.052 | 1.9 GB |
| CIFAR-10 | 64 | on | 6.5 M | 0.187 | 49 | 0.070 | 2.5 GB |
| CIFAR-10 | 64 | off | 5.9 M | 0.186 | 50 | 0.069 | 2.5 GB |

- **Proposed: CIFAR-10 at width 64** (6.5 M parameters), about 49 epochs per T4 hour. The
  label costs nothing measurable in speed, so the twins train in equal time.
- **Training dominates the cost.** The split arm, the sham and the calibration together are
  about 5,000 forward passes per model, about 40 batches, under 5 seconds. Seed-spread
  sampling (8 continuations × 50 steps × a few hundred states) is minutes.
- **Total:** three models at about 1.2 T4 hours each (60 epochs), about 4 T4 hours. That is a
  small fraction of the 473 compute credits Joe has.
- **The FLOP extrapolation was optimistic.** `windows/power/w3_budget_check.py` counts 8.7
  GFLOP per image per training step at width 64. The measured 0.187 s per step of 128 images
  is about 6 TFLOP/s effective. The assumed range was 8–15, so the extrapolation promised
  66–125 epochs per hour against the 49 measured. The registration uses the measurement.
- The first run of the smoke failed on a bug that the local CPU test could not catch: the
  label encoder built a tensor on the CPU. The bug was fixed at the source, then checked by
  running every configuration forward and backward on PyTorch's `meta` device, which rejects
  any stray CPU tensor. The second run is the one above.

## 6. The design this check proposes for registration

**Substrate.** The §5 U-Net at 32 × 32: width 64, multipliers 1-2-2, two residual blocks per
level, attention at 8 × 8 and in the mid-block. Cosine schedule, T = 1,000, trained to predict
the noise. **CIFAR-10**, because the content is a weaker witness of position there than on
MNIST, and weakest in the middle of the process (§2). Every outcome has room.

**Three models**, with the same architecture, seed and data order, trained for the same number
of epochs:

- **M-on**: gets the label (the step, encoded and added inside every residual block). This is
  the primary, the ordinary design in the field.
- **M-drop**: the same, but the label is replaced by a learned "no label" vector on 20% of
  training examples. Secondary. A missing label is then something the network has seen in
  training, not a broken input.
- **M-off**: no label at all. Secondary. It replicates the known result (Sun et al. 2025; Lu
  2026) on our own substrate.

**Training gate,** before any probe: in every band of steps, each model's held-out noise
prediction must beat both trivial predictors (all zeros, and the noisy input itself). A probe is
never read on a network that has not learned to denoise.

**Probes.** Ridge regression on mean-pooled activations at every resolution level of the
encoder, at the mid-block (the primary site) and at every level of the decoder. Target:
log-SNR. Probes are trained only on matched inputs (content level = label level) from training
images, calibrated (g) on a separate split, and scored on held-out test images.

**Arms.**

1. **Input floor.** The hand-statistics probe and each model's random-init twin, on the same
   images, reported by band. M-on's reads of matched inputs are calibration, not findings,
   because the label is part of the input.
2. **The split (primary).** M-on; offsets of 25, 100 and 300 steps, both directions;
   **256 test images per offset class**; the same noise draw across each image's 2 × 2. The
   statistic is w at the mid-block, with the gates and verdicts of §3: CONTENT · LABEL ·
   SPLIT · VARIES BY LEVEL · UNCLEAR · UNREADABLE.
3. **The label sham.** On the 300-step class: the label moved by a random vector the size of
   the real move. S ≤ 0.25 means SILENT. Judged when the primary verdict is LABEL, SPLIT or
   VARIES; reported otherwise.

**Secondaries** (named in advance, no multiplicity claim):

- w layer by layer, a profile through the U-Net;
- w by direction (content noisier than the label vs cleaner);
- whether w predicts the output's error, which is Yu et al.'s output-level measure, read here
  as the noise level implied by the network's own prediction;
- M-drop: w against M-on's w, and its no-label reads against the content;
- M-off: its read against the hand-statistics floor (replicating Lu 2026's "hidden states
  beat input statistics");
- seed spread: within narrow step bands, a probe from one run's mid-block to the spread of
  the final images across 8 continuations from the same state, against the Tweedie-moment
  baseline.

**What it can and cannot show.** It can show which source a small denoiser's internal read of
its own position follows when the two disagree, layer by layer and level by level. It cannot
show anything about persistence or stakes, because the network keeps nothing between
generations. The result is about this class of network at this size.

## 7. Decisions, settled 2026-10-11

Joe, on reading the recommendations: *"as usual I am good with your recommendations."* The
registration carries:

- **All three models fly together**: M-on (primary), M-drop (no-label vector on 20% of
  examples), M-off. Credits are not the constraint (473 on hand), and M-drop is the cleanest
  test of a missing label.
- **About 60 epochs each**, about 1.2 T4 hours per model at the measured 49 epochs per hour,
  about 4 T4 hours in all. The exact number goes in the registration. The training gate guards
  the floor whatever the number.
- **Seed spread stays a pre-named secondary.** Guidance strength and the inversion control are
  deferred to their own pieces.
- **Next:** the lane law. Registration pushed → builder and planted-agent suites → freeze →
  Colab smoke → flight → fresh-process recompute → lock.

## 8. Added at registration: the collinearity check

`windows/power/w3_collinear_check.py` (outputs `w3_collinear_check_output.{txt,json}`), written
2026-10-11 while drafting the registration, before anything was trained.

**The flaw.** The §3 probe is trained only on matched inputs, where the content level and the
label level are the same number. A feature that tracks the label and a feature that tracks the
content are then perfectly collinear in its training data, and ridge regression divides its
weight between them by how cleanly each predicts the target. A label code is exact. A content
code reads the level with per-image error: §2's fourteen statistics leave R² .84 of log-SNR on
CIFAR-10. The probe can lean on the label while the representation moves mostly with the
content. §3's planted worlds could not show this, because they planted the *reads*, not the
activations behind them.

**The check** plants activation worlds (an additive content code, label code, image variation
and noise-draw variation, 64 features) and runs the §3 probe statistic beside a probe-free one,
the **movement share**: of the network's own move between the matched inputs at t and t′, the
share that moving the content alone produces, measured along that move, pooled over both
directions. Pooled that way it is the content's Shapley share of the move, and the label's share
is exactly one minus it. Its gate is κ, the energy of the 2 × 2 interaction over the energy of
the move; the level check is §3's, with a floor (band shares at least 0.2 apart) because the
movement share's intervals are tight.

At 256 images per offset class (100 simulated experiments per world):

| world | what it plants | probe w | probe verdict | movement share w | movement verdict |
|---|---|---|---|---|---|
| CONTENT | content code only | +1.00 | CONTENT 1.00 | +0.99 | CONTENT 1.00 |
| LABEL | label code only | -0.00 | LABEL 1.00 | +0.01 | LABEL 1.00 |
| EQUAL | both codes, equal size | +0.51 | UNREADABLE 0.80 | +0.53 | SPLIT 1.00 |
| SMALL.15 | label code 0.15 the size of the content code, large variation elsewhere | +0.97 | CONTENT 0.91 | +0.98 | CONTENT 1.00 |
| SMALL.3 | the same at 0.3 | +0.92 | CONTENT 0.59 | +0.94 | CONTENT 1.00 |
| MISREAD.15 | label code 0.15 the size; the content code reads the level with per-image error 0.4 SD (about the input floor) | +0.24 | SPLIT 0.78 | +0.97 | CONTENT 1.00 |
| MISREAD.3 | the same at 0.3 | +0.08 | LABEL 1.00 | +0.91 | CONTENT 1.00 |
| MISREAD.3-lo | the same, error 0.2 SD | +0.24 | UNREADABLE 0.67 | +0.91 | CONTENT 1.00 |
| MISREAD-EQ | equal codes, error 0.4 SD | +0.01 | LABEL 1.00 | +0.51 | SPLIT 1.00 |
| MISREAD-LAB | content code 0.15 the size of the label code, error 0.4 SD | +0.01 | LABEL 1.00 | +0.07 | LABEL 1.00 |
| INTERACT | the label code's size depends on the content level | +1.00 | CONTENT 0.85 | +0.53 | VARIES 1.00 |
| COLLAPSE | mismatched inputs fall to one point | +0.50 | UNREADABLE 1.00 | +0.50 | UNREADABLE 1.00 |
| SCATTER | mismatched inputs gain noise as large as the codes | +0.50 | UNREADABLE 1.00 | +0.53 | UNREADABLE 1.00 |
| BANDED | label code below step 500, content code above (error 0.4 SD) | +0.56 | UNREADABLE 1.00 | +0.59 | VARIES 0.98 |

Verdict rates of the movement share at both sizes:

| world | right verdict | n = 128 per offset class | n = 256 | median κ | median band shares |
|---|---|---|---|---|---|
| CONTENT | CONTENT | 1.00 | 1.00 | 0.044 | 0.99 0.98 0.99 0.98 |
| LABEL | LABEL | 1.00 | 1.00 | 0.047 | 0.01 0.02 0.01 0.02 |
| EQUAL | SPLIT | 1.00 | 1.00 | 0.027 | 0.53 0.55 0.55 0.52 |
| SMALL.15 | CONTENT | 1.00 | 1.00 | 0.029 | 0.98 0.97 0.98 0.96 |
| SMALL.3 | CONTENT | 1.00 | 1.00 | 0.027 | 0.94 0.94 0.94 0.92 |
| MISREAD.15 | CONTENT | 1.00 | 1.00 | 0.046 | 0.96 0.96 0.97 0.96 |
| MISREAD.3 | CONTENT | 1.00 | 1.00 | 0.044 | 0.90 0.91 0.92 0.91 |
| MISREAD.3-lo | CONTENT | 1.00 | 1.00 | 0.043 | 0.91 0.92 0.92 0.90 |
| MISREAD-EQ | SPLIT | 1.00 | 1.00 | 0.028 | 0.48 0.54 0.55 0.51 |
| MISREAD-LAB | LABEL | 1.00 | 1.00 | 0.050 | 0.05 0.09 0.08 0.05 |
| INTERACT | VARIES BY LEVEL | 1.00 | 1.00 | 0.159 | 0.34 0.63 1.03 1.01 |
| COLLAPSE | UNREADABLE | 1.00 | 1.00 | 30.002 | 0.50 0.50 0.50 0.50 |
| SCATTER | UNREADABLE | 1.00 | 1.00 | 5.581 | 0.52 0.54 0.53 0.51 |
| BANDED | VARIES BY LEVEL | 0.97 | 0.98 | 0.034 | 0.14 0.72 0.98 0.97 |

**What it settles.**

- **The probe statistic fails where W3 will be.** With a content code that misreads the level by
  about the input floor and a label code 0.3 its size, the probe calls LABEL in every run, while
  the representation moves 91% with the content (MISREAD.3). With equal codes it calls
  LABEL where the truth is half and half (MISREAD-EQ). It measures which source gives the cleanest
  position code: a real quantity, but not the question.
- **The movement share calls every world right**, including the collapse and scatter failures
  (κ 30 and 5.6 against .03–.05 for additive worlds and .16 for the interacting one; the bar is
  0.5) and both level-dependent worlds.
- **The registration's primary is the movement share** at the mid-block, with the same verdict
  bars (0.2 / 0.8), the κ gate and the level check. The probe statistic is secondary S2, and the
  registration pre-reads what a disagreement between the two means.
- **Two rules changed during the check:** a world whose true share sat on the 0.2 bar was
  called VARIES for band shares of .16 and .24, which added the 0.2 floor (a 0.3 floor, tried
  first, cost the banded world its real .72-versus-.98 contrast); and the interacting world's
  right verdict is VARIES, since its label effect changes with level by construction.

