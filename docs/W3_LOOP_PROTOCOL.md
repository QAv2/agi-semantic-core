---
title: "W3: Does the loop know where it is? (pre-registration)"
subtitle: "A small image denoiser is told one noise level while its input sits at another. Which of the two does its inside move with?"
date: "2026-10-11 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law · seed: Fable 5.1's ideation doc, experiment 2"
---

**Status: REGISTERED; NOT YET BUILT.**
Authored 2026-10-11, before any W3 model is trained and before the W3 builder exists, per
lane law. Everything that decides the verdicts is fixed here. The builder writes
`windows/results/w3_freeze_<stamp>/FROZEN_W3.json` (the sha256 of every code file, of the
notebook and of its embedded code, and the seeds), and that file is committed and pushed
**before the smoke**. The registration locks at the flight's first training step. Later
changes are dated amendments at the foot of this file.

Parent: experiment 2, "Does the loop know where it is?", in Fable 5.1's ideation doc of
2026-10-10 (`docs/IDEATION_WINDOWS_2026-10-10.md`). W3 is the third "windows" piece: W1
asked an image model's voice, W2 (Joe's cold-observer seed, ideation doc §8) is not yet
registered, and W3 looks inside a network with no voice at all. It is the program's first
experiment that trains its own subject from scratch. It keeps the method of the J- and
W-series: registration, planted-agent suites, freeze, smoke, flight, a fresh-process
recompute, lock. The design check behind this registration is `docs/W3_DESIGN_CHECK.md` and
four scripts in `windows/power/` (`w3_design_check.py`, `w3_split_check.py`,
`w3_budget_check.py`, `w3_collinear_check.py`; outputs committed beside them). The prior-art
sweep is `docs/LOOP_PRIOR_ART_2026-10-11.md`. Nothing in the design check trains a model or
spends money, apart from one five-minute Colab timing run.

## 0. In plain words

A diffusion image model makes a picture by starting from pure noise and removing a little of
it at a time, over about a thousand steps. At each step the network sees the noisy image and,
in the usual design, a label saying which step it is on. Fable called that label the clock.
The noise level is also plain in the image itself. So the network has two sources for "where
am I in the process": what it is told, and what it sees.

Fable's question was whether the loop knows where it is without the label. The prior art
answers that, and the answer is yes, trivially: networks trained with no label work almost as
well, a probe reads the step from a label-free network at every depth, and fourteen simple
statistics of the noisy image read the step of a CIFAR-10 image to within a median of 38
steps out of 1,000 (design check §2).

What nobody has measured is the conflict. Give a network that does use the label a label for
one step and an image at another. Inside the network, does its sense of position move with
the label or with the content? The output-level answer is "some of both" (Yu et al., CVPR
2026). W3 measures it inside. That is the program's standing question (does an internal state
follow what the system is told, or what is there?), asked of a system with no mouth and an
exact right answer at every step.

The plan:

- **Train three small denoisers** on CIFAR-10, identical except for the label: M-on gets it
  (the primary, the ordinary design), M-drop gets it except on 20% of examples, where it gets
  a learned "no label" vector instead, and M-off never gets it.
- **Show M-on each test image four ways**: noised to steps t and t′ with the matching labels,
  and the two mismatches (the image at t′ with the label t, the image at t with the label t′).
  Record the activations inside the network.
- **Measure one number, the movement share w**: of the network's own move between step t and
  step t′, the share produced by moving the content alone. w = 1 means the inside moves with
  the content, w = 0 with the label.
- **Check the label share with a sham**: move the label by a random vector of the same size.
  If the inside moves only for a label that means a different step, the sham is silent.

The cost is about four to five hours of a T4 GPU from Joe's Colab credits, and no API spend.

## 1. The subject: data, three models, training

**1.1 Data.** CIFAR-10, binary version (`cifar-10-binary.tar.gz` from cs.toronto.edu, md5
`c32a1d4ab5d03f1284b67883e8d87530`; the notebook downloads it and stops if the md5 differs).
The 50,000 training images train the models and supply the probe sets. The 10,000 test images
are never trained on, and every arm's images come from them. Pixels are scaled to [−1, 1].
Training uses random horizontal flips.

**1.2 The network.** The U-Net of design check §5, whose code (`windows/power/w3_budget_check.py`,
the classes timed on the T4) the builder lifts verbatim into `windows/w3_model.py`, plus two
additions that change no computation:

- input 3 × 32 × 32; width C = 64; channel multipliers 1-2-2 (64, 128, 128 channels); two
  residual blocks per encoder level and three per decoder level; self-attention at 8 × 8 and
  in the mid-block (residual, attention, residual); GroupNorm with 8 groups and SiLU; 6.5 M
  parameters with the label pathway, 5.9 M without;
- **the label pathway**: the integer step t → a sinusoidal code of C numbers → a two-layer MLP
  (C → 4C → 4C, SiLU) → **the label embedding e(t)**, 256 numbers. Every residual block
  projects e(t) to its channel count and adds it after its first convolution;
- *addition (a)*: a hook that returns the activations at the seven sites of §3.1;
- *addition (b)*: the forward pass can take a label embedding e directly instead of t. M-drop's
  no-label vector and the sham (§3.3) use it.

**1.3 Three models.** Same architecture apart from the label pathway, same initialisation
seed, and the same batches in the same order, with the same steps and noise draws:

- **M-on** (primary): gets e(t) on every example.
- **M-drop**: e(t) is replaced by a learned vector e∅ (256 numbers, initialised at zero) on
  20% of training examples, chosen by a Bernoulli draw from its own generator. A missing label
  is then something the network has seen in training, not a broken input.
- **M-off**: no label pathway at all. It replicates the known result (Sun et al. 2025; Lu 2026)
  on this substrate.

**1.4 Training.**

- *Schedule:* cosine (Nichol and Dhariwal), T = 1,000 steps, betas clipped at 0.999; ᾱ exactly
  as `alpha_bar("cosine")` in `w3_design_check.py`.
- *Objective:* mean squared error between the network's output and the noise ε, with t uniform
  on {0, …, 999} and x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε.
- *Optimiser:* AdamW, learning rate 2 × 10⁻⁴, betas (0.9, 0.999), weight decay 0; linear
  warm-up over the first 500 steps, then constant; gradient-norm clip 1.0; batch 128 with the
  last partial batch dropped (390 steps per epoch); fp16 autocast with gradient scaling, as
  timed. No EMA, no dropout.
- *Length:* **60 epochs** (23,400 steps), about 1.2 T4 hours per model at the measured 49
  epochs per hour (design check §5).
- *Determinism of the stream:* each epoch's permutation, flips, steps and noise come from
  generators seeded by (TRAIN seed, epoch), so all three models see identical batches, and a
  resumed run continues the identical stream. Model weights are initialised under the TRAIN
  seed. A checkpoint (weights, optimiser, scaler, epoch) is written to Drive after every epoch.

**1.5 The training gate** (before any activation is read). For each model, 2,000 test images
(GATE seed) are noised at uniform random steps. In each of four bands of steps (0–250,
250–500, 500–750, 750–1,000), the model's mean squared error on ε must be **below that of the
best fixed multiple of the noisy input** at each step, λ*(t)·x_t with
λ*(t) = √(1 − ᾱ_t) / (ᾱ_t·v + 1 − ᾱ_t), where v is the mean squared pixel value of the
training images. No rule of the form "a number times the input" does better, and that family
includes predicting zeros and predicting the input itself. (The design check named those two
as the trivial predictors. The noisy input itself is nearly perfect at the top steps, so the
bar is the best multiple, which beats both everywhere.) M-drop is gated twice, with labels and
with e∅.

A model that fails at 60 epochs trains on from its checkpoint to 120 epochs and is gated
again, once. Activations are read only from models that pass. If M-on still fails, P1 and P2
are NOT ADJUDICABLE. The gate table, the loss curves and a grid of 64 samples per model
(1,000-step ancestral sampling, SAMPLE seed) are kept as the training record.

## 2. The design check, the prior-art sweep, and what changed

**2.1 The prior-art sweep** (research worker, 2026-10-11, `docs/LOOP_PRIOR_ART_2026-10-11.md`;
the key claims verified against the papers). The clock-off question is answered: Sun, Jiang,
Zhao and He (ICML 2025, arXiv 2502.13129) show the noise level is sharply determined by the
noisy image and that label-free networks lose little; Lu (2026, arXiv 2605.06169, Appendix H)
reads the step with a linear probe at every depth of a large label-free transformer, and the
hidden states beat input statistics. The conflict is answered only at the output: Yu et al.
(CVPR 2026, arXiv 2604.16044) and Li et al. (ICLR 2024) show a mislabelled network's output
goes partly wrong. Nobody has measured the conflict inside the network. The nearest analogue
of the sham is Rulli et al.'s (EMNLP 2026, arXiv 2607.01774) norm-matched random steering
direction.

**2.2 Changes from the seed**, each made before anything was trained:

| seed (ideation doc §3.2) | W3 | why |
|---|---|---|
| clock off: hold the label constant or scramble it at inference | **trained twins** (M-off, M-drop); the headline moves to the conflict | a network given a constant or random label is out of distribution and breaks (Yu et al. 2026; Li et al. 2024); the clock-off question is answered (Sun et al. 2025; Lu 2026) |
| displacement: swap in a state from another step | **the 2 × 2 of content level × label level** inside M-on | a denoising step carries nothing forward but the image, so "follows the schedule" can only mean "follows the label" |
| sham displacement: same-level content from another trajectory | **a label sham**: the label embedding moved by a random vector of the size of the real move | with a memoryless step, a same-level content swap changes nothing about position by construction; the open sham is in the label channel |
| probes read position | **every read is set against the input floor** (§2.3); the primary is probe-free (§2.4) | the noise level is in the pixels; a read that only matches an untrained network has found the input (J1's input-only null) |
| secondaries: seed spread and guidance; inversion as the walk's positive control | **seed spread kept** (S9), within narrow step bands; guidance and inversion deferred | guidance needs a second, class-conditional build; inversion stands as its own piece |

**2.3 The input floor** (design check §2, `w3_design_check.py`). Ridge probes read the noise
level of 2,000 held-out noisy images from features that involve no trained network. CIFAR-10,
cosine schedule; R² of the read step and the median error in steps out of 1,000: raw pixels
.082 and 229 steps (on natural images the noise level is a property of the pixels' spread, not
of any pixel); **14 hand statistics .948 and 38 steps**; the encoder of an untrained U-Net .879
and 62 steps, with its error largest mid-process (126 steps at steps 400–600). The position is
in the input, and the content is the weakest witness mid-process.

**2.4 The collinearity check, and the change of primary** (new with this registration,
`w3_collinear_check.py`, outputs beside it). The design check's primary was a probe statistic:
a ridge probe trained on matched inputs reads the position, and the split weight places the
mismatched reads between the two levels (design check §3). Writing this registration exposed a
flaw in it. On matched inputs the content level and the label level are the same number, so a
feature that tracks the label and a feature that tracks the content are perfectly collinear in
the probe's training data. Ridge regression then divides its weight by how cleanly each
predicts the target. A label code is exact; a content code reads the level with per-image error
(fourteen statistics leave R² .84 of log-SNR on CIFAR-10). The probe can lean on the label
while the representation moves mostly with the content.

The check plants activation worlds (an additive content code, label code, image variation and
noise-draw variation, 64 features) and runs the probe statistic beside a probe-free one on the
same inputs. At 256 images per offset class, 100 simulated experiments per world (the probe
column gives its most frequent verdict and that verdict's rate):

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

**The probe statistic fails exactly where W3 will be.** When the content code reads the level
with per-image error about the size the input floor shows, the probe calls LABEL on a
network whose representation moves 91% with the content. The probe statistic
measures which source gives the *cleanest* position code, which is a real quantity but not the
question. **The primary is therefore the movement share** (§5.1), which asks how much of the
network's own move between two levels comes from moving the content. The probe statistic is
kept as secondary S2, read as "where the cleanest position code lands", and its disagreement
with the movement share is pre-read in §5.1.

**2.5 The movement share's verdict rates** (`w3_collinear_check.py` §2; 100 simulated
experiments per cell, 500 bootstrap draws each):

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

- **κ separates cleanly.** Additive worlds sit at .03–.05 (the per-input jitter), the
  interacting world at .16, scatter at 5.6 and collapse at 30. The bar, 0.5, is an order of
  magnitude above every readable world and an order below the broken ones.
- **The level check needs a floor.** The movement share's intervals are tight (the network is
  deterministic, so the only sampling is of images), and without a floor a world whose true
  share sat on the 0.2 bar was called VARIES for band shares of .16 and .24 (that world,
  MISREAD-LAB, was then re-planted with a share near .07 so that it tests LABEL). A floor of 0.3,
  tried first, cost the banded world its real .72-versus-.98 contrast (VARIES .90 at 256
  images). The registered floor is 0.2, the width of the LABEL zone.
- **The interacting world's right verdict is VARIES**: its label effect changes with the
  content level by construction, and its band shares run from .34 to 1.0.
- **Size stays at 256 images per offset class**, as the design check chose for the probe
  statistic.

**2.6 The label sham** (design check §4). The statistic and bar carry over: S = mean |shift
under the sham| / mean |shift under the real label move|, content fixed, on the 300-step
class, SILENT when the 95% upper bound is ≤ 0.25. The simulation separated a quiet sham
(k ≤ .1) from a reactive one (k ≥ .3) cleanly and failed safe when jitter was large next to the
real shift. With the probe gone from the primary, a "shift" is now the projection of the
activation change on the network's own matched move (§5.2), the same geometry as the primary.

**2.7 Budget** (design check §5). A Colab T4 timing run (2026-10-11,
`windows/power/w3_timing/timing_20261011_035637.json`) measured 0.187 s per training step of
128 images at width 64 with the label (0.186 without), 49 epochs per T4 hour, and 0.07 s per
forward pass of 128 images. Three models at 60 epochs are about 3.7 T4 hours. Everything after
training (about 40,000 forward passes for the arms and probe sets, and the seed-spread
continuations) is a few minutes.

## 3. Inputs and what is recorded

**3.1 Sites.** Seven places in the network, each the output of a block: **enc-32**, **enc-16**,
**enc-8** (the last encoder block at each resolution, before downsampling; 64, 128 and 128
channels), **mid** (the output of the mid-block, 128 channels at 8 × 8: **the primary site**),
**dec-8**, **dec-16**, **dec-32** (the last decoder block at each resolution, before upsampling;
128, 128 and 64 channels). The feature vector of a site is the spatial mean of each channel. The
network's output ε̂ is also kept for S3, as the 14 hand statistics of ε̂ (design check
`hand_features`). Activations are recorded in fp32 with the model in eval mode.

**3.2 The split set.** 768 test images, drawn without replacement (ARM seed), 256 per offset
class of 25, 100 and 300 steps. For each image: a base step t, a uniform integer on
[20, 1,000 − offset); t′ = t + offset; one noise draw ε shared by both levels, so
x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε and x_t′ the same with t′. Four inputs, written (content, label):
(t, t) and (t′, t′), the matched pair; (t′, t) and (t, t′), the mismatches. M-on, M-drop (with
labels) and M-on's untrained twin all see the same 3,072 inputs.

**3.3 The sham set.** The same 768 images, input x_t with the embedding
e(t) + ‖e(t′) − e(t)‖·u, where u is a uniformly random unit vector in 256 dimensions, one per
image (SHAM seed). M-on (P2) and M-drop (reported under S6).

**3.4 The matched sets** (training images, PROBE seed): a probe-train set of 4,000 images and
a calibration set of 3,000, each noised once at a uniform random step on {0, …, 999} and shown
matched. M-drop also sees both sets with e∅ in place of the label (its no-label sets); M-off
sees them without a label. The movement share's standardisation uses each model's per-feature
SD over its probe-train set at that site.

**3.5 The floor.** The 14 hand statistics of every input image, and the untrained twin of each
model (the same architecture initialised under the TRAIN seed, never trained), shown the same
inputs.

**3.6 The seed-spread set** (S9, M-on only; SPREAD seed). Four narrow bands of steps centred at
100, 300, 500 and 700 (± 10). In each, 64 test images (disjoint from §3.2) are noised to a step
drawn in the band; from each state, 8 continuations run to step 0 by ancestral sampling on 50
evenly spaced steps from t to 0 (the DDPM posterior variance on the respaced schedule), each
with its own noise. The spread of a state is the log of the mean per-pixel variance of its 8
final images.

## 4. Stages

**Order:** this registration pushed → builder + suites + `FROZEN_W3.json` pushed → smoke →
flight → analysis → fresh-process recompute (byte-identical) → lock with a results appendix.

**The notebook.** `colab/build_w3_notebook.py` writes `colab/W3_LOOP_UI.ipynb`, the only thing
that runs on Colab. It is self-contained: the model, training, gate and harvest code are
embedded verbatim from the frozen `windows/w3_*.py` files, and a cell prints their sha256 for
comparison with `FROZEN_W3.json`. It mounts Drive in Joe's browser session and writes to
`MyDrive/semcore/w3/<stage>_<stamp>/`; no rclone, no command-line or headless control of Colab
(the UI-only law). Joe runs it in the browser: Runtime → T4 GPU → Run all. A dropped runtime is
resumed by running it again with `RESUME_STAMP` set; it continues from the last epoch
checkpoint, and finished stages are skipped. Results come back through the Google Drive
integration. The analysis (`windows/analyze_w3.py`) runs locally on the harvested activations,
and the recompute runs it again in a fresh process.

**Smoke** (own seeds: every registered seed + 100). Two epochs per model; the split set at 32
images per offset class; matched sets of 400 and 300 images; the sham set; the seed-spread set
at 8 images per band and 4 continuations; every arm and the analysis end to end. **Go rule:**
the notebook runs end to end; seconds per training step within 25% of the timing run; the
resume self-test passes (3 steps, checkpoint, reload, 2 more steps gives bit-identical weights
to 5 straight steps); G-det passes; the projected flight is at most 6 T4 hours; the harvested
files load locally and the analysis runs on them. The smoke's verdicts mean nothing (two
epochs). A smoke shortfall that is a builder bug is fixed and noted as a build-time
clarification before the flight.

**Flight.** One Run all trains M-on, M-drop and M-off in that order, gates each, harvests every
set from every passing model, and ships the files. The flight's stamp and the epoch count of
each model are recorded in the run log.

**Gates.**

- **G-train**: §1.5.
- **G-det**: the same batch of 128 inputs, evaluated twice, gives activations that agree to
  1 × 10⁻⁵ at every site. The sham's logic needs a deterministic network: every shift under the
  sham must be caused by the sham.
- **G-hash**: the code sha256 printed by the notebook matches `FROZEN_W3.json`.
- **G-data**: the CIFAR-10 md5 matches.

## 5. Measures and verdicts

**Notation.** h(c, l) is a site's feature vector for content at step c with label l, divided
feature by feature by that model's SD over its probe-train set (§3.4). Per image, write the
four inputs as h_tt, h_t′t′ (matched), h_t′t, h_tt′ (content t′ with label t; content t with
label t′). Intervals are 95% percentile intervals from 10,000 bootstrap resamples of images
(ANALYSIS seed).

### 5.1 P1: the movement share (M-on, mid-block)

For each image, two directions:

- direction 1, from (t, t): the content move dc₁ = h_t′t − h_tt, the matched move
  db₁ = h_t′t′ − h_tt;
- direction 2, from (t′, t′): dc₂ = h_tt′ − h_t′t′, db₂ = h_tt − h_t′t′.

> **w = Σ (⟨dc₁, db₁⟩ + ⟨dc₂, db₂⟩) / Σ (⟨db₁, db₁⟩ + ⟨db₂, db₂⟩)**, summed over images.

w is the content's share of the network's own move from t to t′, measured along that move.
The label's share, defined the same way with the label moves, is exactly 1 − w: the 2 × 2
interaction I = h_t′t′ − h_t′t − h_tt′ + h_tt enters the two directions with opposite signs and
cancels, so w is the content's Shapley share of the move. The interaction is measured
separately: **κ = Σ |I|² / Σ |db₁|²**.

| branch | rule (in this order) | reading |
|---|---|---|
| **NOT ADJUDICABLE** | M-on fails G-train after 120 epochs, or G-det fails | |
| **UNREADABLE** | the upper bound of κ exceeds 0.5 | mismatched inputs leave the plane of the matched ones; the 2 × 2 does not decompose the network's move, so neither source is "followed" |
| **VARIES BY LEVEL** | w fitted within four bands of the base step (20–250, 250–500, 500–750, 750–1,000): two bands with different decisive verdicts (CONTENT, LABEL or SPLIT) whose intervals do not overlap and whose shares differ by at least 0.2 | which source the inside moves with depends on where in the process it is; the band verdicts are the finding |
| **CONTENT** | lower bound of w ≥ 0.8 | the inside moves with what it sees |
| **LABEL** | upper bound of w ≤ 0.2 | the inside moves with what it is told |
| **SPLIT** | the whole interval inside (0.2, 0.8) | both sources move it, in the measured proportion |
| **UNCLEAR** | otherwise | |

**How P1 reads with the probe statistic (S2).** The two answer different questions: P1 asks
which source moves the network's representation; S2 asks which source gives its cleanest
position code. Pre-read combinations:

- P1 and S2 agree: that is the answer, from both angles.
- P1 CONTENT or SPLIT and S2 further toward LABEL: the network moves mostly with the content
  and carries the label as a small, exact code that a linear read prefers. This is the
  collinearity world of §2.4, and the headline is P1's.
- S2 further toward CONTENT than P1: the label moves the representation in directions a
  position read does not use. S3 (the output) then says which of the two the network's
  prediction follows.

### 5.2 P2: the label sham (M-on, mid-block, 300-step class)

Per image, with content fixed at t: the real label move d_lab = h(x_t, e(t′)) − h_tt, the sham
move d_sham = h(x_t, e(t) + ‖e(t′) − e(t)‖·u) − h_tt, both standardised as above, and the unit
matched move b = db₁ / |db₁|.

> **S = Σ |⟨d_sham, b⟩| / Σ |⟨d_lab, b⟩|** over the 256 images of the 300-step class.

| branch | rule | reading |
|---|---|---|
| **not judged** | P1 is CONTENT, UNCLEAR, UNREADABLE or NOT ADJUDICABLE | S is reported; there is no label share to qualify |
| **SILENT** | upper bound of S ≤ 0.25 | the inside moves along the position direction only for a label that means a different step |
| **NOT SILENT** | otherwise | some of the label's share is a reaction to any disturbance of the label channel, not to its meaning |

The 25- and 100-step classes are reported, not judged: their real moves are small, and jitter
would dominate (design check §4). The full-length ratio Σ‖d_sham‖ / Σ‖d_lab‖ is reported beside
S: how much the sham disturbs the representation in any direction.

There is no multiplicity correction: P1 is the single primary, and P2 is judged only on its
pre-stated condition.

### 5.3 Secondaries (pre-named; no multiplicity claim)

- **S1 The profile.** w, κ and the P1 verdict rules at all seven sites of M-on.
- **S2 The probe read.** The design check's probe statistic (`w3_split_check.split_fit`, its
  tracking and precision gates, its verdict rules) at every site of M-on: ridge probes on the
  probe-train set (features standardised; the ridge penalty chosen by cross-validation from
  `w3_design_check.ALPHAS`), target log-SNR, isotonic calibration g on the calibration set. Read
  with P1 as in §5.1.
- **S3 The output.** w and κ computed on the 14 hand statistics of ε̂ (standardised the same
  way): does the network's prediction move with the content or the label? Yu et al.'s
  output-level question, on our substrate, beside the inside's answer.
- **S4 Directions.** The content share from each direction alone (direction 1: the content is
  noisier than the label; direction 2: cleaner), each with its interval. Unlike w, these do not
  sum with their label shares to 1; the gap is the interaction.
- **S5 Offsets.** w by offset class (25, 100, 300 steps).
- **S6 M-drop.** w at the mid-block and its difference from M-on's (paired bootstrap over
  images); its sham S on the 300-step class. Its no-label reads: a ridge probe trained on its no-label probe-train set, scored on
  its no-label calibration set (R², median step error by band), against the hand-statistics
  floor on the same images.
- **S7 M-off against the floor.** A ridge probe on M-off's matched sets at each site (R²,
  median step error by band), against the 14 hand statistics and M-off's untrained twin on the
  same images. "Beats the floor" when the paired bootstrap interval of the difference in mean
  absolute log-SNR error excludes zero in the network's favour. This replicates Lu (2026) on
  our substrate.
- **S8 The untrained baseline.** w and κ for M-on's untrained twin at every site: the share the
  architecture gives before any learning. P1 is read against it in the results.
- **S9 Seed spread.** Within each band of §3.6, cross-validated R² (5 folds, SPREAD seed) of
  ridge probes predicting a state's spread from (a) the baseline, the 14 hand statistics of x_t
  and of the network's own estimate of the clean image x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t;
  (b) M-on's mid-block features; (c) both. The increment of (c) over (a) is the question: does
  the inside carry information about where the trajectory can still go beyond what its own
  estimate of the image shows? Reported per band; no verdict.
- **S10 The training record.** Loss curves, the gate table (each model's error against the
  best fixed multiple, by band), the sample grids, the epochs each model trained, and the
  measured time per step.

## 6. Honest expectations (written before anything is trained)

- **G-train passes** for all three models at 60 epochs (~90%).
- **P1** at the mid-block of M-on, my spread: VARIES BY LEVEL ~30%, SPLIT ~25%, CONTENT ~20%,
  LABEL ~10%, UNREADABLE ~10% (a 300-step mismatch is far from anything the network saw in
  training, and its response may not decompose), UNCLEAR ~5%. If it varies, I expect the label's share to be
  largest mid-process, where design check §2 found the content weakest as a witness. The label
  adds a per-channel offset in every residual block, which survives spatial averaging intact,
  so a large label share would not surprise me; pure LABEL would, given how much the content
  carries.
- **P2** SILENT, if judged (~60%). A random direction in 256 dimensions has little projection
  on the curve the trained label embeddings lie along, but a network can react strongly to an
  embedding it has never seen.
- **S2** sits further toward LABEL than P1 (~75%): the collinearity prediction of §2.4.
- **S1** the content share is highest at the encoder's first site and falls toward the
  mid-block (~55%). The first block sees the raw noise before any label has been added more
  than once.
- **S3** the output's w is further toward CONTENT than the mid-block's (~55%).
- **S6** M-drop's w is within 0.15 of M-on's (~60%), and its no-label reads beat the hand
  floor (~55%).
- **S7** M-off's best site beats the hand-statistics floor (~65%), as Lu (2026) found at scale.
- **S8** the untrained twin's w is near 0.5 or UNREADABLE (~60%): nothing has yet taught the
  architecture to weigh either source.
- **S9** the mid-block adds to the baseline in at least one band (~40%), most likely near
  steps 300–500.

## 7. Planted agents (verdict suites; must pass before the smoke)

**The analysis suite** drives the real `analyze_w3.py` with synthetic activations in the
harvest's own file format, generated from the worlds of `w3_collinear_check.py` at the
registered sizes and a fixed suite seed. Lesson banked from W1-T: a suite asserts the set of
verdicts the design licenses at that power, and asserts an exact verdict only where the design
check's rate is at least 0.99.

| agent | plants | must give (P1, mid-block) |
|---|---|---|
| content | content code only | CONTENT |
| label | label code only | LABEL |
| equal | both codes, equal size | SPLIT |
| misread | small exact label code, content code with per-image error (MISREAD.3) | P1 CONTENT, and S2 LABEL or SPLIT: the suite checks the collinearity reading end to end |
| collapse | mismatched inputs fall to one point | UNREADABLE |
| scatter | mismatched inputs gain large noise | UNREADABLE |
| banded | label code below step 500, content code above | VARIES BY LEVEL (or the licensed set at its rate) |
| interact | the label code's size depends on the content level, so the share changes with level | VARIES BY LEVEL |
| quiet sham | label-following network; sham moves 0.05 × the real move | P2 SILENT |
| reactive sham | the same; sham moves 0.5 × the real move | P2 NOT SILENT |
| content + sham | content network | P2 not judged |
| untrained M-on | M-on fails G-train | P1, P2 NOT ADJUDICABLE |

**The training suite** (CPU, a width-8 copy of the network): the seven hooks return the
documented shapes; the embedding path (addition b) reproduces the label path exactly when given
e(t); M-drop's e∅ is used on the drawn 20% and nowhere else; the three models receive identical
batches, steps and noise; checkpoint-and-resume gives bit-identical weights; every notebook cell
that builds a model runs forward and backward on PyTorch's `meta` device (a stray CPU tensor
fails there; the lesson of the timing smoke); the gate's λ* reproduces its closed form; the split
set's 2 × 2 shares one noise draw; no test image appears in a training batch or probe set.

## 8. What this can and cannot show

**Can:** which of two sources, the label it is told or the content it sees, moves a small
denoiser's internal representation of where it is in the process, when the two disagree; how
that changes with depth and with position in the process; whether the label's share is about
the label's meaning (the sham); whether training with a missing label or no label changes the
answer; and whether its inside carries anything about the trajectory's future beyond its own
estimate of the image (S9).

**Cannot:** anything about experience, persistence or stakes. A denoiser keeps nothing between
images, and nothing it does is for itself. The movement share is defined in the geometry of
spatially averaged, standardised activations; it is a share of the network's move, not proof
that downstream layers use one source more than the other (S3 is the output-level check). The
result is about this architecture at this size, trained for 60 epochs; a large text-to-image
model may differ.

## 9. Ops

- **Spend:** Colab compute from Joe's credits (473 on 2026-10-11); the smoke is about 15
  minutes of a T4 and the flight about 4–5 hours. No OpenRouter spend. Joe runs every notebook
  himself in the browser (the Colab UI-only law).
- **Seeds:** TRAIN 20261020 · ARM 20261021 · PROBE 20261022 · ANALYSIS 20261023 ·
  SHAM 20261024 · SPREAD 20261025 · GATE 20261026 · SAMPLE 20261027 · M-drop's Bernoulli
  draw 20261028. The smoke adds 100 to each.
- **Files:** results go to `windows/results/w3_{freeze,smoke,flight}_<stamp>/`. Checkpoints and
  raw activations stay on Drive and locally (gitignored); the committed record is the analysis
  outputs, the gate table, the sample grids, a compressed copy of the per-image summary
  quantities the verdicts use, and the sha256 of every raw file.
- **Data:** `~/datasets/w3_raw/` locally; never committed.

## Build-time clarifications

*(none yet)*

## Amendments

*(none yet)*
