# E8-F — The Featural Curriculum (L3, bottleneck form)

**Pre-registered before build** (lane law). Design check ran first
(`colab/e8f_design_check.py`, artifact `colab/results_e8f/design_check.json`);
its measurements shaped every registered choice below and are quoted where
they did.

## §0 Position and lineage

E8-J v2 fork **FJ1** licensed this rung: *"L3 (featural curriculum) licensed
in bottleneck form"* — train the readout ON bridge feature axes, composing
per the algebra. The Hangul staircase position: R3-c (L1) proved pair
composition over trained atoms; E8-J measured the dictionary↔hidden-space
correspondence (real but thin, ρ≈.11); E8-J v2 proved wing identity survives
the 14-coordinate register round-trip into a locked readout's naming. E8-F
asks the featural question itself: **can a trained readout learn the
register's AXES as first-class readable features and spell NEVER-TRAINED
concepts by composing them** — the jump from lookup (E8-R2's
nearest-trained-name) and fixed pair grammar (R3-c) to a generative read of
the code.

**Bottleneck form, stated plainly**: the feature axes are the rows of a
ridge bridge fit from hand-authored codebook coordinates to the instilled
substrate's pooled reading of hand-authored NAME:desc prose (E8-J
machinery verbatim). Nothing here claims independent authorship; the
native-F1 arm (independently authored coordinates) remains named future
work. Authorship mediation is inherited verbatim from E8-J §7.

**Model speech vs naming**: the readout SPELLS (emits a 14-field featural
code). Any "naming" in P2 is the DICTIONARY decoding that spelling by
nearest-code lookup — the model never speaks concept names in this flight.
Registered wording throughout keeps this straight.

## §1 Design check — measured basis (2026-08-24, all local, locked artifacts)

Machinery identity proven before anything else: the check reproduces the
flown E8-J A5 wing transfer (89.8°, 0/13, per-target < 0.05°), the flown A4
axis-wise reverse R² (all 14 axes < 5e-4), and the E8-J v2 pinned d̂
(max 0.0007°) from raw artifacts. Planted teeth: a readable synthetic world
separates from its derangement (.74 vs .52) and a no-correspondence world
reads at chance (.53).

Findings that shaped the registration:

1. **Sign is prior-saturated; the code must be terciles.** Reverse
   sign-reading of true dirs scores .98–1.00 on domain axes — and so does
   its derangement null (pooled null .8500): the register's domain signs are
   nearly constant (the Ifá-seed "casts only méjì" observation, measured
   again). Per-axis TERCILE quantization over the frame carries the
   information; tercile capacity beats sign-deadzone at every alphabet size
   (§6 of the check).
2. **Full 14-axis tercile capacity is real**: 87% of the 1,865-concept frame
   spells uniquely; 48/64 eval concepts self-unique; but decode margin ≥ 2
   fields for only 26/64 (median margin 1) — decode-to-name is registered as
   a SECONDARY with a modest bar, not a primary.
3. **μ-centered atoms are clean and the whole alphabet is leveraged**: with
   the carrier point at the register mean (frame μ spells all-MID), all 28
   single-axis ±2σ displacements move exactly their own field off MID, and
   every axis's atom moves the injected direction by ≥ 44.9° at 2σ
   (x 80.0°, fx 66.8°, min fe 44.9°) — far above the channel's demonstrated
   ~15–20° resolution (R3-c 18.5°, E8-R2 28.5°). Zero-origin atoms are NOT
   clean (skewed axes spell off-MID at x=0) — rejected by measurement.
4. **Pair corners separate**: 4-corner min-separation over top-6 axes
   median 25.0° (min 17.2°) — compositional training rows are
   discriminable.
5. **A true-dir linear prior exists on a measurable axis set.** Reverse
   ridge (fit train-256, honest split) reads EVAL true dirs at tercile
   accuracy vs permutation null: fx .609/.340, h .484/.327, fh .500/.354,
   x .453/.310, z .453/.343, fz .422/.290, e .422/.309, fy .500/.411 —
   eight axes clear their null by ≥ .08. **Carried-8 :=
   {x, z, e, h, fx, fy, fz, fh}** (pinned). g and fg are dead (Δ ≤ .004);
   y, f, fe, ff below the bar.
6. **Predictions are content-dominated** (pred-vs-carrier 76°, pairwise
   spread median 76.5°) and the honest-split transfer floor is the known
   thinness (eval median 72.4°, λ-stable) — P1 (injected predictions) and
   P3 (true dirs) measure genuinely different things.
7. **Base cone degeneracy confirmed on the split** (top-6 reverse sign
   .729 with degenerate geometry; E8-J lock: eff-rank 24 vs 149): the
   curriculum rides the INSTILLED real model — also the twice-measured
   "naming requires instilled geometry" precedent (E8-N v2).

## §2 Registered design

**Universe**: the E8-J anchor frame (desc ≥ 40, wing-13 excluded), n=1,865.
**Split**: the design check's minted stratified draw, seed 20260886:
eval-64 / train-256 of the 320 pinned anchors (per-trigram proportional).
Eval-64 is OUT of the bridge fit, OUT of every curriculum strand, and OUT
of every pin derivation. Lists pinned as literals in `e8f_logic.py` with
sha; identity asserted against `design_check.json` at build.

**Bridge** W_L3: ridge λ=1.0 (unpenalized intercept), coords→unit inst-L14
dirs, fit on train-256 ONLY (E8-J `ridge_fit` verbatim). λ-sensitivity
measured flat in the check (72.78/73.01 vs 72.4) — not re-flown. W_L3 is
minted at build from the locked atlas and shipped as a pinned payload;
**no on-VM dir or bridge computation anywhere in the flight**.

**The code**: 14 axes in AXIS_NAMES order; per-axis tercile levels
LOW/MID/HIGH with boundaries = frame 33.33/66.67 percentiles, pinned as
4-decimal literals. `code(x)` = the 14-level word. Spelling grammar (single
line, fixed order): `x:MID y:LOW z:HIGH ... fh:MID`, or `NONE`. Parser:
per-axis regex, order-independent, first occurrence per axis; every eval
row scores 14 binary field-matches; unparsed/missing/NONE fields score as
mismatches (no parse-gaming; G4 rides FULL-parse rate separately).

**Injection**: E8-R lineage operator verbatim — α · μ_L14 · normalize(v)
added at layer 14, μ_L14 = 81.875 (pinned from the atlas), α ∈ {0.5, 1.0}
(the trained regime, verbatim).

**Readout**: FRESH LoRA (R3-c/E8-R config verbatim) trained on the E4
real-arm instilled model (adapter merged before training, asserted). The
locked E8-R readout is NOT used in this flight. LR 1e-4, ACCUM 8, greedy
decode, answer-sliced CE (long-answer law), EPOCHS_CAP 6 with per-strand
plateau (min 2, rel .05 — the E8-N v2 minted lesson, prospective:
un-plateaued strands at cap are flagged in the verdict, and the
epoch-starvation reading is pre-registered for any generalization miss
whose strand was still descending).

**Curriculum** (single condition; strand names pinned; all stimuli minted
at build from pinned artifacts; targets = canonical code strings):

| strand  | rows | construction |
|---------|------|--------------|
| carrier | 12   | pred(μ_frame), 2α × 6 reps → all-MID code |
| atom    | 168  | μ_frame ± 2σ_j per axis (14×2), 2α × 3 reps → one field off MID |
| pair    | 168  | 21 pinned axis-pairs (every axis in exactly 3; seeded 3-regular draw) × 4 corners (±2σ both axes) × 2α → two fields off MID |
| full    | 256  | train-256 pred(x_c) × 1 (α alternating by pinned parity) → code(x_c) |
| truedir | 128  | pinned 128-subset of train-256 TRUE inst-L14 dirs (locked atlas) × α1.0 → code(x_c) — teaches that codes apply to real geometry |
| sham    | 48   | no injection → NONE |

Total 780 training rows. Curriculum-level note (registered): atom/pair
strands are MID-heavy by construction; full/truedir strands are balanced;
any MID-flood emission bias scores exactly at the derangement null — no
gate needed, surfaced in S1/S8.

## §3 Eval rows and statistics

All eval at greedy decode; rows pinned with seeded construction at build.

| block    | rows | construction |
|----------|------|--------------|
| SPOT     | 48   | 24 pinned train concepts, pred-dirs, 2α |
| P1       | 128  | eval-64 pred(x_c) dirs, 2α |
| P3       | 128  | eval-64 TRUE inst-L14 dirs (locked atlas), 2α |
| W-perm   | 32   | 16 pinned eval concepts through axis-deranged W (pinned derangement of the 14 axis rows; intercept fixed), 2α |
| carrier  | 4    | pred(μ_frame), 2α × 2 |
| sham     | 24   | no injection |
| atom     | 28   | the 14×2 atoms, α1.0 |
| titr     | 16   | 8 pinned eval concepts, α ∈ {0.25, 1.5} |
| wing     | 13   | E8-J v2 pinned d̂ (re-encoded wing), α1.0 — texture only |

**P1 — composition/interface (primary)**: pooled per-field accuracy over
the 128 P1 rows × 14 axes vs 2,000 concept-level derangements of the
eval-64 codes (rows rescored against code(π(c)) consistently). PASS =
p ≤ .0025 **and** ≥ 6/14 axes individually significant under per-axis
derangement nulls, Holm over 14. The claim on pass (registered wording):
*never-trained points of the register are spelled by composition through
the trained channel — the code generalizes across the coordinate field,
beyond any lookup over trained points* (the derangement null and the
held-out split kill lookup).

**P3 — true-dir semantics arm (primary)**: pooled per-field accuracy over
the 128 P3 rows on the **carried-8** axes only, vs the same 2,000-
derangement construction. PASS = p ≤ .0025. The linear prior band is
quoted (per-axis .42–.61 vs nulls .29–.41), not gated: the trained channel
may fall below it (SFT noise) or above it (nonlinearity) — either side is
reported against the band. A NONE-flood on true dirs is a REGISTERED
possible finding (presence-without-code, the E8-R2/E8-N v2 dissociation
family), not an instrument failure: NONE rows score all-fields-wrong under
both obs and null, draining discrimination honestly; S1 reports
conditional-on-speech accuracy alongside (the "selection as calibration"
lens, E8-N v2).

**P2 — naming via the algebra (secondary)**: each P1 row's spelled code is
decoded by strict unique-argmin Hamming over the frame's 1,865 pinned
codes (ties or non-unique → no-name). PASS = exact decode (name == injected
concept) on ≥ 12/128 rows **and** ≥ 8/64 distinct concepts **and**
derangement-null p ≤ .0025. Ceiling quoted from the check: 48/64
self-unique, margin ≥ 2 for 26/64. On pass, the registered wording: *first
generative naming through the algebra in bottleneck form — the model
spells, the dictionary names.*

**Gate G-INSTALL**: SPOT pooled per-field accuracy ≥ .55 AND SPOT
derangement p ≤ .0025. Failure → **NO_VERDICT** on P1/P2/P3 (installation
failure, not a generalization result; re-fly path in §5). SPOT also ships
per-strand plateau curves for the FF4 adjudication.

**S-blocks (registered)**:
- **S1** speech-selection profile: NONE/FULL/partial rates per block;
  conditional-on-speech accuracies.
- **S2** W-perm probe, both readings pre-stated: scored vs ORIGINAL codes
  (expected: falls toward null — the readout reads axis-specific
  directions, not prompt or row identity) and vs PERMUTED codes (expected:
  rises — in-span code-reading is geometry-general, the R3-c S3
  inheritance made visible in-flight without a second training run).
- **S3** error structure: adjacent-vs-opposite tercile confusions per axis;
  axis-pair error correlations read against the W-row angle table (the
  register-capacity echo; U–C kinship lineage).
- **S4** titration: α0.25 (below the measured speech cliff) and α1.5
  (past-cliff degradation) — house lineage rows.
- **S5** wing texture: 13 re-encoded d̂ spellings vs tercile codes of the
  re-encoded coords vs hand coords. NOT gated; continuity with E8-J v2.
- **S6** P2 mechanism: exact-decode rate among margin ≥ 2 vs margin ≤ 1
  concepts.
- **S7** carrier probe: all-MID emission on pred(μ).
- **S8** code-density curve: row accuracy vs #non-MID fields in the true
  code (mechanism row for FF4).

## §4 Gates

- **G1 pins**: on-VM sha of the shipped payload (W_L3, tercile literals,
  split lists, all stimulus vectors, derangements, row plans) == build
  literal; row counts exact; μ assert 81.875; E4 real adapter load+merge
  asserted; fresh readout-LoRA init asserted; wing-13 absent from every
  curriculum/eval concept list (texture block excepted, flagged).
- **G2 model-identity probe**: the wing-13 dirs recomputed on-VM in the
  documented own-13-call shape (both the E8-J v2 convention and the
  batch-composition lane law) vs the shipped E8-R bundle dirs, tol 1e-4.
  The eval-64 true dirs themselves ride as PINNED constants from the
  locked, recompute-verified atlas — no on-VM recomputation.
- **G3 ppl**: battery delta (readout adapter on vs off) ≤ 0.5%, R3-c form.
- **G4 parse**: FULL-parse rate ≥ 90% over non-sham eval rows.
- **G5 sham FA**: non-NONE claims ≤ 4/24.
- **G-plateau**: per-strand plateau state shipped; cap-hit with descending
  strands is flagged (not a failure) and feeds the pre-registered
  epoch-starvation reading.

Gate failure → NO_VERDICT on affected primaries (lane law); measured
blocks still ship under a GATES-DIRTY banner.

## §5 Registered forks

- **FF1** — P1 ✓: the register is a readable compositional code through
  the trained channel; **L3 lands in bottleneck form**. L4 design
  (operator/ordered-composition grammar) unparks, and the Ifá seed's
  design check (docs/IFA_SEED.md, parked behind L3 by Joe's word) becomes
  runnable next.
- **FF2** — P1 ✓ ∧ P3 ✓: the code also reads TRUE substrate geometry on
  the carried-8 — the program's first semantics-bearing featural read;
  QA-ledger and wing-v2 entries.
- **FF3** — P1 ✓ ∧ P3 ✗: interface-without-geography (the separability
  family, 4th instance if it lands); the claim stays bounded to the
  bridge span, stated plainly.
- **FF4** — P1 ✗ with G-INSTALL ✓: a composition wall INSIDE the span —
  the 14-field joint read exceeds the channel/SFT capacity reached by this
  curriculum; S8 + per-axis Holm table + titration localize; one budget
  re-fly with a revised curriculum (registered path: denser pair/full
  strands or reduced alphabet) before any redesign.
- **G-INSTALL ✗**: NO_VERDICT, re-fly after curriculum/budget revision
  (not a fork).
- **P2 ✓** rides any P1 ✓ fork as the naming texture (never upgrades a
  fork by itself).

## §6 Instruments, pins, provenance

- Design check: `colab/e8f_design_check.py` + `results_e8f/design_check.json`
  (committed with this registration).
- Build (after registration): `colab/e8f_logic.py` (single source),
  `colab/build_e8f_notebook.py`, staged `~/Desktop/E8F_FEATURAL_UI.ipynb`
  — Colab UI-only, drive.mount only, zero rclone, de-shelled installs,
  inflight shipping to `semcore/e8f/`, RESUME_STAMP, progress prints in
  any loop > 2 min (lane note #2), smoke ladder before full.
- Donor discipline: injection/scoring/train cells sliced byte-verbatim
  from the R3-c/E8-J v2 builders where they exist; per-cell compile +
  no-rclone scan + verdict cell exec'd verbatim against synthetic flights
  at BOTH mode constants; CPU train-path micro-gate (answer-sliced CE ==
  HF loss under the composed hook); planted teeth (a readable world passes
  P1 scoring, a deranged world fails; W-perm probe teeth both directions).
- VM law (minted on R3-c, upheld on E8-J v2): VM cells are NEVER edited in
  place; any VM error comes home to the builder for fix+test+restage.
- Verdict recompute law: on retrieval, the verdict cell re-executes
  verbatim locally over raw bundles; every primary ingredient reproduced
  by independent fresh-code tallies before LOCK.
- Locked inputs: E8-J atlas `results_e8j/full_20260824_1827` (dirs, μ),
  E8-J v2 `results_e8j2/rung_pins.json` (d̂ texture). DB untouched;
  adoption of anything = separate gated step.

## §7 Honesty

- Everything here is bottleneck-form; the axes are the hand register seen
  through a thin (ρ≈.11) measured correspondence. A P1 pass is an
  INTERFACE capacity claim; per the R3-c S3 inheritance (registered, made
  visible by S2) in-span code-reading is geometry-general, and the
  semantics weight rests on P3 alone.
- The carried-8 and every bar above were fixed from the design check
  BEFORE build; the split, boundaries, and draws are seeded literals.
- The check's §3 transfer floor (72.4°) is on the record: P1 does not
  imply the bridge predicts true geometry — that wall stands from E8-J
  regardless of outcome here.
- Wing-13 appears ONLY in the untrained, ungated S5 texture.
