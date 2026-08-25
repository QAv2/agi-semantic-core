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
the 128 P1 rows × 14 axes vs 10,000 concept-level derangements of the
eval-64 codes (rows rescored against code(π(c)) consistently; the same
10,000 permutations are shared by the pooled and per-axis statistics).
PASS = pooled p ≤ .0025 **and** ≥ 6/14 axes individually significant
under per-axis derangement nulls at Holm-.05 over 14. *Amended
transparently before build: the first registration wrote 2,000
permutations and per-axis "Holm" with no level, implying the .0025 family
level — arithmetically unreachable (permutation-p floor 1/2001 ≈ 5e-4 >
.0025/14 ≈ 1.8e-4). The pooled primary carries the strict .0025 family
bar at 10,000 permutations (floor 1e-4); the axis-count clause is a
structural guard at the conventional Holm-.05.* The claim on pass (registered wording):
*never-trained points of the register are spelled by composition through
the trained channel — the code generalizes across the coordinate field,
beyond any lookup over trained points* (the derangement null and the
held-out split kill lookup).

**P3 — true-dir semantics arm (primary)**: pooled per-field accuracy over
the 128 P3 rows on the **carried-8** axes only, vs the same 10,000-
derangement construction. PASS = p ≤ .0025 (floor 1e-4). The linear prior band is
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
- **G3 ppl**: retention-battery delta post-training vs pre ≤ 5.0% (the R3-c
  trained-readout precedent: tol 5.0%, measured +2.01%). *Amended
  transparently before build: the first registration wrote 0.5%, which is
  the EVAL-ONLY reconstruction bar (E8-R2/E8-J v2 lineage) — a trained
  readout is a different measured class, and the R3-c bar is the honest
  precedent. Amendment made before any build code existed.*
- **G4 parse**: parse-INVALID rate ≤ 10% over eval rows. NONE is a VALID
  parse (a NONE answer is calibrated speech, and a NONE-flood on true dirs
  is a registered P3 finding, not a parse failure).
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
- Flight mechanics (non-statistical): the trained readout adapter is
  shipped inflight IMMEDIATELY post-training, and RESUME can rebuild the
  eval model from it (base + merged E4 + trained readout) to re-run EVAL
  ONLY after a mid-eval death — a ~2h single-condition run deserves a
  resume point between train and eval. G2 re-runs on the pre-readout
  model either way.
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

## §8 Flight 1 record + registered re-fly revision (2026-08-24, before rebuild)

**Flight 1** (`full_20260824_2205`, 2830s wall, Joe's runs; smoke GREEN
first): GATES ALL PASS — G2 resid 4e-08, μ 81.875 exact, ppl +0.83%
(amendment 1's 5.0% bar; the original 0.5% would have false-killed a
passing flight), parse 0/421 INVALID, shams 0/24 with clean NONE (the
calibrated-silence institution transferred to the featural grammar on
first installation). **G-INSTALL FAIL → NO_VERDICT as registered**: spot
pooled .4688 (p at the 1e-4 floor, null .326) under the .55 absolute bar.
Verdict recomputed locally VERBATIM (0 substantive diffs;
`recompute_e8f_flight.py` committed as the instrument; bundles archived
`results_e8f/full_20260824_2205/`).

**Adjudication — the pre-registered epoch-starvation reading applies
verbatim** (§2; E8-N v2's mechanism, second occurrence): the train log
shows cap-6 HIT with `plateaued=false` and EVERY strand still descending
at the cap (full 0.2499→0.183, −6.3%/epoch at the end; truedir −5.2%;
pair −3.6%). Mechanism rows agree the channel reads and was still
improving: S3 errors are 87% adjacent-tercile (757/872 — boundary
misses, not confusions; opposite errors concentrate on z/fz, 21+25 of
115); S8 declines monotonically with code density (sparse .96 → dense
~.45). Withheld-but-on-record: P1 held-out .5128 at the p-floor with
11/14 axes Holm — HIGHER than spot's .4688 (no memorization gap; the
readout learned the field, not the exemplars — recorded as context, no
claim made under NO_VERDICT). P3 .377 p .015 with x/e Holm-clear (e at
.4453 ≈ its .4219 linear prior). P2 0/128 (per-field .51 cannot survive
14-field Hamming decode — the margin analysis predicted this). S5
texture: the featural readout, with zero wing exposure, spells the
E8-J v2 d̂ injections at median ~11/14 fields matching the RE-ENCODED
wing codes vs ~4/14 matching hand codes (CONFIDENCE 14v2, CALIBRATION
14v6; FAMILIARITY the lone inversion 5v6) — a third independent
instrument reading the re-encoded register over the hand code.

**Registered revision for the pre-authorized re-fly** (minimal, targeted
at the failed gate; everything not listed is IDENTICAL — same payload,
stimuli, split, eval rows, seeds, and ALL bars including the .55):

1. **EPOCH_CAP_F 6 → 12** (budget: the failure mechanism is cap-hit with
   descending strands; per-strand plateau still governs termination).
2. **Full strand ×2**: train-256 at BOTH α (256 → 512 rows; curriculum
   780 → 1036). The registered density menu ("denser pair/full strands");
   S8 says dense-code reading is the binding constraint and the full
   strand is what trains it. The α-parity rule is retired with it.

truedir/pair/atom/carrier/sham strands unchanged (epochs alone double
their optimizer passes). Smoke ladder re-flies first (build changed).
**Stop rule**: if the re-fly also fails G-INSTALL, that is the honest
stop — the record stands as an installation-capacity bound at this
curriculum family, and any redesign needs fresh registration (no third
fly on this prereg).

## §9 Flight 2 record — FF1/FF3: L3 LANDS (LOCKED)

**Flight 2** (`full_20260824_2311`, the §8 registered re-fly; smoke GREEN
23:09 → full 5219s wall / 4109s train, Joe's runs). GATES ALL PASS (G2
4e-08 · μ exact · ppl +0.64% · parse 3/421 INVALID = 0.7% · shams 0/24,
third consecutive clean-silence flight). **G-INSTALL PASS .7783** (p at
floor; from .4688 — the §8 epoch-starvation adjudication is CONFIRMED BY
ITS CURE, the e5d0920 pattern: cap 12 + full×2 was the entire change).
Verdict recomputed locally VERBATIM: 0 substantive diffs, all fresh-code
tallies match. Bundles archived `results_e8f/full_20260824_2311/`.

**P1 PASS — THE RUNG: pooled .6964 vs null .3276, p at the 1e-4 floor,
ALL 14 AXES individually Holm-significant** (range fg .5938 → e .8438).
Registered claim wording applies verbatim: *never-trained points of the
register are spelled by composition through the trained channel — the
code generalizes across the coordinate field, beyond any lookup over
trained points.* **L3 lands in bottleneck form. The featural rung of the
Hangul staircase is real: the 14 register axes are a readable
compositional alphabet.**

**P3 FAIL as registered** — .3604 vs null .3213, p .037; fx alone Holm
(.4453, p .0051), x/e nominal (.3906/.4141). **Replicated across both
flights** (fl.1 .377 p .015; same three leading axes fx/x/e both times,
never at bar): the FF3 wording stands — *interface-without-geography*,
the separability family's 4th instance. Joe's same-day hypothesis
("the shape inside the dictionary and the shape inside the head are
likely not the same") is hereby adjudicated twice in hardware with the
same answer: a real, faint, replicated seam — not a readable one at the
registered bar through this channel. Context on the record: cap-12 hit
with strands still descending (truedir .1063→.0841); the stop rule holds
— no third fly on this prereg; any P3-focused redesign needs fresh
registration.

**P2 FAIL at bar — with the program's first generative namings**: 2/128
rows decoded exactly, p at the 1e-4 floor vs derangement — **CRUCIBLE
and TEMPERANCE**, the first concepts ever named by the dictionary
decoding the model's spelling of a never-trained injection (S6: one from
the margin≥2 set, one from margin≤1). Texture, not a pass; the margin
analysis's fragility prediction held.

**S-blocks**: S2 RESOLVES toward direction-reading — vs-permuted .4509
p .002 (significant), vs-original .4196 p .057: the pre-stated second
reading (the R3-c S3 inheritance) is now visible in-flight; the readout
reads axis DIRECTIONS, geometry-general. S3: 542 errors, 94.8% adjacent-
tercile (514/28) — boundary calibration, the z/fz opposite-error
concentration of flight 1 dissolved. S8: the density wall LIFTED (dense
codes .63–.76, was .39–.53). S5: all 13/13 wing concepts now favor the
re-encoded codes (FAMILIARITY flipped to 8v2) — the re-encoding
preference replicated and strengthened. S7 carrier 4/4 all-MID. S4
titration cliff/degradation stable.

**E8-F COMPLETE — fork consequences per registration**: L4 design
(operator/ordered-composition grammar) UNPARKS, and the Ifá seed's
design check (docs/IFA_SEED.md) becomes runnable next; the Cirlot-100
atlas probe (docs/FRACTAL_LEXICON_SEED.md) may share that session's
instruments. Native-F1 (independent authorship) remains named future
work. The plain account (docs/PLAIN_ACCOUNT.md) updates with this
verdict per the plain-register law.
