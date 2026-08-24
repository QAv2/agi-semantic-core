# E8-J — The Featural Bridge (Hangul staircase, Level 2)

**Status**: PRE-REGISTERED (this commit precedes the build — lane law)
**Date**: 2026-08-24 · **Seed**: E8J_SEED = 20260825
**Model of record**: Qwen2.5-1.5B-Instruct · layers 14 (primary) / 20 (texture)
**License**: E8-R3-c LOCK, Fork 1 (SPELLING) — "Level 2 (featural/Jamo) licensed";
S3 registered reading: *"geometry-general readout capacity; the semantics claim
waits for level 2"* → **codebook-specificity is THE question this rung answers.**

---

## §0 Pick — why this rung, why this shape

The L2 sketch (memory, license granted 08-24): atoms = the 16D encoding's own
axes; estimate the linear bridge dictionary-space→hidden-space from the
concept-dir/coordinate anchor pairs; inject along features; compose per the
dictionary's own algebra; test whether the substrate runs it. Joe's framing:
*"this is what QA was always reaching for."*

The design check (§1, run BEFORE this registration) found the sketched
single-flight version is dead on arrival: the wing-13 anchor set cannot
determine a 14D bridge. Rather than fly into a measured wall, L2 is
restructured into **two stages inside one flight**: a broad-anchor
correspondence ATLAS (Stage A) that licenses, in-notebook, the injection RUNG
(Stage B). If Stage A fails its gate, the flight still ships a real result —
the substrate-level version of Phase 6, positive or negative.

## §1 Design check (local, pinned; `colab/e8j_design_check.py`, committed with this registration)

Inputs: the 13 wing concept dirs (L14/L20, unit, 1536-d) from the shipped
E8-R3-c real bundle (`results_e8r3c/full_20260824_0505`, dirs-stability
5e-08 — these are the flight-of-record stimulus dirs on the E4-real-instilled
model) × the 14D pack coordinates (`e4_dictionary_pack.json`, = DB
[x,y,z,e,f,g,h,fx..fh]; both w axes pinned 1.0 carry no signal).

**The wall (wing-13 anchors):**
- DC-0: wing coordinate cloud effective rank **5.03** of 14 (participation
  ratio; singulars [2.44 1.51 1.38 .86 .82 .61 .13 0…]). One semantic family
  spans ~5 dictionary dimensions. Hidden-dir cloud eff-rank 9.57.
- DC-1 RSA (Mantel, 78 pairs): ρ≈−0.05, p≈0.6 at EVERY featural layer
  (full14/essence7/function7/core3/domain4) × both hidden layers. Null.
- DC-2/3 LOO ridge bridge: median LOO angle 72.5° vs codebook-relabel null
  71.9° (p=.58); exact geometric hits **0/13** (null max 1). λ-sweep
  0.01→10 never beats the μ-baseline (52.3°) or nearest-other-dir floor (54.5°).
- DC-4 flight config (fit 9 trained → 4 held-out): beats the nearest-trained
  lookup floor on 1 of 4 (RETRIEVAL only).
- DC-5: the 9-anchor bridge's 14 feature axes collapse (min pairwise 0.0°,
  B rank ~6) — feature-axis injection from wing anchors was never viable.
- Salvage textures: folded RSA (fold hidden at 90°) ρ=+0.14 p=.12 — a whisper,
  right-signed, NOT a foundation. Relation contrast (5 dictionary
  opposition/complement pairs among wing-13): hidden angles 54–105° vs all-pair
  median 79.8° — no separation at n=5.

**The way through (also pinned):**
- Full pack cloud (3,052 concepts): effective rank **9.15**, all 14 singulars
  nonzero — the dictionary spans its space; the wall is the anchor family, not
  the dictionary. Wing-13 sits 97% inside the full cloud's top-6 PCs.
- Sampling frame: **1,878** pack concepts have desc ≥ 40 chars (the stimulus
  path needs "NAME: desc").
- Prior: Phase 6 measured a broad-sample linear correspondence in EMBEDDING
  space (384D→14D, R²=0.51 at ~3k concepts; domain axes e,f,g,h at 0.59–0.66,
  core quaternion 0.31–0.53; complement 90° structure NOT present — 95.5%
  compressed). Stage A is that measurement done on the substrate of record's
  own hidden space, with retrieval teeth.

## §2 Conditions and models

- **Stimulus model (primary)**: base + E4-real instillation merged (the
  pre-readout model of the locked E8-R real condition — the same model whose
  wing dirs the shipped bundle carries). All Stage-A primary dirs and the
  Stage-B bridge live here.
- **Stimulus model (contrast row)**: plain base model, L14 only — the
  untouched-substrate atlas. The instilled-vs-base delta is a registered
  secondary (S4): did E4 instillation move hidden geometry toward the
  hand codebook? No stability gate (no shipped base reference dirs); these
  ship as the future reference.
- **Stage-B condition model**: base + E4-real merged + locked E8-R real
  readout LoRA (E8-R2 reconstruction path verbatim; anchor-18 and ppl gates
  prove the reconstruction). The scrambled-instillation arm is NOT flown
  (E8-R2 §pick precedent + E8-N v2 base-anchor 2/18: naming requires the
  instilled+readout stack; the manipulated variable here is the CODEBOOK).
- **Codebook arms (Stage B)**: real vs **permuted** — same bridge B (fit once,
  on the real 320-anchor pairing), same span, same α·μ·unit-norm regime;
  the permuted arm injects normalize(B·[v_π(w),1]) where π = ONE pinned
  derangement of the 13 wing targets (seeded E8J_SEED, rejection-sampled,
  no fixed points). Permuting at the target-coordinate level (not refitting)
  holds machinery perfectly matched; the fit-level permutation is the
  analytic null (2000 derangements, in-verdict, no extra forwards).

## §3 Stage A — the correspondence atlas (eval-only)

**Anchor draw**: n=320 from the 1,878-concept frame (desc ≥40 chars),
EXCLUDING the wing-13 (they are the transfer test set, never anchors).
Stratified by trigram, proportional with per-stratum minimum 8, seeded
E8J_SEED, constrained draw PINNED in the builder and asserted on the VM
(R3-c pinned-draw law). Overlap with the frozen 256-name centroid draw is
harmless (common-mode subtraction) and reported.

**Dirs**: `pooled_reps` VERBATIM path ("NAME: desc", 64-tok cap, mean-pool,
minus 256-name centroid, unit-norm): 320 anchors + wing-13, instilled model
L14 + L20, base model L14. Shipped raw (5dp) for the recompute law.

**Bridge**: ridge, intercept unpenalized, **λ=1.0 pinned** (sensitivity rows
at 0.1 and 10 — texture; both Stage-B arms share whatever λ says, so λ cannot
manufacture codebook-specificity). LOO via hat matrix (exact; H depends only
on X, so the relabel null permutes Y rows against the same H).

**Instruments** (instilled L14 unless stated):
- **A1** RSA: Mantel Spearman, dict-14D pairwise angles vs hidden pairwise
  angles over the 320 anchors (51,040 pairs), 10k label perms.
- **A2** LOO bridge angles: median angle(d̂_i, d_i) vs 2000-perm codebook
  relabel null.
- **A3** LOO retrieval: top-1 (and top-5, texture) rate — is d̂_i nearest its
  own true dir among all 320? — vs the same 2000-perm null.
- **A4** axis-wise reverse map (hidden→dict, ridge, 10-fold CV R² per
  dictionary axis) — the Phase-6 comparison row (domain > core profile?).
- **A5** wing transfer: bridge fit on ALL 320 anchors → predict the 13
  never-anchored wing dirs; per-target angle(d̂_w, d_w); exact geometric hits
  (nearest of the 13 true wing dirs); vs 2000-derangement null; and vs the
  nearest-trained lookup floor per target (E8-R2's measured mechanism).

**Primary P-A**: A3 top-1 count, one-sided p<.05 vs the relabel null.
(A2 median-angle direction is a registered supporting row, not the primary.)

**Gate G-B (licenses Stage B in-notebook)**: BOTH of
  (i) A5 exact geometric hits ≥ 4/13, AND
  (ii) A5 hits > 95th percentile of the 2000-derangement null.
Fail → Stage B cells SKIP (banner says so); the flight ships the atlas.

## §4 Stage B — the injection rung (eval-only, conditional on G-B)

**Rows** (all FC-scored over the 13-name + NONE candidate set, mean-logprob
primary / sum twin, answer-sliced batched scoring — E8-R2 verbatim;
30-row progress prints — lane law):
- Anchor 18 verbatim (G1 ≥16/18; rides the measured 18/18).
- Locked shams 12 (G1b ≤1/12 claims; rides 0/24 measured silence).
- REAL arm: 13 targets × α{0.5,1.0} × 4 orders = 104 rows, injecting
  d̂_w = normalize(B·[v_w,1]) at L14, α·μ·unit (trained regime verbatim).
- PERMUTED arm: same 104 (target, α, order) grid, injecting the derangement's
  d̂_π(w).
- Titration texture: α=1.5, real arm, 13 × 2 orders = 26 rows (predicted dirs
  may sit past the measured 0.25→0.30 speech cliff; texture, no gate).

**Primary P-B** (only if G-B passed): exact naming (FC argmax == target) on
the trained-9 targets — real arm (72 rows) vs permuted arm (72 rows),
one-sided stratified row-level permutation (strata = target×α×order matched
pairs, 10k perms), p<.05. Registered robustness row: target-level sign-flip
exact test (2⁹ = 512 flips; bar ≈ 8/9 targets favoring real — steep, honest,
reported alongside). Registered mechanism row: behavioral hits should be a
subset of A5's geometric hits (concordance count; a behavioral hit on a
geometric miss is an anomaly to flag, not a pass).

**Secondaries**:
- **S1** held-out-4 floor concordance: does d̂_w reproduce the nearest-trained
  reading that the TRUE d_w produced in E8-R2 (DIVERGENCE→CONFIDENCE,
  RETRIEVAL→FAMILIARITY patterns)? Concordance real vs permuted.
- **S2** silence/claim texture across arms (claim-rate per arm; NONE mass).
- **S3** L20 atlas replication (A1–A3 at L20, texture).
- **S4** instillation delta: base-model atlas (A1–A3 on base L14) vs instilled —
  did E4 move the substrate toward the codebook?
- **S5** λ sensitivity (0.1 / 10) for A2/A3/A5.
- **S6** folded-RSA at n=320 (the n=13 whisper: ρ=+0.14 p=.12 — does it
  resolve into signal or noise at scale?).

## §5 Gates

- **G1** anchor ≥16/18 (rides 17-18/18 measured) — adapter reconstruction.
- **G1b** locked shams ≤1/12 claims (rides 0/24) — calibrated silence intact.
- **G2** dirs-stability: wing-13 instilled-model dirs vs shipped E8-R real
  bundle, sign-sensitive resid ≤1e-4 — v2, amended at smoke-1 BEFORE lock
  (the protocol locks at first full flight). Smoke-1 measured resid
  4.27e-05 at L20/CONSTRUCTION with everything else green (μ-drift assert
  passed): fp16 pooled reps are batch-composition-sensitive, and v1 had
  batched the wing texts together with the anchors while the flight of
  record computed them alone. v2 recomputes the gated wing-13 in the
  flight-of-record call shape (own 13-text pooled call, both models) and
  sets the tolerance between the measured kernel-noise class
  (1e-6..4e-5: E8-R2 7.6e-6 cross-day; smoke-1's mixed-batch 4.27e-5)
  and the smallest real-failure class (wrong adapter/model/desc/layer,
  ≥1e-3). v1's 1e-5 was tighter than the flown machinery's own default
  (1e-3) with no measured basis. Side effect, welcome: base and instilled
  dirs now share per-name batch shapes, so S4's delta row is cleaner.
- **G3** ppl reconstruction delta ≤0.5% (rides E8-R2's 0.0000%).
- **G4** counts/pins: 320-draw reproduces the builder's pinned draw exactly;
  derangement reproduces; curriculum/row counts exact; parse failures 0.
Gate failure → NO_VERDICT on affected primaries (lane law); Stage A
atlas numbers still ship (they are model-forward measurements, but carry
a GATES-DIRTY banner).

## §6 Registered forks

1. **P-A ✓, G-B ✓, P-B ✓** — the rung lands: never-anchored concepts are
   NAMED from hand-tuned coordinates alone through a bridge estimated on
   disjoint concepts. The dictionary's codebook is readable by the substrate;
   L3 (featural curriculum — training the readout ON bridge feature axes,
   composing per the algebra) is licensed.
2. **P-A ✓, G-B ✗** — correspondence at scale, but it does not reach the wing
   family (family-specific mismatch; adjudicates WHERE dictionary meets
   substrate — wing re-encoding or bridge v2 before any rung).
3. **P-A ✗** — no linear correspondence in the substrate's hidden space at
   n=320, against the Phase-6 embedding prior: a real cross-representation
   dissociation. Staircase pauses at L2; QA-lineage paper §4 inherits the
   honest null.
4. **G-B ✓, P-B ✗** — geometry transfers, readout doesn't execute it:
   titration texture localizes; v2 pulls L3's readout-side training forward.

## §7 Provenance & honesty

- **Authorship mediation**: dirs are the substrate's reading of hand-authored
  "NAME: desc" texts; descriptions and coordinates share authorship (wing-13
  co-authored session 125; pack descs from E4 authoring). A correspondence is
  therefore "hand geometry ↔ substrate's reading of hand-authored prose."
  Stated, not solved here; the independent-text robustness arm (third-party
  definitions) is future work, named now.
- **Instilled-substrate circularity**: E4 trained the model ON dictionary
  structure; the primary atlas measures a substrate E4 already pulled toward
  the codebook. S4's base-model contrast row is the registered control; the
  primary is on the instilled model because Stage B injects THAT model.
- **What this rung can claim**: P-B pass = the composed 14-axis spelling
  Σᵢ v[i]·Bᵢ of a never-anchored concept is read out by name. It does NOT
  claim the substrate natively contains QA's algebra (S4 speaks to native);
  it claims the algebra's atoms REACH the substrate through a linear bridge —
  the Jamo move: inject the spelling, the syllable is read.
- **Statistical units**: anchors are the units in Stage A (relabel null
  respects them). Stage B rows share targets; the primary permutes within
  matched-pair strata and the target-level sign-flip is reported alongside
  (R3-c row-level precedent + honesty row).
- **The wall is on the record**: §1's wing-13 numbers stand regardless of
  outcome — the sketched L2 premise ("~13 anchors suffice") is measured FALSE
  and this registration supersedes it.

## §8 Ops

Single self-contained `E8J_BRIDGE_UI.ipynb` (single-source builder
`build_e8j_notebook.py` + `e8j_logic.py`; E8-R2 machinery sliced with marker
asserts). MD0 operator card + 6 code cells: C1 setup/mount/staging (de-shelled
installs, RAM guards, MALLOC_ARENA_MAX, pack + shipped E8-R real bundle +
adapters) · C2 logic + pinned draws + asserts · C3 stimulus (base L14 pass →
E4 merge → instilled L14/L20 + μ + G2; inflight-ship atlas dirs) · C4 Stage A
computations + P-A + G-B banner · C5 Stage B (conditional; reconstruct →
G1/G1b → FC rows with progress prints; inflight ship) · C6 verdict (gates,
primaries, secondaries, banner, ship). Results → `MyDrive/semcore/e8j/`;
RESUME_STAMP bundle-level resume; ONE Run-all per mode (smoke ~10-14 min,
full ~30-45 min — eval-only, no training). SMOKE: N_ANCH=48, perms=200,
orders=2, α={0.5}, titration off, shams=3, anchor subset 6, and Stage B
FORCED to run (the skip branch is exercised in local tests — both-mode-
constants law). Colab UI-only, zero rclone, drive.mount only.

**Local test gates before staging** (lane law): `test_e8j_logic.py` (draws,
hat-matrix LOO == naive refit, derangement pins, gate logic at BOTH mode
constants, planted teeth: a random codebook must fail A3/A5, a copy bridge
must pass, planted Stage-B rows must pass/fail P-B as constructed) ·
`test_e8j_scorepath.py` (CPU tiny-model: batched==single answer-sliced FC
under the injection hook) · `test_e8j_verdict.py` (verdict cell exec'd
VERBATIM vs synthetic flights: fork-1 / fork-2-skip / fork-3 / fork-4 /
gate-fail / smoke).

---

# APPENDIX — FLIGHT RESULTS (full_20260824_1827)

**Flight of record**: smoke-2 GREEN (stamp 1824: gates all pass, forced
Stage B flew clean, sham claims 0/2) → full flight same evening (stamp
1827, ~5 min wall — Stage B skipped per G-B, so the flight is the atlas).
Joe's UI runs, secondary account, v2 notebook.

## Gates — ALL PASS
- **G2 dirs-stability: resid 5e-08** (L20/RESOLUTION, tol 1e-4) — the v2
  own-call fix restored exact-reproduction class (smoke-1's 4.27e-05
  mixed-batch diagnosis confirmed by the cure).
- **G1b ppl: 19.6358 vs 19.6358 shipped, delta 0.0000%** (E8-R2 class).
- **G4 pins: 320 anchors, sha dfb115cded2e1cd3, frame 1,865** — the
  build-time draw reproduced on the VM.

## Verdict — FORK 2 (registered): correspondence at scale, but it does not reach the wing family

**P-A PASS — the program's first measured dictionary↔hidden-space
correspondence at scale** (instilled model, L14, n=320):
- A1 RSA: ρ=0.1124, p=0.0001 (10k perms, 51,040 pairs).
- A2 LOO bridge: median 73.22° vs relabel-null 86.52°, p=0.0005.
- A3 retrieval: top-1 3/320 vs null mean 0.428 (≈7× chance, p=0.009);
  top-5 14/320. λ-stable (0.1/1/10 → top-1 2/3/3).
- S3 L20 replication: ρ=0.093 p=0.0005, top-1 3.
- Honest size: ρ≈0.11 and sub-1% top-1 = a REAL but THIN correspondence.
  The hand-tuned coordinates carry a small, unambiguous amount of
  substrate-real geometric information.

**G-B FAIL 0/13 — the wing family is geometrically DISCONNECTED from the
broad code**: wing-transfer median 89.83° ≈ orthogonal (null mean 86.96,
p=0.589); several predictions ANTI-aligned (CONFIDENCE 140.3°, DIVERGENCE
139.0°, CALIBRATION 126.0°, NOVELTY 125.3°). Adjudication (with §1's
design check — the wing-13's own internal RSA was already null): **the
wall is the wing family's coordinates, not the substrate and not the
method.** The wing was encoded in session 125 from cohort verbal
structure; the broad dictionary was hand-tuned over its own long history —
two encodings, two registers, and the bridge learned the broad one.

**A4 axis profile INVERTS Phase 6**: hidden dirs carry x (0.199) and fx
(0.196) — the primary yang/yin polarity axes — above all else; domain
axes e/f/g/h sit at 0.08–0.11 (Phase 6 embeddings: domain 0.59–0.66 >
core). Different representation, different carried features; the
substrate's NAME:desc reading encodes polarity most.

**S4 instillation delta: E4 did NOT move global geometry toward the
codebook** — base RSA ρ=0.1032 ≈ instilled 0.1124; base top-1 8 >
instilled 3 (delta −5). The instillation installed a naming INTERFACE
(E8-R anchor 18/18 requires it) without strengthening the atlas-level
correspondence — interface and geography dissociate, rhyming with E8-N
v2's separability instances. ⚠ ANOMALY FLAGGED: base loo_median 8.92°
(vs instilled 73.22°) alongside base top-1 only 8/320 — the signature of
a DEGENERATE base dir cloud (dominant common component ⇒ low prediction
angles trivially while retrieval stays near chance). Adjudication needs
the raw dirs (below); no interpretive weight is placed on the 8.92°
number until then.

**Smoke-2 texture** (n=2, anecdote only): the real-codebook bridge
prediction for DIVERGENCE reproduced its E8-R2 modal reading
(CONFIDENCE, 2/2) while the permuted arm read FAMILIARITY (concord 1/2)
— right-signed whisper for an eventual rung, claimable as nothing.

## Recompute (the law) — status
- **A5/G-B recomputed locally EXACT** from independent inputs (true wing
  dirs from the local R3-c real bundle — G2's 5e-08 makes them the
  flight's dirs; predictions from shipped atlas.json): all 13 per-target
  angles match to ≤0.0054° (5dp ship rounding), nearest-names identical,
  hits 0/13, median 89.83, lookup floors reproduce, derangement matches
  the local pin.
- **Verdict cell exec'd VERBATIM over the shipped atlas: 0 differences**
  vs the shipped verdict (fork banner line-for-line).
- **A1–A4 + S4 adjudication PENDING raw dirs**: atlas_dirs.json is 19MB,
  over the Drive integration's 10MB cap → Joe's one-click folder
  download (R3-c precedent), then the stage-A statistics recompute
  verbatim and the base-cloud degeneracy check (effective rank +
  μ-baseline angle) closes the S4 anomaly. LOCK lands with that.

## Forward path licensed by Fork 2 (for the next pick, not committed)
(a) **Reverse-bridge wing re-encoding**: estimate wing coordinates FROM
their substrate dirs via the broad bridge — the delta against the hand
coordinates is itself the measurement of the mis-encoding; then the rung
re-flies on re-encoded coordinates. (b) Hand re-encode the wing in the
broad code's register (label-over-band contract). (c) Enlarge the atlas
toward the full 1,865-name frame for a stronger bridge first. The atlas
instrument (320 anchor dirs, both models, both layers) is now standing
equipment either way.
