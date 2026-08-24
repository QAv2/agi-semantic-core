# E8-J v2 — Reverse-Bridge Wing Re-Encoding (pre-registration)

**License**: E8-J LOCK, Fork 2 forward menu item (a) — "estimate wing
coordinates FROM their substrate dirs via the broad bridge — the delta
against the hand coordinates is itself the measurement of the mis-encoding;
then the rung re-flies on re-encoded coordinates." Joe's pick, verbatim
("take the fork, go with (a)"), 2026-08-24, session 132.

**Status**: PRE-REGISTERED before the measurement script exists (lane law).
The design check (§1) is committed alongside; it is anchor-side only — no
reverse estimate of any wing dir has been computed anywhere.

**Equipment**: the locked E8-J atlas (`colab/results_e8j/full_20260824_1827/`
— raw dirs, 320 anchors + wing-13, both models, both layers), the E8-J
machinery (`e8j_logic.py`, single source), the e4 pack coordinates, and (for
the conditional Part 2 flight) the locked E8-R real readout bundle.

---

## §0 Question

E8-J adjudicated its G-B fail (wing transfer 0/13, median 89.8°, several
predictions anti-aligned) as a register problem: the session-125 wing
coordinates are encoded in a different register than the broad dictionary
the bridge learned. That adjudication rested on the design check's
wing-internal nulls plus the fail itself. This rung asks the quantitative
version, in two parts:

- **Part 1 (local measurement)**: read the wing's coordinates OUT of the
  substrate through the reverse bridge. Is the hand encoding's deviation
  from the substrate-derived encoding LARGER than the bridge's own noise
  floor? Where (which axes), and how (coherent transform vs incoherent)?
- **Part 2 (conditional flight)**: do re-encoded coordinates round-trip
  through the forward bridge into the locked E8-R readout's *behavior* —
  the rung, re-flown in bottleneck form (§4 restates the claim honestly).

## §1 Design check (measured before this registration; anchor-side only)

`colab/e8j2_design_check.py`, summary `colab/results_e8j2_design_check.json`:

- **Machinery teeth GREEN**: exact reverse LOO == naive refit (3 λ);
  planted in-register world passes the relabel-null logic (p=.005), a
  permuted world fails (p=.269). A4 axis-wise row recomputes EXACT vs the
  flown atlas.
- **G-M1 measured (instrument gate): PASS** — reverse LOO (instilled L14,
  λ=1.0) standardized-Δ median 3.163 vs relabel-null 3.769 (null min
  3.609), p=.0005/2000 perms. The reverse bridge exists.
- **λ_rev = 1.0 pinned** (best of {1,10,100} by median standardized Δ;
  symmetric with the forward LAM=1.0). Per-axis LOO R² .06–.26, all 14
  positive; shrinkage: LOO predictions carry ~42% of register SD (median).
- **Register structure on the record**: function block ≈ 0.80 × essence
  (corr .85–.97 per pair; anchor coord cloud eff-rank 8.40/14). The
  register is ~8 effective dimensions wearing 14 coordinates.
- **Region control (changes the primary's floor)**: anchors NEAR the wing's
  hand-coordinate region are EASIER to predict (rank-corr distance↔error
  +.576; near-20% floor Δstd median 2.258 vs full-floor 3.163) — the
  shrunken estimator favors non-extreme coordinates and the wing region is
  central. Consequences, fixed BEFORE any wing read: (i) the primary uses
  the REGION-MATCHED floor (the empirically correct one); (ii) a wing
  delta excess cannot be attributed to region hardness; (iii) the
  full-floor comparison is kept as the conservative secondary.
- **Round-trip calibration (calibrates G-M2 and Part 2 expectations)**:
  double-LOO forward∘reverse round trip on anchors — own-angle median
  57.9°, retrieval top-1 8/320; random in-register 13-sets clear the
  ≥4/13 within-set retrieval bar only **52%** of the time (median 4/13).
  A G-M2 fail is therefore a stage-gate outcome, not a disproof; the bar
  is kept anyway because it is the flight's own G-B bar (continuity), and
  flying a re-encode that cannot clear it locally is pointless.
- **L20 reverse similar** (Δstd med 3.172); **base L14 worse** (3.504,
  shrink .177) — consistent with the locked cone-degeneracy finding.

## §2 Part 1 — the measurement (local, pure numpy, no flight)

**Estimator (M1)**: x̂_w = raw affine reverse ridge (unpenalized intercept,
NO row normalization), fit on ALL 320 anchors, instilled L14 dirs → 14D
coords, λ_rev=1.0. The wing-13 was never in any bridge fit (never-anchored,
asserted). Standardization σ = per-axis SD (ddof=1) of the 320 anchor TRUE
coordinates. Δstd(c) = ‖(x̂_c − x_c^hand)/σ‖₂.

**Primary P-M (the mis-encoding test)**: one-sided rank-sum, wing Δstd
(n=13) vs NEAR-REGION anchor LOO Δstd (n=64: bottom-20% standardized
distance to the hand wing centroid — the design-check §7 population),
permutation p over 10,000 group-label shuffles (seed E8J_SEED+55), α=.05.
- **Reading if wing > floor (expected, from A5 0/13 + anti-alignments)**:
  mis-encoding measured beyond bridge noise — E8-J's Fork-2 adjudication
  quantified.
- **Reading if within noise**: the hand coordinates are consistent with
  the broad register at bridge resolution — the adjudication WEAKENS and
  the G-B wall must be relocated (bridge thinness on this family, or
  family dir-side idiosyncrasy). Registered honestly as FM3.

**Secondary floor (conservative)**: k = #wing concepts with Δstd above the
FULL 320-anchor LOO median; exact one-sided binomial(13, .5), pass at
k ≥ 10 (P = .046). Split-floor reading registered as FM4: the
region-matched primary is operative (justified §1, fixed pre-read); the
split is reported.

**Gate G-M1 (re-verify in-run)**: relabel null, 2000 perms, fresh seed
(E8J_SEED+50); PASS = p<.05. Expected ≈ .0005 (§1). Fail → NO_VERDICT on
P-M (instrument dead on the day; investigate, do not interpret).

**Gate G-M2 (stage gate for Part 2)**: A5 VERBATIM with re-encoded
coordinates — `wing_transfer(Xa, Ya14, x̂_w, Dw14, λ=1.0)` (forward bridge
all-320, predict wing dirs from x̂_w) + `derangement_null_transfer` (2000,
seed E8J_SEED+51). PASS = exact geometric hits ≥4/13 AND > 95th pctile of
the derangement null (G-B bar, continuity). Directly comparable to the
flown A5 on hand coordinates (0/13, median 89.8°). Calibration on the
record: healthy in-register 13-sets pass ~52% (§1). Pass → Part 2 stages.
Fail → FM2 (no flight; the measurement is the product).

**Secondaries** (all registered, all local):
- **S-M1 axis profile**: per-axis signed mean z and mean |z| of
  (x̂ − x_hand)/σ over the 13 — WHERE the mis-encoding lives. Prediction
  (texture): the dims the 5.03-eff-rank hand family collapsed.
- **S-M2 systematicity**: orthogonal Procrustes (numpy SVD, centered
  clouds, rotation only) mapping the estimate cloud onto the hand cloud;
  residual = ‖X̂cR − Xc‖_F/‖Xc‖_F vs 2000 name-permutation nulls (seed
  E8J_SEED+52). Below-null residual = coherent re-registration (a
  transformed register); at-null = incoherent per-concept mis-assignment.
  n=13 in 14D overfits the rotation; the null uses the same machinery, so
  the comparison is calibrated (stated).
- **S-M3 cloud diagnostics**: eff-rank of the x̂ cloud (hand cloud: 5.03);
  13×13 Mantel RSA (2000 perms): hand-coords vs true wing dirs (expected
  ≈ null — the design-check-era finding, now on the measured record; seed
  +53) and x̂ vs true wing dirs (expected positive BY CONSTRUCTION —
  texture only, marked as such; seed +54).
- **S-M4 depth replication**: x̂ from instilled L20; per-concept cosine to
  the L14 estimate in standardized centered coords (median), per-axis
  Pearson across the 13. Register-level consistency across depth.
- **S-M5 base negative control**: x̂ from base L14; per-axis SD(x̂)/σ
  (median) vs the instilled row — expected collapse toward the anchor
  mean (cone degeneracy, locked finding).
- **S-M6 shrinkage sensitivity**: per-axis un-shrink by ANCHOR-calibrated
  factors σ/SD(LOO-pred axis) at λ_rev; recompute P-M primary and G-M2.
  Texture (raw estimator remains the registered artifact).
- **S-M7 modal concordance**: for the held-out 4 (DIVERGENCE, NOVELTY,
  RETRIEVAL, TENSION): nearest TRAINED-9 neighbor in standardized coord
  space under x̂ vs under hand coords, each compared to the locked E8-R2
  modal readings (DIVERGENCE→CONFIDENCE, NOVELTY→CONFABULATION,
  RETRIEVAL→FAMILIARITY, TENSION→CALIBRATION). Prediction: x̂ concordance
  ≥ hand concordance (the estimates come from the geometry the readout
  actually reads).
- **S-M8 validator report (label-over-band)**: the 5 intra-wing DB
  relations (4 oppositions + TENSION–RESOLUTION complement): core-3D and
  essence-7 angles under hand vs x̂; opposition passes either regime
  (>60°, i.e. not-kinship), complement band 60–120° core. Report-only —
  **the DB is not touched** (§6).

**Artifact**: `colab/results_e8j2/wing_reencode_v1.json` — x̂_w (L14
primary; L20/base rows included), λ_rev, seeds, σ, atlas provenance sha,
G-M1/G-M2/P-M verdicts. This is a FLIGHT INPUT artifact, not a dictionary
update.

## §3 Part 2 — the rung re-fly (conditional on G-M2 PASS)

`E8J2_RUNG_UI.ipynb`, eval-only, E8-J §4 Stage B machinery VERBATIM
(e8r2 scoring, answer-sliced FC over the 13-name+NONE candidate set,
mean-logprob primary / sum twin, 30-row progress prints), with one
substitution and one simplification:

- **Substitution**: injected dirs are the PINNED forward-bridge
  predictions from re-encoded coordinates — real arm
  d̂_w = normalize(B·[x̂_w,1]), permuted arm d̂_π(w) with π = the SAME
  pinned `wing_derangement` E8-J carried. Both minted LOCALLY at
  measurement time from locked artifacts (bridge W: forward ridge all-320,
  λ=1.0 — the flown A5's W), embedded in the notebook, sha-pinned.
  (Row-wise identity: predicting permuted coords == permuting predictions;
  the arms are machinery-identical by construction.)
- **Simplification (registered difference from E8-J)**: NO on-VM dir
  recompute and therefore NO G2 dirs-stability gate — the flight computes
  no stimulus pass; it injects pinned vectors into the locked readout.
  G4 covers the pins instead (sha recompute of embedded vectors, unit-norm
  asserts, derangement identity, row counts, parse 0). G1 anchor-18
  ≥16/18, G1b locked shams ≤1/12 claims, G3 ppl reconstruction ≤0.5%
  ride unchanged.

**Rows**: anchor-18 · locked shams 12 · REAL 104 (13 targets × α{0.5,1.0}
× 4 orders) · PERMUTED 104 (same grid, deranged dirs) · titration texture
26 (α=1.5, 2 orders).

**Primary P-B′**: exact naming on the trained-9 targets, real (72) vs
permuted (72), one-sided stratified matched-pair permutation (strata =
target×α×order, 10k), p<.05. Robustness row: target-level sign-flip exact
(2⁹; bar 8/9 favoring real). Mechanism row: behavioral hits ⊆ the LOCAL
geometric hits (G-M2's per-target hit list, passed in as pins); a
behavioral hit on a geometric miss is an anomaly to flag.

**Secondaries**: S1′ held-out-4 behavioral reading vs the E8-R2 modals,
real vs permuted concordance · S2′ claim-rate/NONE-mass per arm ·
titration curve texture.

**Smoke**: SMOKE=True path with reduced constants forces the full cell
path; local tests exercise both mode constants (lane law).

## §4 Claim discipline (honesty)

- **The circularity, stated plainly**: x̂_w is derived FROM the substrate's
  own wing dirs. Part 2 therefore does NOT re-establish E8-J Fork 1's
  claim ("named from hand-tuned coordinates alone"). What Part 2 can
  claim if P-B′ passes: **wing identity survives compression into the
  14-coordinate dictionary register and returns through the forward
  bridge into the locked readout's naming behavior** — a
  codebook-capacity result (the register is rich enough to carry these
  identities end-to-end), not an independent-authorship result. The
  native-F1 rung (independently authored coordinates) remains open and is
  named as future work.
- **Shrinkage**: the estimator carries ~42% of register SD; deltas are
  measured against floors built from the SAME estimator (fair), and S-M6
  reports the un-shrunk sensitivity.
- **Floor choice**: region-matched primary fixed from anchor-side evidence
  BEFORE any wing read (§1); the conservative full floor rides as
  secondary; FM4 registers the split reading.
- **Family-vs-anchors caveat**: the wing is one tight semantic family;
  anchor LOO floors come from a spread draw. S-M2 (systematicity) and
  S-M5 (base control) bracket this; residual limitation stated.
- **Authorship mediation** (inherited from E8-J §7): dirs remain the
  substrate's reading of hand-authored NAME:desc prose.
- **The wall stands regardless**: E8-J's G-B 0/13 on hand coordinates and
  the wing-13 eff-rank 5.03 are on the record whatever Part 1 finds.

## §5 Registered forks

- **FM1**: P-M mis-encoding + G-M2 PASS → Part 2 stages; the flight
  decides the bottleneck-form rung. Flight forks: **FJ1** P-B′ pass —
  the rung lands in bottleneck form; L3 (featural curriculum) licensed in
  bottleneck form, native-F1 stays open. **FJ2** P-B′ fail — geometry
  transfers, readout doesn't execute (E8-J fork-4 analog); titration
  texture localizes.
- **FM2**: P-M mis-encoding + G-M2 fail → no flight; the delta
  measurement + axis profile + S-M7 are the product, feeding menu (b)
  (hand re-encode, guided) or (c) (atlas enlargement, then retry).
- **FM3**: P-M within noise (both floors) → the register adjudication
  weakens; wall relocates (bridge thinness / family dir idiosyncrasy);
  menu (c) becomes the path.
- **FM4**: floors split → region-matched primary is operative
  (pre-justified §1/§2); split reported verbatim.

## §6 Instruments, pins, provenance

- Scripts: `colab/e8j2_design_check.py` (committed with this
  registration), `colab/e8j2_measure.py` (written AFTER this registration;
  reuses `e8j_logic` verbatim where roles allow — `ridge_fit`,
  `hat_matrix`, `wing_transfer`, `derangement_null_transfer`,
  `mantel_spearman`, `angles_rowwise`; raw-affine reverse variants carry
  exact-LOO==refit teeth in the design check).
- Seeds: design check E8J_SEED+40..42; measurement +50 (G-M1 re-verify),
  +51 (G-M2 derangement), +52 (Procrustes null), +53/+54 (Mantel), +55
  (P-M label permutation). Disjoint from all flown streams (≤ +13).
- Results archive: `colab/results_e8j2/` (measurement verdict json +
  re-encode artifact; flight bundles later if Part 2 flies).
- **The DB is untouched**: wing v1 coordinates stay as-is in
  `db/semantic.db`. Adoption of any re-encoding into the dictionary is a
  SEPARATE future step gated on (i) the rung's outcome, (ii) the
  label-over-band validator, (iii) Joe's word — noting the standing
  encoding rules protect existing concepts precisely against casual
  re-encoding.
- Push/public remains Joe's gate; commits local.
