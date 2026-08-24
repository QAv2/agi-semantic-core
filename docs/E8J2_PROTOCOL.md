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

---

## APPENDIX — Part 1 measurement: FM1 (2026-08-24, session 132) — LOCKED

Run: `colab/e8j2_measure.py` → `results_e8j2/e8j2_verdict.json` (5.8s
local). Machinery teeth green in-run (predict∘permute identity; rank-sum
planted/null p=.0005/.43). **G-M1 re-verify PASS p=.0005** (obs 3.163,
null 3.769 — matches §1 exactly on a fresh seed).

**P-M: MIS-ENCODING MEASURED.** Wing Δstd median 4.364 vs region floor
2.258 (n=64), p=.0001 (10k label perms); conservative secondary agrees —
11/13 above the full-anchor median, binomial p=.0112. **No FM4 split.**
Per-concept percentiles vs the anchor-LOO distribution: worst = SATURATION
99.4, RESOLUTION 97.8, CONSTRUCTION 96.6, RETRIEVAL 96.2; most in-register
= CONFABULATION 42.2, CAPTURE 49.7. E8-J's Fork-2 adjudication is now a
quantity, not an inference.

**Where and how it is mis-encoded**:
- **S-M1 axis profile**: the mismatch concentrates on core z/fz (signed
  +0.38/+0.28 — the substrate reads the wing HIGHER on z than the hand
  code) and x/fx (−0.40/−0.36 — the substrate reads the wing MORE YIN),
  then g/fg (−0.36/−0.48).
- **S-M2 systematicity**: Procrustes resid .854 vs null .882, p=.0855 —
  NOT a coherent rotation at α=.05. Registered reading applies:
  per-concept mis-assignment, not a family-level rotated frame. "Two
  registers" sharpens to: **there is no clean transform between them**.
- **S-M3**: hand↔dirs RSA ρ=−.056 p=.63 (the design-check-era null, now
  on the measured record); x̂ cloud eff-rank 5.57 (hand 5.08); x̂↔dirs
  ρ=.570 p<.001 texture (by construction, marked).
- **S-M4 depth replication STRONG**: x̂(L20) vs x̂(L14) per-concept cos
  median .884, per-axis r median .851 — the reverse read is
  register-stable across depth, not layer noise.
- **S-M5 base control**: reverse estimates from the base cone collapse to
  spread .011 (vs instilled .315) — the locked degeneracy finding
  reproduced in reverse, as predicted.
- **S-M6 shrinkage sensitivity**: un-shrunk P-M p=.0001, G-M2 4/13 @
  67.5° — all conclusions shrinkage-robust.

**G-M2 PASS AT THE BAR → Part 2 STAGES.** 4/13 exact geometric hits
(bar ≥4; derangement null p95 = 2.0), median angle 67.5°, vs the flown
hand-coordinate A5 at 0/13 @ 89.8°. Per §1 calibration, 4/13 IS the
healthy in-register 13-set median — the re-encoded wing round-trips like
a native part of the broad register. Hits = **RESOLUTION, RETRIEVAL,
CONSTRUCTION, FAMILIARITY** — three of the four worst-mis-encoded
concepts: where the hand code was most wrong, the re-encode moves
farthest and the round trip lands. Coherent mechanism, on the record.

**Prediction outcomes (honesty)**:
- P-M mis-encoding: predicted, CONFIRMED.
- S-M5 base collapse: predicted, CONFIRMED.
- **S-M7 modal concordance: prediction MISSED** — x̂ 1/4 vs hand 1/4 (no
  improvement; only TENSION→CALIBRATION matches under x̂). The
  coord-space nearest under shrinkage is not the dir-space FC floor
  geometry. The flight's S1′ behavioral row is the operative test.
- G-M2: registered as uncertain (52% calibration); passed at the median.

**S-M8 + post-hoc shrinkage sensitivity (labeled, unregistered row)**:
raw-x̂ bands break 3/5, but the un-shrunk variant restores
FAMILIARITY–NOVELTY (136°), RETRIEVAL–CONSTRUCTION (103°), and
TENSION–RESOLUTION (89°) — those three collapses were estimator
shrinkage. Two genuine findings survive:
- **UNCERTAINTY–CONFIDENCE: the substrate disagrees with the label.**
  Their full 1536-D dirs sit at **54.5°** — kinship, not opposition
  (un-shrunk coords 48°). A real code-vs-substrate disagreement for the
  wing-v2 ledger, not an artifact.
- **CONFABULATION–CALIBRATION: a register-capacity limit.** Full dirs at
  79.9° (healthy complement-band separation) but merged in the readable
  subspace (9° raw / 18° un-shrunk) — their difference lives in
  dimensions the broad bridge cannot see. Distinct mechanism from the
  above, both on the record.

**Part-2 power note**: trained-9 ∩ geometric hits = {RESOLUTION,
CONSTRUCTION, FAMILIARITY} — the mechanism model expects behavioral wins
concentrated in those ~24/72 real rows; the stratified primary permutes
within matched pairs, so concentration is what it is powered for.

**Artifacts**: `results_e8j2/wing_reencode_v1.json` (x̂ L14/L20/base, σ,
provenance) · `results_e8j2/rung_pins.json` (d̂ real arm, pinned
derangement, geometric-hit list, sha `685255b9eec94a3a`).

**Part 1 LOCKED. FM1 → Part 2 staged: `E8J2_RUNG_UI.ipynb`.**

---

## APPENDIX — Part 2 flight: FJ1 (2026-08-24, `full_20260824_2006`) — LOCKED

**Flight record**: smoke GREEN (Joe's run, ~1956). Full attempt 1 died at
PARSE time — IndentationError in the FLY cell. Diagnosed from the staged
bytes without touching the VM: the staged cell compiles clean locally, and
the VM's reported error line (27) vs the staged line (36) showed ~9 lines
missing — the VM's cell text had diverged post-smoke (the R3-c divergence
class, caught pre-execution this time). Remedy per lane law: NO in-place
repair — fresh upload of the byte-identical staged notebook, straight to
full (the green smoke had validated this exact build). Full flew clean,
352s. Nothing partial preceded it (parse error = the flight never started).

**GATES ALL PASS, pristine**: anchor 18/18 · locked shams 0/12 claims ·
ppl reconstruction delta 0.0000% (19.6358 exact) · G4 pins (on-VM sha
recompute == `685255b9eec94a3a`, 208/208 rows, all inject hooks fired,
shams uninjected, titration parse fails 0).

**Primary P-B′ PASS p=.0001** — real arm 19 exact vs permuted arm 0 over
the trained-9 matched pairs (72 vs 72); alpha-balanced (10 @ α0.5, 9 @
α1.0 — not an alpha artifact). Sign-flip robustness p=.01562 = 8/512:
ALL 6 nonzero targets favor real (CONSTRUCTION +7, CALIBRATION +4,
CONFIDENCE +3, CAPTURE +3, FAMILIARITY +1, SATURATION +1); the registered
"≈8/9 targets" prose bar is not met (6/9 nonzero) and is reported
alongside as registered — the exhaustive test is significant.

**Mechanism row CLEAN**: behavioral hits {CONSTRUCTION, 7/8} ⊆ geometric
hits {CONSTRUCTION, FAMILIARITY, RESOLUTION}; anomalies none. Both
sub-flag dissociation directions are on the record as texture:
- RESOLUTION: geometric hit, 0 behavioral — the geometry lands where the
  readout does not execute it.
- CALIBRATION 4/8, CONFIDENCE 3/8, CAPTURE 3/8: behavioral success
  WITHOUT a geometric hit (all under the ≥5 flag bar) — the readout's
  decision regions are its own geometry, not nearest-in-13 angles.

**Secondaries**:
- **S1′ held-out concordance: real 4/32 vs perm 8/32** — the E8-R2 modal
  confusion structure does NOT survive the bottleneck (real-arm held-out
  modals collapse to a CALIBRATION attractor; TENSION reads CONFIDENCE).
  Coherent with Part 1's S-M7 miss: the re-encode carries trained-name
  IDENTITY through the register; it does not carry the held-out confusion
  geometry. The perm arm's 8/32 includes a coincidental TENSION 5/8
  (deranged dir happens to read as TENSION's modal).
- S2: none_top rate 0.0 both arms (injection always claims — E8-R2
  regime confirmed). Held-out-4 real exact 0/32 (texture): untrained
  names still carry no output mass, E8-R2's finding intact.
- S_titr α=1.5: claim .154, exact 0, invalid 22/26 — past-cliff
  degradation, texture as registered.

**RECOMPUTE CLEAN (the law)**: the staged notebook's verdict cell exec'd
VERBATIM over the raw shipped `rungb.json` → **0 differences** against
the shipped verdict, and independent fresh-code tallies reproduce every
primary ingredient (19/0/19, 18/18, 0/12, hooks, alpha split).
`colab/recompute_e8j2_flight.py` committed as the instrument; bundles
archived `results_e8j2/full_20260824_2006/` (byte sizes match the Drive
listing; pins sha verified in-bundle).

**FORK: FJ1 — the rung lands in BOTTLENECK FORM** (registered wording
verbatim): wing identity survives compression into the 14-coordinate
dictionary register and returns through the forward bridge into the
locked readout's naming. Codebook-capacity claim — NOT independent
authorship; native-F1 (independently authored coordinates) remains named
future work. **L3 (featural curriculum) licensed in bottleneck form.**

**Part 2 LOCKED. E8-J v2 complete: the mis-encoding measured (Part 1),
and the re-encoded register executes through the readout (Part 2).**
