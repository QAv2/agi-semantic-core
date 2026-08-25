# E8-O2 — The Mixed-Figure Curriculum (breaking the méjì shortcut)

**Pre-registered before build** (lane law). Fresh registration: E8-O's
forks were terminal. Design check ran first (`colab/e8o2_design_check.py`,
artifact `colab/results_e8o2/design_check.json`).

## §0 Position

E8-O locked FO2 with the mechanism measured: the E8-F readout reads
composed points by the ESSENCE leg and IMPUTES the function leg as the
senior donor's méjì completion (.4420 vs .3824 truth) — a shortcut the
register's f≈0.80e redundancy made correct on 98.7% of everything it was
ever trained on. E8-O2 is the registered cure, the E8-F precedent applied
to composition: **continue the locked readout's training on leg-decoupled
(mixed-figure) points, where the shortcut pays chance, then re-fly the
flown order eval.** The claim at stake completes L4: order discrimination
through the trained channel.

## §1 Design check — measured basis (2026-08-25, all local)

- **Draws pinned**: TRAINPAIR48 = a seeded perfect matching over 96
  DISTINCT train-256 concepts (sha 8e1dbade408eeb69; maximal diversity,
  degree exactly 1); POFRESH12 = fresh eval-64 pairs, pair-disjoint from
  the flown PAIR24, degree 1 (sha fe6f36106ca209dd); SPOTMIX12 ⊂
  TRAINPAIR48.
- **Train-pair geometry matches the eval geometry**: order-separation
  median 109.3° through the pinned W, 100% ≥ the 20° channel bar (eval
  pairs: 100.8°, carried from the E8-O check).
- **The shortcut is STARVED by construction**: across the mixed training
  targets, the function code matches the senior's méjì completion at
  .360 per-field — tercile chance — versus 1.000 on every concept row in
  the readout's entire schooling. The curriculum removes exactly the
  food the shortcut ate.
- **S-MEJI reversal statistic toothed**: leg-faithful world +.685,
  shortcut world −.685, NONE-flood 0.000.
- **Flight-1 baselines carried as the comparison row** (quoted, not
  re-flown): P-O1 .4464 (essence ~.51 / function ~.38), P-O2 d .0275
  p .019, fun-vs-B .3824 / fun-vs-A-méjì .4420 (S-MEJI ≈ −.06),
  G-ID .5982.

## §2 Registered design — CONTINUE-training flight

**Arm (the mechanism claim)**: the locked E8-F readout
(`e8f/inflight_20260824_2311/readout_real`) is loaded TRAINABLE
(`PeftModel.from_pretrained(..., is_trainable=True)`) on the E4-real
merged base and continues training on the mixed curriculum. Fresh-init is
NOT flown; it is named only as a fallback consideration under the single
re-fly clause below. Injection/scoring machinery: E8-F/E8-O verbatim
(same payload, W, terciles, μ assert, Injector, α ∈ {0.5, 1.0}).

**Curriculum** (single condition; strands ['mixed','replay','sham'];
per-strand plateau min 2 rel .05, cap 8; LR 1e-4, ACCUM 8, answer-sliced
CE — all E8-F verbatim):

| strand | rows | construction |
|--------|------|--------------|
| mixed  | 192  | TRAINPAIR48 × BOTH orders × 2α — injected pred(compose_A), target = the EXACT composed code (A's 7 + B's 7) |
| replay | 52   | 24 pinned train concepts × 2α full-code rows + 4 carrier rows (anti-forgetting ballast) |
| sham   | 24   | no injection → NONE |

268 rows total; est. train ≤ 15 min, flight ~30 min.

**Eval** (198 rows, pinned):

| block      | rows | construction |
|------------|------|--------------|
| spot_mixed | 24   | SPOTMIX12 × both orders × α1.0 (installation gate) |
| po         | 96   | the FLOWN PAIR24 × both orders × 2α — VERBATIM the E8-O primary block (direct before/after) |
| po_fresh   | 48   | POFRESH12 × both orders × 2α (consistency secondary) |
| ident      | 16   | the same 16 re-flown E8-F rows (retention gate) |
| sham       | 12   | → NONE |
| carrier    | 2    | → all-MID |

**Primaries**:

- **P-M1 — ORDER (the L4 claim, completing FO2)**: the E8-O P-O2
  statistic VERBATIM on the po block (per-pair own-vs-flipped accuracy
  delta, 10k sign-flips over the 24 pairs). PASS = p ≤ .0025.
- **P-M2 — FUNCTION-LEG READING (the starved leg)**: pooled accuracy on
  the SEVEN FUNCTION AXES of the po block vs 10k pair-derangements
  (the P-O1 construction restricted to fun axes). PASS = p ≤ .0025.

**Registered secondaries**: **S-MEJI** — the reversal statistic goes
POSITIVE (sign-flip over pairs, p ≤ .0025; baseline ≈ −.06): the readout
reads the junior leg's truth better than the senior's méjì completion.
**S-FRESH** — po_fresh reproduces the po directions (d > 0 and
function-leg above its null, nominal — consistency, no Holm).
**S-ESS** — essence-leg accuracy maintained (texture vs .51 baseline).
Before/after table vs every §1 baseline ships in the verdict.

**Gates**: G1 pins (payload + both pair shas + curriculum counts +
init-adapter path asserted) · G2 wing identity probe (standing
convention) · **G-ID retention ≥ .55** on the ident block (the readout
must still read plain concepts after the new schooling) · **G-INSTALL-M**:
spot_mixed pooled ≥ .55 AND derangement p ≤ .0025 (trained mixed pairs
must be readable — else NO_VERDICT, installation not generalization) ·
G3 ppl ≤ 5.0% (trained-readout class) · G4 parse-INVALID ≤ 10% ·
G5 sham ≤ 2/12.

**Single re-fly clause**: ONE pre-authorized re-fly total, triggered
only by G-ID retention failure (revision: doubled replay strand) or
G-INSTALL-M failure (revision: epochs/budget — the E8-F pattern). A
second failure of either ends this prereg. All other outcomes are
terminal forks.

**Registered forks**:

- **FV1** — P-M1 ✓ ∧ P-M2 ✓: **L4 LANDS in bottleneck form** — both legs
  read, order behaviorally discriminated; the QA algebra's first measured
  asymmetric operator, completing what FO2 left; the méjì shortcut is a
  CURRICULUM artifact, cured by schooling (the E8-F precedent's second
  confirmation-by-cure).
- **FV2** — P-M2 ✓ ∧ P-M1 ✗: the function leg becomes readable but order
  STILL collapses — the wall is deeper than curriculum (channel/
  architecture class); honest stop for this approach; any further attempt
  needs a new design, not a re-fly.
- **FV3** — P-M1 ✓ ∧ P-M2 ✗: order discriminated through essence-side
  cues alone — partial, localized by the per-axis table; claim bounded.
- **FV4** — gate-triggered NO_VERDICT paths per the re-fly clause.

## §3 Claim discipline

Bottleneck form; register-side operator (the algebra composes, the
substrate reads); an FV1 pass claims order-discrimination THROUGH THE
TRAINED CHANNEL, not native phrase composition (future, named). The
before/after comparison rides the same rows, same stats, same locked
stimuli — the only change between E8-O and E8-O2's po block is the
readout's schooling. DB untouched.

## §4 Instruments

Design check + json committed with this registration. Build (after):
`e8o2_logic.py` = `e8o_logic.py` verbatim + E8-O2 block, single-source
into `E8O2_MIXED_UI.ipynb` (builder + logic/trainpath/verdict suites,
smoke ladder, zero rclone, recompute law on retrieval). Seeds:
E8O2_SEED = 20260910 stream. Locked inputs: E8-F payload + readout
(flight of record), E8-R real bundle (G2), the flown PAIR24.
