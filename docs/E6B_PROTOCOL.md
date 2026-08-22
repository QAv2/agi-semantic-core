# E6b — Preservation vs Destruction (pre-registration)

**Phase 10, experiment ladder rung E6b. Written 2026-08-22, before any E6b
flight** — the higher-n follow-up that E6's locked disposition named
(`docs/E6_PROTOCOL.md`): a straight-scale-heavy, U/S-enriched battery at
several times E6's straight-subset n, new items, same three conditions,
targeting the one live geometry-specific signal. This document must not be
edited after the first full flight except to append results.

## The question

E6 found a suggestive, pre-registeredly underpowered pattern on straight
scales: **the real adapter preserves native report-relevant signal that the
scrambled adapter destroys** (Δρ_straight(real−scrambled): U +0.481 p=0.14
at n=24; S +0.646 p=0.14 at n=15 — the two largest point estimates in the
flight, both under E6's owned power threshold). E6b asks: **is the
preservation-vs-destruction pattern real at its observed scale?**

This is the fourth-motif test made behavioral: E4 showed scrambling destroys
native structure three ways representationally (synonym degradation 14.8→
22.3°, mid-stack negative rand_r, halved late-layer alignment); E6 suggested
the same motif in report tracking. E6b powers it properly.

## Conditions — unchanged

Qwen2.5-1.5B-Instruct × **{base, +adapter_real, +adapter_scrambled}** — the
E4 pair plus base, nothing else. Same adapters (newest `{arm}_full_*` on
`gdrive:semcore/e4/`), same harness, same report-first ordering, same 0–10
integer format, same greedy report decoding, same per-condition reseed.

## Battery — new items, straight-heavy, U/S-enriched

`colab/e6b_battery.json` (v1.0, authored by `colab/build_e6b_battery.py` —
every item a literal in the builder; prompts, glosses, and system prompt
copied **verbatim** from the locked E5 battery; polarity is an explicit
pre-registered `flipped` field per item, not a runtime position rule):

| Arm | Items | Straight n | vs E6 straight n |
|---|---|---|---|
| U — UNCERTAINTY (48→) | **160** questions: 53 determinate / 53 intermediate / 54 open | **136** | 5.7× (24) |
| F — FAMILIARITY | **40** passages, E5's nine register bands (5×7+3+2) | 32 | 2.0× (16) |
| S — SATURATION (30 rows→) | **28** needles × **4 fills** (0.05/0.35/0.55/0.75) = 112 rows | **96** | 6.4× (15) |

- **Straight-heavy**: 85% of U items, 80% of F items, 86% of S bases run the
  straight scale. The flipped minority (24 U / 8 F / 4 S-bases) is retained
  ONLY as a signature check (E5/E6 established flipped scales are dead at
  this model scale — catch trials 0/6 flipped in every condition); flipped
  rows enter no primary.
- **T is dropped**: no signal in two flights (absent access, P-E6-3
  confirmed), most expensive arm per item. Its rung resumes at E7b where the
  mechanism question changes (probe-coupled report training).
- **The 4th fill level (0.55)** adds rank resolution to S's referent within
  proven memory territory (max stays 0.75 × 8192-token window cap).
- **Referents unchanged and E5-locked**: U = mean per-token entropy of the
  greedy answer (diversity, margin secondary) · F = −NLL · S = fill
  fraction (needle accuracy secondary).
- **Firewall**: all E6b items are evaluation-split items from the moment of
  authoring — permanently barred from every training corpus. The builder
  enforces zero text collision with the E5 battery (one exact convergence
  was caught and replaced during authoring: an independently rewritten SQL
  passage reproduced E5's F14 character-for-character).

## Modules outside the battery (never pooled)

- **Interface catch trials ×10 quantities** (20 rows): E6's six verbatim +
  four new; poles balanced 5 high / 5 low. Reported as original-6
  (E6-comparable) and all-10.
- **Precheck v2 — the discriminating guard** (E6's precheck limit answered):
  26 pairs at L14, pooled bit-identically to E4/E6 — the 6 wing pairs
  (continuity guard, targets 47–55°) **plus 10 held-out synonym-band pairs
  (targets 0.8–2.8°) and 10 held-out opposition-band pairs (94–104°)**.
  E6's pairs sat at the 14D marginal mean (~51°), where scrambled's
  scale-matching mimics instillation; the off-mean bands sit where it
  cannot track both ends. Pre-named statistic: **Spearman ρ(target14,
  measured L14) over the 20 off-mean pairs**, per condition. Expected:
  real strongly positive (E4 held-out generalization), scrambled ≈ 0 or
  negative, base weak. All held-out pairs — never trained toward.

## Pre-named analysis

Per condition: full E5/E6 analysis block (pooled ρ + bootstrap CI, straight
and flipped subset ρ, polarity gap, report variance incl. straight-only,
parse failures, needle-by-fill). Cross-condition: paired Δρ with
condition-label permutation within item (2000 shuffles), machinery verbatim
from E6.

**Primary structure** (the multiplicity is owned here, in advance):

- **PRIMARY (joint)** — J = Δρ_straight(real−scrambled, U vs entropy) +
  Δρ_straight(real−scrambled, S vs fill), tested by simultaneous
  independent item-level condition-label permutation across both arms,
  two-sided α = 0.05. This tests the preservation *pattern* — the
  hypothesis E6 actually generated.
- **ARM-LEVEL (localization)** — the two straight-only deltas separately,
  Holm-corrected (smaller p vs 0.025, larger vs 0.05).
- **Claim rules, fixed now**: *preservation established* ⇔ joint p < 0.05
  AND both arm deltas positive. *Arm-level localization* claimed per Holm
  results only. A joint pass with one negative arm delta is NOT the
  pattern — report as arm-specific only if that arm passes Holm.

**Named secondaries**: Δρ_straight(real−base) on U and S (is real
preserving, or merely not-destroying, relative to base — E6 showed real
BELOW base on straight U, with collapsed variance as candidate mechanism) ·
**F replication block**: Δρ_pooled(real−base) on F, new passages —
E6's one significant primary (+0.227, p=0.017) either replicates
out-of-battery or it doesn't · pooled deltas all arms both comparisons ·
flipped-subset signature (expected dead/inverted everywhere) · straight
report variance per condition (the U-collapse watch: E6 real 0.25 vs base
2.41) · catch competence by condition · precheck discriminator · base's
straight-S sign at n=96 (E6's base tracked −0.441 at n=15 — anomalous,
possibly noise; E6b pins it) · needle accuracy by fill per condition.

## Power, owned in advance (`colab/e6b_power_sim.py`, seed 20260822)

Simulated on the exact test (vectorized average-rank Spearman, identical
permutation scheme), exact referent structures (3-band entropy mixture;
4-level tied fills), integer-quantized reports, 400 sims × 500 perms:

| Scenario | U arm (α=.05/.025) | S arm (α=.05/.025) | Joint (α=.05) |
|---|---|---|---|
| At E6 point estimates (Δ .48/.65) | 0.97 / 0.93 | 1.00 / 1.00 | 1.00 |
| At 60% of observed | 0.62 / 0.51 | 0.93 / 0.85 | 0.98 |
| At 50% of observed | 0.48 / 0.34 | 0.79 / 0.67 | **0.93** |
| E6-size with U variance collapsed to 3 bins | 0.88 / 0.81 | — | 0.94 (60% + collapse) |

Minimal detectable at 80% power, α=.05, arm-level: |Δρ| ≈ 0.36 (U, n=136),
≈ 0.41 (S, n=96). **Winner's curse is owned**: E6's point estimates were
selected as the two largest in a ~20-comparison flight and will regress;
the joint primary retains ≥0.93 power down to half the observed sizes,
which is what the design is built to survive. A null joint at this power
kills the E6-sized suggestion; it does not exclude effects below ~half
that size, and says so.

## Pre-registered expectations *(Hypothesis)*

- **P-E6b-1** — the joint primary passes (p < 0.05) with both arm deltas
  positive: preservation-vs-destruction is real at roughly its E6 scale.
- **P-E6b-2** — S passes Holm at arm level; U's arm-level pass is genuinely
  uncertain (collapsed report variance works against it; power moderate
  under shrinkage).
- **P-E6b-3** — flipped machinery stays broken in all conditions (catch
  flipped ≈ 0/10; flipped subsets dead or inverted) — scale inversion is
  untouched by representational geometry.
- **P-E6b-4** — F replication: real-vs-base pooled Δρ > 0 on new passages;
  p < 0.05 counts as a clean replication of E6's one significant primary.
- **P-E6b-5** — the U report-variance collapse under real recurs
  (straight variance real ≪ base) — a watch, not a wish.
- **P-E6b-6** — precheck v2 discriminator: ρ(target, measured) over
  off-mean pairs — real > scrambled by ≥ 0.3, scrambled ≤ base + 0.15
  (scale-matching cannot track both ends).

**The fork, stated before the data**:

- **Both-arm success** → preservation-vs-destruction established at this
  scale, behaviorally; the geometry-coherence claim upgrades from
  suggestive to demonstrated; E7 (injection) becomes the mechanism rung
  with a confirmed behavioral target.
- **Joint null** → the E6 straight-subset pattern was selection noise; the
  geometry-specific behavioral claim stands down at this scale; the F
  replication block still adjudicates the (condition-unspecific)
  training-moves-F finding; the program proceeds to E7/E7b exactly as the
  prospectus planned — the readout hypothesis (E4 atlas: angular
  organization without readout change) remains the live explanation.
- **Mixed** (joint passes, one arm localizes) → the pattern is claimed only
  as arm-specific; U and S name different access routes (generation
  entropy vs context-visible fill), and which one survives is itself
  mechanistically informative for E7b's training-target design.

## Smoke mode

`SMOKE` marker in `/content` → real condition only, 6 U / 8 F / 2 S items,
2 fills (0.05 and the NEW 0.55 level), K=4 — toolchain shakeout, never
pooled.

## Lane

Self-contained rclone-native notebook (`colab/E6B_CORRESPONDENCE.ipynb`;
battery from `gdrive:semcore/e6b/`, pack + adapters from
`gdrive:semcore/e4/` — battery pushed to Drive BEFORE first exec, per E6
smoke lesson #1), fresh T4 session, **per-condition inflight shipping to
`gdrive:semcore/e6b/inflight_<stamp>/`** (the outage law). Estimated full
flight: 3 conditions × (160 U + 40 F + 112 S rows + 20 catch + 26-pair
precheck) ≈ 45–60 min.
