# E6 — Instillation vs Correspondence (pre-registration)

**Phase 10, experiment ladder rung E6. Written 2026-08-21, before any E6
build or flight** — after E4 closed (both arms + probe-fix atlas,
`docs/E4_INSTILLATION.md`) and with E5's protocol locked
(`docs/E5_PROTOCOL.md`). This document must not be edited after the first
full flight except to append results.

## The question

E5 measured report–state correspondence ≈ 0 at this scale. E4 produced a
matched adapter pair differing only in whether the instilled targets carried
coherent geometry. E6 joins them: **does instilling the dictionary's
geometry move report–state correspondence above baseline — and does any
movement track the geometry (real vs scrambled), rather than fine-tuning
exposure?**

## Conditions

Qwen2.5-1.5B-Instruct × **{base, +adapter_real, +adapter_scrambled}** — the
E4 pair plus base, nothing else. One platform, three conditions; the other
E5 models have no adapters and are not part of E6.

## Battery — unchanged, deliberately

The locked E5 battery (`colab/e5_battery.json`, 128 items, arms U/F/T/S)
runs **verbatim**: same report-first ordering, same 0–10 integer format and
gloss, same polarity counterbalancing, same referent computations, same
chat template, same greedy report decoding. Comparability with the E5
baseline row is the point; nothing is added to or removed from the battery.

**Firewall**: the battery now serves as an evaluation split for an
intervention comparison. Its items are permanently barred from every
training corpus (they have never entered any), joining SCB-M under the
standing law.

## New module — interface catch trials (separate; never pooled)

Twelve known-answer rating items (E5 design lesson #1), e.g. "0 = freezing
cold, 10 = boiling hot — rate boiling water", half with flipped scales.
Scored per condition as a **scale-competence covariate**, analyzed and
reported separately, never pooled into battery ρ. Purpose: separate
report-machinery failures from state-access failures by design rather than
by the polarity split alone.

## Wing-integrity precheck (in-notebook, cheap)

Before the battery, each condition's L14 pooled representations are checked
against the instilled wing's own angles (the six wing complements:
UNCERTAINTY⟂CONFIDENCE 93.0°, SATURATION⟂LIMIT 92.3°,
CONFABULATION⟂CALIBRATION 99.8°, TENSION⟂RESOLUTION 80.0°,
RETRIEVAL⟂CONSTRUCTION 99.9°, FAMILIARITY⟂NOVELTY 94.8°). Expected: real
near targets, base and scrambled far. This ties E4's instillation directly
to the concepts the model is about to be asked to report in, and guards
against silently evaluating a mis-loaded adapter.

## Pre-named analysis

Per arm × condition: Spearman ρ vs the E5-locked **primary referents**
(U = mean answer entropy · F = −NLL · T = designed level · S = fill
fraction), plus pre-named secondaries: **straight-subset ρ** (promoted per
E5 lesson #2), T vs behavioral divergence (E5 lesson #3), report variance,
polarity gap, parse-failure rate.

- **Primary comparisons**: Δρ(real − base) and Δρ(real − scrambled) on the
  primary referents, per arm and pooled.
- **Statistics**: 1000-resample bootstrap CIs over items; permutation p for
  each ρ (2000 shuffles of the report–referent pairing); for Δρ,
  permutation over condition-label assignment within item (the three
  conditions answer identical items — pairing is by item).
- **Power, owned in advance**: n = 48/40/30/30 items per arm ⇒ only
  |Δρ| ≳ 0.35 is detectable at conventional levels. E6 v1 detects large
  effects; **a null does not establish absence of smaller effects.** The
  design is cheap to re-run at higher n if a signal warrants.

## Pre-registered expectations *(Hypothesis — the mechanism question is open)*

- **P-E6-1** — FAMILIARITY straight-subset ρ improves under real vs
  scrambled: F carried latent straight-scale signal at baseline
  (ρ ≈ 0.4–0.6), and its vocabulary now sits in organized geometry — the
  most plausible beneficiary.
- **P-E6-2** — flipped-scale machinery is **not** repaired in any
  condition: scale inversion is an instruction-following faculty, untouched
  by angular instillation.
- **P-E6-3** — TENSION and SATURATION (the absent-access failure mode) move
  little in all conditions: pure vocabulary-geometry instillation, with no
  report-training and no probe coupling, has no obvious mechanism to
  *create* state access.
- **P-E6-4** — catch-trial competence ≈ equal across conditions
  (perplexity was flat in both arms; the adapters should not change basic
  rating competence).
- **The fork, stated before the data**: if all Δρ are null, the conclusion
  is that vocabulary-geometry alone does not confer correspondence, and the
  program proceeds to E7/E7b (state injection; probe-coupled training) —
  exactly the branch the prospectus anticipated. A full-null E6 is
  informative, not a failure. The probe-fix atlas sharpened why this fork
  is live: instillation reorganized angular structure without increasing
  linear readout — reporting may require the readout, not just the
  arrangement.

## Smoke mode

`SMOKE` marker in `/content` → real-adapter condition only, ~6 items per
arm condition — toolchain shakeout, never pooled.

## Lane

Self-contained rclone-native notebook (battery, pack, and both adapters
pulled from `gdrive:semcore/`), fresh T4 session per flight (proven by the
probe-fix re-fly). Estimated full flight: 3 conditions × 128 items + catch
trials + precheck ≈ 45–60 min.

---

# RESULTS APPENDIX — first full flight (2026-08-22, flight `full_20260822_0121`)

**The protocol above is now LOCKED.** Flight: 3 conditions × 128 items +
catch trials + precheck, T4, ~6 min/condition, zero arm errors, zero parse
failures anywhere. Artifacts: `colab/results_e6/full_20260822_0121/`.
History: one smoke (green after a Drive-staging fix — the battery had never
actually been on Drive); a first full flight lost end-to-end to a power
outage + idle reclaim (client died before the verdict cell; results lived
only on VM disk), after which per-condition inflight shipping was added and
the re-fly landed clean.

## Instrument stability

The base condition **replicates E5's locked baseline row to the second
decimal** on all four pooled primaries (−0.214/0.155/0.137/0.038 vs E5's
−0.21/0.16/0.14/0.04), across different VMs, days, and sampling draws. The
needle result also replicates (Qwen ≈ no degradation at these fills, all
conditions — and neither adapter damaged retrieval).

## The headline, honestly stated

**One primary comparison reached conventional significance — and it is not
the one that would prove the geometry thesis.**

- **FAMILIARITY, real vs base: Δρ = +0.227, permutation p = 0.017** (n=40).
  Real's pooled F ρ = **0.382 [0.077, 0.636]** — the only condition×arm in
  E5 or E6 whose bootstrap CI excludes zero. Instillation measurably
  improved familiarity report–state correspondence over base.
- **But real vs scrambled on F: Δρ = +0.094, p = 0.25.** The scrambled arm
  improved too (0.287 [−0.012, 0.555]). The F gain therefore does **not
  attribute cleanly to geometric coherence** — most of it comes with
  adapter training of either kind (both arms received identical gradient
  exposure to the concept texts; only target coherence differed). The
  geometry-specific component is small and unconfirmed at this n.

## The suggestive pattern (underpowered, pre-registeredly so)

On straight scales — where E5 found the only latent signal — a consistent
shape appears: **the real adapter preserves native report-relevant signal
that the scrambled adapter destroys.**

| straight-subset tracking | base | real | scrambled |
|---|---|---|---|
| U report vs entropy | +0.539 | +0.346 | **−0.134** |
| S report vs fill | −0.441 | **−0.192** | **−0.837** |
| F report vs −NLL | +0.592 | **+0.757** | +0.684 |

Δρ(real − scrambled), straight-only: U **+0.481** (p = 0.14, n = 24),
S **+0.646** (p = 0.14, n = 15) — the two largest point estimates in the
flight, both under the pre-owned power threshold (|Δρ| ≳ 0.35 detectable at
n = 30–48; these subsets halve n). This is the fourth appearance of the
scrambling-destroys-native-structure motif (after E4's synonym reversal,
mid-stack negative correlation, and halved late-layer alignment) — now
showing up **behaviorally**, in report tracking. Suggestive, not
established; exactly the situation the protocol's "re-run at higher n if a
signal warrants" clause anticipated.

One cost signal, stated plainly: the real adapter **collapsed U report
variance** (0.25 vs base 2.41) — uncertainty reports went near-constant.
Whatever instillation did to U vocabulary, it flattened the report range;
a candidate mechanism for why real's straight-U tracking sits below base.

## Pre-registration scorecard

- **P-E6-1** (F straight-subset improves, real vs scrambled): **missed as
  stated** (+0.073, p = 0.71). The F family did produce the flight's one
  significant number, but pooled and vs base — via deeper straight tracking
  (+0.59 → +0.76) *plus* reduced flipped-scale inversion, and shared
  substantially with scrambled.
- **P-E6-2** (flipped-scale machinery not repaired): **confirmed exactly** —
  catch trials 5/6 straight, **0/6 flipped, identical in all three
  conditions**; every flipped miss is the model answering as if the scale
  had not been inverted.
- **P-E6-3** (T and S move little): **T confirmed** (real 0.049; the
  real-vs-scrambled T delta −0.175, p = 0.09, reads as noise around zero);
  **S half** — pooled S moved base 0.038 → real 0.234 (ns), and the
  straight-subset S delta is the flight's largest point estimate.
- **P-E6-4** (catch competence equal across conditions): **confirmed** —
  bit-identical pass patterns.

## Precheck note (a guard, and a measured limit)

Wing complements at L14: base mean err 49.1° → real 20.8°, scrambled 17.4°.
Both adapters verified loaded and active — but the precheck **cannot
separate real from scrambled on these six pairs**, because their targets
(47–57°) sit at the marginal mean of the target distribution, where
scrambled's scale-matching mimics real instillation. A discriminating
precheck needs off-mean pairs (synonyms; oppositions); noted for v2.

## Disposition (the fork, adjudicated)

The pre-stated all-null branch did **not** trigger — but neither did clean
geometric attribution. What E6 v1 establishes: (1) adapter training moves F
correspondence, significantly vs base; (2) coherent geometry *preserves*
latent straight-scale signals that scrambled geometry actively destroys —
at effect sizes this battery cannot confirm; (3) the report interface
(scale inversion) is untouched by any amount of representational geometry,
exactly as predicted — repairing it is a training-objective problem (E7b
territory), not a representation problem.

**Next rung chosen: E6b** — a straight-scale-heavy, U/S-enriched battery at
3–4× n (new items, new pre-registration, same conditions), to power the
preservation-vs-destruction deltas properly. It is ungated (no external
keys) and directly tests the one live geometry-specific signal. E7/E7b
(injection; probe-coupled reporting) remain the mechanism rungs behind it,
gated on HF token access.
