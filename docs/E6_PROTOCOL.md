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
