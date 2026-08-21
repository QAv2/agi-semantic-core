# E4 — Decoder Instillation: results record (both arms)

**Phase 10, experiment ladder rung E4. Flights: 2026-08-21.**
Pre-registration: the design cell of `colab/E4_INSTILL.ipynb` (locked at first
full flight). Design context: `docs/PHASE_10_PROSPECTUS.md` §5. This document
is the results record for the two-arm pair and, with E5's baseline, the input
to E6.

## The question

Does the E1 recipe — the dictionary's angular contract as a training target
plus a KL retention channel — port from the 22M encoder to a decoder-only
instruct model at LoRA rank? And is what gets learned the *geometry*, rather
than an artifact of fine-tuning exposure? The second question is the scrambled
arm's job: identical recipe, identical steps and data volume, train targets
replaced by a seeded derangement — same marginal target distribution, zero
pair-specific structure.

## Setup (both arms identical except targets)

- **Model**: Qwen2.5-1.5B-Instruct · LoRA r16 α32 on q/k/v/o
  (4.36M trainable, 0.28%) · T4, session `semcore`.
- **Geometry loss**: MSE between mean-pooled hidden-state cosines at layer 14
  and grounded 14D target cosines, on 4,213 train relations (80% split).
- **Retention**: chunked KL toward the same model with adapters disabled
  (teacher without a second copy), wikitext, λ = 1.0.
- **Schedule**: 800 steps, GEO_BS 24, RET 6×256, lr 1e-4 (~26 min/arm).
- **Evals**: 1,058 held-out relations (mean |rep angle − 14D target angle| by
  type) · 2,000 held-out random pairs (Pearson r between rep angles and 14D
  angles — pairs never trained in either arm) · wikitext perplexity · ridge
  probe R² (instrument failed in-flight; see below).
- Two smoke flights preceded the real arm (env pins; an eval-indexing fix);
  smoke data not pooled.

## Results

Flights `real_full_20260821_2144` and `scrambled_full_20260821_2221`
(artifacts in `colab/results_e4/`, adapters 17MB each; the scrambled dir's
duplicate `adapter_real` from the shared VM was deduped locally).

| metric | baseline | **real** post | **scrambled** post |
|---|---|---|---|
| held-out complement err (°) | 60.60 | **15.43** | 23.77 |
| held-out opposition err (°) | 74.20 | 31.03 | 25.83 |
| held-out adjacent err (°) | 50.77 | 15.60 | 17.21 |
| held-out affinity err (°) | 28.29 | 12.66 | 16.96 |
| held-out homonym err (°) | 41.55 | 18.54 | 16.67 |
| held-out isomorphic err (°) | 29.37 | 14.87 | 15.72 |
| held-out synonym err (°) | 14.80 | 13.44 | **22.27 (worse)** |
| held-out random-pair r | −0.015 | **+0.198** | −0.002 |
| perplexity | 10.658 | 10.663 (+0.05%) | 10.661 (+0.03%) |

Training dynamics: the real arm converges — clean back-half descent to geo
0.023 at step 800. The scrambled arm plateaus noisily ~2× higher (final
0.039, oscillating 0.03–0.12 through the back half): a partial fit with a
floor, consistent with the reading below.

## The reading — three findings

**1. The recipe ports.** Every held-out relation type improves under real
targets; correlation on 2,000 never-trained random pairs goes from zero to
+0.198; capability cost is +0.05% perplexity. The E1 result — geometry
trainable at zero semantic cost — holds in a decoder at LoRA rank.

**2. The dissociation lands, and it lives in the correlation metric.** The
scrambled arm's held-out |err| gains are a **scale artifact**, not learning:
with mutually incoherent targets, the optimum available to the network is to
match the *marginal distribution* of target angles — spread the base model's
narrow anisotropy cone (~30° typical pairwise) toward the target mean —
which mechanically shrinks |rep − target| against mid-range targets while
knowing nothing about which pair goes where. Two measurements expose it:

- **Random-pair correlation is immune to scale matching** (it rewards only
  pair-specific ordering): real **+0.198**, scrambled **−0.002** — dead
  exactly where the pre-registration looked.
- **The synonym reversal is the signature of indiscriminate spreading.**
  Synonyms are the one relation type base models already satisfy natively
  (similar meaning → nearby reps → small angles ≈ small targets; baseline
  err 14.8°, the best row in the table). Real training preserves that
  structure (→ 13.4°). Scrambled training *destroys* it (→ 22.3°): the
  spreading that fakes improvement on mid-range targets pushes genuinely
  close pairs apart. Scrambled learned the statistics of the answer sheet;
  real learned the geometry.

**3. Capability preservation does not discriminate the arms — geometry
does.** Both arms hold perplexity flat: the retention channel works
regardless of target coherence. So "the model still works" is no evidence
that anything real was instilled; the held-out correlation is.

## Pre-registered scorecard (real arm, criteria as written)

- Held-out complement + synonym err drop materially: **complement ✓**
  (60.6 → 15.43); synonym drops but was near floor at baseline
  (14.8 → 13.4) — *materially* applies to complement only. Pass, stated
  precisely.
- Random-pair r rises: **✓** (−0.015 → +0.198).
- Perplexity within +5%: **✓** (+0.05%).
- Probe R² at loss layer rises: **✗ instrument failure** — the in-flight
  ridge probe (α = 1.0, unstandardized fp16 reps) was ill-conditioned
  (LinAlgWarnings, rcond ~1e-8; negative R² at every layer in every
  condition, baseline included). Both readings are meaningless; neither
  criterion direction can be scored on a broken instrument. Repaired
  instrument = `colab/E4_PROBEFIX.ipynb` (StandardScaler + RidgeCV over an
  alpha grid, the E0 method) flying a three-condition per-layer depth atlas
  (base / real / scrambled); results appended below when landed.
- **Kill condition** (ppl > +10% or geometry does not move): not triggered.

An in-flight prediction, scored honestly: the working prediction while the
scrambled arm flew was "train loss falls (memorization) but held-out numbers
stay dead." Half right, and instructively wrong: scrambled train loss fell
only partway (the scale-matching floor), and held-out |err| *did* move — by
the same artifact — while the metric that stayed dead was the correlation.
Lesson carried forward: **|err|-style metrics conflate scale with structure;
correlation-style metrics are primary for every future geometry eval** (the
probe-fix atlas reports both).

## What this sets up

E6 now has exactly the comparison pair it needs: two adapters matched on
recipe, steps, data volume, and capability cost, differing only in whether
the instilled targets carried coherent geometry. If the real-arm model's
report–state correspondence (E5 battery, re-run) moves and the scrambled
arm's does not, the delta attributes to the geometry itself — not to
fine-tuning exposure, vocabulary familiarity, or anything the scrambled arm
shares. E5's baseline ≈ 0 is the floor both arms are measured against.

---

*Probe-fix depth-atlas appendix: pending flight `E4_PROBEFIX` — to be
appended on landing.*
