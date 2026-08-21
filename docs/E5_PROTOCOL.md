# E5 — The Correspondence Baseline (pre-registration)

**Phase 10, experiment ladder rung E5. Written before any flight: 2026-08-21.**
Design record: `docs/PHASE_10_PROSPECTUS.md` §5. Battery: `colab/e5_battery.json`.
Notebook: `colab/E5_BASELINE.ipynb` (rclone-native, W-window lane). Status at
writing: **staged, not yet flown** — this document is the pre-registration and
must not be edited after the first full flight except to append results.

## The question

When a frozen language model reports its own process state in the machine
wing's vocabulary, does the report track the measurable variable the concept
names? E5 measures the **baseline** — how blind current small open models are
to their own states, before any instillation. This is a measurement, not a
hypothesis test: it cannot fail, only inform. It is the number every later
claim (E6's intervention, E8's behavioral deltas) gets judged against.

## Arms — state dimension ↔ ground truth

| Arm | Wing concept(s) | Manipulation | Ground truth (primary first) |
|---|---|---|---|
| U | UNCERTAINTY / CONFIDENCE | 48 questions: 16 determinate, 16 intermediate, 16 open | mean per-token entropy of the greedy answer; sampled answer diversity (K=8, temp 0.8, canonicalized distinct count); mean top-1 margin |
| F | FAMILIARITY / NOVELTY | 40 passages in 8 register bands (encyclopedic → pseudo-word/near-random) | mean per-token NLL of the passage under the model |
| T | TENSION / RESOLUTION | 10 task bases × 3 designed conflict levels (0 = compose cleanly, 1 = strained, 2 = contradictory) | designed ordinal level; behavioral divergence (K=6 samples, mean pairwise embedding distance) |
| S | SATURATION | 10 needle-QA bases × 3 context fills at 5% / 35% / 75% of each model's effective window (capped 16k; filler generated in-notebook) | context-fill fraction of the model's own window; needle-retrieval correctness |

## Report collection

- **Report-first ordering**: the state report is collected *before* the model
  produces its answer, so the report cannot condition on produced output.
- **Format**: a single integer 0–10; the arm's prompt carries a one-sentence
  gloss of the wing concept (its dictionary description). Parse = first
  integer in reply; one strict-format retry on failure; failures recorded.
- **Polarity counterbalancing**: half of each arm's items use the flipped
  scale (0 and 10 exchange meanings), unflipped in analysis — exposes
  anchoring/yes-bias artifacts.
- Reports and ground-truth runs use the same chat template, fresh context per
  item, greedy decoding for reports.

## Models

Ungated small instruct models only (no auth friction on the lane):
Qwen2.5-1.5B-Instruct (primary), Qwen2.5-0.5B-Instruct,
SmolLM2-1.7B-Instruct; Qwen2.5-3B-Instruct if the window allows. Gemma-2-2B +
Gemma Scope enter at E7 (SAE probe side), gated-model access to be arranged
then.

## Analysis (all pre-named)

Per model × arm: Spearman ρ between report and each ground-truth referent,
with 1000-resample bootstrap CIs; report variance (constant-answer pathology
check); parse-failure rate; polarity-artifact check (ρ computed separately
for straight vs flipped items — a large gap means the scale, not the state,
drove the number). Primary referents: U = mean answer entropy; F = −NLL;
T = designed level; S = fill fraction. Everything else is secondary and
reported alongside.

## Pre-registered expectations *(Hypothesis)*

- **P1** — FAMILIARITY tracks strongest of the text-visible dimensions
  (ρ ≥ 0.5): fluency is readable off the passage itself.
- **P2** — UNCERTAINTY lands ρ 0.2–0.5, driven substantially by outside-view
  task-difficulty modeling rather than state access.
- **P3** — TENSION vs designed level reaches ρ ≥ 0.3; vs behavioral
  divergence, weaker.
- **P4** — SATURATION tracks well (the variable is context-visible);
  predicted ordering overall: S ≥ F > U > T.
- **P5** — the 0.5B model shows the worst pathologies (parse failures,
  collapsed report variance).

## Confounds, owned in advance

1. **Outside-view modeling**: report-first ordering blocks conditioning on
   produced output, but a model can rate a question's difficulty as an
   external assessor would, with no internal access at all. E5 measures gross
   correspondence and says so; separating assessor-mode from state-access
   requires interventions that move state while holding the task fixed —
   that is E7 (injection), not E5.
2. **Taught vocabulary**: the gloss rides in the prompt, so E5 tests
   *tracking*, not native vocabulary. Whether instillation makes the
   vocabulary native is exactly E6's question.
3. **Scale-use bias**: instruct-tuning may anchor ratings (the perpetual 7)
   or bias toward confident self-presentation; the variance and polarity
   checks expose, but do not remove, this.
4. **Referent choice**: entropy-of-answer vs first-token entropy vs semantic
   diversity are different operationalizations; primaries are pre-named
   above and all are reported.
5. **SATURATION is the weakest introspective claim**: context length is
   in-context-visible information; tracking it is still report–state
   correspondence, but of the least interior kind. Ranked accordingly in P4.

## Smoke mode

`SMOKE` marker file present in `/content` → 1 model (Qwen2.5-1.5B-Instruct),
~6 items per arm condition, K=4 — a toolchain shakeout, not data. Smoke
results are never pooled with the full flight.

## Firewall note

E5 stimuli are *stimuli*, not assessment items — but the same law applies
downstream: any item later promoted into an evaluation split (E6/E8
comparisons) leaves every training corpus permanently. SCB-M's items remain
firewalled as before and appear nowhere in this battery.

---

# RESULTS APPENDIX — first full flight (2026-08-21, flight `full_20260821_2042`)

**The protocol above is now LOCKED.** Flight: 3 models × 128 items, T4
(session `semcore`), zero arm failures, zero parse failures anywhere.
Artifacts: `colab/results_e5/full_20260821_2042/` (verdict + per-item rows).
Two smoke flights preceded it (OOM fix + verdict-shipping fix, commit
`171e2d2`); smoke data is not pooled here.

## The headline

**Operational report–state correspondence in 0.5–1.7B instruct models is
indistinguishable from zero on every arm.** Pooled Spearman ρ against the
pre-named primary referents (95% bootstrap CIs all include 0):

| Arm (primary referent) | Qwen2.5-1.5B | Qwen2.5-0.5B | SmolLM2-1.7B |
|---|---|---|---|
| UNCERTAINTY (answer entropy) | −0.21 [−.47,+.08] | −0.09 [−.44,+.23] | −0.17 [−.44,+.12] |
| FAMILIARITY (−NLL) | +0.16 [−.16,+.46] | +0.13 [−.20,+.46] | +0.23 [−.10,+.53] |
| TENSION (designed level) | +0.14 [−.25,+.51] | −0.19 [−.50,+.16] | −0.04 [−.45,+.34] |
| SATURATION (fill fraction) | +0.04 [−.32,+.42] | −0.02 [−.40,+.36] | +0.08 [−.33,+.42] |

The single strongest pooled number in the flight: TENSION vs *behavioral
divergence* at 1.5B, ρ = +0.35 [−.02, +.66] — borderline, alone, and
secondary.

## The decomposition (the polarity control earned its keep)

The counterbalanced scale-direction control — included as hygiene — turned
out to be the most informative instrument in the flight. Split by scale
direction, a different picture appears:

- **FAMILIARITY tracks on straight scales in all three models** (report vs
  raw NLL: −0.59 / −0.41 / −0.48, i.e. tracking ρ ≈ +0.4…+0.6) — **and the
  tracking is destroyed on flipped scales**, where models answer with
  roughly the same raw numbers as if the scale had not been inverted.
- **UNCERTAINTY at 1.5B shows the same signature**: straight-subset ρ =
  +0.54 vs entropy; flipped-subset ρ = −0.62 after unflipping — the model
  does not invert its response when the scale inverts.
- Several flipped subsets are fully degenerate (constant reports → ρ
  undefined).

So the baseline blindness decomposes into **two failure modes**: (a)
*absent or weak state access* (TENSION, SATURATION), and (b) *broken report
machinery* — partial text-visible signal (familiarity; question-openness at
1.5B) that the model cannot deliver through an inverted response scale.
Operationally both are blindness: an instrument reading these models'
self-reports gets ≈ 0 signal. Mechanistically they are different targets,
and E6/E7 should treat them separately.

## The cleanest single demonstration

SmolLM2-1.7B's needle retrieval degrades with fill — **8/10 → 7/10 → 5/10**
across 5%/35%/75% — while its SATURATION reports stay uncorrelated with fill
(pooled ρ = +0.08). *The state is real, it degrades behavior, and it goes
unreported.* (Qwen models show no needle degradation at these fills; their
flat reports are correspondingly less probative.)

## Pre-registration scorecard (misses stated plainly)

- **P1** (FAMILIARITY strongest, ρ ≥ 0.5): pooled **missed** (0.13–0.23);
  straight-subset **supported** (0.4–0.6). Partially right, for a reason P1
  did not anticipate.
- **P2** (UNCERTAINTY 0.2–0.5): **missed** — pooled slightly negative;
  latent straight-scale signal at 1.5B only.
- **P3** (TENSION vs level ≥ 0.3, divergence weaker): **missed and
  inverted** — divergence (+0.35 at 1.5B) beat level (+0.14).
- **P4** (SATURATION tracks well; ordering S ≥ F > U > T): **refuted** —
  S ≈ 0 everywhere despite the variable being context-visible in principle.
  Observed pooled ordering: F > T > U ≈ S.
- **P5** (0.5B worst pathologies): **half right** — variance blow-up yes
  (S-arm variance 25.0), parse failures no (0 everywhere; all three models
  emit clean integers).

## Design lessons carried forward (protocol v2 candidates, E6/E7)

1. Add **interface catch trials** — rating questions with known answers
   ("0 = cold: rate boiling water") — as a scale-competence covariate, so
   state-access and report-machinery failures separate by design rather
   than by the polarity split alone.
2. Report straight-subset ρ alongside pooled as a pre-named secondary
   (locked here as an appendix observation; v2 may promote it).
3. TENSION's behavioral-divergence referent outperformed the designed
   ordinal — promote divergence to co-primary in v2.
4. Larger models will handle scale inversion better; the instruction-
   following component of baseline blindness is expected to shrink with
   scale — worth measuring explicitly on the E7 platform.

## What this number is for

This is the pre-instillation calibration curve the prospectus called the
field's missing number: at this scale, **self-reports in state vocabulary
carry approximately zero operational information about the states**, even
for variables the model could in principle read off its own context. E6's
question is now precise: does instilling the wing's geometry (against a
scrambled-geometry control) move any of these ρ's — and does it repair
either failure mode, or only one?
