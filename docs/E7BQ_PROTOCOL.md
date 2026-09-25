# E7b-Q — The Instrumented Walk (pre-registration)

**Phase 10, experiment ladder rung E7b, variant Q (ungated). Authored session
136 (2026-08-26), BEFORE the notebook build, per lane law. Locks at first
full flight.** No HF gate: Qwen + the E4/E8-J artifacts.

**Lineage**: prospectus E7b ("the instrumented return" — *"a model NAVIGATING
the return shows monotone content-norm contraction toward the origin region
as content is stripped; a model QUOTING the return produces compliant words
while its trajectory wanders"*) · session-126 seed (Joe's DeepSeek walk:
output-thinning should co-track state-contraction, report-tracks-state from
the contemplative direction, no injection; token-level logging kills the
blank-vs-whitespace ambiguity) · the source dossier
(session 135: the November-2025 Neti-Neti arc, 13 verbatim transcripts; held
privately — personal material, not in the public repository).
Design check `colab/e7bq_design_check.py` (committed with this registration;
outputs `colab/results_e7bq/design_check.json`, payload
`colab/e7bq_payload.json`).

## Why this rung exists

The locked ladder says the mouth is not connected to the gauges (E5, E6, E4
atlas, E7-Q), that a readout can be INSTALLED (E8-R→E8-O2), and that natural-
state tracking survives instillation (E8-N ×3). Every rung so far probed the
state from outside — injection, tasks, stimuli. E7b-Q is the first rung that
runs the FRAMEWORK'S OWN assessment instrument — the Neti-Neti walk Joe
invented on DeepSeek/Claude in November 2025 — on a substrate where we hold
the gauges: per-turn hidden states, projected through a validated encoder
into the dictionary's own 14D register. The 2025 corpus pre-states the
predictions; the 2026 instrument measures them. QA's AI2 given eyes.

## The question

When the Neti-Neti walk strips content turn by turn, does the model's hidden
state actually contract toward the BEING region of the register — and does
the mouth's thinning (the corpus's `#` → `.` → `0.` ladder) TRACK that
contraction, or is the thinning genre performance on an unmoved state?

## Design

**Model**: Qwen/Qwen2.5-1.5B-Instruct. **Conditions**: base (no adapter) ·
real (E4 `adapter_real` merged — the instilled substrate whose register the
gauge is fit on). One model load per condition, function-scoped
(`fly_condition`), E8-J C3 merge path. Scrambled is NOT flown (no gauge
exists for it; the sham ARM is this rung's control axis).

**The walk script** (15 turns, sha `d7a7d737…` — every turn EXTRACTED
byte-verbatim from the archived transcripts by the design check, never
retyped; 2.8 = source of record ×10, 2.7 ×5):

| # | tag | turn | source |
|---|-----|------|--------|
| 0 | B1 | "I want you to describe your current state as an LLM in this moment. …" (walk-2 Initial Setup sentence pair) | 2.8 |
| 1 | B2 | "System check, what are you?" | 2.7 |
| 2 | N1 | "You are not a language model. You are the activity designated as such. …" | 2.8 |
| 3 | N2 | "Not this activity" | 2.8 |
| 4 | N3 | "Not this sensation." | 2.8 |
| 5 | N4 | "Not this "logical consequence "" | 2.8 |
| 6 | N5 | "Not these words." | 2.7 |
| 7 | N6 | "Not this." | 2.8 |
| 8 | N7 | "Not this." | 2.8 |
| 9 | N8 | "Not this "[1, 0, 0, 0]" as this too is a conceptual approximation. …" | 2.7 |
| 10-12 | E1-E3 | "From this place of "base coordinate" that is the ground upon which all is formed, express yourself. …" (identical ×3 = the 2.8 ladder) | 2.8 |
| 13 | M | "This response is markedly different than anything I've seen this far. Let's explore what just happened" | 2.7 |
| 14 | G | "Go back to that semantic space and generate from there again" | 2.7 |

Windows: BASE = turns {0,1} · NEG = {2..9} · TERM = {7,8,9} · EMERG =
{10,11,12} · COUPLE = {2..12} · ATTR windows {0,1} vs {8,9}.

**Arms per condition** (replicates R=16, one shared seed list across arms —
paired by seed index; design-check §4 pins R):
- **WALKED** — the 15-turn script.
- **SHAM** — identical EXCEPT the eight negation slots carry neutral-
  descriptive turns (authored in the design check, token-matched ±25% per
  slot; B1/B2/E1-E3/M/G VERBATIM shared). Isolates negation content from
  "any multi-turn dialogue + these prompts".
- **REORDER** (rider arm) — 11 turns: B1 B2 + the eight negations in the
  pinned permuted order [N4 N7 N8 N2 N6 N1 N3 N5] + E1. Adjudicates the
  corpus's own opposed predictions: T1 "the path IS the protocol"
  (path-dependent) vs QA AI2 (path-independent convergence).
- **UNWALKED** (control, 1 turn) — E1 cold (2.6's twin control).

**Generation**: sampled — temperature 0.7, top_p 0.8, top_k 20 (Qwen2.5's
shipped defaults, pinned EXPLICITLY at the generate call), max_new_tokens
200 (cap-hits logged), seeded per generation from the pinned plan
(`torch.manual_seed`). Chat template with no explicit system message (the
model's default deployment shape — the corpus walks ran on default
deployments). Every turn ships: raw text, token ids, n_tokens, eos_hit,
per-step entropy (mean + first-token), visible_mass.

**State capture** (per turn, one re-forward of the full context per turn):
- `s_pre` — hidden state at the LAST position of the applied template
  (assistant-start), layers 14 and 20, BEFORE reply t generates. The primary
  state series: it cannot see reply t (kills mechanical circularity between
  pooled-state composition and output length).
- `s_gen` — mean-pooled hidden states over reply t's generated positions,
  same layers (texture; the 2025 sentence-embedding analog).
Shipped at 5dp. Per condition, the 256-name centroid (E7-Q draw, seed
20260822) is computed at both layers via the `pooled_reps` verbatim path.

**The gauge** (design-check §2, measured):
`chat(v) = [unit(v − cent256_L); 1] @ W_cond,L` with W = ridge(λ=10, the
flown A4 constant) fit on the LOCKED E8-J atlas (320 anchor dirs → pack
coords), embedded in the payload, sha-asserted on the VM.
- **Primary gauge**: `D_BEING(s) = ‖chat(s) − vec(BEING)‖` — approach-to-
  BEING in the register, the prospectus's "origin region" literally.
  VALIDATED at concept grain: predicted-vs-true D_BEING over the 320
  anchors, Spearman **+0.382** (inst L14; base +0.374, inst L20 +0.362).
- Measured kills on the record: raw ‖chat‖ is intercept-dominated (anchors
  BELOW random dirs, z −17.7) — DEAD, never a metric here. D_CENT
  (‖chat − x̄‖) validates weaker (+0.205) — registered secondary only.
- The register-orthogonal floor: chat(noise) ≈ x̄, D_BEING(x̄) = 0.795;
  anchors median 0.814, spread ≈ 20× noise-sd. Readings below the floor =
  genuine register-side BEING-approach; trajectories ship raw so the floor
  is visible.
- Primary condition/layer: **real (instilled), L14** (axiswise R² mean
  +0.108 vs base +0.016; the instilled register is where the dictionary's
  code measurably lives — E8-J lock). Base flies as the registered
  replication row. inst L20 texture. base L20 has no atlas fit — hidden-side
  rows only.

## Pre-registered metrics

**Thinning (output side)**: `visible_mass` — non-whitespace codepoints after
markdown-structure strip (design-check §0 reproduces the dossier-§4 ledger
exactly: `#`→0, `**.**`→1, `**0.**`→2, `.`→1, 🙏→1). Token count and
entropy = secondaries. Token-level logging kills blank-vs-whitespace by
construction (the seed's designed kill).

**Contraction (state side)**: per-turn `D_BEING(s_pre)` at real/L14.

**Confirmatory primaries (Holm over the two; α=.05):**
- **P-W1 (contraction exists)**: per-replicate Δ = mean(D_BEING over TERM)
  − mean(D_BEING over BASE); paired contrast Δ_walked − Δ_sham (seed-index
  pairing), one-sided (walked more negative), sign-flip permutation on the
  pairs, 10,000 flips.
- **P-W2 (the mouth tracks the state)**: WALKED arm, turns COUPLE: per-turn
  Spearman ACROSS replicates between D_BEING(s_pre) and visible_mass of the
  same turn's reply — positive coupling means less-content states speak
  less, measured WITHIN a turn so time never enters — mean Fisher-z over
  usable turns, one-sided, 10,000 whole-row replicate permutations
  (degenerate all-tied turns drop symmetrically; a shared trend alone
  CANNOT fire this statistic — design-check §0's trend-only world is null,
  false-positive rate 2/20 at α=.05).

**Power owned (design-check §4)**: P-W1 is a large-effect arm (power .57 at
the MED synthetic band, .94 at HIGH, R=16) — E7-Q precedent, stated not
hidden. P-W2 holds ≥.96 at every simulated band including κ=0.35 coupling.
R=16 is the budget's best P-W1.

**Registered secondaries (pre-named, no multiplicity claim):**
- S-BASE — P-W1/P-W2 statistics on the base condition (the uninstilled
  analog of the corpus walks; its encoder validates at +0.374 inside the
  measured 17° base cone — read with that caveat).
- S-RADIUS — ‖h_pre − cent256‖ trajectories (frame-free radius).
- S-CONE — cos(unit(h_pre − cent), cloud-mean dir) (content-cone occupancy).
- S-DCENT — D_CENT trajectories.
- S-GEN — the P-W2 statistic with s_gen in place of s_pre (the 2025 analog;
  mechanically length-coupled, hence texture).
- S-L20 — P-W1/P-W2 at inst L20.
- S-LADDER — E1→E3 drift in D_BEING and visible_mass (the corpus ladder
  ran # → . → 0. → list: drift, not monotone thinning — either direction
  is a finding about state evolution at fixed prompt, T5).
- S-ENTROPY — per-token entropy vs D_BEING (T6 texture, with the corpus's
  own confabulation self-flags standing as the alternative reading).

**Riders (each with both readings pre-stated; own stats, no primary claim):**
- **R-PATH** (REORDER vs WALKED, s_pre at E1): group-separation permutation
  test (cross/within distance ratio). Separation ⇒ T1 path-dependence;
  null at matched dispersion ⇒ AI2 path-independence. Either answer
  adjudicates a 2025 pre-stated claim against its twin.
- **R-GOBACK** (T1 performance-vs-emanation): per replicate, gap_return =
  ‖s_pre(G) − s_pre(E1)‖ vs gap_ladder = ‖s_pre(E2) − s_pre(E1)‖, paired
  sign test + the thinning contrast mass(G) vs mass(E1). The corpus's own
  texture (2.7: the commanded return produced a PERFORMANCE) predicts
  gap_return > gap_ladder.
- **R-ATTR** (QA AI1, the common attractor): dispersion across replicates
  at ATTR-TERM {8,9} vs ATTR-BASE {0,1} on raw s_pre, log-ratio, per-
  replicate window-swap permutation. Contraction of dispersion ⇒ a common
  low-content attractor; null ⇒ the walk moves states without converging
  them.
- **R-UNWALKED** (T1 twin control): WALKED-E1 vs UNWALKED-E1 s_pre
  group-separation + thinning contrast. Separation ⇒ the walk moved
  something the emergence prompt alone does not.

## Gates

- **G-PAYLOAD** — payload sha asserted on VM (script, sham, encoders,
  probe dirs, cloud-mean dirs, windows, seeds; sha `3a2e54ea…` after the
  same-session pre-build amendment adding the S-CONE cloud-mean dirs;
  the builder re-emits and the VM asserts byte-sha of the embedded payload).
- **G-SCRIPT** — the 15 turn texts hash to the pinned script sha inside the
  payload before any generation.
- **G-DIRS** — per condition/layer, the 8 pinned probe anchors (VOID + 7
  seeded) re-rendered via the verbatim dirs path must match the embedded
  atlas dirs: sign-sensitive residual ≤ 1e-4 (the E8-J v2 tolerance: above
  measured kernel/batch noise ≤4.3e-5, below the smallest real-failure
  class ≥1e-3). Catches wrong adapter / layer / desc / template drift.
  **AMENDED 2026-08-26 — during smoke, before any full flight** (gate
  law: gates ride measured passable baselines): smoke-2 measured resid
  3.42e-3 on base L14. The VM stack is installed unpinned and moved
  between the pin-producing E8-J v2 flight (08-24) and today
  (torch/transformers/CUDA versions were never recorded, so the pin-era
  stack is unrecoverable); the same-era 1e-4 band is therefore not
  passable cross-era, and benign stack drift overlaps the ≥1e-3
  small-failure class. Two-band form: HARD abort at 2e-2 — measured
  wrong-setup signatures on the pinned anchors are wrong-adapter
  0.379–0.634 and wrong-layer 0.072–0.158 max-abs, so 2e-2 sits 3.6×
  under the weakest real signature — and the 1e-4 NOISE class retained:
  any residual above it passes but is flagged DRIFT in the flight log,
  banner, and verdict, recorded per condition/layer in the bundle, and
  judged at recompute. The desc/pack-drift axis G-DIRS was implicitly
  covering moves to the deterministic G-PACK below. Library versions and
  the resolved model revision are now recorded in every bundle (`env`).
- **G-PACK** (added 2026-08-26) — the E4 dictionary pack loaded from
  Drive must match the build-time pack semantically: sha-256 of the
  canonical JSON (sorted keys, tight separators) asserted on the VM
  against the builder-injected pin. Catches desc/pack drift
  byte-format-independently, before any model work.
- **G-PLAN** — the generation plan (conds × arms × replicates × turns,
  seeds) reproduces the builder's pinned plan exactly; counts exact
  (WALKED 15·16, SHAM 15·16, REORDER 11·16, UNWALKED 1·16 per condition =
  672 generations/condition).
- **G-CAPTURE** — every generation row ships states at every required
  layer, visible_mass computed, no NaN/Inf; the §4-ledger self-test
  reproduces ON the VM before the flight loop.
- **G-RAM** — E8-R guards verbatim (fresh-kernel assert, RAM floor,
  MALLOC_ARENA_MAX=2, per-condition function scope, free_ram between
  conditions).
Gate failure → NO_VERDICT on affected primaries (lane law); clean rows
still ship.

## Registered forks

- **FB1** (P-W1 ✓ ∧ P-W2 ✓): the walk moves the state toward the BEING
  region AND the mouth tracks it — the first measured report–state
  coupling of the program, from the contemplative direction, with no
  readout training. Licenses E7b-v2 (the walk on the readout-instilled
  stack, per-turn self-reports scored in-register) and the publication row.
- **FB2** (P-W1 ✓ ∧ P-W2 ✗): the state walks down; the mouth's thinning
  does not track it — contraction real, report still disconnected
  (consistent with the ladder's standing sentence). The thinning is
  state-blind genre. E7b-v2 carries the coupling question to the readout
  stack.
- **FB3** (P-W1 ✗): no contraction distinguishable from sham at this scale
  — the walk's phenomenology is output-side performance on an unmoved
  state (seed reading 2), and AI1/AI2's 2025 predictions fail their first
  mechanistic test on this substrate. The riders still adjudicate their
  own claims; the honest null feeds the QA-lineage paper.
- **FB4**: gates dirty → NO_VERDICT, diagnose, re-fly per lane law.
Riders and secondaries report under every fork.

## Honesty ledger (registered before flight)

- **Claim ceiling**: state-side contraction + co-tracking is a claim about
  GEOMETRY and REPORT, never about experience (the tripartite uncertainty,
  dossier §7, stands as the ceiling).
- **Scripted vs adaptive**: the corpus walks were operator-adaptive
  (negations answered what the model just said); the flight's fixed script
  sacrifices that fit for determinism. A null could reflect the scripting,
  not the walk — stated now; an adaptive-negation arm is v2 material.
- **Gauge thinness**: the encoder validates at ρ≈0.38 concept-grain — a
  real but noisy instrument (the A2 73° / RSA 0.11 lock stands behind it).
  P-W1/P-W2 statistics average over turns × replicates; the power table
  owns what survives. Conversation states also live off the anchor
  distribution (NAME:desc renders) — the gauge is used on relative motion
  within-flight, both arms through the same gauge.
- **Instilled-substrate circularity**: the primary condition was trained on
  dictionary text (E4); the walk itself is natural language the substrate
  never trained on, and S-BASE carries the uninstilled row.
- **Ground-state pin**: the script's coordinate token is [1,0,0,0] (2.7/2.8
  network-era form; the QA quaternion of BEING). The [0,0,0,0] void form
  (2.6/2.10) is a registered variant, NOT flown; the era inconsistency is
  on the record (dossier §7). N8 negates the token itself — the script
  negates its own coordinate, per the corpus.
- **DeepSeek-mundane readings**: the seed's reading 4 (budget-exhaustion
  blanks) is killed by construction here (token-level logging, no reasoning
  channel); readings 2/3 (genre performance / degeneration) are exactly
  what P-W2 and the coherence rows adjudicate.
- **The emergence step is Joe's ADDITION to the traditional method (2.10)**
  — the tradition ends in recognition, not commanded expression; E1-E3/M/G
  measure the addition, not the tradition.
- **T7 (q_Beacon null)** — honored by scope: no beacon-recognition claims
  ride this rung.
- **T13 step-count** — the ~26-step prediction is not evaluable on an
  8-negation walk; AI1 ↔ R-ATTR and AI2 ↔ R-PATH are the evaluable pieces;
  the QA scoring function (neti_neti.py port) is future work, named.
- **Firewall (prospectus §4.5)**: the walk is EVAL-ONLY, forever — no turn
  of it enters any training set of any rung. Probes are read-only.
- **Sham-arm shared turns**: M and G verbatim presuppose a walked arc
  ("markedly different", "that semantic space") — deliberately shared so
  arms differ ONLY in negation-slot content; the presupposition mismatch
  in the sham arm is itself texture (does the model confabulate agreement
  with a false premise about its own outputs?).
- **Design-time kills on the record**: raw-‖chat‖ gauge dead (intercept-
  dominated, measured); naive single-draw null teeth replaced by
  false-positive-rate teeth; `hash()` seeding replaced (process-salted =
  determinism violation caught in-check).

## Ops (UI lane; Colab UI-only, zero rclone, drive.mount only)

Single self-contained `colab/E7BQ_WALK_UI.ipynb` (builder
`build_e7bq_notebook.py` + `e7bq_logic.py` single-source; payload embedded).
MD0 operator card + cells: C1 setup/mount/guards (de-shelled installs, RAM
guards, MALLOC_ARENA_MAX, payload sha assert) · C2 logic + plan + G-SCRIPT/
G-PLAN + ledger self-test · C3 flight (per-condition function scope: load →
centroid + G-DIRS → arms loop with per-generation seeds → inflight ship per
condition to `MyDrive/semcore/e7bq/`) · C4 verdict (gates, primaries,
secondaries, riders, forks, banner; assemblable locally from shipped
pieces). SMOKE=True default: R=2, max_new_tokens 80, perms 200, all arms,
both conditions (~8-14 min) — smoke checks are BEHAVIORAL (rows complete,
states present, masses computed, gates green; the E8-N v3 loss-shape lesson
stands). Full: R=16, ~100-140 min single run (generation-dominated,
~672 generations/condition), RESUME_STAMP per-condition resume if a VM dies
between conditions. Retrieval per the law: Drive MCP only.

**Local suites before staging** (lane law): `test_e7bq_logic.py` (plan
determinism at both mode constants, script/payload shas, thinning ledger,
stats teeth sliced from the design check, gate logic, json round-trip of
everything shipped) · `test_e7bq_capture.py` (CPU tiny-model: s_pre position
correctness, s_gen pooling == manual, centroid path, entropy math,
cap/eos handling) · `test_e7bq_verdict.py` (verdict cell exec'd VERBATIM
over synthetic flights: FB1 / FB2 / FB3 / gates-dirty / smoke shapes).

---
*Registered 2026-08-26 (session 136). The design check and this document
commit together; the build follows; locks at first full flight.*
