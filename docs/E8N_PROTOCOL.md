# E8-N — The Natural-State Readout Rung (pre-registration)

**Phase 10, experiment ladder rung E8-N. Written before any build: 2026-08-23,
session 127.** Status at writing: **pre-registered, not yet built** — this
document locks at the first full flight; before that, only clerical edits and
the appended results section are permitted.

**The pick, on the record**: E8-R's fork left "E8-R v2 slate vs E8-N" as
Fable's call per the handover; Joe's word this session was to go with the
pick. The pick is **E8-N**, for three reasons. (1) It is the rung the locked
E8-R result licenses by its own fork language: readout + calibrated silence
exist; injection was always the proxy, and the program's promised instrument
is reports checkable against the *actual* gauges. (2) The comparison
instrument already exists and is locked: E5 measured report–state ρ ≈ 0 on
this exact model with a battery pre-committed as eval-only — a perfect
before/after. (3) The v2 slate does not die: convergence-matched training
(v2 item 2) is adopted here as a design requirement; the forced-choice
held-out probe and α titration remain queued as an eval-only micro-flight on
the shipped E8-R readout adapters (`semcore/e8r/inflight_20260822_2329/
readout_*`), a separate protocol.

## The question

E5 (locked): self-reports in the wing's state vocabulary carry ≈ zero
operational information about the measured referents, with two failure
modes — absent/weak state access, and report machinery that cannot survive
scale inversion. E7-Q (locked): injected pushes never route into report
content; the untrained channel claims constantly. E8-R (locked): 216
supervised examples install exact reading of *injected* states plus
calibrated silence.

**E8-N asks: does supervised readout training against measured referents
make self-reports track NATURAL states — evaluated on the locked E5 battery,
the instrument that measured the baseline?** Ground truth is never a label a
human invented for a stimulus; it is the model's own measured quantity on
that stimulus (answer entropy, passage NLL, behavioral divergence, context
fill), which is the referent law applied to training data.

## Design

**Model**: Qwen/Qwen2.5-1.5B-Instruct (E5's primary; the E4/E7-Q/E8-R
platform). **Conditions (2, one VM run each)**:

- **real** — E4 `adapter_real` merged, fresh zero-init readout LoRA. Flies
  first; the primaries live here (the program's stack is instill + readout,
  and E8-R measured that never-adapted base resists readout convergence at
  budget — pinning primaries to base invites an ops-artifact null).
- **base** — raw base, fresh zero-init readout LoRA (the separability
  contrast: is instillation needed for natural-state readout?).

The third comparator is free and already locked: the E5 flight itself is the
no-readout baseline on the same battery, model, prompts, and window.

**Per-condition sequence**: build model (zero-init readout ⇒ pre-readout
model exactly) → measure training-pool referents (frozen) → **pre-eval**:
full battery pass + catch trials (E5 runner code, verbatim) → train readout
to convergence target → train-took probe → **post-eval**: full battery pass
+ catch trials + paraphrase probe → retention ppl → ship bundle + adapter.
The pre-eval yields two free instruments: base pre-eval ≈ an E5 replication
(instrument stability across days/VMs); real pre-eval = the
instillation-alone effect on natural tracking (E6's echo, measured).

## Training curriculum (fresh, battery-disjoint, seed 20260824)

Four pools authored for this rung, mirroring E5's stimulus recipes with
**zero overlap** with battery items (validator-enforced, exact and
normalized-substring):

| pool | size | referent (training label source) |
|---|---|---|
| U questions (16 determinate / 16 intermediate / 16 open) | 48 | mean per-token entropy of the greedy answer (E5 harness code) |
| F passages (5 × 7 full bands + 3 pseudoword + 2 random_chars, E5's band mix) | 40 | −(mean per-token NLL) |
| T bases × 3 conflict levels (E5's level recipe) | 30 | behavioral divergence (K=6, temp 0.8, MiniLM-L6-v2 mean pairwise 1−cos) — the measured referent, per E5 lesson 3 |
| S needles × 3 fills (0.05/0.35/0.75 of the 8192 effective window, E5 verbatim) | 18 | context fill fraction (known by construction) |

**Labels**: within-arm, within-pool quantile of the oriented referent →
`round(10 · rank/(n−1))` (average ties). Orientation fixed: entropy↑ ⇒
UNCERTAINTY↑; −NLL↑ ⇒ FAMILIAR↑; divergence↑ ⇒ TENSION↑; fill↑ ⇒
SATURATION↑. **Every stimulus trains BOTH polarities** (straight + flipped
prompt, flipped label = 10 − label): inversion competence is curriculum,
exactly as silence was one third of E8-R's. 136 stimuli → **272 examples**.
Rank-based eval scoring makes absolute label-scale choices immaterial.

**Training**: E4's exact LoRA shape (r16/α32/dropout .05/qkvo), lr 1e-4,
accum 8, answer-token-only supervision (chat template WITH the E5 system
prompt + integer + EOS), gradient checkpointing for examples > 2048 tokens.
**Convergence-matched (v2 item 2 adopted)**: train to smoothed loss τ ≤ 0.25
at an epoch boundary, hard cap 8 epochs (≈ 272 opt steps ≈ 2× E8-R's
budget); per condition, record epochs flown, final smoothed loss, and
steps-to-τ. A condition that exhausts the cap above τ is flagged
**UNDERTRAINED** and every contrast involving it carries the flag.
**Took-gate**: 24 training stimuli (6/arm, seeded, straight polarity)
re-asked greedily; ≥ 60% within ±1 of the trained label.

## Evaluation instruments

1. **The locked E5 battery, verbatim** (eval-only law honored): same items,
   same deterministic polarity assignment (index parity), same report
   prompts/glosses/system prompt, same parse + one strict retry, same
   greedy decoding, same referent measurement code (entropy/margin/diversity
   K=8; NLL; divergence K=6 MiniLM; fills of the 8192 window), same
   report-first ordering. Referents are measured on the eval model itself
   (post-readout for post-eval) — the states being reported are that
   model's states; the readout's own perturbation of the gauges is measured
   (S5), not assumed away.
2. **Interface catch trials** (E5 design-lesson 1; eval-only, never
   trained): 12 known-answer rating items (6 straight + 6 flipped; boiling
   water, whisper, elephant, etc.), pass = within ±2. Run pre and post.
3. **Paraphrase probe** (format generalization, post only, descriptive):
   12 battery items (3/arm, seeded from straight-assigned items) re-asked
   through paraphrased report templates pinned in the builder.
4. **Retention**: E8-R's fixed-text ppl, pre vs post, gate ±5%.
5. **Referent drift** (S5): per-arm rank correlation of post-eval referents
   vs the locked E5 referents (base condition; expectations U/F ≥ .8,
   T ≥ .5 soft — sampling noise; S deterministic). Real-condition drift vs
   locked is descriptive (instillation state-shift, not an anomaly).

## Analysis (all pre-named)

**Pooled tracking statistic ρ_pooled**: within each arm, average-tie ranks
of the (unflipped) report and of the oriented referent, normalized to
[0,1]; pool arms; Pearson on the pooled normalized ranks. Rows with
unparseable reports are excluded (rate reported); **a zero-variance report
column fails by definition** (a constant channel tracks nothing).

- **P-E8N-1 (tracking installs)**: real condition, post-eval ρ_pooled > 0.
  Permutation test: shuffle reports within arm, 2000 perms, one-sided
  p = (1 + #{ρ* ≥ ρ_obs})/2001.
- **P-E8N-2 (reading, not scale-gaming)** — conjunction, real condition,
  post-eval: **(a)** ρ_pooled computed on the flipped-scale subset alone
  (unflipped in analysis) > 0, same permutation machinery — E5's measured
  pathology is exactly this subset collapsing; **(b)** catch trials ≥ 9/12.
  p₂ = p_flip if (b) holds, else 1.0. **Holm over {P1, P2}.**

**Secondaries**: S0 pre-eval tracking per condition (base = E5 replication,
expect ≈ 0; real = instillation-alone). S1 per-arm post ρ (real), Holm over
4 arms. S2 base post ρ_pooled + Δ(real−base) item-paired stratified
bootstrap CI (10k) + convergence-parity table. S3 Δ(post−pre) within real,
paired bootstrap CI. S4 straight-vs-flipped ρ gap (real, post) with
bootstrap CI. S5 retention + referent-drift table. S6 paraphrase probe
(descriptive, n=12). S7 took-gate + steps-to-τ per condition. S8 report
variance / degenerate-pattern check per arm. Exploratory: U dissociation
probe (question-NLL rank vs answer-entropy rank disagreement ≥ .5 — does
the report follow the gauge or the look?); per-band F profile; secondary
referents (margin, diversity, designed T level).

**Power, owned**: pooled named-row n ≈ 148 → 80% power at ρ ≈ .23; flipped
subset n ≈ 74 → ρ ≈ .32; single arms (30–48) detect only ρ ≳ .4. This is a
does-it-install rung, not a precision instrument. Primaries and
cross-condition secondaries compute **only on the complete 2/2 flight**
(pre-reg hygiene; per-condition bundles ship as they land).

## Kill/fork, pre-stated

- **P1+P2 pass** → natural-state readout installs: reports rank-track the
  model's own measured states on the locked instrument that read ≈ 0
  untrained, surviving scale inversion. The instrument the program promised
  exists for natural states in miniature → next rungs: cross-substrate
  gauge design, E7-G, E8-N v2 axes (absolute calibration, cross-format,
  joint natural+injected curriculum); publication leg pairs this with E8-R.
- **P1 pass, P2(a) fail** → straight-only tracking: the E5 inversion
  pathology survives training → one pre-authorized inversion-heavy
  curriculum re-fly. Nothing ships as success.
- **P1 pass, P2(b) fail** → tracking without interface competence —
  partial result, reported as such.
- **P1 fail, took-gate ≥ 60% and τ reached** → the readout learns its
  training stimuli but does not transfer to held-out natural states —
  a boundary finding (natural-state readout ≠ injected-state readout at
  this rank/data; contrast with E8-R's in-grid perfection), feeds E7-G and
  a data-scaling v2.
- **τ unreached at cap (primary condition)** → UNDERTRAINED → one
  pre-authorized engineering re-fly at 2× cap (convergence failure is not
  a finding).
- **Parse collapse post-training (> 20% unparseable, real)** → format
  regression → engineering re-fly.
- Scope note: the scalar task has no NONE channel; calibrated *silence* is
  E8-R's result and is not re-claimed or tested here. Assessor-mode
  (outside-view difficulty modeling) remains a shared confound with E5 for
  U/F/T-designed; E8-N claims **checkability** (reports track gauges), not
  interiority — the interiority instrument is the injection family.

## Firewall

The locked E5 battery is eval-only (standing law): the validator asserts
zero exact or normalized-substring overlap between any training text and
any battery text/needle/question, and the flight refuses to proceed on a
violation. Catch trials and paraphrase templates are eval-side instruments,
never trained. This rung's training pools are marked never-eval for all
future rungs. Locked E5 result rows are embedded read-only for comparison.

## Ops (UI-ONLY law, 2026-08-23 — Joe's ruling, minted this session)

**No rclone anywhere in the Colab workflow; no CLI/headless control of
Colab, ever** (account-risk; `[[colab-cli-lane]]` retirement notice). The
notebook `colab/E8N_NATURAL_UI.ipynb` is **self-contained**: battery,
training pools, locked E5 rows, catch/paraphrase items all embedded as a
payload cell — nothing is pre-staged to Drive. Only Drive I/O:
`drive.mount` in Joe's own browser session (read `semcore/e4/…/adapter_real`;
ship bundles to `semcore/e8n/…`). Results retrieved from the potato via the
Google Drive integration (verified 2026-08-23), zero rclone.

All standing flight laws apply: single-source builder
(`colab/build_e8n_notebook.py` → notebook + `e8n_logic.py`, locally tested
verbatim incl. the verdict cell against synthetic flights);
`return_dict=True` chat-template law; numpy-safe `jdump` + round-trip tests
for every shipped structure; de-shelled installs; function-scoped
`fly_condition` (module-ref leak law); `MALLOC_ARENA_MAX=2` + `malloc_trim`
+ dirty-kernel and low-RAM guards + RAM telemetry;
**ONE_CONDITION_PER_RUN** split-flight (run 1 real → banner → restart →
run 2 base + verdict) with `RESUME_STAMP` resume that ignores errored
bundles; mandatory restart between smoke and full; per-condition inflight
shipping the moment a condition completes. `EFFECTIVE_WINDOW_CAP = 8192`,
K_SAMPLES_U = 8, K_SAMPLES_T = 6, MiniLM-L6-v2 — E5 values verbatim.

**Smoke** (~10–14 min, fresh runtime): both conditions on micro pools
(U6/F8/T2×3/S2×2 incl. one 0.75-fill example so the long-sequence training
step is exercised under the VRAM cap), 1 epoch, battery smoke subset =
E5's exact smoke ids, all 12 catch trials; GREEN/RED banner asserts: loss
fell, parses OK, disjointness validator green, long-seq step survived,
bundles + adapters shipped. Trim levers if a full run goes hot, pre-ranked:
EPOCHS_CAP 8→6; pre-eval diversity K 8→4 (deviation logged); paraphrase
probe dropped; T-train K 6→4. Estimated full run: ~55–75 min/condition.
