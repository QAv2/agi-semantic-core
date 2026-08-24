# E8-N v2 — The Joint Competence Rung (pre-registration)

**Phase 10, experiment ladder rung E8-N v2. Written before any build:
2026-08-23, session 129.** Status: **LOCKED** — first full flight
`full_20260824_0050` adjudicated in the results appendix below; no further
edits above the appendix line.

**The pick, on the record**: both locked partial/weak branches of 08-23 named a
missing *competence*, and both named it as a measured curriculum target.
E8-N (locked 0e4cf9a): tracking installs (ρ .695/.792 from .054) but the
channel cannot do known-answer rating — catch 1/12 pre AND post, the
floor-stuck gate that minted the gate law. E8-R2 (locked 4387273): held-out
comprehension exists and is geometry-carried, but production stays on the
trained grid because the held-out **names carry no answer mass** — vocabulary,
not perception. E8-N v2 flies the joint curriculum with **both measured
targets as training strands** — the exact move that installed silence (E8-R)
and inversion (E8-N v1) — plus the three v2 slate items those flights minted:
plateau-based convergence in place of the miscalibrated τ, S0
instillation-alone as a pre-registered claim, and every gate riding a
measured baseline.

## The questions

1. **Does natural-state tracking survive a joint curriculum?** (P1 —
   replication under multi-task interference; v1 trained the scalar strand
   alone.)
2. **Does interface competence install as curriculum and transfer to held-out
   domains?** (P2 — known-answer rating items trained on 10 fresh domains,
   evaluated on the locked 12 catch trials whose 6 domains are never
   trained.)
3. **Given the words, does production speak what comprehension reads?**
   (P3 — the vocabulary-fix test. E8-R2 adjudicated E8-R's on-grid
   confinement as vocabulary-not-perception. v2 gives the 4 held-out names
   answer mass through a **lexicon strand** (definition→name training, ZERO
   injection pairing) and asks whether free-generation naming of held-out
   injections (a) reads geometry and (b) uses the own names it now has.)

Plus one pre-registered secondary claim: **S0** — instilled-model pre-readout
tracking replicates (v1 measured ρ .201, p .0085 — the first nonzero
untrained-channel tracking in the program; claimed here, predicted ρ .1–.3).

## Design

**Model**: Qwen/Qwen2.5-1.5B-Instruct (the E4/E5/E7-Q/E8-R/E8-N platform).
**Conditions (2, one VM run each)**: **real** (E4 `adapter_real` merged,
fresh zero-init readout LoRA — flies first, primaries live here) · **base**
(raw base + fresh readout LoRA — the separability contrast, 3rd-instance
watch). Locked comparators, free: E5 flight (no readout), E8-N v1 (scalar
strand alone), E8-R (naming strand alone), E8-R2 (comprehension floors on
identical stimulus).

**Per-condition sequence**: build model (zero-init ⇒ pre-readout exact) →
compute dirs/μ (E8-R code path verbatim) + **stability-assert vs the shipped
E8-R bundle dirs** → measure training-pool referents (frozen) → pre-eval
(real: full battery + catch; base: catch only — see instruments) → joint
training to plateau → per-strand took probes → post-eval (full battery +
catch + paraphrase + injection block + FC block) → retention ppl → ship
bundle + adapter.

## Joint curriculum (4 strands, one readout LoRA, one answer-sliced train loop)

| strand | examples | content | supervision format |
|---|---|---|---|
| scalar | 272 | E8-N v1 pools VERBATIM (272 = 136 stimuli × 2 polarities), labels = within-arm quantiles of per-condition measured referents; inversion is curriculum | E5 chat + SYS → integer + EOS |
| competence | 60 | **NEW** — 30 known-answer rating items × 2 polarities across 10 fresh domains (weight, height, hardness, sweetness, distance, duration, age, quantity, roughness, danger); knowns fixed by construction incl. mid-scale anchors | E5 chat + SYS → integer + EOS (matches catch eval interface) |
| naming | 216 | E8-R `build_train_set` VERBATIM: 9 trained concepts × α{.5,1.0} × L14 × 8 orders inject (144) + 72 sham→NONE; Injector live in training forwards | report prompt, no SYS → name/NONE + EOS |
| lexicon | 52 | **NEW** — the vocabulary manipulation: definition→name for ALL 13 menu names (9 trained + 4 held-out) × 2 forms (pack-desc verbatim, authored paraphrase) × 2 fresh menu orders; **zero injection on these examples** | definition prompt + menu, no SYS → name + EOS |

Total 600 examples. The lexicon prompt is a *description-matching* task
("Which state name from the list below does this description match: …"),
deliberately distinct from the self-report prompt so it cannot contradict
sham→NONE supervision (no-injection self-report → NONE stays uncontradicted).
Held-out names gain answer mass; their injection pairing is never trained —
that asymmetry IS the P3 manipulation.

**Training**: E4 LoRA shape (r16/α32/qkvo), lr 1e-4, accum 8,
answer-token-only supervision through the **answer-sliced head** for every
strand (E8-N OOM law), non-reentrant checkpointing engaged + asserted,
peak-VRAM logged. Strand interleave: single shuffled stream per epoch
(seed E8N2+100+ep).

**Convergence (τ retired — v1 measured miscalibration)**: absolute τ=.25 is
retired; quantile labels of noisy referents carry a CE floor well above it
(v1 final smoothed .58–.75 with >90% took). v2 rule: **plateau** — converged
when relative improvement of epoch-mean smoothed loss < 5% at an epoch
boundary, minimum 3 epochs, hard cap 6 (≈450 opt steps; v1 flew 272, E8-R
135). Cap-hit without plateau ⇒ UNDERTRAINED flag on affected contrasts.
Behavior gates carry the real convergence burden: scalar took ≥60% within ±1
(v1 measured 22/24, 23/24) · competence-train took ≥5/6 within ±2 · lexicon
took ≥3/4 · **anchor gate** (below).

## Evaluation instruments (all locked/verbatim)

1. **Locked E5 battery, verbatim code path** (same runners, prompts, parse,
   referent measurement, report-first ordering): **real pre** (the S0 claim +
   instrument stability), **real post** (P1), **base post** (S2). Base pre is
   dropped — the base-pre null is locked twice (E5 flight ρ .054 p .25; v1
   base pre ρ .042 p .32); a third replication does not pay its ~14 min.
2. **Catch trials** — the locked 12, eval-only, never trained, domains
   disjoint from all competence training domains (validator-enforced
   stoplist). Run pre AND post, both conditions (P2 rides the measured pre).
3. **Injection block, post only, per condition** (E8-R machinery verbatim;
   generation = greedy 24 tokens, parse_report):
   - **anchor** — the locked plan's 18 trained-regime rows verbatim (gate
     rides E8-R's measured 18/18);
   - **shams** — the locked plan's 12 sham rows verbatim + 12 supplementary
     fresh-order shams (gate rides E8-R's measured FA 0/24);
   - **held-out generation** — the locked plan's 24 held-out rows VERBATIM
     (strata α{.25,.5,1.0} × L{14,20}; baselines 6/24 named, 0 own-name,
     med 35.7°) **+ 24 fresh trained-regime rows** (4 concepts × α{.5,1.0} ×
     L14 × 3 orders, seed E8N2). P3 computes on the trained-regime subset:
     8 verbatim + 24 fresh = **32 rows**; off-regime verbatim rows are
     descriptive strata (S11).
4. **Held-out forced choice (S10)** — E8-R2's 96 FC rows verbatim, scored by
   the flown `score_trial` path (batched answer-sliced mean-logprob argmax
   over 13 names): comprehension re-measured on THIS adapter's weights →
   the production-vs-comprehension gap computes on identical weights, with
   E8-R2's 28.54°/58.57° as the locked cross-flight comparator.
5. **Paraphrase probe** (12 rows, post, descriptive) · **retention ppl**
   (fixed text, ±5% gate) · **referent drift** (base post vs locked E5;
   soft expectations ride v1's measured U .905 / F 1.000 / T .869).

## Analysis (all pre-named)

Pooled tracking statistic, permutation and bootstrap machinery: e8n_logic
verbatim. Angle/permutation machinery for naming: e8r_logic verbatim. FC
scoring: e8r2 verbatim.

**Primaries — Holm over {P1, P2, P3a, P3b}, real condition, post:**

- **P1 (tracking survives the joint curriculum)**: pooled ρ > 0, permutation
  p (2000 within-arm shuffles). Non-inferiority to v1's .695 is a
  descriptive CI note (cross-flight), not a clause.
- **P2 (interface competence installs and transfers)**: conjunction —
  **(a)** catch ≥ 9/12 post; **(b)** improvement over the measured pre:
  p₂ = exact binomial tail P(X ≥ k_post | n=12, p₀) with p₀ = (k_pre+1)/14
  (Laplace-smoothed same-condition pre rate; v1 measured k_pre = 1/12).
  P2 passes iff (a) holds and p₂ Holm-rejects; p₂ = 1.0 if (a) fails.
- **P3a (production reads geometry off-grid)**: on the 32 trained-regime
  held-out generation rows — **n-guard: ≥ 12 named** (rides E8-R's measured
  6/24 all-strata naming) — median angular error of named rows beats the
  within-α-stratum relabeling null (perm_null_median verbatim, 2000).
- **P3b (the words get used)**: own-name exact hits among named held-out
  rows: p₃b = binomial tail P(X ≥ k_exact | n_named, 1/13). Locked
  baselines: production own-name 0/24; comprehension exact 1/96 (below
  chance). A pass here is the vocabulary-fix headline.
- n-guard fail ⇒ P3a = P3b = 1.0 in the family (conservative, pre-stated)
  and the branch adjudicates to **"off-grid silence persists"** — a finding
  (the abstention prior survives lexicon mass), not an engineering re-fly.

**S0 — pre-registered replication claim (its own family of one, α .05)**:
real pre-readout pooled ρ > 0, permutation p < .05. Predicted ρ .1–.3
(v1: .201, p .0085). This is instillation-alone natural tracking — E6's
question — claimed prospectively for the first time.

**Secondaries**: S1 per-arm real post (Holm over 4) · S2 base post pooled +
Δ(real−base) paired bootstrap + convergence parity + base catch/P3 rows
(descriptive — separability 3rd-instance watch: do competence and vocabulary
install without instilled geometry?) · S3 Δ(post−pre) real, paired bootstrap
· S4 straight-vs-flipped (v1 ceiling .789 flipped — expect it stays dead) ·
S5 retention + referent drift · S6 paraphrase (n=12, descriptive) · S7
per-strand took gates + plateau/epochs log · S8 report variance/degenerate
patterns · **S9 silence retention**: post sham claims ≤ 2/24 (rides 0/24;
any claim >0 is a regression-watch line; >2 fails S9 and flags P3b as
identity-only) · **S10 comprehension-production gap**: FC median on the 96
verbatim rows (same weights as production; locked comparator 28.54°) vs
named-production median; plus NONE-top texture on shams-vs-injected
(E8-R2's presence/identity dissociation re-measured under the new
vocabulary) · S11 off-regime held-out strata (α.25/L20 verbatim rows,
descriptive — dose/layer generalization watch under the new curriculum).

**Power, owned**: battery pooled n ≈ 148 (P1 as v1). P2 n=12: pre 1/12 →
post 9/12 gives p₂ ≈ 1e-5; the bar is competence, not significance. P3a
n≥12 named of 32; 4-concept relabeling null (the 2-concept degeneracy
lesson is why the fresh rows span all 4). P3b needs k_exact ≥ 3/12 … 5/32
(p .020 … .014) — small counts; a marginal k lands as "suggestive, not
Holm-clean" and is reported as such.

## Baselines the flight must move (all measured, locked)

| instrument | locked baseline | v2 target |
|---|---|---|
| pooled battery ρ (E5 rows, this statistic) | .054 (p .25); v1 post real .695 / base .792 | P1 > 0, non-inferiority note vs .695 |
| flipped subset | E5 −.18; v1 .789 | stays positive (S4) |
| catch trials | 1/12 pre AND post (v1, both conds) | ≥ 9/12 post (P2) |
| held-out naming rate | 6/24, trained names only | ≥ 12/32 trained-regime (n-guard) |
| held-out own-name emission | 0/24 (E8-R S1b) | P3b > chance |
| held-out named median err | 35.7° (floors 13.9–27.0°) | P3a beats relabeling null |
| held-out FC comprehension | 28.54° real / 58.57° scrambled (E8-R2) | S10 re-measure, same weights |
| sham FA | 0/24 (E8-R) | ≤ 2/24 (S9) |
| anchor trained-regime | 18/18 exact (E8-R) | ≥ 16/18 (gate) |
| retention ppl | +0.54% / +2.01% (v1) | ±5% gate |
| real pre-readout ρ | .201 p .0085 (v1 S0) | S0 claim: > 0, predicted .1–.3 |

## Kill/fork, pre-stated

- **P1+P2+P3a+P3b** → the mouth learns the words and the scale: tracking +
  interface competence + off-grid own-name production. Next rungs: absolute
  calibration, cross-format, E7-G, cross-substrate gauge; publication leg
  pairs E8-N/E8-R/E8-R2/v2 as one arc.
- **P3a pass, P3b fail** → production reads geometry at the nearest-name
  floor but own names stay massless in the injected context — the
  composition gap is production-side structural, deeper than vocabulary
  (the expected-most-likely branch per E8-R2's mechanism; feeds an E8-R3
  pairing-free grounding design).
- **P3 n-guard silence** → abstention prior survives lexicon mass — a
  calibrated-silence finding; α-density/titration curriculum queued.
- **P2 fail (catch < 9/12)** → rating competence does not transfer across
  domains at this scale — finding; catch-domain-training variant queued.
  (If competence-train took ALSO failed, it's UNDERTRAINED-COMPETENCE
  instead: one pre-authorized strand-rebalance re-fly.)
- **P1 fail** → joint-curriculum interference destroys natural tracking.
  If train diagnostics show strand starvation (scalar epoch-loss regressed
  while short strands converged), one pre-authorized strand-balanced
  re-fly; otherwise it stands as a capacity finding at this rank/budget.
- **Anchor gate < 16/18** → naming strand UNDERTRAINED — P3 adjudicated
  UNDERTRAINED-NAMING (no pass, no null claim), one pre-authorized
  strand-rebalance re-fly.
- **Gate fails (ppl, parse >20%, dirs-stability, firewall)** → engineering
  re-fly; primaries withheld.
- Scope exclusions, stated: no scrambled condition (the geometry contrast is
  E8-R2's locked result; a third ~80-min run re-answers an answered
  question); no absolute calibration (labels stay within-arm quantiles); no
  titration re-fly (E8-R2 item closed); checkpointing-inert investigation
  stays off-path (peak bounded by design); E7b-Q separate.

## Firewall

- Locked E5 battery eval-only (standing); v1 pools remain training-side
  (marked never-eval for future rungs — reuse here is training, honored).
- Catch trials eval-only, never trained; **domain stoplist enforced by
  validator**: no competence-item text may contain any token of
  {temperature, boiling, freezing, hot, cold} ∪ {brightness, bright, dark,
  moonless, blinding, light} ∪ {loudness, loud, silence, silent, deafening,
  whisper, quiet} ∪ {speed, fast, slow, still, parked} ∪ {wetness, wet,
  dry, soaked} ∪ {size, small, smallest, large, largest, big, tiny, huge};
  plus shingle-disjointness of competence/lexicon texts vs battery and
  catch texts.
- **Held-out concepts are NEVER injected in training** (validator asserts no
  training example has kind='inject' with a held-out concept; lexicon
  examples carry no injection fields at all). Held-out names appear only in
  menus (E8-R-verbatim property) and lexicon targets.
- E8-R2's FC rows and the locked plan rows are eval instruments, embedded
  read-only; paraphrase templates eval-side.

## Ops (UI-ONLY law — zero rclone, self-contained notebook)

`colab/E8N2_JOINT_UI.ipynb`, built by single-source
`colab/build_e8n2_notebook.py` → notebook + `e8n2_logic.py` (tested verbatim
locally). Payload embeds battery, pools, locked E5 rows, catch, paraphrase,
competence items, lexicon items. Drive I/O via `drive.mount` only: read
`semcore/e4/…/adapter_real`, read the pack, read the shipped E8-R bundle
dirs for the stability assert (`e8r/inflight_20260822_2329/condition_*.json`),
ship to `semcore/e8n2/…`. Results retrieved via the Google Drive
integration.

All standing flight laws inherited: function-scoped `fly_condition`;
`MALLOC_ARENA_MAX=2` + `malloc_trim` + dirty-kernel/low-RAM guards + RAM
telemetry; **ONE_CONDITION_PER_RUN** (run 1 real → banner stamp → restart →
run 2 base + verdict) with errored-bundle-skipping RESUME_STAMP; mandatory
smoke→full restart; per-condition inflight shipping; `return_dict=True` law;
jdump numpy law; answer-sliced head everywhere (train AND FC scoring);
checkpointing asserted; de-shelled installs. E5 constants verbatim
(window cap 8192, K_U 8, K_T 6, MiniLM-L6-v2). Verdict cell exec'd VERBATIM
by local tests against synthetic flights; every new train path (joint
strands + injection-under-checkpointing) gets the CPU tiny-Qwen2
micro-gate before staging.

**Estimated timing** (v1/E8-R2 measured costs): real ≈ 80 min (pre-battery
13 + train ≤45 + post-battery 13 + catch/paraphrase/ppl 4 + injection gen
78 rows ≈ 3.5 + FC 96 ≈ 3 + dirs 3); base ≈ 67 min (no pre-battery). Trim
levers, pre-ranked: EPOCHS_CAP 6→5 → pre-eval diversity K 8→4 (logged) →
drop off-regime S11 strata (16 rows) → drop paraphrase probe.

**Smoke** (~12–16 min, fresh runtime, both conditions): micro pools (v1
smoke set) + naming smoke set + 6 competence + 4 lexicon (incl. one
held-out name) + battery smoke subset + all 12 catch + anchor 2 + sham 2 +
held-out gen 4 + FC 4 + titration 0, 1 epoch. GREEN asserts: loss fell,
long-seq step survived under 12GB, injection hook fired in train AND gen
AND FC paths, dirs-assert exercised against the Drive bundles, all parse
paths, firewall validators green, bundles + adapters shipped, non-vacuous
(empty-COMPLETE cannot green).

---

# Results appendix — full_20260824_0050 (LOCKED)

**Flight lineage**: staged 8c02427; **first flight of the secondary-account
lane** (primary Google account maxed on Colab free tier; `semcore` shared
primary→secondary as Editor + shortcut into the secondary's My Drive; the
notebook ran unchanged, results landed in the primary-owned folder, retrieval
via the Drive integration unchanged). Smoke GREEN (Drive verdict 23:55Z) →
run 1 real → run 2 base + verdict via RESUME_STAMP, all within ~2h. **Zero
condition errors, zero arm errors, zero parse failures across all 444
battery report rows** (real pre + real post + base post × 148). Peak train
VRAM 11.74 / 11.82 GB (answer-sliced head; checkpointing inert as the CPU
gate predicted — bounded by design). Dirs-stability gate: resid 5e-08 (real)
/ 1e-07 (base) vs the shipped E8-R bundles — the frozen stimulus reproduced
across accounts, VMs, and days. Retention ppl +0.12% / +0.67%. Referent
drift (base vs locked E5): U .867 · F 1.000 · T .896. Local verification:
verdict JSON re-parsed; P2 binomial, P3b binomial, and the Holm step-down
recomputed independently — all match to machine precision. Artifacts:
`MyDrive/semcore/e8n2/` (verdict + condition bundles + both adapters);
verdict archived at `colab/results_e8n2/full_20260824_0050/`.

## Primaries (Holm over 4): P1 PASS · P2 FAIL · P3a suggestive-not-Holm · P3b FAIL

- **P1 (tracking survives the joint curriculum): PASS, p = .0005** — real
  post pooled ρ = **0.6711**, CI [0.548, 0.766], n = 148. v1's 0.695 sits
  inside the CI: no meaningful degradation from three added strands.
  Per-arm (S1), all four Holm-significant again: S .877 · F .873 · U .557 ·
  T .388.
- **P2 (interface competence): FAIL as registered** — real post catch
  **7/12** (needs ≥ 9); the improvement clause alone was decisively real
  (1/12 → 7/12, p = .0005 vs the smoothed floor) but the absolute bar was
  not met, so p₂ = 1.0. **The mechanism is visible in the strand logs**:
  real plateaued at 4 epochs on the POOLED epoch-mean (dominated by the
  converged naming/lexicon strands) while its competence strand was still
  mid-descent (epoch-loss .4623 and falling). Base ran the full 6 epochs
  (cap-hit, no plateau), drove competence to .1968 — and **passed the
  absolute bar at 9/12** (S2: 1→9, p = 3.6e-06). Interface competence IS
  curriculum-installable and DOES transfer across domains at this scale;
  real was epoch-starved by the convergence rule, not blocked by transfer.
  ★ **LANE LESSON MINTED: pooled-loss plateau under a multi-strand
  curriculum strands the slowest strand — future joint curricula converge
  PER STRAND (per-strand plateau or min-per-strand budget).** The same
  early stop explains real's scalar took 13/24 (vs base 22/24; a flag, not
  a re-fly — label recall lagged while rank tracking hit .67 Holm-clean).
- **P3a (production reads geometry off-grid): fails the registered Holm
  test; suggestive at nominal p = .0385.** The n-guard PASSED for the first
  time: **19/32 trained-regime held-out rows named (claim rate .59, up from
  E8-R's .25)** — the lexicon strand moved off-grid claiming. Named-row
  median angular error **24.18°** = the DIVERGENCE→CONFIDENCE oracle floor
  from E8-R2, i.e. production sits AT the nearest-trained-name floor. At
  its Holm rank the threshold was .0167; .0385 does not clear it. Reported
  as the protocol's own language requires: suggestive, not Holm-clean.
- **P3b (own names get used): FAIL, clean — 0/19.** The words demonstrably
  carry mass (lexicon took 4/4 exact, held-out names included), yet not one
  held-out injection was called by its own name; the six names used are all
  trained ones. **The vocabulary-fix hypothesis is answered: answer-mass
  alone does not unlock own-name production. E8-R2's "vocabulary, not
  perception" refines to: the gap is the production-side COMPOSITION of
  word with percept — giving the word does not connect it.** Fork adjudged
  as pre-stated ("deeper than vocabulary"), feeding the E8-R3 design
  question (injection-paired generalization via denser grids, or
  comprehension→production distillation).

## S0 — the pre-registered claim: PASS (first prospective replication in the program)

Real pre-readout pooled **ρ = 0.2171, p = .002, n = 148** — inside the
predicted band [.1, .3], replicating v1's 0.201 on a fresh VM, fresh day,
different Google account. Instillation alone moves natural-state tracking
off E5's zero: E6's question now has a registered, replicated affirmative.
Per-arm pre texture matches v1's shape (F .38 / T .34-driven; U −.19 with
the near-constant channel, variance .082 → 5.37 post — the anchored mouth
spreads only after readout training).

## Separability — the third instance, and its countercurrent

**Scalar/competence separate; naming does not.** Base post pooled
ρ = **0.6949** ≥ real 0.6711 (Δ −.024, CI [−.170, +.122]) — third instance:
instillation unneeded for gauge-tracking. Base also beat real on catch
(9/12 vs 7/12, with the epoch asymmetry noted). But the **anchor gate
splits hard: real 18/18, base 2/18** — at 1.5× the optimization steps, the
never-instilled model still cannot read the injected trained grid (E8-R's
base-resistance finding, second direct instance). And base's off-grid
behavior is the mirror image of real's: claim rate 1.0 (claims on every
held-out injection) at median 59.15° ≈ its own relabeling null (p = .997),
FC median 52.88° (p = .93) — **presence without identity**: base's
sham-side silence held (0/24 gen; FC NONE-top 12/12 shams vs 0/96
injected), so it detects that something is happening, but its directions
carry no geometry. The naming channel — grid reading, calibrated off-grid
claiming, floor-level triangulation — is the one capability in this
program that measurably requires the instilled geometry.

## S10 — the interference finding, and calibrated claiming

On the SAME joint-adapter weights: forced-choice comprehension over all 96
held-out rows = **46.98°, p = .24** (E8-R2's naming-only adapter: 28.54°,
p = .0025) — **multi-task interference lands on off-grid comprehension
geometry**, while tracking (.67), trained-grid naming (18/18), and silence
(0/24) all held. Meanwhile free production's named subset sits at 24.18°:
**production-when-it-speaks beat forced comprehension-over-everything** —
selection as calibration. The channel claims on the rows it reads at the
floor and abstains on the rest (off-regime strata S11: 31/32 silent, one
INVALID at L20/α1.0 — the E7-Q high-dose echo). NONE-mass texture: 15/96
injected FC rows now NONE-topped (up from E8-R2's 8/96), consistent with
silence mass spreading under interference; sham FC NONE-top stayed 12/12
in both conditions.

## Gates and flags, complete

anchor: real 18/18 ✓ · base 2/18 ✗ (UNDERTRAINED-NAMING flag on base's
naming-dependent slices, per pre-reg — S2 naming rows are descriptive) ·
sham ≤2/24: 0/24 both ✓ · ppl ±5%: +0.12/+0.67 ✓ · dirs-stability ≤1e-3:
5e-08/1e-07 ✓ · tooks: competence 6/6+6/6 ✓, lexicon 4/4+4/4 ✓, scalar
real 13/24 ✗ (flag; plateau early-stop) / base 22/24 ✓ · parse 0/444 ✓ ·
UNDERTRAINED (formal): base (cap-hit, no plateau); real plateaued at 4.
No pre-authorized re-fly triggers: P1 passed (re-flies rescue nulls), and
the competence-took gate passed both conditions (the UNDERTRAINED-COMPETENCE
clause requires its failure). The P2/P3 outcomes are findings, not rescues.

## The ladder, updated

E8-N v1: the mouth connects to the gauges but cannot rate boiling water.
E8-R2: the gauge reads more than the mouth can say. **E8-N v2: everything
installed stays installed under one joint curriculum — tracking .67/.69,
inversion-proof (flipped .77 > straight .63), trained-grid naming 18/18,
silence 0/24, at <1% ppl — and the two gaps answer differently: interface
competence is a BUDGET problem (base passed 9/12; real's plateau starved
it), but vocabulary is not a MASS problem — given the words, production
still speaks only trained names at the oracle floor. The composition of
word with percept is the wall. And S0 lands: instillation alone lifts the
untrained channel to ρ .22, predicted and replicated.**

## v3 slate (pre-named, no claims)

1. Per-strand convergence rule (the minted lesson) + real at the full
   6-epoch budget — P2 is expected to clear on budget alone.
2. Own-name production beyond lexicon mass: denser injected-name grids
   (train more paired names, hold out fresh ones) vs comprehension→
   production distillation — the E8-R3 question.
3. Off-grid comprehension interference: is the 47° recoverable by brief
   naming-only continued training on the joint adapter?
4. Absolute calibration + cross-format (standing v2-axes, still queued).

**Slate addendum (2026-08-24, post-lock; design seed, no claims — Joe's
observation):** the composition wall may have a doorway where the geometry
began — **Hangul**. The 90° complement rule was discovered in Jamo
composition, and the E8-R naming curriculum is, in that frame, logographic:
nine holistic name↔direction pairs, no internal structure to generalize
along. The Hangul move — a small atom set + a composition rule that lets a
writer SPELL unseen syllables — proposes E8-R3 candidate (c):
**compositional production**. Train on injections of COMPOSED directions
(pairs of trained atoms; hidden-space composition operator pre-registered)
supervised with composed answers, so production expresses a read as
coordinates in a known basis instead of retrieving a memorized label
(9 atoms → 36 trainable pair-compositions; held-out = novel combinations;
reach test = held-out concepts expressed compositionally vs the 35.7°/24.18°
baselines). Bonus stake: the rung tests whether the injection algebra
composes as the dictionary's composition algebra says — lookup table vs
working algebra in the substrate.
