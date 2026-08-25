# E8-N v3 — pre-registration (LOCKED at first full flight)

**Rung**: the per-strand budget cure applied to E8-N v2's P2 (interface
competence), completing the E8-N program's registered slate item 1 — plus
two backlog riders measured through the same flight: S10′ (FC-interference
recovery, slate item 3's discriminating measurement) and S-ABS (absolute
calibration, slate item 4's prospective replication — the retrospective
half was MEASURED at design-check time from the locked v2 rows, results
below). Fresh registration; v2's forks were honored and closed.

**Provenance**: E8N2_PROTOCOL.md v3 slate (items 1, 3, 4) · Joe's backlog
directive (session 134: "navigate to the backlog… finish everything the
original research set out for") · design check `e8n3_design_check.py` +
`results_e8n3/design_check.json`, committed WITH this prereg, BEFORE build.

## The one training change (everything else v2-verbatim)

- **Convergence**: pooled plateau (v2) → **per-strand plateau** — training
  stops only when EVERY strand's epoch-mean smoothed loss improves < 5%
  relative at an epoch boundary (min 3 epochs). Donor:
  `e8o2_logic.per_strand_plateau` VERBATIM (flew E8-F v2 and E8-O2; the
  design check proves it returns not-plateaued on v2's flown curves at
  both mode constants — the cure demonstrated on the record).
- **EPOCH_CAP = 12** (design check: competence reaches base's passing loss
  ~ep 11, own plateau ~ep 15; cap-hit-still-descending is acceptable per
  the E8-F/E8-O2 precedent — the catch bar carries the verdict; cap-hit
  without full plateau still sets the UNDERTRAINED flag, honest).
- Curriculum (600 rows, 4 strands), LoRA shape, lr, answer-sliced head,
  interleave, seeds policy: **byte-identical to v2's builder path**.

## Scope

- **REAL condition only.** Base's P2 answer is locked (9/12 at 6 epochs,
  p 3.6e-06); base tracking/separability locked. All base comparisons ride
  locked numbers. Single run, ONE condition, RESUME_STAMP for mid-eval
  death. Worst-case ~116 min (measured v2 secs, cap-12 projection).
- **Eval blocks kept (verbatim code paths)**: pre battery 148 + pre catch
  12 → train → post battery 148 + post catch 12 · anchor 18 · shams 12+12
  · FC 96 heldout + 12 sham_fc · retention ppl. **Dropped**: held-out
  generation 48 (P3's questions belong to the staircase lineage
  R3-c→E8-F→E8-O→E8-O2; carrying the block without registered claims
  invites fishing) · paraphrase 12 (v2 descriptive, no open question).

## Primaries (Holm .05 over 2, real post)

- **P-V1 (the rung — P2 on budget)**: conjunction, v2-verbatim —
  (a) catch ≥ 9/12 post; (b) exact binomial tail
  P(X ≥ k_post | n=12, p₀), p₀ = (k_pre+1)/14 from THIS flight's measured
  pre catch. Passes iff (a) and Holm-rejected (b); p = 1.0 if (a) fails.
  v2 baselines: pre 1/12, post 7/12 (design check reproduces both via the
  flown `catch_score`).
- **P-V2 (tracking retention)**: post pooled ρ > 0, permutation p (2000
  within-arm shuffles, machinery verbatim). Descriptive non-inferiority
  CI vs v2's .6711 (cross-flight note, not a clause).

**S0 (own family of one, α .05)**: pre-readout pooled ρ > 0 — THIRD
prospective replication of instillation-alone natural tracking; predicted
band [.1, .3] (v1 .201, v2 .2171).

## Registered secondaries

- **S10′ — FC-interference (backlog slate item 3, the discriminating
  measurement)**: the 96 held-out FC rows verbatim on the v3 adapter;
  per-tid paired comparison against the PINNED v2 error vector
  (`results_e8n3/fc_baseline_v2.json`, sha 02ae2e7a0374a249, median
  46.98°; locked E8-R2 naming-only comparator 28.54°). Statistic: paired
  sign-flip test over the 96 tids, one-sided (v3 < v2), α .05, plus
  median. Pre-stated readings:
  - **RECOVERED** (p ≤ .05 AND v3 median ≤ 35°): the interference was
    epoch-starvation — slate item 3 CLOSED by the same cure, no recovery
    probe flies.
  - **PARTIAL** (p ≤ .05, median > 35°): budget helps, interference has a
    budget-independent component — the naming-only recovery probe stays
    queued with its target sharpened to the residual.
  - **PERSISTS** (p > .05): multi-task interference is budget-independent
    — the naming-only continued-training probe (E8-O2 continue-load
    pattern on `e8n2/inflight_20260823_2356/readout_real`) is the queued
    next rung for this item.
  - sham_fc NONE-top texture re-measured alongside (presence/identity
    dissociation watch).
- **S-ABS — absolute calibration (backlog slate item 4, prospective
  half)**: the design check MEASURED the retrospective half from the
  locked v2 flight-of-record rows (the trained target is
  report = 10·rank01(oriented referent), so calibration is a property of
  the flown data): **the readout's absolute scale is ARM-SPLIT** —
  familiarity slope .874 / raw-Pearson .9027 and saturation .724 / .9083
  are near-calibrated (interval-faithful, small LOO transfer penalty),
  while uncertainty .399 / .356 and tension .383 / .343 are
  scale-compressed toward mid-scale (rank tracking without interval
  scale; uncertainty rank-ρ .557 vs raw-Pearson .356). Pre-readout rows
  carry no absolute structure (slopes ≤ .29; uncertainty variance .08).
  v3 re-runs the SAME measurement on its post rows. **Registered
  predictions**: (i) the arm split REPLICATES (familiarity+saturation
  slopes ≥ .6; uncertainty+tension ≤ .5); (ii) slopes are STABLE under
  the extended budget — the compression is a property of the
  quantile-label design, not epoch starvation. Either outcome of (ii) is
  a finding: stable ⇒ absolute-scale training needs a label-design rung
  (queued as the item's remaining half alongside cross-format); moved ⇒
  budget was the bottleneck here too. No gate rides S-ABS.
- S4 flipped-subset stays positive (v2 .77) · S7 per-strand took gates +
  plateau/epochs log (competence-took ≥5/6, scalar ≥60%, lexicon ≥3/4) ·
  S8 report variance · S9 sham claims ≤ 2/24 (rides 0/24 ×2) · S3
  post−pre paired bootstrap (descriptive).

## Gates (fail ⇒ NO_VERDICT, primaries withheld)

G1 pins: pools/curriculum shas + battery row counts + kept-block tids
verbatim (design-check pins asserted on-VM) · G2 dirs-stability (naming
dirs vs shipped E8-R bundle) · G3 retention ppl ±5% · G4 parse (battery
parse-fail ≤ 20%, injection INVALID ≤ 10%) · G5 sham claims ≤ 2/24 ·
anchor ≥ 16/18 · **catch-pre sanity**: k_pre ≤ 4 (a pre catch ≥ 5/12
means the pre frame moved — engineering NO_VERDICT, not a fork).

## Kill/fork (pre-stated)

- **FN1** — P-V1 + P-V2 pass → **E8-N COMPLETE**: natural-state tracking
  (v1, v2) + interface competence (v3) both installed; competence was a
  budget problem — THIRD confirmation-by-cure. S10′/S-ABS readings
  annotate the lock.
- **FN2** — P-V1 fails with competence strand PLATEAUED (< 5% before cap)
  → the wall is deeper than budget at this scale/rank — honest stop for
  P2; the catch-domain-training variant stays queued as future work.
- **FN3** — P-V1 fails CAP-HIT with competence still descending ≥ 5% →
  epoch-starved AGAIN at 12: ONE pre-authorized re-fly at cap 18 with the
  competence strand doubled (120 rows), everything else identical; a
  second failure ends this prereg.
- **FN4** — P-V2 fails (tracking regresses under extended joint training)
  → budget-interference finding; P-V1 reported but the joint-curriculum
  frame is the headline casualty.
- Gates dirty → NO_VERDICT, engineering re-fly.

## Claim discipline

P-V1's claim is interface competence on the locked catch domains through
the joint curriculum — not general numeracy, not calibration-at-large.
S10′ closes/queues slate item 3 by measurement; S-ABS closes slate item
4's measurement half (retrospective measured at design check,
prospective replication in-flight) — the label-design rung and
cross-format remain queued and are NOT claimed by this flight.
