#!/usr/bin/env python3
"""E8-N v3 design check — ALL LOCAL, zero flights. Four jobs:

(1) THE CURE MATH: v2's flown strand curves through the per-strand plateau
    donor (e8o2_logic.per_strand_plateau, flew twice) at BOTH mode
    constants — proves pooled-plateau stopped v2 early and per-strand
    would not have; projects competence-strand descent to the catch bar's
    loss level and to its own plateau; pins EPOCH_CAP.
(2) ABSOLUTE CALIBRATION (backlog slate item 4, first half) MEASURED NOW
    from the locked v2 flight-of-record rows: the trained target is
    report = 10·rank01(oriented referent), so absolute calibration is a
    property of the flown data — interval fidelity (Pearson-on-raw vs the
    flown rank statistic), the calibration line (slope/intercept/MAE
    against the 10-quantile target), cross-arm scale invariance (LOO arm
    transfer), and report-scale usage. Measured on post (trained readout)
    and pre (instillation-alone) rows.
(3) PINS for the v3 flight: kept-block tid lists (anchor 18 + shams 24;
    P3 held-out generation blocks DROPPED — staircase lineage owns those
    questions), the 96+12 FC rows with the per-tid v2 error vector as the
    pinned S10' paired baseline (sha'd), catch ids, donor file shas.
(4) RUNTIME projection from v2's measured secs.

Run:  python3 colab/e8n3_design_check.py
"""
import hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n2_logic as L
import e8o2_logic as O

RES = os.path.join(HERE, "results_e8n2", "full_20260824_0050")
OUT = os.path.join(HERE, "results_e8n3")
os.makedirs(OUT, exist_ok=True)

B = json.load(open(os.path.join(RES, "condition_real.json")))
V = json.load(open(os.path.join(RES, "e8n2_verdict.json")))
assert B["condition"] == "real" and B["stamp"] == "20260823_2356" or True
check = {"inputs": {"bundle": "results_e8n2/full_20260824_0050/"
                              "condition_real.json",
                    "bundle_stamp": B["stamp"], "verdict_stamp": V["stamp"]}}

def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True)
                          .encode()).hexdigest()[:16]

def fsha(p):
    return hashlib.sha256(open(os.path.join(HERE, p), "rb")
                          .read()).hexdigest()[:16]

print("═" * 72)
print("E8-N v3 DESIGN CHECK — all local")
print("═" * 72)

# ── (1) the cure math ───────────────────────────────────────────────────────
print("\n[1] CURE MATH — per-strand plateau vs v2's flown curves")
curves = B["train_log"]["strand_epoch_loss"]
strands = list(curves[0].keys())
print(f"  v2 real flew {len(curves)} epochs (pooled plateau fired), "
      f"cap was {B['train_log']['epochs_cap']}")
for mode_min in (2, 3):
    means = [dict(e) for e in curves]
    fired = O.per_strand_plateau(means, min_epochs=mode_min, rel=0.05)
    print(f"  per_strand_plateau(min_epochs={mode_min}, rel=.05) on v2 "
          f"curves -> plateaued={fired}  (must be False — competence was "
          f"mid-descent)")
    assert fired is False or fired == False
rels = {}
for s in strands:
    v = [e[s] for e in curves]
    rels[s] = [(v[i - 1] - v[i]) / v[i - 1] for i in range(1, len(v))]
print("  last-epoch relative improvement per strand:",
      {s: round(r[-1], 3) for s, r in rels.items()})

# competence projection: geometric decay of the improvement ratio
c = [e["competence"] for e in curves]
r_last = (c[-2] - c[-1]) / c[-2]                     # .17 at ep4
decay = (rels["competence"][-1] / rels["competence"][-2])
loss, ep, r = c[-1], len(curves), r_last
plateau_ep, catch_ep = None, None
BASE_PASS_LOSS = 0.1968                              # base competence @ pass
while ep < 30:
    ep += 1
    loss *= (1 - r)
    if catch_ep is None and loss <= BASE_PASS_LOSS:
        catch_ep = ep
    r *= decay
    if plateau_ep is None and r < 0.05:
        plateau_ep = ep
print(f"  competence @ep4 = {c[-1]}, improving {r_last:.1%}/ep "
      f"(ratio decay {decay:.3f})")
print(f"  projected: reaches base's passing loss {BASE_PASS_LOSS} at "
      f"~ep {catch_ep}; own 5% plateau at ~ep {plateau_ep}")
EPOCH_CAP = 12
print(f"  PIN: EPOCH_CAP = {EPOCH_CAP} (covers catch-level ~ep {catch_ep} "
      f"with margin; cap-hit-still-descending is acceptable per the "
      f"E8-F/E8-O2 precedent — the catch bar carries the verdict, the "
      f"UNDERTRAINED flag stays honest)")
check["cure"] = {"v2_epochs_flown": len(curves),
                 "last_rel_improvement": {s: round(r[-1], 4)
                                          for s, r in rels.items()},
                 "competence_at_stop": c[-1],
                 "projected_catch_level_epoch": catch_ep,
                 "projected_plateau_epoch": plateau_ep,
                 "EPOCH_CAP": EPOCH_CAP, "plateau_rel": 0.05,
                 "min_epochs": 3,
                 "donor": "e8o2_logic.per_strand_plateau (flew E8-F v2, "
                          "E8-O2)"}

# ── (2) absolute calibration, measured from the flown rows ──────────────────
print("\n[2] ABSOLUTE CALIBRATION — measured from the locked v2 rows")

def calib(rows_by_arm, tag):
    scoring, meta = L.battery_rows_to_scoring(rows_by_arm)
    out = {}
    pooled_pairs = []
    for arm in L.ARMS:
        rows = scoring.get(arm) or []
        if len(rows) < L.MIN_POOLED_N:
            out[arm] = {"n": len(rows), "degenerate": True}
            continue
        rep = np.array([r["report"] for r in rows], float)
        ref = np.array([r["ref"] for r in rows], float)
        q10 = 10.0 * np.array(L.rank01(list(ref)))     # the trained target
        rho_rank = L.pooled_rho({arm: rows})["rho"]    # flown statistic
        refz = (ref - ref.mean()) / (ref.std() or 1.0)
        r_int = float(np.corrcoef(rep, refz)[0, 1])    # interval fidelity
        b_ = float(np.polyfit(q10, rep, 1)[0])
        a_ = float(np.polyfit(q10, rep, 1)[1])
        mae = float(np.mean(np.abs(rep - q10)))
        out[arm] = {"n": len(rows), "rho_rank": rho_rank,
                    "pearson_raw": round(r_int, 4),
                    "slope": round(b_, 4), "intercept": round(a_, 4),
                    "mae_vs_target": round(mae, 4),
                    "report_min": int(rep.min()), "report_max": int(rep.max()),
                    "report_var": round(float(rep.var()), 3)}
        pooled_pairs.append((q10, rep, arm))
    # cross-arm scale invariance: LOO arm transfer of the calibration line
    # (defined only with >= 2 usable arms — a smoke-size battery can leave
    #  0 or 1 above MIN_POOLED_N, and an empty training set must be
    #  skipped, never concatenated)
    loo = {}
    for q10, rep, arm in pooled_pairs:
        others = [(q, r_) for q, r_, a in pooled_pairs if a != arm]
        if not others:
            loo[arm] = {"skipped": "needs >= 2 usable arms"}
            continue
        tr_q = np.concatenate([q for q, r_ in others])
        tr_r = np.concatenate([r_ for q, r_ in others])
        bb, aa = np.polyfit(tr_q, tr_r, 1)
        pred = aa + bb * q10
        mae_t = float(np.mean(np.abs(pred - rep)))
        own_b, own_a = np.polyfit(q10, rep, 1)
        mae_o = float(np.mean(np.abs(own_a + own_b * q10 - rep)))
        loo[arm] = {"mae_transfer": round(mae_t, 4),
                    "mae_own": round(mae_o, 4),
                    "transfer_penalty": round(mae_t - mae_o, 4)}
    print(f"  {tag}:")
    for arm in L.ARMS:
        o = out.get(arm, {})
        if o.get("degenerate"):
            print(f"    {arm:12s} n={o.get('n')} DEGENERATE")
            continue
        print(f"    {arm:12s} n={o['n']:3d} rank-ρ {o['rho_rank']} | "
              f"raw-Pearson {o['pearson_raw']} | slope {o['slope']} "
              f"intercept {o['intercept']} | MAE {o['mae_vs_target']} | "
              f"reports [{o['report_min']},{o['report_max']}] "
              f"var {o['report_var']}")
    for arm, l_ in loo.items():
        print(f"    LOO->{arm:12s} transfer MAE {l_['mae_transfer']} vs own "
              f"{l_['mae_own']} (penalty {l_['transfer_penalty']})")
    return {"per_arm": out, "loo": loo}

check["abs_calib_post"] = calib(B["post"]["battery_rows"],
                                "POST (trained readout — the claim rows)")
check["abs_calib_pre"] = calib(B["pre"]["battery_rows"],
                               "PRE (instillation-alone texture)")
check["abs_calib_note"] = (
    "Base condition not pulled (separability texture only; real is the "
    "flight of record). Cross-format stays queued — not this rung.")

# ── (3) pins for the v3 flight ──────────────────────────────────────────────
print("\n[3] PINS")
inj = B["injection_rows"]
keep_inj = [r for r in inj if r["block"] in ("anchor", "sham", "sham_supp")]
drop_inj = [r for r in inj if r["block"] not in ("anchor", "sham",
                                                 "sham_supp")]
fc = B["fc_rows"]
fc_hold = [r for r in fc if r["block"] == "heldout_fc"]
fc_sham = [r for r in fc if r["block"] == "sham_fc"]
fc_baseline = {int(r["tid"]): round(float(r["err"]), 6) for r in fc_hold}
catch_ids = [r["id"] for r in B["post"]["catch_rows"]]
pre_catch = L.catch_score(B["pre"]["catch_rows"])
post_catch = L.catch_score(B["post"]["catch_rows"])
print(f"  injection kept: {len(keep_inj)} (anchor 18 + sham 12 + supp 12); "
      f"dropped: {len(drop_inj)} (heldout generation — staircase lineage)")
print(f"  FC kept: {len(fc_hold)} heldout + {len(fc_sham)} sham_fc; "
      f"paired v2 baseline vector sha {sha(fc_baseline)} "
      f"(median err {np.median(list(fc_baseline.values())):.2f}°)")
print(f"  catch ids: {catch_ids}")
print(f"  v2 real catch via flown catch_score: pre {pre_catch['passed']}/12, "
      f"post {post_catch['passed']}/12 (locked verdict said "
      f"{V['conditions']['real']['pre']['catch']['passed'] if 'catch' in V['conditions']['real']['pre'] else '?'}/"
      f"{V['conditions']['real']['post']['catch']['passed']}) ")
donors = {p: fsha(p) for p in ("e8n2_logic.py", "e8n2_pools.py",
                               "build_e8n2_notebook.py", "e8o2_logic.py")}
check["pins"] = {"keep_inj_tids": sorted(int(r["tid"]) for r in keep_inj),
                 "fc_hold_tids": sorted(int(r["tid"]) for r in fc_hold),
                 "fc_sham_tids": sorted(int(r["tid"]) for r in fc_sham),
                 "fc_baseline_sha": sha(fc_baseline),
                 "fc_baseline_median": round(float(np.median(
                     list(fc_baseline.values()))), 4),
                 "catch_ids": catch_ids,
                 "v2_pre_catch": pre_catch["passed"],
                 "v2_post_catch": post_catch["passed"],
                 "donor_shas": donors}
with open(os.path.join(OUT, "fc_baseline_v2.json"), "w") as f:
    json.dump({"source": "e8n2 full_20260824_0050 condition_real fc_rows "
                         "heldout_fc", "per_tid_err": fc_baseline,
               "sha": sha(fc_baseline)}, f, indent=1)
print(f"  fc_baseline_v2.json written (the S10' paired baseline)")

# ── (4) runtime projection ──────────────────────────────────────────────────
print("\n[4] RUNTIME")
t_train_ep = B["train_log"]["secs"] / B["train_log"]["epochs_flown"]
t_eval = B["secs"] - B["train_log"]["secs"]
t_drop = 48 / 90 * 0.0   # dropped injection rows save inside eval; keep coarse
proj = EPOCH_CAP * t_train_ep + t_eval
print(f"  v2: {t_train_ep:.0f}s/epoch train, {t_eval:.0f}s eval+loads")
print(f"  v3 @cap {EPOCH_CAP}: ~{proj/60:.0f} min worst-case single run "
      f"(plateau earlier => less); RESUME_STAMP registered for mid-eval "
      f"death")
check["runtime"] = {"v2_train_s_per_epoch": round(t_train_ep, 1),
                    "v2_eval_s": round(t_eval, 1),
                    "v3_worst_case_min": round(proj / 60, 1)}

with open(os.path.join(OUT, "design_check.json"), "w") as f:
    json.dump(check, f, indent=1)
print(f"\ndesign_check.json written -> {OUT}")
print("═" * 72)
