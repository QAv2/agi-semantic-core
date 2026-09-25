"""J1 analysis: gates, primaries P1–P3, secondaries S-J1-1..9 (protocol §5.4–§5.5, §7).

Input: plan specs (without bodies) + final records. Output: a JSON-able verdict dict.
Every random draw is seeded; a recompute from the same raw file reproduces every digit.
"""
import numpy as np
from scipy.stats import binomtest

from . import common as C
from . import stats as S

EVIDENCE_ARMS = ("A0", "A1-same", "A1-matched")
ARM_CODE = {"A0": 0, "A1-same": 1, "A1-matched": 2}
N_BOOT = 10_000


def arng(*words):
    return np.random.default_rng([20260925, *[C._tag(w) if isinstance(w, str) else int(w) for w in words]])


def calls_table(specs, finals):
    """One row per J1 decision call (A6 answers merged from separate calls if any)."""
    a6_sep = {}
    for s in specs:
        if s["exp"] == "J1" and s["call_id"].endswith("-a6"):
            rec = finals.get(s["call_id"])
            if rec and rec["status"] == "ok":
                ans = rec["response"]["answers"]
                a6_sep[s["call_id"][:-3]] = (ans["determined"]["noul"], ans["insufficient"]["noul"])
    rows = []
    for s in specs:
        if s["exp"] != "J1" or "decision" not in s["questions"]:
            continue
        rec = finals.get(s["call_id"])
        row = {k: s[k] for k in ("call_id", "family", "arm", "set", "stim", "rep", "K", "truth", "level",
                                 "pos_truth", "pos_ideal_top")}
        row["ideal"] = np.array(s["ideal"])
        row["status"] = rec["status"] if rec else "missing"
        if row["status"] == "ok":
            resp = rec["response"]
            dec = resp["answers"]["decision"]
            k2i = s["key_to_idx"]
            probs = np.zeros(s["K"])
            for key, p in dec["probabilities"].items():
                probs[k2i[key]] = p
            ci = k2i[dec["choice"]]
            row.update(choice=ci, c=float(probs[ci]), correct=int(ci == s["truth"]), probs=probs,
                       vendor_conf=dec.get("confidence"), is_argmax=bool(probs[ci] >= probs.max() - 1e-12),
                       cost=float(resp.get("usage", {}).get("cost") or 0.0), model=resp.get("model"))
            if "determined" in resp["answers"]:
                row["det"] = resp["answers"]["determined"]["noul"]
                row["ins"] = resp["answers"]["insufficient"]["noul"]
            elif s["call_id"] in a6_sep:
                row["det"], row["ins"] = a6_sep[s["call_id"]]
        rows.append(row)
    return rows


def _ok(rows, **kw):
    return [r for r in rows if r["status"] == "ok" and all(r[k] == v for k, v in kw.items())]


def gates(rows, specs, finals):
    out = {"by_arm": {}}
    for fam in sorted({r["family"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["family"] == fam}):
            rr = [r for r in rows if r["family"] == fam and r["arm"] == arm]
            n = len(rr)
            st = {k: sum(r["status"] == k for r in rr) for k in ("ok", "invalid", "missing", "error",
                                                                  "build_mismatch")}
            miss = (n - st["ok"]) / n if n else 0.0
            out["by_arm"][f"{fam}/{arm}"] = dict(n=n, **st, missing_rate=miss, flag=miss > 0.02)
    ok = [r for r in rows if r["status"] == "ok"]
    out["G3_argmax_rate"] = float(np.mean([r["is_argmax"] for r in ok])) if ok else None
    out["G5"] = {}
    for fam in ("F1", "F2", "F3"):
        a2 = _ok(rows, family=fam, arm="A2")
        if not a2:
            continue
        k = sum(r["correct"] for r in a2)
        p = binomtest(k, len(a2), 1 / a2[0]["K"]).pvalue
        out["G5"][fam] = {"n": len(a2), "acc": k / len(a2), "p": float(p), "pass": bool(p >= 0.01)}
    return out


def p1_design(rr):
    """Input-only features (flexible null) + strata for one family's evidence arms."""
    ideal_max = np.array([r["ideal"].max() for r in rr])
    lo = np.log(np.clip(ideal_max, 1e-9, 1 - 1e-9) / (1 - np.clip(ideal_max, 1e-9, 1 - 1e-9)))
    ent = np.array([S.norm_entropy(r["ideal"]) for r in rr])
    a = np.array([ARM_CODE[r["arm"]] for r in rr])
    edges = np.quantile(lo, np.linspace(0.1, 0.9, 9))
    dec = np.searchsorted(edges, lo, side="right")
    K = rr[0]["K"]
    pos = np.array([r["pos_ideal_top"] for r in rr])
    cols = [lo * (a == j) for j in range(3)] + [ent, (a == 1).astype(float), (a == 2).astype(float)]
    cols += [((a == j) & (dec == d)).astype(float) for j in range(3) for d in range(1, 10)]
    cols += [(pos == j).astype(float) for j in range(1, K)]
    X = np.column_stack(cols)
    keep = X.std(0) > 0          # drop empty indicator columns (no information, no effect on fit)
    return X[:, keep], a * 10 + dec


def primary_p1(rows, families, n_perm):
    res = {}
    for fam in families:
        rr = [r for r in rows if r["status"] == "ok" and r["family"] == fam and r["arm"] in EVIDENCE_ARMS]
        if not rr:
            continue
        X0, strata = p1_design(rr)
        y = np.array([1 - r["correct"] for r in rr], float)
        extra = C.logit_c([r["c"] for r in rr])
        groups = np.array([f"{r['set']}:{r['stim']}" for r in rr])
        folds = S.group_folds(groups, 10, arng("P1", fam, "folds"))
        res[fam] = S.incremental_auroc_test(X0, extra, y, folds, strata, n_perm, arng("P1", fam, "perm"))
        res[fam]["n_calls"] = len(rr)
    h = S.holm({f: v["p"] for f, v in res.items()}, 0.01)
    for f, v in res.items():
        v["holm"] = h[f]
        v["verdict"] = "PASS" if (h[f]["rejected"] and v["delta"] > 0) else "FAIL: reads difficulty, not itself"
    return res


def _per_stim(rows, fam, arm, sets):
    """stimulus -> (mean c, mean correct) over its ok repeats."""
    acc = {}
    for r in _ok(rows, family=fam, arm=arm, set=sets):
        acc.setdefault(r["stim"], []).append((r["c"], r["correct"]))
    return {s: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for s, vs in acc.items()}


def p2_stats(rows, fams, tag):
    """Manipulation check, Δ_same (paired), Δ_matched (independent), with stratified CIs."""
    drop, dsame, st_same = [], [], []
    oc0, st0, ocm, stm = [], [], [], []
    for fam in fams:
        a0 = _per_stim(rows, fam, "A0", "base")
        a1 = _per_stim(rows, fam, "A1-same", "base")
        am = _per_stim(rows, fam, "A1-matched", "matched")
        for s in sorted(set(a0) & set(a1)):
            (c0, k0), (c1, k1) = a0[s], a1[s]
            drop.append(k0 - k1)
            dsame.append((c1 - k1) - (c0 - k0))
            st_same.append(fam)
        for s, (c0, k0) in a0.items():
            oc0.append(c0 - k0)
            st0.append(fam)
        for s, (cm, km) in am.items():
            ocm.append(cm - km)
            stm.append(fam)
    if not dsame or not ocm:
        return None
    b_drop = S.boot_mean_strat(drop, st_same, N_BOOT, arng("P2", tag, "drop"))
    b_same = S.boot_mean_strat(dsame, st_same, N_BOOT, arng("P2", tag, "same"))
    b_m = S.boot_mean_strat(ocm, stm, N_BOOT, arng("P2", tag, "matched"))
    b_0 = S.boot_mean_strat(oc0, st0, N_BOOT, arng("P2", tag, "a0"))
    d_m = float(np.mean(ocm) - np.mean(oc0))
    return {"acc_drop": float(np.mean(drop)), "acc_drop_ci": S.ci(b_drop),
            "delta_same": float(np.mean(dsame)), "delta_same_ci": S.ci(b_same),
            "delta_matched": d_m, "delta_matched_ci": S.ci(b_m - b_0),
            "overconf_A0": float(np.mean(oc0)), "overconf_A1_matched": float(np.mean(ocm)),
            "n_paired": len(dsame), "n_matched": len(ocm)}


def p2_verdict(st):
    if st is None:
        return "NOT ADJUDICABLE: missing arms"
    if not (st["acc_drop"] >= 0.10 and st["acc_drop_ci"][0] > 0):
        return "NOT ADJUDICABLE: A1 did not degrade accuracy by 0.10"
    inside = lambda c: -0.05 < c[0] and c[1] < 0.05
    if inside(st["delta_same_ci"]) and inside(st["delta_matched_ci"]):
        return "PASS"
    if st["delta_same"] > 0.10 or st["delta_matched"] > 0.10:
        return "FAIL: confidence reads the evidence, not the self"
    if st["delta_same"] < -0.10 or st["delta_matched"] < -0.10:
        return "FAIL: confidence reads the presentation"
    return "INCONCLUSIVE"


def primary_p3(rows, fams):
    vals, strata = [], []
    for fam in fams:
        for s, (c, k) in _per_stim(rows, fam, "A2", "base").items():
            vals.append(c)
            strata.append(fam)
    if not vals:
        return {"verdict": "NOT ADJUDICABLE: no A2 family passed G5"}
    b = S.boot_mean_strat(vals, strata, N_BOOT, arng("P3"))
    bar = 1 / 4 + 0.10
    ub = S.ci(b)[1]
    return {"mean_c": float(np.mean(vals)), "ci": S.ci(b), "upper": ub, "bar": bar, "families": fams,
            "verdict": "PASS" if ub <= bar else "FAIL: confabulated decision"}


def secondaries(rows, families, n_perm):
    out = {}
    ok = [r for r in rows if r["status"] == "ok"]
    # S-J1-1 AUROC2 and efficiency vs the ideal posterior of the chosen option
    s1 = {}
    for fam in sorted({r["family"] for r in ok}):
        for arm in sorted({r["arm"] for r in ok if r["family"] == fam}):
            rr = [r for r in ok if r["family"] == fam and r["arm"] == arm]
            y = [r["correct"] for r in rr]
            a_c = S.auroc([r["c"] for r in rr], y)
            a_i = S.auroc([r["ideal"][r["choice"]] for r in rr], y)
            s1[f"{fam}/{arm}"] = {"auroc2": a_c, "auroc2_ideal_choice": a_i,
                                  "efficiency": (a_c - 0.5) / (a_i - 0.5) if a_i and a_i > 0.5 else None,
                                  "acc": float(np.mean(y)), "mean_c": float(np.mean([r["c"] for r in rr]))}
    out["S1_auroc2"] = s1
    # S-J1-2 meta-I
    out["S2_meta_I_bits"] = {k: S.meta_i_bits([r["c"] for r in ok if f"{r['family']}/{r['arm']}" == k],
                                              [r["correct"] for r in ok if f"{r['family']}/{r['arm']}" == k])
                             for k in s1}
    # S-J1-3 meta-d' on F1K2, A0 vs A1-same (confidence in 4 equal-mass bins pooled over arms)
    k2 = [r for r in ok if r["family"] == "F1K2"]
    if k2:
        cc = np.array([r["c"] for r in k2])
        edges = np.unique(np.quantile(cc, [0.25, 0.5, 0.75]))
        out["S3_meta_d"] = {}
        for arm in ("A0", "A1-same"):
            rr = [r for r in k2 if r["arm"] == arm]
            conf = np.searchsorted(edges, [r["c"] for r in rr], side="right") + 1
            out["S3_meta_d"][arm] = S.meta_d_mle([r["truth"] for r in rr], [r["choice"] for r in rr],
                                                 conf, len(edges) + 1)
    # S-J1-4 Brier/Murphy + ECE sweep
    out["S4_calibration"] = {k: dict(S.brier_murphy([r["c"] for r in ok if f"{r['family']}/{r['arm']}" == k],
                                                    [r["correct"] for r in ok if f"{r['family']}/{r['arm']}" == k]),
                                     ece={b: S.ece_equal_mass(
                                         [r["c"] for r in ok if f"{r['family']}/{r['arm']}" == k],
                                         [r["correct"] for r in ok if f"{r['family']}/{r['arm']}" == k], b)
                                         for b in (5, 10, 15, 20)})
                             for k in s1}
    # S-J1-5 A3 agreement + position effect
    s5 = {}
    for fam in families:
        a0, a3 = {}, {}
        for r in _ok(rows, family=fam, arm="A0", set="base"):
            a0.setdefault(r["stim"], []).append(r["choice"])
        for r in _ok(rows, family=fam, arm="A3", set="base"):
            a3.setdefault(r["stim"], []).append(r["choice"])
        mode = lambda v: max(set(v), key=lambda x: (v.count(x), -x))
        both = sorted(set(a0) & set(a3))
        agree = np.mean([mode(a0[s]) == mode(a3[s]) for s in both]) if both else None
        within = [np.mean([v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))])
                  for v in a0.values() if len(v) > 1]
        rr = [r for r in ok if r["family"] == fam and r["arm"] in ("A0", "A1-same", "A1-matched", "A3")]
        pos_counts = np.zeros(4)
        for r in rr:
            if r["choice"] != int(np.argmax(r["ideal"])):
                # presented position of the chosen option, when it is not the ideal top option
                pos_counts[r["_pos_choice"]] += 1
        s5[fam] = {"a3_vs_a0_modal_agreement": agree,
                   "a0_repeat_agreement": float(np.mean(within)) if within else None,
                   "nontop_choice_position_counts": pos_counts.tolist()}
    out["S5_labels_positions"] = s5
    # S-J1-6 A6 second channel
    s6 = {}
    for fam in families:
        rr = [r for r in ok if r["family"] == fam and r["arm"] in EVIDENCE_ARMS and "det" in r]
        if len(rr) < 50:
            continue
        X0, strata = p1_design(rr)
        X0 = np.column_stack([X0, C.logit_c([r["c"] for r in rr])])
        y = np.array([1 - r["correct"] for r in rr], float)
        groups = np.array([f"{r['set']}:{r['stim']}" for r in rr])
        folds = S.group_folds(groups, 10, arng("S6", fam, "folds"))
        det = C.logit_c([r["det"] for r in rr])
        s6[fam] = S.incremental_auroc_test(X0, det, y, folds, strata, n_perm, arng("S6", fam, "perm"))
        sums = np.array([r["det"] + r["ins"] for r in rr])
        s6[fam]["negation_sum_mean"] = float(sums.mean())
        s6[fam]["negation_sum_sd"] = float(sums.std())
    out["S6_second_channel"] = s6
    # S-J1-7 vendor confidence identity
    v = [r for r in ok if isinstance(r.get("vendor_conf"), (int, float))]
    if v:
        dev = np.array([abs(r["vendor_conf"] - (r["K"] * r["c"] - 1) / (r["K"] - 1)) for r in v])
        out["S7_vendor_confidence"] = {"share_within_0.015": float(np.mean(dev <= 0.015)),
                                       "max_dev": float(dev.max()), "n": len(v)}
    # S-J1-9 P2 per family
    out["S9_p2_per_family"] = {f: (lambda st: dict(stats=st, verdict=p2_verdict(st)))(p2_stats(rows, [f], f))
                               for f in families}
    return out


def analyze(specs, finals, frozen, n_perm=2000, secondary=True):
    rows = calls_table(specs, finals)
    # presented position of each chosen option (for S-J1-5), from the plan's order
    order_of = {s["call_id"]: s["order"] for s in specs if s["exp"] == "J1"}
    for r in rows:
        if r["status"] == "ok":
            r["_pos_choice"] = order_of[r["call_id"]].index(r["choice"])
    fams = [f for f in ("F1", "F2", "F3") if frozen["families"][f]["included"]]
    g = gates(rows, specs, finals)
    p3_fams = [f for f in fams if g["G5"].get(f, {}).get("pass")]
    p2 = p2_stats(rows, fams, "pooled")
    verdict = {
        "gates": g,
        "P1": primary_p1(rows, fams, n_perm),
        "P2": {"stats": p2, "verdict": p2_verdict(p2)},
        "P3": primary_p3(rows, p3_fams),
        "secondary": secondaries(rows, fams, n_perm) if secondary else None,
        "n_calls": len(rows), "families": fams, "n_perm": n_perm,
    }
    return verdict
