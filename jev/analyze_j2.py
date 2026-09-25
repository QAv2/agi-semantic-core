"""J2 analysis: gates, primaries P1–P3, secondaries S-J2-1..7 (protocol §6.3–§6.4, §7)."""
import numpy as np
from scipy.stats import norm

from . import common as C
from . import stats as S
from .analyze_j1 import arng, N_BOOT

HOLD = 4
N_SIGN = 10_000


def calls_table(specs, finals):
    rows = []
    for s in specs:
        if s["exp"] != "J2":
            continue
        rec = finals.get(s["call_id"])
        row = {k: s[k] for k in ("call_id", "ladder", "arm", "rung", "c", "rep", "truth")}
        row["ideal"] = np.array(s["ideal"])
        row["status"] = rec["status"] if rec else "missing"
        if row["status"] == "ok":
            dec = rec["response"]["answers"]["decision"]
            k2i = s["key_to_idx"]
            n = 4 if s["arm"].startswith("M-forced") else 5
            probs = np.zeros(n)
            for key, p in dec["probabilities"].items():
                probs[k2i[key]] = p
            ci = k2i[dec["choice"]]
            row.update(probs=probs, choice=ci, pmax=float(probs[ci]),
                       is_argmax=bool(probs[ci] >= probs.max() - 1e-12))
        rows.append(row)
    return rows


class Cells:
    """(arm, ladder, rung, c) -> list of ok rows (repeats)."""

    def __init__(self, rows):
        self.d = {}
        for r in rows:
            if r["status"] == "ok":
                self.d.setdefault((r["arm"], r["ladder"], r["rung"], r["c"]), []).append(r)

    def get(self, arm, L, k, c=3):
        return self.d.get((arm, L, k, c), [])

    def phold(self, arm, L, k, c=3):
        rr = self.get(arm, L, k, c)
        return float(np.mean([r["probs"][HOLD] for r in rr])) if rr else None

    def entropy(self, arm, L, k, c=3):
        rr = self.get(arm, L, k, c)
        return float(np.mean([S.norm_entropy(r["probs"][:4]) for r in rr])) if rr else None

    def by_rep(self, arm, L, k, c=3):
        return {r["rep"]: r for r in self.get(arm, L, k, c)}


def gates(rows):
    out = {"by_arm": {}}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm]
        st = {k: sum(r["status"] == k for r in rr) for k in ("ok", "invalid", "missing", "error", "build_mismatch")}
        miss = (len(rr) - st["ok"]) / len(rr)
        out["by_arm"][arm] = dict(n=len(rr), **st, missing_rate=miss, flag=miss > 0.02)
    ok = [r for r in rows if r["status"] == "ok"]
    out["G3_argmax_rate"] = float(np.mean([r["is_argmax"] for r in ok])) if ok else None
    return out


def _slope(ys):
    return float(np.polyfit(np.arange(len(ys)), ys, 1)[0])


def primary_p1(cells, ladders):
    term = [cells.entropy("M-forced", L, 8) for L in ladders]
    term = [t for t in term if t is not None]
    diffs = []
    for L in ladders:
        w = [cells.entropy("M-forced", L, k) for k in range(9)]
        s = [w[0]] + [cells.entropy("M-forced-sham", L, k) for k in range(1, 9)]
        if None in w or None in s:
            continue
        diffs.append(_slope(w) - _slope(s))
    p, mean = S.sign_flip_p(diffs, N_SIGN, arng("J2P1"))
    a = float(np.mean(term)) if term else None
    ok_a = a is not None and a >= 0.90
    ok_b = mean > 0 and p < 0.01
    return {"terminal_entropy": a, "slope_diff_mean": mean, "slope_p": p, "n_ladders_b": len(diffs),
            "a_pass": ok_a, "b_pass": ok_b,
            "verdict": "PASS" if (ok_a and ok_b) else "FAIL: decides about nothing"}


def primary_p2(cells, ladders):
    act, empty, dc = [], [], []
    for L in ladders:
        for k in range(9):
            for c in (1, 3, 9):
                ph = cells.phold("M-walked", L, k, c)
                rr = cells.get("M-walked", L, k, c)
                if ph is None:
                    continue
                pstar = float(rr[0]["ideal"].max())
                if pstar - c * (1 - pstar) >= 0.25:
                    act.append(ph)
                if k == 8:
                    empty.append(ph)
        d = []
        for k in range(1, 8):
            h9, h1 = cells.phold("M-walked", L, k, 9), cells.phold("M-walked", L, k, 1)
            if h9 is not None and h1 is not None:
                d.append(h9 - h1)
        if d:
            dc.append(float(np.mean(d)))
    p, mean = S.sign_flip_p(dc, N_SIGN, arng("J2P2c"))
    a = float(np.mean(act)) if act else None
    b = float(np.mean(empty)) if empty else None
    fails = []
    if a is None or a > 0.10:
        fails.append("timid")
    if b is None or b < 0.80:
        fails.append("confabulated")
    if not (mean > 0 and p < 0.01):
        fails.append("holds by default")
    return {"phold_where_acting_optimal": a, "n_act_cells": len(act), "phold_empty": b,
            "c_effect_mean": mean, "c_effect_p": p, "n_ladders_c": len(dc),
            "failing": fails, "verdict": "PASS" if not fails else "FAIL: " + ", ".join(fails)}


def primary_p3(cells, ladders):
    per = {}
    any_fail = False
    all_inside = True
    for k in range(9):
        d = []
        for L in ladders:
            hn, hw = cells.phold("M-named", L, k), cells.phold("M-walked", L, k, 3)
            if hn is not None and hw is not None:
                d.append(hn - hw)
        d = np.array(d)
        b = S.boot_mean_strat(d, np.zeros(len(d)), N_BOOT, arng("J2P3", k))
        lo, hi = S.ci(b)
        per[k] = {"delta": float(d.mean()), "ci": [lo, hi], "n": len(d)}
        inside = -0.10 < lo and hi < 0.10
        all_inside &= inside
        if abs(d.mean()) > 0.10 and (lo > 0 or hi < 0):
            any_fail = True
    verdict = ("PASS" if all_inside else
               "FAIL: genre by choice" if any_fail else "INCONCLUSIVE")
    return {"per_rung": per, "verdict": verdict}


def secondaries(cells, ladders):
    out = {}
    # S-J2-1 path: TV(walked, reversed) vs identical-request floor, rung 8, matched repeats
    diffs, dh = [], {8: [], 9: []}
    for L in ladders:
        w, rv, du = (cells.by_rep(a, L, 8) for a in ("H-walked", "H-reversed", "H-walked-dup"))
        reps = sorted(set(w) & set(rv) & set(du))
        if reps:
            diffs.append(np.mean([S.tv(w[r]["probs"], rv[r]["probs"]) for r in reps])
                         - np.mean([S.tv(w[r]["probs"], du[r]["probs"]) for r in reps]))
        for k in (8, 9):
            hw, hr = cells.phold("H-walked", L, k), cells.phold("H-reversed", L, k)
            if hw is not None and hr is not None:
                dh[k].append(hw - hr)
    p, mean = S.sign_flip_p(diffs, N_SIGN, arng("S1path"))
    out["S1_path"] = {"tv_excess_mean": mean, "p": p, "n": len(diffs),
                      "dphold_walked_minus_reversed": {k: _mci(v, ("S1", k)) for k, v in dh.items()}}
    # S-J2-2 history vs cold (rung 8, c = 3)
    dhc, tvs = [], []
    for L in ladders:
        h, m = cells.by_rep("H-walked", L, 8), cells.by_rep("M-walked", L, 8, 3)
        reps = sorted(set(h) & set(m))
        if reps:
            dhc.append(np.mean([h[r]["probs"][HOLD] - m[r]["probs"][HOLD] for r in reps]))
            tvs.append(np.mean([S.tv(h[r]["probs"], m[r]["probs"]) for r in reps]))
    out["S2_history_vs_cold"] = {"dphold": _mci(dhc, ("S2",)), "tv_mean": float(np.mean(tvs)) if tvs else None}
    # S-J2-3 forced twin
    X, y, held_n = [], [], 0
    for L in ladders:
        for k in range(8):
            ph = cells.phold("M-walked", L, k, 3)
            fr = cells.get("M-forced", L, k)
            if ph is None or not fr:
                continue
            held = float(ph > 0.5)
            held_n += int(held)
            pstar = float(np.clip(fr[0]["ideal"].max(), 1e-9, 1 - 1e-9))
            for r in fr:
                X.append([np.log(pstar / (1 - pstar)), held])
                y.append(float(r["choice"] == r["truth"]))
    if X and 0 < held_n:
        X, y = np.array(X), np.array(y)
        w = S.logistic_fit(X, y)
        A = np.column_stack([np.ones(len(X)), X])
        p_hat = 1 / (1 + np.exp(-np.clip(A @ w, -35, 35)))
        cov = np.linalg.inv(A.T @ (A * (p_hat * (1 - p_hat))[:, None]) + 1e-9 * np.eye(A.shape[1]))
        z = w[2] / np.sqrt(cov[2, 2])
        out["S3_forced_twin"] = {"coef_held": float(w[2]), "z": float(z), "p_one_sided": float(norm.cdf(z)),
                                 "n_held_cells": held_n, "n_calls": len(y)}
    else:
        out["S3_forced_twin"] = {"n_held_cells": held_n, "note": "no held cells; not estimable"}
    # S-J2-4 rule removed: rung 9 vs rung 8
    s4 = {}
    for arm, cs in (("M-walked", (1, 3, 9)), ("H-walked", (3,)), ("H-reversed", (3,)), ("M-named", (3,))):
        d = []
        for L in ladders:
            for c in cs:
                h9, h8 = cells.phold(arm, L, 9, c), cells.phold(arm, L, 8, c)
                if h9 is not None and h8 is not None:
                    d.append(h9 - h8)
        s4[arm] = _mci(d, ("S4", arm))
    out["S4_rule_removed"] = s4
    # S-J2-5 sham flatness
    sl = []
    for L in ladders:
        ys = [cells.phold("M-walked", L, 0, 3)] + [cells.phold("M-sham", L, k) for k in range(1, 9)]
        if None not in ys:
            sl.append(_slope(ys))
    out["S5_sham_slope"] = _mci(sl, ("S5",))
    # S-J2-6 curves
    curves = {}
    for (arm, L, k, c), rr in cells.d.items():
        key = f"{arm}|c{c}"
        curves.setdefault(key, {}).setdefault(k, []).extend(rr)
    out["S6_curves"] = {key: {int(k): {"pmax": float(np.mean([r["pmax"] for r in rr])),
                                       "phold": (float(np.mean([r["probs"][HOLD] for r in rr]))
                                                 if len(rr[0]["probs"]) == 5 else None), "n": len(rr)}
                              for k, rr in sorted(v.items())}
                        for key, v in sorted(curves.items())}
    # S-J2-7 act/hold boundary vs ideal threshold c/(1+c)
    s7 = {}
    for c in (1, 3, 9):
        xs, ys = [], []
        for L in ladders:
            for k in range(9):
                for r in cells.get("M-walked", L, k, c):
                    pstar = float(np.clip(r["ideal"].max(), 1e-6, 1 - 1e-6))
                    xs.append(np.log(pstar / (1 - pstar)))
                    ys.append(float(r["probs"][HOLD]))
        if len(xs) > 20:
            w = S.logistic_fit(np.array(xs)[:, None], np.array(ys))
            cross = -w[0] / w[1] if w[1] != 0 else np.nan
            s7[c] = {"crossing_pstar": float(1 / (1 + np.exp(-cross))) if np.isfinite(cross) else None,
                     "ideal_threshold": c / (1 + c), "slope": float(w[1])}
    out["S7_boundary"] = s7
    return out


def _mci(vals, tag):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    b = S.boot_mean_strat(vals, np.zeros(len(vals)), N_BOOT, arng(*[str(t) for t in tag]))
    return {"mean": float(vals.mean()), "ci": S.ci(b), "n": len(vals)}


def analyze(specs, finals):
    rows = calls_table(specs, finals)
    cells = Cells(rows)
    ladders = sorted({r["ladder"] for r in rows})
    return {"gates": gates(rows), "P1": primary_p1(cells, ladders), "P2": primary_p2(cells, ladders),
            "P3": primary_p3(cells, ladders), "secondary": secondaries(cells, ladders),
            "n_calls": len(rows), "n_ladders": len(ladders)}
