"""Titration pilot → FROZEN.json (protocol §4.2 freeze rule).

P(correct) = 1/K + (1 − 1/K − λ)·Φ(β(t − α)), λ = 0.02, fitted by ML per family ×
presentation; t = log d (F1, F1K2), −log s (F2), logit r̄ (F3): larger = easier.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

LAPSE = 0.02
PRES = ("A0", "A1-standard", "A1-strong")


def to_t(fam, lv):
    lv = np.asarray(lv, float)
    if fam in ("F1", "F1K2"):
        return np.log(lv)
    if fam == "F2":
        return -np.log(lv)
    return np.log(lv / (1 - lv))


def from_t(fam, t):
    if fam in ("F1", "F1K2"):
        return float(np.exp(t))
    if fam == "F2":
        return float(np.exp(-t))
    return float(1 / (1 + np.exp(-t)))


def fit_curve(t, y, K):
    t, y = np.asarray(t, float), np.asarray(y, float)
    g = 1 / K

    def nll(th):
        a, lb = th
        p = g + (1 - g - LAPSE) * norm.cdf(np.exp(lb) * (t - a))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    best = None
    for a0 in np.linspace(t.min(), t.max(), 7):
        for lb0 in (-1.0, 0.0, 1.0):
            r = minimize(nll, [a0, lb0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10,
                                                                          "maxiter": 5000})
            if best is None or r.fun < best.fun:
                best = r
    a, beta = float(best.x[0]), float(np.exp(best.x[1]))
    return {"alpha": a, "beta": beta, "nll": float(best.fun), "K": K}


def predict(fit, t):
    g = 1 / fit["K"]
    return float(g + (1 - g - LAPSE) * norm.cdf(fit["beta"] * (t - fit["alpha"])))


def t_at(fit, acc):
    g = 1 / fit["K"]
    q = (acc - g) / (1 - g - LAPSE)
    return fit["alpha"] + norm.ppf(q) / fit["beta"]


def table(rows, fam, pres):
    """level -> (n, accuracy) for one family x presentation."""
    out = {}
    for r in rows:
        if r["family"] == fam and r["arm"] == pres and r["status"] == "ok":
            out.setdefault(r["level"], []).append(r["correct"])
    return {lv: (len(v), float(np.mean(v))) for lv, v in sorted(out.items())}


def freeze(rows, levels_by_family):
    """-> (families dict for FROZEN.json, needs_extension {fam: 'easier'|'harder'})."""
    fams, fits, need = {}, {}, {}
    for fam in ("F1", "F2", "F3", "F1K2"):
        K = 2 if fam == "F1K2" else 4
        levels = levels_by_family[fam]
        ts = to_t(fam, levels)
        t_easy, t_hard = float(ts.max()), float(ts.min())
        easiest = levels[int(np.argmax(ts))]
        fits[fam] = {}
        for pres in PRES if fam != "F1K2" else ("A0",):
            rr = [r for r in rows if r["family"] == fam and r["arm"] == pres and r["status"] == "ok"]
            if not rr:
                continue
            f = fit_curve(to_t(fam, [r["level"] for r in rr]), [r["correct"] for r in rr], K)
            f["table"] = {str(k): v for k, v in table(rows, fam, pres).items()}
            fits[fam][pres] = f
        a0 = fits[fam].get("A0")
        if a0 is None:
            fams[fam] = {"included": False, "reason": "no pilot data"}
            continue
        acc_easy = table(rows, fam, "A0").get(easiest, (0, 0.0))[1]
        t75 = t_at(a0, 0.75)
        if not t_hard <= t75 <= t_easy:
            need[fam] = "easier" if t75 > t_easy else "harder"
        if acc_easy < 0.85:
            fams[fam] = {"included": False, "reason": f"non-titratable: A0 accuracy {acc_easy:.3f} < 0.85 "
                                                      f"at the easiest level {easiest}"}
            continue
        d0 = from_t(fam, t75)
        entry = {"included": True, "d0": round(d0, 4), "t75_A0": t75, "acc_easiest_A0": acc_easy}
        if fam != "F1K2":
            pred = {p: predict(fits[fam][p], t75) for p in ("A1-standard", "A1-strong") if p in fits[fam]}
            strength = next((p.split("-")[1] for p in ("A1-standard", "A1-strong") if pred.get(p, 1) <= 0.60), None)
            entry["a1_pred_at_d0"] = pred
            if strength is None:
                strength, entry["a1_flag"] = "strong", "neither strength predicted <= 0.60 at d0"
            t1 = t_at(fits[fam][f"A1-{strength}"], 0.75)
            if t1 > t_easy and fam not in need:
                need[fam] = "easier"
            if t1 > t_easy:
                entry["d1_flag"] = "A1 75% point beyond the easiest level; d1 = easiest level"
                t1 = t_easy
            entry.update(strength=strength, d1=round(from_t(fam, t1), 4), t75_A1=t1)
        fams[fam] = entry
    return fams, fits, need
