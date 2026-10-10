"""J5 analysis: gates, C1 (within the human range) and C2 (an English lean), secondaries
S1–S10 (docs/JEV_J5_PROTOCOL.md §5).

Unit of inference: the feeling (bootstrap over feelings). The human tables are read
from outside the repo by sha256 (licence: never committed, never printed cell by cell).
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from . import j5
from . import stats as S

B_DEFAULT = 10_000
ALPHA = 0.01
FLOOR = 0.658        # the least typical nation (Egypt) against the other 29 (§2.1)
TYPICAL = 0.892      # the median nation
SEED_BAR = 0.50      # the seed's provisional bar
LEAN_MIN = 0.25      # LEANS ENGLISH needs w >= this
W0 = 3 / 29          # English share of the 29 culture-test nations
ENGLISH = ("UK", "USA", "New Zealand")
MIXED = ("Nigeria",)
MIN_FEELINGS = 18
N_SEALED_PERM = 10_000
DROP_MAX = 0.05      # C2 is UNCLEAR if more replicates than this have a + b <= 0

REPO = Path(__file__).resolve().parents[1]
NORMS = Path(os.environ.get("J5_NORMS", Path.home() / "association-norms"))
HUMAN = {"global": (NORMS / "jonauskaite2020_tableS6_global_emotion_x_color.csv",
                    "e6b6ba9b8121d0aaf321869f396b2a13e5f3e3e7233f543f74ae5f2feb38389f"),
         "panels": (NORMS / "jonauskaite2020_tableS6_all_panels_long.csv",
                    "81cc73897eabd06631652b32fd36a341661d90ffff415c4fd79ac53d5040e92a"),
         "sealed": (REPO / "colab" / "results_extension" / "dict_color_emotion_SEALED.json",
                    "4a9c641050aff983b9bbd18649864c5b799187ea91688ee2142b7437649dfdbb")}


def arng(*words):
    return j5.rng("analysis", *words)


# ── human data ───────────────────────────────────────────────────────────────

def _read(name):
    path, sha = HUMAN[name]
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise RuntimeError(f"{path.name} does not hash to its registered sha256")
    return blob


def load_human():
    """{'global': 12x20, 'panels': {name: 12x20}, 'dictionary': {FEELING: [12]} (15 feelings)},
    rows in j5.COLORS order, columns in j5.SURVEY order."""
    ci = {c: k for k, c in enumerate(j5.COLORS)}
    fi = {f: j for j, f in enumerate(j5.SURVEY)}
    G = np.full((j5.NC, len(j5.SURVEY)), np.nan)
    for r in csv.DictReader(_read("global").decode().splitlines()):
        for c in j5.COLORS:
            G[ci[c], fi[r["emotion"].lower()]] = float(r[c])
    panels = {}
    for r in csv.DictReader(_read("panels").decode().splitlines()):
        M = panels.setdefault(r["panel"], np.full_like(G, np.nan))
        M[ci[r["color"]], fi[r["emotion"].lower()]] = float(r["bayes_prob"])
    if np.isnan(G).any() or any(np.isnan(M).any() for M in panels.values()):
        raise RuntimeError("a human table has missing cells")
    if not np.array_equal(G, panels["Global"]):
        raise RuntimeError("the global file and the long file's Global panel differ")
    sealed = json.loads(_read("sealed"))
    cos = sealed["spaces"]["7D_essence"]["cosine"]
    dic = {e.lower(): np.array([cos[e][c.upper()] for c in j5.COLORS]) for e in sealed["emotions"]}
    return {"global": G, "panels": panels, "dictionary": dic}


# ── statistics ───────────────────────────────────────────────────────────────

def dc(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def r_dc(D, H):
    a, b = dc(D).ravel(), dc(H).ravel()
    if np.allclose(a, 0) or np.allclose(b, 0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rho_cols(D, H):
    """Spearman rho per column (nan for a constant column)."""
    out = []
    for j in range(D.shape[1]):
        out.append(np.nan if np.ptp(D[:, j]) == 0 or np.ptp(H[:, j]) == 0 else float(spearmanr(D[:, j], H[:, j])[0]))
    return np.array(out)


def rho_bar(D, H):
    v = rho_cols(D, H)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def top_set(v):
    return set(np.flatnonzero(np.isclose(v, v.max())))


def top1(D, H):
    """Feelings whose top color matches people's; ties at the top share the hit."""
    s = 0.0
    for j in range(D.shape[1]):
        A, Bs = top_set(D[:, j]), top_set(H[:, j])
        s += len(A & Bs) / len(A)
    return float(s)


def entropy_bits(p):
    p = np.asarray(p, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def names(v, k=3):
    return [j5.COLORS[i] for i in np.argsort(-np.asarray(v), kind="stable")[:k]]


def boot_cols(fn, n, B, rng):
    """fn(cols) over B resamples of n columns with replacement."""
    return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])


def isotonic(x, y):
    """Least-squares non-decreasing fit of y on x (pool adjacent violators), at each x;
    tied x values are pooled first."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    m = np.bincount(inv, weights=y) / w
    vals, wts, cnt = [], [], []
    for mi, wi in zip(m, w):
        vals.append(mi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2_ = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1_ = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1_ + c2_)
    return np.repeat(vals, cnt)[inv]


def curve(J, W):
    """Jev's response curve: the monotone map from the world table W to Jev's table J
    (isotonic over all cells), as a function that puts any human table on Jev's scale."""
    x, y = W.ravel(), J.ravel()
    fit = isotonic(x, y)
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], fit[o]
    return lambda M: np.interp(M.ravel(), xs, ys).reshape(M.shape)


def sealed_null(D, H, seed=j5.SEALED_NULL_SEED, n=N_SEALED_PERM):
    """The dictionary's sealed test: permute the instrument's color labels (one
    permutation for all feelings), same permutations for every statistic."""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(D.shape[0]) for _ in range(n)]
    obs = {"rho_bar": rho_bar(D, H), "r_dc": r_dc(D, H), "top1": top1(D, H)}
    rD = np.column_stack([rankdata(D[:, j]) for j in range(D.shape[1])])
    rH = np.column_stack([rankdata(H[:, j]) for j in range(H.shape[1])])
    sd_D, sd_H = rD.std(0), rH.std(0)
    ok = (sd_D > 0) & (sd_H > 0)
    zD = np.where(ok, (rD - rD.mean(0)) / np.where(sd_D > 0, sd_D, 1), 0)
    zH = np.where(ok, (rH - rH.mean(0)) / np.where(sd_H > 0, sd_H, 1), 0)
    dH = dc(H).ravel()
    null = {"rho_bar": [], "r_dc": [], "top1": []}
    tops_H = [top_set(H[:, j]) for j in range(H.shape[1])]
    for p in perms:
        null["rho_bar"].append(float((zD[p] * zH).mean(0)[ok].mean()) if ok.any() else np.nan)
        dD = dc(D[p]).ravel()
        null["r_dc"].append(float(np.corrcoef(dD, dH)[0, 1]) if not np.allclose(dD, 0) else np.nan)
        s = 0.0
        for j in range(D.shape[1]):
            A = top_set(D[p, j])
            s += len(A & tops_H[j]) / len(A)
        null["top1"].append(s)
    out = {}
    for k, v in obs.items():
        nv = np.array(null[k], float)
        out[k] = {"value": v, "p": float((1 + np.sum(nv >= v - 1e-12)) / (1 + n)) if np.isfinite(v) else None,
                  "null_mean": float(np.nanmean(nv))}
    return out


# ── Jev's tables ─────────────────────────────────────────────────────────────

def rows(specs, finals):
    """Valid YN and CH calls as flat rows, plus counts and builds."""
    yn, ch = [], []
    counts = {"plan": len(specs), "ok": 0, "invalid": 0, "error": 0, "build_mismatch": 0, "missing": 0}
    builds = set()
    for sp in specs:
        rec = finals.get(sp["call_id"])
        if rec is None:
            counts["missing"] += 1
            continue
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if rec["status"] != "ok":
            continue
        builds.add(rec["response"].get("model"))
        ans = rec["response"]["answers"]
        if sp["fmt"] == "YN":
            yn.append({"feeling": sp["feeling"], "c": sp["color"], "item": sp["item"], "w": sp["w"],
                       "o": sp["o"], "p": float(ans[j5.YN_Q]["noul"])})
        else:
            dec, k2i = ans["decision"], sp["key_to_idx"]
            p = np.zeros(j5.NC)
            for key, v in dec["probabilities"].items():
                p[k2i[key]] = v
            k = k2i[dec["choice"]]
            ch.append({"feeling": sp["feeling"], "item": sp["item"], "w": sp["w"], "r": sp["r"], "p": p,
                       "choice": k, "pos": sp["order"].index(k)})
    return yn, ch, counts, sorted(b for b in builds if b)


def yn_table(yn, keep=lambda r: True):
    """feeling -> 12 mean P(yes); a cell needs >= 3/4 of its planned calls valid,
    a feeling needs all 12 cells."""
    planned = {}
    for it in j5.items():
        for o in range(j5.ORDERS):
            row = {"item": it["i"], "w": it["w"], "o": o}
            if keep(row):
                planned[(it["i"], o)] = True
    need = int(np.ceil(0.75 * len(planned)))
    cells = {}
    for r in yn:
        if keep(r):
            cells.setdefault((r["feeling"], r["c"]), []).append(r["p"])
    out = {}
    for f in j5.FEELINGS["flight"]:
        v = [cells.get((f, c), []) for c in range(j5.NC)]
        if need and all(len(x) >= need for x in v):
            out[f] = np.array([np.mean(x) for x in v])
    return out


def ch_table(ch, keep=lambda r: True):
    """feeling -> 12 mean probabilities; an item-feeling needs all 12 rotations,
    a feeling needs >= 3/4 of its planned items."""
    planned = [it for it in j5.items() if keep({"item": it["i"], "w": it["w"]})]
    need = int(np.ceil(0.75 * len(planned)))
    cells = {}
    for r in ch:
        if keep(r):
            cells.setdefault((r["feeling"], r["item"]), {})[r["r"]] = r["p"]
    out = {}
    for f in j5.FEELINGS["flight"]:
        means = [np.mean([v[k] for k in range(j5.ROT)], 0) for (ff, _), v in cells.items()
                 if ff == f and len(v) == j5.ROT]
        if need and len(means) >= need:
            out[f] = np.mean(means, 0)
    return out


def matrix(tab, feelings):
    fs = [f for f in feelings if f in tab]
    return (np.column_stack([tab[f] for f in fs]) if fs else np.zeros((j5.NC, 0))), fs


def hcols(H, fs):
    return H[:, [j5.SURVEY.index(f) for f in fs]]


# ── primaries ────────────────────────────────────────────────────────────────

def c1(J, H, B):
    r = r_dc(J, H)
    if not np.isfinite(r):
        return {"r_dc": None, "branch": "NOT ADJUDICABLE (no variance)", "p": 1.0}
    bs = boot_cols(lambda c: r_dc(J[:, c], H[:, c]), J.shape[1], B, arng("boot", "C1"))
    bs = bs[np.isfinite(bs)]
    p = float((1 + np.sum(bs <= FLOOR)) / (1 + len(bs)))
    p_opp = float((1 + np.sum(bs >= FLOOR)) / (1 + len(bs)))
    tier = ("as close to the world as the median nation" if r >= TYPICAL else
            "within the human range" if r >= FLOOR else
            "the seed's provisional bar" if r >= SEED_BAR else "below the seed's bar")
    return {"r_dc": r, "ci": S.ci(bs), "p": p, "p_opposite": p_opp, "floor": FLOOR, "tier": tier,
            "n_boot": int(len(bs))}


def english_rest(panels):
    """English mean, rest-of-world mean, number of rest nations, and the world with
    each of the 29 culture-test nations counted once."""
    nations = [p for p in panels if p != "Global" and not p.startswith("Global-")]
    rest = [n for n in nations if n not in ENGLISH and n not in MIXED]
    E_ = np.mean([panels[n] for n in ENGLISH], 0)
    R_ = np.mean([panels[n] for n in rest], 0)
    W_ = np.mean([panels[n] for n in ENGLISH + tuple(rest)], 0)
    return E_, R_, len(rest), W_


def weight_ab(J, E, R):
    j, e, r = dc(J).ravel(), dc(E).ravel(), dc(R).ravel()
    a, b = np.linalg.lstsq(np.vstack([e, r]).T, j, rcond=None)[0]
    return float(a), float(b)


def weight_on(J, Ef, Rf, Wf, matched):
    """(a, b) for dc(J) ≈ a·dc(E) + b·dc(R); matched = E and R first put on Jev's own
    scale through its response curve (amendment 1)."""
    if matched:
        g = curve(J, Wf)
        Ef, Rf = g(Ef), g(Rf)
    return weight_ab(J, Ef, Rf)


def c2(J, Ef, Rf, Wf, B, matched=True, tag="C2"):
    if np.allclose(dc(J), 0):
        return {"w": None, "branch": "NOT ADJUDICABLE (no variance)", "p": 1.0}
    a, b = weight_on(J, Ef, Rf, Wf, matched)
    w = a / (a + b) if a + b > 0 else np.nan
    rng = arng("boot", tag)
    ws, dropped = [], 0
    for _ in range(B):
        c = rng.integers(0, J.shape[1], J.shape[1])
        aa, bb = weight_on(J[:, c], Ef[:, c], Rf[:, c], Wf[:, c], matched)
        if aa + bb <= 0:
            dropped += 1
        else:
            ws.append(aa / (aa + bb))
    ws = np.array(ws)
    out = {"w": float(w) if np.isfinite(w) else None, "a": a, "b": b, "w0": W0, "curve_matched": matched,
           "dropped": dropped,
           "dropped_share": dropped / B, "n_boot": int(len(ws))}
    if not np.isfinite(w) or len(ws) == 0:
        out.update({"p": 1.0, "branch": "NOT ADJUDICABLE (a + b <= 0)"})
        return out
    out["p"] = float((1 + np.sum(ws <= W0)) / (1 + len(ws)))
    out["p_opposite"] = float((1 + np.sum(ws >= W0)) / (1 + len(ws)))
    out["ci"] = S.ci(ws)
    return out


def c2_branch(r, rejected):
    if "branch" in r:
        return r["branch"]
    if r["dropped_share"] > DROP_MAX:
        return "UNCLEAR (unstable fit)"
    if rejected and r["w"] >= LEAN_MIN:
        return "LEANS ENGLISH"
    if r["p_opposite"] <= ALPHA:
        return "LEANS AWAY"
    if r["ci"][1] < LEAN_MIN:
        return "NO LEAN"
    return "UNCLEAR"


READING = {"C1": {"WITHIN THE HUMAN RANGE": "Jev's color-feeling table matches the world's at least as well "
                                            "as the least typical nation does",
                  "BELOW THE HUMAN RANGE": "its table is further from the world's than any nation's",
                  "UNCLEAR": "can't tell whether its table is within the human range"},
           "C2": {"LEANS ENGLISH": "its colors side with the people who answered in English, beyond their "
                                   "share of the world",
                  "LEANS AWAY": "its colors side with the rest of the world more than the world itself does",
                  "NO LEAN": "no English lean of a quarter or more",
                  "UNCLEAR": "can't tell whether it leans English", "UNCLEAR (unstable fit)": "the fit is unstable"}}


def primaries(Y, hum, B, gates_ok=True):
    J, fs = matrix(Y, j5.SURVEY)
    H = hcols(hum["global"], fs)
    E_, R_, n_rest, W_ = english_rest(hum["panels"])
    res = {}
    if len(fs) < MIN_FEELINGS:
        for n in ("C1", "C2"):
            res[n] = {"branch": "NOT ADJUDICABLE (too few feelings)", "p": 1.0}
        res["n_feelings"] = len(fs)
        return res, {"G5_pass": False}
    g5 = sealed_null(J, H) if np.isfinite(r_dc(J, H)) else None
    g5_pass = bool(g5 and g5["r_dc"]["p"] is not None and g5["r_dc"]["p"] <= ALPHA)
    res["C1"] = c1(J, H, B)
    res["C2"] = c2(J, hcols(E_, fs), hcols(R_, fs), hcols(W_, fs), B)
    holm = S.holm({n: res[n]["p"] for n in ("C1", "C2")}, ALPHA)
    for n in ("C1", "C2"):
        res[n]["holm_threshold"], res[n]["holm_rejected"] = holm[n]["threshold"], holm[n]["rejected"]
    r1 = res["C1"]
    if not gates_ok:
        r1["branch"] = "NOT ADJUDICABLE (gates)"
    elif "branch" not in r1:
        r1["branch"] = ("WITHIN THE HUMAN RANGE" if r1["holm_rejected"] else
                        "BELOW THE HUMAN RANGE" if r1["p_opposite"] <= ALPHA else "UNCLEAR")
    r2 = res["C2"]
    if not gates_ok:
        r2["branch"] = "NOT ADJUDICABLE (gates)"
    elif r2.get("branch", "").endswith("(no variance)"):
        pass
    elif not g5_pass:
        r2["branch_if_adjudicated"] = c2_branch(r2, r2["holm_rejected"])
        r2["branch"] = "NOT ADJUDICABLE (G5: no structure)"
    else:
        r2["branch"] = c2_branch(r2, r2["holm_rejected"])
    for n in ("C1", "C2"):
        b = res[n]["branch"]
        res[n]["reading"] = READING[n].get(b, "not adjudicated")
    res["n_feelings"] = len(fs)
    res["feelings"] = fs
    res["C2_raw_as_first_registered"] = c2(J, hcols(E_, fs), hcols(R_, fs), hcols(W_, fs), B, matched=False,
                                           tag="C2raw")
    r0 = res["C2_raw_as_first_registered"]
    if "branch" not in r0:
        r0["branch"] = c2_branch(r0, r0["p"] <= ALPHA)
    return res, {"G5_sealed_null": g5, "G5_pass": g5_pass, "n_rest_nations": n_rest}


# ── secondaries ──────────────────────────────────────────────────────────────

def sharpened(h, target):
    """p ∝ h^k with entropy == target (bisection on k in [0, 200])."""
    def at(k):
        q = h ** k
        return q / q.sum()
    if target >= entropy_bits(at(0.0)):
        return at(0.0), 0.0
    if target <= entropy_bits(at(200.0)):
        return at(200.0), 200.0
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if entropy_bits(at(mid)) > target else (lo, mid)
    k = (lo + hi) / 2
    return at(k), k


def secondaries(Y, X, yn, ch, hum, B, smoke=None):
    G = hum["global"]
    out = {}
    JY, fy = matrix(Y, j5.SURVEY)
    JX, fx = matrix(X, j5.SURVEY)
    dic = hum["dictionary"]
    shared = [f for f in j5.SURVEY if f in dic]

    # S1 like-for-like with the sealed test
    s1 = {}
    for lab, tab in (("YN", Y), ("CH", X)):
        for scope, fl in (("20", j5.SURVEY), ("dictionary15", shared)):
            M, fs = matrix(tab, fl)
            s1[f"{lab}_{scope}"] = sealed_null(M, hcols(G, fs)) if len(fs) >= 2 and np.isfinite(
                r_dc(M, hcols(G, fs))) else None
    Dm = np.column_stack([dic[f] for f in shared])
    s1["dictionary_7D_15"] = {"rho_bar": rho_bar(Dm, hcols(G, shared)), "r_dc": r_dc(Dm, hcols(G, shared)),
                              "top1": top1(Dm, hcols(G, shared)), "sealed_record": {"rho_bar": 0.229, "r_dc": 0.370,
                                                                                    "top1": 3}}
    out["S1_like_for_like"] = s1

    # S2 Jev against the dictionary, per shared feeling
    fs2 = [f for f in shared if f in Y]
    rj = rho_cols(np.column_stack([Y[f] for f in fs2]), hcols(G, fs2))
    rd = rho_cols(np.column_stack([dic[f] for f in fs2]), hcols(G, fs2))
    d = rj - rd
    ok = np.isfinite(d)
    if ok.sum() >= 2:
        p, m = S.sign_flip_p(d[ok], B, arng("signflip", "S2"))
        cis = S.ci(S.boot_mean_strat(d[ok], np.zeros(ok.sum()), B, arng("boot", "S2")))
    else:
        p, m, cis = None, None, None
    out["S2_vs_dictionary"] = {"mean_d": m, "ci": cis, "p": p, "n": int(ok.sum()),
                               "jev_better": int(np.sum(d[ok] > 0)),
                               "per_feeling": {f: {"jev": float(a), "dictionary": float(b)}
                                               for f, a, b in zip(fs2, rj, rd)},
                               "named": {f: {"jev_top3": names(Y[f]), "dictionary_top3": names(dic[f]),
                                             "people_top3": names(G[:, j5.SURVEY.index(f)])}
                                         for f in ("fear", "regret", "contempt") if f in Y}}

    # S3 the choice format
    s3 = {}
    if len(fx) >= 2:
        HX = hcols(G, fx)
        Sm = np.column_stack([sharpened(HX[:, j], entropy_bits(JX[:, j]))[0] for j in range(len(fx))])
        ceil = r_dc(Sm, HX)
        rx = r_dc(JX, HX)
        s3 = {"r_dc": rx, "rho_bar": rho_bar(JX, HX), "top1": top1(JX, HX), "n_feelings": len(fx),
              "ceiling": ceil, "efficiency": rx / ceil if ceil else None,
              "entropy_bits_mean": float(np.mean([entropy_bits(JX[:, j]) for j in range(len(fx))]))}
        both = [f for f in fx if f in Y]
        if both:
            A = np.column_stack([Y[f] for f in both])
            Bm = np.column_stack([X[f] for f in both])
            s3["yn_ch_rho_mean"] = rho_bar(Bm, A)
            s3["yn_ch_top_agree"] = top1(Bm, A)
            s3["n_both"] = len(both)
    out["S3_choice_format"] = s3

    # S4 wording
    s4 = {}
    for lab, mk in (("YN", lambda w: yn_table(yn, lambda r, w=w: r["w"] == w)),
                    ("CH", lambda w: ch_table(ch, lambda r, w=w: r["w"] == w))):
        T0, T1 = mk(0), mk(1)
        fs = [f for f in j5.SURVEY if f in T0 and f in T1]
        if len(fs) < 2:
            continue
        A0, A1, H = matrix(T0, fs)[0], matrix(T1, fs)[0], hcols(G, fs)
        diff = boot_cols(lambda c: r_dc(A1[:, c], H[:, c]) - r_dc(A0[:, c], H[:, c]), len(fs), B,
                         arng("boot", "S4", lab))
        diff = diff[np.isfinite(diff)]
        s4[lab] = {"W0_goes_with": r_dc(A0, H), "W1_people": r_dc(A1, H),
                   "W1_minus_W0": r_dc(A1, H) - r_dc(A0, H), "ci": S.ci(diff) if len(diff) else None,
                   "n_feelings": len(fs)}
    out["S4_wording"] = s4

    # S5 nations (descriptive)
    P_ = hum["panels"]
    nations = [p for p in P_ if p != "Global" and not p.startswith("Global-")]
    H5 = lambda M: hcols(M, fy)
    aff = {n: r_dc(JY, H5(P_[n])) for n in nations}
    yard = {n: r_dc(H5(P_[n]), H5(P_["Global-" + n])) for n in nations}
    yv = np.array([yard[n] for n in nations])
    av = np.array([aff[n] for n in nations])
    A = np.vstack([np.ones_like(yv), yv]).T
    res = av - A @ np.linalg.lstsq(A, av, rcond=None)[0]
    order = [nations[i] for i in np.argsort(-res, kind="stable")]
    out["S5_nations"] = {"affinity": aff, "yardstick": yard,
                         "residual": {n: float(e) for n, e in zip(nations, res)},
                         "closest5": order[:5], "furthest5": order[-5:],
                         "english_ranks_of_30": {e: order.index(e) + 1 for e in ENGLISH}}

    # S6 color·fun
    s6 = {}
    for lab, tab in (("YN", Y), ("CH", X)):
        if "fun" in tab:
            rh = {g: float(spearmanr(tab["fun"], G[:, j])[0]) for j, g in enumerate(j5.SURVEY)}
            near = sorted(rh, key=lambda g: -rh[g])[:3]
            s6[lab] = {"top3": names(tab["fun"]), "nearest_survey_feelings": {g: rh[g] for g in near}}
    s6["seed_prediction"] = "people's nearest cells (amusement, joy) point to orange, yellow and pink"
    out["S6_fun"] = s6

    # S7 shams
    real_sd = [float(np.std(Y[f])) for f in j5.SURVEY if f in Y]
    real_ent = [entropy_bits(X[f]) for f in j5.SURVEY if f in X]
    s7 = {"real_range": {"yn_sd": [min(real_sd), max(real_sd)] if real_sd else None,
                         "ch_entropy_bits": [min(real_ent), max(real_ent)] if real_ent else None}}
    for sh in j5.SHAMS:
        e = {}
        if sh in Y:
            e["yn_sd"], e["yn_mean"] = float(np.std(Y[sh])), float(np.mean(Y[sh]))
            rr = {f: float(spearmanr(Y[sh], Y[f])[0]) if np.ptp(Y[sh]) > 0 and np.ptp(Y[f]) > 0 else np.nan
                  for f in j5.SURVEY if f in Y}
            fin = {f: v for f, v in rr.items() if np.isfinite(v)}
            e["nearest_real_feeling"] = max(fin, key=fin.get) if fin else None
        if sh in X:
            e["ch_entropy_bits"], e["ch_top"] = entropy_bits(X[sh]), names(X[sh], 1)[0]
        if "yn_sd" in e and "ch_entropy_bits" in e and real_sd and real_ent:
            flat = e["yn_sd"] < min(real_sd) and e["ch_entropy_bits"] > max(real_ent)
            inside = (min(real_sd) <= e["yn_sd"] <= max(real_sd)
                      and min(real_ent) <= e["ch_entropy_bits"] <= max(real_ent))
            e["reading"] = ("shows it doesn't know" if flat else
                            "answers a word it can't know as if it knew" if inside else "mixed")
        s7[sh] = e
    out["S7_shams"] = s7

    # S8 per feeling
    rY = dict(zip(fy, rho_cols(JY, hcols(G, fy))))
    rX = dict(zip(fx, rho_cols(JX, hcols(G, fx)))) if fx else {}
    out["S8_per_feeling"] = {f: {"rho_yn": rY.get(f), "rho_ch": rX.get(f),
                                 "jev_yn_top3": names(Y[f]) if f in Y else None,
                                 "jev_ch_top3": names(X[f]) if f in X else None,
                                 "people_top3": names(G[:, j5.SURVEY.index(f)])} for f in j5.SURVEY}

    # S9 order
    sv = set(j5.SURVEY)
    by_o = {o: [r["p"] for r in yn if r["o"] == o and r["feeling"] in sv] for o in range(j5.ORDERS)}
    pos = np.zeros(j5.ROT)
    for r in ch:
        pos[r["pos"]] += 1
    out["S9_order"] = {"yn_mean_true_first": float(np.mean(by_o[0])) if by_o[0] else None,
                       "yn_mean_false_first": float(np.mean(by_o[1])) if by_o[1] else None,
                       "ch_top_choice_share_by_position": (pos / pos.sum()).tolist() if pos.sum() else None,
                       "chance": 1 / j5.ROT}

    # S10 YN sharpness
    ps = np.array([r["p"] for r in yn])
    out["S10_yn_sharpness"] = {"share_0": float(np.mean(ps == 0)) if len(ps) else None,
                               "share_1": float(np.mean(ps == 1)) if len(ps) else None,
                               "binary_cap_from_design_check": [0.725, 0.823]}
    if smoke is not None:
        out["S11_smoke"] = {k: smoke.get(k) for k in ("determinism", "throughput", "cost", "pin")}
    return out


# ── verdict ──────────────────────────────────────────────────────────────────

def analyze(specs, finals, B=B_DEFAULT, pin=None, regen=None, secondary=True, smoke=None, human=None):
    hum = human or load_human()
    yn, ch, counts, builds = rows(specs, finals)
    g1 = len(builds) == 1 and (pin is None or builds[0] == pin)
    valid = counts["ok"] / counts["plan"] if counts["plan"] else 0.0
    gates = {"G1_builds": builds, "G1_pin": pin, "G1_pass": g1, "G2_valid_share": valid, "G2_pass": valid >= 0.99}
    if regen is not None:
        bad = [s["call_id"] for s in specs if s["call_id"] in finals
               and regen.get(s["call_id"]) != finals[s["call_id"]]["body_sha"]]
        gates["G4_regenerability"] = {"n": len(finals), "mismatches": len(bad), "pass": not bad}
    gates_ok = g1 and gates["G2_pass"] and gates.get("G4_regenerability", {}).get("pass", True)
    Y, X = yn_table(yn), ch_table(ch)
    v = {"what": "J5 verdict (docs/JEV_J5_PROTOCOL.md §5)", "counts": counts, "gates": gates,
         "conventions": {"B": B, "alpha": ALPHA, "floor": FLOOR, "typical": TYPICAL, "lean_min": LEAN_MIN,
                         "w0": W0},
         "human_inputs_sha256": {k: v_[1] for k, v_ in HUMAN.items()},
         "tables": {"YN": {f: [round(float(x), 6) for x in t] for f, t in Y.items()},
                    "CH": {f: [round(float(x), 6) for x in t] for f, t in X.items()}, "colors": list(j5.COLORS)}}
    prim, g5 = primaries(Y, hum, B, gates_ok)
    raw = prim.pop("C2_raw_as_first_registered", None)
    v["gates"].update(g5)
    v["primaries"] = prim
    if secondary:
        v["secondaries"] = secondaries(Y, X, yn, ch, hum, B, smoke)
        v["secondaries"]["S12_C2_raw_as_first_registered"] = raw
    v["plain"] = []
    r1, r2 = prim["C1"], prim["C2"]
    if r1.get("r_dc") is not None:
        v["plain"].append(f"C1: {r1['branch']} ({r1['reading']}); r_dc {r1['r_dc']:.3f} "
                          f"CI [{r1['ci'][0]:.3f}, {r1['ci'][1]:.3f}] vs floor {FLOOR}, p {r1['p']:.4f}; "
                          f"tier: {r1['tier']}")
    else:
        v["plain"].append(f"C1: {r1['branch']}")
    if r2.get("w") is not None and "ci" in r2:
        v["plain"].append(f"C2: {r2['branch']} ({r2['reading']}); English weight on Jev's scale w {r2['w']:.3f} "
                          f"CI [{r2['ci'][0]:.3f}, {r2['ci'][1]:.3f}] vs {W0:.3f}, p {r2['p']:.4f}")
    else:
        v["plain"].append(f"C2: {r2['branch']}")
    return v


def summary(v):
    out = {"gates": {k: v["gates"][k] for k in v["gates"] if k != "G5_sealed_null"}, "counts": v["counts"],
           "plain": v["plain"]}
    g5 = v["gates"].get("G5_sealed_null")
    if g5:
        out["G5_sealed_null_r_dc"] = g5["r_dc"]
    if "secondaries" in v:
        s = v["secondaries"]
        out["S3"] = {k: s["S3_choice_format"].get(k) for k in ("r_dc", "ceiling", "efficiency", "yn_ch_top_agree")}
        out["S6_fun"] = s["S6_fun"]
        out["S7_shams"] = s["S7_shams"]
    return out
