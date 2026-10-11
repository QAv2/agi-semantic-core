#!/usr/bin/env python3
"""W3 collinearity check: what the probe's split weight w measures when the label and the content
are carried in different places, and the probe-free movement share that replaces it as the
primary. Lane law: BEFORE registration, BEFORE build. Simulation only; no network is touched.

The worry, found while writing the registration (2026-10-11): the probe of w3_split_check.py is
trained only on matched inputs, where the content level and the label level are the same number.
Features that track the label and features that track the content are then perfectly collinear in
the probe's training data, and ridge regression divides its weight between them by how cleanly
each predicts the target. A label code is exact; a content code reads the level with some
per-image error (design check §1: fourteen input statistics leave R2 .84 of log-SNR on CIFAR-10).
The probe can then lean on the label while the representation moves mostly with the content.

§1 plants activation worlds and compares, on the same 2 x 2 inputs per image:

    w      the probe statistic of w3_split_check (split_fit), "where the position read lands"
    w_rep  the movement share, probe-free. Features are standardised by their SD over matched
           training inputs. Per image, with c = l = t and c' = l' = t',
               direction 1 (reference (t, t)):    dc = h(t', t) - h(t, t)    db = h(t', t') - h(t, t)
               direction 2 (reference (t', t')):  dc = h(t, t') - h(t', t')  db = h(t, t) - h(t', t')
           where h(content, label); w_rep = sum <dc, db> / sum <db, db> over both directions.
           Pooled over the two directions, w_rep + l_rep = 1 exactly (the 2 x 2 interaction
           enters the two directions with opposite signs), so w_rep is the content's Shapley share
           of the network's own move from t to t', measured along that move.

§2 the movement share's verdict rules and their teeth. Gates, then bands, as in split check:
    interaction gate  kappa = sum |I|^2 / sum |db|^2 with I = h(t', t') - h(t', t) - h(t, t') +
                      h(t, t), the 2 x 2 interaction. 0 for a network whose content and label add;
                      large when mismatched inputs leave the plane of the matched ones (collapse,
                      scatter). The 95% upper bound must be <= KAPPA_BAR, else UNREADABLE.
    level check       w_rep within four bands of the base step; two decisive band verdicts that
                      differ, with non-overlapping CIs and shares at least DELTA_VARIES apart,
                      give VARIES BY LEVEL. (The floor was added after the first run: the
                      movement share's intervals are tight, and a world whose true share sat on
                      the 0.2 bar was called VARIES for band shares of .16 and .24. A floor of 0.3,
                      tried second, cost BANDED its genuine .72-versus-.98 contrast; 0.2, the
                      width of the LABEL zone, is the floor.)
    verdict           CONTENT lower >= .8 · LABEL upper <= .2 · SPLIT inside (.2, .8) · UNCLEAR.

Run:  ~/venvs/semcore/bin/python windows/power/w3_collinear_check.py
"""
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w3_split_check as sc  # noqa: E402
from w3_design_check import ALPHAS, T, alpha_bar  # noqa: E402

SEED = 20261013
D = 64
N_TR, N_CAL = 4000, 3000
REPS, B = 100, 500
KAPPA_BAR = 0.5
DELTA_VARIES = 0.2        # band shares must also differ by this much (the movement share's CIs are tight)
STEP_BANDS = sc.STEP_BANDS
NS = (128, 256)
# lam_A content code, lam_B label code, lam_E image variation, noise-draw scale, misread (per-image
# error of the content code's level, in SDs of log-SNR), special
WORLDS = {
    "CONTENT":      (1.0, 0.0, 1.0, 1.0, 0.0, None),
    "LABEL":        (0.0, 1.0, 1.0, 1.0, 0.0, None),
    "EQUAL":        (1.0, 1.0, 1.0, 1.0, 0.0, None),
    "SMALL.15":     (1.0, 0.15, 2.0, 2.0, 0.0, None),
    "SMALL.3":      (1.0, 0.3, 2.0, 2.0, 0.0, None),
    "MISREAD.15":   (1.0, 0.15, 1.0, 1.0, 0.4, None),
    "MISREAD.3":    (1.0, 0.3, 1.0, 1.0, 0.4, None),
    "MISREAD.3-lo": (1.0, 0.3, 1.0, 1.0, 0.2, None),
    "MISREAD-EQ":   (1.0, 1.0, 1.0, 1.0, 0.4, None),
    "MISREAD-LAB":  (0.15, 1.0, 1.0, 0.5, 0.4, None),
    "INTERACT":     (1.0, 1.0, 1.0, 1.0, 0.0, "interact"),
    "COLLAPSE":     (1.0, 1.0, 1.0, 1.0, 0.0, "collapse"),
    "SCATTER":      (1.0, 1.0, 1.0, 1.0, 0.0, "scatter"),
    "BANDED":       (1.0, 1.0, 1.0, 1.0, 0.4, "banded"),
}
EXPECT = {"CONTENT": "CONTENT", "LABEL": "LABEL", "EQUAL": "SPLIT", "SMALL.15": "CONTENT",
          "SMALL.3": "CONTENT", "MISREAD.15": "CONTENT", "MISREAD.3": "CONTENT",
          "MISREAD.3-lo": "CONTENT", "MISREAD-EQ": "SPLIT", "MISREAD-LAB": "LABEL",
          "INTERACT": "VARIES", "COLLAPSE": "UNREADABLE", "SCATTER": "UNREADABLE",
          "BANDED": "VARIES"}


def basis(s):
    z = (s - MU) / SD
    return np.stack([z, np.tanh(z), np.tanh(2 * z - 1)], axis=1)


def step_of(s):
    return np.searchsorted(-LS, -s)


class World:
    def __init__(self, rng, lam_a, lam_b, lam_e, noise, misread, special):
        self.A = rng.normal(0, 1, (3, D)) * lam_a / np.sqrt(3)
        self.B = rng.normal(0, 1, (3, D)) * lam_b / np.sqrt(3)
        self.E = rng.normal(0, 1, (8, D)) * lam_e / np.sqrt(8)
        self.N = rng.normal(0, 1, (8, D)) * noise / np.sqrt(8)
        self.v = rng.normal(0, 1, 8)
        self.v /= np.linalg.norm(self.v)
        self.label_only, self.misread, self.special = lam_a == 0, misread, special
        self.junk = rng.normal(0, 1, D)

    def h(self, rng, s_c, s_l, z, zeta, mismatched=False):
        s_read = s_c + self.misread * SD * (zeta @ self.v)
        con, lab = basis(s_read) @ self.A, basis(s_l) @ self.B
        if self.special == "interact":
            lab = lab * (1 + np.tanh((s_c - MU) / SD))[:, None]
        if self.special == "banded":          # label code below step 500, content code above
            con = con * np.where(step_of(s_c) >= 500, 1.0, 0.15)[:, None]
            lab = lab * np.where(step_of(s_l) >= 500, 0.15, 1.0)[:, None]
        nz = np.ones(len(s_c)) if self.label_only else 1 / (1 + np.exp(s_c / 2))
        out = con + lab + z @ self.E + nz[:, None] * (zeta @ self.N) + rng.normal(0, 0.05, (len(s_c), D))
        if mismatched and self.special == "collapse":
            out = np.tile(self.junk, (len(s_c), 1)) + rng.normal(0, 0.05, (len(s_c), D))
        if mismatched and self.special == "scatter":
            out = out + rng.normal(0, 1.0, (len(s_c), D))
        return out


def boot(rng, m, num, den):
    C = rng.multinomial(m, np.full(m, 1 / m), size=B)
    return (C @ num) / (C @ den)


def movement(rng, h_tt, h_pp, h_pt, h_tp, sd, sel=None):
    if sel is not None:
        h_tt, h_pp, h_pt, h_tp = h_tt[sel], h_pp[sel], h_pt[sel], h_tp[sel]
    u = lambda a: a / sd  # noqa: E731
    db1, db2 = u(h_pp - h_tt), u(h_tt - h_pp)
    a_c = (u(h_pt - h_tt) * db1).sum(1) + (u(h_tp - h_pp) * db2).sum(1)
    a_l = (u(h_tp - h_tt) * db1).sum(1) + (u(h_pt - h_pp) * db2).sum(1)
    a_b = 2 * (db1 * db1).sum(1)
    inter = u(h_pp - h_pt - h_tp + h_tt)
    a_i = 2 * (inter * inter).sum(1)
    m = len(a_b)
    wb, kb = boot(rng, m, a_c, a_b), boot(rng, m, a_i, a_b)
    return {"w": float(a_c.sum() / a_b.sum()), "l": float(a_l.sum() / a_b.sum()),
            "lo": float(np.percentile(wb, 2.5)), "hi": float(np.percentile(wb, 97.5)),
            "kappa": float(a_i.sum() / a_b.sum()), "kappa_hi": float(np.percentile(kb, 97.5))}


def movement_verdict(pooled, bands):
    if pooled["kappa_hi"] > KAPPA_BAR:
        return "UNREADABLE"
    calls = [(sc.decisive(b), b) for b in bands if b is not None]
    calls = [(v, b) for v, b in calls if v != "UNCLEAR"]
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            (v1, b1), (v2, b2) = calls[i], calls[j]
            if v1 != v2 and (b1["hi"] < b2["lo"] or b2["hi"] < b1["lo"]) \
                    and abs(b1["w"] - b2["w"]) >= DELTA_VARIES:
                return "VARIES"
    return sc.decisive(pooled)


def experiment(rng, world, n):
    W = World(rng, *WORLDS[world])
    rnd = lambda k: (rng.normal(0, 1, (k, 8)), rng.normal(0, 1, (k, 8)))  # noqa: E731
    ttr = rng.integers(0, T, N_TR)
    htr = W.h(rng, LS[ttr], LS[ttr], *rnd(N_TR))
    scl = StandardScaler().fit(htr)
    pr = RidgeCV(alphas=ALPHAS).fit(scl.transform(htr), LS[ttr])
    read = lambda h: pr.predict(scl.transform(h))  # noqa: E731
    tc = rng.integers(0, T, N_CAL)
    g = IsotonicRegression(out_of_bounds="clip").fit(LS[tc], read(W.h(rng, LS[tc], LS[tc], *rnd(N_CAL)))).predict
    t, tp, _ = sc.draw_pairs(rng, n)
    m = len(t)
    z, zeta = rnd(m)
    s_t, s_p = LS[t], LS[tp]
    h_tt, h_pp = W.h(rng, s_t, s_t, z, zeta), W.h(rng, s_p, s_p, z, zeta)
    h_pt = W.h(rng, s_p, s_t, z, zeta, mismatched=True)     # (content t', label t)
    h_tp = W.h(rng, s_t, s_p, z, zeta, mismatched=True)     # (content t, label t')
    # the probe statistic, with its tracking gate
    r_tt, r_pp = read(h_tt), read(h_pp)
    sd_matched = np.std(np.concatenate([r_tt - g(s_t), r_pp - g(s_p)]))
    scn, sln = np.concatenate([s_p, s_t]), np.concatenate([s_t, s_p])
    r = np.concatenate([read(h_pt), read(h_tp)])
    img = np.concatenate([np.arange(m), np.arange(m)])
    C = rng.multinomial(m, np.full(m, 1 / m), size=B)
    f = sc.split_fit(r - g(sln), g(scn) - g(sln), r, g(scn), g(sln), img, m, C, sd_matched)
    probe_v = "UNREADABLE" if (f["track_lo"] < sc.TRACK_BAR or f["prec"] > sc.GATE) else sc.decisive(f)
    # the movement share
    sd = htr.std(axis=0)
    pooled = movement(rng, h_tt, h_pp, h_pt, h_tp, sd)
    bands = []
    for lo_, hi_ in STEP_BANDS:
        sel = (t >= lo_) & (t < hi_)
        bands.append(movement(rng, h_tt, h_pp, h_pt, h_tp, sd, sel) if sel.sum() >= 8 else None)
    return {"w": f["w"], "track": f["track"], "probe": probe_v, "w_rep": pooled["w"],
            "l_rep": pooled["l"], "kappa": pooled["kappa"], "rep": movement_verdict(pooled, bands),
            "band_w": [None if b is None else b["w"] for b in bands]}


def main():
    global LS, MU, SD
    ab = alpha_bar("cosine")
    LS = np.log(ab / (1 - ab))
    MU, SD = LS.mean(), LS.std()
    sc.LS = LS
    rng = np.random.default_rng(SEED)
    lines, res = [], {}

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("W3 collinearity check — §1 the probe's w against the probe-free movement share w_rep; "
        "§2 the movement share's verdict rates")
    log(f"seed {SEED}; activation dim {D}; probe train {N_TR}, calibration {N_CAL}; reps {REPS}, "
        f"bootstrap {B}; bars {sc.LO}/{sc.HI}; probe gates: tracking {sc.TRACK_BAR}, precision "
        f"{sc.GATE}; movement gate: kappa upper bound <= {KAPPA_BAR}; VARIES floor {DELTA_VARIES}; step bands {STEP_BANDS}")
    for world in WORLDS:
        log(f"  {world:12s} (movement should be {EXPECT[world]})")
        for n in NS:
            runs = [experiment(rng, world, n) for _ in range(REPS)]
            med = {k: float(np.median([x[k] for x in runs])) for k in ("w", "track", "w_rep", "l_rep", "kappa")}
            bw = np.array([[np.nan if v is None else v for v in x["band_w"]] for x in runs])
            med["band_w_rep"] = [float(v) for v in np.nanmedian(bw, axis=0)]
            pv, rv = [x["probe"] for x in runs], [x["rep"] for x in runs]
            pr = {k: pv.count(k) / REPS for k in sorted(set(pv))}
            rr = {k: rv.count(k) / REPS for k in sorted(set(rv))}
            res[f"{world}/{n}"] = {"median": med, "probe_verdicts": pr, "movement_verdicts": rr,
                                   "movement_right": rr.get(EXPECT[world], 0.0)}
            log(f"    n {n:3d}  probe w {med['w']:+.2f} {pr}")
            log(f"           w_rep {med['w_rep']:+.2f} (l_rep {med['l_rep']:+.2f}, kappa {med['kappa']:.3f}, "
                f"bands {' '.join(f'{v:+.2f}' for v in med['band_w_rep'])}) {rr}  "
                f"RIGHT {rr.get(EXPECT[world], 0.0):.2f}")
    with open(os.path.join(HERE, "w3_collinear_check_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "w3_collinear_check_output.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
