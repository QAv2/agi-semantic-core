#!/usr/bin/env python3
"""W3 split check: the statistics of the label-versus-content arm. Lane law: BEFORE
registration, BEFORE build. Simulation only; no network is touched.

The arm, recast after the 2026-10-11 prior-art sweep (docs/LOOP_PRIOR_ART_2026-10-11.md): a
clock-on denoiser is shown content at one noise level c with the label of another level l. A
linear probe trained only on matched inputs (c = l) reads the internal position. The split
weight w places the mismatched read between the network's own matched reads:

    w = slope through the origin of  r(c, l) - g(s_l)  on  g(s_c) - g(s_l)

where s is log-SNR and g is the probe's calibration curve, E[read | s] on matched inputs, fitted
by isotonic regression on a separate calibration split. w = 1: the read follows the content;
w = 0: it follows the label. Using g instead of the identity removes the probe's shrinkage. Using
the fitted g in the denominator, rather than each image's own matched reads, avoids
errors-in-variables attenuation, which would bias w toward "label".

Verdict (per layer; primary = the mid-block), from 95% bootstrap CIs over images:
    CONTENT  lower >= 0.8        LABEL  upper <= 0.2        SPLIT  0.2 < lower and upper < 0.8
    UNCLEAR  otherwise
Two gates first, because a network pushed off its training distribution may stop producing
reads that are positions at all, and the arm must not score that as a split:
    tracking gate  the joint fit  r = alpha + a g(s_c) + b g(s_l)  on the mismatched reads gives
                   the total tracking a + b (1 for any mix of content and label, 0 when the reads
                   ignore both); its CI lower bound must be >= TRACK_BAR, else UNREADABLE.
    precision gate the mismatched reads' residual SD around the fitted mix, over the matched
                   reads' residual SD around g, must be <= GATE, else UNREADABLE.
Level dependence: w is also fitted within four bands of the base step. If two bands get
different decisive verdicts with non-overlapping CIs, the verdict is VARIES BY LEVEL (and the
band verdicts are the finding); the precision gate is then judged within bands, because a
split that changes with level inflates the pooled residual by construction.

§2 planted worlds (teeth for the verdict): CONTENT, LABEL, SPLIT .5, the edges .3 and .7,
COLLAPSE (mismatched reads fall to the mean), SCATTER (mismatched reads lose precision 3x),
BANDED (content above step 500, label below) and BANDED300 (the boundary inside a band).
Grid: probe noise sigma in log-SNR units × images per offset class; offsets 25/100/300 steps.

§3 the label sham (Rulli et al. 2026's norm-matched random direction, in the label channel): the
label embedding moves by a random vector with the norm of the real move emb(t') - emb(t). A
deterministic network reads the same input identically, so every sham shift is caused by the
perturbation. Statistic, per offset class: S = mean|sham shift| / mean|real label-only shift|;
primary on the 300-step class (where the real move is large enough to measure against), 95%
bootstrap upper bound <= SHAM_BAR ⇒ SILENT. Informative only where the label moves the read
(w < 1); on a CONTENT verdict it is reported, not judged.
Worlds: SILENT, REACTIVE k (the sham moves the read k × as far as the real label move does).

Run:  ~/venvs/semcore/bin/python windows/power/w3_split_check.py
"""
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w3_design_check import T, alpha_bar  # noqa: E402

SEED = 20261012
OFFSETS = (25, 100, 300)
SHRINK = 0.9               # a ridge probe's slope of read on truth
N_CAL = 3000
REPS = 200
B = 500
GATE = 1.5
TRACK_BAR = 0.8
LO, HI = 0.2, 0.8
SHAM_BAR = 0.25
STEP_BANDS = ((20, 250), (250, 500), (500, 750), (750, T))
SIGMAS = (0.5, 1.0, 2.0)
NS = (64, 128, 256)
WORLDS = ("CONTENT", "LABEL", "SPLIT.5", "SPLIT.3", "SPLIT.7", "COLLAPSE", "SCATTER", "BANDED",
          "BANDED300")
EXPECT = {"CONTENT": "CONTENT", "LABEL": "LABEL", "SPLIT.5": "SPLIT", "SPLIT.3": "SPLIT",
          "SPLIT.7": "SPLIT", "COLLAPSE": "UNREADABLE", "SCATTER": "UNREADABLE",
          "BANDED": "VARIES", "BANDED300": "VARIES"}

LS = None


def g_true(s, mean):
    return mean + SHRINK * (s - mean)


def draw_pairs(rng, n):
    """n images per offset class: base step t, displaced step t' = t + offset."""
    t, tp, off = [], [], []
    for o in OFFSETS:
        b = rng.integers(20, T - o, n)
        t.append(b)
        tp.append(b + o)
        off.append(np.full(n, o))
    return np.concatenate(t), np.concatenate(tp), np.concatenate(off)


def mismatched_mean(world, sc, sl, base_step, mean):
    gc, gl = g_true(sc, mean), g_true(sl, mean)
    if world == "CONTENT":
        return gc
    if world == "LABEL":
        return gl
    if world.startswith("SPLIT"):
        w = float(world[5:])
        return w * gc + (1 - w) * gl
    if world == "COLLAPSE":
        return np.full_like(sc, mean)
    if world == "SCATTER":
        return gc
    if world == "BANDED":
        return np.where(base_step >= 500, gc, gl)
    if world == "BANDED300":
        return np.where(base_step >= 300, gc, gl)
    raise ValueError(world)


def experiment(rng, world, sigma, n, mean):
    # calibration split: matched reads of other images, uniform steps
    tc = rng.integers(0, T, N_CAL)
    rc = g_true(LS[tc], mean) + rng.normal(0, sigma, N_CAL)
    iso = IsotonicRegression(out_of_bounds="clip").fit(LS[tc], rc)
    g = iso.predict

    t, tp, off = draw_pairs(rng, n)
    m = len(t)
    s_t, s_tp = LS[t], LS[tp]
    # matched reads of the test images (the gate's reference)
    r_tt = g_true(s_t, mean) + rng.normal(0, sigma, m)
    r_pp = g_true(s_tp, mean) + rng.normal(0, sigma, m)
    sd_matched = np.std(np.concatenate([r_tt - g(s_t), r_pp - g(s_tp)]))
    # two mismatched reads per image: (content t', label t) and (content t, label t')
    noise = sigma * (3.0 if world == "SCATTER" else 1.0)
    sc = np.concatenate([s_tp, s_t])
    sl = np.concatenate([s_t, s_tp])
    base = np.concatenate([t, t])
    r = mismatched_mean(world, sc, sl, base, mean) + rng.normal(0, noise, 2 * m)
    y = r - g(sl)
    d = g(sc) - g(sl)
    C = rng.multinomial(m, np.full(m, 1 / m), size=B)
    img = np.concatenate([np.arange(m), np.arange(m)])
    pooled = split_fit(y, d, r, g(sc), g(sl), img, m, C, sd_matched)
    bands = []
    for lo_, hi_ in STEP_BANDS:
        sel_img = (t >= lo_) & (t < hi_)
        if sel_img.sum() < 8:
            bands.append(None)
            continue
        sel = np.concatenate([sel_img, sel_img])
        keep = np.flatnonzero(sel_img)
        remap = -np.ones(m, int)
        remap[keep] = np.arange(len(keep))
        Cb = rng.multinomial(len(keep), np.full(len(keep), 1 / len(keep)), size=B)
        bands.append(split_fit(y[sel], d[sel], r[sel], g(sc)[sel], g(sl)[sel], remap[img[sel]],
                               len(keep), Cb, sd_matched))
    out = dict(pooled)
    out["verdict"] = final_verdict(pooled, bands)
    out["bands"] = [None if b is None else round(b["w"], 3) for b in bands]
    return out


def split_fit(y, d, r, gc, gl, img, m, C, sd_matched):
    """w (slope of y on d through the origin), its CI; tracking a + b from the joint fit
    r ~ 1 + gc + gl, its CI; precision ratio. Bootstrap over images via per-image sums."""
    def per_image(v):
        return np.bincount(img, weights=v, minlength=m)
    a_i, b_i = per_image(y * d), per_image(d * d)
    w = a_i.sum() / b_i.sum()
    wb = (C @ a_i) / (C @ b_i)
    X = np.stack([np.ones_like(gc), gc, gl], axis=1)
    xx = np.stack([per_image(X[:, i] * X[:, j]) for i in range(3) for j in range(3)], axis=1)
    xy = np.stack([per_image(X[:, i] * r) for i in range(3)], axis=1)
    coef = np.linalg.solve(xx.sum(0).reshape(3, 3), xy.sum(0))
    XXb = (C @ xx).reshape(-1, 3, 3)
    XYb = C @ xy
    coefb = np.linalg.solve(XXb + 1e-9 * np.eye(3), XYb[..., None])[..., 0]
    track_b = coefb[:, 1] + coefb[:, 2]
    lo, hi = np.percentile(wb, [2.5, 97.5])
    return {"w": float(w), "lo": float(lo), "hi": float(hi),
            "track": float(coef[1] + coef[2]), "track_lo": float(np.percentile(track_b, 2.5)),
            "prec": float(np.std(y - w * d) / sd_matched)}


def decisive(f):
    if f["lo"] >= HI:
        return "CONTENT"
    if f["hi"] <= LO:
        return "LABEL"
    if f["lo"] > LO and f["hi"] < HI:
        return "SPLIT"
    return "UNCLEAR"


def final_verdict(pooled, bands):
    if pooled["track_lo"] < TRACK_BAR:
        return "UNREADABLE"
    # within a band the levels span little range, so a + b is loose there: bands are judged on
    # precision only, and tracking is a pooled property
    good = [b for b in bands if b is not None and b["prec"] <= GATE]
    calls = [(decisive(b), b) for b in good]
    calls = [(v, b) for v, b in calls if v != "UNCLEAR"]
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            (v1, b1), (v2, b2) = calls[i], calls[j]
            if v1 != v2 and (b1["hi"] < b2["lo"] or b2["hi"] < b1["lo"]):
                return "VARIES"
    if pooled["prec"] > GATE:
        return "UNREADABLE"
    return decisive(pooled)


def sham_experiment(rng, k, sigma_shift, n, mean, real_w):
    """Label-only moves on the same content: real (label t -> t') and sham (norm-matched random
    direction). Read shift under the real move = (1 - real_w) x (g(s_t') - g(s_t)); the sham
    moves it k x as far, sign random per image. Shift noise sigma_shift."""
    t, tp, off = draw_pairs(rng, n)
    m = len(t)
    full = g_true(LS[tp], mean) - g_true(LS[t], mean)
    real = (1 - real_w) * full + rng.normal(0, sigma_shift, m)
    sham = k * (1 - real_w) * full * rng.choice([-1, 1], m) + rng.normal(0, sigma_shift, m)
    out = {}
    for o in OFFSETS:
        sel = off == o
        a, b = np.abs(sham[sel]), np.abs(real[sel])
        C = rng.multinomial(sel.sum(), np.full(sel.sum(), 1 / sel.sum()), size=B)
        hi = np.percentile((C @ a) / (C @ b), 97.5)
        out[o] = {"S": float(a.sum() / b.sum()), "silent": bool(hi <= SHAM_BAR)}
    return out


def main():
    global LS
    rng = np.random.default_rng(SEED)
    ab = alpha_bar("cosine")
    LS = np.log(ab / (1 - ab))
    mean = LS.mean()
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("W3 split check — §2 the label-versus-content split weight, §3 the label sham")
    log(f"seed {SEED}; cosine schedule, T {T}, log-SNR {LS.min():.1f}..{LS.max():.1f} "
        f"(SD over uniform steps {LS.std():.2f}); offsets {OFFSETS} steps; "
        f"reps {REPS}, bootstrap {B}; precision gate {GATE}; tracking bar {TRACK_BAR}; "
        f"bars {LO}/{HI}; step bands {STEP_BANDS}")
    s1 = os.path.join(HERE, "w3_design_check_s1_mnist.json")
    if os.path.exists(s1):
        r1 = json.load(open(s1))
        for key in ("mnist/cosine/randnet_mean_s1", "mnist/cosine/hand"):
            r2 = r1[key]["r2_logsnr"]
            log(f"  §1 reference: {key} R2 {r2:.3f} ⇒ residual SD ≈ "
                f"{np.sqrt(1 - r2) * LS.std():.2f} log-SNR (input-level probe noise, no trained net)")
    log()
    log("§2 verdict rates over planted worlds (rows: world × sigma; columns: images per offset "
        "class, 3 classes)")
    res = {"s2": {}, "s3": {}}
    for world in WORLDS:
        log(f"  {world:9s} (should be {EXPECT[world]})")
        for sigma in SIGMAS:
            row = []
            for n in NS:
                runs = [experiment(rng, world, sigma, n, mean) for _ in range(REPS)]
                v = [x["verdict"] for x in runs]
                rates = {k: v.count(k) / REPS for k in
                         ("CONTENT", "LABEL", "SPLIT", "VARIES", "UNCLEAR", "UNREADABLE")}
                bands = np.array([[np.nan if b is None else b for b in x["bands"]] for x in runs])
                cell = {"rates": rates,
                        "w_median": float(np.median([x["w"] for x in runs])),
                        "ci_halfwidth_median": float(np.median([(x["hi"] - x["lo"]) / 2
                                                                for x in runs])),
                        "track_median": float(np.median([x["track"] for x in runs])),
                        "prec_median": float(np.median([x["prec"] for x in runs])),
                        "band_w_median": [float(v) for v in np.nanmedian(bands, axis=0)]}
                res["s2"][f"{world}/{sigma}/{n}"] = cell
                top = max(rates, key=rates.get)
                row.append(f"n{n:<3d} {top[:5]:5s} {rates[top]:4.2f} "
                           f"(w {cell['w_median']:+.2f} ±{cell['ci_halfwidth_median']:.2f} "
                           f"tr {cell['track_median']:.2f} p {cell['prec_median']:.1f})")
            log(f"    σ {sigma:4.2f}  " + " | ".join(row))
        c = res["s2"][f"{world}/1.0/128"]
        log("    band w at σ 1, n 128: " + " ".join(f"{v:+.2f}" for v in c["band_w_median"]))
    log()
    log("§3 label sham: P(SILENT) per offset class (95% upper bound of S <= "
        f"{SHAM_BAR}; primary = 300 steps), by k × real w × shift noise; n 128 per class")
    for real_w in (0.0, 0.5):
        for sig in (0.05, 0.25, 0.5):
            for k in (0.0, 0.1, 0.3, 0.5):
                runs = [sham_experiment(rng, k, sig, 128, mean, real_w) for _ in range(REPS)]
                cells = []
                for o in OFFSETS:
                    p = float(np.mean([x[o]["silent"] for x in runs]))
                    Sm = float(np.median([x[o]["S"] for x in runs]))
                    res["s3"][f"w{real_w}/sig{sig}/k{k}/off{o}"] = {"p_silent": p, "S_median": Sm}
                    cells.append(f"off {o:3d}: P {p:4.2f} (S {Sm:.2f})")
                log(f"  real w {real_w:.1f}  shift noise {sig:4.2f}  k {k:.1f}  " + " | ".join(cells))
    with open(os.path.join(HERE, "w3_split_check_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "w3_split_check_output.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
