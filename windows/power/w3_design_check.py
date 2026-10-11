#!/usr/bin/env python3
"""W3 design check ("Does the loop know where it is?", ideation experiment 2): lane law,
BEFORE registration, BEFORE build. No trained diffusion model is touched here.

§1  How much of the clock is already in the input. Before asking whether a denoiser keeps its
    own clock, measure how well the noise level of x_t can be read from x_t itself: linear
    probes on (a) the raw pixels, (b) a dozen hand statistics of high-frequency energy, and
    (c) the pooled features of a RANDOM-INIT network of the flight's architecture class with no
    time input. (c) is the floor a clock-off claim has to clear: J1's input-only null, on this
    substrate. Reported per band of steps, because the input can't carry position everywhere
    (near t = 0 the noise hides under the image's own texture; near t = T every x_t is noise).

Data: MNIST train images (idx) and CIFAR-10 batch 1 (binary), both raw bytes read with numpy,
from --data (default ~/datasets/w3_raw). Two schedules: DDPM linear (1e-4 to 0.02, T = 1000)
and cosine (Nichol & Dhariwal 2021, s = 0.008).

Run:  ~/venvs/semcore/bin/python windows/power/w3_design_check.py [--data DIR] [--n 6000]
      [--sets mnist,cifar10]   (each dataset draws from its own seeded stream)
"""
import argparse
import gzip
import json
import os
import tarfile
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
W3_SEED = 20261011
T = 1000
BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 900),
         (900, 1000)]
ALPHAS = np.logspace(-3, 5, 17)


# ---- data ---------------------------------------------------------------------------------
def load_mnist(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    n, h, w = (int.from_bytes(raw[i:i + 4], "big") for i in (4, 8, 12))
    x = np.frombuffer(raw, np.uint8, offset=16).reshape(n, 1, h, w)
    return x


def load_cifar(path):
    with tarfile.open(path, "r:gz") as tf:
        member = tf.getmember("cifar-10-batches-bin/data_batch_1.bin")
        raw = tf.extractfile(member).read()
    rec = np.frombuffer(raw, np.uint8).reshape(-1, 1 + 3072)
    return rec[:, 1:].reshape(-1, 3, 32, 32)


def to_unit(x):
    return x.astype(np.float32) / 127.5 - 1.0


# ---- schedules ----------------------------------------------------------------------------
def alpha_bar(kind):
    if kind == "linear":
        return np.cumprod(1.0 - np.linspace(1e-4, 0.02, T))
    s = 0.008
    f = np.cos(((np.arange(T + 1) / T) + s) / (1 + s) * np.pi / 2) ** 2
    betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    return np.cumprod(1.0 - betas)


def noisy(x0, t, ab, rng):
    a = ab[t].reshape(-1, 1, 1, 1).astype(np.float32)
    eps = rng.standard_normal(x0.shape).astype(np.float32)
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps


# ---- features -----------------------------------------------------------------------------
def hand_features(x):
    """Per-image statistics of overall and high-frequency energy (channel-averaged)."""
    g = x.mean(axis=1)
    dx = np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2))
    lap = np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1]
                 - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]).mean(axis=(1, 2))
    p = np.abs(np.fft.fftshift(np.fft.fft2(g), axes=(1, 2))) ** 2
    h, w = g.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    edges = np.linspace(0, r.max() + 1e-6, 9)
    bins = [np.log(p[:, (r >= lo) & (r < hi)].mean(axis=1) + 1e-8)
            for lo, hi in zip(edges[:-1], edges[1:])]
    feats = [x.std(axis=(1, 2, 3)), np.abs(x).mean(axis=(1, 2, 3)), dx, dy, lap, *bins,
             bins[-1] - bins[0]]
    return np.stack(feats, axis=1)


class RandNet(torch.nn.Module):
    """The encoder half of a small DDPM U-Net (conv, GroupNorm, SiLU, two downsamples) up to
    its mid-block, randomly initialised, with no time input."""

    def __init__(self, cin, seed):
        super().__init__()
        torch.manual_seed(seed)

        def blk(a, b, stride):
            return torch.nn.Sequential(torch.nn.Conv2d(a, b, 3, stride, 1),
                                       torch.nn.GroupNorm(8, b), torch.nn.SiLU())
        self.net = torch.nn.Sequential(blk(cin, 64, 1), blk(64, 128, 2), blk(128, 256, 2),
                                       blk(256, 256, 1))

    @torch.no_grad()
    def forward(self, x, pool):
        out = []
        for i in range(0, len(x), 500):
            h = self.net(torch.from_numpy(x[i:i + 500]))
            f = [h.mean(dim=(2, 3))]
            if pool == "mean+std":
                f.append(h.std(dim=(2, 3)))
            out.append(torch.cat(f, dim=1).numpy())
        return np.concatenate(out)


# ---- probe --------------------------------------------------------------------------------
def probe(ftr, fte, ytr, yte):
    sc = StandardScaler().fit(ftr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(ftr), ytr)
    return m.predict(sc.transform(fte)), float(m.alpha_)


def step_of(logsnr_pred, logsnr_table):
    """Predicted log-SNR -> the nearest step of the schedule (log-SNR falls with t)."""
    idx = np.searchsorted(-logsnr_table, -logsnr_pred)
    idx = np.clip(idx, 1, T - 1)
    lo, hi = logsnr_table[idx - 1], logsnr_table[idx]
    return np.where(np.abs(logsnr_pred - lo) < np.abs(logsnr_pred - hi), idx - 1, idx)


def score(pred, t_te, ls_te, ls_table):
    st = step_of(pred, ls_table)
    err = np.abs(st - t_te)
    bands = {f"{lo}-{hi}": float(np.median(err[(t_te >= lo) & (t_te < hi)])) for lo, hi in BANDS}
    return {"r2_logsnr": float(r2_score(ls_te, pred)),
            "r2_step": float(r2_score(t_te, st)),
            "median_step_err": float(np.median(err)),
            "band_median_step_err": bands}


LOADERS = {"mnist": (load_mnist, "mnist-train-images.gz"),
           "cifar10": (load_cifar, "cifar-10-binary.tar.gz")}


def section1(data_dir, n, names, log):
    out = {}
    for dname in names:
        rng = np.random.default_rng([W3_SEED, list(LOADERS).index(dname)])
        load, fname = LOADERS[dname]
        raw = load(os.path.join(data_dir, fname))
        pick = rng.choice(len(raw), n, replace=False)
        x0 = to_unit(raw[pick])
        ntr = int(n * 2 / 3)
        for kind in ("linear", "cosine"):
            ab = alpha_bar(kind)
            ls_table = np.log(ab / (1 - ab))
            t = rng.integers(0, T, n)
            xt = noisy(x0, t, ab, rng)
            ls = ls_table[t]
            feats = {"pixels": xt.reshape(n, -1), "hand": hand_features(xt)}
            for seed in (1, 2, 3):
                net = RandNet(xt.shape[1], W3_SEED + seed)
                feats[f"randnet_mean_s{seed}"] = net(xt, "mean")
                if seed == 1:
                    feats["randnet_mean+std_s1"] = net(xt, "mean+std")
            for fname, f in feats.items():
                t0 = time.time()
                pred, alpha = probe(f[:ntr], f[ntr:], ls[:ntr], ls[ntr:])
                r = score(pred, t[ntr:], ls[ntr:], ls_table)
                r.update(dims=int(f.shape[1]), alpha=alpha, secs=round(time.time() - t0, 1))
                out[f"{dname}/{kind}/{fname}"] = r
                b = r["band_median_step_err"]
                log(f"{dname:8s} {kind:6s} {fname:20s} dims {f.shape[1]:5d}  "
                    f"R2(logSNR) {r['r2_logsnr']:.3f}  R2(step) {r['r2_step']:.3f}  "
                    f"median |err| {r['median_step_err']:5.0f} steps  by band: "
                    + " ".join(f"{v:4.0f}" for v in b.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets/w3_raw"))
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--sets", default="mnist,cifar10")
    args = ap.parse_args()
    names = args.sets.split(",")
    torch.set_num_threads(3)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("W3 design check — §1 input legibility of position")
    log(f"seed {W3_SEED}; n {args.n} images per dataset (2/3 train, 1/3 test; fresh noise); "
        f"T {T}; ridge alphas {ALPHAS[0]:g}..{ALPHAS[-1]:g}")
    log("bands (steps): " + " ".join(f"{lo}-{hi}" for lo, hi in BANDS))
    res = section1(args.data, args.n, names, log)
    tag = "_".join(names)
    with open(os.path.join(HERE, f"w3_design_check_s1_{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, f"w3_design_check_s1_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
