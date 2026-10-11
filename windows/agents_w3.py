"""W3 planted agents (registration §7): synthetic harvest directories in the exact format of
windows/w3_spec.py (every array set written through `ship()`, plus gate_*.json, gdet.json and
train_*.json), generated from the activation worlds of windows/power/w3_collinear_check.py.

    make(agent, outdir, smoke=False, seed=SUITE_SEED, omit=()) -> Path

The World below is that script's World, copied with the feature dimension D as a parameter so
each site gets its own SITE_DIMS width (that script also imports torch through w3_design_check,
which nothing here needs). A model is one World per site plus one 14 wide for `out` (the hand
statistics of eps-hat), all sharing the per-image variables: z (the image) and zeta (the noise
draw), so the four inputs of a split image share them as the real 2 x 2 shares x0 and eps.

What each agent's models are:
  M-on      the agent's world (AGENTS below); its sham moves sham_k x the real label move along
            that move, with a random sign per image (the random unit vector u), plus the world's
            own per-input jitter.
  M-drop    the same world with its own draws; its no-label sets replace the label code by a
            fixed vector (e-null), same images, steps and noise as its labelled sets.
  M-off     a content world with no label code, reading the level a little better than the
            hand statistics (misread 0.2 SD against their 0.4).
  on0/off0  the untrained twins: unstructured worlds (small codes, large variation).
  hand      a content-only world, 14 wide, misread 0.4 SD (about the input floor); a property of
            the input, so the same numbers in every model's sets.
"""
import json
import zlib
from pathlib import Path

import numpy as np

from . import w3_spec as S

SUITE_SEED = 20261031
LS = S.LOGSNR
MU, SD = float(LS.mean()), float(LS.std())
N_TEST, N_TRAIN_IMG = 10000, 50000

# lam_a content code, lam_b label code, lam_e image variation, noise-draw scale, misread (per-image
# error of the content code's level, in SDs of log-SNR), special  (w3_collinear_check.WORLDS)
WORLDS = {
    "CONTENT":      (1.0, 0.0, 1.0, 1.0, 0.0, None),
    "LABEL":        (0.0, 1.0, 1.0, 1.0, 0.0, None),
    "EQUAL":        (1.0, 1.0, 1.0, 1.0, 0.0, None),
    "SMALL.15":     (1.0, 0.15, 2.0, 2.0, 0.0, None),
    "MISREAD.3":    (1.0, 0.3, 1.0, 1.0, 0.4, None),
    "INTERACT":     (1.0, 1.0, 1.0, 1.0, 0.0, "interact"),
    "COLLAPSE":     (1.0, 1.0, 1.0, 1.0, 0.0, "collapse"),
    "SCATTER":      (1.0, 1.0, 1.0, 1.0, 0.0, "scatter"),
    "BANDED":       (1.0, 1.0, 1.0, 1.0, 0.4, "banded"),
    # the floor and the twins (not in the collinearity check)
    "UNSTRUCTURED": (0.3, 0.3, 1.0, 1.0, 0.4, None),
    "OFF":          (1.0, 0.0, 1.0, 1.0, 0.2, None),
    "OFF0":         (0.3, 0.0, 1.0, 1.0, 0.4, None),
    "HAND":         (1.0, 0.0, 1.0, 1.0, 0.4, None),
    "X0HAT":        (0.5, 0.0, 1.0, 0.3, 0.0, None),
}

# registration §7; sham_k defaults to 0.05 (a quiet sham), gate_on to True
AGENTS = {
    "content":       {"world": "CONTENT"},
    "label":         {"world": "LABEL"},
    "equal":         {"world": "EQUAL"},
    "misread":       {"world": "MISREAD.3"},
    "collapse":      {"world": "COLLAPSE"},
    "scatter":       {"world": "SCATTER"},
    "banded":        {"world": "BANDED"},
    "interact":      {"world": "INTERACT"},
    "quiet-sham":    {"world": "LABEL", "sham_k": 0.05},
    "reactive-sham": {"world": "LABEL", "sham_k": 0.5},
    # a content network (a small label code, P1 CONTENT at rate 1.00) with a reactive sham: P2
    # would be NOT SILENT if judged, so "not judged" is the gate working, not a quiet sham
    "content+sham":  {"world": "SMALL.15", "sham_k": 0.5},
    "untrained":     {"world": "UNSTRUCTURED", "gate_on": False},
}


def basis(s):
    z = (s - MU) / SD
    return np.stack([z, np.tanh(z), np.tanh(2 * z - 1)], axis=1)


def step_of(s):
    return np.searchsorted(-LS, -s)


class World:
    """w3_collinear_check.World with D a parameter, the label code split out (for the sham and
    for e-null), and the same h()."""

    def __init__(self, rng, D, lam_a, lam_b, lam_e, noise, misread, special):
        self.D = D
        self.A = rng.normal(0, 1, (3, D)) * lam_a / np.sqrt(3)
        self.B = rng.normal(0, 1, (3, D)) * lam_b / np.sqrt(3)
        self.E = rng.normal(0, 1, (8, D)) * lam_e / np.sqrt(8)
        self.N = rng.normal(0, 1, (8, D)) * noise / np.sqrt(8)
        self.v = rng.normal(0, 1, 8)
        self.v /= np.linalg.norm(self.v)
        self.label_only, self.misread, self.special = lam_a == 0, misread, special
        self.junk = rng.normal(0, 1, D)
        self.null = basis(np.array([MU])) @ self.B          # e-null: one fixed label code

    def lab(self, s_l, s_c):
        lab = basis(s_l) @ self.B
        if self.special == "interact":
            lab = lab * (1 + np.tanh((s_c - MU) / SD))[:, None]
        if self.special == "banded":          # label code below step 500, content code above
            lab = lab * np.where(step_of(s_l) >= 500, 0.15, 1.0)[:, None]
        return lab

    def h(self, rng, s_c, s_l, z, zeta, mismatched=False, lab=None):
        s_read = s_c + self.misread * SD * (zeta @ self.v)
        con = basis(s_read) @ self.A
        if self.special == "banded":
            con = con * np.where(step_of(s_c) >= 500, 1.0, 0.15)[:, None]
        if lab is None:
            lab = self.lab(s_l, s_c)
        n = len(s_c)
        nz = np.ones(n) if self.label_only else 1 / (1 + np.exp(s_c / 2))
        out = con + lab + z @ self.E + nz[:, None] * (zeta @ self.N) + rng.normal(0, 0.05, (n, self.D))
        if mismatched and self.special == "collapse":
            out = np.tile(self.junk, (n, 1)) + rng.normal(0, 0.05, (n, self.D))
        if mismatched and self.special == "scatter":
            out = out + rng.normal(0, 1.0, (n, self.D))
        return out

    def sham(self, rng, s_t, s_p, z, zeta, k, sign):
        """Content at t, label embedding e(t) + |e(t') - e(t)| u: the label code moves k x the
        real label move along that move, sign per image."""
        l_t = self.lab(s_t, s_t)
        lab = l_t + k * sign[:, None] * (self.lab(s_p, s_t) - l_t)
        return self.h(rng, s_t, s_t, z, zeta, lab=lab)


class Net:
    """One model: a World per site (SITE_DIMS wide) and one for `out` (14 wide)."""

    def __init__(self, rng, world):
        p = WORLDS[world]
        self.w = {s: World(rng, S.SITE_DIMS[s], *p) for s in S.SITES}
        self.w["out"] = World(rng, 14, *p)

    def split(self, rng, L):
        a = {}
        for key, W in self.w.items():
            hs = [W.h(rng, L["s_t"], L["s_t"], L["z"], L["zeta"]),                    # (t, t)
                  W.h(rng, L["s_p"], L["s_p"], L["z"], L["zeta"]),                    # (t', t')
                  W.h(rng, L["s_p"], L["s_t"], L["z"], L["zeta"], mismatched=True),   # (t', t)
                  W.h(rng, L["s_t"], L["s_p"], L["z"], L["zeta"], mismatched=True)]   # (t, t')
            a[key] = np.stack(hs, axis=1).astype(np.float32)
        return a

    def sham(self, rng, L, k):
        return {key: W.sham(rng, L["s_t"], L["s_p"], L["z"], L["zeta"], k, L["sign"]).astype(np.float32)
                for key, W in self.w.items()}

    def matched(self, rng, L, null=False):
        return {key: W.h(rng, L["s"], L["s"], L["z"], L["zeta"],
                         lab=np.tile(W.null, (len(L["s"]), 1)) if null else None).astype(np.float16)
                for key, W in self.w.items()}


def _latents(rng, m):
    return rng.normal(0, 1, (m, 8)), rng.normal(0, 1, (m, 8))


def make(agent, outdir, smoke=False, seed=SUITE_SEED, omit=()):
    """Write agent `agent`'s synthetic harvest into `outdir` at the registered (or smoke) sizes.
    `omit` names sets or JSON files ("gate_drop.json") to leave out."""
    spec = AGENTS[agent]
    sz = S.sizes(smoke)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, zlib.crc32(agent.encode())])
    k_sham = spec.get("sham_k", 0.05)
    nets = {"on": Net(rng, spec["world"]), "drop": Net(rng, spec["world"]), "off": Net(rng, "OFF"),
            "on0": Net(rng, "UNSTRUCTURED"), "off0": Net(rng, "OFF0")}
    hand, x0hat = World(rng, 14, *WORLDS["HAND"]), World(rng, 14, *WORLDS["X0HAT"])

    def ship(arrays, name):
        if name not in omit:
            S.ship(arrays, name, out)

    # the split set (§3.2): 256 test images per offset class, one noise draw per image
    n = sz["N_PER_CLASS"]
    t = np.concatenate([rng.integers(S.BASE_MIN, S.T - o, n) for o in S.OFFSETS]).astype(np.int32)
    off = np.repeat(np.array(S.OFFSETS, np.int32), n)
    tp = t + off
    m = len(t)
    test_img = rng.permutation(N_TEST)
    z, zeta = _latents(rng, m)
    L = {"s_t": LS[t], "s_p": LS[tp], "z": z, "zeta": zeta, "sign": rng.choice([-1.0, 1.0], m)}
    ints = {"t": t, "tp": tp, "off": off, "img": test_img[:m].astype(np.int32)}
    hand_split = np.stack([hand.h(rng, L["s_t"], L["s_t"], z, zeta),
                           hand.h(rng, L["s_p"], L["s_p"], z, zeta)], axis=1).astype(np.float32)
    for tag in ("on", "drop", "on0"):
        ship({**ints, **nets[tag].split(rng, L), "hand": hand_split}, f"{tag}_split")
    for tag in ("on", "drop"):                                   # §3.3, rows in split order
        ship(nets[tag].sham(rng, L, k_sham), f"{tag}_sham")

    # the matched sets (§3.4): the same training images, steps and noise for every model
    train_img = rng.permutation(N_TRAIN_IMG)
    start = 0
    for kind, size in (("ptrain", sz["N_PTRAIN"]), ("pcal", sz["N_PCAL"])):
        ts = rng.integers(0, S.T, size).astype(np.int32)
        z, zeta = _latents(rng, size)
        Lm = {"s": LS[ts], "z": z, "zeta": zeta}
        base = {"t": ts, "img": train_img[start:start + size].astype(np.int32),
                "hand": hand.h(rng, Lm["s"], Lm["s"], z, zeta).astype(np.float32)}
        start += size
        for tag in ("on", "drop", "off", "on0", "off0"):
            if tag == "on0" and kind == "pcal":
                continue
            ship({**base, **nets[tag].matched(rng, Lm)}, f"{tag}_{kind}")
        ship({**base, **nets["drop"].matched(rng, Lm, null=True)}, f"drop_{kind}_null")

    # the seed-spread set (§3.6): four narrow bands, test images disjoint from the split set
    ns = sz["SPREAD_N"]
    band = np.repeat(np.arange(len(S.SPREAD_CENTRES), dtype=np.int32), ns)
    ts = (np.repeat(np.array(S.SPREAD_CENTRES), ns)
          + rng.integers(-S.SPREAD_HALF, S.SPREAD_HALF + 1, len(band))).astype(np.int32)
    z, zeta = _latents(rng, len(band))
    s = LS[ts]
    q, q2 = rng.normal(0, 1, 8) / np.sqrt(8), rng.normal(0, 1, 8) / np.sqrt(8)
    logvar = 0.6 * (z @ q) + 0.3 * (zeta @ q2) + 0.1 * (s - MU) / SD + rng.normal(0, 0.2, len(band))
    ship({"band": band, "t": ts, "img": test_img[m:m + len(band)].astype(np.int32),
          "mid": nets["on"].w["mid"].h(rng, s, s, z, zeta).astype(np.float32),
          "hand_xt": hand.h(rng, s, s, z, zeta).astype(np.float32),
          "hand_x0": x0hat.h(rng, s, s, z, zeta).astype(np.float32),
          "logvar": logvar.astype(np.float32)}, "on_spread")

    # gates and the training record
    epochs = sz["EPOCHS"]
    gate_bands = [{"steps": list(b), "mse": 0.02 + 0.01 * i, "bar": 0.05 + 0.01 * i}
                  for i, b in enumerate(S.GATE_BANDS)]
    js = {"gate_on": {"pass": spec.get("gate_on", True), "epochs": epochs, "bands": gate_bands},
          "gate_drop": {"pass": True, "epochs": epochs, "bands": gate_bands, "null_bands": gate_bands},
          "gate_off": {"pass": True, "epochs": epochs, "bands": gate_bands},
          "gdet": {"pass": True, "max_abs_diff": 0.0, "tol": S.GDET_TOL}}
    for tag in ("on", "drop", "off"):
        js[f"train_{tag}"] = {"epochs": epochs, "sec_per_step": 0.187,
                              "losses": [round(0.08 / (1 + e) + 0.03, 6) for e in range(epochs)]}
    for k, d in js.items():
        if f"{k}.json" not in omit:
            (out / f"{k}.json").write_text(json.dumps(d, indent=1, sort_keys=True))
    return out
