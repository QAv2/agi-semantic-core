"""W3 "Does the loop know where it is?": the shared spec (docs/W3_LOOP_PROTOCOL.md).

Everything the Colab side (w3_model.py, w3_core.py, the notebook) and the local side
(analyze_w3.py, agents_w3.py) must agree on lives here: seeds, sizes, bars, sites, the noise
schedule, the input statistics, and the harvest's file format. numpy only, so it runs in both
places. The notebook embeds this file verbatim.

THE HARVEST FORMAT. Every array set is shipped by `ship()` as numbered .npz parts of at most
SHIP_CAP bytes (the ORC ship-size law: the Drive integration kills responses of ~7.5 MB of
base64), split along the first axis, plus one line per part in `manifest.json`. `load()` puts a
set back together and checks every part's sha256. Set names, by model tag (MODELS):

  {on,drop,on0}_split   768 test images (§3.2). Per-image ints t, tp, off, img; per site
                        float32 (n, 4, dim) in INPUT_ORDER; `out` float32 (n, 4, 14) = hand
                        statistics of the network's output eps-hat; `hand` float32 (n, 2, 14) =
                        hand statistics of the two noisy images x_t, x_t'.
  {on,drop}_sham        the same 768 images (§3.3), rows in the split set's order: `img`; per
                        site float32 (n, dim); `out` (n, 14).
  {M}_ptrain, {M}_pcal  the matched sets (§3.4): ints t, img; per site float16 (n, dim); `out`
                        float16 (n, 14); `hand` float32 (n, 14). For on, drop, off, on0 (ptrain
                        only), off0. M-drop also ships drop_ptrain_null / drop_pcal_null (e-null
                        in place of the label; same images, steps and noise as its labelled sets).
  on_spread             §3.6: ints band (the index 0-3 of SPREAD_CENTRES), t, img; `mid` float32
                        (n, 128); `hand_xt`, `hand_x0` float32 (n, 14); `logvar` float32 (n,).

All integer arrays are int32.

JSON beside the parts: gate_{on,drop,off}.json (§1.5; key "pass"), train_{on,drop,off}.json
(keys "epochs", "losses" per epoch, "sec_per_step"), gdet.json, env.json, code_sha.json, resume_test.json.
Model tags: on / drop / off are the trained models, on0 / off0 the untrained twins of M-on and
M-off (initialised under the TRAIN seed, never trained).
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

# ---- the registration's numbers ----------------------------------------------------------
SEEDS = {"TRAIN": 20261020, "ARM": 20261021, "PROBE": 20261022, "ANALYSIS": 20261023,
         "SHAM": 20261024, "SPREAD": 20261025, "GATE": 20261026, "SAMPLE": 20261027,
         "DROP": 20261028}
SMOKE_SEED_OFFSET = 100

T = 1000
WIDTH = 64
MULT = (1, 2, 2)
EMB_DIM = 4 * WIDTH
EPOCHS = 60
EPOCHS_MAX = 120               # one further try for a model that fails its gate (§1.5)
BATCH = 128
LR = 2e-4
WARMUP = 500
CLIP = 1.0
DROP_P = 0.2

OFFSETS = (25, 100, 300)
BASE_MIN = 20                  # base step t uniform on [BASE_MIN, T - offset)
N_PER_CLASS = 256
N_PTRAIN, N_PCAL = 4000, 3000
N_GATE = 2000
GATE_BANDS = ((0, 250), (250, 500), (500, 750), (750, 1000))
STEP_BANDS = ((20, 250), (250, 500), (500, 750), (750, 1000))
SPREAD_CENTRES = (100, 300, 500, 700)
SPREAD_HALF = 10
SPREAD_N = 64                  # states per band
SPREAD_K = 8                   # continuations per state
SPREAD_STEPS = 50
N_SAMPLES = 64                 # sample grid per model (training record)

LO, HI = 0.2, 0.8              # verdict bars
KAPPA_BAR = 0.5                # interaction gate (upper bound of kappa)
DELTA_VARIES = 0.2             # band shares must differ by at least this for VARIES
SHAM_BAR = 0.25
SHAM_CLASS = 300
BOOT = 10000
GDET_TOL = 1e-5
ALPHAS = np.logspace(-3, 5, 17)   # ridge penalties (w3_design_check.ALPHAS)
CV_FOLDS = 5

SITES = ("enc32", "enc16", "enc8", "mid", "dec8", "dec16", "dec32")
SITE_DIMS = {"enc32": WIDTH, "enc16": 2 * WIDTH, "enc8": 2 * WIDTH, "mid": 2 * WIDTH,
             "dec8": 2 * WIDTH, "dec16": 2 * WIDTH, "dec32": WIDTH}
PRIMARY_SITE = "mid"
# the four inputs of a split image, written (content step, label step)
INPUT_ORDER = ("tt", "pp", "pt", "tp")     # (t,t), (t',t'), (t',t), (t,t')
MODELS = ("on", "drop", "off", "on0", "off0")

SMOKE = {"EPOCHS": 2, "N_PER_CLASS": 32, "N_PTRAIN": 400, "N_PCAL": 300, "N_GATE": 400,
         "SPREAD_N": 8, "SPREAD_K": 4, "N_SAMPLES": 16, "BOOT": 1000}

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
SHIP_CAP = 3_800_000


def seeds(smoke=False):
    return {k: v + (SMOKE_SEED_OFFSET if smoke else 0) for k, v in SEEDS.items()}


def sizes(smoke=False):
    base = {"EPOCHS": EPOCHS, "N_PER_CLASS": N_PER_CLASS, "N_PTRAIN": N_PTRAIN, "N_PCAL": N_PCAL,
            "N_GATE": N_GATE, "SPREAD_N": SPREAD_N, "SPREAD_K": SPREAD_K, "N_SAMPLES": N_SAMPLES,
            "BOOT": BOOT}
    if smoke:
        base.update(SMOKE)
    return base


# ---- the noise schedule (verbatim from windows/power/w3_design_check.py) -------------------
def alpha_bar(kind="cosine"):
    if kind == "linear":
        return np.cumprod(1.0 - np.linspace(1e-4, 0.02, T))
    s = 0.008
    f = np.cos(((np.arange(T + 1) / T) + s) / (1 + s) * np.pi / 2) ** 2
    betas = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    return np.cumprod(1.0 - betas)


AB = alpha_bar("cosine")
LOGSNR = np.log(AB / (1 - AB))


# ---- input statistics (verbatim from windows/power/w3_design_check.py) ---------------------
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


# ---- shipping ------------------------------------------------------------------------------
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _npz_bytes(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def ship(arrays, name, outdir, cap=SHIP_CAP):
    """Write `arrays` (all sharing the first axis) as name.partNN.npz files of at most `cap`
    bytes, and append their lines to outdir/manifest.json. Returns the manifest lines."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = {len(v) for v in arrays.values()}
    assert len(n) == 1, f"{name}: arrays disagree on their first axis {n}"
    n = n.pop()
    row_bytes = sum(v[:1].nbytes for v in arrays.values()) or 1
    per = max(1, int(cap * 0.97) // row_bytes)
    lines, k, start = [], 0, 0
    while start < n or (n == 0 and k == 0):
        stop = min(n, start + per)
        b = _npz_bytes({key: v[start:stop] for key, v in arrays.items()})
        while len(b) > cap and stop - start > 1:          # npz headers: shrink until it fits
            stop = start + max(1, (stop - start) * 9 // 10)
            b = _npz_bytes({key: v[start:stop] for key, v in arrays.items()})
        fname = f"{name}.part{k:02d}.npz"
        (outdir / fname).write_bytes(b)
        lines.append({"set": name, "part": k, "file": fname, "rows": [start, stop],
                      "bytes": len(b), "sha256": sha256_bytes(b),
                      "arrays": {key: [str(v.dtype), list(v.shape[1:])] for key, v in arrays.items()}})
        k, start = k + 1, stop
        if n == 0:
            break
    man = outdir / "manifest.json"
    old = json.loads(man.read_text()) if man.exists() else []
    old = [x for x in old if x["set"] != name] + lines
    man.write_text(json.dumps(old, indent=1, sort_keys=True))
    return lines


def load(name, indir):
    """Reassemble a shipped set, checking each part's sha256 against the manifest."""
    indir = Path(indir)
    lines = sorted((x for x in json.loads((indir / "manifest.json").read_text()) if x["set"] == name),
                   key=lambda x: x["part"])
    if not lines:
        raise KeyError(f"set {name!r} is not in {indir / 'manifest.json'}")
    parts = []
    for x in lines:
        b = (indir / x["file"]).read_bytes()
        if sha256_bytes(b) != x["sha256"]:
            raise ValueError(f"{x['file']}: sha256 differs from the manifest")
        with np.load(io.BytesIO(b)) as z:
            parts.append({k: z[k] for k in z.files})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def has(name, indir):
    man = Path(indir) / "manifest.json"
    return man.exists() and any(x["set"] == name for x in json.loads(man.read_text()))
