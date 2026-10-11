"""W3 Colab-side core (docs/W3_LOOP_PROTOCOL.md §1, §3, §4): data, the deterministic batch stream,
training with checkpoints and resume, the training gate, the four input sets, the harvest, the
sampler and the seed-spread continuations, G-det, and the resume self-test. The notebook
(colab/W3_LOOP_UI.ipynb) embeds this file, w3_model.py and w3_spec.py verbatim and calls run().
Everything here also runs on a CPU, which is how the training suite tests it.
"""
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import w3_spec as S
from .w3_model import make_model

TAGS = ("on", "drop", "off")


# ---- data ----------------------------------------------------------------------------------
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cifar(path):
    """-> (train, test) as float32 arrays in [-1, 1], shapes (50000, 3, 32, 32), (10000, ...)."""
    def parse(raw):
        rec = np.frombuffer(raw, np.uint8).reshape(-1, 1 + 3072)
        return rec[:, 1:].reshape(-1, 3, 32, 32)
    with tarfile.open(path, "r:gz") as tf:
        tr = [parse(tf.extractfile(f"cifar-10-batches-bin/data_batch_{i}.bin").read()) for i in range(1, 6)]
        te = parse(tf.extractfile("cifar-10-batches-bin/test_batch.bin").read())
    unit = lambda a: a.astype(np.float32) / 127.5 - 1.0  # noqa: E731
    return unit(np.concatenate(tr)), unit(te)


def noisy(x0, t, eps):
    """x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps (numpy, per image)."""
    a = S.AB[t].reshape(-1, 1, 1, 1).astype(np.float32)
    return (np.sqrt(a) * x0 + np.sqrt(1 - a) * eps).astype(np.float32)


# ---- the batch stream (identical for all three models; resumable at any epoch) -------------
def epoch_batches(seed, epoch, n_train, batch=S.BATCH):
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch))
    perm = torch.randperm(n_train, generator=g)
    for i in range(n_train // batch):
        idx = perm[i * batch:(i + 1) * batch]
        flip = torch.rand(batch, generator=g) < 0.5
        t = torch.randint(0, S.T, (batch,), generator=g)
        eps = torch.randn(batch, 3, 32, 32, generator=g)
        yield idx, flip, t, eps


def drop_masks(seed, epoch, steps, batch=S.BATCH):
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch))
    return torch.rand(steps, batch, generator=g) < S.DROP_P


def amp_ctx(device):
    """fp16 autocast on CUDA (as timed); off elsewhere (CPU tests, the meta-device check)."""
    cuda = device.type == "cuda"
    return torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.float16, enabled=cuda)


def lr_at(step):
    return S.LR * min(1.0, (step + 1) / S.WARMUP)


# ---- training ------------------------------------------------------------------------------
def _atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def new_state(tag, device, C=S.WIDTH, seed=S.SEEDS["TRAIN"]):
    model = make_model(tag, C=C, mult=S.MULT, seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=S.LR, betas=(0.9, 0.999), weight_decay=0.0)
    cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if cuda else "cpu", enabled=cuda)
    return {"model": model, "opt": opt, "scaler": scaler, "epoch": 0, "losses": [], "sec_per_step": []}


def state_dict(st):
    return {"model": st["model"].state_dict(), "opt": st["opt"].state_dict(),
            "scaler": st["scaler"].state_dict(), "epoch": st["epoch"], "losses": st["losses"],
            "sec_per_step": st["sec_per_step"]}


def load_state(st, path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    st["model"].load_state_dict(d["model"])
    st["opt"].load_state_dict(d["opt"])
    st["scaler"].load_state_dict(d["scaler"])
    st["epoch"], st["losses"], st["sec_per_step"] = d["epoch"], d["losses"], d["sec_per_step"]
    return st


def train_epochs(st, tag, X, until, seeds, device, ckpt=None, drive_ckpt=None, drive_every=5,
                 log=print, n_train=None, batch=S.BATCH):
    """Train st['model'] from st['epoch'] to `until` epochs on X (a float32 tensor on `device`).
    Writes a local checkpoint every epoch and a Drive copy every `drive_every` epochs and at the
    end, so a dropped runtime resumes from the last Drive copy."""
    model, opt, scaler = st["model"], st["opt"], st["scaler"]
    n_train = n_train or len(X)
    steps = n_train // batch
    ab = torch.tensor(S.AB, dtype=torch.float32, device=device)
    model.train()
    while st["epoch"] < until:
        e = st["epoch"]
        masks = drop_masks(seeds["DROP"], e, steps, batch) if tag == "drop" else None
        t0, tot = time.time(), torch.zeros((), device=device)
        for i, (idx, flip, t, eps) in enumerate(epoch_batches(seeds["TRAIN"], e, n_train, batch)):
            gstep = e * steps + i
            for pg in opt.param_groups:
                pg["lr"] = lr_at(gstep)
            x0 = X[idx.to(device)]
            x0 = torch.where(flip.to(device)[:, None, None, None], x0.flip(3), x0)
            t, eps = t.to(device), eps.to(device)
            a = ab[t][:, None, None, None]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            nm = masks[i].to(device) if masks is not None else None
            with amp_ctx(device):
                out = model(xt, t, null_mask=nm)
            loss = F.mse_loss(out.float(), eps)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), S.CLIP)
            scaler.step(opt)
            scaler.update()
            tot = tot + loss.detach()        # no per-step sync: read once per epoch
        if device.type == "cuda":
            torch.cuda.synchronize()
        st["losses"].append(float(tot) / steps)
        st["sec_per_step"].append((time.time() - t0) / steps)
        st["epoch"] = e + 1
        log(f"  {tag} epoch {e + 1}/{until}: loss {st['losses'][-1]:.4f}, {st['sec_per_step'][-1]:.3f} s/step")
        if ckpt:
            _atomic_save(state_dict(st), ckpt)
            if drive_ckpt and (st["epoch"] % drive_every == 0 or st["epoch"] == until):
                shutil.copyfile(ckpt, str(drive_ckpt) + ".tmp")
                os.replace(str(drive_ckpt) + ".tmp", drive_ckpt)
    model.eval()
    return st


# ---- the training gate (§1.5) ---------------------------------------------------------------
@torch.no_grad()
def eps_hat(model, x, t=None, emb=None, null=False, device=None, bs=256):
    device = device or next(model.parameters()).device
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        tb = torch.from_numpy(np.asarray(t[i:i + bs])).long().to(device) if t is not None else None
        nm = torch.ones(len(xb), dtype=torch.bool, device=device) if null else None
        if null:
            tb = torch.zeros(len(xb), dtype=torch.long, device=device)
        outs.append(model(xb, tb, emb=None if emb is None else emb[i:i + bs].to(device), null_mask=nm).float().cpu().numpy())
    return np.concatenate(outs)


def gate(model, Xte, v, seeds, n, null=False):
    """Mean squared error on eps by band of steps, against the best fixed multiple of the noisy
    input, lambda*(t) x_t with lambda* = sqrt(1 - ab) / (ab v + 1 - ab)."""
    rng = np.random.default_rng(seeds["GATE"])
    idx = rng.choice(len(Xte), n, replace=False)
    t = rng.integers(0, S.T, n)
    eps = rng.standard_normal((n, 3, 32, 32), dtype=np.float32)
    xt = noisy(Xte[idx], t, eps)
    model.eval()
    pred = eps_hat(model, xt, t if model.clock else None, null=null)
    a = S.AB[t]
    lam = (np.sqrt(1 - a) / (a * v + 1 - a)).astype(np.float32).reshape(-1, 1, 1, 1)
    mse_m = ((pred - eps) ** 2).mean(axis=(1, 2, 3))
    mse_l = ((lam * xt - eps) ** 2).mean(axis=(1, 2, 3))
    bands = []
    for lo, hi in S.GATE_BANDS:
        sel = (t >= lo) & (t < hi)
        bands.append({"steps": [lo, hi], "n": int(sel.sum()), "mse_model": float(mse_m[sel].mean()),
                      "mse_best_multiple": float(mse_l[sel].mean()),
                      "pass": bool(mse_m[sel].mean() < mse_l[sel].mean())})
    return {"bands": bands, "pass": all(b["pass"] for b in bands), "v": float(v), "n": int(n),
            "null_label": bool(null)}


# ---- the input sets (§3) ---------------------------------------------------------------------
def split_set(seeds, npc, n_pool=10000):
    rng = np.random.default_rng(seeds["ARM"])
    n = npc * len(S.OFFSETS)
    img = rng.choice(n_pool, n, replace=False)
    off = np.repeat(np.array(S.OFFSETS), npc)
    t = np.concatenate([rng.integers(S.BASE_MIN, S.T - o, npc) for o in S.OFFSETS])
    eps = rng.standard_normal((n, 3, 32, 32), dtype=np.float32)
    return {"img": img.astype(np.int32), "off": off.astype(np.int32), "t": t.astype(np.int32),
            "tp": (t + off).astype(np.int32), "eps": eps}


def sham_dirs(seeds, n, dim=S.EMB_DIM):
    rng = np.random.default_rng(seeds["SHAM"])
    u = rng.standard_normal((n, dim))
    return (u / np.linalg.norm(u, axis=1, keepdims=True)).astype(np.float32)


def probe_sets(seeds, n_tr, n_cal, n_pool=50000):
    rng = np.random.default_rng(seeds["PROBE"])
    img = rng.choice(n_pool, n_tr + n_cal, replace=False)
    t = rng.integers(0, S.T, n_tr + n_cal)
    eps = rng.standard_normal((n_tr + n_cal, 3, 32, 32), dtype=np.float32)
    cut = n_tr
    return ({"img": img[:cut].astype(np.int32), "t": t[:cut].astype(np.int32), "eps": eps[:cut]},
            {"img": img[cut:].astype(np.int32), "t": t[cut:].astype(np.int32), "eps": eps[cut:]})


def spread_set(seeds, n_per_band, exclude, n_pool=10000):
    rng = np.random.default_rng(seeds["SPREAD"])
    cand = np.setdiff1d(np.arange(n_pool), exclude)
    n = n_per_band * len(S.SPREAD_CENTRES)
    img = rng.choice(cand, n, replace=False)
    band = np.repeat(np.arange(len(S.SPREAD_CENTRES)), n_per_band)
    t = np.array(S.SPREAD_CENTRES)[band] + rng.integers(-S.SPREAD_HALF, S.SPREAD_HALF + 1, n)
    eps = rng.standard_normal((n, 3, 32, 32), dtype=np.float32)
    return {"img": img.astype(np.int32), "band": band.astype(np.int32), "t": t.astype(np.int32), "eps": eps}


# ---- harvest ---------------------------------------------------------------------------------
@torch.no_grad()
def features(model, x, t=None, emb=None, null=False, bs=256):
    """Site features (mean-pooled, fp32) and the 14 hand statistics of eps-hat."""
    device = next(model.parameters()).device
    model.eval()
    feats = {s: [] for s in S.SITES}
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        tb = torch.from_numpy(np.asarray(t[i:i + bs])).long().to(device) if t is not None else None
        nm = None
        if null:
            nm = torch.ones(len(xb), dtype=torch.bool, device=device)
            tb = torch.zeros(len(xb), dtype=torch.long, device=device)
        eb = None if emb is None else emb[i:i + bs].to(device)
        out, rec = model(xb, tb, emb=eb, null_mask=nm, feats=True)
        for s in S.SITES:
            feats[s].append(rec[s].float().cpu().numpy())
        outs.append(S.hand_features(out.float().cpu().numpy()))
    return {s: np.concatenate(v) for s, v in feats.items()}, np.concatenate(outs).astype(np.float32)


def harvest_split(model, X0, sp):
    """The 2 x 2 per image: (t,t), (t',t'), (t',t), (t,t') in S.INPUT_ORDER."""
    x0 = X0[sp["img"]]
    xt, xp = noisy(x0, sp["t"], sp["eps"]), noisy(x0, sp["tp"], sp["eps"])
    lab = model.clock
    inputs = {"tt": (xt, sp["t"]), "pp": (xp, sp["tp"]), "pt": (xp, sp["t"]), "tp": (xt, sp["tp"])}
    got = {k: features(model, xx, tt if lab else None) for k, (xx, tt) in inputs.items()}
    arr = {k: sp[k] for k in ("t", "tp", "off", "img")}
    for s in S.SITES:
        arr[s] = np.stack([got[k][0][s] for k in S.INPUT_ORDER], axis=1).astype(np.float32)
    arr["out"] = np.stack([got[k][1] for k in S.INPUT_ORDER], axis=1).astype(np.float32)
    arr["hand"] = np.stack([S.hand_features(xt), S.hand_features(xp)], axis=1).astype(np.float32)
    return arr


@torch.no_grad()
def harvest_sham(model, X0, sp, u):
    device = next(model.parameters()).device
    x0 = X0[sp["img"]]
    xt = noisy(x0, sp["t"], sp["eps"])
    e_t = model.embed(torch.from_numpy(sp["t"]).long().to(device)).float()
    e_p = model.embed(torch.from_numpy(sp["tp"]).long().to(device)).float()
    assert u.shape == tuple(e_t.shape), f"sham directions {u.shape} vs embeddings {tuple(e_t.shape)}"
    emb = e_t + (e_p - e_t).norm(dim=1, keepdim=True) * torch.from_numpy(u).to(device)
    f, out = features(model, xt, emb=emb.cpu())
    arr = {s: f[s].astype(np.float32) for s in S.SITES}
    arr["out"] = out
    arr["img"] = sp["img"]               # rows follow the split set's order; analysis checks it
    return arr


def harvest_matched(model, X0, ps, null=False):
    x = noisy(X0[ps["img"]], ps["t"], ps["eps"])
    f, out = features(model, x, ps["t"] if model.clock and not null else None, null=null)
    arr = {"t": ps["t"], "img": ps["img"]}
    for s in S.SITES:
        arr[s] = f[s].astype(np.float16)
    arr["out"] = out.astype(np.float16)
    arr["hand"] = S.hand_features(x).astype(np.float32)
    return arr


# ---- sampling --------------------------------------------------------------------------------
@torch.no_grad()
def ancestral(model, x, steps, gen, bs=512):
    """DDPM ancestral sampling along the decreasing step sequence `steps` (respaced posterior,
    x0-hat clipped to [-1, 1]); x is a torch tensor on the model's device at steps[0]."""
    device = x.device
    ab = torch.tensor(S.AB, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].clone()
        for k in range(len(steps) - 1):
            s, s2 = int(steps[k]), int(steps[k + 1])
            a, a2 = ab[s], ab[s2]
            tb = torch.full((len(xb),), s, dtype=torch.long, device=device)
            with amp_ctx(device):
                e = model(xb, tb if model.clock else None).float()
            x0 = ((xb - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
            beta = 1 - a / a2
            mean = (a2.sqrt() * beta / (1 - a)) * x0 + ((1 - beta).sqrt() * (1 - a2) / (1 - a)) * xb
            var = beta * (1 - a2) / (1 - a)
            z = torch.randn(xb.shape, generator=gen, device=device)
            xb = mean + var.sqrt() * z
        # the last move lands on step steps[-1]; its x0-hat is the image
        s = int(steps[-1])
        tb = torch.full((len(xb),), s, dtype=torch.long, device=device)
        with amp_ctx(device):
            e = model(xb, tb if model.clock else None).float()
        out.append(((xb - (1 - ab[s]).sqrt() * e) / ab[s].sqrt()).clamp(-1, 1))
    return torch.cat(out)


def sample_grid(model, n, seed, device):
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(n, 3, 32, 32, generator=gen, device=device)
    return ancestral(model, x, list(range(S.T - 1, -1, -1)), gen)


def harvest_spread(model, X0, sp, k, seed):
    device = next(model.parameters()).device
    x0 = X0[sp["img"]]
    xt = noisy(x0, sp["t"], sp["eps"])
    f, _ = features(model, xt, sp["t"])
    eh = eps_hat(model, xt, sp["t"])
    a = S.AB[sp["t"]].reshape(-1, 1, 1, 1).astype(np.float32)
    x0h = (xt - np.sqrt(1 - a) * eh) / np.sqrt(a)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    logvar = np.zeros(len(xt), np.float32)
    for i in range(len(xt)):
        steps = np.unique(np.round(np.linspace(int(sp["t"][i]), 0, S.SPREAD_STEPS + 1)).astype(int))[::-1]
        xs = torch.from_numpy(np.repeat(xt[i:i + 1], k, axis=0)).to(device)
        fin = ancestral(model, xs, steps, gen).float().cpu().numpy()
        logvar[i] = np.log(fin.var(axis=0).mean())
    return {"band": sp["band"], "t": sp["t"], "img": sp["img"], "mid": f["mid"].astype(np.float32),
            "hand_xt": S.hand_features(xt).astype(np.float32),
            "hand_x0": S.hand_features(x0h.astype(np.float32)).astype(np.float32), "logvar": logvar}


# ---- checks ----------------------------------------------------------------------------------
def gdet(model, X0, sp):
    x0 = X0[sp["img"][:128]]
    xt = noisy(x0, sp["t"][:128], sp["eps"][:128])
    f1, _ = features(model, xt, sp["t"][:128])
    f2, _ = features(model, xt, sp["t"][:128])
    d = max(float(np.abs(f1[s] - f2[s]).max()) for s in S.SITES)
    return {"max_abs_diff": d, "tol": S.GDET_TOL, "pass": d <= S.GDET_TOL}


def resume_selftest(X, seeds, n_steps=3, C=8):
    """CPU, a width-8 M-drop: two epochs of n_steps straight, against one epoch, checkpoint,
    reload into a fresh state, one more epoch. The weights must agree bit for bit."""
    cpu = torch.device("cpu")
    n = n_steps * S.BATCH
    Xc = torch.from_numpy(X[:n]).to(cpu)
    quiet = lambda *a, **k: None  # noqa: E731
    a = train_epochs(new_state("drop", cpu, C=C, seed=seeds["TRAIN"]), "drop", Xc, 2, seeds, cpu,
                     log=quiet, n_train=n)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"w3_resume_selftest_{os.getpid()}.pt"
    b = train_epochs(new_state("drop", cpu, C=C, seed=seeds["TRAIN"]), "drop", Xc, 1, seeds, cpu,
                     ckpt=tmp, log=quiet, n_train=n)
    c = load_state(new_state("drop", cpu, C=C, seed=seeds["TRAIN"] + 1), tmp, cpu)
    c = train_epochs(c, "drop", Xc, 2, seeds, cpu, log=quiet, n_train=n)
    tmp.unlink(missing_ok=True)
    sa, sc_ = a["model"].state_dict(), c["model"].state_dict()
    same = all(torch.equal(sa[k], sc_[k]) for k in sa)
    return {"pass": bool(same), "steps_per_epoch": n_steps, "width": C, "device": "cpu"}


# ---- the run (called by the notebook) --------------------------------------------------------
def run(out, cifar_path, smoke=True, device=None, log=print, code_sha=None,
        width=S.WIDTH, n_train=None, sizes=None, expect_md5=S.CIFAR_MD5):
    """Train, gate, check and harvest everything into `out` (a Drive folder). Resume-safe: a
    finished model, gate or set is skipped on a re-run; training resumes from the last Drive
    checkpoint. width / n_train / sizes / expect_md5 exist for the CPU training suite only; the
    notebook passes none of them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    sd, sz = S.seeds(smoke), dict(S.sizes(smoke), **(sizes or {}))
    md5 = md5_file(cifar_path)
    if md5 != expect_md5:
        raise SystemExit(f"G-data: CIFAR-10 md5 {md5} differs from {expect_md5}")
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "smoke": smoke, "cifar_md5": md5, "seeds": sd, "sizes": sz, "code_sha": code_sha,
           "width": width, "n_train": n_train}
    (out / "env.json").write_text(json.dumps(env, indent=1, sort_keys=True))
    Xtr, Xte = load_cifar(cifar_path)
    v = float((Xtr.astype(np.float64) ** 2).mean())
    if not (out / "resume_test.json").exists():
        rt = resume_selftest(Xtr, sd)
        (out / "resume_test.json").write_text(json.dumps(rt, indent=1, sort_keys=True))
        log(f"resume self-test: {'PASS' if rt['pass'] else 'FAIL'}")
    Xg = torch.from_numpy(Xtr).to(device)
    models = {}
    local = Path("/content/w3_ckpt") if Path("/content").exists() else out / "ckpt_local"
    local.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        st = new_state(tag, device, C=width, seed=sd["TRAIN"])
        dck = out / f"ckpt_{tag}.pt"
        if dck.exists():
            load_state(st, dck, device)
            log(f"{tag}: resumed at epoch {st['epoch']}")
        gfile = out / f"gate_{tag}.json"
        target = sz["EPOCHS"]
        while True:
            if st["epoch"] < target:
                train_epochs(st, tag, Xg, target, sd, device, ckpt=local / f"{tag}.pt", drive_ckpt=dck, log=log,
                             n_train=n_train)
            g = gate(st["model"], Xte, v, sd, sz["N_GATE"])
            if tag == "drop":
                g["null"] = gate(st["model"], Xte, v, sd, sz["N_GATE"], null=True)
                g["pass"] = g["pass"] and g["null"]["pass"]
            g["epochs"] = st["epoch"]
            if g["pass"] or target >= (S.EPOCHS_MAX if not smoke else 2 * sz["EPOCHS"]):
                break
            log(f"{tag}: gate FAILED at {st['epoch']} epochs; training on to the registered maximum")
            target = S.EPOCHS_MAX if not smoke else 2 * sz["EPOCHS"]
        gfile.write_text(json.dumps(g, indent=1, sort_keys=True))
        (out / f"train_{tag}.json").write_text(json.dumps(
            {"epochs": st["epoch"], "losses": st["losses"], "sec_per_step": st["sec_per_step"]}, indent=1))
        log(f"{tag}: gate {'PASS' if g['pass'] else 'FAIL'} at {st['epoch']} epochs")
        models[tag] = (st["model"].eval(), g["pass"])
    for tag in ("on0", "off0"):
        models[tag] = (make_model(tag[:-1], C=width, mult=S.MULT, seed=sd["TRAIN"]).to(device).eval(), True)
    sp = split_set(sd, sz["N_PER_CLASS"], n_pool=len(Xte))
    u = sham_dirs(sd, len(sp["t"]), dim=4 * width)
    ptr, pcal = probe_sets(sd, sz["N_PTRAIN"], sz["N_PCAL"], n_pool=len(Xtr))
    if not (out / "gdet.json").exists():          # determinism belongs to the code: always checked
        (out / "gdet.json").write_text(json.dumps(gdet(models["on"][0], Xte, sp), indent=1, sort_keys=True))
    plan = [("on", "split"), ("on", "sham"), ("on", "ptrain"), ("on", "pcal"), ("on", "spread"),
            ("drop", "split"), ("drop", "sham"), ("drop", "ptrain"), ("drop", "pcal"),
            ("drop", "ptrain_null"), ("drop", "pcal_null"), ("off", "ptrain"), ("off", "pcal"),
            ("on0", "split"), ("on0", "ptrain"), ("off0", "ptrain"), ("off0", "pcal")]
    for tag, what in plan:
        name = f"{tag}_{what}"
        model, ok = models[tag]
        if not ok or S.has(name, out):
            continue
        t0 = time.time()
        if what == "split":
            arr = harvest_split(model, Xte, sp)
        elif what == "sham":
            arr = harvest_sham(model, Xte, sp, u)
        elif what == "spread":
            ss = spread_set(sd, sz["SPREAD_N"], sp["img"], n_pool=len(Xte))
            arr = harvest_spread(model, Xte, ss, sz["SPREAD_K"], sd["SPREAD"])
        else:
            ps = ptr if what.startswith("ptrain") else pcal
            arr = harvest_matched(model, Xtr, ps, null=what.endswith("_null"))
        S.ship(arr, name, out)
        log(f"shipped {name} ({time.time() - t0:.0f} s)")
    for tag in TAGS:
        f = out / f"samples_{tag}.npy"
        if models[tag][1] and not f.exists():
            np.save(f, sample_grid(models[tag][0], sz["N_SAMPLES"], sd["SAMPLE"], device).cpu().numpy().astype(np.float16))
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return out
