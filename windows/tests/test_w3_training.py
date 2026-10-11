"""W3 training suite (docs/W3_LOOP_PROTOCOL.md §7): the Colab-side code on a CPU, before any GPU
minute is spent. Width-8 networks unless a test needs the registered width."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from windows import w3_core as c
from windows import w3_model as m
from windows import w3_spec as S

ROOT = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


def test_lift_is_verbatim():
    src = (ROOT / "windows/power/w3_budget_check.py").read_text()
    lifted = src[src.index("def sinusoid"):src.index("def main")].rstrip()
    assert lifted in (ROOT / "windows/w3_model.py").read_text()


def test_spec_matches_design_check():
    import sys
    sys.path.insert(0, str(ROOT / "windows/power"))
    import w3_design_check as d
    import w3_split_check as sc
    assert np.array_equal(S.alpha_bar(), d.alpha_bar("cosine"))
    x = np.random.default_rng(0).standard_normal((3, 3, 32, 32)).astype(np.float32)
    assert np.array_equal(S.hand_features(x), d.hand_features(x))
    assert np.array_equal(S.ALPHAS, d.ALPHAS)
    assert (S.LO, S.HI, S.SHAM_BAR, S.OFFSETS, S.STEP_BANDS) == (sc.LO, sc.HI, sc.SHAM_BAR, sc.OFFSETS, sc.STEP_BANDS)


@pytest.mark.parametrize("tag", ["on", "drop", "off"])
def test_forward_equals_the_original(tag):
    import sys
    sys.path.insert(0, str(ROOT / "windows/power"))
    import w3_budget_check as b
    net = m.make_model(tag, C=8, seed=1).eval()
    ref = b.UNet(cin=3, C=8, clock=(tag != "off")).eval()
    ref.load_state_dict({k: v for k, v in net.state_dict().items() if k != "null_emb"})
    x, t = torch.randn(3, 3, 32, 32), torch.randint(0, 1000, (3,))
    out, f = net(x, t, feats=True)
    assert torch.equal(out, ref(x, t))
    assert torch.equal(out, net(x, t))
    if tag != "off":
        assert torch.equal(out, net(x, emb=net.embed(t)))


def test_sites_at_the_registered_width():
    net = m.make_model("on", C=S.WIDTH, seed=0).eval()
    _, f = net(torch.randn(2, 3, 32, 32), torch.tensor([5, 900]), feats=True)
    assert tuple(f) == S.SITES
    assert {k: v.shape[1] for k, v in f.items()} == S.SITE_DIMS
    assert net.temb[-1].out_features == S.EMB_DIM
    n_on = sum(p.numel() for p in net.parameters())
    n_off = sum(p.numel() for p in m.make_model("off", C=S.WIDTH).parameters())
    assert round(n_on / 1e6, 1) == 6.5 and round(n_off / 1e6, 1) == 5.9


def test_null_vector_only_where_drawn():
    net = m.make_model("drop", C=8, seed=0).eval()
    with torch.no_grad():
        net.null_emb.normal_()
    x, t = torch.randn(4, 3, 32, 32), torch.randint(0, 1000, (4,))
    mask = torch.tensor([True, False, True, False])
    got = net(x, t, null_mask=mask)
    lab = net(x, t)
    nul = net(x, emb=net.null_emb.expand(4, -1))
    assert torch.equal(got[~mask], lab[~mask]) and torch.equal(got[mask], nul[mask])


def test_stream_is_shared_and_resumable():
    a = list(c.epoch_batches(7, 3, 512, batch=128))
    b = list(c.epoch_batches(7, 3, 512, batch=128))
    assert len(a) == 4
    for x, y in zip(a, b):
        assert all(torch.equal(p, q) for p, q in zip(x, y))
    other = next(c.epoch_batches(7, 4, 512, batch=128))
    assert not torch.equal(a[0][0], other[0])
    masks = c.drop_masks(S.SEEDS["DROP"], 0, 390)
    assert abs(float(masks.float().mean()) - S.DROP_P) < 0.01


def test_lr_schedule():
    assert c.lr_at(0) == pytest.approx(S.LR / S.WARMUP)
    assert c.lr_at(S.WARMUP - 1) == pytest.approx(S.LR)
    assert c.lr_at(10_000) == S.LR


def test_resume_selftest_passes():
    X = np.random.default_rng(0).uniform(-1, 1, (3 * S.BATCH, 3, 32, 32)).astype(np.float32)
    assert c.resume_selftest(X, S.seeds(True))["pass"]


@pytest.mark.parametrize("tag", ["on", "drop", "off"])
def test_meta_device_forward_backward(tag):
    """A stray CPU-made tensor fails on the meta device (the timing smoke's lesson)."""
    meta = torch.device("meta")
    net = m.make_model(tag, C=S.WIDTH).to(meta)
    x = torch.randn(4, 3, 32, 32, device=meta)
    t = torch.randint(0, 1000, (4,), device=meta)
    nm = torch.ones(4, dtype=torch.bool, device=meta) if tag == "drop" else None
    out, f = net(x, t, null_mask=nm, feats=True)
    out.mean().backward()
    if tag != "off":
        net(x, emb=net.embed(t)).sum().backward()


def test_meta_device_training_step():
    """One full training step (stream, noising, null mask, forward, backward, clip, AdamW) on the
    meta device. It can only stop at the loss read-out, which no meta tensor can do: reaching it
    means no CPU-made tensor met a device tensor on the way."""
    meta = torch.device("meta")
    st = c.new_state("drop", meta, C=8)
    X = torch.empty(256, 3, 32, 32, device=meta)
    with pytest.raises(RuntimeError, match=r"item\(\) cannot be called on meta"):
        c.train_epochs(st, "drop", X, 1, S.seeds(True), meta, log=lambda *a: None, n_train=128)
    assert all(p.grad is not None for p in st["model"].parameters() if p.requires_grad and p is not st["model"].null_emb)


def test_gate_multiple_is_the_best_multiple():
    rng = np.random.default_rng(3)
    x0 = rng.uniform(-1, 1, (400, 3, 32, 32)).astype(np.float32) * 0.7
    v = float((x0.astype(np.float64) ** 2).mean())
    for t in (5, 200, 500, 900, 999):
        eps = rng.standard_normal(x0.shape).astype(np.float32)
        xt = c.noisy(x0, np.full(400, t), eps)
        a = S.AB[t]
        lam = np.sqrt(1 - a) / (a * v + 1 - a)
        mse = lambda k: float(((k * xt - eps) ** 2).mean())  # noqa: E731
        best = mse(lam)
        assert best <= mse(0.0) + 1e-6 and best <= mse(1.0) + 1e-6
        assert best <= mse(lam * 1.02) + 1e-5 and best <= mse(lam * 0.98) + 1e-5


def test_split_set_shares_one_noise_draw():
    sp = c.split_set(S.seeds(), S.N_PER_CLASS)
    assert len(sp["t"]) == 3 * S.N_PER_CLASS and len(set(sp["img"])) == len(sp["img"])
    for o in S.OFFSETS:
        sel = sp["off"] == o
        assert sel.sum() == S.N_PER_CLASS
        assert sp["t"][sel].min() >= S.BASE_MIN and sp["t"][sel].max() < S.T - o
        assert np.all(sp["tp"][sel] == sp["t"][sel] + o)
    x0 = np.random.default_rng(0).uniform(-1, 1, (5, 3, 32, 32)).astype(np.float32)
    xp = c.noisy(x0, sp["tp"][:5], sp["eps"][:5])
    a = S.AB[sp["tp"][:5]].reshape(-1, 1, 1, 1)
    assert np.allclose((xp - np.sqrt(a) * x0) / np.sqrt(1 - a), sp["eps"][:5], atol=1e-4)


def test_sets_are_disjoint_where_registered():
    sd = S.seeds()
    sp = c.split_set(sd, S.N_PER_CLASS)
    ss = c.spread_set(sd, S.SPREAD_N, sp["img"])
    assert not set(ss["img"]) & set(sp["img"])
    ptr, pcal = c.probe_sets(sd, S.N_PTRAIN, S.N_PCAL)
    assert not set(ptr["img"]) & set(pcal["img"])
    assert ptr["img"].max() < 50000 and sp["img"].max() < 10000
    u = c.sham_dirs(sd, len(sp["t"]))
    assert u.shape == (768, S.EMB_DIM) and np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-5)


def test_ship_respects_the_cap(tmp_path):
    arr = {"t": np.arange(768, dtype=np.int32),
           **{s: np.zeros((768, 4, S.SITE_DIMS[s]), np.float32) for s in S.SITES}}
    lines = S.ship(arr, "on_split", tmp_path)
    assert all(x["bytes"] <= S.SHIP_CAP for x in lines) and len(lines) >= 3
    back = S.load("on_split", tmp_path)
    assert all(np.array_equal(arr[k], back[k]) for k in arr)
    (tmp_path / lines[0]["file"]).write_bytes(b"x" + (tmp_path / lines[0]["file"]).read_bytes()[1:])
    with pytest.raises(ValueError):
        S.load("on_split", tmp_path)


def _fake_cifar(path, n_train_per_batch, n_test, seed=0):
    rng = np.random.default_rng(seed)

    def rec(n):
        lab = rng.integers(0, 10, (n, 1), dtype=np.uint8)
        # smooth-ish images so a denoiser has something to learn
        base = rng.integers(0, 256, (n, 3, 4, 4)).astype(np.float32)
        img = np.kron(base, np.ones((1, 1, 8, 8))).clip(0, 255).astype(np.uint8)
        return np.concatenate([lab, img.reshape(n, -1)], axis=1).tobytes()
    with tarfile.open(path, "w:gz") as tf:
        for name, n in [(f"data_batch_{i}.bin", n_train_per_batch) for i in range(1, 6)] + [("test_batch.bin", n_test)]:
            b = rec(n)
            ti = tarfile.TarInfo(f"cifar-10-batches-bin/{name}")
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))


def test_run_end_to_end_on_cpu(tmp_path, monkeypatch):
    """The whole notebook path at width 8 on a fake CIFAR: train, gate, G-det, every set shipped,
    samples, DONE; a re-run skips everything already done. A width-8 net trained for two steps
    can't pass the real gate, so the gate is computed for real and then marked passed."""
    real_gate = c.gate

    def passing_gate(*a, **k):
        g = real_gate(*a, **k)
        g["pass"] = True
        return g
    monkeypatch.setattr(c, "gate", passing_gate)
    fake = tmp_path / "cifar.tar.gz"
    _fake_cifar(fake, 160, 1200)
    out = tmp_path / "smoke_x"
    sizes = {"EPOCHS": 1, "N_PER_CLASS": 8, "N_PTRAIN": 40, "N_PCAL": 30, "N_GATE": 40,
             "SPREAD_N": 2, "SPREAD_K": 2, "N_SAMPLES": 2}
    logs = []
    c.run(out, fake, smoke=True, device=CPU, log=logs.append, width=8, n_train=256, sizes=sizes,
          expect_md5=c.md5_file(fake))
    assert (out / "DONE").exists()
    assert json.loads((out / "resume_test.json").read_text())["pass"]
    for tag in c.TAGS:
        g = json.loads((out / f"gate_{tag}.json").read_text())
        assert [b["steps"] for b in g["bands"]] == [list(b) for b in S.GATE_BANDS]
        assert g["epochs"] >= 1
    man = json.loads((out / "manifest.json").read_text())
    shipped = {x["set"] for x in man}
    assert shipped == {"on_split", "on_sham", "on_ptrain", "on_pcal", "on_spread", "drop_split",
                       "drop_sham", "drop_ptrain", "drop_pcal", "drop_ptrain_null", "drop_pcal_null",
                       "off_ptrain", "off_pcal", "on0_split", "on0_ptrain", "off0_ptrain", "off0_pcal"}
    assert json.loads((out / "gdet.json").read_text())["pass"]
    assert all((out / f"samples_{t}.npy").exists() for t in c.TAGS)
    assert "null" in json.loads((out / "gate_drop.json").read_text())
    nul, lab = S.load("drop_ptrain_null", out), S.load("drop_ptrain", out)
    assert np.array_equal(nul["t"], lab["t"]) and not np.array_equal(nul["mid"], lab["mid"])
    assert np.array_equal(S.load("on_sham", out)["img"], S.load("on_split", out)["img"])
    # the two halves agree on the format: the local analysis reads this harvest end to end
    from windows import analyze_w3
    v = analyze_w3.analyze(out, smoke=True)
    assert v["P1"]["verdict"] in ("CONTENT", "LABEL", "SPLIT", "VARIES BY LEVEL", "UNCLEAR", "UNREADABLE")
    assert v["P2"]["verdict"] in ("SILENT", "NOT SILENT", "not judged")
    assert {k.split("_")[0] for k in v["secondaries"]} == {f"S{i}" for i in range(1, 11)}
    sp = S.load("on0_split", out)
    assert sp["mid"].shape == (24, 4, 16) and sp["out"].shape == (24, 4, 14) and sp["hand"].shape == (24, 2, 14)
    assert S.load("off0_ptrain", out)["mid"].dtype == np.float16
    # resume: a second run trains nothing and ships nothing new
    n_lines = len(man)
    logs2 = []
    c.run(out, fake, smoke=True, device=CPU, log=logs2.append, width=8, n_train=256, sizes=sizes,
          expect_md5=c.md5_file(fake))
    assert not any("epoch" in s and "/" in s and "loss" in s for s in logs2)
    assert len(json.loads((out / "manifest.json").read_text())) == n_lines


def test_run_refuses_wrong_data(tmp_path):
    fake = tmp_path / "cifar.tar.gz"
    _fake_cifar(fake, 2, 2)
    with pytest.raises(SystemExit):
        c.run(tmp_path / "x", fake, smoke=True, device=CPU)
