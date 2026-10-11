"""W3 analysis suite (registration §7): the real analyze_w3 on synthetic harvests written by
agents_w3 in w3_spec's own format, at the registered sizes and the fixed suite seed.

What is asserted (lesson banked from W1-T): the exact licensed verdict where the design check's
rate at 256 images per offset class is >= 0.99 (registration §2.5: every world but BANDED, 1.00),
and for BANDED (.98) the registered verdict at the suite's fixed seed, never re-seeded to pass.
The verdict agents run at the registered bootstrap size (10,000) on the sections they test;
each statistic draws from its own generator, so those numbers are the ones a full run reports.
The complete analysis (every section) runs in the fresh-process recompute test and on a
smoke-sized harvest. Every harvest lives in pytest's tmp dirs and is deleted after use (each is
~95 MB and /tmp is RAM on this machine).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from windows import agents_w3 as A
from windows import analyze_w3 as AN
from windows import w3_spec as S

REPO = Path(__file__).resolve().parents[2]
P12 = ("P1", "P2")
P1_VERDICTS = {"NOT ADJUDICABLE", "UNREADABLE", "VARIES BY LEVEL", "CONTENT", "LABEL", "SPLIT", "UNCLEAR"}
ALL_SECTIONS = {"S1_profile", "S2_probe", "S3_output", "S4_directions", "S5_offsets", "S6_drop",
                "S7_off", "S8_untrained", "S9_spread", "S10_record"}
_CACHE = {}


@pytest.fixture(autouse=True)
def _empty_tmp_path(request):
    """Empty each test's tmp_path afterwards: harvests are large and /tmp is RAM here."""
    tp = request.getfixturevalue("tmp_path") if "tmp_path" in request.fixturenames else None
    yield
    if tp is not None:
        for f in tp.iterdir():
            shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink()


def run_agent(agent, tmp_path_factory, parts=P12):
    """Write the agent's harvest, analyse it at the registered bootstrap size, delete it."""
    key = (agent, parts)
    if key not in _CACHE:
        d = tmp_path_factory.mktemp(agent.replace("+", "_"))
        try:
            A.make(agent, d)
            _CACHE[key] = AN.analyze(d, parts=parts)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    v = _CACHE[key]
    assert v["stage"]["boot"] == S.BOOT and v["stage"]["smoke"] is False
    return v


def p1(v):
    return v["P1"]["verdict"]


def p2(v):
    return v["P2"]["verdict"]


# ---- P1 (registration §7) --------------------------------------------------------------------
@pytest.mark.parametrize("agent,want", [("content", "CONTENT"), ("label", "LABEL"), ("equal", "SPLIT")])
def test_p1_primary_agents(agent, want, tmp_path_factory):
    v = run_agent(agent, tmp_path_factory)
    assert p1(v) == want, v["P1"]
    assert v["P1"]["site"] == "mid"
    assert v["P1"]["kappa_hi"] <= S.KAPPA_BAR


def test_misread_collinearity_end_to_end(tmp_path_factory):
    """MISREAD.3: the representation moves ~90% with the content, the probe leans on the label."""
    v = run_agent("misread", tmp_path_factory, parts=("P1", "P2", "S2"))
    assert p1(v) == "CONTENT", v["P1"]
    s2 = v["secondaries"]["S2_probe"]["mid"]
    assert s2["verdict"] in {"LABEL", "SPLIT"}, s2
    assert s2["w"] < v["P1"]["w"]                 # S2 further toward LABEL than P1 (§5.1 pre-read)


@pytest.mark.parametrize("agent", ["collapse", "scatter"])
def test_unreadable(agent, tmp_path_factory):
    v = run_agent(agent, tmp_path_factory)
    assert p1(v) == "UNREADABLE", v["P1"]
    assert v["P1"]["kappa_hi"] > S.KAPPA_BAR


def test_banded(tmp_path_factory):
    """Design rate .98 at 256 per class: asserted at the fixed suite seed, not re-seeded."""
    v = run_agent("banded", tmp_path_factory)
    assert p1(v) == "VARIES BY LEVEL", v["P1"]


def test_interact(tmp_path_factory):
    v = run_agent("interact", tmp_path_factory)
    assert p1(v) == "VARIES BY LEVEL", v["P1"]
    assert v["P1"]["kappa_hi"] <= S.KAPPA_BAR     # the interaction is real but under the bar


# ---- P2 --------------------------------------------------------------------------------------
def test_quiet_sham(tmp_path_factory):
    v = run_agent("quiet-sham", tmp_path_factory)
    assert p1(v) == "LABEL"
    assert p2(v) == "SILENT", v["P2"]
    assert v["P2"]["upper"] <= S.SHAM_BAR and v["P2"]["class"] == S.SHAM_CLASS


def test_reactive_sham(tmp_path_factory):
    v = run_agent("reactive-sham", tmp_path_factory)
    assert p1(v) == "LABEL"
    assert p2(v) == "NOT SILENT", v["P2"]


def test_content_sham_not_judged(tmp_path_factory):
    v = run_agent("content+sham", tmp_path_factory)
    assert p1(v) == "CONTENT"
    assert p2(v) == "not judged", v["P2"]
    assert isinstance(v["P2"]["S"], float)              # S still reported
    assert v["P2"]["upper"] > S.SHAM_BAR                # a reactive sham: only the P1 gate withholds it
    assert set(v["P2"]["by_offset"]) == {str(o) for o in S.OFFSETS}


def test_untrained_not_adjudicable(tmp_path_factory):
    v = run_agent("untrained", tmp_path_factory)
    assert p1(v) == "NOT ADJUDICABLE" and p2(v) == "NOT ADJUDICABLE"
    assert "M-on failed G-train" in v["P1"]["reasons"]
    assert v["gates"]["gate_on"] is False


# ---- identities and plumbing -----------------------------------------------------------------
def test_shapley_identity(tmp_path_factory):
    for agent in ("content", "label", "equal", "interact", "collapse"):
        v = run_agent(agent, tmp_path_factory)
        rows = [v["P1"]] + [b for b in v["P1"]["bands"] if not b["skipped"]]
        for r in rows:
            assert abs(r["w"] + r["l"] - 1) < 1e-8, (agent, r)
    rng = np.random.default_rng(0)                      # any 2 x 2, additive or not
    X = rng.normal(size=(64, 4, 16)) * rng.uniform(0.1, 10, 16)
    t = AN.move_terms(X, AN.inv_sd(X[:, 0]))
    w = (t["c1"] + t["c2"]).sum() / (t["b1"] + t["b2"]).sum()
    lab = (t["l1"] + t["l2"]).sum() / (t["b1"] + t["b2"]).sum()
    assert abs(w + lab - 1) < 1e-12
    assert np.allclose(t["b1"], t["b2"])


def test_to_step_inverts_the_schedule():
    assert np.array_equal(AN.to_step(S.LOGSNR), np.arange(S.T))
    mid = (S.LOGSNR[:-1] + S.LOGSNR[1:]) / 2
    assert np.all(np.isin(AN.to_step(mid) - np.arange(S.T - 1), (0, 1)))
    assert AN.to_step(np.array([1e3, -1e3])).tolist() == [0, S.T - 1]


def test_fresh_process_recompute_all_sections(tmp_path):
    """Every section at the registered sizes (smoke bootstrap), then the CLI in a fresh process
    must write a byte-identical verdict.json and report.md."""
    d = tmp_path / "content"
    try:
        A.make("content", d)
        v = AN.write(d, smoke=True)
        first, rep = (d / "verdict.json").read_bytes(), (d / "report.md").read_bytes()
        (d / "verdict.json").unlink()
        (d / "report.md").unlink()
        r = subprocess.run([sys.executable, "-m", "windows.analyze_w3", str(d), "--smoke"],
                           cwd=REPO, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        assert (d / "verdict.json").read_bytes() == first
        assert (d / "report.md").read_bytes() == rep
        assert json.loads(first) == v                    # python numbers only, nothing lost
        assert v["stage"]["smoke"] and v["stage"]["boot"] == S.sizes(True)["BOOT"]
        assert v["sets"]["on_split"]["parts"] > 1        # multi-part sets reassembled
        sec = v["secondaries"]
        assert set(sec) == ALL_SECTIONS
        assert p1(v) == "CONTENT" and p2(v) == "not judged"
        assert set(sec["S1_profile"]) == set(S.SITES)
        assert sec["S1_profile"]["mid"]["w"] == v["P1"]["w"]
        assert set(sec["S2_probe"]) == set(S.SITES)
        assert sec["S3_output"]["verdict"] in P1_VERDICTS
        assert all(sec["S4_directions"][k]["w"] >= S.HI for k in ("dir1", "dir2"))
        assert all(sec["S5_offsets"][str(o)]["n"] == S.N_PER_CLASS for o in S.OFFSETS)
        s6 = sec["S6_drop"]
        assert s6["status"] == "ok" and s6["w_mid"]["verdict"] == "CONTENT"
        assert abs(s6["diff_from_on"]["diff"]) < 0.1
        assert set(s6["no_label_probe"]["sites"]) == set(S.SITES)
        s7 = sec["S7_off"]
        assert s7["status"] == "ok" and s7["twin_paired"] and s7["beats_hand_sites"]
        assert set(sec["S8_untrained"]["sites"]) == set(S.SITES)
        assert len(sec["S9_spread"]["bands"]) == len(S.SPREAD_CENTRES)
        assert {"gate_on", "gdet", "train_on"} <= set(sec["S10_record"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tampered_part_fails_load(tmp_path):
    d = tmp_path / "tamper"
    A.make("content", d, smoke=True)
    line = next(x for x in json.loads((d / "manifest.json").read_text()) if x["set"] == "on_split")
    f = d / line["file"]
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="sha256"):
        S.load("on_split", d)
    with pytest.raises(ValueError, match="sha256"):
        AN.analyze(d, smoke=True, parts=P12)


def test_missing_optional_sets_do_not_crash(tmp_path):
    d = tmp_path / "missing"
    omit = ("on_pcal", "on_spread", "drop_split", "drop_sham", "drop_ptrain_null", "drop_pcal_null",
            "off_ptrain", "off_pcal", "on0_split", "train_drop.json")
    A.make("content", d, smoke=True, omit=omit)
    v = AN.analyze(d, smoke=True)
    assert p1(v) in P1_VERDICTS - {"NOT ADJUDICABLE"}
    sec = v["secondaries"]
    assert sec["S2_probe"] == {"status": "missing", "sets": ["on_pcal"]}
    assert sec["S9_spread"]["status"] == "missing"
    assert sec["S8_untrained"]["status"] == "missing"
    assert sec["S7_off"]["status"] == "missing"
    s6 = sec["S6_drop"]
    assert s6["w_mid"]["status"] == "missing" and s6["no_label_probe"]["status"] == "missing"
    assert "train_drop" not in sec["S10_record"]
    assert not v["sets"]["on_spread"]["present"]
    # only the off0 twin missing: S7 still runs against the hand floor
    d2 = tmp_path / "missing2"
    A.make("content", d2, smoke=True, omit=("off0_ptrain", "off0_pcal", "on_sham"))
    v2 = AN.analyze(d2, smoke=True, parts=("P1", "P2", "S7"))
    s7 = v2["secondaries"]["S7_off"]
    assert s7["status"] == "ok" and s7["twin"]["status"] == "missing"
    assert p2(v2) == "NOT ADJUDICABLE" and v2["P2"]["reasons"] == ["on_sham missing"]


def test_gate_failures(tmp_path):
    d = tmp_path / "gates"
    A.make("label", d, smoke=True)
    (d / "gdet.json").write_text(json.dumps({"pass": False}))
    v = AN.analyze(d, smoke=True, parts=("P1", "P2", "S1", "S8"))
    assert p1(v) == "NOT ADJUDICABLE" and p2(v) == "NOT ADJUDICABLE"
    assert "G-det failed" in v["P1"]["reasons"]
    assert v["secondaries"]["S1_profile"]["status"] == "not adjudicable"
    assert v["secondaries"]["S8_untrained"]["status"] == "not adjudicable"
    (d / "gdet.json").write_text(json.dumps({"pass": True}))
    (d / "gate_drop.json").write_text(json.dumps({"pass": False}))
    (d / "gate_off.json").unlink()
    v = AN.analyze(d, smoke=True, parts=("P1", "P2", "S6", "S7"))
    assert p1(v) in P1_VERDICTS - {"NOT ADJUDICABLE"}
    assert v["secondaries"]["S6_drop"] == {"status": "not adjudicable", "reasons": ["M-drop failed G-train"]}
    assert v["secondaries"]["S7_off"] == {"status": "not adjudicable", "reasons": ["gate_off.json missing"]}
    (d / "gate_on.json").unlink()
    v = AN.analyze(d, smoke=True, parts=P12)
    assert p1(v) == "NOT ADJUDICABLE" and "gate_on.json missing" in v["P1"]["reasons"]


def test_m_on_failure_as_harvested(tmp_path):
    """The harvest's own layout when M-on fails G-train: no M-on sets and no gdet.json (G-det runs
    on M-on only when it passes). P1, P2 and M-on's secondaries are NOT ADJUDICABLE; M-drop,
    M-off and the untrained twin are still read."""
    d = tmp_path / "onfail"
    A.make("untrained", d, smoke=True,
           omit=("on_split", "on_sham", "on_ptrain", "on_pcal", "on_spread", "gdet.json"))
    v = AN.analyze(d, smoke=True)
    assert p1(v) == "NOT ADJUDICABLE" and p2(v) == "NOT ADJUDICABLE"
    assert {"M-on failed G-train", "gdet.json missing", "on_split missing"} <= set(v["P1"]["reasons"])
    sec = v["secondaries"]
    for k in ("S1_profile", "S2_probe", "S3_output", "S4_directions", "S5_offsets", "S9_spread"):
        assert sec[k]["status"] == "not adjudicable", k
    assert sec["S6_drop"]["status"] == "ok" and sec["S6_drop"]["diff_from_on"]["status"] == "not adjudicable"
    assert sec["S7_off"]["status"] == "ok" and sec["S8_untrained"]["status"] == "ok"
    assert v["gates"]["gdet"] is None


def test_smoke_sized_harvest_runs_end_to_end(tmp_path):
    """The smoke go rule: a smoke-sized harvest loads and every section runs."""
    d = tmp_path / "smoke"
    A.make("equal", d, smoke=True)
    v = AN.analyze(d, smoke=True)
    sz = S.sizes(True)
    assert v["sets"]["on_split"]["rows"] == 3 * sz["N_PER_CLASS"]
    assert v["sets"]["on_ptrain"]["rows"] == sz["N_PTRAIN"]
    assert v["sets"]["on_spread"]["rows"] == len(S.SPREAD_CENTRES) * sz["SPREAD_N"]
    assert p1(v) in P1_VERDICTS - {"NOT ADJUDICABLE"}
    assert set(v["secondaries"]) == ALL_SECTIONS
    assert all(s.get("status", "ok") == "ok" for s in v["secondaries"].values())
