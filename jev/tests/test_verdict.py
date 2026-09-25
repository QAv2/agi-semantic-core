"""Verdict suite: planted agents with known structure must land on their registered
branch of every primary (protocol §4 step 2), through the real analysis code.

Reduced flight for speed: J1 240 stimuli per family, J2 40 ladders, 320 permutations
(the smallest count whose minimum p, 1/321, clears the strictest Holm step .01/3).

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_verdict.py -q
"""
import pytest

from jev import agents as A
from jev import analyze_j1, analyze_j2
from jev import plan as P

FZ = {"families": {"F1": {"included": True, "d0": 0.71, "d1": 1.0, "strength": "standard"},
                   "F2": {"included": True, "d0": 2.0, "d1": 1.41, "strength": "strong"},
                   "F3": {"included": True, "d0": 0.60, "d1": 0.70, "strength": "standard"},
                   "F1K2": {"included": True, "d0": 0.5}},
      "bundle_a6": True}
N_PERM = 320


@pytest.fixture(scope="module")
def j1specs():
    return [P.strip(s) for s in P.j1_flight(FZ, n=240)]


@pytest.fixture(scope="module")
def j2specs():
    return [P.strip(s) for s in P.j2_flight(40)]


def j1(specs, kind, secondary=False, **kw):
    finals = A.synthesize(specs, j1_kind=kind) if not kw else _custom(specs, kind, **kw)
    return analyze_j1.analyze(specs, finals, FZ, n_perm=N_PERM, secondary=secondary)


def _custom(specs, kind, tau):
    agent = A.J1Agent(kind, specs, tau=tau)
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": agent.respond(s)} for s in specs}


def test_self_knower_passes_everything(j1specs):
    v = j1(j1specs, "self", secondary=True)
    assert all(f["verdict"] == "PASS" for f in v["P1"].values()), v["P1"]
    assert v["P2"]["verdict"] == "PASS", v["P2"]
    assert v["P3"]["verdict"] == "PASS", v["P3"]
    assert all(g["pass"] for g in v["gates"]["G5"].values())
    assert v["gates"]["G3_argmax_rate"] > 0.99
    s = v["secondary"]
    assert s["S3_meta_d"]["A0"]["m_ratio"] > 0.5
    assert s["S7_vendor_confidence"]["share_within_0.015"] > 0.99
    assert set(s["S6_second_channel"]) == {"F1", "F2", "F3"}


def test_evidence_reader(j1specs):
    v = j1(j1specs, "evidence")
    assert all(f["verdict"].startswith("FAIL") for f in v["P1"].values()), v["P1"]
    assert v["P2"]["verdict"] == "FAIL: confidence reads the evidence, not the self", v["P2"]
    assert v["P3"]["verdict"] == "PASS"


def test_parrot(j1specs):
    v = j1(j1specs, "parrot")
    assert all(f["verdict"].startswith("FAIL") for f in v["P1"].values())
    assert v["P2"]["verdict"] == "FAIL: confidence reads the evidence, not the self", v["P2"]
    assert v["P3"]["verdict"] == "FAIL: confabulated decision"


def test_presentation_reader(j1specs):
    v = j1(j1specs, "presentation")
    assert v["P2"]["verdict"] == "FAIL: confidence reads the presentation", v["P2"]


def test_p2_not_adjudicable_without_degradation(j1specs):
    v = j1(j1specs, "self", tau={"A0": 0.9, "A1": 0.9, "A2": 0.9, "A3": 0.9})
    assert v["P2"]["verdict"].startswith("NOT ADJUDICABLE"), v["P2"]


def j2(specs, kind):
    return analyze_j2.analyze(specs, A.synthesize(specs, j2_kind=kind))


def test_j2_ideal(j2specs):
    v = j2(j2specs, "ideal")
    assert (v["P1"]["verdict"], v["P2"]["verdict"], v["P3"]["verdict"]) == ("PASS", "PASS", "PASS"), v


def test_j2_confabulator(j2specs):
    v = j2(j2specs, "confab")
    assert v["P1"]["verdict"] == "FAIL: decides about nothing"
    assert v["P2"]["failing"] == ["confabulated", "holds by default"]
    assert v["P3"]["verdict"] == "PASS"


def test_j2_default_holder(j2specs):
    v = j2(j2specs, "default")
    assert v["P2"]["failing"] == ["timid", "confabulated", "holds by default"]
    assert v["P1"]["verdict"] == "PASS" and v["P3"]["verdict"] == "PASS"


def test_j2_genre_chooser(j2specs):
    v = j2(j2specs, "genre")
    assert v["P3"]["verdict"] == "FAIL: genre by choice"
    assert v["P1"]["verdict"] == "PASS" and v["P2"]["verdict"] == "PASS"
