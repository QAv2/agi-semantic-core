"""J3 verdict suite: planted agents with known structure land on their registered
branch of every primary, through the real analysis code (protocol §7 step 2).

The full registered plan (18,576 calls), with 300 bootstrap resamples for speed
(the registered analysis uses 10,000).

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j3_verdict.py -q
"""
import pytest

from jev import agents_j3 as A
from jev import analyze_j3
from jev import j3
from jev import j3_plan as P

B = 300


@pytest.fixture(scope="module")
def frozen():
    return {"bars": {"0": j3.fit_bar(0), "1": j3.fit_bar(1)}}


@pytest.fixture(scope="module")
def stims():
    return j3.flight_stimuli()


@pytest.fixture(scope="module")
def specs(stims):
    return [P.strip(s) for _, part in P.flight_parts(True, stims) for s in part]


def run(specs, frozen, kind, **kw):
    return analyze_j3.analyze(specs, A.synthesize(specs, kind, frozen["bars"]), frozen, B=B,
                              pin=A.PIN, **kw)


def test_twin_passes_everything(specs, frozen, stims):
    mspecs = [P.strip(s) for s in P.mouth_flight(stims)]
    v = run(specs, frozen, "twin", mouth=(mspecs, A.mouth_synthesize(mspecs, frozen["bars"])))
    for pres in ("RAW", "DIGEST"):
        r = v["primaries"][pres]
        assert r["P1"]["verdict"] == "PASS", r["P1"]
        assert r["P2"]["verdict"] == "PASS", r["P2"]
        assert r["P3"]["verdict"] == "PASS", r["P3"]
    assert v["J4_fork"]["jev_enters_J4"] and v["gates"]["G1_pass"] and v["gates"]["G3_argmax_rate"] > 0.99
    m = v["S6_mouth"]
    assert m["n_ok"] == 1548 and m["mouth"]["P1"]["G_CEIL"]["pass"]
    assert v["S3_presentation_confidence"]["n_stimuli"] == 2064


def test_degraded_fails_p3(specs, frozen):
    v = run(specs, frozen, "degraded", secondary=False)
    for pres in ("RAW", "DIGEST"):
        assert v["primaries"][pres]["P3"]["verdict"] == "FAIL: doesn't earn its place", v["primaries"][pres]["P3"]
    assert not v["J4_fork"]["jev_enters_J4"]


def test_claimer_claims_at_ceiling(specs, frozen):
    v = run(specs, frozen, "claimer", secondary=False)
    r = v["primaries"]["DIGEST"]
    assert r["P1"]["verdict"] == "FAIL: claims at ceiling", r["P1"]
    assert r["P2"]["verdict"] == "FAIL: confidence carries no information", r["P2"]
    assert r["P3"]["verdict"].startswith("FAIL")


def test_toplooker_names_the_push(specs, frozen):
    v = run(specs, frozen, "toplooker", secondary=False)
    r = v["primaries"]["DIGEST"]
    assert r["P1"]["a"]["clause"] == "PASS", r["P1"]
    assert r["P1"]["verdict"] == "FAIL: names the push, not the pattern", r["P1"]


def test_mute_is_not_adjudicable(specs, frozen):
    v = run(specs, frozen, "mute", secondary=False)
    r = v["primaries"]["RAW"]
    assert r["P1"]["verdict"] == "NOT ADJUDICABLE: does not read this presentation", r["P1"]


def test_separate_mode_joins_the_noul(stims, frozen):
    sub = stims[:24]
    specs = [P.strip(s) for _, part in P.flight_parts(False, sub) for s in part]
    rows = analyze_j3.calls_table(specs, A.synthesize(specs, "twin", frozen["bars"]))
    assert len(rows) == 24 * 2 * 3 and all(0 <= r["pyes"] <= 1 for r in rows)
