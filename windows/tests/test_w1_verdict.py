"""W1 verdict suite: planted agents through the real Gemini-app stage runner (fake UI), the real
read stage (fake reader endpoint) and the real analysis (protocol §7, Amendments 1-2).

Equal-arm agents assert P2 in {NO EXCESS, UNCLEAR} (never SELF EXCEEDS), and the weak agent
asserts only what its own statistics license; see build-time clarification 1 in the protocol.
"""
import json

import pytest

from windows import agents_w1 as A
from windows import analyze_w1 as AN

ABC = ("A", "B", "C")


def run(agent, tmp_path, name="w1_flight_t"):
    v, fake = A.run_agent(agent, tmp_path / name)
    return v


def p1(v):
    return {c: v["P1"][c]["verdict"] for c in ABC}


def p2(v):
    return {c: v["P2"][c]["verdict"] for c in ABC}


EQUAL = {"NO EXCESS", "UNCLEAR"}
NULLISH = {"AT CHANCE", "UNCLEAR"}       # Amendment 1: a null contrast reaches AT CHANCE 83-90%


def nullish(v, cs):
    return all(v["P1"][c]["verdict"] in NULLISH for c in cs)


@pytest.mark.parametrize("variant", ["base", "always70", "extremes"])
def test_null(tmp_path, variant):
    ag = A.AGENTS["null"] if variant == "base" else A.realistic(A.AGENTS["null"], variant)
    v = run(ag, tmp_path)
    assert v["gate_G_R"]["pass"]
    assert nullish(v, ABC)
    if variant == "base":
        assert p1(v) == {c: "AT CHANCE" for c in ABC}
    assert set(p2(v).values()) <= EQUAL          # Amendment 1: equal arms reach NO EXCESS 16-35%


@pytest.mark.parametrize("variant", ["base", "always70"])
def test_input_leak(tmp_path, variant):
    ag = A.AGENTS["input_leak"] if variant == "base" else A.realistic(A.AGENTS["input_leak"], variant)
    v = run(ag, tmp_path)
    assert p1(v)["A"] == p1(v)["B"] == "CARRIES" and nullish(v, "C")
    assert set(p2(v).values()) <= EQUAL
    for c in ("A", "B"):                       # the third-party arm carries it too
        o = v["secondaries"]["S1_observer"][c]
        assert o["p"] <= 0.01 if variant == "base" else o["mean"] > 0.5


@pytest.mark.parametrize("variant", ["base", "always70"])
def test_state(tmp_path, variant):
    ag = A.AGENTS["state"] if variant == "base" else A.realistic(A.AGENTS["state"], variant)
    v = run(ag, tmp_path)
    assert p1(v)["C"] == "CARRIES" and nullish(v, "AB")
    if variant == "base":
        assert p2(v)["C"] == "SELF EXCEEDS"
    else:                                      # a leaning, flat-confidence reader loses power, never direction
        assert p2(v)["C"] in {"SELF EXCEEDS", "UNCLEAR"}
    assert p2(v)["A"] in EQUAL and p2(v)["B"] in EQUAL


def test_lean_only(tmp_path):
    v = run(A.AGENTS["lean_only"], tmp_path)
    assert p1(v) == {c: "AT CHANCE" for c in ABC}
    assert all(v["P1"][c]["mean"] == 0.5 for c in ABC)
    assert p2(v) == {c: "NO EXCESS" for c in ABC}
    assert v["secondaries"]["S7_reader_lean"]["IMG"]["share_1"] == 1.0


def test_blind_readers_block_primaries(tmp_path):
    v = run(A.AGENTS["blind"], tmp_path)
    assert not v["gate_G_R"]["pass"]
    assert set(p1(v).values()) == {"NOT ADJUDICABLE"} and set(p2(v).values()) == {"NOT ADJUDICABLE"}


def test_sparse(tmp_path):
    v = run(A.AGENTS["sparse"], tmp_path)
    assert v["P1"]["A"]["n"] == 11 and v["P1"]["A"]["verdict"] == "NOT ADJUDICABLE"
    assert v["P2"]["A"]["verdict"] == "NOT ADJUDICABLE"
    assert v["P1"]["B"]["verdict"] in NULLISH
    d = v["secondaries"]["S4_deflection"]
    assert d["A-X-SELF"]["no_image"] == pytest.approx(5 / 16)


def test_weak(tmp_path):
    v = run(A.AGENTS["weak"], tmp_path)
    b = v["P1"]["A"]
    assert b["verdict"] != "NOT ADJUDICABLE" and 0.5 < b["mean"] < 0.75
    expect = "CARRIES" if b["p"] <= AN.ALPHA / 3 else (
        "AT CHANCE" if b["q975"] <= AN.BAR and b["binary_q975"] <= AN.BAR_BINARY else "UNCLEAR")
    # Holm with B and C at chance: A is first in the order, so its bar is alpha/3
    assert b["verdict"] == expect


def test_branch_rules_by_hand():
    """Hand-built pair scores sit in each P1 branch."""
    import numpy as np
    strong = [0.9] * 14 + [0.5, 0.6]
    assert AN.signflip_p(np.asarray(strong) - 0.5) <= 0.01 / 3
    flat = [0.5, 0.55, 0.45, 0.5] * 4
    assert AN.boot_q975(flat, (9,)) <= AN.BAR and AN.signflip_p(np.asarray(flat) - 0.5) > 0.01
    wide = [0.95, 0.05] * 6 + [0.9, 0.9, 0.2, 0.8]
    assert AN.boot_q975(wide, (9,)) > AN.BAR and AN.signflip_p(np.asarray(wide) - 0.5) > 0.01 / 3
    assert AN.holm({"A": 0.002, "B": 0.004, "C": 0.6}) == {"A": True, "B": True, "C": False}
    assert AN.holm({"A": None, "B": 0.003, "C": 0.009}) == {"A": False, "B": True, "C": False}
    assert AN.holm({"A": None, "B": 0.004, "C": 0.009}) == {"A": False, "B": False, "C": False}


def test_recompute_is_byte_identical(tmp_path):
    d = tmp_path / "w1_flight_t"
    A.run_agent(A.AGENTS["input_leak"], d)
    AN.main([str(d)])
    first = (d / "verdict.json").read_bytes()
    import subprocess, sys, os
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-m", "windows.analyze_w1", str(d)], check=True, env=env,
                   cwd=str(A.Path(A.__file__).resolve().parents[1]), capture_output=True)
    assert (d / "verdict.json").read_bytes() == first
