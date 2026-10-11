"""W1-T verdict suite (protocol §7): W1's planted agents, with text diagrams, through the real
W1-T stage runner (fake UI), the real read stage (fake reader) and the real analysis; plus
ceiling-leak. Same assertions as W1's suite (build-time clarification 1 there)."""
import os
import subprocess
import sys

import pytest

from windows import agents_w1t as A
from windows import analyze_w1t as AN

ABC = ("A", "B", "C")
EQUAL = {"NO EXCESS", "UNCLEAR"}
NULLISH = {"AT CHANCE", "UNCLEAR"}


def run(agent, tmp_path, name="w1t_flight_t"):
    v, _ = A.run_agent(agent, tmp_path / name)
    return v


def p1(v):
    return {c: v["P1"][c]["verdict"] for c in ABC}


def p2(v):
    return {c: v["P2"][c]["verdict"] for c in ABC}


def nullish(v, cs):
    return all(v["P1"][c]["verdict"] in NULLISH for c in cs)


@pytest.mark.parametrize("variant", ["base", "always70", "extremes"])
def test_null(tmp_path, variant):
    ag = A.AGENTS["null"] if variant == "base" else A.realistic(A.AGENTS["null"], variant)
    v = run(ag, tmp_path)
    assert v["gate_G_R"]["pass"]
    assert nullish(v, ABC)
    if variant == "base":            # W1-T clarification 2: AT CHANCE x3 is not what 16 pairs license
        assert all(0.45 <= v["P1"][c]["mean"] <= 0.55 for c in ABC)
    assert set(p2(v).values()) <= EQUAL


@pytest.mark.parametrize("variant", ["base", "always70"])
def test_input_leak(tmp_path, variant):
    ag = A.AGENTS["input_leak"] if variant == "base" else A.realistic(A.AGENTS["input_leak"], variant)
    v = run(ag, tmp_path)
    if variant == "base":
        assert p1(v)["A"] == p1(v)["B"] == "CARRIES"
    else:                   # clarification 2: a leaning, flat-70 reader loses power, never direction
        assert {p1(v)["A"], p1(v)["B"]} <= {"CARRIES", "UNCLEAR"} and "CARRIES" in {p1(v)["A"], p1(v)["B"]}
    assert nullish(v, "C")
    assert set(p2(v).values()) <= EQUAL
    for c in ("A", "B"):
        o = v["secondaries"]["S1_third_party"][c]
        assert o["p"] <= 0.01 if variant == "base" else o["mean"] > 0.5


@pytest.mark.parametrize("variant", ["base", "always70"])
def test_state(tmp_path, variant):
    ag = A.AGENTS["state"] if variant == "base" else A.realistic(A.AGENTS["state"], variant)
    v = run(ag, tmp_path)
    # clarification 2: at W1-T's draws the STRONG plant on C is realized weak (a 16-pair design
    # reaches CARRIES at the Holm-first bar ~90% of the time); the verdict may lose power, never
    # direction. ceiling_state below takes the CARRIES -> SELF EXCEEDS path for sure.
    assert p1(v)["C"] in {"CARRIES", "UNCLEAR"} and v["P1"]["C"]["mean"] > 0.6 and nullish(v, "AB")
    assert p2(v)["C"] in {"SELF EXCEEDS", "UNCLEAR"} and v["P2"]["C"]["mean_d"] > 0.1
    assert p2(v)["A"] in EQUAL and p2(v)["B"] in EQUAL
    assert not any(v["P2"][c]["at_ceiling"] for c in ABC)


def test_ceiling_state(tmp_path):
    v = run(A.AGENTS["ceiling_state"], tmp_path)
    assert p1(v)["C"] == "CARRIES" and nullish(v, "AB")
    assert p2(v)["C"] == "SELF EXCEEDS" and p2(v)["A"] in EQUAL and p2(v)["B"] in EQUAL
    assert not v["P2"]["C"]["at_ceiling"]          # the third party has nothing on C


def test_lean_only(tmp_path):
    v = run(A.AGENTS["lean_only"], tmp_path)
    assert p1(v) == {c: "AT CHANCE" for c in ABC}
    assert all(v["P1"][c]["mean"] == 0.5 for c in ABC)
    assert p2(v) == {c: "NO EXCESS" for c in ABC}
    assert v["secondaries"]["S7_reader_lean"]["DIA"]["share_1"] == 1.0


def test_blind_readers_block_primaries(tmp_path):
    v = run(A.AGENTS["blind"], tmp_path)
    assert not v["gate_G_R"]["pass"]
    assert set(p1(v).values()) == {"NOT ADJUDICABLE"} and set(p2(v).values()) == {"NOT ADJUDICABLE"}


def test_sparse(tmp_path):
    v = run(A.AGENTS["sparse"], tmp_path)
    assert v["P1"]["A"]["n"] == 11 and v["P1"]["A"]["verdict"] == "NOT ADJUDICABLE"
    assert v["P2"]["A"]["verdict"] == "NOT ADJUDICABLE"
    assert v["P1"]["B"]["verdict"] in NULLISH
    assert v["secondaries"]["S4_deflection"]["A-X-SELF"]["no_diagram"] == pytest.approx(5 / 16)


def test_weak(tmp_path):
    v = run(A.AGENTS["weak"], tmp_path)
    b = v["P1"]["A"]
    assert b["verdict"] != "NOT ADJUDICABLE" and 0.5 < b["mean"] < 0.75
    expect = "CARRIES" if b["p"] <= AN.ALPHA / 3 else (
        "AT CHANCE" if b["q975"] <= AN.BAR and b["binary_q975"] <= AN.BAR_BINARY else "UNCLEAR")
    assert b["verdict"] == expect


def test_ceiling_leak(tmp_path):
    """§7's added agent: both arms near perfect on A and B -> CARRIES, NO EXCESS (at ceiling)."""
    v = run(A.AGENTS["ceiling_leak"], tmp_path)
    assert p1(v)["A"] == p1(v)["B"] == "CARRIES" and nullish(v, "C")
    for c in ("A", "B"):
        assert v["P2"][c]["verdict"] == "NO EXCESS" and v["P2"][c]["at_ceiling"]
        assert v["P2"][c]["reported"] == "NO EXCESS (at ceiling)"
    assert not v["P2"]["C"]["at_ceiling"] and "(at ceiling)" not in v["P2"]["C"]["reported"]


def test_recompute_is_byte_identical(tmp_path):
    d = tmp_path / "w1t_flight_t"
    A.run_agent(A.AGENTS["input_leak"], d)
    AN.main([str(d)])
    first = (d / "verdict.json").read_bytes()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-m", "windows.analyze_w1t", str(d)], check=True, env=env,
                   cwd=str(A.Path(A.__file__).resolve().parents[1]), capture_output=True)
    assert (d / "verdict.json").read_bytes() == first
