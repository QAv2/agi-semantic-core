"""J5 verdict suite: planted agents land on their registered branches through the real
analysis code (docs/JEV_J5_PROTOCOL.md §7). Full plan, B = 1,000 for speed.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j5_verdict.py -q
"""
import pytest

from jev import agents_j5 as A
from jev import analyze_j5 as Z
from jev import j5_plan as P

B = 1000


@pytest.fixture(scope="module")
def specs():
    return [P.strip(s) for s in P.flight()]


@pytest.fixture(scope="module")
def human():
    return Z.load_human()


def run(specs, human, kind):
    return Z.analyze(specs, A.synthesize(specs, kind), B=B, pin=A.PIN, human=human)


def br(v):
    return {n: v["primaries"][n]["branch"] for n in ("C1", "C2")}


def test_people(specs, human):
    v = run(specs, human, "people")
    assert v["gates"]["G5_pass"]
    assert br(v) == {"C1": "WITHIN THE HUMAN RANGE", "C2": "NO LEAN"}, v["plain"]


def test_english(specs, human):
    v = run(specs, human, "english")
    assert br(v) == {"C1": "WITHIN THE HUMAN RANGE", "C2": "LEANS ENGLISH"}, v["plain"]


def test_rest(specs, human):
    v = run(specs, human, "rest")
    assert br(v) == {"C1": "WITHIN THE HUMAN RANGE", "C2": "LEANS AWAY"}, v["plain"]


def test_sharp(specs, human):
    v = run(specs, human, "sharp")
    b = br(v)
    assert b["C1"] == "WITHIN THE HUMAN RANGE" and b["C2"] != "LEANS ENGLISH", v["plain"]
    assert abs(v["secondaries"]["S3_choice_format"]["efficiency"] - 1) <= 0.05


def test_scrambled(specs, human):
    v = run(specs, human, "scrambled")
    assert not v["gates"]["G5_pass"]
    b = br(v)
    assert b["C1"] == "BELOW THE HUMAN RANGE" and b["C2"].startswith("NOT ADJUDICABLE"), v["plain"]


def test_fixed(specs, human):
    v = run(specs, human, "fixed")
    assert br(v) == {"C1": "NOT ADJUDICABLE (no variance)", "C2": "NOT ADJUDICABLE (no variance)"}, v["plain"]
