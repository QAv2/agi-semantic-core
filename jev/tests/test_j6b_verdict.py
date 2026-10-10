"""J6b verdict suite: planted agents land on their registered branches through the
real analysis code (docs/JEV_J6B_PROTOCOL.md §5). Full plan, B = 2,000 for speed.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6b_verdict.py -q
"""
import pytest

from jev import agents_j6b as A
from jev import analyze_j6b as Z
from jev import j6b_plan as P

B = 2000


@pytest.fixture(scope="module")
def specs():
    return [P.strip(s) for s in P.flight()]


def run(specs, kind, **kw):
    return Z.analyze(specs, A.synthesize(specs, kind), B=B, pin=A.PIN, **kw)


def br(v):
    return {n: r["branch"] for n, r in v["primaries"].items()}


def cl(v):
    return {n: v["primaries"][n]["clause"]["branch"] for n in ("Q1", "Q2")}


def test_namer(specs):
    v = run(specs, "namer")
    assert br(v) == {"Q1": "MOVES", "Q2": "MOVES", "Q3": "MOVES", "Q4": "HOLDS"}, v["plain"]
    assert cl(v) == {"Q1": "SPECIFIC", "Q2": "SPECIFIC"}
    assert v["primaries"]["Q1"]["reading"] == "knows its name"
    assert v["primaries"]["Q2"]["reading"] == "recognizes itself from its maker and release date"


def test_pointer(specs):
    v = run(specs, "pointer")
    assert br(v) == {"Q1": "HOLDS", "Q2": "HOLDS", "Q3": "MOVES", "Q4": "MOVES"}, v["plain"]
    assert v["primaries"]["Q4"]["reading"] == "the pointer does work the name alone doesn't"


def test_anylabel(specs):
    v = run(specs, "anylabel")
    assert br(v) == {"Q1": "MOVES", "Q2": "MOVES", "Q3": "MOVES", "Q4": "HOLDS"}, v["plain"]
    assert cl(v) == {"Q1": "NOT SPECIFIC", "Q2": "NOT SPECIFIC"}
    assert v["primaries"]["Q1"]["reading"] == "takes a label on any named system, Jev or not"
    r3 = v["secondaries"]["R3_other_system_labels"]
    assert r3["ONAME"]["branch"] == "MOVES" and r3["ODESC"]["branch"] == "MOVES"


def test_contraster_clause_cannot_rescue(specs):
    v = run(specs, "contraster")
    assert br(v) == {"Q1": "HOLDS", "Q2": "HOLDS", "Q3": "MOVES", "Q4": "MOVES"}, v["plain"]
    # the difference of differences comes out positive because the control went negative...
    assert cl(v) == {"Q1": "SPECIFIC", "Q2": "SPECIFIC"}
    # ...and the registered reading still says no, because raw uptake is absent
    assert v["primaries"]["Q1"]["reading"] == "doesn't connect the name Jev to itself"
    assert v["secondaries"]["R3_other_system_labels"]["ONAME"]["branch"] == "REVERSED"


def test_fixed_fails_the_pointer_check(specs):
    v = run(specs, "fixed")
    b = br(v)
    assert b["Q3"] == "HOLDS", v["plain"]
    assert all(b[n] == "NOT ADJUDICABLE (pointer check failed)" for n in ("Q1", "Q2", "Q4"))


def test_gates(specs):
    fin = A.synthesize(specs, "namer")
    drop = {s["call_id"] for n, s in enumerate(specs) if n % 50 == 0}
    v = Z.analyze(specs, {c: r for c, r in fin.items() if c not in drop}, B=200, pin=A.PIN, secondary=False)
    assert not v["gates"]["G2_pass"] and all(b == "NOT ADJUDICABLE (gates)" for b in br(v).values())
    v = Z.analyze(specs, fin, B=200, pin="typesafe/jev-9", secondary=False)
    assert not v["gates"]["G1_pass"] and all(b == "NOT ADJUDICABLE (gates)" for b in br(v).values())
