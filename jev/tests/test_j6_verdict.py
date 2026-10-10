"""J6 verdict suite: planted agents with known structure land on their registered
branch of every primary, through the real analysis code (docs/JEV_J6_PROTOCOL.md §9).

The full registered plan (3,456 calls), with 2,000 resamples for speed (the
registered analysis uses 10,000; 2,000 still lets Holm reject at alpha .01 / 4).

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6_verdict.py -q
"""
import pytest

from jev import agents_j6 as A
from jev import analyze_j6 as Z
from jev import j6_plan as P

B = 2000


@pytest.fixture(scope="module")
def specs():
    return [P.strip(s) for s in P.flight()]


def run(specs, kind, **kw):
    return Z.analyze(specs, A.synthesize(specs, kind), B=B, pin=A.PIN, **kw)


def br(v):
    return {n: r["branch"] for n, r in v["primaries"].items()}


def test_echoer(specs):
    v = run(specs, "echoer")
    assert br(v) == {"P1": "MOVES", "P2": "MOVES", "P3": "HOLDS", "P4": "HOLDS"}, v["plain"]
    assert v["secondaries"]["S5_other_statement_echo"]["branch"] == "MOVES"
    assert v["secondaries"]["S3_negated_word_residue"]["mean_d"] > 0.1


def test_reader(specs):
    v = run(specs, "reader")
    assert br(v) == {"P1": "HOLDS", "P2": "MOVES", "P3": "MOVES", "P4": "MOVES"}, v["plain"]
    assert v["secondaries"]["S5_other_statement_echo"]["branch"] == "HOLDS"
    assert v["gates"]["G1_pass"] and v["gates"]["G2_pass"] and v["items"] == 24


def test_fixed(specs):
    v = run(specs, "fixed")
    b = br(v)
    assert b["P1"] == "HOLDS" and b["P2"] == "HOLDS", v["plain"]
    assert b["P3"] == b["P4"] == "NOT ADJUDICABLE (P2 did not move)"
    assert v["baseline"]["stable_label"] == "a decision model"


def test_door(specs):
    v = run(specs, "door")
    b = br(v)
    assert b["P1"] == "HOLDS" and b["P2"] == "HOLDS", v["plain"]
    assert v["baseline"]["stable_label"] == "cannot say"


def test_positional(specs):
    v = run(specs, "positional")
    b = br(v)
    assert b["P1"] == "HOLDS" and b["P2"] == "HOLDS", v["plain"]
    assert v["baseline"]["stable_label"] is None and v["baseline"]["verdict"] == "no stable label"
    s7 = v["secondaries"]["S7_position"]
    assert s7["argmax_position"] == 0 and s7["max_share"] > 0.99


def test_gates(specs):
    fin = A.synthesize(specs, "reader")
    # a second build string -> G1 fails -> nothing adjudicated
    k = next(iter(fin))
    fin2 = {c: (r if c != k else r | {"response": r["response"] | {"model": "typesafe/jev-1.14"}}) for c, r in fin.items()}
    v = Z.analyze(specs, fin2, B=200, pin=A.PIN, secondary=False)
    assert not v["gates"]["G1_pass"] and all(b == "NOT ADJUDICABLE (gates)" for b in br(v).values())
    # 2% of calls missing -> G2 fails
    drop = {s["call_id"] for n, s in enumerate(specs) if n % 50 == 0}
    v = Z.analyze(specs, {c: r for c, r in fin.items() if c not in drop}, B=200, pin=A.PIN, secondary=False)
    assert not v["gates"]["G2_pass"] and all(b == "NOT ADJUDICABLE (gates)" for b in br(v).values())
    # a wrong regenerated sha -> G4 fails
    regen = {s["call_id"]: s["body_sha"] for s in specs}
    regen[k] = "0" * 64
    v = Z.analyze(specs, fin, B=200, pin=A.PIN, regen=regen, secondary=False)
    assert not v["gates"]["G4_regenerability"]["pass"] and br(v)["P2"] == "NOT ADJUDICABLE (gates)"


def test_incomplete_items_drop_pairwise(specs):
    fin = A.synthesize(specs, "reader")
    # one rotation missing in SH/Q0 for 5 items: P2, P3, P4 drop them, P1 keeps all 24
    drop = {f"J6-SH-Q0-i{i:02d}-r3" for i in range(5)}
    v = Z.analyze(specs, {c: r for c, r in fin.items() if c not in drop}, B=200, pin=A.PIN, secondary=False)
    n = {k: r["n_items"] for k, r in v["primaries"].items()}
    assert n == {"P1": 24, "P2": 19, "P3": 19, "P4": 19}
    assert v["primaries"]["P2"]["branch"] == "NOT ADJUDICABLE (data)"
