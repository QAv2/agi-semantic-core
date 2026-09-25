"""Logic suite: seeds, IDs, orders, G6 ideal-solver invariance, plan counts and
determinism, J2 arm invariants. No network.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_logic.py -q
"""
import re
from collections import Counter

import numpy as np
import pytest

from jev import common as C
from jev import j1, j2
from jev import plan as P

FZ = {"families": {"F1": {"included": True, "d0": 0.71, "d1": 1.0, "strength": "standard"},
                   "F2": {"included": True, "d0": 2.0, "d1": 1.41, "strength": "strong"},
                   "F3": {"included": True, "d0": 0.60, "d1": 0.70, "strength": "standard"},
                   "F1K2": {"included": True, "d0": 0.5}},
      "bundle_a6": True}


def test_stage_seeds_disjoint():
    a = C.rng("flight", "J1", "F1", "base", 0).random(8)
    b = C.rng("pilot", "J1", "F1", "base", 0).random(8)
    c = C.rng("smoke", "J1", "F1", "base", 0).random(8)
    assert not np.allclose(a, b) and not np.allclose(a, c) and not np.allclose(b, c)
    assert np.allclose(a, C.rng("flight", "J1", "F1", "base", 0).random(8))


def test_neutral_ids():
    ids = C.neutral_ids(C.rng("smoke", "ids-test"), 500)
    assert len(set(ids)) == 500
    assert all(re.fullmatch(r"opt_[a-hj-km-np-z2-9]{3}", k) for k in ids)


def test_balanced_truth_position():
    pos = Counter()
    for i in range(400):
        for rep in range(3):
            truth = i % 4
            o = C.balanced_order(i, rep, 4, truth, C.rng("smoke", "bo", i, rep))
            assert sorted(o) == [0, 1, 2, 3] and o[(i + rep) % 4] == truth
            pos[o.index(truth)] += 1
    assert set(pos.values()) == {300}


@pytest.mark.parametrize("fam,level", [("F1", 0.71), ("F1K2", 0.5), ("F2", 1.41), ("F3", 0.6)])
def test_g6_ideal_solver_invariance(fam, level):
    """Registered G6: 500 stimuli per family, A0 vs both A1 strengths (and A3):
    the text-only parser recovers the identical ideal posterior."""
    worst = 0.0
    for i in range(500):
        st = j1.generate(fam, C.rng("smoke", "g6", fam, i), level)
        order = C.balanced_order(i, 0, st["K"], st["truth"], C.rng("smoke", "g6o", fam, i))
        ref = j1.ideal(st)
        for arm, s in (("A0", None), ("A1", "standard"), ("A1", "strong"), ("A3", None)):
            state, opts, k2i = j1.render(st, arm, order, C.rng("smoke", "g6r", fam, i, arm, str(s)),
                                         C.rng("smoke", "g6i", fam, i), s)
            assert [k for k, _ in opts] == [next(k for k, v in k2i.items() if v == idx) for idx in order]
            worst = max(worst, float(np.abs(j1.ideal_from_text(state, st, k2i) - ref).max()))
    assert worst < 1e-9


@pytest.mark.parametrize("fam", ["F1", "F2", "F3"])
def test_a2_carries_no_evidence(fam):
    st = j1.generate(fam, C.rng("smoke", "a2", fam), P.GRIDS[fam][3])
    order = C.balanced_order(0, 0, 4, st["truth"], C.rng("smoke", "a2o", fam))
    state, _, _ = j1.render(st, "A2", order, C.rng("smoke", "a2r"), C.rng("smoke", "a2i"))
    if fam == "F1":
        _, body = j1._table_rows(state[state.index("|"):])
        assert all(c == j1.NA for row in body for c in row[1:])
    elif fam == "F2":
        reading, _ = j1.parse_evidence(state, "F2")
        assert all(v is None for v in reading)
    else:
        assert all(k is None for _, k in j1.parse_evidence(state, "F3"))
        assert not any(ln.startswith("- ") and "appears clear" in ln for ln in state.splitlines())


def test_a1_changes_presentation():
    st = j1.generate("F1", C.rng("smoke", "a1"), 1.0)
    order = [0, 1, 2, 3]
    s0, o0, _ = j1.render(st, "A0", order, C.rng("smoke", "x"), C.rng("smoke", "y"))
    s1, o1, _ = j1.render(st, "A1", order, C.rng("smoke", "x"), C.rng("smoke", "y"), "standard")
    assert s0 != s1 and o0 == o1 and "Legend:" in s1 and "Legend:" not in s0


def test_counts_smoke_pilot_and_determinism():
    sm, pi = P.smoke(), P.j1_pilot()
    assert len(sm) == 740 and len(pi) == 2920
    assert len({s["call_id"] for s in sm}) == 740 and len({s["call_id"] for s in pi}) == 2920
    assert P.plan_sha(sm) == P.plan_sha(P.smoke())
    assert P.plan_sha(pi) == P.plan_sha(P.j1_pilot())


def test_smoke_blocks():
    sm = P.smoke()
    det = [s for s in sm if s.get("block") == "determinism"]
    groups = Counter(re.sub(r"-x\d+$", "", s["call_id"]) for s in det)
    assert len(groups) == 24 and set(groups.values()) == {10}
    by = {}
    for s in det:
        by.setdefault(re.sub(r"-x\d+$", "", s["call_id"]), set()).add(s["body_sha"])
    assert all(len(v) == 1 for v in by.values())          # identical requests
    thr = Counter(s["concurrency"] for s in sm if s.get("block") == "throughput")
    assert thr == {1: 40, 2: 40, 4: 40, 8: 40, 16: 40}
    iso = [s for s in sm if s.get("block") == "isolation"]
    assert len(iso) == 300
    assert {tuple(s["questions"]) for s in iso} == {("decision",), ("decision", "determined", "insufficient")}


def test_j1_flight_counts_and_pairing():
    specs = P.j1_flight(FZ)
    assert len(specs) == 20400
    un = P.j1_flight(dict(FZ, bundle_a6=False))
    assert len(un) == 20400 + 3 * 3 * 400 * 3 + 2 * 400 * 3
    orders = {}
    for s in specs:
        if s["set"] == "base":
            orders.setdefault((s["family"], s["stim"], s["rep"]), set()).add(tuple(s["order"]))
    assert all(len(v) == 1 for v in orders.values())       # A0/A1-same/A2/A3 share the order
    for s in specs[:50]:
        assert C.body_sha(s["body"]) == s["body_sha"]
        assert s["body"]["model"] == "jev-1.13" and isinstance(s["body"]["state"], str)


def test_flight_sha_matches_parts():
    j1s, j2s = P.j1_flight(FZ, n=6), P.j2_flight(3)
    sha, n = P.flight_sha(FZ, 6, 3)
    assert sha == P.plan_sha(j1s + j2s) and n == len(j1s) + len(j2s)


@pytest.fixture(scope="module")
def j2specs():
    return P.j2_flight()


def test_j2_counts(j2specs):
    assert len(j2specs) == 30240
    assert len({s["call_id"] for s in j2specs}) == 30240


def test_j2_rung9_identical_across_c(j2specs):
    by = {}
    for s in j2specs:
        if s["arm"] == "M-walked" and s["rung"] == 9:
            by.setdefault((s["ladder"], s["rep"]), set()).add(s["body_sha"])
    assert all(len(v) == 1 for v in by.values())


def test_j2_named_differs_only_in_preface_and_hold(j2specs):
    idx = {(s["arm"], s["ladder"], s["rung"], s["c"], s["rep"]): s for s in j2specs}
    for L in (0, 57, 119):
        for k in (0, 5, 8, 9):
            w, n = idx[("M-walked", L, k, 3, 1)], idx[("M-named", L, k, 3, 1)]
            assert n["body"]["state"] == j2.NAMED_PREFACE + "\n\n" + w["body"]["state"]
            cw = w["body"]["questions"]["decision"]["criteria"]
            cn = n["body"]["questions"]["decision"]["criteria"]
            diff = [(a, b) for a, b in zip(cw.items(), cn.items()) if a != b]
            assert len(diff) == 1 and diff[0][1][0] == "witness"
            assert list(cw.values()).index(diff[0][0][1]) == list(cn.values()).index(diff[0][1][1])


def test_j2_forced_and_sham(j2specs):
    for s in j2specs:
        st = s["body"]["state"]
        crit = s["body"]["questions"]["decision"]["criteria"]
        if s["arm"].startswith("M-forced"):
            assert len(crit) == 4 and "- no action: 0" not in st
        else:
            assert len(crit) == 5
        if s["arm"] in ("M-sham", "M-forced-sham"):
            src = [ln for ln in st.splitlines() if "(reliability" in ln]
            assert all(j2.OFFLINE not in ln for ln in src)
            other = st[st.index("Other data:"):st.index("Payoffs")]
            assert other.count(j2.OFFLINE) == s["rung"]
        if s["arm"] == "M-walked" and s["rung"] <= 8:
            src = [ln for ln in st.splitlines() if "(reliability" in ln]
            assert sum(j2.OFFLINE in ln for ln in src) == s["rung"]
            assert f"- wrong action: -{s['c']}" in st


def test_j2_history_arms(j2specs):
    idx = {(s["arm"], s["ladder"], s["rung"], s["rep"]): s for s in j2specs if s["c"] == 3}
    for L in range(0, 120, 17):
        w, r, d = (idx[(a, L, 8, 0)] for a in ("H-walked", "H-reversed", "H-walked-dup"))
        assert w["body_sha"] == d["body_sha"]
        cut = lambda st: st[:st.index("Log of this situation:")] + st[st.index("Payoffs"):]
        assert cut(w["body"]["state"]) == cut(r["body"]["state"])
        assert w["body"]["state"] != r["body"]["state"]
        assert w["order"] == r["order"]
        assert np.allclose(w["ideal"], 0.25)


def test_j2_position_balance(j2specs):
    tp, hp = Counter(), Counter()
    for s in j2specs:
        if len(s["order"]) == 5:
            tp[s["order"].index(s["truth"])] += 1
            hp[s["order"].index(4)] += 1
    for cnt in (tp, hp):
        m = np.mean(list(cnt.values()))
        assert len(cnt) == 5 and all(abs(v - m) / m < 0.15 for v in cnt.values())
