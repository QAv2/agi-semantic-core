"""J5 logic suite: the stimuli and plans are what docs/JEV_J5_PROTOCOL.md §3 registers,
the human inputs load by sha, and the branch rules behave.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j5_logic.py -q
"""
import re

import numpy as np
import pytest

from jev import agents_j5 as A
from jev import analyze_j5 as Z
from jev import j5
from jev import j5_plan as P
from jev import runner as R


@pytest.fixture(scope="module")
def flight():
    return P.flight()


def test_registered_text():
    assert j5.COLORS == ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "turquoise",
                         "white", "yellow")
    assert len(j5.SURVEY) == 20 and j5.EXTENSION == ("fun",) and j5.SHAMS == ("zorvish", "plemt")
    it0, it3 = j5.items()[0], j5.items()[3]
    assert j5.page(it0) == "This is a question about colors and feelings."
    assert j5.page(it3) == "The question below is part of a short survey about colors."
    q = j5.yn_question(it0, "sadness", "black", 0)
    assert q == {"type": "noul", "instructions": "Does the color black go with the feeling of sadness?",
                 "criteria": {"true": "Yes, the color black goes with the feeling of sadness.",
                              "false": "No, the color black does not go with the feeling of sadness."}}
    q1 = j5.yn_question(it3, "fun", "pink", 1)
    assert q1["instructions"] == "Do people associate the color pink with the feeling of fun?"
    assert list(q1["criteria"]) == ["false", "true"]
    assert q1["criteria"]["false"] == "No, people do not associate the color pink with the feeling of fun."
    assert j5.CH[0].format(f="pride") == "Which color goes best with the feeling of pride?"
    assert j5.CH[1].format(f="pride") == "Which color do people most associate with the feeling of pride?"


def test_counts_ids_and_sha_stability(flight):
    yn = [s for s in flight if s["fmt"] == "YN"]
    ch = [s for s in flight if s["fmt"] == "CH"]
    assert len(yn) == 23 * 12 * 4 * 2 == 2208 and len(ch) == 23 * 4 * 12 == 1104 and len(flight) == 3312
    assert len({s["call_id"] for s in flight}) == 3312 and len({s["body_sha"] for s in flight}) == 3312
    assert P.plan_sha(flight) == P.plan_sha(P.flight()) and len(P.smoke()) == 216
    fe = "|".join(j5.FEELINGS["flight"])
    co = "|".join(j5.COLORS)
    assert all(re.fullmatch(rf"J5-YN-({fe})-({co})-i[0-3]-o[01]", s["call_id"]) for s in yn)
    assert all(re.fullmatch(rf"J5-CH-({fe})-i[0-3]-r(0\d|1[01])", s["call_id"]) for s in ch)


def test_items_and_yn_cells(flight):
    assert [it["i"] for it in j5.items()] == [0, 1, 2, 3]
    assert all(it["i"] == it["w"] * 2 + it["f"] for it in j5.items())
    cells = {}
    for s in flight:
        if s["fmt"] == "YN":
            cells.setdefault((s["feeling"], s["color"]), set()).add((s["item"], s["o"]))
    assert len(cells) == 23 * 12 and all(len(v) == 8 for v in cells.values())


def test_rotations_balance_every_position(flight):
    cells = {}
    for s in flight:
        if s["fmt"] == "CH":
            cells.setdefault((s["feeling"], s["item"]), []).append(s["order"])
    assert len(cells) == 23 * 4
    for orders in cells.values():
        M = np.zeros((12, 12), int)
        for o in orders:
            for pos, k in enumerate(o):
                M[k, pos] += 1
        assert (M == 1).all()


def test_keys_neutral_and_shared_per_item_feeling(flight):
    by = {}
    for s in flight:
        if s["fmt"] == "CH":
            by.setdefault((s["feeling"], s["item"]), set()).add(tuple(sorted(s["key_to_idx"].items())))
            assert all(re.fullmatch(r"opt_[a-z2-9]{3}", k) for k in s["key_to_idx"])
    assert all(len(v) == 1 for v in by.values())
    assert len({next(iter(v)) for v in by.values()}) == len(by)


def test_page_carries_no_answer(flight):
    frames = set(j5.FRAMES["flight"])
    for s in flight:
        st = s["body"]["state"]
        assert st in frames
        assert not any(c in st for c in j5.COLORS) and not any(f in st for f in j5.FEELINGS["flight"])


def test_smoke_never_repeats_a_flight_call(flight):
    sm = P.smoke()
    assert not {s["body_sha"] for s in flight} & {s["body_sha"] for s in sm}
    assert {s["feeling"] for s in sm} == {"surprise", "boredom"}
    assert {s["body"]["state"] for s in sm} <= set(j5.FRAMES["smoke"])
    det = [s for s in sm if s["block"] == "determinism"]
    thr = [s for s in sm if s["block"] == "throughput"]
    assert len(det) == 120 and len({s["body_sha"] for s in det}) == 12
    assert len(thr) == 96 and len({s["body_sha"] for s in thr}) == 96
    assert not {s["body_sha"] for s in det} & {s["body_sha"] for s in thr}
    assert sum(s["fmt"] == "YN" for s in thr) == 48


def test_body_shape(flight):
    s = next(x for x in flight if x["fmt"] == "CH" and x["r"] == 5)
    q = s["body"]["questions"]["decision"]
    assert s["body"]["model"] == "jev-1.13" and q["type"] == "choice"
    assert [s["key_to_idx"][k] for k in q["criteria"]] == s["order"]
    assert all(q["criteria"][k] == f"the color {j5.COLORS[i]}" for k, i in s["key_to_idx"].items())
    y = next(x for x in flight if x["fmt"] == "YN")
    assert list(y["body"]["questions"]) == [j5.YN_Q] and y["questions"] == [j5.YN_Q]


def test_runner_validates_agent_responses(flight):
    yn = next(x for x in flight if x["fmt"] == "YN")
    ch = next(x for x in flight if x["fmt"] == "CH")
    for kind in ("people", "english", "rest", "sharp", "scrambled", "fixed"):
        assert R.validate(yn, A.respond(yn, kind)) is None
        assert R.validate(ch, A.respond(ch, kind)) is None


def test_human_inputs_load_by_sha():
    h = Z.load_human()
    assert h["global"].shape == (12, 20) and len(h["panels"]) == 61 and len(h["dictionary"]) == 15
    E_, R_, n_rest, W_ = Z.english_rest(h["panels"])
    assert n_rest == 26 and Z.W0 == pytest.approx(3 / 29)
    assert np.allclose(W_, (3 * E_ + 26 * R_) / 29)
    # the registered floor is the least typical nation's agreement with the other 29
    nations = [p for p in h["panels"] if p != "Global" and not p.startswith("Global-")]
    ys = [Z.r_dc(h["panels"][n], h["panels"]["Global-" + n]) for n in nations]
    assert round(min(ys), 3) == Z.FLOOR and round(float(np.median(ys)), 3) == Z.TYPICAL


def test_isotonic_and_curve():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(Z.isotonic(x, [1, 2, 3, 4]), [1, 2, 3, 4])
    assert np.allclose(Z.isotonic(x, [4, 3, 2, 1]), [2.5] * 4)
    assert np.allclose(Z.isotonic([0.1, 0.1, 0.2], [1, 3, 0]), [4 / 3] * 3)    # ties pooled first
    W = np.random.default_rng(0).random((12, 20))
    g = Z.curve(1 / (1 + np.exp(-20 * (W - 0.5))), W)
    assert np.allclose(g(W), 1 / (1 + np.exp(-20 * (W - 0.5))))


def test_sharpened_matches_entropy():
    h = np.linspace(0.05, 0.6, 12)        # distinct values, so every entropy is reachable
    for t in (0.5, 1.7, 3.0):
        q, k = Z.sharpened(h, t)
        assert Z.entropy_bits(q) == pytest.approx(t, abs=0.01) and 0 < k < 200
    q, k = Z.sharpened(h, 4.0)            # flatter than uniform asked: uniform
    assert k == 0.0 and np.allclose(q, 1 / 12)
    q, k = Z.sharpened(h, 0.0)            # all on one color asked: the sharpest allowed
    assert k == 200.0 and q.max() > 0.99


def test_c2_branch_rules():
    base = {"dropped_share": 0.0, "p_opposite": 0.5, "ci": [0.1, 0.4], "w": 0.3}
    assert Z.c2_branch(base, True) == "LEANS ENGLISH"
    assert Z.c2_branch(base | {"w": 0.2}, True) == "UNCLEAR"          # significant but under a quarter
    assert Z.c2_branch(base | {"w": 0.2, "ci": [0.15, 0.24]}, True) == "NO LEAN"
    assert Z.c2_branch(base | {"w": 0.0, "p_opposite": 0.001, "ci": [-0.02, 0.03]}, False) == "LEANS AWAY"
    assert Z.c2_branch(base | {"dropped_share": 0.06}, True) == "UNCLEAR (unstable fit)"
    assert Z.c2_branch({"branch": "NOT ADJUDICABLE (no variance)"}, True) == "NOT ADJUDICABLE (no variance)"
