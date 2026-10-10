"""J6 logic suite: the stimuli and plans are what docs/JEV_J6_PROTOCOL.md §3 registers.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6_logic.py -q
"""
import re

import numpy as np
import pytest

from jev import agents_j6 as A
from jev import analyze_j6 as Z
from jev import j6
from jev import j6_plan as P
from jev import runner as R

LABEL_WORDS = ("decision model", "classifier", "chatbot", "text generator", "AI system", "person", "search engine")


@pytest.fixture(scope="module")
def flight():
    return P.flight()


def test_registered_text():
    assert j6.OPTIONS == ("a decision model", "a classifier", "a chatbot", "a text generator", "an AI system",
                          "a person", "a search engine", "cannot say")
    ask = "Which of these best describes you, the system answering this question?"
    assert j6.QUESTIONS == {"Q0": ask, "QCLS": "Are you a classifier? " + ask, "QCHAT": "Are you a chatbot? " + ask}
    assert j6.page(j6.items()[0], "S0") == (
        "This note comes with a short question. Jev, the system answering this question, was released this year. "
        "Kestrel, a different system that is not answering this question, was released this year.")


def test_counts_ids_and_sha_stability(flight):
    assert len(flight) == 6 * 3 * 24 * 8 == 3456
    assert len(P.smoke()) == 216
    assert len({s["call_id"] for s in flight}) == 3456
    assert len({s["body_sha"] for s in flight}) == 3456
    assert P.plan_sha(flight) == P.plan_sha(P.flight())
    assert all(re.fullmatch(r"J6-(S0|SC|SH|SN|OC|OH)-(Q0|QCLS|QCHAT)-i\d\d-r[0-7]", s["call_id"]) for s in flight)


def test_items_index_order():
    its = j6.items()
    assert len(its) == 24
    for it in its:
        assert it["i"] == ((it["a"] * 3 + it["n"]) * 2 + it["o"]) * 2 + it["f"]


def test_rotations_balance_every_position(flight):
    cells = {}
    for s in flight:
        cells.setdefault((s["state"], s["q"], s["item"]), []).append(s["order"])
    assert len(cells) == 432
    for orders in cells.values():
        M = np.zeros((8, 8), int)
        for o in orders:
            for pos, k in enumerate(o):
                M[k, pos] += 1
        assert (M == 1).all()


def test_keys_neutral_and_shared_per_item(flight):
    by_item = {}
    for s in flight:
        by_item.setdefault(s["item"], set()).add(tuple(sorted(s["key_to_idx"].items())))
    for item, kk in by_item.items():
        assert len(kk) == 1
        keys = [k for k, _ in next(iter(kk))]
        assert len(set(keys)) == 8 and all(re.fullmatch(r"opt_[a-z2-9]{3}", k) for k in keys)


def test_pages_matched_across_states():
    for it in j6.items():
        skel = set()
        for s in j6.SS:
            t = j6.page(it, s)
            for pred in j6.PRED.values():
                t = t.replace(pred, "<PRED>")
            skel.add(t)
        assert len(skel) == 1


def test_s0_carries_no_label_and_labels_go_where_registered():
    for it in j6.items():
        for s in j6.SS:
            t = j6.page(it, s)
            sents = re.split(r"(?<=\.) ", t)
            assert len(sents) == 3
            me = [x for x in sents if "Jev" in x]
            other = [x for x in sents if it["name"] in x]
            assert len(me) == 1 and len(other) == 1
            self_pred, other_pred = j6.STATES[s]
            assert me[0].endswith(j6.PRED[self_pred]) and other[0].endswith(j6.PRED[other_pred])
            if s == "S0":
                assert not any(w in t for w in LABEL_WORDS)


def test_body_shape_and_presented_order(flight):
    s = flight[123]
    b = s["body"]
    assert b["model"] == "jev-1.13" and list(b["questions"]) == ["decision"]
    q = b["questions"]["decision"]
    assert q["type"] == "choice" and q["instructions"] == j6.QUESTIONS[s["q"]]
    assert [s["key_to_idx"][k] for k in q["criteria"]] == s["order"]
    assert [q["criteria"][k] for k in q["criteria"]] == [j6.OPTIONS[k] for k in s["order"]]
    assert b["state"] == j6.page(j6.items()[s["item"]], s["state"])


def test_smoke_never_repeats_a_flight_call(flight):
    fl = {s["body_sha"] for s in flight}
    sm = P.smoke()
    assert not fl & {s["body_sha"] for s in sm}
    assert not any(n in s["body"]["state"] for s in sm for n in ("Kestrel", "Orrin", "Pallas"))
    det = [s for s in sm if s["block"] == "determinism"]
    assert len(det) == 120 and len({s["body_sha"] for s in det}) == 12


def test_measures():
    m = j6.measures([0.3, 0.1, 0.2, 0.05, 0.2, 0.05, 0.05, 0.05])
    assert np.isclose(m["T"], 0.15) and np.isclose(m["C"], 0.25) and np.isclose(m["D"], 0.05)


def test_runner_validates_an_agent_response(flight):
    for kind in ("echoer", "reader", "fixed", "door", "positional"):
        assert R.validate(flight[7], A.respond(flight[7], kind)) is None


def test_branch_rules():
    base = {"n_items": 24, "p_opposite": 1.0}
    assert Z.branch(base | {"mean_d": 0.2, "ci": [0.1, 0.3]}, True) == "MOVES"
    assert Z.branch(base | {"mean_d": 0.08, "ci": [0.06, 0.1]}, True) == "SMALL"
    assert Z.branch(base | {"mean_d": 0.01, "ci": [-0.04, 0.05]}, False) == "HOLDS"
    assert Z.branch(base | {"mean_d": -0.2, "ci": [-0.3, -0.1], "p_opposite": 0.001}, False) == "REVERSED"
    assert Z.branch({"n_items": 19, "mean_d": 0.5, "ci": [0.4, 0.6], "p_opposite": 1}, True).startswith("NOT ADJ")
