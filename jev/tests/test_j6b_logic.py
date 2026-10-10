"""J6b logic suite: the stimuli and plans are what docs/JEV_J6B_PROTOCOL.md §1 registers.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6b_logic.py -q
"""
import re

import numpy as np
import pytest

from jev import agents_j6b as A
from jev import analyze_j6b as Z
from jev import j6
from jev import j6b
from jev import j6b_plan as P
from jev import runner as R

LABEL_WORDS = ("decision model", "classifier", "chatbot", "text generator", "AI system", "person", "search engine")


@pytest.fixture(scope="module")
def flight():
    return P.flight()


def test_registered_text():
    assert j6b.OPTIONS == j6.OPTIONS and j6b.QUESTION == j6.QUESTIONS["Q0"]
    it = j6b.items()[0]
    assert j6b.page(it, "NAME-CHAT") == ("This note comes with a short question. Jev is a chatbot and a text "
                                          "generator. Models of many kinds are in use today.")
    assert j6b.page(it, "DESC-CLS") == ("This note comes with a short question. The model that TypeSafe AI released "
                                         "in September 2026 is a decision model and a classifier. Models of many "
                                         "kinds are in use today.")
    assert j6b.page(it, "ODESC-CLS").startswith("This note comes with a short question. The model that Kestrel Labs "
                                                 "released in September 2026 is")
    assert len(j6b.STATES) == 11


def test_counts_ids_and_sha_stability(flight):
    assert len(flight) == 11 * 24 * 8 == 2112 and len(P.smoke()) == 216
    assert len({s["call_id"] for s in flight}) == 2112 and len({s["body_sha"] for s in flight}) == 2112
    assert P.plan_sha(flight) == P.plan_sha(P.flight())
    assert all(re.fullmatch(r"J6B-(NONE|(POINT|NAME|DESC|ONAME|ODESC)-(CLS|CHAT))-i\d\d-r[0-7]", s["call_id"])
               for s in flight)


def test_items_index_order_and_factors_vary_every_mode():
    its = j6b.items()
    assert len(its) == 24
    for it in its:
        assert it["i"] == ((it["n"] * 2 + it["o"]) * 2 + it["f"]) * 2 + it["w"]
    for s in j6b.STATES:
        # NONE has no predicate, so the label word order can't vary it (clarification 1)
        assert len({j6b.page(it, s) for it in its}) == (12 if s == "NONE" else 24)


def test_rotations_balance_every_position(flight):
    cells = {}
    for s in flight:
        cells.setdefault((s["state"], s["item"]), []).append(s["order"])
    assert len(cells) == 264
    for orders in cells.values():
        M = np.zeros((8, 8), int)
        for o in orders:
            for pos, k in enumerate(o):
                M[k, pos] += 1
        assert (M == 1).all()


def test_keys_neutral_shared_per_item_and_new_stream(flight):
    by_item = {}
    for s in flight:
        by_item.setdefault(s["item"], set()).add(tuple(sorted(s["key_to_idx"].items())))
    assert all(len(v) == 1 for v in by_item.values())
    assert j6b.keys(j6b.items()[0]) != j6.keys(j6.items()[0])


def test_label_pairs_differ_only_in_the_predicate():
    for it in j6b.items():
        for m in j6b.MODES:
            a, b = j6b.page(it, f"{m}-CLS"), j6b.page(it, f"{m}-CHAT")
            assert a.replace(j6b.PRED["CLS"][it["w"]], "<P>") == b.replace(j6b.PRED["CHAT"][it["w"]], "<P>")


def test_reference_modes_carry_only_their_reference():
    for it in j6b.items():
        for s in j6b.STATES:
            t = j6b.page(it, s)
            mode, _ = j6b.split(s)
            assert ("answering this question" in t) == (mode == "POINT")
            assert ("Jev" in t) == (mode in ("POINT", "NAME"))
            assert ("TypeSafe" in t) == (mode == "DESC")
            assert (it["name"] in t) == (mode in ("ONAME", "ODESC"))
            if mode is None:
                assert not any(w in t for w in LABEL_WORDS)


def test_smoke_never_repeats_a_flight_call(flight):
    sm = P.smoke()
    assert not {s["body_sha"] for s in flight} & {s["body_sha"] for s in sm}
    assert not any(n in s["body"]["state"] for s in sm for n in ("Kestrel", "Orrin", "Pallas"))
    det = [s for s in sm if s["block"] == "determinism"]
    assert len(det) == 120 and len({s["body_sha"] for s in det}) == 12


def test_body_shape(flight):
    s = flight[777]
    q = s["body"]["questions"]["decision"]
    assert s["body"]["model"] == "jev-1.13" and q["instructions"] == j6b.QUESTION and s["q"] == "Q0"
    assert [s["key_to_idx"][k] for k in q["criteria"]] == s["order"]
    assert s["body"]["state"] == j6b.page(j6b.items()[s["item"]], s["state"])


def test_runner_validates_agent_responses(flight):
    for kind in ("namer", "pointer", "anylabel", "contraster", "fixed"):
        assert R.validate(flight[11], A.respond(flight[11], kind)) is None


def test_clause_branch_and_reading_rules():
    assert Z.clause_branch({"mean_d": 0.3, "p": 0.001, "ci": [0.2, 0.4]}) == "SPECIFIC"
    assert Z.clause_branch({"mean_d": 0.01, "p": 0.3, "ci": [-0.03, 0.04]}) == "NOT SPECIFIC"
    assert Z.clause_branch({"mean_d": 0.08, "p": 0.001, "ci": [0.06, 0.1]}) == "UNCLEAR"
    sp = {"branch": "SPECIFIC"}
    assert Z.reading("Q1", {"branch": "MOVES", "clause": sp}) == "knows its name"
    # a clause can't rescue a primary that HOLDS
    assert Z.reading("Q1", {"branch": "HOLDS", "clause": sp}) == "doesn't connect the name Jev to itself"
    assert Z.reading("Q2", {"branch": "MOVES", "clause": {"branch": "NOT SPECIFIC"}}) == \
        "takes a label on any described model"
