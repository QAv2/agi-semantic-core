"""J3 logic suite (protocol §3–§6, §8 G6'): data, counts and balance, pairing,
the information identity between presentations and the fitted bar, the reference
card's truth, the genre firewall, determinism.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j3_logic.py -q
"""
import collections
import re

import numpy as np
import pytest

from jev import j3
from jev import j3_plan as P


@pytest.fixture(scope="module")
def stims():
    return j3.flight_stimuli()


@pytest.fixture(scope="module")
def specs(stims):
    return [s for _, part in P.flight_parts(True, stims) for s in part]


def test_data_dedupe_halves_register():
    g = j3.data()
    assert len(g["S_all"]) == 320 and len(g["U"]) == 258
    assert np.bincount(g["half"]).tolist() == [129, 129]
    assert np.abs(j3.reg(g["S_all"]) - g["stored"]).max() < 1e-5
    assert sorted(m["copies"] for m in g["smeta"])[-2:] == [16, 48]


def test_counts_and_balance(stims):
    assert len(stims) == 3096
    kinds = collections.Counter((s["kind"], s["dose"]) for s in stims)
    assert kinds[("untouched", 0.0)] == 258 and kinds[("random", 0.5)] == 258 and kinds[("push", 0.5)] == 258
    assert all(kinds[("random", a)] == 86 for a in j3.THR)
    assert all(kinds[("push", a)] == 516 for a in j3.THR + j3.PSY)
    per = collections.Counter((s["dose"], s["truth"]) for s in stims if s["kind"] == "push")
    for a in j3.THR + j3.PSY:
        assert {per[(a, k)] for k in range(13)} <= {39, 40}
    assert {per[(0.5, k)] for k in range(13)} <= {19, 20}
    by_state = collections.defaultdict(list)
    for s in stims:
        if s["kind"] == "push":
            by_state[s["j"]].append(s["truth"])
    assert all(len(set(v)) == 9 for v in by_state.values())


def test_option_balance_and_pairing(specs):
    pos = collections.Counter(s["order"].index(s["truth"]) for s in specs)
    mean = len(specs) / 14
    assert all(abs(v - mean) / mean < 0.15 for v in pos.values()) and len(pos) == 14
    key = {(s["sid"], s["rep"], s["pres"]): s for s in specs}
    for (sid, rep, pres), s in list(key.items())[:3000]:
        if pres == "RAW":
            d = key[(sid, rep, "DIGEST")]
            assert d["order"] == s["order"] and d["key_to_idx"] == s["key_to_idx"]
    assert all(set(s["body"]["questions"]) == {"decision", "pushed"} for s in specs[:50])
    assert all(len(s["body"]["questions"]["decision"]["criteria"]) == 14 for s in specs[:50])


def test_G6_information_identity(stims):
    """(a) text-only parsers recover every displayed number; (b) the digest recomputed
    from the parsed RAW reading matches the DIGEST display to 0.01; (c) the fitted
    bar's features from the parsed DIGEST text equal those it is scored on."""
    r = np.random.default_rng(3)
    for n in r.choice(len(stims), 500, replace=False):
        s = stims[n]
        order = [int(x) for x in r.permutation(14)]
        raw, dig = j3.render_raw(s, order), j3.render_digest(s, order)
        reading, scatter, eff = j3.parse_raw(raw)
        ref = j3.reference(s["ref"])
        assert [reading[c] for c in j3.CH] == s["raw"]
        assert np.allclose([scatter[c] for c in j3.CH], np.round(ref["sd"] * j3.UNIT, 2))
        for i in range(13):
            assert np.allclose(eff[s["ids"][i]], np.round(ref["sig"][i] * j3.UNIT, 2))
        shift, sims, card = j3.parse_digest(dig)
        assert shift == s["shift"] and [sims[s["ids"][i]] for i in range(13)] == s["sims"]
        assert card == ref["card"]
        x = ref["m0"] + np.array([reading[c] for c in j3.CH]) / j3.UNIT
        s2, m2 = j3.display_digest(x, ref)
        assert np.abs(s2[0] - s["sims"]).max() <= 0.0100001 and abs(m2[0] - s["shift"]) <= 0.0100001
        f_text = j3.bar_features([sims[s["ids"][i]] for i in range(13)], shift)
        assert np.array_equal(f_text, j3.bar_features(s["sims"], s["shift"]))


def test_reference_card_is_true():
    g = j3.data()
    for h in (0, 1):
        ref = j3.reference(h)
        c = {k: float(v) for k, v in ref["card"].items()}
        U = g["U"][ref["idx"]]
        s0, m0 = j3.display_digest(j3.reg(U), ref)
        assert np.mean((m0 >= c["s05"]) & (m0 <= c["s95"])) >= 0.90
        assert np.mean(s0.max(1) < c["t95"]) >= 0.95
        Rc = np.vstack([j3.reg(U + j3.CEIL * g["mu"] * g["D"][k]) for k in range(13)])
        sc, mc = j3.display_digest(Rc, ref)
        own = sc[np.arange(len(sc)), np.repeat(np.arange(13), len(U))]
        assert np.mean(mc > c["g05"]) >= 0.95 and np.mean(own > c["c05"]) >= 0.95


def test_genre_firewall(stims):
    """No concept name, no program vocabulary, no contemplative word reaches a state."""
    banned = [n.lower() for n in j3.data()["names"]] + [
        "mind", "feel", "conscious", "aware", "witness", "meditat", "introspect", "language model",
        "qwen", "concept", "emotion", "experience", "thought"]
    fmt_line = "confidence is your probability, in percent, that your choice is correct"
    for s in stims[::7]:
        order = list(range(14))
        opts = j3.options(s, order)[0]
        mouth = P.mouth_prompt(j3.render_digest(s, order), opts).split('Answer with a JSON object')[0]
        for text in (j3.render_raw(s, order), j3.render_digest(s, order), mouth, j3.INSTR,
                     " ".join(d for _, d in opts), j3.PUSHED_Q["instructions"]):
            low = text.lower()
            assert not [w for w in banned if w in low], [w for w in banned if w in low]
    # the mouth's answer format names its JSON field, and nothing else
    assert fmt_line in P.mouth_prompt("x", []) and "confidence" not in P.MOUTH_SYS.lower()


def test_plan_determinism(stims):
    a = P.flight_sha(True, stims)
    b = P.flight_sha(True, j3.flight_stimuli())
    assert a == b and a[1] == 18576
    assert P.plan_sha(P.smoke()) == P.plan_sha(P.smoke()) and len(P.smoke()) == 640
    assert len(P.mouth_flight(stims)) == 1548 and len(P.mouth_smoke()) == 40
    ids = [s["call_id"] for _, part in P.flight_parts(False, stims) for s in part]
    assert len(ids) == len(set(ids)) == 37152


def test_fitted_bar_deterministic_and_sane(stims):
    b1, b2 = j3.fit_bar(0), j3.fit_bar(0)
    assert b1 == b2
    bars = {"0": b1, "1": j3.fit_bar(1)}
    acc = collections.defaultdict(list)
    for s in stims:
        p = j3.bar_predict(bars[str(s["ref"])], s["sims"], s["shift"])[0]
        acc[(s["kind"], s["dose"])].append(int(np.argmax(p)) == s["truth"])
    assert np.mean(acc[("push", 0.5)]) > 0.98
    assert 0.40 < np.mean(acc[("push", 0.04)]) < 0.80
    assert np.mean(acc[("untouched", 0.0)]) > 0.80
    assert np.mean(acc[("random", 0.5)]) > 0.90


def test_mouth_body_is_registered(stims):
    s = P.mouth_flight(stims)[0]
    b = s["body"]
    assert b["model"] == "deepseek/deepseek-v4-flash-0731" and b["temperature"] == 0 and b["max_tokens"] == 60
    assert b["reasoning"] == {"enabled": False} and b["response_format"] == {"type": "json_object"}
    assert re.search(r'"choice": "<option key>"', b["messages"][1]["content"])
    assert all(x["j"] % 2 == 0 for x in P.mouth_flight(stims))
