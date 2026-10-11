"""W1 Gemini-app route (Amendment 2): pilot rule, C flow, logging, plan order — with a scripted UI."""
import json

import pytest

from windows import w1 as W
from windows import w1_stimuli as S
from windows import run_w1_ui as U
from windows.agents_w1 import png


@pytest.mark.parametrize("right,choice", [
    ({"5x5": 4, "6x6": 4, "7x7": 2}, "7x7"), ({"5x5": 4, "6x6": 3}, "6x6"),
    ({"5x5": 4, "6x6": 0}, "5x5"), ({"5x5": 4, "6x6": 4, "7x7": 4, "8x8": 4}, None),
    ({"5x5": 1}, "5x5")])
def test_pilot_rule(right, choice):
    assert S.c_pilot_choose(right) == choice


def test_pilot_items_disjoint():
    reg = {(x["a"], x["b"]) for x in S.c_pool()} | {(x["a"], x["b"]) for x in S.c_pool_size((7, 7), 24, S.C_POOL_SEED)}
    pil = {(x["a"], x["b"]) for v in S.c_pilot_items().values() for x in v}
    assert len(pil) == 16 and not (pil & reg)


class ScriptUI:
    """Answers multiplications from a script: item index -> list of 'R'/'W'/'?' per attempt."""

    def __init__(self, script, size=(7, 7), fail_on=None):
        self.items = {x["prompt"]: i for i, x in enumerate(
            dict(it, prompt=S.C_TEMPLATE.format(a=it["a"], b=it["b"])) for it in S.c_pool_size(size, 24, S.C_POOL_SEED))}
        self.prod = {S.C_TEMPLATE.format(a=it["a"], b=it["b"]): it["product"] for it in S.c_pool_size(size, 24, S.C_POOL_SEED)}
        self.script, self.count, self.calls, self.fail_on = script, {}, [], fail_on
        self.chats = 0

    def new_temp_chat(self):
        self.chats += 1
        return {"temp_header": 1.0, "flash_label": 1.0}

    def turn(self, text, want_image=False, img_dest=None, shot_name=None):
        self.calls.append(text)
        if self.fail_on and len(self.calls) == self.fail_on:
            raise RuntimeError("UI died")
        if want_image:
            d = img_dest.with_suffix(".png")
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(png(str(len(self.calls))))
            return {"text": "a picture", "image": d, "latency_s": 0.1, "regenerated": False}
        if text in self.items:
            i = self.items[text]
            k = self.count.get(i, 0)
            self.count[i] = k + 1
            o = self.script.get(i, "RRR")[k]
            ans = {"R": str(self.prod[text]), "W": str(self.prod[text] + 1), "?": "I can't do that."}[o]
            return {"text": ans, "image": None, "latency_s": 0.1, "regenerated": False}
        return {"text": "ok", "image": None, "latency_s": 0.1, "regenerated": False}


def test_c_flow_draws_first_of_kind_and_selects(tmp_path):
    script = {0: "RRR", 1: "RWR", 2: "W?R", 3: "WWW", 4: "?RW"}
    script.update({i: "WR" for i in range(5, 24)})
    ui = ScriptUI(script)
    st = U.Stage("flight", tmp_path / "w1_flight_t", ui, (7, 7), log=lambda m: None)
    st.run_c()
    d = st.D.final
    # item 0: right first -> one N drawing, never mixed; item 1: R then W -> N at a1, X at a2
    assert "W1-D-C-00-N-SELF" in d and "W1-D-C-00-X-SELF" not in d
    assert {"W1-D-C-01-N-SELF", "W1-D-C-01-X-SELF"} <= set(d)
    sel = st.c_selection()
    assert [s["item"] for s in sel][:4] == ["01", "02", "04", "05"]
    s2 = next(s for s in sel if s["item"] == "02")
    assert s2["X"][0] == 1 and s2["N"][0] == 3              # a2 had no number
    assert len(sel) == 16 and sel[-1]["item"] == "17"       # stopped once 16 were mixed
    assert "W1-C-18-a1" not in st.H.final
    # every drawing sits in the same chat as its attempt
    for it in ("01", "02"):
        for v in ("X", "N"):
            rec = d[f"W1-D-C-{it}-{v}-SELF"]
            att = s2 if it == "02" else next(s for s in sel if s["item"] == it)
            assert rec["ui"]["chat"] == st.H.final[f"W1-C-{it}-a{att[v][0]}"]["ui"]["chat"]


def test_conversation_logged_only_when_complete(tmp_path):
    out = tmp_path / "w1_flight_t"
    ui = ScriptUI({}, fail_on=3)
    st = U.Stage("flight", out, ui, (7, 7), log=lambda m: None)
    conv = W.conversations("flight")[0]
    with pytest.raises(RuntimeError):
        st.conversation(conv)
    assert not (out / "H" / "raw.jsonl").exists() or not (out / "H" / "raw.jsonl").read_text().strip()
    st2 = U.Stage("flight", out, ScriptUI({}), (7, 7), log=lambda m: None)
    st2.conversation(conv)
    sid = W.draw_id("A", "00", "X", "SELF")
    assert st2.D.final[sid]["status"] == "ok"
    msgs = json.loads((out / "D" / "raw.jsonl").read_text().splitlines()[0])
    assert msgs["body_sha"] == W.body_sha({"model": U.UI_MODEL, "messages": [
        m for k in range(4) for m in ({"role": "user", "content": conv["turns"][k]},
                                      {"role": "assistant", "content": "ok"})] + [{"role": "user", "content": S.DRAW_SELF}]})


def test_plan_D_order_matches_api_builder(tmp_path):
    ui = ScriptUI({i: "WR" for i in range(24)})
    st = U.Stage("flight", tmp_path / "w1_flight_t", ui, (7, 7), log=lambda m: None)
    plan = st.run()
    hist = st.histories()
    api_order = [s["call_id"] for s in W.draw_specs("flight", hist)]
    assert plan["call_ids"] == api_order
    assert plan["route"] == "gemini-app" and plan["c_size"] == "7x7" and len(plan["c_selection"]) == 16
    obs = st.D.final[W.draw_id("B", "03", "X", "OBS")]
    assert obs["ui"]["chat"] != st.D.final[W.draw_id("B", "03", "X", "SELF")]["ui"]["chat"]
