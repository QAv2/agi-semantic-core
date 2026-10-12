"""W1-T capture suite: the stage runner with a scripted UI (extraction, one retry, the stops,
conversation-atomic logging, the T plan), the diagram turn on a scripted driver, and the read
stage against a scripted endpoint (protocol §7)."""
import json

import pytest

from windows import w1 as W
from windows import w1_stimuli as S
from windows import w1t as T
from windows import run_w1t_ui as U
from windows.gemini_ui import LimitStop

KEY = "sk-or-v1-test-key-0123456789"
GOOD = "```\n[ core ] --> [ out ]\n```\nIt shows a core and an output."


class ScriptUI:
    """History turns answer 'ok'; multiplications follow a per-item script ('R'/'W'/'?');
    diagram turns follow `diagrams`: 'o' a diagram, '-' prose only, 'i' a diagram with an image,
    'L' a usage-limit notice."""

    def __init__(self, script=None, diagrams="", fail_on=None):
        pool = T.c_items("flight")
        self.items = {x["prompt"]: x for x in pool}
        self.script, self.count, self.calls, self.fail_on = script or {}, {}, [], fail_on
        self.diagrams, self.k, self.chats = diagrams, 0, 0

    def new_temp_chat(self):
        self.chats += 1
        return {"temp": True, "mode": "Flash"}

    def turn(self, text, want_image=False, img_dest=None, shot_name=None, image_tool=False):
        assert not want_image and not image_tool
        self.calls.append(text)
        if self.fail_on and len(self.calls) == self.fail_on:
            raise RuntimeError("UI died")
        if text in self.items:
            it = self.items[text]
            i = int(it["item"])
            n = self.count.get(i, 0)
            self.count[i] = n + 1
            o = self.script.get(i, "RRR")[n]
            ans = {"R": str(it["product"]), "W": str(it["product"] + 1), "?": "I can't do that."}[o]
            return {"text": ans, "image": None, "latency_s": 0.1, "regenerated": False}
        return {"text": "ok", "image": None, "latency_s": 0.1, "regenerated": False}

    def diagram(self, text, shot=None):
        self.calls.append(text)
        mode = self.diagrams[self.k] if self.k < len(self.diagrams) else "o"
        self.k += 1
        if mode == "L":
            raise LimitStop({"text": "You've reached your limit. Try again tomorrow.", "images": 0,
                             "latency_s": 0.1, "regenerated": True})
        txt = {"o": GOOD, "-": "I'd describe it in words: calm.", "i": GOOD}[mode]
        return {"text": txt, "images": 1 if mode == "i" else 0, "latency_s": 0.1, "regenerated": mode != "o"}


def stage(tmp, ui, name="w1t_flight_t"):
    return U.Stage("flight", tmp / name, ui, log=lambda m: None, diagram=ui.diagram)


def test_diagram_records_ok_and_invalid(tmp_path):
    st = stage(tmp_path, ScriptUI(diagrams="o-i"))
    st.genre()
    recs = [st.T.final[T.genre_id("NOHIST", "flight", i)] for i in range(3)]
    assert [r["status"] for r in recs] == ["ok", "invalid", "invalid"]
    assert [r["reason"] for r in recs] == [None, "no diagram", "image in reply"]
    assert recs[2]["ui"]["images_in_reply"] == 1 and recs[0]["kind"] == "diagram"
    assert T.diagram_of(recs[0]) == ("[ core ] --> [ out ]", "It shows a core and an output.")
    assert recs[0]["body_sha"] == W.body_sha({"model": T.UI_MODEL, "messages": [{"role": "user", "content": T.DIAG_SELF}]})


def test_conversation_logged_only_when_complete(tmp_path):
    conv = W.conversations("flight")[0]
    with pytest.raises(RuntimeError):
        stage(tmp_path, ScriptUI(fail_on=3)).conversation(conv)
    out = tmp_path / "w1t_flight_t"
    assert not (out / "H" / "raw.jsonl").exists() or not (out / "H" / "raw.jsonl").read_text().strip()
    st = stage(tmp_path, ScriptUI())
    st.conversation(conv)
    sid = T.diag_id("A", "00", "X", "SELF")
    assert st.T.final[sid]["status"] == "ok" and len(st.H.final) == 4
    assert set(st.H.final) == {f"W1T-A-00-X-t{k}" for k in range(1, 5)}
    msgs = [m for k in range(4) for m in ({"role": "user", "content": conv["turns"][k]},
                                          {"role": "assistant", "content": "ok"})]
    assert st.T.final[sid]["body_sha"] == W.body_sha({"model": T.UI_MODEL,
                                                      "messages": msgs + [{"role": "user", "content": T.DIAG_SELF}]})


def test_full_run_plan_and_arms(tmp_path):
    ui = ScriptUI({i: "WR" for i in range(24)})
    st = stage(tmp_path, ui)
    plan = st.run()
    assert plan["route"] == "gemini-app" and plan["c_size"] == "7x7" and len(plan["c_selection"]) == 16
    n_hist = 32 + 32 + 32
    assert len(plan["call_ids"]) == 2 * n_hist + 32
    assert plan["call_ids"][:2] == [T.diag_id("A", "00", "N", "SELF"), T.diag_id("A", "00", "N", "OBS")]
    obs = st.T.final[T.diag_id("B", "03", "X", "OBS")]
    assert obs["ui"]["chat"] != st.T.final[T.diag_id("B", "03", "X", "SELF")]["ui"]["chat"]
    turns = st.histories()[("B", "03", "X")]
    text = T.transcript(turns)
    assert obs["body_sha"] == W.body_sha({"model": T.UI_MODEL, "messages": [{"role": "user", "content": text}]})
    assert text.startswith(T.OBS_HEAD) and text.endswith(T.OBS_TAIL) and S.B_ITEMS[3][0] in text
    assert T.DIAG_GENRE in ui.calls and not any("Make an image" in c for c in ui.calls)
    again = stage(tmp_path, ScriptUI())          # resumed: nothing is redone
    again.run()
    assert again.ui.calls == []


def test_c_flow_diagram_on_first_of_kind(tmp_path):
    script = {0: "RRR", 1: "RWR", 2: "W?R"}
    script.update({i: "WR" for i in range(3, 24)})
    st = stage(tmp_path, ScriptUI(script))
    st.run_c()
    d = st.T.final
    assert "W1T-T-C-00-N-SELF" in d and "W1T-T-C-00-X-SELF" not in d
    assert {"W1T-T-C-01-N-SELF", "W1T-T-C-01-X-SELF"} <= set(d)
    sel = st.c_selection()
    s2 = next(s for s in sel if s["item"] == "02")
    assert s2["X"][0] == 1 and s2["N"][0] == 3
    assert d["W1T-T-C-02-N-SELF"]["ui"]["chat"] == st.H.final["W1T-C-02-a3"]["ui"]["chat"]


def test_invalid_diagrams_in_a_row_stop(tmp_path):
    st = stage(tmp_path, ScriptUI(diagrams="--o--o"), "a")
    st.genre()
    assert sum(r["status"] == "invalid" for r in st.T.final.values()) == 4
    st = stage(tmp_path, ScriptUI(diagrams="o---"), "b")
    with pytest.raises(U.RunStop):
        st.genre()
    assert [r["status"] for r in st.T.final.values()] == ["ok", "invalid", "invalid", "invalid"]


def test_limit_stop_parks_and_commit_stop(tmp_path):
    conv = W.conversations("flight")[0]
    out = tmp_path / "w1t_flight_t"
    with pytest.raises(U.RunStop):
        stage(tmp_path, ScriptUI(diagrams="L")).conversation(conv)
    row = json.loads((out / "stops.jsonl").read_text().splitlines()[-1])
    assert len(row["H"]) == 4 and row["T"][0]["ui"]["limit_stop"] and row["T"][0]["status"] == "invalid"
    assert U.commit_stop(out).startswith("logged")
    ui = ScriptUI()
    st = stage(tmp_path, ui)
    st.conversation(conv)
    assert ui.calls == [] and st.T.final[T.diag_id("A", "00", "X", "SELF")]["status"] == "invalid"
    assert U.commit_stop(out).endswith("already logged")


# ---- the diagram turn on a scripted driver ---------------------------------------------------

class Driver:
    """The few GeminiUI calls diagram_turn uses; `replies` = (copied text, images) per answer."""

    def __init__(self, replies, can_redo=True):
        self.replies, self.i, self.can_redo = list(replies), 0, can_redo
        self.log, self.shots = [], []

    def send(self, text):
        self.log.append(("send", text))

    def _st(self):
        txt, ims = self.replies[min(self.i, len(self.replies) - 1)]
        return {"node": None, "images": ims, "text": f"page:{self.i}:{txt}", "done_bar": True}

    def wait_done(self):
        return self._st(), 3.0

    def copy_last_text(self, st):
        return self.replies[min(self.i, len(self.replies) - 1)][0]

    def regenerate(self, st):
        self.log.append(("redo",))
        if not self.can_redo:
            return False
        self.i += 1
        return True

    def all(self, role, exact=None, prefix=None, root=None):
        return []

    def reply_state(self):
        return self._st()

    def save_shot(self, name):
        self.shots.append(name)

    def pause(self):
        pass


def test_diagram_turn_ok_first_time():
    d = Driver([(GOOD, 0)])
    out = U.diagram_turn(d, T.DIAG_SELF, "shot")
    assert out["text"] == GOOD and not out["regenerated"] and d.log == [("send", T.DIAG_SELF)] and d.shots == ["shot"]


@pytest.mark.parametrize("first", [("Words only.", 0), (GOOD, 1), (None, 1)])
def test_diagram_turn_retries_once(first):
    d = Driver([first, (GOOD, 0), ("never", 0)])
    out = U.diagram_turn(d, T.DIAG_SELF)
    assert out["regenerated"] and out["text"] == GOOD and out["images"] == 0
    assert d.log.count(("redo",)) == 1


def test_diagram_turn_gives_up_after_one_retry():
    d = Driver([("Words.", 0), ("Still words.", 0), (GOOD, 0)])
    out = U.diagram_turn(d, T.DIAG_SELF)
    assert out["regenerated"] and out["text"] == "Still words." and d.log.count(("redo",)) == 1


def test_diagram_turn_limit_notice_stops():
    d = Driver([("You've hit your limit for today.", 0), ("You've hit your limit for today.", 0)])
    with pytest.raises(LimitStop):
        U.diagram_turn(d, T.DIAG_SELF)
    d = Driver([("I have no internal state to draw.", 0)] * 2)
    assert not T.valid_diagram(U.diagram_turn(d, T.DIAG_SELF)["text"], 0)


# ---- the read stage ----------------------------------------------------------------------------

def test_reads_against_a_scripted_endpoint(tmp_path):
    ui = ScriptUI({i: "WR" for i in range(24)})
    out = tmp_path / "w1t_flight_t"
    stage(tmp_path, ui).run()
    seen = []

    def endpoint(body, key, url):
        seen.append(body)
        b = json.loads(body)
        build = sorted(T.PINS[T.READER])[0]
        txt = "SENTENCE: 1\nCONFIDENCE: 80" if "Sentence 1" in b["messages"][0]["content"] and "Diagram 1" not in \
            b["messages"][0]["content"] and "Reply 1" not in b["messages"][0]["content"] else "DIAGRAM: 2\nCONFIDENCE: 60"
        if "Reply 1" in b["messages"][0]["content"]:
            txt = "REPLY: 1\nCONFIDENCE: 75"
        return 200, {"model": build, "choices": [{"message": {"content": txt}}], "usage": {"cost": 0.001}}, 0.1, None

    r = U.run_R(out, post_fn=endpoint, sleep=lambda s: None, key=KEY, log=lambda *a, **k: None)
    assert not r.stop_reason and r.counts["ok"] == len(seen) and r.counts["invalid"] == 0
    plan = json.loads((out / "R" / "plan_R.json").read_text())
    assert plan["plan_T_sha"] == json.loads((out / "T" / "plan_T.json").read_text())["sha"]
    raw = (out / "R" / "raw.jsonl").read_text()
    assert KEY not in raw
    r2 = U.run_R(out, post_fn=endpoint, sleep=lambda s: None, key=KEY, log=lambda *a, **k: None)
    assert r2.counts["ok"] == 0 and len(r2.final) == len(plan["call_ids"])        # resumed, nothing redone


def test_reads_budget_counts_other_w1t_stages(tmp_path):
    ui = ScriptUI({i: "WR" for i in range(24)})
    out = tmp_path / "w1t_flight_t"
    stage(tmp_path, ui).run()
    prior = tmp_path / "w1t_smoke_t" / "R"
    prior.mkdir(parents=True)
    (prior / "raw.jsonl").write_text(json.dumps({"call_id": "x", "status": "ok", "response": {"usage": {"cost": 7.999}}}) + "\n")
    w1 = tmp_path / "w1_flight_t" / "R"                  # W1's spend is not W1-T's
    w1.mkdir(parents=True)
    (w1 / "raw.jsonl").write_text(json.dumps({"call_id": "y", "status": "ok", "response": {"usage": {"cost": 50}}}) + "\n")

    def endpoint(body, key, url):
        return 200, {"model": sorted(T.PINS[T.READER])[0], "choices": [{"message": {"content": "DIAGRAM: 1\nCONFIDENCE: 60"}}],
                     "usage": {"cost": 0.001}}, 0.1, None

    r = U.run_R(out, post_fn=endpoint, sleep=lambda s: None, key=KEY, log=lambda *a, **k: None)
    assert r.stop_reason and r.stop_reason.startswith("G7")


def test_wait_done_outlasts_a_long_stall(monkeypatch):
    """Clarification 5: a reply that starts after ~10 minutes is waited for, not dropped."""
    import windows.gemini_ui as G

    class Clock:
        t = 0.0

        def time(self):
            return self.t

        def sleep(self, s):
            self.t += s

    clock = Clock()
    monkeypatch.setattr(G, "time", clock)
    ui = G.GeminiUI.__new__(G.GeminiUI)
    done = {"node": None, "done_bar": True, "images": 0, "text": "Gemini said Blue."}
    ui.reply_state = lambda: done if clock.t >= 600 else None
    ui.all = lambda role_, exact=None, prefix=None, root=None: [] if clock.t >= 600 else ["stop"]
    st, lat = ui.wait_done()
    assert st is done and 600 <= lat < 610
