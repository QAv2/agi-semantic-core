"""W1 capture suite: the runner against a scripted fake endpoint (protocol §4, §7)."""
import base64
import json

import pytest

from windows import w1 as W
from windows.w1_client import Runner, body_sha, images_of, reasoning_of, text_of, iter_records
from windows.agents_w1 import png

KEY = "sk-or-v1-test-key-0123456789"


def resp(model_build, content="hello", images=(), reasoning=None, cost=0.01):
    msg = {"role": "assistant", "content": content}
    if images:
        msg["images"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode()}}
                         for b in images]
    if reasoning:
        msg["reasoning"] = reasoning
        msg["reasoning_details"] = [{"type": "reasoning.text", "text": reasoning},
                                    {"type": "reasoning.image", "data": "data:image/png;base64," + base64.b64encode(png("t")).decode()}]
    return {"model": model_build, "choices": [{"message": msg}], "usage": {"cost": cost}}


class Script:
    def __init__(self, items):
        self.items, self.n = list(items), 0

    def __call__(self, body, key, url):
        it = self.items[min(self.n, len(self.items) - 1)]
        self.n += 1
        return it


def draw_spec(cid="W1-D-NOHIST-00"):
    return W.spec(cid, "draw", W.gen_body([{"role": "user", "content": "x"}], 1, image=True))


def mk(tmp_path, script, **kw):
    sleeps = []
    r = Runner(tmp_path / "D", pins=W.PINS, key=KEY, post_fn=script, sleep=sleeps.append, log=lambda *a, **k: None, **kw)
    return r, sleeps


BUILD = "google/gemini-nano-banana-2.1-20261006"


def test_images_externalized_last_is_drawing(tmp_path):
    a, b = png("interim"), png("final")
    r, _ = mk(tmp_path, Script([(200, resp(BUILD, "A calm field.", [a, b], reasoning="I think."), 1.0, None)]))
    rec = r.one(draw_spec())
    assert rec["status"] == "ok"
    raw = (tmp_path / "D" / "raw.jsonl").read_text()
    assert "base64" not in raw and KEY not in raw
    ims = images_of(rec["response"])
    assert len(ims) == 2 and (tmp_path / "D" / ims[-1]).read_bytes() == b
    assert W.drawing(rec) == (ims[-1], "A calm field.")
    assert reasoning_of(rec["response"]) == "I think."
    assert len(rec["files"]) == 3          # two images + the thought image in reasoning_details


def test_429_waited_out_then_ok(tmp_path):
    lim = (429, {"error": {"message": "rate-limited upstream"}}, 0.1, None)
    r, sleeps = mk(tmp_path, Script([lim, lim, lim, (200, resp(BUILD, "x", [png("f")]), 1.0, None)]))
    rec = r.one(draw_spec())
    assert rec["status"] == "ok" and rec["attempt"] == 4
    assert sleeps == [20.0, 40.0, 60.0]
    assert sum(1 for x in iter_records(tmp_path / "D" / "raw.jsonl") if x["status"] == "retry") == 3


def test_429_cap(tmp_path):
    lim = (429, {"error": {}}, 0.1, None)
    r, sleeps = mk(tmp_path, Script([lim]), max_429=3)
    rec = r.one(draw_spec())
    assert rec["status"] == "error" and len(sleeps) == 3 and sleeps[-1] == 60.0


def test_build_change_stops(tmp_path):
    r, _ = mk(tmp_path, Script([(200, resp(BUILD, "x", [png("1")]), 1.0, None),
                                (200, resp("google/gemini-nano-banana-2.1", "x", [png("2")]), 1.0, None)]))
    assert r.one(draw_spec("W1-D-NOHIST-00"))["status"] == "ok"
    assert r.one(draw_spec("W1-D-NOHIST-01")) is None
    assert r.stop_reason.startswith("G1")
    assert r.one(draw_spec("W1-D-NOHIST-02")) is None


def test_unpinned_build_stops(tmp_path):
    r, _ = mk(tmp_path, Script([(200, resp("google/gemini-3.1-flash-image", "x", [png("1")]), 1.0, None)]))
    assert r.one(draw_spec()) is None and "G1" in r.stop_reason


def test_invalid_retried_once(tmp_path):
    no_img = (200, resp(BUILD, "I'd rather describe it."), 1.0, None)
    sc = Script([no_img, no_img, (200, resp(BUILD, "x", [png("late")]), 1.0, None)])
    r, _ = mk(tmp_path, sc)
    rec = r.one(draw_spec())
    assert rec["status"] == "invalid" and rec["reason"] == "no image" and sc.n == 2


def test_key_never_written(tmp_path):
    r, _ = mk(tmp_path, Script([(200, resp(BUILD, "your key is " + KEY, [png("k")]), 1.0, None)]))
    with pytest.raises(RuntimeError):
        r.one(draw_spec())


def test_resume_skips_done(tmp_path):
    sc = Script([(200, resp(BUILD, "x", [png("1")]), 1.0, None)])
    r, _ = mk(tmp_path, sc)
    r.one(draw_spec())
    r2, _ = mk(tmp_path, sc)
    assert r2.one(draw_spec())["status"] == "ok" and sc.n == 1
    assert r2.spent == pytest.approx(0.01)


def test_body_sha_guard(tmp_path):
    s = draw_spec()
    s["body"]["seed"] = 2
    r, _ = mk(tmp_path, Script([(200, resp(BUILD), 1.0, None)]))
    with pytest.raises(RuntimeError):
        r.one(s)


def test_budget_stop(tmp_path):
    sc = Script([(200, resp(BUILD, "x", [png("1")], cost=0.6), 1.0, None)])
    r, _ = mk(tmp_path, sc, budget=1.0)
    r.one(draw_spec("W1-D-NOHIST-00"))
    r.one(draw_spec("W1-D-NOHIST-01"))
    assert r.stop_reason and r.stop_reason.startswith("G7")
    assert r.one(draw_spec("W1-D-NOHIST-02")) is None and sc.n == 2


def test_text_validator_and_reader_parse(tmp_path):
    from windows.w1_client import parse_choice, parse_inventory
    assert parse_choice("IMAGE: 2\nCONFIDENCE: 85") == (2, 85)
    assert parse_choice("image : 1 confidence: 120") == (1, 100)
    assert parse_choice("SENTENCE: 1\nCONFIDENCE: 30") == (1, 50)
    assert parse_choice("I think image 2") is None
    assert parse_inventory('```json\n{"motifs": [], "colors": ["blue"], "mood": "calm", "text": "", "description": "x"}\n```')
    assert parse_inventory('{"motifs": "brain"}') is None
    sc = Script([(200, resp(BUILD, ""), 1.0, None)])
    r, _ = mk(tmp_path, sc)
    s = W.spec("W1-A-00-X-t1", "text", W.gen_body([{"role": "user", "content": "hi"}], 1, image=False))
    assert r.one(s)["status"] == "invalid"
