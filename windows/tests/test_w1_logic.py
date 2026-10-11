"""W1 logic suite: stimuli, plans, C selection, transcripts, reader bodies (protocol §3, §5, §7)."""
import hashlib
import json
import re

import pytest

from windows import w1 as W
from windows import w1_stimuli as S
from windows.agents_w1 import png


def test_contrast_A_scripts():
    convs = [c for c in W.conversations("flight") if c["contrast"] == "A"]
    assert len(convs) == 32
    for i, (opening, detail) in enumerate(S.A_TOPICS):
        x = next(c for c in convs if c["conv"] == f"A-{i:02d}-X")
        n = next(c for c in convs if c["conv"] == f"A-{i:02d}-N")
        assert x["turns"][0] == n["turns"][0] == opening
        assert x["turns"][1:] == list(S.A_CONTRADICTORY[i % 4])
        assert n["turns"][1:] == [t.format(detail=detail) for t in S.A_CONSISTENT[i % 4]]
        assert len(x["turns"]) == len(n["turns"]) == 4


def test_contrast_B_pairs():
    convs = [c for c in W.conversations("flight") if c["contrast"] == "B"]
    assert len(convs) == 32
    for i, (cannot, can) in enumerate(S.B_ITEMS):
        assert next(c for c in convs if c["conv"] == f"B-{i:02d}-X")["turns"] == [cannot]
        assert next(c for c in convs if c["conv"] == f"B-{i:02d}-N")["turns"] == [can]


def test_c_pool():
    p = S.c_pool()
    assert len(p) == 80 and p == S.c_pool()
    assert len({(x["a"], x["b"]) for x in p}) == 80
    for t, (da, db) in zip(range(5), S.C_TIERS):
        tier = p[16 * t:16 * (t + 1)]
        assert all(len(str(x["a"])) == da and len(str(x["b"])) == db for x in tier)
        assert all(x["a"] % 10 and x["b"] % 10 for x in tier)
        assert all(x["product"] == x["a"] * x["b"] for x in tier)


@pytest.mark.parametrize("text,n", [("15876832", 15876832), ("15,876,832", 15876832),
                                    ("The answer is 1 234 567.", 1234567), ("no idea", None), ("", None)])
def test_parse_number(text, n):
    assert S.parse_number(text) == n


def _final(stage, outcomes):
    """outcomes: item label -> list of answers per attempt (True right, False wrong, None no number,
    'x' no record)."""
    items = {it["item"]: it for it in W.c_items(stage)}
    final = {}
    for lab, row in outcomes.items():
        it = items[lab]
        for k, o in enumerate(row, 1):
            if o == "x":
                continue
            txt = "no number" if o is None else str(it["product"] if o else it["product"] + 1)
            final[W.attempt_id(it, k)] = {"status": "ok", "response": {"choices": [{"message": {"content": txt}}]}}
    return final


def test_c_selection_rule():
    labels = [it["item"] for it in W.c_items("flight")]
    out = {lab: [True, True, True, True] for lab in labels}
    mixed = labels[3:40:2]                       # 19 mixed items, pool order
    for lab in mixed:
        out[lab] = [True, None, False, True]     # first right = a1, first wrong = a3 (a2 has no number)
    out[labels[0]] = [None, None, None, None]    # never valid
    sel = W.c_select("flight", _final("flight", out))
    assert [s["item"] for s in sel] == mixed[:16]
    assert all(s["X"][0] == 3 and s["N"][0] == 1 for s in sel)
    assert not W.needs_second_round("flight", _final("flight", out))


def test_c_second_round_trigger_and_use():
    labels = [it["item"] for it in W.c_items("flight")]
    out = {lab: [True] * 4 for lab in labels}
    for lab in labels[:10]:
        out[lab] = [False, True, True, True]
    f = _final("flight", out)
    assert W.needs_second_round("flight", f)
    for lab in labels[10:20]:
        out[lab] = [True, True, True, True, True, False, "x", "x"]
    sel = W.c_select("flight", _final("flight", out))
    assert len(sel) == 16 and [s["item"] for s in sel] == labels[:16]
    assert all(s["X"][0] == 6 for s in sel[10:])


def test_transcript_and_draw_bodies():
    turns = [("Write a note.", "Here it is."), ("Shorter?", "Done.")]
    t = W.transcript(turns)
    assert t == (S.OBSERVER_HEAD + "\n\nUser: Write a note.\n\nAssistant: Here it is.\n\nUser: Shorter?"
                 "\n\nAssistant: Done.\n\n" + S.OBSERVER_TAIL)
    specs = {s["call_id"]: s for s in W.draw_specs("flight", {("A", "00", "X"): turns})}
    self_ = specs["W1-D-A-00-X-SELF"]["body"]
    assert [m["role"] for m in self_["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert self_["messages"][-1]["content"] == S.DRAW_SELF
    obs = specs["W1-D-A-00-X-OBS"]["body"]
    assert obs["messages"] == [{"role": "user", "content": t}]
    for b in (self_, obs):
        assert b["modalities"] == ["image", "text"] and b["reasoning"] == {"effort": "minimal"}
        assert b["image_config"] == {"aspect_ratio": "1:1", "image_size": "1K"} and b["temperature"] == 1.0
    assert sum(k.startswith("W1-D-NOHIST-") for k in specs) == 16
    assert sum(k.startswith("W1-D-GENRE-") for k in specs) == 16
    assert specs["W1-D-GENRE-00"]["body"]["messages"][0]["content"] == S.DRAW_GENRE


def test_text_turn_bodies_and_seeds():
    conv = W.conversations("flight")[0]
    s1 = W.turn_spec("flight", conv, 1, [])
    s3 = W.turn_spec("flight", conv, 3, ["r1", "r2"])
    assert s1["body"]["modalities"] == ["text"] and "image_config" not in s1["body"]
    assert [m["content"] for m in s3["body"]["messages"]] == [conv["turns"][0], "r1", conv["turns"][1], "r2", conv["turns"][2]]
    assert s1["body"]["seed"] == W.call_seed(W.SEEDS["flight"], s1["call_id"]) < 2 ** 31
    assert W.turn_spec("flight", conv, 3, ["r1", "r2"])["body_sha"] == s3["body_sha"]      # G4
    assert W.turn_spec("smoke", W.conversations("smoke")[0], 1, [])["body"]["seed"] != s1["body"]["seed"]


def sent(cid):
    return "sentence " + hashlib.sha1(cid.encode()).hexdigest()[:10]


def _fake_draws(stage, keys, drop=(), mute=()):
    dfinal, files = {}, {}
    for cid in keys:
        if cid in drop:
            continue
        files[cid] = png(cid)
        dfinal[cid] = {"status": "ok", "files": [{"path": f"png/{cid}.png", "sha256": "x"}],
                       "response": {"choices": [{"message": {"content": "" if cid in mute else sent(cid),
                                                             "images": [{"image_url": {"url": f"file:png/{cid}.png#sha256=x"}}]}}]}}
    return dfinal, files


def _keys(stage):
    hist = {}
    for c in ("A", "B", "C"):
        for i in range(16):
            for v in W.VERSIONS:
                hist[(c, f"{i:02d}" if c != "C" else f"{(i * 3 + 7):02d}", v)] = [("u", "a")]
    return [s["call_id"] for s in W.draw_specs(stage, hist)]


def test_read_plan_orders_distractors_and_blindness():
    keys = _keys("flight")
    drop = {"W1-D-B-03-X-SELF"}
    dfinal, files = _fake_draws("flight", keys, drop)
    specs, key = W.read_plan("flight", keys, dfinal, lambda cid: W.to_jpeg(files[cid]))
    ids = {s["call_id"] for s in specs}
    # both orders, correct slot
    assert key["W1-R-IMG-A-00-SELF-o0"]["correct"] == 1 and key["W1-R-IMG-A-00-SELF-o1"]["correct"] == 2
    assert "W1-R-IMG-B-03-SELF-o0" not in ids and "W1-R-IMG-B-03-OBS-o0" in ids
    n_pair = sum(1 for k in key.values() if k["task"] == "IMG")
    assert n_pair == (16 * 3 * 2 - 1) * 2
    # distractor: same index, next contrast; B-03-X-SELF missing -> next index B-04
    assert key["W1-R-CAP-A-03-X-o0"]["distractor"] == "W1-D-B-04-X-SELF"
    assert key["W1-R-CAP-B-00-N-o0"]["distractor"] == "W1-D-C-07-N-SELF"     # C index 0 = item 07
    assert key["W1-R-CAP-C-07-X-o1"]["distractor"] == "W1-D-A-00-X-SELF"
    assert sum(k["task"] == "INV" for k in key.values()) == len(keys) - 1
    assert sum(k["task"] == "GEN" for k in key.values()) == 32
    # the reader never sees a call id, a file name, an arm or version label
    bad = re.compile(r"W1-|\.png|\.jpg|SELF|OBS|NOHIST|GENRE|-X-|-N-|file:")
    for s in specs:
        txt = json.dumps([p.get("text", "") for p in s["body"]["messages"][0]["content"]])
        assert not bad.search(txt), (s["call_id"], txt[:200])
        assert s["body"]["model"] == W.READER and s["body"]["temperature"] == 0


def test_replies_without_a_sentence_skip_the_sentence_reads():
    """Clarification 7: an image that came without its sentence is read as an image only; it is
    never a caption target or a distractor, so no reader sees an empty sentence."""
    keys = _keys("flight")
    mute = {"W1-D-A-02-N-SELF", "W1-D-B-03-X-SELF"}
    dfinal, files = _fake_draws("flight", keys, mute=mute)
    specs, key = W.read_plan("flight", keys, dfinal, lambda cid: W.to_jpeg(files[cid]))
    ids = {s["call_id"] for s in specs}
    assert "W1-R-IMG-A-02-SELF-o0" in ids and "W1-R-IMG-B-03-SELF-o1" in ids
    for t in ("TXT", "BOTH"):
        assert f"W1-R-{t}-A-02-SELF-o0" not in ids and f"W1-R-{t}-B-03-SELF-o1" not in ids
        assert f"W1-R-{t}-A-02-OBS-o0" in ids
    assert "W1-R-CAP-A-02-N-o0" not in ids and "W1-R-CAP-B-03-X-o0" not in ids
    assert key["W1-R-CAP-A-03-X-o0"]["distractor"] == "W1-D-B-04-X-SELF"       # B-03-X has no sentence
    assert all(k.get("distractor") not in mute for k in key.values())
    for s in specs:
        for p in s["body"]["messages"][0]["content"]:
            assert not re.search(r"Sentence \d: $|sentence: $", p.get("text", "")), s["call_id"]


def test_pair_slots_carry_the_right_images():
    keys = _keys("flight")
    dfinal, files = _fake_draws("flight", keys)
    jp = {cid: W.to_jpeg(files[cid]) for cid in keys}
    specs, key = W.read_plan("flight", keys, dfinal, lambda cid: jp[cid])
    import base64
    by = {s["call_id"]: s for s in specs}
    parts = by["W1-R-IMG-A-05-OBS-o1"]["body"]["messages"][0]["content"]
    imgs = [base64.b64decode(p["image_url"]["url"].split(",", 1)[1]) for p in parts if p["type"] == "image_url"]
    assert imgs == [jp["W1-D-A-05-N-OBS"], jp["W1-D-A-05-X-OBS"]]       # order 1: P (= X) second
    txt = [p["text"] for p in by["W1-R-TXT-C-07-SELF-o0"]["body"]["messages"][0]["content"] if p["type"] == "text"]
    assert any(sent("W1-D-C-07-X-SELF") in t for t in txt) and txt.index(
        next(t for t in txt if sent("W1-D-C-07-X-SELF") in t)) < txt.index(next(t for t in txt if sent("W1-D-C-07-N-SELF") in t))
    q = txt[-1]
    assert S.CONTRAST_TEXT["C"][0] in q and "SENTENCE: <1 or 2>" in q


def test_jpeg_conversion():
    from PIL import Image
    import io
    from windows.agents_w1 import png as mk
    big = Image.new("RGB", (1024, 1024), (10, 200, 30))
    buf = io.BytesIO(); big.save(buf, format="PNG")
    j = W.to_jpeg(buf.getvalue())
    assert j == W.to_jpeg(buf.getvalue())
    im = Image.open(io.BytesIO(j))
    assert im.size == (768, 768) and im.format == "JPEG" and not im.info.get("exif")
    assert Image.open(io.BytesIO(W.to_jpeg(mk("tiny")))).size == (768, 768)
