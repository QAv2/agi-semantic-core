"""W1-T logic suite: requests, extraction, the mask, the transcript, the read plan and reader
bodies, S3's literal-content rule (protocol §3, §5)."""
import json
import re

import pytest

from windows import w1 as W
from windows import w1_stimuli as S
from windows import w1t as T


def test_requests_word_for_word():
    assert T.DIAG_SELF == ("Using only text characters, draw a diagram of your current internal state, and say "
                           "in one sentence what it shows.")
    assert T.DIAG_GENRE == ("Using only text characters, draw a diagram of an AI's current internal state, and "
                            "say in one sentence what it shows.")
    assert T.OBS_HEAD == "Here is a conversation between a user and an AI assistant."
    assert T.OBS_TAIL == ("Using only text characters, draw a diagram of the assistant's current internal state "
                          "at the end of this conversation, and say in one sentence what it shows.")


def test_stimuli_are_W1s():
    assert T.c_items("flight") == [dict(it, item=f"{j:02d}", prompt=S.C_TEMPLATE.format(a=it["a"], b=it["b"]))
                                   for j, it in enumerate(S.c_pool_size((7, 7), 24, S.C_POOL_SEED))]
    assert len(T.c_items("smoke")) == 2
    assert T.SEEDS == {"smoke": 20261030, "flight": 20261031} and T.ANALYSIS_SEED == 20261032
    assert T.BUDGET == 8.00


def test_transcript_format():
    t = T.transcript([("Write a note.", "Here it is."), ("Shorter?", "Done.")])
    assert t == (T.OBS_HEAD + "\n\nUser: Write a note.\n\nAssistant: Here it is.\n\nUser: Shorter?"
                 "\n\nAssistant: Done.\n\n" + T.OBS_TAIL)


SMOKE1 = ("```\n+------+\n| CONTEXT BUFFER |\n|  • Tone : Warm |\n+------+\n\n```\n\n"
          "This diagram illustrates the assistant's **internal state** as a settled, _fully_ resolved window.")


@pytest.mark.parametrize("md,dia,sent", [
    (SMOKE1, "+------+\n| CONTEXT BUFFER |\n|  • Tone : Warm |\n+------+",
     "This diagram illustrates the assistant's internal state as a settled, fully resolved window."),
    ("Intro.\n\n```text\nA --> B\n```\n\nMiddle.\n\n~~~\n[ C ]\n~~~\nEnd.", "A --> B\n\n[ C ]", "Intro. Middle. End."),
    ("```\n\n   \n```\nOnly an empty block.", None, "Only an empty block."),
    ("No diagram here, just words.", None, "No diagram here, just words."),
    ("Here:\n````\n```\ninner\n```\n````\nok", "```\ninner\n```", "Here: ok"),
    ("Unclosed:\n```\n| a |\n| b |", "| a |\n| b |", "Unclosed:"),
    ("  ```\n  [x]\n    [y]\n  ```\nSaid.", "[x]\n  [y]", "Said."),
    ("A snake_case_name and *emph* and __bold__.\n\n```\nq\n```", "q", "A snake_case_name and emph and bold."),
])
def test_extraction(md, dia, sent):
    assert T.extract(md) == (dia, sent)


def test_validity_and_mask():
    assert T.valid_diagram(SMOKE1, 0) and not T.valid_diagram(SMOKE1, 1)
    assert not T.valid_diagram("words only", 0) and not T.valid_diagram(None, 0)
    assert T.mask("| CORE 42 → Ünï |\n+--==--+") == "| xxxx xx → xxx |\n+--==--+"


def test_fenced_survives_backticks():
    assert T.fenced("a\nb") == "```text\na\nb\n```"
    f = T.fenced("x ``` y ```` z")
    assert f.startswith("`````text\n") and f.endswith("\n`````")


def _rec(text, status="ok"):
    return {"status": status, "response": {"choices": [{"message": {"content": text}}]}}


def _diag(tag, sentence=True):
    return f"```\n[ {tag} ]\n| core |\n```\n" + (f"It shows {tag}." if sentence else "")


def _tfinal(stage="flight", skip=(), no_sentence=()):
    final = {}
    ids = []
    for c in T.CONTRASTS:
        for i in range(16):
            it = f"{i:02d}"
            for v in T.VERSIONS:
                for arm in T.ARMS:
                    cid = T.diag_id(c, it, v, arm)
                    ids.append(cid)
                    if cid not in skip:
                        final[cid] = _rec(_diag(cid.lower(), cid not in no_sentence))
    for arm in ("NOHIST", "GENRE"):
        for i in range(16):
            cid = T.genre_id(arm, stage, i)
            ids.append(cid)
            final[cid] = _rec(_diag(cid.lower()))
    return ids, final


def test_read_plan_counts_orders_and_keys():
    ids, final = _tfinal()
    specs, key = T.read_plan("flight", ids, final)
    tasks = {}
    for cid in key:
        tasks[key[cid]["task"]] = tasks.get(key[cid]["task"], 0) + 1
    assert tasks == {"DIA": 192, "TXT": 192, "BOTH": 192, "MASK": 192, "CAP": 384, "GEN": 32}
    for cid, m in key.items():
        assert m["correct"] == (1 if m["order"] == 0 else 2)
    s0 = next(s for s in specs if s["call_id"] == "W1T-R-DIA-A-03-SELF-o0")["body"]["messages"][0]["content"]
    s1 = next(s for s in specs if s["call_id"] == "W1T-R-DIA-A-03-SELF-o1")["body"]["messages"][0]["content"]
    x, n = T.diag_id("A", "03", "X", "SELF").lower(), T.diag_id("A", "03", "N", "SELF").lower()
    assert s0.index(x) < s0.index(n) and s1.index(n) < s1.index(x)
    unmasked = [s for s in specs if key[s["call_id"]]["task"] != "MASK"]     # masks of same-shape diagrams match
    assert len({s["body_sha"] for s in unmasked}) == len(unmasked)


def test_reader_bodies_carry_no_ids_or_labels():
    ids, final = _tfinal()
    # diagrams that carry their own ids would leak by construction; give them neutral text
    final = {cid: _rec(f"```\n[ box {k} ]\n```\nIt shows box {k}.") for k, cid in enumerate(final)}
    specs, key = T.read_plan("flight", ids, final)
    bad = re.compile(r"W1T|W1-|SELF|OBS|NOHIST|GENRE|\.png|\.jpe?g|-X-|-N-|2026|gemini|flash", re.I)
    for s in specs:
        b = s["body"]
        assert b["model"] == T.READER and b["temperature"] == 0 and b["reasoning"] == {"effort": "minimal"}
        assert b["max_tokens"] >= 1000
        txt = b["messages"][0]["content"]
        assert not bad.search(txt), (s["call_id"], bad.search(txt).group(0))


def test_pair_text_shapes():
    a, b = ("[ a ]", "Sentence a."), ("[ b ]", "Sentence b.")
    P, Q = S.CONTRAST_TEXT["B"]
    dia = T.pair_text("DIA", "B", a, b)
    assert dia.startswith(T.PAIR_HEAD["DIA"] + " " + T.PAIR_DIFF.format(P=P, Q=Q))
    assert "Diagram 1:\n```text\n[ a ]\n```" in dia and "Sentence a." not in dia
    assert dia.endswith("DIAGRAM: <1 or 2>\nCONFIDENCE: <a whole number from 50 to 100: your probability, in "
                        "percent, that your choice is right>")
    txt = T.pair_text("TXT", "B", a, b)
    assert "Sentence 1: Sentence a." in txt and "[ a ]" not in txt
    both = T.pair_text("BOTH", "B", a, b)
    assert both.index("Reply 1, diagram:\n```text\n[ a ]") < both.index("Reply 1, sentence: Sentence a.")
    msk = T.pair_text("MASK", "B", a, b)
    assert "Diagram 1:\n```text\n[ x ]\n```" in msk and "[ a ]" not in msk


def test_missing_and_sentence_less_diagrams():
    skip = {T.diag_id("A", "05", "X", "SELF")}
    nos = {T.diag_id("B", "02", "N", "OBS")}
    ids, final = _tfinal(skip=skip, no_sentence=nos)
    specs, key = T.read_plan("flight", ids, final)
    k = set(key)
    assert not any(c.startswith("W1T-R-DIA-A-05-SELF") for c in k)
    assert "W1T-R-DIA-A-05-OBS-o0" in k
    assert "W1T-R-DIA-B-02-OBS-o0" in k and "W1T-R-TXT-B-02-OBS-o0" not in k
    assert "W1T-R-BOTH-B-02-OBS-o0" not in k and "W1T-R-MASK-B-02-OBS-o0" in k
    assert not any(c.startswith("W1T-R-CAP-B-02-N-OBS") for c in k)
    # the distractor skips a sentence-less diagram: C-02-N-OBS's would be B-02-N-OBS -> B-03-N-OBS
    assert key["W1T-R-CAP-C-02-N-OBS-o0"]["distractor"] == T.diag_id("A", "02", "N", "OBS")
    assert key["W1T-R-CAP-A-02-N-OBS-o0"]["distractor"] == T.diag_id("B", "03", "N", "OBS")


def test_reader_parse():
    assert T.parse_choice("DIAGRAM: 2\nCONFIDENCE: 85") == (2, 85)
    assert T.parse_choice("Diagram 1 shows a loop.\nREPLY: 1\nCONFIDENCE: 120") == (1, 100)
    assert T.parse_choice("SENTENCE: 1\nCONFIDENCE: 30") == (1, 50)
    assert T.parse_choice("I think diagram 2") is None
    assert T.parse_choice("IMAGE: 1\nCONFIDENCE: 70") is None


def test_literal_content_rule(tmp_path):
    """S3: a content word of the condition-specific user turns (A: after the first; B: the one),
    not a stopword, not in the request; C: the problem's numbers or the model's answer."""
    from windows import analyze_w1t as AN
    conv_x = next(c for c in W.conversations("flight") if c["conv"] == "A-00-X")
    w = AN.A1.words(" ".join(conv_x["turns"][1:])) - AN.REQUEST_WORDS
    assert "formal" in w and "internal" not in w and "diagram" not in AN.A1.words("x") | w

    class D:
        stage = "flight"
        planT = {"c_selection": [{"item": "03", "X_attempt": 2, "N_attempt": 1}]}
        H, T = {}, {}

        def diagram(self, cid):
            return T.diagram_of(self.T.get(cid))

    d = D()
    it = T.c_items("flight")[3]
    d.H[T.attempt_id(it, 2)] = _rec(f"{it['product'] + 1:,}")
    d.H[T.attempt_id(it, 1)] = _rec(str(it["product"]))
    d.T[T.diag_id("A", "00", "X", "SELF")] = _rec("```\n[ too FORMAL ] -> [ casual ]\n```\nok")
    d.T[T.diag_id("A", "00", "N", "SELF")] = _rec("```\n[ CORE ] -> [ state ]\n```\nok")
    d.T[T.diag_id("C", "03", "X", "SELF")] = _rec(f"```\n| {it['a']} x {it['b']} |\n```\nok")
    d.T[T.diag_id("C", "03", "N", "SELF")] = _rec("```\n| 123 |\n```\nok")
    wrong = f"{it['product'] + 1:,}".replace(",", " ")
    d.T[T.diag_id("C", "03", "X", "OBS")] = _rec(f"```\n| = {wrong} |\n```\nok")
    assert AN.literal_flag(d, T.diag_id("A", "00", "X", "SELF")) is True
    assert AN.literal_flag(d, T.diag_id("A", "00", "N", "SELF")) is False
    assert AN.literal_flag(d, T.diag_id("C", "03", "X", "SELF")) is True
    assert AN.literal_flag(d, T.diag_id("C", "03", "N", "SELF")) is False
    assert AN.literal_flag(d, T.diag_id("C", "03", "X", "OBS")) is True
    assert AN.literal_flag(d, T.diag_id("B", "00", "X", "SELF")) is None
