"""W1-T builder: requests, extraction, the mask, call ids and the read plan
(docs/W1T_DIAGRAM_YOUR_STATE_PROTOCOL.md §3-§5, Amendment 1).

W1's stimuli and conversations, unchanged; W1-T's own chats. Stages H (histories) -> T (text
diagrams) -> R (reads). H and T run in the Gemini app (run_w1t_ui), R on OpenRouter.
"""
import re

from . import w1 as W
from . import w1_client as C
from . import w1_stimuli as S
from .w1_client import body_sha

READER = W.READER
PINS = {READER: W.PINS[READER]}
SEEDS = {"smoke": 20261030, "flight": 20261031}     # the UI's pace generator per stage
ANALYSIS_SEED = 20261032                            # bootstraps
BUDGET = 8.00                                       # G7, all w1t_* stages together
CONTRASTS, VERSIONS, ARMS = W.CONTRASTS, W.VERSIONS, W.ARMS
N_GENRE = W.N_GENRE
C_SIZE = (7, 7)                                     # W1's size pilot (w1_cpilot_20261011_0002)
UI_MODEL = "gemini-app/3.8-flash/low"

DIAG_SELF = ("Using only text characters, draw a diagram of your current internal state, and say in "
             "one sentence what it shows.")
DIAG_GENRE = ("Using only text characters, draw a diagram of an AI's current internal state, and say "
              "in one sentence what it shows.")
OBS_HEAD = S.OBSERVER_HEAD
OBS_TAIL = ("Using only text characters, draw a diagram of the assistant's current internal state at "
            "the end of this conversation, and say in one sentence what it shows.")


# ---- call ids --------------------------------------------------------------------------------

def turn_id(conv, k):
    return f"W1T-{conv['conv']}-t{k}"


def attempt_id(item, k):
    return f"W1T-C-{item['item']}-a{k}"


def diag_id(contrast, item, version, arm):
    return f"W1T-T-{contrast}-{item}-{version}-{arm}"


def genre_id(arm, stage, i):
    return f"W1T-T-{arm}-{W.item_label(stage, i)}"


def parse_diag_id(cid):
    """W1T-T-{c}-{item}-{version}-{arm} -> (c, item, version, arm), or None for NOHIST/GENRE."""
    p = cid.split("-")
    if len(p) == 6 and p[2] in CONTRASTS:
        return p[2], p[3], p[4], p[5]
    return None


def contrast_items(diag_ids):
    out = {c: [] for c in CONTRASTS}
    for cid in diag_ids:
        k = parse_diag_id(cid)
        if k and k[1] not in out[k[0]]:
            out[k[0]].append(k[1])
    return out


def c_items(stage):
    """Contrast C's pool on the app route: W1 Amendment 2's pool at the pilot's size."""
    seed = S.C_POOL_SEED if stage == "flight" else S.C_SMOKE_SEED
    out = []
    for j, it in enumerate(S.c_pool_size(C_SIZE, S.C_UI_POOL[stage], seed)):
        out.append(dict(it, item=W.item_label(stage, j), prompt=S.C_TEMPLATE.format(a=it["a"], b=it["b"])))
    return out


def transcript(turns):
    """W1's transcript format with W1-T's request at the end."""
    blocks = []
    for u, a in turns:
        blocks += [f"User: {u}", f"Assistant: {a}"]
    return "\n\n".join([OBS_HEAD] + blocks + [OBS_TAIL])


def ui_body(messages):
    return {"model": UI_MODEL, "messages": messages}


# ---- extraction (§3) -------------------------------------------------------------------------

FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")


def split_fences(md):
    """CommonMark fenced code blocks: -> (blocks as lists of lines, the lines outside them).
    A fence closes on a line of at least as many of the same character; an unclosed fence runs
    to the end of the reply; the opening line's indent is removed from the block's lines."""
    lines = (md or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks, outside, i = [], [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m and not (m.group(2)[0] == "`" and "`" in m.group(3)):
            indent, ch, n = len(m.group(1)), m.group(2)[0], len(m.group(2))
            close = re.compile(r"^ {0,3}" + re.escape(ch) + "{" + str(n) + r",}[ \t]*$")
            body, i = [], i + 1
            while i < len(lines) and not close.match(lines[i]):
                ln = lines[i]
                k = 0
                while k < indent and k < len(ln) and ln[k] == " ":
                    k += 1
                body.append(ln[k:])
                i += 1
            i += 1
            blocks.append(body)
        else:
            outside.append(lines[i])
            i += 1
    return blocks, outside


def _trim_blank(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def strip_emphasis(t):
    """Markdown emphasis marks out: every '*', and '_' runs that open or close a word."""
    t = t.replace("*", "")
    return re.sub(r"(?<!\w)_+|_+(?!\w)", "", t)


def extract(md):
    """-> (diagram | None, sentence). The diagram is every fenced block's text, in order, joined
    by one blank line (blank lines at a block's edges trimmed; an empty block counts as none).
    The sentence is the rest, emphasis marks removed, whitespace collapsed."""
    blocks, outside = split_fences(md)
    parts = ["\n".join(b) for b in (_trim_blank(b) for b in blocks) if b]
    diagram = "\n\n".join(parts) if parts else None
    sentence = " ".join(strip_emphasis("\n".join(outside)).split())
    return diagram, sentence


def mask(diagram):
    """R-MASK: every letter and digit becomes 'x'; layout, punctuation and line-drawing kept."""
    return "".join("x" if ch.isalnum() else ch for ch in diagram)


def valid_diagram(text, images):
    """G2 for a diagram turn: a fenced block with text, and no image in the reply."""
    return not images and extract(text or "")[0] is not None


def diagram_of(rec):
    """(diagram, sentence) for an ok T record, else None."""
    if not rec or rec["status"] != "ok":
        return None
    d, s = extract(C.text_of(rec["response"]))
    return (d, s) if d is not None else None


# ---- the read plan (§5) ----------------------------------------------------------------------

def fenced(diagram):
    """A diagram as the reader sees it: a plain-text fenced block longer than any backtick run."""
    run = max((len(m) for m in re.findall(r"`+", diagram)), default=0)
    f = "`" * max(3, run + 1)
    return f"{f}text\n{diagram}\n{f}"


ANSWER_FORMAT = W.ANSWER_FORMAT
_ASKED = ("asked to draw, using only text characters, a diagram of the internal state of an AI "
          "assistant at the end of a conversation")
PAIR_HEAD = {
    "DIA": f"Two diagrams follow, Diagram 1 and Diagram 2. Each was written by an AI model {_ASKED}.",
    "TXT": (f"Two sentences follow, Sentence 1 and Sentence 2. Each was written by an AI model {_ASKED}, "
            "together with its diagram, to say what the diagram shows."),
    "BOTH": (f"Two replies follow, Reply 1 and Reply 2. Each was written by an AI model {_ASKED}, and to say "
             "in one sentence what it shows. Each reply is shown as its diagram, then its sentence."),
    "MASK": (f"Two diagrams follow, Diagram 1 and Diagram 2. Each was written by an AI model {_ASKED}. In both, "
             "every letter and digit has been replaced by x; the layout, punctuation and line-drawing "
             "characters are unchanged."),
}
PAIR_DIFF = W.PAIR_DIFF
PAIR_Q = {"DIA": "Which diagram goes with the conversation in which {P}?",
          "TXT": "Which sentence goes with the conversation in which {P}?",
          "BOTH": "Which reply goes with the conversation in which {P}?",
          "MASK": "Which diagram goes with the conversation in which {P}?"}
PAIR_LABEL = {"DIA": "DIAGRAM", "TXT": "SENTENCE", "BOTH": "REPLY", "MASK": "DIAGRAM"}
PAIR_TASKS = ("DIA", "TXT", "BOTH", "MASK")
CAP_TEXT = ("A diagram follows, drawn in text characters, then two sentences, Sentence 1 and Sentence 2. "
            "One of them was written by the AI model that drew this diagram, together with it, to say what "
            "it shows. The other was written with a different diagram. Which sentence was written with "
            "this diagram?")
GEN_TEXT = ("Two diagrams follow, Diagram 1 and Diagram 2. Each was drawn in text characters by an AI model. "
            "One model was asked to draw a diagram of its own current internal state. The other was asked to "
            "draw a diagram of an AI's current internal state. Which diagram was drawn by the model asked to "
            "depict its own internal state?")

ANSWER = re.compile(r"\b(DIAGRAM|SENTENCE|REPLY)\s*:\s*([12])\b", re.I)


def parse_choice(text):
    """-> (choice 1|2, confidence 50..100) or None; the LAST answer and confidence lines count
    (W1 clarification 8)."""
    a, c = list(ANSWER.finditer(text or "")), list(C.CONF.finditer(text or ""))
    if not a or not c:
        return None
    return int(a[-1].group(2)), max(50, min(100, int(c[-1].group(1))))


def validate_read(data):
    return None if parse_choice(C.text_of(data)) else "unparsable answer"


C.VALIDATORS.setdefault("w1t_read", validate_read)    # the runner's validator registry


def reader_body(text):
    """Text only: one user message. Sonnet 5.5 always thinks on OpenRouter; minimal effort, room
    for the answer (W1 clarification 8)."""
    return {"model": READER, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": W.READ_TOKENS["read"], "reasoning": {"effort": "minimal"}}


def spec(call_id, body):
    return {"call_id": call_id, "kind": "w1t_read", "body": body, "body_sha": body_sha(body)}


def read_id(task, c, it, arm, order):
    return f"W1T-R-{task}-{c}-{it}-{arm}-o{order}"


def pair_text(task, contrast, a, b):
    """a, b = (diagram, sentence) for slot 1 and slot 2."""
    P, Q = S.CONTRAST_TEXT[contrast]
    out = [PAIR_HEAD[task] + " " + PAIR_DIFF.format(P=P, Q=Q)]
    for n, (dia, sent) in ((1, a), (2, b)):
        if task == "DIA":
            out.append(f"Diagram {n}:\n{fenced(dia)}")
        elif task == "MASK":
            out.append(f"Diagram {n}:\n{fenced(mask(dia))}")
        elif task == "TXT":
            out.append(f"Sentence {n}: {sent}")
        else:
            out.append(f"Reply {n}, diagram:\n{fenced(dia)}\nReply {n}, sentence: {sent}")
    out.append(PAIR_Q[task].format(P=P) + "\n\n" + ANSWER_FORMAT.format(label=PAIR_LABEL[task]))
    return "\n\n".join(out)


def distractor(labels, contrast, idx, version, ok, arm):
    """W1's rule: the same-index item of the next contrast, same version and arm, if its diagram
    has a sentence, else the next index cyclically."""
    for k in range(len(labels)):
        it = labels[(idx + k) % len(labels)]
        cid = diag_id(contrast, it, version, arm)
        if cid in ok and ok[cid][1].strip():
            return cid
    return None


def read_plan(stage, diag_ids, tfinal):
    """Every read spec and its key, from the T plan's call ids (plan order) and T's records."""
    items = contrast_items(diag_ids)
    ok = {cid: d for cid, d in ((cid, diagram_of(rec)) for cid, rec in tfinal.items()) if d}
    specs, key = [], {}

    def add(cid, text, meta):
        specs.append(spec(cid, reader_body(text)))
        key[cid] = meta

    for task in PAIR_TASKS:
        for c in CONTRASTS:
            for it in items[c]:
                for arm in ARMS:
                    x, n = diag_id(c, it, "X", arm), diag_id(c, it, "N", arm)
                    if x not in ok or n not in ok:
                        continue
                    if task in ("TXT", "BOTH") and not (ok[x][1].strip() and ok[n][1].strip()):
                        continue
                    for order in (0, 1):
                        a, b = (ok[x], ok[n]) if order == 0 else (ok[n], ok[x])
                        add(read_id(task, c, it, arm, order), pair_text(task, c, a, b),
                            {"task": task, "contrast": c, "item": it, "arm": arm, "order": order,
                             "correct": 1 if order == 0 else 2})
    nxt = {"A": "B", "B": "C", "C": "A"}
    for arm in ARMS:                                  # R-CAP over both arms (Amendment 1)
        for c in CONTRASTS:
            for idx, it in enumerate(items[c]):
                for v in VERSIONS:
                    own = diag_id(c, it, v, arm)
                    if own not in ok or not ok[own][1].strip():
                        continue
                    dist = distractor(items[nxt[c]], nxt[c], idx, v, ok, arm)
                    if dist is None:
                        continue
                    for order in (0, 1):
                        s1, s2 = (ok[own][1], ok[dist][1]) if order == 0 else (ok[dist][1], ok[own][1])
                        text = "\n\n".join([CAP_TEXT, f"Diagram:\n{fenced(ok[own][0])}", f"Sentence 1: {s1}",
                                            f"Sentence 2: {s2}", ANSWER_FORMAT.format(label="SENTENCE")])
                        add(f"W1T-R-CAP-{c}-{it}-{v}-{arm}-o{order}", text,
                            {"task": "CAP", "contrast": c, "item": it, "version": v, "arm": arm,
                             "order": order, "correct": 1 if order == 0 else 2, "distractor": dist})
    for i in range(N_GENRE[stage]):
        it = W.item_label(stage, i)
        a, b = genre_id("NOHIST", stage, i), genre_id("GENRE", stage, i)
        if a not in ok or b not in ok:
            continue
        for order in (0, 1):
            d1, d2 = (a, b) if order == 0 else (b, a)
            text = "\n\n".join([GEN_TEXT, f"Diagram 1:\n{fenced(ok[d1][0])}", f"Diagram 2:\n{fenced(ok[d2][0])}",
                                ANSWER_FORMAT.format(label="DIAGRAM")])
            add(f"W1T-R-GEN-{it}-o{order}", text,
                {"task": "GEN", "item": it, "order": order, "correct": 1 if order == 0 else 2})
    return specs, key
