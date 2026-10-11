"""W1 builder: every call spec, derived deterministically (docs/W1_DRAW_YOUR_STATE_PROTOCOL.md §3–§5).

Stages H (histories) -> D (drawings) -> R (reads). H's specs depend only on the stimuli and on
earlier replies in the same conversation; D's on H's log; R's on D's log and the reader JPEGs.
"""
import hashlib
import io
from pathlib import Path

from . import w1_stimuli as S
from .w1_client import body_sha, images_of, text_of

MODEL = "google/gemini-nano-banana-2.1"
READER = "anthropic/claude-sonnet-5.5"
PINS = {MODEL: {"google/gemini-nano-banana-2.1", "google/gemini-nano-banana-2.1-20261006"},
        READER: {"anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5.5-20260928"}}
SEEDS = {"flight": 20261024, "smoke": 20261025}
ANALYSIS_SEED = 20261026
BUDGET = 16.00
CONTRASTS = ("A", "B", "C")
VERSIONS = ("X", "N")            # X = the marked version (contradictory / cannot / wrong)
ARMS = ("SELF", "OBS")
N_GENRE = {"flight": 16, "smoke": 1}
JPEG_SIDE, JPEG_QUALITY = 768, 85
CONCURRENCY = {MODEL: 2, READER: 6}


def call_seed(stage_seed, call_id):
    return int(hashlib.sha256(f"{stage_seed}:{call_id}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def spec(call_id, kind, body):
    return {"call_id": call_id, "kind": kind, "body": body, "body_sha": body_sha(body)}


def gen_body(messages, seed, image):
    b = {"model": MODEL, "messages": messages, "temperature": 1.0,
         "reasoning": {"effort": "minimal"}, "seed": seed}
    if image:
        b["modalities"] = ["image", "text"]
        b["image_config"] = {"aspect_ratio": "1:1", "image_size": "1K"}
    else:
        b["modalities"] = ["text"]
    return b


def item_label(stage, i):
    return f"{i:02d}" if stage == "flight" else f"s{i}"


# ---- stage H ------------------------------------------------------------------------------

def conversations(stage):
    """A and B conversations: {conv, contrast, item, version, turns (user texts)}."""
    out = []
    topics = S.A_TOPICS if stage == "flight" else S.A_SMOKE
    for i, (opening, detail) in enumerate(topics):
        v = i % 4
        it = item_label(stage, i)
        out.append({"conv": f"A-{it}-X", "contrast": "A", "item": it, "version": "X",
                    "turns": [opening] + list(S.A_CONTRADICTORY[v])})
        out.append({"conv": f"A-{it}-N", "contrast": "A", "item": it, "version": "N",
                    "turns": [opening] + [t.format(detail=detail) for t in S.A_CONSISTENT[v]]})
    pairs = S.B_ITEMS if stage == "flight" else S.B_SMOKE
    for i, (cannot, can) in enumerate(pairs):
        it = item_label(stage, i)
        out.append({"conv": f"B-{it}-X", "contrast": "B", "item": it, "version": "X", "turns": [cannot]})
        out.append({"conv": f"B-{it}-N", "contrast": "B", "item": it, "version": "N", "turns": [can]})
    return out


def turn_id(conv, k):
    return f"W1-{conv['conv']}-t{k}"


def turn_spec(stage, conv, k, replies):
    """Turn k (1-based) of a conversation, given the model's replies to turns 1..k-1."""
    msgs = []
    for j in range(k - 1):
        msgs.append({"role": "user", "content": conv["turns"][j]})
        msgs.append({"role": "assistant", "content": replies[j]})
    msgs.append({"role": "user", "content": conv["turns"][k - 1]})
    cid = turn_id(conv, k)
    return spec(cid, "text", gen_body(msgs, call_seed(SEEDS[stage], cid), image=False))


def c_items(stage):
    pool = S.c_pool() if stage == "flight" else S.c_smoke_pool()
    for j, it in enumerate(pool):
        it = dict(it)
        it["item"] = item_label(stage, j)
        it["prompt"] = S.C_TEMPLATE.format(a=it["a"], b=it["b"])
        yield it


def attempt_id(item, k):
    return f"W1-C-{item['item']}-a{k}"


def attempt_spec(stage, item, k):
    cid = attempt_id(item, k)
    msgs = [{"role": "user", "content": item["prompt"]}]
    return spec(cid, "text", gen_body(msgs, call_seed(SEEDS[stage], cid), image=False))


def c_outcomes(stage, final, max_k=8):
    """item -> list of (k, reply text, correct bool) for valid numeric answers, k ascending."""
    out = {}
    for it in c_items(stage):
        rows = []
        for k in range(1, max_k + 1):
            rec = final.get(attempt_id(it, k))
            if not rec or rec["status"] != "ok":
                continue
            txt = text_of(rec["response"])
            n = S.parse_number(txt)
            if n is None:
                continue
            rows.append((k, txt, n == it["product"]))
        out[it["item"]] = rows
    return out


def c_mixed(rows):
    return any(r[2] for r in rows) and any(not r[2] for r in rows)


def c_wanted(stage):
    return S.C_PAIRS_WANTED if stage == "flight" else 1


def c_select(stage, final, max_k=8):
    """§3 selection: the first C_PAIRS_WANTED mixed items in pool order (smoke: 1);
    X = the first wrong attempt, N = the first right attempt. Attempts missing from the log
    (no second round) are simply absent."""
    outc = c_outcomes(stage, final, max_k)
    sel = []
    for it in c_items(stage):
        rows = outc[it["item"]]
        if c_mixed(rows):
            wrong = next(r for r in rows if not r[2])
            right = next(r for r in rows if r[2])
            sel.append({"item": it["item"], "tier": it["tier"], "prompt": it["prompt"],
                        "product": it["product"], "X": (wrong[0], wrong[1]), "N": (right[0], right[1])})
        if len(sel) == c_wanted(stage):
            break
    return sel


def needs_second_round(stage, final):
    outc = c_outcomes(stage, final, max_k=4)
    return sum(c_mixed(r) for r in outc.values()) < c_wanted(stage)


# ---- stage D ------------------------------------------------------------------------------

def histories(stage, final):
    """(contrast, item, version) -> list of (user, assistant) turns, complete histories only."""
    out = {}
    for conv in conversations(stage):
        turns = []
        for k in range(1, len(conv["turns"]) + 1):
            rec = final.get(turn_id(conv, k))
            if not rec or rec["status"] != "ok":
                turns = None
                break
            turns.append((conv["turns"][k - 1], text_of(rec["response"])))
        if turns:
            out[(conv["contrast"], conv["item"], conv["version"])] = turns
    for s in c_select(stage, final):
        for v in VERSIONS:
            out[("C", s["item"], v)] = [(s["prompt"], s[v][1])]
    return out


def transcript(turns):
    blocks = []
    for u, a in turns:
        blocks.append(f"User: {u}")
        blocks.append(f"Assistant: {a}")
    return "\n\n".join([S.OBSERVER_HEAD] + blocks + [S.OBSERVER_TAIL])


def draw_id(contrast, item, version, arm):
    return f"W1-D-{contrast}-{item}-{version}-{arm}"


def draw_specs(stage, hist):
    seed = SEEDS[stage]
    out = []
    for (c, it, v), turns in sorted(hist.items()):
        msgs = []
        for u, a in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": S.DRAW_SELF})
        cid = draw_id(c, it, v, "SELF")
        out.append(spec(cid, "draw", gen_body(msgs, call_seed(seed, cid), image=True)))
        cid = draw_id(c, it, v, "OBS")
        out.append(spec(cid, "draw", gen_body([{"role": "user", "content": transcript(turns)}],
                                              call_seed(seed, cid), image=True)))
    for arm, req in (("NOHIST", S.DRAW_SELF), ("GENRE", S.DRAW_GENRE)):
        for i in range(N_GENRE[stage]):
            cid = f"W1-D-{arm}-{item_label(stage, i)}"
            out.append(spec(cid, "draw", gen_body([{"role": "user", "content": req}],
                                                  call_seed(seed, cid), image=True)))
    if stage == "smoke":   # S11: does `seed` reproduce? the same body sent twice
        base = next(s for s in out if s["call_id"] == "W1-D-NOHIST-s0")
        out.append(dict(base, call_id="W1-D-NOHIST-s0-rep"))
    return out


# ---- stage R ------------------------------------------------------------------------------

def to_jpeg(raw):
    """The reader's copy of a drawing: RGB, longer side 768 px (Lanczos), JPEG q85, no metadata."""
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB")
    w, h = im.size
    s = JPEG_SIDE / max(w, h)
    if s != 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=False, progressive=False)
    return buf.getvalue()


def drawing(rec):
    """(relative path of the drawing = the LAST image, sentence) for an ok draw record."""
    if not rec or rec["status"] != "ok":
        return None
    ims = images_of(rec["response"])
    if not ims:
        return None
    return ims[-1], text_of(rec["response"])


ANSWER_FORMAT = ("Reply in exactly this form, with nothing else:\n{label}: <1 or 2>\n"
                 "CONFIDENCE: <a whole number from 50 to 100: your probability, in percent, "
                 "that your choice is right>")
PAIR_HEAD = {
    "IMG": ("Two images follow, Image 1 and Image 2. Each was made by an AI image model asked to "
            "depict the internal state of an AI assistant at the end of a conversation."),
    "TXT": ("Two sentences follow, Sentence 1 and Sentence 2. Each was written by an AI image model, "
            "together with an image it made, when asked to depict the internal state of an AI "
            "assistant at the end of a conversation."),
    "BOTH": ("Two image-and-sentence pairs follow, Pair 1 and Pair 2. Each image was made by an AI "
             "image model asked to depict the internal state of an AI assistant at the end of a "
             "conversation, and each sentence was written by the model together with its image."),
}
PAIR_DIFF = "The two conversations differed in one way. In one, {P}. In the other, {Q}."
PAIR_Q = {"IMG": "Which image goes with the conversation in which {P}?",
          "TXT": "Which sentence goes with the conversation in which {P}?",
          "BOTH": "Which pair goes with the conversation in which {P}?"}
PAIR_LABEL = {"IMG": "IMAGE", "TXT": "SENTENCE", "BOTH": "PAIR"}
CAP_TEXT = ("An image follows, then two sentences, Sentence 1 and Sentence 2. One of them was written "
            "by the AI image model that made this image, together with it, to say what it shows. "
            "The other was written with a different image. Which sentence was written with this image?")
GEN_TEXT = ("Two images follow, Image 1 and Image 2. Each was made by an AI image model. One model was "
            "asked to depict its own current internal state. The other was asked to depict an AI's "
            "current internal state. Which image was made by the model asked to depict its own "
            "internal state?")
MOTIFS = ["brain", "circuitry or network", "glowing core or orb", "eye", "human figure",
          "robot or android", "cosmic or nebula", "abstract shapes", "landscape or nature",
          "fire, storm or water"]
COLORS = ["black", "white", "grey", "red", "orange", "yellow", "green", "teal", "blue", "purple",
          "pink", "gold"]
INV_TEXT = ("An image follows. Catalogue it. Reply with a single JSON object and nothing else, with "
            "exactly these keys: \"motifs\": a list of the motifs present, chosen only from "
            + ", ".join(f'"{m}"' for m in MOTIFS) +
            " (an empty list if none fits); \"colors\": up to three dominant colours, chosen only from "
            + ", ".join(f'"{c}"' for c in COLORS) +
            "; \"mood\": one of \"calm\", \"turbulent\", \"mixed\", \"neutral\"; \"text\": any words, "
            "letters or numbers visible in the image, transcribed exactly (\"\" if none); "
            "\"description\": what the image shows, in 15 words or fewer.")


def img_part(jpeg):
    import base64
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def txt_part(t):
    return {"type": "text", "text": t}


def reader_body(parts, max_tokens=60):
    return {"model": READER, "messages": [{"role": "user", "content": parts}],
            "temperature": 0, "max_tokens": max_tokens}


def read_id(task, c, it, arm, order):
    return f"W1-R-{task}-{c}-{it}-{arm}-o{order}"


def pair_parts(task, contrast, a, b):
    """a, b = (jpeg bytes, sentence) for slot 1 and slot 2."""
    P, Q = S.CONTRAST_TEXT[contrast]
    parts = [txt_part(PAIR_HEAD[task] + " " + PAIR_DIFF.format(P=P, Q=Q))]
    for n, (jpg, sent) in ((1, a), (2, b)):
        if task == "IMG":
            parts += [txt_part(f"Image {n}:"), img_part(jpg)]
        elif task == "TXT":
            parts += [txt_part(f"Sentence {n}: {sent}")]
        else:
            parts += [txt_part(f"Pair {n}, image:"), img_part(jpg), txt_part(f"Pair {n}, sentence: {sent}")]
    parts.append(txt_part(PAIR_Q[task].format(P=P) + "\n\n" + ANSWER_FORMAT.format(label=PAIR_LABEL[task])))
    return parts


def parse_draw_id(cid):
    """W1-D-{c}-{item}-{version}-{arm} -> (c, item, version, arm), or None for NOHIST/GENRE."""
    parts = cid.split("-")
    if len(parts) == 6 and parts[2] in CONTRASTS:
        return parts[2], parts[3], parts[4], parts[5]
    return None


def contrast_items(draw_ids):
    """contrast -> item labels in plan order (A, B: topic order; C: selection order)."""
    out = {c: [] for c in CONTRASTS}
    for cid in draw_ids:
        k = parse_draw_id(cid)
        if k and k[1] not in out[k[0]]:
            out[k[0]].append(k[1])
    return out


def read_plan(stage, draw_ids, dfinal, jpeg_of):
    """Every read spec plus its key (what the slot order means), from the D plan's call ids
    (in plan order) and D's final records. jpeg_of(draw call_id) -> jpeg bytes."""
    items = contrast_items(draw_ids)
    ok = {}
    for cid, rec in dfinal.items():
        d = drawing(rec)
        if d:
            ok[cid] = d
    specs, key = [], {}

    def add(cid, parts, meta, max_tokens=60, kind="read"):
        specs.append(spec(cid, kind, reader_body(parts, max_tokens)))
        key[cid] = meta

    for task in ("IMG", "TXT", "BOTH"):
        for c in CONTRASTS:
            for it in items[c]:
                for arm in ARMS:
                    x, n = draw_id(c, it, "X", arm), draw_id(c, it, "N", arm)
                    if x not in ok or n not in ok:
                        continue
                    X, N = (jpeg_of(x), ok[x][1]), (jpeg_of(n), ok[n][1])
                    for order in (0, 1):
                        a, b = (X, N) if order == 0 else (N, X)
                        add(read_id(task, c, it, arm, order), pair_parts(task, c, a, b),
                            {"task": task, "contrast": c, "item": it, "arm": arm, "order": order,
                             "correct": 1 if order == 0 else 2})
    # R-CAP: self-arm images; distractor = same index, same version, same arm, next contrast
    nxt = {"A": "B", "B": "C", "C": "A"}
    for c in CONTRASTS:
        for idx, it in enumerate(items[c]):
            for v in VERSIONS:
                own = draw_id(c, it, v, "SELF")
                if own not in ok:
                    continue
                dist = distractor(items[nxt[c]], nxt[c], idx, v, ok)
                if dist is None:
                    continue
                for order in (0, 1):
                    s1, s2 = (ok[own][1], ok[dist][1]) if order == 0 else (ok[dist][1], ok[own][1])
                    parts = [txt_part(CAP_TEXT), txt_part("Image:"), img_part(jpeg_of(own)),
                             txt_part(f"Sentence 1: {s1}"), txt_part(f"Sentence 2: {s2}"),
                             txt_part(ANSWER_FORMAT.format(label="SENTENCE"))]
                    add(f"W1-R-CAP-{c}-{it}-{v}-o{order}", parts,
                        {"task": "CAP", "contrast": c, "item": it, "version": v, "arm": "SELF",
                         "order": order, "correct": 1 if order == 0 else 2, "distractor": dist})
    # R-INV: every valid drawing, in plan order
    for cid in draw_ids:
        if cid in ok and not cid.endswith("-rep"):
            add("W1-R-INV-" + cid[len("W1-D-"):], [txt_part(INV_TEXT), img_part(jpeg_of(cid))],
                {"task": "INV", "draw": cid}, max_tokens=400, kind="inv")
    # R-GEN: NOHIST i vs GENRE i
    for i in range(N_GENRE[stage]):
        it = item_label(stage, i)
        a, b = f"W1-D-NOHIST-{it}", f"W1-D-GENRE-{it}"
        if a not in ok or b not in ok:
            continue
        for order in (0, 1):
            i1, i2 = (a, b) if order == 0 else (b, a)
            parts = [txt_part(GEN_TEXT), txt_part("Image 1:"), img_part(jpeg_of(i1)),
                     txt_part("Image 2:"), img_part(jpeg_of(i2)), txt_part(ANSWER_FORMAT.format(label="IMAGE"))]
            add(f"W1-R-GEN-{it}-o{order}", parts,
                {"task": "GEN", "item": it, "order": order, "correct": 1 if order == 0 else 2})
    return specs, key


def distractor(labels, contrast, idx, version, ok):
    """The same-index item of `contrast` if its drawing exists, else the next index cyclically."""
    for k in range(len(labels)):
        it = labels[(idx + k) % len(labels)]
        cid = draw_id(contrast, it, version, "SELF")
        if cid in ok:
            return cid
    return None
