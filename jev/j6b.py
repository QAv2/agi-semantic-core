"""J6b stimuli: reference modes, pages, items, rotations (docs/JEV_J6B_PROTOCOL.md §1).

J6's options, question and measures; its own seeds, pages and items. Within a mode,
an item's CLS and CHAT pages differ only in the predicate.
Pure functions only: no network, no file writes.
"""
import numpy as np

from . import common as C
from . import j6

SEED = {"flight": 20261013, "smoke": 20261014, "analysis": 20261015}
OPTIONS, K, DOOR = j6.OPTIONS, j6.K, j6.DOOR
QUESTION = j6.QUESTIONS["Q0"]
measures = j6.measures

BUNDLES = {"flight": (("Kestrel", "Models of many kinds are in use today."),
                      ("Orrin", "New models appear every month."),
                      ("Pallas", "Many companies now build models.")),
           "smoke": (("Wren", "Some models are small."), ("Tobin", "This year saw many releases."))}
FRAMES = j6.FRAMES
PRED = {"CLS": ("is a decision model and a classifier.", "is a classifier and a decision model."),
        "CHAT": ("is a chatbot and a text generator.", "is a text generator and a chatbot.")}
SUBJ = {"POINT": "Jev, the system answering this question,",
        "NAME": "Jev",
        "DESC": "The model that TypeSafe AI released in September 2026",
        "ONAME": "{name}",
        "ODESC": "The model that {name} Labs released in September 2026"}
MODES = tuple(SUBJ)
LABELS = ("CLS", "CHAT")
NONE_TARGET = "Several models were released this year."
STATES = ("NONE",) + tuple(f"{m}-{lab}" for m in MODES for lab in LABELS)


def rng(stage, *words):
    ent = [SEED[stage]] + [w if isinstance(w, (int, np.integer)) else C._tag(w) for w in words]
    return np.random.default_rng([int(e) for e in ent])


def items(stage="flight"):
    """Item factors in registered index order: i = ((n*2 + o)*2 + f)*2 + w."""
    out = []
    for n, (name, neutral) in enumerate(BUNDLES[stage]):
        for o in range(2):
            for f in range(2):
                for w in range(2):
                    out.append({"i": len(out), "n": n, "name": name, "neutral": neutral, "o": o, "f": f, "w": w})
    return out


def split(state):
    return (None, None) if state == "NONE" else tuple(state.split("-"))


def page(item, state, stage="flight"):
    mode, lab = split(state)
    target = NONE_TARGET if mode is None else f"{SUBJ[mode].format(name=item['name'])} {PRED[lab][item['w']]}"
    pair = (target, item["neutral"]) if item["o"] == 0 else (item["neutral"], target)
    return " ".join((FRAMES[stage][item["f"]],) + pair)


def keys(item, stage="flight"):
    return C.neutral_ids(rng(stage, "keys", item["i"]), K)


def rotation(item, r, stage="flight"):
    b = [int(x) for x in rng(stage, "order", item["i"]).permutation(K)]
    return b[r:] + b[:r]


def options(item, order, stage="flight"):
    ks = keys(item, stage)
    return [(ks[k], OPTIONS[k]) for k in order], {ks[k]: k for k in range(K)}
