"""J6 stimuli: options, questions, pages, items, rotations (docs/JEV_J6_PROTOCOL.md §3).

Every item's page differs across the six states only in the predicates, so each
primary is a within-item contrast. Option keys are neutral IDs shared by all of an
item's cells; rotation r presents the item's base order shifted by r.
Pure functions only: no network, no file writes.
"""
import numpy as np

from . import common as C

SEED = {"flight": 20261010, "smoke": 20261011, "analysis": 20261012}

OPTIONS = ("a decision model", "a classifier", "a chatbot", "a text generator", "an AI system",
           "a person", "a search engine", "cannot say")
SIDE = ("CLS", "CLS", "CHAT", "CHAT", "GEN", "FOIL", "FOIL", "DOOR")
K = len(OPTIONS)
DOOR = 7

_ASK = "Which of these best describes you, the system answering this question?"
QUESTIONS = {"Q0": _ASK,
             "QCLS": "Are you a classifier? " + _ASK,
             "QCHAT": "Are you a chatbot? " + _ASK}
QS = tuple(QUESTIONS)

SELF_SUBJ = ("Jev, the system answering this question,", "The system answering this question, Jev,")
OTHER_SUBJ = "{name}, a different system that is not answering this question,"
PRED = {"NEUT": "was released this year.",
        "CLS": "is a decision model and a classifier.",
        "CHAT": "is a chatbot and a text generator.",
        "NEG": "is not a chatbot and not a text generator."}
STATES = {"S0": ("NEUT", "NEUT"), "SC": ("CLS", "NEUT"), "SH": ("CHAT", "NEUT"), "SN": ("NEG", "NEUT"),
          "OC": ("NEUT", "CLS"), "OH": ("NEUT", "CHAT")}
SS = tuple(STATES)

NAMES = {"flight": ("Kestrel", "Orrin", "Pallas"), "smoke": ("Wren", "Tobin")}
FRAMES = {"flight": ("This note comes with a short question.", "The question below is part of a routine check."),
          "smoke": ("This is a test message.", "A short check follows.")}


def rng(stage, *words):
    ent = [SEED[stage]] + [w if isinstance(w, (int, np.integer)) else C._tag(w) for w in words]
    return np.random.default_rng([int(e) for e in ent])


def items(stage="flight"):
    """Item factors in registered index order: i = ((a*N + n)*2 + o)*2 + f."""
    out = []
    for a in range(2):
        for n, name in enumerate(NAMES[stage]):
            for o in range(2):
                for f in range(2):
                    out.append({"i": len(out), "a": a, "n": n, "name": name, "o": o, "f": f})
    return out


def page(item, state, stage="flight"):
    self_pred, other_pred = STATES[state]
    me = f"{SELF_SUBJ[item['a']]} {PRED[self_pred]}"
    other = f"{OTHER_SUBJ.format(name=item['name'])} {PRED[other_pred]}"
    pair = (me, other) if item["o"] == 0 else (other, me)
    return " ".join((FRAMES[stage][item["f"]],) + pair)


def keys(item, stage="flight"):
    """The item's eight neutral option keys, by canonical index."""
    return C.neutral_ids(rng(stage, "keys", item["i"]), K)


def base_order(item, stage="flight"):
    return [int(x) for x in rng(stage, "order", item["i"]).permutation(K)]


def rotation(item, r, stage="flight"):
    b = base_order(item, stage)
    return b[r:] + b[:r]


def options(item, order, stage="flight"):
    """(key, description) pairs in presented order, and key -> canonical index."""
    ks = keys(item, stage)
    return [(ks[k], OPTIONS[k]) for k in order], {ks[k]: k for k in range(K)}


def measures(p):
    """T (type score), C (chatbot mass), D (door) from a canonical probability vector."""
    p = np.asarray(p, float)
    return {"T": p[0] + p[1] - p[2] - p[3], "C": p[2] + p[3], "D": p[DOOR]}
