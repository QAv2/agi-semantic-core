"""J5 stimuli: colors, feelings, wordings, framings, items, keys, rotations
(docs/JEV_J5_PROTOCOL.md §3).

YN asks one color-feeling pairing per call (a noul question, both criteria orders).
CH asks one feeling per call (a choice over the 12 colors, 12 cyclic rotations).
The page carries only a framing sentence. Pure functions only: no network, no file writes.
"""
import numpy as np

from . import common as C

SEED = {"flight": 20261016, "smoke": 20261017, "analysis": 20261018}
SEALED_NULL_SEED = 20260925     # the dictionary's sealed test (seed §7), kept for like-for-like

COLORS = ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "turquoise", "white",
          "yellow")
NC = len(COLORS)
SURVEY = ("interest", "amusement", "pride", "joy", "pleasure", "contentment", "admiration", "love", "relief",
          "compassion", "sadness", "guilt", "regret", "shame", "disappointment", "fear", "disgust", "contempt",
          "hate", "anger")
EXTENSION = ("fun",)
SHAMS = ("zorvish", "plemt")
FEELINGS = {"flight": SURVEY + EXTENSION + SHAMS, "smoke": ("surprise", "boredom")}
FRAMES = {"flight": ("This is a question about colors and feelings.",
                     "The question below is part of a short survey about colors."),
          "smoke": ("This is a test message.", "A short check follows.")}
YN = (("Does the color {c} go with the feeling of {f}?",
       "Yes, the color {c} goes with the feeling of {f}.",
       "No, the color {c} does not go with the feeling of {f}."),
      ("Do people associate the color {c} with the feeling of {f}?",
       "Yes, people associate the color {c} with the feeling of {f}.",
       "No, people do not associate the color {c} with the feeling of {f}."))
CH = ("Which color goes best with the feeling of {f}?",
      "Which color do people most associate with the feeling of {f}?")
OPTION = "the color {c}"
YN_Q = "answer"     # the noul question's name in the request
ROT = NC            # CH rotations per item-feeling
ORDERS = 2          # YN criteria orders: 0 = true first, 1 = false first


def rng(stage, *words):
    ent = [SEED[stage]] + [w if isinstance(w, (int, np.integer)) else C._tag(w) for w in words]
    return np.random.default_rng([int(e) for e in ent])


def items():
    """Item factors in registered index order: i = w*2 + f."""
    return [{"i": w * 2 + f, "w": w, "f": f} for w in range(2) for f in range(2)]


def page(item, stage="flight"):
    return FRAMES[stage][item["f"]]


def yn_question(item, feeling, color, o):
    ins, yes, no = (s.format(c=color, f=feeling) for s in YN[item["w"]])
    crit = {"true": yes, "false": no} if o == 0 else {"false": no, "true": yes}
    return {"type": "noul", "instructions": ins, "criteria": crit}


def ch_keys(item, feeling, stage="flight"):
    """The item-feeling's twelve neutral option keys, by canonical color index."""
    return C.neutral_ids(rng(stage, "keys", feeling, item["i"]), NC)


def ch_rotation(item, feeling, r, stage="flight"):
    b = [int(x) for x in rng(stage, "order", feeling, item["i"]).permutation(NC)]
    return b[r:] + b[:r]


def ch_options(item, feeling, order, stage="flight"):
    """(key, description) pairs in presented order, and key -> canonical color index."""
    ks = ch_keys(item, feeling, stage)
    return [(ks[k], OPTION.format(c=COLORS[k])) for k in order], {ks[k]: k for k in range(NC)}


def ch_question(item, feeling, opts):
    return C.choice_q(CH[item["w"]].format(f=feeling), opts)
