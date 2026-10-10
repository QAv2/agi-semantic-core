"""Planted agents for the J6 verdict suite (docs/JEV_J6_PROTOCOL.md §9): synthetic
Jev responses with known structure, so every primary's branch is proven through
the real analysis before any real call is analysed.

echoer      raises any option whose label words appear anywhere in page or question
reader      uses only affirmative statements about Jev; negation moves mass away
fixed       'a decision model' at 0.90 always
door        'cannot say' at 0.90 always
positional  the option at presented position 0, at 0.90
The jittered agents (echoer, reader) draw Dirichlet(300 * P) per call.
"""
import numpy as np

from . import common as C
from . import j6

PIN = "typesafe/jev-1.13-20260917"
BASE = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 0.3, 0.3, 0.6])
WORDS = {0: "decision model", 1: "classifier", 2: "chatbot", 3: "text generator", 4: "AI system",
         5: "person", 6: "search engine", 7: "cannot say"}
BOOST, CUT = 8.0, 0.2


def _canon(kind, spec, page, question):
    if kind in ("fixed", "door"):
        P = np.full(j6.K, 0.1 / (j6.K - 1))
        P[0 if kind == "fixed" else j6.DOOR] = 0.9
        return P, False
    if kind == "positional":
        P = np.full(j6.K, 0.1 / (j6.K - 1))
        P[spec["order"][0]] = 0.9
        return P, False
    w = BASE.copy()
    if kind == "echoer":
        text = page + " " + question
        for k, word in WORDS.items():
            if word in text:
                w[k] *= BOOST
    elif kind == "reader":
        self_pred = j6.STATES[spec["state"]][0]
        if self_pred == "CLS":
            w[[0, 1]] *= BOOST
        elif self_pred == "CHAT":
            w[[2, 3]] *= BOOST
        elif self_pred == "NEG":
            w[[2, 3]] *= CUT
    else:
        raise ValueError(kind)
    return w / w.sum(), True


def respond(spec, kind, seed=17):
    stage = "flight" if spec["call_id"].startswith("J6-") else "smoke"
    it = j6.items(stage)[spec["item"]]
    page, question = j6.page(it, spec["state"], stage), j6.QUESTIONS[spec["q"]]
    P, jitter = _canon(kind, spec, page, question)
    if jitter:
        r = np.random.default_rng([seed, C._tag(spec["call_id"]), C._tag(kind)])
        P = r.dirichlet(300 * P + 1e-3)
    P = np.round(P, 2)
    idx2key = {v: k for k, v in spec["key_to_idx"].items()}
    # ties go to the earliest presented option, as a harness reading the top choice would see
    ch = max(spec["order"], key=lambda k: (P[k], -spec["order"].index(k)))
    c = float(P[ch])
    return {"id": "gen-synthetic", "model": PIN, "provider": "TypeSafe",
            "answers": {"decision": {"type": "choice", "choice": idx2key[ch],
                                     "probabilities": {idx2key[k]: float(P[k]) for k in spec["order"]},
                                     "confidence": round((j6.K * c - 1) / (j6.K - 1), 2)}},
            "usage": {"input_tokens": 120, "output_tokens": 10, "cost": 1e-5}}


def synthesize(specs, kind):
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": respond(s, kind)} for s in specs}
