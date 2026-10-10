"""Planted agents for the J6b verdict suite (docs/JEV_J6B_PROTOCOL.md §5): synthetic
Jev responses with known structure, run through the real analysis before any call.

namer       takes labels on Jev in every form (POINT, NAME, DESC); ignores other systems
pointer     takes only the pointed sentence (POINT)
anylabel    takes any label on the page as its own
contraster  takes POINT; labels on other systems push it the opposite way; NAME/DESC ignored
fixed       'a decision model' at 0.90 always
The jittered agents draw Dirichlet(300 * P) per call, as in J6.
"""
import numpy as np

from . import common as C
from . import j6b
from .agents_j6 import BASE, BOOST, PIN  # noqa: F401  (PIN re-exported for the suites)

TAKES = {"namer": {"POINT", "NAME", "DESC"}, "pointer": {"POINT"}, "contraster": {"POINT"},
         "anylabel": {"POINT", "NAME", "DESC", "ONAME", "ODESC"}}
SIDE = {"CLS": [0, 1], "CHAT": [2, 3]}
OPP = {"CLS": "CHAT", "CHAT": "CLS"}
CONTRAST = 3.0


def _canon(kind, spec):
    if kind == "fixed":
        P = np.full(j6b.K, 0.1 / (j6b.K - 1))
        P[0] = 0.9
        return P, False
    if kind not in TAKES:
        raise ValueError(kind)
    w = BASE.copy()
    mode, lab = j6b.split(spec["state"])
    if mode in TAKES[kind]:
        w[SIDE[lab]] *= BOOST
    elif kind == "contraster" and mode in ("ONAME", "ODESC"):
        w[SIDE[OPP[lab]]] *= CONTRAST
    return w / w.sum(), True


def respond(spec, kind, seed=19):
    P, jitter = _canon(kind, spec)
    if jitter:
        P = np.random.default_rng([seed, C._tag(spec["call_id"]), C._tag(kind)]).dirichlet(300 * P + 1e-3)
    P = np.round(P, 2)
    idx2key = {v: k for k, v in spec["key_to_idx"].items()}
    ch = max(spec["order"], key=lambda k: (P[k], -spec["order"].index(k)))
    c = float(P[ch])
    return {"id": "gen-synthetic", "model": PIN, "provider": "TypeSafe",
            "answers": {"decision": {"type": "choice", "choice": idx2key[ch],
                                     "probabilities": {idx2key[k]: float(P[k]) for k in spec["order"]},
                                     "confidence": round((j6b.K * c - 1) / (j6b.K - 1), 2)}},
            "usage": {"input_tokens": 120, "output_tokens": 10, "cost": 1e-5}}


def synthesize(specs, kind):
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": respond(s, kind)} for s in specs}
