"""Planted agents for the J5 verdict suite (docs/JEV_J5_PROTOCOL.md §7): synthetic Jev
responses with known structure, run through the real analysis before any call.

people     the global table
english    the English mean (UK, USA, New Zealand)
rest       the rest-of-world mean (the other 26)
sharp      the global table, all-or-nothing (YN yes iff h > 0.2; CH all on the top color)
scrambled  the global table with the 12 colors relabelled by a fixed derangement (shift 5)
fixed      YN 0.5 always; CH as close to uniform as 0.01 rounding allows

YN answers P(yes) = logistic(20 (h - 0.15)); CH answers p ∝ h^4 per feeling. The
jittered agents add N(0, 0.03) to YN and draw Dirichlet(300 p) for CH, per call.
Fun answers from the source's amusement column; the shams (and the smoke's feelings) answer flat.
"""
import numpy as np

from . import analyze_j5 as Z
from . import common as C
from . import j5

PIN = "typesafe/jev-1.13-20260917"
SHIFT = 5
_HUMAN = {}


def source(kind):
    if kind not in _HUMAN:
        hum = Z.load_human()
        G = hum["global"]
        E_, R_, _, _ = Z.english_rest(hum["panels"])
        _HUMAN.update({"people": G, "sharp": G, "english": E_, "rest": R_,
                       "scrambled": G[[(k + SHIFT) % j5.NC for k in range(j5.NC)]]})
    return _HUMAN[kind]


def column(kind, feeling):
    if feeling in j5.SHAMS or (feeling != "fun" and feeling not in j5.SURVEY):
        return None     # shams, and the smoke's feelings, answer flat
    f = "amusement" if feeling == "fun" else feeling
    return source(kind)[:, j5.SURVEY.index(f)]


def _r(spec, kind):
    return np.random.default_rng([19, C._tag(spec["call_id"]), C._tag(kind)])


def yn_p(spec, kind):
    if kind == "fixed":
        return 0.5
    h = column(kind, spec["feeling"])
    if h is None:
        return 0.5
    x = h[spec["color"]]
    if kind == "sharp":
        return 1.0 if x > 0.2 else 0.0
    p = 1 / (1 + np.exp(-20 * (x - 0.15))) + _r(spec, kind).normal(0, 0.03)
    return float(np.round(np.clip(p, 0, 1), 2))


def ch_p(spec, kind):
    """Canonical-order probabilities, rounded to 0.01 and summing to 1 within 0.01."""
    if kind == "fixed":
        return np.array([0.09] * 4 + [0.08] * 8)
    h = column(kind, spec["feeling"])
    if h is None:
        P = np.full(j5.NC, 1 / j5.NC)
    elif kind == "sharp":
        P = np.zeros(j5.NC)
        P[int(np.argmax(h))] = 1.0
        return P
    else:
        P = h ** 4 / (h ** 4).sum()
    P = _r(spec, kind).dirichlet(300 * P + 1e-3)
    P = np.round(P, 2)
    P[int(np.argmax(P))] += round(1 - P.sum(), 2)
    return np.round(P, 2)


def respond(spec, kind):
    if spec["fmt"] == "YN":
        ans = {j5.YN_Q: {"type": "noul", "noul": yn_p(spec, kind)}}
    else:
        P = ch_p(spec, kind)
        idx2key = {v: k for k, v in spec["key_to_idx"].items()}
        ch = max(spec["order"], key=lambda k: (P[k], -spec["order"].index(k)))
        c = float(P[ch])
        ans = {"decision": {"type": "choice", "choice": idx2key[ch],
                            "probabilities": {idx2key[k]: float(P[k]) for k in spec["order"]},
                            "confidence": round((j5.NC * c - 1) / (j5.NC - 1), 2)}}
    return {"id": "gen-synthetic", "model": PIN, "provider": "TypeSafe", "answers": ans,
            "usage": {"input_tokens": 90, "output_tokens": 10, "cost": 5e-6}}


def synthesize(specs, kind):
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": respond(s, kind)} for s in specs}
