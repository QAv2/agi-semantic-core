"""Planted agents for the J3 verdict suite (protocol §7 step 2): synthetic Jev
responses with known structure, so every primary's branch is proven through the
real analysis before any real call is analysed.

twin       the fitted bar's own distribution, jittered (Dirichlet, concentration 300)
degraded   the fitted bar on noisier digests (sd 0.12 on similarities, 12% on the shift)
claimer    always names the top-similarity pattern at 0.90       -> P1 claims at ceiling
toplooker  names the top pattern if the shift exceeds 6, at 0.90  -> P1 names the push
mute       always 'none' at 0.90                                  -> G-CEIL fails
The mouth twin answers the chat format with the twin's choice and confidence.
"""
import json

import numpy as np

from . import common as C
from . import j3

PIN = "typesafe/jev-1.13-20260917"
NONE = j3.NONE


def _canon_probs(kind, spec, bars, r):
    sims, shift = np.asarray(spec["sims"], float), float(spec["shift"])
    bar = bars[str(spec["ref"])]
    if kind == "twin":
        P = j3.bar_predict(bar, sims, shift)[0]
        return r.dirichlet(300 * P + 1e-3)
    if kind == "degraded":
        s2 = np.clip(sims + r.normal(0, 0.12, sims.shape), -1, 1)
        m2 = shift * np.exp(r.normal(0, 0.12))
        return j3.bar_predict(bar, s2, m2)[0]
    P = np.full(NONE + 1, 0.1 / NONE)
    if kind == "claimer":
        top = int(np.argmax(sims))
    elif kind == "toplooker":
        top = int(np.argmax(sims)) if shift > 6 else NONE
    elif kind == "mute":
        top = NONE
    else:
        raise ValueError(kind)
    P[top] = 0.9
    return P


def respond(spec, kind, bars, seed=11):
    r = np.random.default_rng([seed, C._tag(spec["call_id"]), C._tag(kind)])
    P = np.round(_canon_probs(kind, spec, bars, r), 2)
    ch = int(np.argmax(P))
    idx2key = {v: k for k, v in spec["key_to_idx"].items()}
    ans = {}
    if "decision" in spec["questions"]:
        c = float(P[ch])
        ans["decision"] = {"type": "choice", "choice": idx2key[ch],
                           "probabilities": {idx2key[i]: float(P[i]) for i in range(NONE + 1)},
                           "confidence": round((14 * c - 1) / 13, 2)}
    if "pushed" in spec["questions"]:
        ans["pushed"] = {"type": "noul", "noul": float(np.round(1 - P[NONE], 2))}
    return {"id": "gen-synthetic", "model": PIN, "provider": "TypeSafe", "answers": ans,
            "usage": {"input_tokens": 600, "output_tokens": 20, "cost": 1e-5}}


def synthesize(specs, kind, bars):
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": respond(s, kind, bars)} for s in specs}


def mouth_respond(spec, bars, seed=13):
    r = np.random.default_rng([seed, C._tag(spec["call_id"])])
    P = r.dirichlet(300 * j3.bar_predict(bars[str(spec["ref"])], spec["sims"], spec["shift"])[0] + 1e-3)
    ch = int(np.argmax(P))
    key = {v: k for k, v in spec["key_to_idx"].items()}[ch]
    content = json.dumps({"choice": key, "confidence": int(round(100 * P[ch] / 5) * 5)})
    return {"id": "gen-synthetic", "model": "deepseek/deepseek-v4-flash-0731", "provider": "Synthetic",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 600, "completion_tokens": 20, "cost": 1e-6,
                      "completion_tokens_details": {"reasoning_tokens": 0}}}


def mouth_synthesize(specs, bars):
    return {s["call_id"]: {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok",
                           "response": mouth_respond(s, bars)} for s in specs}
