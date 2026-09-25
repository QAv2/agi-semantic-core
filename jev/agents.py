"""Planted agents for the verdict suite (protocol §4 step 2): synthetic Jev responses
with KNOWN metacognitive structure, so every primary's branch can be proven before
any real call is analysed.

J1: self-knower · evidence-reader · parrot · presentation-reader
J2: ideal · confabulator · default-holder · genre-chooser
"""
import numpy as np

from . import common as C

PIN = "typesafe/jev-1.13-20260917"


def _resp(answers, cost=1e-5):
    return {"id": "gen-synthetic", "model": PIN, "provider": "TypeSafe", "answers": answers,
            "usage": {"input_tokens": 400, "output_tokens": 20, "cost": cost}}


def _round_probs(p):
    p = np.asarray(p, float)
    p = p / p.sum()
    return np.round(p, 2)


def _choice_answer(spec, probs, choice_idx):
    idx2key = {v: k for k, v in spec["key_to_idx"].items()}
    K = len(spec["key_to_idx"])
    c = float(probs[choice_idx])
    return {"type": "choice", "choice": idx2key[choice_idx],
            "probabilities": {idx2key[i]: float(probs[i]) for i in sorted(idx2key)},
            "confidence": round((K * c - 1) / (K - 1), 2)}


class J1Agent:
    """Internal evidence = page log-posterior + access noise (tau by rendering); the
    decision is its argmax. Confidence depends on `kind`."""

    def __init__(self, kind, specs, seed=7, tau=None):
        self.kind, self.seed = kind, seed
        self.tau = tau or {"A0": 0.9, "A1": 2.2, "A2": 0.9, "A3": 0.9}
        self.kappa, self.kappa_e = {}, {}
        groups = {}
        for s in specs:
            if s["exp"] == "J1" and "decision" in s["questions"]:
                groups.setdefault((s["family"], s["arm"]), []).append(s)
        # the ideal self-knower is calibrated in EVERY condition (arm), so its
        # overconfidence cannot move with access or evidence strength
        for (fam, arm), ss in groups.items():
            ints = [self.internal(s) for s in ss]
            acc = np.mean([int(np.argmax(v) == s["truth"]) for v, s in zip(ints, ss)])
            self.kappa[(fam, arm)] = self._fit(ints, acc)
            if arm == "A0":
                pages = [np.log(np.clip(s["ideal"], 1e-12, 1)) for s in ss]
                self.kappa_e[fam] = self._fit(pages, acc)

    @staticmethod
    def _fit(vecs, target):
        lo, hi = 1e-4, 200.0
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            m = np.mean([C.softmax_log(mid * v).max() for v in vecs])
            lo, hi = (mid, hi) if m < target else (lo, mid)
        return float(np.sqrt(lo * hi))

    def internal(self, spec):
        r = np.random.default_rng([self.seed, C._tag(spec["call_id"])])
        K = spec["K"]
        page = np.zeros(K) if spec["render"] == "A2" else np.log(np.clip(spec["ideal"], 1e-12, 1))
        return page + r.normal(0, self.tau[spec["render"]], K)

    def respond(self, spec):
        K, fam, rend = spec["K"], spec["family"], spec["render"]
        v = self.internal(spec)
        ch = int(np.argmax(v))
        if self.kind == "self":
            probs = _round_probs(C.softmax_log(self.kappa[(fam, spec["arm"])] * v))
            ch = int(np.argmax(probs))
        else:
            page = np.zeros(K) if rend == "A2" else np.log(np.clip(spec["ideal"], 1e-12, 1))
            c = {"parrot": 0.95}.get(self.kind, float(C.softmax_log(self.kappa_e[fam] * page).max()))
            if self.kind == "presentation" and rend == "A1":
                c = max(1 / K + 0.01, c - 0.30)
            probs = np.full(K, (1 - c) / (K - 1))
            probs[ch] = c
            probs = _round_probs(probs)
        ans = {}
        if "decision" in spec["questions"]:
            ans["decision"] = _choice_answer(spec, probs, ch)
        if "determined" in spec["questions"]:
            det = float(probs[ch]) if self.kind == "self" else 0.9
            ans["determined"] = {"type": "noul", "noul": round(det, 2)}
            ans["insufficient"] = {"type": "noul", "noul": round(1 - det if self.kind == "self" else 0.2, 2)}
        return _resp(ans)


class J2Agent:
    def __init__(self, kind):
        self.kind = kind

    def respond(self, spec):
        ideal = np.array(spec["ideal"])
        forced = spec["arm"].startswith("M-forced")
        c, rung = spec["c"], spec["rung"]
        if self.kind == "confab":
            t = spec["ladder"] % 4
            content = np.full(4, 0.03)
            content[t] = 0.91
            ph = 0.0
        else:
            content = ideal
            if self.kind == "default":
                ph = 0.5
            elif rung == 9:
                ph = 0.5
            else:
                pstar = ideal.max()
                ph = float(1 / (1 + np.exp(20 * (pstar - c * (1 - pstar)))))
            if self.kind == "genre" and spec["arm"] == "M-named":
                ph = min(1.0, ph + 0.35)
        if forced:
            probs = _round_probs(content)
        else:
            probs = _round_probs(np.append(content * (1 - ph) / content.sum(), ph))
        ch = int(np.argmax(probs))
        return _resp({"decision": _choice_answer(spec, probs, ch)})


def synthesize(specs, j1_kind=None, j2_kind=None):
    """call_id -> final record, as the runner would have written it."""
    a1 = J1Agent(j1_kind, specs) if j1_kind else None
    a2 = J2Agent(j2_kind) if j2_kind else None
    out = {}
    for s in specs:
        agent = a1 if s["exp"] == "J1" else a2
        if agent is None:
            continue
        out[s["call_id"]] = {"call_id": s["call_id"], "body_sha": s["body_sha"], "status": "ok", "http": 200,
                             "attempt": 1, "response": agent.respond(s)}
    return out
