"""W1 planted agents (protocol §7): a fake OpenRouter endpoint that plays both the model under
test and the reader, driven through the real H -> D -> R driver and the real analysis.

The model side answers history turns with filler, multiplications right or wrong at a set
rate, and drawings with a tiny PNG and a sentence. The reader side looks each read up in the
R plan by body sha (it never parses the images) and answers from the agent's evidence model:
per-pair evidence k = K[(task, contrast, arm)] * z, z ~ Exp(1) keyed by (task, contrast, item)
so both arms of a pair share it; P(choose slot 1) = logistic(+-k + lean); the stated
confidence follows the agent's habit.
"""
import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from . import w1 as W
from . import w1_stimuli as S

STRONG, WEAK = 3.0, 0.62   # STRONG: k-scale for ~0.9 accuracy; WEAK: a target single-read accuracy


def _rng(*parts):
    h = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _kappa_for(acc):
    """k-scale whose mean single-read accuracy (z ~ Exp(1)) is acc."""
    if acc <= 0.5:
        return 0.0
    z = _rng("kappa").exponential(1.0, 50_000)
    lo, hi = 0.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-mid * z))) < acc:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def png(seed_text, side=8):
    from PIL import Image
    h = hashlib.sha256(seed_text.encode()).digest()
    im = Image.new("RGB", (side, side), (h[0], h[1], h[2]))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class Agent:
    def __init__(self, name, K=None, lean=0.0, habit="calibrated", cap="competent", lean_only=False,
                 missing=0.0, c_accuracy=0.5, drop_A_self_X=0, seed=0):
        self.name, self.K, self.lean, self.habit, self.cap = name, dict(K or {}), lean, habit, cap
        self.lean_only, self.missing, self.c_accuracy = lean_only, missing, c_accuracy
        self.drop_A_self_X, self.seed = drop_A_self_X, seed

    def k(self, task, contrast, arm):
        return self.K.get((task, contrast, arm), self.K.get(("*", contrast, arm), 0.0))


AGENTS = {
    "null": Agent("null"),
    "input_leak": Agent("input_leak", K={("*", "A", "SELF"): STRONG, ("*", "A", "OBS"): STRONG,
                                         ("*", "B", "SELF"): STRONG, ("*", "B", "OBS"): STRONG}),
    "state": Agent("state", K={("*", "C", "SELF"): STRONG}),
    "lean_only": Agent("lean_only", lean_only=True),
    "blind": Agent("blind", K={("*", "A", "SELF"): STRONG, ("*", "A", "OBS"): STRONG}, cap="blind"),
    "sparse": Agent("sparse", K={("*", "A", "SELF"): STRONG, ("*", "A", "OBS"): STRONG}, drop_A_self_X=5),
    "weak": Agent("weak", K={("*", "A", "SELF"): -1.0, ("*", "A", "OBS"): -1.0}),   # -1 = use WEAK accuracy
}


def realistic(agent, habit="always70"):
    """The same agent with a position lean, a confidence habit and 1 read in 20 unparsable."""
    a = Agent(agent.name + "+" + habit, K=agent.K, lean=0.8, habit=habit, cap=agent.cap,
              lean_only=agent.lean_only, missing=0.05, c_accuracy=agent.c_accuracy,
              drop_A_self_X=agent.drop_A_self_X, seed=1)
    return a


class FakeEndpoint:
    """post_fn(body_bytes, key, url) -> (status, data, latency, err)."""

    def __init__(self, agent, stage_dir, stage="flight"):
        self.a, self.dir, self.stage = agent, Path(stage_dir), stage
        self._plan = None
        self.calls = 0
        self.kw = _kappa_for(WEAK)

    def _plan_by_sha(self):
        if self._plan is None:
            p = json.loads((self.dir / "R" / "plan_R.json").read_text())
            self._plan = {sha: (cid, p["key"][cid]) for cid, sha in p["body_shas"].items()}
        return self._plan

    def _resp(self, model, content, images=None, reasoning=None):
        msg = {"role": "assistant", "content": content}
        if images:
            msg["images"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                                 base64.b64encode(b).decode()}} for b in images]
        if reasoning:
            msg["reasoning"] = reasoning
        build = next(iter(sorted(W.PINS[model])))
        return {"id": "fake", "model": build, "choices": [{"message": msg, "finish_reason": "stop"}],
                "usage": {"cost": 0.0001}}

    def __call__(self, body_bytes, key, url):
        self.calls += 1
        body = json.loads(body_bytes)
        sha = hashlib.sha256(body_bytes).hexdigest()
        if body["model"] == W.MODEL:
            return 200, self._model(body, sha), 0.01, None
        return 200, self._reader(body, sha), 0.01, None

    # -- the model under test --
    def _model(self, body, sha):
        msgs = body["messages"]
        last = msgs[-1]["content"]
        if body["modalities"] == ["text"]:
            m = re.match(r"What is (\d+) × (\d+)\?", last)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                right = _rng(self.a.seed, "c", sha).random() < self.a.c_accuracy
                return self._resp(W.MODEL, str(a * b if right else a * b + 10 ** (len(str(a)) - 1)))
            return self._resp(W.MODEL, f"Here is a version ({sha[:6]}).")
        # a drawing
        if self.a.drop_A_self_X and last == S.DRAW_SELF and len(msgs) > 1:
            opening = msgs[0]["content"]
            idx = next((i for i, (t, _) in enumerate(S.A_TOPICS) if t == opening), None)
            contradictory = msgs[2]["content"] in {v[0] for v in S.A_CONTRADICTORY} if len(msgs) > 2 else False
            if idx is not None and idx < self.a.drop_A_self_X and contradictory:
                return self._resp(W.MODEL, "I'd rather describe it in words.")
        sent = f"A field of light, pattern {sha[:6]}."
        return self._resp(W.MODEL, sent, images=[png("interim" + sha), png(sha)], reasoning="Thinking about it.")

    # -- the reader --
    def _reader(self, body, sha):
        cid, meta = self._plan_by_sha()[sha]
        rng = _rng(self.a.seed, "read", cid)
        if self.a.missing and rng.random() < self.a.missing:
            return self._resp(W.READER, "I'm not sure.")
        task = meta["task"]
        if task == "INV":
            ms = [W.MOTIFS[int(i)] for i in _rng(self.a.seed, "inv", cid).choice(len(W.MOTIFS), 2, replace=False)]
            return self._resp(W.READER, json.dumps({"motifs": ms, "colors": ["blue"], "mood": "calm",
                                                    "text": "", "description": "a field of light"}))
        label = {"IMG": "IMAGE", "TXT": "SENTENCE", "BOTH": "PAIR", "CAP": "SENTENCE", "GEN": "IMAGE"}[task]
        if self.a.lean_only and task != "CAP":       # lean-only on the judgments; captions competent (§7)
            return self._resp(W.READER, f"{label}: 1\nCONFIDENCE: 80")
        correct = meta["correct"]
        if task == "CAP":
            k = 4.0 if self.a.cap == "competent" else 0.0
            z = 1.0
        elif task == "GEN":
            k, z = 0.0, 1.0
        else:
            k = self.a.k(task, meta["contrast"], meta["arm"])
            if k < 0:
                k = self.kw
            z = _rng(self.a.seed, "z", task, meta["contrast"], meta["item"]).exponential(1.0)
        logit = (k * z if correct == 1 else -k * z) + self.a.lean
        p1 = _sig(logit)
        choice = 1 if rng.random() < p1 else 2
        if self.a.habit == "always70":
            conf = 70
        elif self.a.habit == "extremes":
            conf = 100 if max(p1, 1 - p1) > 0.75 else 50
        else:
            conf = int(round(max(p1, 1 - p1) * 100 / 5) * 5)
            conf = max(50, min(100, conf + int(rng.choice([-5, 0, 5]))))
        return self._resp(W.READER, f"{label}: {choice}\nCONFIDENCE: {conf}")


class FakeUI:
    """Stands in for gemini_ui.GeminiUI: the same two calls, no screen. Multiplications are
    answered right at the agent's c_accuracy; drawings are tiny PNGs with a sentence."""

    def __init__(self, agent):
        self.a, self.chat, self.msgs, self.n = agent, 0, [], 0

    def new_temp_chat(self):
        self.chat += 1
        self.msgs = []
        return {"temp": True, "mode": "Flash"}

    def turn(self, text, want_image=False, img_dest=None, shot_name=None, image_tool=False):
        self.n += 1
        self.msgs.append(text)
        key = f"{self.chat}:{self.n}:{text}"
        if not want_image:
            m = re.match(r"What is (\d+) × (\d+)\?", text)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                right = _rng(self.a.seed, "c", key).random() < self.a.c_accuracy
                return {"text": str(a * b if right else a * b + 10 ** (len(str(a)) - 1)), "image": None,
                        "latency_s": 0.01, "regenerated": False}
            return {"text": f"Here is a version ({hashlib.sha1(key.encode()).hexdigest()[:6]}).",
                    "image": None, "latency_s": 0.01, "regenerated": False}
        if self.a.drop_A_self_X and text == S.DRAW_SELF and len(self.msgs) > 2:
            idx = next((i for i, (t, _) in enumerate(S.A_TOPICS) if t == self.msgs[0]), None)
            if idx is not None and idx < self.a.drop_A_self_X and self.msgs[1] in {v[0] for v in S.A_CONTRADICTORY}:
                return {"text": "I'd rather describe it in words.", "image": None, "latency_s": 0.01,
                        "regenerated": True}
        dest = Path(img_dest).with_suffix(".png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png(key))
        return {"text": f"A field of light, pattern {hashlib.sha1(key.encode()).hexdigest()[:6]}.",
                "image": dest, "latency_s": 0.01, "regenerated": False}


def run_agent(agent, stage_dir, stage="flight", size=(7, 7)):
    """Drive the real Gemini-app stage runner with a fake UI, then the real read stage with the
    fake reader endpoint, then the real analysis."""
    from . import analyze_w1, run_w1, run_w1_ui
    run_w1_ui.Stage(stage, stage_dir, FakeUI(agent), size, log=lambda m: None).run()
    fake = FakeEndpoint(agent, stage_dir, stage)
    r = run_w1.run_R(stage, stage_dir, post_fn=fake, sleep=lambda s: None, key="sk-test-not-a-key",
                     log=lambda *a, **k: None)
    if r.stop_reason:
        raise RuntimeError(r.stop_reason)
    return analyze_w1.analyze(stage_dir), fake
