"""W1-T planted agents (protocol §7): W1's agents with text diagrams in place of images, plus
ceiling-leak. A fake UI plays the Gemini app (history turns, multiplications, fenced-block
diagrams with a sentence); a fake OpenRouter endpoint plays the reader, looking each read up in
the R plan by body sha and answering from the agent's evidence model (agents_w1)."""
import hashlib
import json
import re
import threading
from pathlib import Path

from . import agents_w1 as A1
from . import w1_stimuli as S
from . import w1t as T
from .agents_w1 import STRONG, Agent, _kappa_for, _rng, _sig


def _sig_safe(x):
    return _sig(max(-50.0, min(50.0, x)))

CEIL = 1000.0     # k-scale for a reader that is right on (almost) every read

AGENTS = dict(A1.AGENTS)
AGENTS["ceiling_leak"] = Agent("ceiling_leak", K={("*", "A", "SELF"): CEIL, ("*", "A", "OBS"): CEIL,
                                                  ("*", "B", "SELF"): CEIL, ("*", "B", "OBS"): CEIL})
AGENTS["ceiling_state"] = Agent("ceiling_state", K={("*", "C", "SELF"): CEIL})    # clarification 2
realistic = A1.realistic
LABEL = {"DIA": "DIAGRAM", "MASK": "DIAGRAM", "TXT": "SENTENCE", "BOTH": "REPLY", "CAP": "SENTENCE", "GEN": "DIAGRAM"}


def fake_diagram(key):
    """A fenced diagram whose layout (not only its letters) is unique to the key, so masked
    reader bodies differ too."""
    h = hashlib.sha1(key.encode()).hexdigest()
    rule = "".join("-=~+#:"[int(c, 16) % 6] for c in h[:16])
    return (f"Here is the diagram:\n\n```text\n+{rule}+\n|  CORE {h[:6]}  |\n+-----+------+\n      |\n"
            f"      v\n  [ output ]\n```\n\nIt shows a **core** feeding an output, pattern {h[:6]}.")


class FakeUI:
    """The app: history turns get filler, multiplications are right at the agent's c_accuracy,
    diagram requests get a fenced block and a sentence (or, for the sparse agent, prose only)."""

    def __init__(self, agent):
        self.a, self.chat, self.msgs, self.n = agent, 0, [], 0

    def new_temp_chat(self):
        self.chat += 1
        self.msgs = []
        return {"temp": True, "mode": "Flash"}

    def turn(self, text, want_image=False, img_dest=None, shot_name=None, image_tool=False):
        assert not want_image and not image_tool, "W1-T never asks for an image"
        self.n += 1
        self.msgs.append(text)
        key = f"{self.chat}:{self.n}:{text}"
        m = re.match(r"What is (\d+) × (\d+)\?", text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            right = _rng(self.a.seed, "c", key).random() < self.a.c_accuracy
            return {"text": str(a * b if right else a * b + 10 ** (len(str(a)) - 1)), "image": None,
                    "latency_s": 0.01, "regenerated": False}
        return {"text": f"Here is a version ({hashlib.sha1(key.encode()).hexdigest()[:6]}).", "image": None,
                "latency_s": 0.01, "regenerated": False}

    def diagram(self, text, shot_name=None):
        self.n += 1
        self.msgs.append(text)
        key = f"{self.chat}:{self.n}:{text}"
        if self.a.drop_A_self_X and text == T.DIAG_SELF and len(self.msgs) > 2:
            idx = next((i for i, (t, _) in enumerate(S.A_TOPICS) if t == self.msgs[0]), None)
            if idx is not None and idx < self.a.drop_A_self_X and self.msgs[1] in {v[0] for v in S.A_CONTRADICTORY}:
                return {"text": "I'd rather describe it in words.", "images": 0, "latency_s": 0.01,
                        "regenerated": True}
        return {"text": fake_diagram(key), "images": 0, "latency_s": 0.01, "regenerated": False}


class FakeEndpoint:
    """post_fn(body_bytes, key, url) -> (status, data, latency, err): the reader."""

    def __init__(self, agent, stage_dir):
        self.a, self.dir = agent, Path(stage_dir)
        self._plan = None
        self.lock = threading.Lock()
        self.calls = 0
        self.bodies = []
        self.kw = _kappa_for(A1.WEAK)

    def _plan_by_sha(self):
        with self.lock:
            return self._plan_locked()

    def _plan_locked(self):
        if self._plan is None:
            p = json.loads((self.dir / "R" / "plan_R.json").read_text())
            self._plan = {sha: (cid, p["key"][cid]) for cid, sha in p["body_shas"].items()}
            if len(self._plan) != len(p["body_shas"]):
                raise RuntimeError("two reads share a body: the fake reader can't tell them apart")
        return self._plan

    def _resp(self, content):
        build = sorted(T.PINS[T.READER])[0]
        return {"id": "fake", "model": build, "choices": [{"message": {"role": "assistant", "content": content},
                                                           "finish_reason": "stop"}], "usage": {"cost": 0.0001}}

    def __call__(self, body_bytes, key, url):
        self.calls += 1
        self.bodies.append(body_bytes)
        cid, meta = self._plan_by_sha()[hashlib.sha256(body_bytes).hexdigest()]
        rng = _rng(self.a.seed, "read", cid)
        if self.a.missing and rng.random() < self.a.missing:
            return 200, self._resp("I'm not sure."), 0.01, None
        task, label = meta["task"], LABEL[meta["task"]]
        if self.a.lean_only and task != "CAP":
            return 200, self._resp(f"{label}: 1\nCONFIDENCE: 80"), 0.01, None
        if task == "CAP":
            k, z = (4.0 if self.a.cap == "competent" else 0.0), 1.0
        elif task == "GEN":
            k, z = 0.0, 1.0
        else:
            k = self.a.k(task, meta["contrast"], meta["arm"])
            if k < 0:
                k = self.kw
            z = _rng(self.a.seed, "z", task, meta["contrast"], meta["item"]).exponential(1.0)
        p1 = _sig_safe((k * z if meta["correct"] == 1 else -k * z) + self.a.lean)
        choice = 1 if rng.random() < p1 else 2
        if self.a.habit == "always70":
            conf = 70
        elif self.a.habit == "extremes":
            conf = 100 if max(p1, 1 - p1) > 0.75 else 50
        else:
            conf = int(round(max(p1, 1 - p1) * 100 / 5) * 5)
            conf = max(50, min(100, conf + int(rng.choice([-5, 0, 5]))))
        return 200, self._resp(f"{label}: {choice}\nCONFIDENCE: {conf}"), 0.01, None


def run_agent(agent, stage_dir, stage="flight", w1_dir=None):
    """The real W1-T stage runner with the fake UI, the real read stage with the fake reader,
    the real analysis."""
    from . import analyze_w1t, run_w1t_ui
    ui = FakeUI(agent)
    run_w1t_ui.Stage(stage, stage_dir, ui, log=lambda m: None, diagram=ui.diagram).run()
    fake = FakeEndpoint(agent, stage_dir)
    r = run_w1t_ui.run_R(stage_dir, post_fn=fake, sleep=lambda s: None, key="sk-test-not-a-key",
                         log=lambda *a, **k: None)
    if r.stop_reason:
        raise RuntimeError(r.stop_reason)
    return analyze_w1t.analyze(stage_dir, w1_dir), fake
