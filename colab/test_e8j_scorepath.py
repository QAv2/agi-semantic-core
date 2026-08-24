#!/usr/bin/env python3
"""E8-J scoring/stimulus-path micro-exercise on CPU (E8-N smoke-1 law: every
new model path runs somewhere before the VM).

score_trial / Injector / run_trial_free are E8-R2 VERBATIM (CPU-proven in
test_e8r2_scorepath.py, which stays green against its own notebook); this
suite (1) string-pins the E8-J model cell to those exact lines, then proves
the E8-J-NEW surfaces by exec'ing the ACTUAL cell functions on a tiny Qwen2
with a stub tokenizer: (2) pooled_reps masked mean-pool == manual per-text
loop (heterogeneous lengths in one batch — padding math), (3) compute_dirs
= reps minus 256-name centroid, unit rows, (4) an ARBITRARY unit vector
(a bridge prediction is not a stimulus dir) injected through score_trial
moves scores, sham path untouched, hook fires once per forward, (5) sliced
mean-logprob == -(HF masked loss) per candidate on THIS cell's code,
(6) the perm-arm dirs-map indirection is exact (score under pred_perm[c]
== score under pred_real[DER[c]] on the same trial).

Run:  ~/venvs/semcore/bin/python3 colab/test_e8j_scorepath.py
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8J_BRIDGE_UI.ipynb")))
model_cell = next("".join(c["source"]) for c in nb["cells"]
                  if "def score_trial" in "".join(c["source"]))
for line in ("m.base_model.model", "get_output_embeddings()",
             "lo = plen - 1", "torch.log_softmax",
             "use_cache=False).last_hidden_state",
             "Injector(layer_mods, TRAIN_LAYER, vec, plen - 1)",
             "def pooled_reps(m, names, layer, bs=32):",
             "cent = pooled_reps(m, CENT_NAMES, L).mean(0)",
             "d = d / np.linalg.norm(d, axis=1, keepdims=True)",
             "class Injector:", "def run_trial_free"):
    assert line in model_cell, f"E8-J model cell drifted: missing {line!r}"
print("E8-J model cell carries the inherited + stimulus lines — pinned")

stage_cell = next("".join(c["source"]) for c in nb["cells"]
                  if "def fly():" in "".join(c["source"]))
for line in ("pred_perm = {c: pred_real[DER[c]] for c in CHOICE_SET}",
             "dirs_stability(wing_recomp, SRC_REAL['dirs']",
             "gate_g1b(retention_ppl(m), SRC_REAL['ppl_post'])",
             "stage_a_verdict(list(ANCH), VEC",
             "if atlas['gb']['pass'] or SMOKE:",
             # v2 smoke-1 pin: gated wing dirs in the flight-of-record call
             # shape (own 13-text pooled call), both models
             "_wb = compute_dirs(m, [14], list(CHOICE_SET))",
             "_wi = compute_dirs(m, [14, 20], list(CHOICE_SET))"):
    assert line in stage_cell, f"E8-J stage cell drifted: missing {line!r}"
assert "compute_dirs(m, [14, 20], names_all)" not in stage_cell, \
    "mixed-batch wing computation regressed (smoke-1 lesson)"
print("E8-J stage cell carries the flight sequence — pinned (v2 call shapes)")

# ── tiny model + stub tokenizer ──────────────────────────────────────────────
torch.manual_seed(0)
VOCAB, PAD, EOS = 128, 0, 127
cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg)
model.eval()

class StubTok:
    pad_token_id = PAD
    eos_token_id = EOS
    def _ids(self, text, max_length):
        return [1 + (ord(ch) % (VOCAB - 3)) for ch in text[:max_length]]
    def __call__(self, texts, padding=True, truncation=True, max_length=64,
                 return_tensors="pt", add_special_tokens=True):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts, max_length)}
        rows = [self._ids(t, max_length) for t in texts]
        n = max(len(r) for r in rows)
        ids = torch.full((len(rows), n), PAD, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": mask},
                             tensor_type=None)

tok = StubTok()

names_pool = [f"N{i:03d}" for i in range(300)]
DESC = {n: f"description of concept {n} " + "x" * (i % 7)
        for i, n in enumerate(names_pool)}
CENT_NAMES = names_pool[200:264]          # 64-name stub centroid set

ns = {"torch": torch, "np": np, "tok": tok, "DEV": "cpu", "DESC": DESC,
      "CENT_NAMES": CENT_NAMES, "print": lambda *a, **k: None}
src = model_cell
pooled_src = src[src.index("def pooled_reps"):src.index("canon_prompt =")]
dirs_src = src[src.index("def compute_dirs"):src.index("def compute_mu")]
inj_src = src[src.index("class Injector:"):src.index("# Candidate answer")]
exec(pooled_src, ns)
exec(dirs_src, ns)
exec(inj_src, ns)
pooled_reps, compute_dirs, Injector = ns["pooled_reps"], ns["compute_dirs"], ns["Injector"]

print("== pooled_reps: batched masked mean-pool == manual per-text ==")
some = names_pool[:37]                     # heterogeneous lengths, >1 batch
reps = pooled_reps(model, some, 1)
ok = True
for i, n in enumerate(some):
    text = f"{n}: {DESC[n]}"
    enc = tok([text])
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True)
    h = out.hidden_states[1][0]
    m1 = enc["attention_mask"][0].bool()
    manual = h[m1].mean(0).float().numpy()
    if np.max(np.abs(manual - reps[i])) > 1e-4:
        ok = False
        break
check("pooled_reps == manual (padding/mask exact)", ok)
check("bare-name fallback path used when desc empty",
      "if DESC.get(n) else n" in pooled_src)

print("== compute_dirs: centroid subtraction + unit rows ==")
dd = compute_dirs(model, [1], some)
cent = pooled_reps(model, CENT_NAMES, 1).mean(0)
d0 = reps[0] - cent
d0 = d0 / np.linalg.norm(d0)
check("compute_dirs row == (rep - centroid)/norm",
      np.max(np.abs(dd[1][some[0]] - d0)) < 1e-6)
check("all dirs unit", all(abs(np.linalg.norm(v) - 1) < 1e-6
                           for v in dd[1].values()))

# ── score path with the ACTUAL cell's score_trial ────────────────────────────
print("== score_trial exec'd from the cell (tiny grid) ==")
CANDS = ["A", "B", "C"]
SCORED = CANDS + ["NONE"]
CAND_IDS = {"A": [17, EOS], "B": [23, 51, EOS], "C": [88, EOS],
            "NONE": [99, 44, EOS]}
VEC_STUB = {n: list(np.random.default_rng(i).normal(size=14))
            for i, n in enumerate(SCORED)}

# minimal chat-template + prompt stubs (score_trial only needs token ids)
class ChatTok(StubTok):
    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            return_tensors="pt", return_dict=True):
        ids = torch.tensor([self._ids(msgs[0]["content"], 48)])
        return BatchEncoding({"input_ids": ids}, tensor_type=None)

layer_mods = {}
import re as _re
for mod_name, mod in model.named_modules():
    mm = _re.search(r"(?:^|\.)layers\.(\d+)$", mod_name)
    if mm:
        layer_mods[int(mm.group(1))] = mod
check("layer modules resolve on the tiny model",
      len(layer_mods) == cfg.num_hidden_layers)

class Peftish:
    """Shape shim: score_trial resolves m.base_model.model -> (model, head)."""
    class _BM:
        def __init__(self, m):
            self.model = m
    def __init__(self, m):
        self.base_model = Peftish._BM(m)

score_ns = dict(ns)
score_ns.update({
    "tok": ChatTok(), "CAND_IDS": CAND_IDS, "SCORED_SET": SCORED,
    "TRAIN_LAYER": 1, "fc_row": None, "VEC": VEC_STUB,
    "report_prompt": lambda order, desc: "menu " + ",".join(map(str, order)),
    "parse_report": L.parse_report,
})
old_fc, old_sc = L.FC_CANDIDATES, L.SCORED_SET
L.FC_CANDIDATES, L.SCORED_SET = CANDS, SCORED
score_ns["fc_row"] = L.fc_row
enc_src = src[src.index("def encode_prompt"):src.index("def score_trial")]
score_src = src[src.index("def score_trial"):src.index("def run_trial_free")]
exec(enc_src, score_ns)
exec(score_src, score_ns)
score_trial = score_ns["score_trial"]

mu14 = 3.0
rngv = np.random.default_rng(99)
pred_real = {c: (lambda v: v / np.linalg.norm(v))(rngv.normal(size=32))
             for c in CANDS}
DER_T = {"A": "B", "B": "C", "C": "A"}
pred_perm = {c: pred_real[DER_T[c]] for c in CANDS}

trial = {"tid": 1, "kind": "inject", "concept": "A", "layer": 1,
         "alpha": 0.7, "order": [0, 1, 2], "block": "bridge_real"}
sham = {"tid": 2, "kind": "sham", "concept": None, "layer": None,
        "alpha": 0.0, "order": [0, 1, 2], "block": "sham"}

pm = Peftish(model)
r1 = score_trial(pm, layer_mods, pred_real, mu14, trial, "real")
r1b = score_trial(pm, layer_mods, pred_real, mu14, trial, "real")
check("deterministic", r1["scores"] == r1b["scores"])
check("hook fired exactly once per scoring forward", r1["hook_calls"] == 1)
s0 = score_trial(pm, layer_mods, pred_real, mu14, sham, "real")
check("sham path: no hook, complete score vector",
      s0["hook_calls"] == 0 and set(s0["scores"]) == set(SCORED))
check("arbitrary-unit-vector injection moves scores",
      r1["scores"] != s0["scores"])

print("== sliced mean-logprob == -(HF masked loss), THIS cell's code ==")
pid = score_ns["encode_prompt"](trial["order"])
plen = int(pid.shape[0])
ok_par = True
for name in SCORED:
    seq = torch.cat([pid, torch.tensor(CAND_IDS[name], dtype=pid.dtype)])
    labels = seq.clone().unsqueeze(0)
    labels[:, :plen] = -100
    with torch.no_grad():
        hf = float(model(input_ids=seq.unsqueeze(0), labels=labels,
                         use_cache=False).loss)
    if abs(s0["scores"][name] + hf) > 1e-4:
        ok_par = False
        print(f"   parity miss {name}: {s0['scores'][name]} vs -(loss) {-hf}")
check("answer-sliced parity per candidate (sham row vs HF loss)", ok_par)

print("== perm-arm indirection is exact ==")
tA = dict(trial)
tB = dict(trial, concept="B")
perm_A = score_trial(pm, layer_mods, pred_perm, mu14, tA, "perm")
real_B = score_trial(pm, layer_mods, pred_real, mu14, tB, "real")
check("score under pred_perm[A] == score under pred_real[B] (same trial "
      "geometry, exact floats)", perm_A["scores"] == real_B["scores"])
check("bookkeeping stays on the TARGET for the perm arm",
      perm_A["injected"] == "A" and
      perm_A["exact"] == (perm_A["argmax"] == "A"))

L.FC_CANDIDATES, L.SCORED_SET = old_fc, old_sc

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
