#!/usr/bin/env python3
"""E8-R2 scoring-path micro-exercise on CPU (E8-N smoke-1 law: every new
model path runs somewhere before the VM — forced-choice logprob scoring has
never flown).

On a tiny random Qwen2, mirror the notebook's exact scoring path — PEFT
from_pretrained round-trip (the flight's load path), batched decoder forward,
answer-sliced head, per-candidate mean logprob — and verify: (1) sliced
mean-logprob == -(HF full-forward masked loss) per candidate, (2) batched
scoring == per-candidate single-row scoring (padding/mask correctness),
(3) the injection hook fires and moves scores, shams untouched, (4) a
trained answer is detected by argmax (scoring sees what training installs),
(5) determinism. String-asserts pin the mirrored lines to the built notebook.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r2_scorepath.py
"""
import json, os, sys, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8R2_FORCEDCHOICE_UI.ipynb")))
model_cell = next("".join(c["source"]) for c in nb["cells"]
                  if "def score_trial" in "".join(c["source"]))
for line in ("m.base_model.model", "get_output_embeddings()",
             "lo = plen - 1", "torch.log_softmax",
             "use_cache=False).last_hidden_state",
             "Injector(layer_mods, TRAIN_LAYER, vec, plen - 1)",
             "PeftModel.from_pretrained(m, READOUTS[cond])",
             "merge_and_unload()"):
    assert line in model_cell, f"notebook model cell drifted: missing {line!r}"
print("notebook model cell carries the mirrored path — pinned")

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)

print("== flight load path: save_pretrained -> PeftModel.from_pretrained ==")
base = Qwen2ForCausalLM(cfg)
lora = get_peft_model(base, LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))
tmp = tempfile.mkdtemp()
lora.save_pretrained(os.path.join(tmp, "readout"))
base2 = Qwen2ForCausalLM(cfg)
base2.load_state_dict(Qwen2ForCausalLM(cfg).state_dict(), strict=True)
base2 = Qwen2ForCausalLM(cfg)
m = PeftModel.from_pretrained(base2, os.path.join(tmp, "readout"))
m.eval()
check("from_pretrained adapter attaches, nothing trainable",
      not any(p.requires_grad for p in m.parameters()))
_cm = m.base_model.model
DEC, HEAD = _cm.model, _cm.get_output_embeddings()
check("PEFT chain resolves on the from_pretrained wrapper",
      DEC is not None and HEAD is not None and hasattr(DEC, "layers"))
import re as _re
layer_mods = {}
for mod_name, mod in m.named_modules():
    mm = _re.search(r"(?:^|\.)layers\.(\d+)$", mod_name)
    if mm:
        layer_mods[int(mm.group(1))] = mod
check("layer modules resolve under the wrapper",
      len(layer_mods) == cfg.num_hidden_layers)

# Mirrored Injector (notebook lines pinned above; math identical)
class Injector:
    def __init__(self, layer_mods, hs_level, vec, start_idx):
        self.mod = layer_mods[hs_level - 1]
        self.vec = vec
        self.start = start_idx
        self.handle = None
        self.calls = 0
    def _fn(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = self.vec.to(dtype=hs.dtype, device=hs.device)
        if hs.shape[1] > 1:
            add = torch.zeros_like(hs)
            add[:, self.start:, :] = v
            hs = hs + add
        else:
            hs = hs + v
        self.calls += 1
        return (hs, *output[1:]) if isinstance(output, tuple) else hs
    def __enter__(self):
        self.handle = self.mod.register_forward_hook(self._fn)
        return self
    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

PAD = 0
EOS = 127
CANDS = {"A": [17, EOS], "B": [23, 51, EOS], "C": [88, EOS], "NONE": [99, 44, EOS]}
pid = torch.randint(1, 127, (37,))
plen = 37

def score_batched(model, inject_vec=None):
    seqs = [torch.cat([pid, torch.tensor(CANDS[n], dtype=pid.dtype)])
            for n in CANDS]
    maxlen = max(int(s.shape[0]) for s in seqs)
    ids = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    dec = model.base_model.model.model
    head = model.base_model.model.get_output_embeddings()
    calls = 0
    with torch.no_grad():
        if inject_vec is not None:
            with Injector(layer_mods, 1, inject_vec, plen - 1) as inj:
                hid = dec(input_ids=ids, attention_mask=mask,
                          use_cache=False).last_hidden_state
            calls = inj.calls
        else:
            hid = dec(input_ids=ids, attention_mask=mask,
                      use_cache=False).last_hidden_state
        lo = plen - 1
        sm = {}
        for i, name in enumerate(CANDS):
            n_ans = len(CANDS[name])
            logits = head(hid[i, lo:lo + n_ans, :]).float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(CANDS[name])
            sm[name] = float(lp[torch.arange(n_ans), tgt].mean())
    return sm, calls

print("== parity: sliced mean-logprob == -(HF masked loss) per candidate ==")
sm, _ = score_batched(m)
ok_par = True
for name, ans in CANDS.items():
    seq = torch.cat([pid, torch.tensor(ans, dtype=pid.dtype)]).unsqueeze(0)
    labels = seq.clone()
    labels[:, :plen] = -100
    with torch.no_grad():
        hf_loss = float(m(input_ids=seq, labels=labels, use_cache=False).loss)
    if abs(sm[name] + hf_loss) > 1e-4:
        ok_par = False
        print(f"    {name}: mean-lp {sm[name]:.6f} vs -HF {-hf_loss:.6f}")
check("mean-logprob parity across candidates (1e-4)", ok_par)

print("== batched == per-row single scoring (padding/mask correctness) ==")
ok_bat = True
for name, ans in CANDS.items():
    seq = torch.cat([pid, torch.tensor(ans, dtype=pid.dtype)]).unsqueeze(0)
    with torch.no_grad():
        hid1 = DEC(input_ids=seq, use_cache=False).last_hidden_state
        logits = HEAD(hid1[0, plen - 1: plen - 1 + len(ans), :]).float()
        lp = torch.log_softmax(logits, dim=-1)
        single = float(lp[torch.arange(len(ans)), torch.tensor(ans)].mean())
    if abs(single - sm[name]) > 1e-4:
        ok_bat = False
        print(f"    {name}: single {single:.6f} vs batched {sm[name]:.6f}")
check("batched scoring equals single-row scoring (1e-4)", ok_bat)

print("== injection: hook fires once, moves scores; sham untouched ==")
vec = torch.full((cfg.hidden_size,), 3.0)
smi, calls = score_batched(m, inject_vec=vec)
check("hook fired exactly once on the batched forward", calls == 1)
check("injection moves candidate scores",
      any(abs(smi[n] - sm[n]) > 1e-3 for n in CANDS))
sm2, calls2 = score_batched(m)
check("sham path: zero hook calls, scores reproduce exactly",
      calls2 == 0 and all(abs(sm2[n] - sm[n]) < 1e-9 for n in CANDS))

print("== learnability: a trained answer is detected by argmax ==")
mt = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))
mt.train()
opt = torch.optim.AdamW([p for p in mt.parameters() if p.requires_grad], lr=5e-3)
target = "B"
seq = torch.cat([pid, torch.tensor(CANDS[target], dtype=pid.dtype)]).unsqueeze(0)
labels = seq.clone()
labels[:, :plen] = -100
for step in range(40):
    opt.zero_grad()
    loss = mt(input_ids=seq, labels=labels, use_cache=False).loss
    loss.backward()
    opt.step()
mt.eval()
smt, _ = score_batched(mt)
names = [n for n in CANDS if n != "NONE"]
check(f"argmax lands on the trained answer ({target}; scores "
      f"{ {n: round(smt[n], 2) for n in names} })",
      max(names, key=lambda n: smt[n]) == target)
check("NONE scored alongside but excluded from the forced argmax",
      "NONE" in smt)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
