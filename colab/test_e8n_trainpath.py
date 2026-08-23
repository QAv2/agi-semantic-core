#!/usr/bin/env python3
"""E8-N v2 train-path micro-exercise on CPU (smoke-1 lesson: the training
forward had never executed anywhere before the VM — this closes that gap).

On a tiny random Qwen2, mirror the notebook's exact v2 path — PEFT attribute
chain, non-reentrant gradient-checkpointing enable + engagement ASSERT,
decoder forward, answer_slice, head-on-slice, manual CE, backward — and
verify: (1) sliced CE == HF full-forward masked loss (parity with v1
semantics), (2) grads land on LoRA params only, (3) a long sequence trains
under checkpointing, (4) the toy example is learnable. String-asserts pin
the mirrored lines to the built notebook so drift is detected.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n_trainpath.py
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n_logic as L

from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8N_NATURAL_UI.ipynb")))
train_cell = next("".join(c["source"]) for c in nb["cells"]
                  if "def train_readout" in "".join(c["source"]))
for line in ("m.base_model.model", "get_output_embeddings()",
             "gradient_checkpointing_kwargs={'use_reentrant': False}",
             "answer_slice(plen", "hid[:, lo:hi, :]"):
    assert line in train_cell, f"notebook train cell drifted: missing {line!r}"
print("notebook train cell carries the mirrored path — pinned")

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg)
m = get_peft_model(base, LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))

print("== attribute chain + checkpointing engagement (the smoke-1 killer) ==")
_cm = m.base_model.model
DEC, HEAD = _cm.model, _cm.get_output_embeddings()
check("PEFT chain resolves: CausalLM under wrapper, decoder, head",
      DEC is not None and HEAD is not None and hasattr(DEC, "layers"))
lora_named = [n for n, p in m.named_parameters() if p.requires_grad]
check("only LoRA params trainable", lora_named and all("lora" in n for n in lora_named))
m.train()
try:
    _cm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    _cm.gradient_checkpointing_enable()
try:
    _cm.enable_input_require_grads()
except Exception as e:
    print("  (input-require-grads unavailable:", e, ")")
check("gradient checkpointing ENGAGED (flag set on decoder)",
      bool(getattr(DEC, "gradient_checkpointing", False)))

print("== parity: sliced-head CE == HF full-forward masked loss ==")
m.eval()   # deterministic (no lora dropout) for the parity comparison
ids = torch.randint(1, 127, (1, 40))
plen = 37
ans = ids[:, plen:]
lo, hi = L.answer_slice(plen, ids.shape[1])
with torch.no_grad():
    hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
    sliced_loss = Fnn.cross_entropy(
        HEAD(hid[:, lo:hi, :]).float().view(-1, cfg.vocab_size), ans.reshape(-1))
    labels = ids.clone()
    labels[:, :plen] = -100
    hf_loss = m(input_ids=ids, labels=labels, use_cache=False).loss
check(f"losses match ({float(sliced_loss):.6f} vs {float(hf_loss):.6f})",
      abs(float(sliced_loss) - float(hf_loss)) < 1e-4)

print("== backward: grads land on LoRA only, long seq trains checkpointed ==")
m.train()
long_ids = torch.randint(1, 127, (1, 600))
lplen = 597
llo, lhi = L.answer_slice(lplen, long_ids.shape[1])
hid = DEC(input_ids=long_ids, use_cache=False).last_hidden_state
loss = Fnn.cross_entropy(
    HEAD(hid[:, llo:lhi, :]).float().view(-1, cfg.vocab_size),
    long_ids[:, lplen:].reshape(-1))
loss.backward()
gnorm = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None)
frozen_with_grad = [n for n, p in m.named_parameters()
                    if not p.requires_grad and p.grad is not None
                    and float(p.grad.abs().sum()) > 0]
check("long-seq (600 tok) forward+backward OK under checkpointing, loss finite",
      torch.isfinite(loss).item())
check("nonzero grads reached the LoRA params", gnorm > 0)
check("no grads on frozen base params", not frozen_with_grad)

print("== learnability: fixed toy example, loss falls ==")
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-3)
first = None
for step in range(30):
    opt.zero_grad()
    hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
    loss = Fnn.cross_entropy(
        HEAD(hid[:, lo:hi, :]).float().view(-1, cfg.vocab_size), ans.reshape(-1))
    loss.backward()
    opt.step()
    if first is None:
        first = float(loss)
check(f"loss falls over 30 steps ({first:.3f} -> {float(loss):.3f})",
      float(loss) < first)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
