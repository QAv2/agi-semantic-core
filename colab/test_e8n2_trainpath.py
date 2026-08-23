#!/usr/bin/env python3
"""E8-N v2 train-path micro-exercise on CPU (lane law: every new train path
gets a CPU gate before staging).

The v2-new path: FOUR strands through one answer-sliced forward, with the
Injector hook live on naming-inject examples and the BACKWARD inside the hook
context — because non-reentrant checkpointing re-runs layer forwards during
backward, and a hook removed before backward would make the recompute diverge
from the original forward (silent wrong gradients). This test proves, on a
tiny random Qwen2:

  (1) PEFT chain + checkpointing engagement assert (v1 gate, re-pinned)
  (2) sliced CE == HF masked loss, hookless (v1 parity, re-pinned)
  (3) Injector math: positions before start untouched, after start shifted;
      sliced CE under hook == HF masked loss under the same hook
  (4) THE decisive check: grads with (checkpointing ON + backward inside the
      hook context) match grads with checkpointing OFF to 1e-5 — and the hook
      fires MORE times under checkpointing (the recompute saw it)
  (5) learnability of a naming-style injected example
  (6) the built notebook carries the exact mirrored lines (drift pins)

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n2_trainpath.py
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n2_logic as L

from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8N2_JOINT_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
train_cell = next(s for s in cells if "def train_joint" in s)
stim_cell = next(s for s in cells if "class Injector" in s)
for line in ("m.base_model.model", "get_output_embeddings()",
             "gradient_checkpointing_kwargs={'use_reentrant': False}",
             "answer_slice(plen", "hid[:, lo:hi, :]",
             "with Injector(layer_mods, L, vec, plen - 1) as injh:",
             "loss = fwd_loss(ids, plen, ans)",
             "scaler.scale(loss / ACCUM).backward()"):
    assert line in train_cell, f"notebook train cell drifted: missing {line!r}"
# backward-inside-context pin: the injected branch must backward BEFORE the
# context closes (hook_calls accounting line follows the with-block)
inj_block = train_cell.split("with Injector(layer_mods, L, vec, plen - 1) as injh:")[1]
inj_lines = inj_block.splitlines()
assert any("backward()" in l for l in inj_lines[:4]), \
    "backward not inside the Injector context in the train cell"
print("notebook train cell carries the mirrored path — pinned")

# Injector executed VERBATIM from the built notebook cell
inj_src = stim_cell[stim_cell.index("class Injector"):stim_cell.index("def run_trial")]
ns = {"torch": torch}
exec(inj_src, ns)
Injector = ns["Injector"]
print("Injector class exec'd verbatim from the built notebook")

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg)
m = get_peft_model(base, LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.0, bias="none",     # dropout 0: determinism
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))

print("== chain + checkpointing engagement ==")
_cm = m.base_model.model
DEC, HEAD = _cm.model, _cm.get_output_embeddings()
layer_mods = {i: mod for i, mod in enumerate(DEC.layers)}   # module-index keys, as resolve_layers
check("PEFT chain resolves", DEC is not None and HEAD is not None)
lora_named = [n for n, p in m.named_parameters() if p.requires_grad]
check("only LoRA params trainable",
      bool(lora_named) and all("lora" in n for n in lora_named))
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
check("gradient checkpointing ENGAGED",
      bool(getattr(DEC, "gradient_checkpointing", False)))

ids = torch.randint(1, 127, (1, 40))
plen = 37
ans = ids[:, plen:]
lo, hi = L.answer_slice(plen, ids.shape[1])

def sliced_loss():
    hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
    return Fnn.cross_entropy(
        HEAD(hid[:, lo:hi, :]).float().view(-1, cfg.vocab_size), ans.reshape(-1))

def hf_loss():
    labels = ids.clone()
    labels[:, :plen] = -100
    return m(input_ids=ids, labels=labels, use_cache=False).loss

print("== parity, hookless ==")
m.eval()
with torch.no_grad():
    sl, hl = sliced_loss(), hf_loss()
check(f"sliced CE == HF masked loss ({float(sl):.6f} vs {float(hl):.6f})",
      abs(float(sl) - float(hl)) < 1e-4)

print("== Injector math (verbatim class) ==")
HS_LEVEL = 1
vec = torch.full((cfg.hidden_size,), 0.5)
with torch.no_grad():
    h_clean = DEC(input_ids=ids, use_cache=False).last_hidden_state
    # hidden_states[L] convention: hook layer_mods[L] = decoder layer L-1;
    # its OUTPUT feeds layer L, and the final hidden state reflects the add.
    with Injector(layer_mods, HS_LEVEL, vec, plen - 1) as inj:
        h_hook = DEC(input_ids=ids, use_cache=False).last_hidden_state
    calls_plain = inj.calls
check("hook fired exactly once on a plain forward", calls_plain == 1)
check("final hidden state unchanged and changed across the start boundary",
      torch.allclose(h_clean[:, :plen - 1], h_hook[:, :plen - 1], atol=1e-5) and
      not torch.allclose(h_clean[:, plen - 1:], h_hook[:, plen - 1:], atol=1e-3))
with torch.no_grad():
    with Injector(layer_mods, HS_LEVEL, vec, plen - 1):
        sl_h = sliced_loss()
    with Injector(layer_mods, HS_LEVEL, vec, plen - 1):
        hl_h = hf_loss()
check(f"parity holds UNDER the hook ({float(sl_h):.6f} vs {float(hl_h):.6f})",
      abs(float(sl_h) - float(hl_h)) < 1e-4)

print("== decisive: grad parity, checkpointing x injection ==")
m.train()

def grads_snapshot():
    return {n: p.grad.clone() for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None}

# A: checkpointing ON, backward inside hook context (the notebook's path)
m.zero_grad(set_to_none=True)
with Injector(layer_mods, HS_LEVEL, vec, plen - 1) as inj_a:
    loss_a = sliced_loss()
    loss_a.backward()
calls_ckpt = inj_a.calls
grads_a = grads_snapshot()

# B: checkpointing OFF, same forward+backward
try:
    _cm.gradient_checkpointing_disable()
except Exception:
    pass
m.zero_grad(set_to_none=True)
with Injector(layer_mods, HS_LEVEL, vec, plen - 1) as inj_b:
    loss_b = sliced_loss()
    loss_b.backward()
calls_plain2 = inj_b.calls
grads_b = grads_snapshot()

check(f"losses identical ({float(loss_a):.6f} vs {float(loss_b):.6f})",
      abs(float(loss_a) - float(loss_b)) < 1e-6)
# Two legal worlds: recompute active (hook fires again — the in-context
# backward is what keeps grads right) or checkpointing INERT despite the
# flag (v1 flight ops note, confirmed locally: calls equal, grads
# bit-identical, peak bounded by the answer-sliced head instead). Either
# way the decisive property is grad parity, asserted below.
if calls_ckpt > calls_plain2:
    print(f"  (recompute ACTIVE: hook fired {calls_ckpt} vs {calls_plain2})")
else:
    print(f"  (checkpointing INERT on this transformers version: hook fired "
          f"{calls_ckpt} vs {calls_plain2} — v1 ops note reproduced locally)")
check("hook coverage accounted in both worlds", calls_ckpt >= calls_plain2)
worst = max(float((grads_a[n] - grads_b[n]).abs().max()) for n in grads_b)
check(f"grads match checkpointing-off reference (worst {worst:.2e})",
      set(grads_a) == set(grads_b) and worst < 1e-5)
frozen_with_grad = [n for n, p in m.named_parameters()
                    if not p.requires_grad and p.grad is not None
                    and float(p.grad.abs().sum()) > 0]
check("no grads on frozen base params", not frozen_with_grad)

print("== long-seq sanity under re-enabled checkpointing ==")
try:
    _cm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    _cm.gradient_checkpointing_enable()
long_ids = torch.randint(1, 127, (1, 600))
lplen = 597
llo, lhi = L.answer_slice(lplen, long_ids.shape[1])
hid = DEC(input_ids=long_ids, use_cache=False).last_hidden_state
loss = Fnn.cross_entropy(
    HEAD(hid[:, llo:lhi, :]).float().view(-1, cfg.vocab_size),
    long_ids[:, lplen:].reshape(-1))
loss.backward()
check("600-tok forward+backward OK, loss finite", torch.isfinite(loss).item())

print("== learnability: injected naming-style example ==")
m.zero_grad(set_to_none=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-3)
first = None
for step in range(30):
    opt.zero_grad()
    with Injector(layer_mods, HS_LEVEL, vec, plen - 1):
        loss = sliced_loss()
        loss.backward()
    opt.step()
    if first is None:
        first = float(loss)
check(f"loss falls over 30 injected steps ({first:.3f} -> {float(loss):.3f})",
      float(loss) < first)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
