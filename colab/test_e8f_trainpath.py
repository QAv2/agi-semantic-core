#!/usr/bin/env python3
"""E8-F train/score-path micro-exercise on CPU (lane law: every train path
gets a CPU micro-exercise before staging; the first long forward must not
happen on the VM).

On a tiny random Qwen2, mirror the notebook's exact paths and verify:
(1) long-answer sliced-CE == HF full-forward masked loss, hookless AND under
a pinned-stimulus injection hook (the E8-F stimulus is a precomputed unit
vector — no on-VM composition); (2) hook fires once per forward and moves
the loss; (3) grad parity checkpointing-on vs -off with the backward INSIDE
the Injector context (E8-N v2 law), LoRA-only grads; (4) learnability +
GENERATION round trip: a micro-train on injected code-length answers drops
loss and greedy decode emits the trained answer for the trained stimulus
(E8-F eval is generation, not forced choice); (5) the eval-resume reload
path (save_pretrained -> fresh base -> PeftModel.from_pretrained) preserves
logits exactly. String-asserts pin the mirrored lines to the built notebook.

Run:  python3 colab/test_e8f_trainpath.py
"""
import copy, json, os, sys, tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
train_cell = next(c for c in cells if "def train_readout_feat" in c)
for line in ("def encode_featex", "def build_eval_model",
             "vec = ex['alpha'] * mu14 * torch.tensor(STIM[skey])",
             "with Injector(layer_mods, ex['layer'], vec, plen - 1) as injh:",
             "scaler.scale(loss / ACCUM).backward()",
             "assert getattr(DEC, 'gradient_checkpointing', False)",
             "per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL)",
             "answer_slice(plen, ids.shape[1])",
             "hid[:, lo:hi, :]",
             "PeftModel.from_pretrained(m, str(readout_dir))"):
    assert line in train_cell, f"train cell drifted: missing {line!r}"
run_cell = next(c for c in cells if "def run_trial_feat" in c)
for line in ("max_new_tokens=110, do_sample=False",
             "vec = trial['alpha'] * mu14 * torch.tensor(STIM[trial['skey']])",
             "Injector(layer_mods, trial['layer'], vec, ids.shape[1] - 1)",
             "assert calls >= 1"):
    assert line in run_cell, f"run cell drifted: missing {line!r}"
print("notebook train + run cells carry the mirrored paths — pinned")

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg)
BASE_SD = copy.deepcopy(base.state_dict())
m = get_peft_model(base, LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))
for p in m.parameters():
    if p.requires_grad:
        p.data = p.data.float()
_cm = m.base_model.model
DEC, HEAD = _cm.model, _cm.get_output_embeddings()

import re as _re
layer_mods = {}
for mod_name, mod in m.named_modules():
    mm = _re.search(r"(?:^|\.)layers\.(\d+)$", mod_name)
    if mm:
        layer_mods[int(mm.group(1))] = mod
check("layer modules resolve under the PEFT wrapper",
      len(layer_mods) == cfg.num_hidden_layers)


class Injector:  # mirrored; pinned to the notebook by the logic suite markers
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


rng = np.random.default_rng(20260885)
# pinned-stimulus analog: precomputed unit vectors (E8-F ships them minted)
STIM = {}
for k in ("full:X", "atom:x:+", "true:Y"):
    v = rng.standard_normal(cfg.hidden_size)
    STIM[k] = v / np.linalg.norm(v)
MU, EOS = 4.0, 127
prompt = torch.randint(1, 127, (37,))
PLEN = 37
# code-length answers (12-14 tokens, the E8-F regime; distinct per stimulus)
ANS = {"full:X": [17, 44, 9, 61, 88, 12, 23, 51, EOS],
       "atom:x:+": [23, 51, 9, 61, 12, 88, 44, 17, 30, EOS],
       "true:Y": [88, 12, 61, 9, 51, 23, 17, 44, EOS]}


def mirrored_fwd_loss(ids, plen, ans):
    lo, hi = plen - 1, ids.shape[1] - 1
    hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
    logits = HEAD(hid[:, lo:hi, :]).float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))


def example(skey):
    ans = torch.tensor(ANS[skey])
    ids = torch.cat([prompt, ans]).unsqueeze(0)
    return ids, PLEN, ans.unsqueeze(0)


def hf_masked_loss(ids, plen):
    labels = ids.clone()
    labels[:, :plen] = -100
    with torch.no_grad():
        return float(m(input_ids=ids, labels=labels, use_cache=False).loss)


print("== (1) code-length sliced-CE == HF masked loss ==")
m.eval()
ok = True
for skey in ANS:
    ids, plen, ans = example(skey)
    with torch.no_grad():
        sl = float(mirrored_fwd_loss(ids, plen, ans))
    hf = hf_masked_loss(ids, plen)
    if abs(sl - hf) > 1e-4:
        ok = False
        print(f"   mismatch {skey}: sliced {sl:.6f} vs HF {hf:.6f}")
check("hookless parity to 1e-4 on all code-length answers", ok)

vec = torch.tensor(1.0 * MU * STIM["full:X"])
ids, plen, ans = example("full:X")
with torch.no_grad():
    with Injector(layer_mods, 1, vec, plen - 1) as inj1:
        sl_h = float(mirrored_fwd_loss(ids, plen, ans))
    labels = ids.clone()
    labels[:, :plen] = -100
    with Injector(layer_mods, 1, vec, plen - 1) as inj2:
        hf_h = float(m(input_ids=ids, labels=labels, use_cache=False).loss)
check("parity holds UNDER the pinned-stimulus hook", abs(sl_h - hf_h) < 1e-4)
check("hook fires exactly once per forward", inj1.calls == 1 and inj2.calls == 1)
with torch.no_grad():
    sl_nohook = float(mirrored_fwd_loss(ids, plen, ans))
check("injection moves the loss (vector lands)", abs(sl_h - sl_nohook) > 1e-6)

print("== (2) grad parity: checkpointing on vs off, backward in-context ==")
def grads_snapshot(use_ckpt):
    m.train()
    if use_ckpt:
        try:
            _cm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            _cm.gradient_checkpointing_enable()
        try:
            _cm.enable_input_require_grads()
        except Exception:
            pass
        assert getattr(DEC, "gradient_checkpointing", False)
    else:
        try:
            _cm.gradient_checkpointing_disable()
        except Exception:
            pass
    m.zero_grad(set_to_none=True)
    ids_, plen_, ans_ = example("full:X")
    with Injector(layer_mods, 1, vec, plen_ - 1) as inj:
        loss = mirrored_fwd_loss(ids_, plen_, ans_)
        loss.backward()
    assert inj.calls >= 1
    out = {n: p.grad.detach().clone() for n, p in m.named_parameters()
           if p.requires_grad and p.grad is not None}
    m.zero_grad(set_to_none=True)
    return out

g_off = grads_snapshot(False)
g_on = grads_snapshot(True)
check("LoRA grads exist both ways", len(g_off) > 0 and set(g_off) == set(g_on))
worst = max(float((g_off[n] - g_on[n]).abs().max()) for n in g_off)
check(f"grad parity ckpt-on == ckpt-off (worst {worst:.2e})", worst < 1e-5)
nonzero = sum(1 for n in g_on if float(g_on[n].abs().max()) > 0)
check("gradients are nonzero through the checkpointed hook path", nonzero > 0)

print("== (3) micro-train -> generation round trip ==")
# The flown bar (test_e8n/e8r3c_trainpath): a fixed toy example, loss FALLS
# through the mirrored loop, and the trained answer comes back out — here via
# GREEDY GENERATION, E8-F's eval mode. Conditional stimulus-discrimination is
# a scale capability already proven in flight (E8-R 18/18 at 1.5B); a 2-layer
# random model is not asked to show it.
try:
    _cm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    _cm.gradient_checkpointing_enable()
try:
    _cm.enable_input_require_grads()
except Exception:
    pass
m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
SKEY = "full:X"
ids_, plen_, ans_ = example(SKEY)
v_tr = torch.tensor(1.0 * MU * STIM[SKEY])
lvs = []
for step in range(150):
    with Injector(layer_mods, 1, v_tr, plen_ - 1):
        loss = mirrored_fwd_loss(ids_, plen_, ans_)
        loss.backward()
    opt.step(); opt.zero_grad()
    lvs.append(float(loss.detach()))
first, last = lvs[0], float(np.mean(lvs[-10:]))
try:
    _cm.gradient_checkpointing_disable()
except Exception:
    pass
m.eval()
check(f"loss falls through the mirrored checkpointed loop "
      f"({first:.2f} -> {last:.2f})", last < first * 0.95)  # flown bar (e8n/e8r3c)
with torch.no_grad():
    with Injector(layer_mods, 1, v_tr, PLEN - 1) as inj:
        out = m.generate(input_ids=prompt.unsqueeze(0),
                         max_new_tokens=len(ANS[SKEY]) + 2,
                         do_sample=False, pad_token_id=0, use_cache=True)
got = out[0][PLEN:].tolist()
check("greedy generation emits the trained answer (train -> gen round trip)",
      got[:len(ANS[SKEY]) - 1] == ANS[SKEY][:-1] and inj.calls >= 1)

print("== (4) eval-resume reload path preserves logits ==")
with tempfile.TemporaryDirectory() as td:
    m.save_pretrained(td)
    base2 = Qwen2ForCausalLM(cfg)
    base2.load_state_dict(BASE_SD)
    m2 = PeftModel.from_pretrained(base2, td)
    m2.eval()
    ids_, _, _ = example("full:X")
    with torch.no_grad():
        l1 = m(input_ids=ids_).logits
        l2 = m2(input_ids=ids_).logits
    worst2 = float((l1 - l2).abs().max())
    check(f"save -> fresh base -> from_pretrained logits equal "
          f"(worst {worst2:.2e})", worst2 < 1e-4)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
