#!/usr/bin/env python3
"""E8-O2 train-path micro-exercise on CPU (lane law). The NEW path this
flight adds over E8-F's proven loop is CONTINUE-TRAINING: load a saved
adapter with is_trainable=True and train it further. Verify on a tiny
Qwen2: (1) the loaded adapter has trainable params and grads FLOW through
the Injector-context backward; (2) continued training moves loss on a NEW
target while the previously trained answer still generates (replay-free
short horizon); (3) the float32 cast of trainable params after loading
(the notebook's exact line) leaves eval outputs unchanged; (4) save ->
reload round trip stays logit-exact. String-pins tie the mirrored lines
to the built notebook.

Run:  python3 colab/test_e8o2_trainpath.py
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

nb = json.load(open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
mc = next(c for c in cells if "def train_readout_mixed" in c)
fl = next(c for c in cells if "def fly" in c)
for line in ("scaler.scale(loss / ACCUM).backward()",
             "assert getattr(DEC, 'gradient_checkpointing', False)",
             "assert params, 'no trainable params — is_trainable load failed'",
             "per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL)",
             "hid[:, lo:hi, :]"):
    assert line in mc, f"model cfg cell drifted: {line!r}"
for line in ("PeftModel.from_pretrained(m0, str(READOUT_DIR), is_trainable=True)",
             "p.data = p.data.float()",
             "validate_no_eval_in_train_o2(examples)",
             "ship(OUT / 'readout_mixed'"):
    assert line in fl, f"flight cell drifted: {line!r}"
print("notebook cells carry the mirrored lines — pinned")

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

import re as _re
layer_mods = {}
for mod_name, mod in m.named_modules():
    mm = _re.search(r"(?:^|\.)layers\.(\d+)$", mod_name)
    if mm:
        layer_mods[int(mm.group(1))] = mod


class Injector:  # mirrored
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


rng = np.random.default_rng(20260910)
STIM = {}
for k in ("old", "new"):
    v = rng.standard_normal(cfg.hidden_size)
    STIM[k] = v / np.linalg.norm(v)
MU, EOS = 4.0, 127
prompt = torch.randint(1, 127, (31,))
PLEN = 31
ANS = {"old": [17, 44, 9, 61, 88, 12, EOS], "new": [23, 51, 30, 5, 77, 41, EOS]}
DEC0 = m.base_model.model.model
HEAD0 = m.base_model.model.get_output_embeddings()


def fwd_loss(mm, skey):
    dec = mm.base_model.model.model
    head = mm.base_model.model.get_output_embeddings()
    ans = torch.tensor(ANS[skey])
    ids = torch.cat([prompt, ans]).unsqueeze(0)
    lo, hi = PLEN - 1, ids.shape[1] - 1
    hid = dec(input_ids=ids, use_cache=False).last_hidden_state
    logits = head(hid[:, lo:hi, :]).float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                           ans.unsqueeze(0).view(-1)), ids


def train_on(mm, skey, steps, lr=1e-2):
    mm.train()
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad],
                            lr=lr)
    first = last = None
    for _ in range(steps):
        v = torch.tensor(1.0 * MU * STIM[skey])
        with Injector(layer_mods_for(mm), 1, v, PLEN - 1):
            loss, _ = fwd_loss(mm, skey)
            loss.backward()
        opt.step(); opt.zero_grad()
        lv = float(loss.detach())
        first = lv if first is None else first
        last = lv
    mm.eval()
    return first, last


def layer_mods_for(mm):
    out = {}
    for mod_name, mod in mm.named_modules():
        r = _re.search(r"(?:^|\.)layers\.(\d+)$", mod_name)
        if r:
            out[int(r.group(1))] = mod
    return out


def gen(mm, skey, n):
    v = torch.tensor(1.0 * MU * STIM[skey])
    with torch.no_grad():
        with Injector(layer_mods_for(mm), 1, v, PLEN - 1):
            out = mm.generate(input_ids=prompt.unsqueeze(0), max_new_tokens=n,
                              do_sample=False, pad_token_id=0, use_cache=True)
    return out[0][PLEN:].tolist()


print("== phase 1: initial training on 'old' ==")
f1, l1 = train_on(m, "old", 150)
check(f"phase-1 loss falls ({f1:.2f} -> {l1:.2f})", l1 < f1 * 0.95)
check("phase-1 answer generates", gen(m, "old", 6) == ANS["old"][:6])

with tempfile.TemporaryDirectory() as td:
    m.save_pretrained(td)
    base2 = Qwen2ForCausalLM(cfg)
    base2.load_state_dict(BASE_SD)
    m2 = PeftModel.from_pretrained(base2, td, is_trainable=True)
    params2 = [p for p in m2.parameters() if p.requires_grad]
    check("continue-load: trainable params exist", len(params2) > 0)
    for p in params2:
        p.data = p.data.float()          # the notebook's exact cast
    m2.eval()
    ids_ = torch.cat([prompt, torch.tensor(ANS["old"])]).unsqueeze(0)
    with torch.no_grad():
        la = m(input_ids=ids_).logits
        lb = m2(input_ids=ids_).logits
    check(f"float32 cast + reload leave logits equal "
          f"(worst {float((la-lb).abs().max()):.2e})",
          float((la - lb).abs().max()) < 1e-4)
    # grads flow through the loaded adapter under the hook
    m2.train()
    v = torch.tensor(1.0 * MU * STIM["new"])
    with Injector(layer_mods_for(m2), 1, v, PLEN - 1) as inj:
        loss, _ = fwd_loss(m2, "new")
        loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in params2
                if p.grad is not None)
    check("grads flow through the loaded adapter in-context",
          inj.calls >= 1 and gnorm > 0)
    m2.zero_grad(set_to_none=True)
    print("== phase 2: CONTINUE-training on 'new' ==")
    f2, l2 = train_on(m2, "new", 150)
    check(f"phase-2 loss falls on the new target ({f2:.2f} -> {l2:.2f})",
          l2 < f2 * 0.95)
    check("new answer generates after continue-training",
          gen(m2, "new", 6) == ANS["new"][:6])

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
