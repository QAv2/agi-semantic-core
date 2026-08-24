#!/usr/bin/env python3
"""E8-R3-c train/score-path micro-exercise on CPU (lane law: every train path
gets a CPU micro-exercise before staging; smoke-1 lesson — the first long
forward must not happen on the VM).

On a tiny random Qwen2, mirror the notebook's exact paths and verify:
(1) pair-answer sliced-CE == HF full-forward masked loss (multi-token
answers), hookless AND under a composed-direction injection hook;
(2) composed-vec math: inject_vec = alpha*mu*normalize(d_a+d_b), hook fires
exactly once per forward; (3) grad parity checkpointing-on vs -off with the
backward INSIDE the Injector context (E8-N v2 law; the inert-checkpointing
finding reproduced locally), LoRA-only grads; (4) learnability: a micro-train
on composed injections with multi-token pair answers drops loss and the
trained answers are detected by the batched scoring-path argmax (train ->
score round trip); (5) batched scoring == single-row scoring with
heterogeneous candidate lengths (padding proof). String-asserts pin the
mirrored lines to the built notebook.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r3c_trainpath.py
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
train_cell = next(c for c in cells if "def train_readout_pairs" in c)
for line in ("def encode_pairex", "def inject_vec",
             "pair_dvec(dirs[L], *what)",
             "with Injector(layer_mods, inj[1], vec, plen - 1) as injh:",
             "scaler.scale(loss / ACCUM).backward()",
             "assert getattr(DEC, 'gradient_checkpointing', False)",
             "per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL)",
             "answer_slice(plen, ids.shape[1])",
             "hid[:, lo:hi, :]"):
    assert line in train_cell, f"train cell drifted: missing {line!r}"
fc_cell = next(c for c in cells if "def score_pair_trial" in c)
for line in ("lo = plen - 1", "torch.log_softmax", "PAIR_CAND_IDS[name]",
             "use_cache=False).last_hidden_state",
             "Injector(layer_mods, TRAIN_LAYER, vec, plen - 1)"):
    assert line in fc_cell, f"FC cell drifted: missing {line!r}"
print("notebook train + FC cells carry the mirrored paths — pinned")

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024,
                  tie_word_embeddings=True)
base = Qwen2ForCausalLM(cfg)
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


# Mirrored Injector (pinned to the stimulus cell by the logic suite)
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


def pair_dvec_np(dirs_L, a, b):
    v = np.asarray(dirs_L[a], float) + np.asarray(dirs_L[b], float)
    return v / np.linalg.norm(v)


rng = np.random.default_rng(20260825)
ATOMS = {}
for name in ("A", "B", "C"):
    v = rng.standard_normal(cfg.hidden_size)
    ATOMS[name] = v / np.linalg.norm(v)
MU, ALPHA = 4.0, 1.0
PAD, EOS = 0, 127
prompt = torch.randint(1, 127, (31,))
PLEN = 31
PAIR_ANS = {("A", "B"): [17, 44, EOS], ("A", "C"): [23, 51, 9, EOS],
            ("B", "C"): [88, 12, EOS]}


def mirrored_fwd_loss(ids, plen, ans):
    lo, hi = plen - 1, ids.shape[1] - 1
    hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
    logits = HEAD(hid[:, lo:hi, :]).float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))


def example(pair):
    ans = torch.tensor(PAIR_ANS[pair])
    ids = torch.cat([prompt, ans]).unsqueeze(0)
    return ids, PLEN, ans.unsqueeze(0)


def hf_masked_loss(ids, plen):
    labels = ids.clone()
    labels[:, :plen] = -100
    with torch.no_grad():
        return float(m(input_ids=ids, labels=labels, use_cache=False).loss)


print("== (1) pair-answer sliced-CE == HF masked loss ==")
m.eval()
ok = True
for pair in PAIR_ANS:
    ids, plen, ans = example(pair)
    with torch.no_grad():
        sl = float(mirrored_fwd_loss(ids, plen, ans))
    hf = hf_masked_loss(ids, plen)
    if abs(sl - hf) > 1e-4:
        ok = False
        print(f"   mismatch {pair}: sliced {sl:.6f} vs HF {hf:.6f}")
check("hookless parity to 1e-4 on all multi-token answers", ok)

vecAB = torch.tensor(ALPHA * MU * pair_dvec_np(ATOMS, "A", "B"))
ids, plen, ans = example(("A", "B"))
with torch.no_grad():
    with Injector(layer_mods, 1, vecAB, plen - 1) as inj1:
        sl_h = float(mirrored_fwd_loss(ids, plen, ans))
    labels = ids.clone()
    labels[:, :plen] = -100
    with Injector(layer_mods, 1, vecAB, plen - 1) as inj2:
        hf_h = float(m(input_ids=ids, labels=labels, use_cache=False).loss)
check("parity holds UNDER the composed-injection hook",
      abs(sl_h - hf_h) < 1e-4)
check("hook fires exactly once per forward",
      inj1.calls == 1 and inj2.calls == 1)
with torch.no_grad():
    sl_nohook = float(mirrored_fwd_loss(ids, plen, ans))
check("composed injection moves the loss (vector lands)",
      abs(sl_h - sl_nohook) > 1e-6)

print("== (2) composed-vec math ==")
dv = pair_dvec_np(ATOMS, "A", "B")
check("unit norm + symmetry",
      abs(np.linalg.norm(dv) - 1) < 1e-12
      and np.allclose(dv, pair_dvec_np(ATOMS, "B", "A")))
check("midpoint geometry: equal angles to both atoms",
      abs(float(np.dot(dv, ATOMS["A"])) - float(np.dot(dv, ATOMS["B"]))) < 1e-12)

print("== (3) grad parity: checkpointing on vs off, backward in-context ==")
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
    ids, plen, ans = example(("A", "B"))
    with Injector(layer_mods, 1, vecAB, plen - 1) as inj:
        loss = mirrored_fwd_loss(ids, plen, ans)
        loss.backward()
    g = {n: p.grad.detach().clone() for n, p in m.named_parameters()
         if p.grad is not None}
    return float(loss.detach()), g, inj.calls

l_on, g_on, c_on = grads_snapshot(True)
l_off, g_off, c_off = grads_snapshot(False)
check("loss identical on/off", abs(l_on - l_off) < 1e-7)
check("grads bit-close on/off (inert-checkpointing finding reproduced; "
      "in-context backward correct either way)",
      set(g_on) == set(g_off)
      and all(torch.allclose(g_on[k], g_off[k], atol=1e-6) for k in g_on))
check("hook fired once per training forward (recompute would double it)",
      c_on == 1 and c_off == 1)
check("LoRA-only grads", all("lora" in k for k in g_on))
m.zero_grad(set_to_none=True)

print("== (4) learnability: composed injection -> multi-token answer ==")
# The flown bar (test_e8n_trainpath): a fixed toy example, loss FALLS through
# the mirrored loop. Conditional pair-discrimination is a scale capability
# already proven in flight (E8-R: 144 injected examples -> 18/18 exact at
# 1.5B); a 2-layer random model is not asked to show it.
try:
    _cm.gradient_checkpointing_disable()
except Exception:
    pass
m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
TRAINED_PAIR = ("A", "B")
ids, plen, ans = example(TRAINED_PAIR)
vec_tr = torch.tensor(ALPHA * MU * pair_dvec_np(ATOMS, *TRAINED_PAIR))
first, last = None, None
for step in range(60):
    with Injector(layer_mods, 1, vec_tr, plen - 1):
        loss = mirrored_fwd_loss(ids, plen, ans)
        loss.backward()
    opt.step()
    opt.zero_grad()
    lv = float(loss.detach())
    if first is None:
        first = lv
    last = lv
m.eval()
check(f"loss falls through the mirrored loop ({first:.3f} -> {last:.3f})",
      last < 0.95 * first)


def score_pairs(vec=None):
    seqs = [torch.cat([prompt, torch.tensor(PAIR_ANS[p], dtype=prompt.dtype)])
            for p in PAIR_ANS]
    maxlen = max(int(s.shape[0]) for s in seqs)
    ids = torch.full((len(seqs), maxlen), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    with torch.no_grad():
        if vec is not None:
            with Injector(layer_mods, 1, vec, PLEN - 1) as inj:
                hid = DEC(input_ids=ids, attention_mask=mask,
                          use_cache=False).last_hidden_state
            assert inj.calls == 1
        else:
            hid = DEC(input_ids=ids, attention_mask=mask,
                      use_cache=False).last_hidden_state
        lo = PLEN - 1
        out = {}
        for i, p in enumerate(PAIR_ANS):
            n_ans = len(PAIR_ANS[p])
            logits = HEAD(hid[i, lo:lo + n_ans, :]).float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(PAIR_ANS[p])
            out[p] = float(lp[torch.arange(n_ans), tgt].mean())
    return out


scores = score_pairs(vec_tr)
check("train -> score round trip: trained answer argmax under its own "
      "composed injection", max(scores, key=scores.get) == TRAINED_PAIR)

print("== (5) batched == single-row scoring (heterogeneous lengths) ==")
vec = torch.tensor(ALPHA * MU * pair_dvec_np(ATOMS, "A", "B"))
batched = score_pairs(vec)
ok = True
for p in PAIR_ANS:
    seq = torch.cat([prompt, torch.tensor(PAIR_ANS[p], dtype=prompt.dtype)]
                    ).unsqueeze(0)
    with torch.no_grad():
        with Injector(layer_mods, 1, vec, PLEN - 1):
            hid = DEC(input_ids=seq, use_cache=False).last_hidden_state
        lo = PLEN - 1
        n_ans = len(PAIR_ANS[p])
        logits = HEAD(hid[0, lo:lo + n_ans, :]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(PAIR_ANS[p])
        single = float(lp[torch.arange(n_ans), tgt].mean())
    if abs(single - batched[p]) > 1e-5:
        ok = False
        print(f"   mismatch {p}: single {single:.6f} vs batched {batched[p]:.6f}")
check("padding/mask correctness across 3/4-token candidates", ok)
s1 = score_pairs(vec)
s2 = score_pairs(vec)
check("determinism", all(abs(s1[p] - s2[p]) < 1e-9 for p in s1))

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{'='*60}\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
