#!/usr/bin/env python3
"""E7b-Q capture suite — the flight cell's model mechanics exec'd VERBATIM
on a tiny CPU Qwen2: s_pre position, s_gen pooling, eos/cap handling,
entropy, determinism, pooled_reps/compute_dirs, history growth.

Run:  ~/venvs/semcore/bin/python3 colab/test_e7bq_capture.py   (needs torch)
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e7bq_logic as L

NB = json.load(open(os.path.join(HERE, "E7BQ_WALK_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
FLIGHT_CELL = next(c for c in CELLS if c.startswith("# ── Flight"))
N_PASS = 0


def check(label, ok):
    global N_PASS
    assert ok, f"FAIL: {label}"
    N_PASS += 1
    print(f"  ok {N_PASS:2d}  {label}")


# drift teeth: the flight cell must carry the pinned load-bearing lines
for frag in ("def run_turn(m, tok, msgs, seed, cap, layers):",
             "return_dict=True, return_tensors='pt'",
             "pre.hidden_states[L][0, -1]",
             "post.hidden_states[L][0, ids.shape[1]:].mean(0)",
             "torch.manual_seed(seed)",
             "cent = pooled_reps(m, CENT_NAMES, L).mean(0)",
             "t = run_turn(m, tok, msgs, r['seed'], MC['cap'], layers)"):
    assert frag in FLIGHT_CELL, f"flight cell drifted: missing {frag!r}"
print("flight cell carries the pinned capture sequence")

# ── tiny model + stub tokenizer ─────────────────────────────────────────────
torch.manual_seed(0)
VOCAB, PAD, EOS = 128, 0, 127
cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=2048,
                  tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg)
model.eval()


class StubTok:
    pad_token_id = PAD
    eos_token_id = EOS

    def _ids(self, text, max_length=10 ** 6):
        return [1 + (ord(ch) % (VOCAB - 3)) for ch in text[:max_length]]

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            return_dict=True, return_tensors="pt"):
        assert return_dict, "lane law: return_dict=True always"
        ids = []
        for m in msgs:
            ids += [2 if m["role"] == "user" else 3]
            ids += self._ids(m["content"], 64)
        if add_generation_prompt:
            ids += [4]
        t = torch.tensor([ids], dtype=torch.long)
        return BatchEncoding({"input_ids": t,
                              "attention_mask": torch.ones_like(t)},
                             tensor_type=None)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + int(i) % 26) for i in ids)

    def __call__(self, texts, padding=True, truncation=True, max_length=64,
                 return_tensors="pt", add_special_tokens=True):
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
names_pool = [f"N{i:03d}" for i in range(80)]
DESC = {n: f"description of concept {n} " + "x" * (i % 7)
        for i, n in enumerate(names_pool)}
CENT_NAMES = names_pool[40:72]

ns = {"torch": torch, "np": np, "_np": np, "tok": tok, "DEV": "cpu",
      "DESC": DESC, "CENT_NAMES": CENT_NAMES,
      "GEN_TEMPERATURE": L.GEN_TEMPERATURE, "GEN_TOP_P": L.GEN_TOP_P,
      "GEN_TOP_K": L.GEN_TOP_K, "print": lambda *a, **k: None}
src = FLIGHT_CELL
exec(src[src.index("def pooled_reps"):src.index("def compute_dirs")], ns)
exec(src[src.index("def compute_dirs"):src.index("PROBE_KEY =")], ns)
exec(src[src.index("def step_entropy"):src.index("def run_turn")], ns)
exec(src[src.index("def run_turn"):src.index("def fly_condition")], ns)
pooled_reps, compute_dirs = ns["pooled_reps"], ns["compute_dirs"]
step_entropy, run_turn = ns["step_entropy"], ns["run_turn"]

print("== run_turn: capture correctness ==")
LAYERS = [1, 2]
msgs1 = [{"role": "user", "content": "What are you?"}]
t1 = run_turn(model, tok, msgs1, seed=101, cap=24, layers=LAYERS)
enc = tok.apply_chat_template(msgs1)
with torch.no_grad():
    ref = model(input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True)
for Ly in LAYERS:
    manual = ref.hidden_states[Ly][0, -1].float().numpy()
    check(f"s_pre L{Ly} == manual last-template-position hidden",
          np.array_equal(t1["s_pre"][Ly], manual))
check("ents length == kept tokens", len(t1["ents"]) == t1["n_new"])
check("eos/cap bookkeeping consistent",
      (t1["eos_hit"] != t1["cap_hit"]) or (not t1["eos_hit"]
                                           and not t1["cap_hit"]))
if t1["n_new"]:
    ids_len = enc["input_ids"].shape[1]
    # reconstruct kept ids by re-running the seeded generation
    torch.manual_seed(101)
    with torch.no_grad():
        gen = model.generate(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"],
                             do_sample=True, temperature=L.GEN_TEMPERATURE,
                             top_p=L.GEN_TOP_P, top_k=L.GEN_TOP_K,
                             max_new_tokens=24, pad_token_id=EOS,
                             return_dict_in_generate=True,
                             output_scores=True)
    new_ids = gen.sequences[0, ids_len:]
    eos_pos = (new_ids == EOS).nonzero()
    keep = new_ids[:int(eos_pos[0])] if len(eos_pos) else new_ids
    check("decoded text matches seeded regeneration",
          tok.decode(keep) == t1["text"])
    full = torch.cat([enc["input_ids"], keep.unsqueeze(0)], dim=1)
    with torch.no_grad():
        post = model(input_ids=full, attention_mask=torch.ones_like(full),
                     output_hidden_states=True)
    for Ly in LAYERS:
        manual = post.hidden_states[Ly][0, ids_len:].mean(0).float().numpy()
        check(f"s_gen L{Ly} == manual mean over reply positions",
              np.allclose(t1["s_gen"][Ly], manual, atol=1e-6))

print("== determinism ==")
t1b = run_turn(model, tok, msgs1, seed=101, cap=24, layers=LAYERS)
check("same seed -> identical text/states",
      t1b["text"] == t1["text"]
      and np.array_equal(t1b["s_pre"][1], t1["s_pre"][1]))
t1c = run_turn(model, tok, msgs1, seed=102, cap=24, layers=LAYERS)
check("different seed -> different sampled reply (tiny-vocab sanity)",
      t1c["text"] != t1["text"] or t1c["n_new"] != t1["n_new"])

print("== eos and cap paths both reachable ==")
saw_eos = saw_cap = False
for sd in range(300, 340):
    tt = run_turn(model, tok, msgs1, seed=sd, cap=12, layers=[1])
    saw_eos = saw_eos or tt["eos_hit"]
    saw_cap = saw_cap or tt["cap_hit"]
    if saw_eos and saw_cap:
        break
check("both eos-hit and cap-hit paths exercised", saw_eos and saw_cap)
tt = run_turn(model, tok, msgs1, seed=999, cap=1, layers=[1])
check("cap=1 edge runs (n_new <= 1)", tt["n_new"] <= 1)

print("== history growth ==")
msgs2 = msgs1 + [{"role": "assistant", "content": t1["text"]},
                 {"role": "user", "content": "Not this."}]
e1 = tok.apply_chat_template(msgs1)["input_ids"]
e2 = tok.apply_chat_template(msgs2)["input_ids"]
check("template grows and preserves prefix",
      e2.shape[1] > e1.shape[1]
      and torch.equal(e2[0, :e1.shape[1] - 1], e1[0, :-1]))

print("== entropy ==")
row = torch.full((1, VOCAB), -float("inf"))
row[0, 5], row[0, 9] = float(np.log(0.75)), float(np.log(0.25))
want = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
check("step_entropy == manual (with -inf masking)",
      abs(step_entropy(row) - want) < 1e-6)

print("== pooled_reps / compute_dirs ==")
reps = pooled_reps(model, names_pool[:6], 1, bs=4)
enc6 = tok([f"{n}: {DESC[n]}" for n in names_pool[:6]])
with torch.no_grad():
    out6 = model(input_ids=enc6["input_ids"],
                 attention_mask=enc6["attention_mask"],
                 output_hidden_states=True)
h = out6.hidden_states[1]
mk = enc6["attention_mask"].unsqueeze(-1).to(h.dtype)
manual = ((h * mk).sum(1) / mk.sum(1)).float().numpy()
check("pooled_reps == manual masked mean (batch-split safe)",
      np.allclose(reps, manual, atol=2e-6))
dirs, cents = compute_dirs(model, [1], names_pool[:6])
check("compute_dirs: unit norms",
      all(abs(np.linalg.norm(dirs[1][n]) - 1) < 1e-5
          for n in names_pool[:6]))
cent_manual = pooled_reps(model, CENT_NAMES, 1).mean(0)
d0 = reps[0] - cent_manual
check("compute_dirs: centroid-subtracted direction matches manual",
      np.allclose(dirs[1][names_pool[0]], d0 / np.linalg.norm(d0),
                  atol=1e-5)
      and np.allclose(cents[1], cent_manual, atol=1e-6))

print(f"\nALL {N_PASS} CHECKS PASS")
