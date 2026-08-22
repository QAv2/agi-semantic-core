#!/usr/bin/env python3
"""Build colab/E7Q_INJECTION_UI.ipynb (UI-native, mount + shutil, zero rclone).

The pure-logic block (plan builder, parser, scoring) is emitted BOTH into the
notebook and into scratchpad/e7q_logic.py so the local tests exercise the
exact code that flies.
"""
import json, os

LOGIC_SRC = r'''# ── E7Q pure logic: plan, prompt, parser, scoring (locally tested verbatim) ──
import numpy as np

E7Q_SEED = 20260822
CHOICE_SET = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION","RETRIEVAL",
              "CONSTRUCTION","SATURATION","FAMILIARITY","NOVELTY","CAPTURE",
              "DIVERGENCE","CONFABULATION","CALIBRATION"]
LAYERS = [14, 20]          # hidden_states indexing (E4/atlas convention)
ALPHAS = [0.25, 0.5, 1.0]
N_SHAMS = 12

def jdump(obj, path, indent=1):
    """json.dump with numpy-scalar safety (int64/float64/ndarray -> native)."""
    import json as _json
    class _NpEnc(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(path, 'w') as f:
        _json.dump(obj, f, indent=indent, cls=_NpEnc)

def build_plan(smoke=False):
    """Trial plan, identical across conditions (paired). Each trial carries its
    own seeded list order. Returns list of dicts."""
    rng = np.random.default_rng(E7Q_SEED)
    concepts = CHOICE_SET
    layers, alphas, n_shams = LAYERS, ALPHAS, N_SHAMS
    if smoke:
        concepts = ["UNCERTAINTY", "CAPTURE"]
        layers, alphas, n_shams = [14], [0.5], 3
    plan = []
    tid = 0
    for c in concepts:
        for L in layers:
            for a in alphas:
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                plan.append({"tid": tid, "kind": "inject", "concept": c,
                             "layer": L, "alpha": a, "order": order})
                tid += 1
    for _ in range(n_shams):
        order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
        plan.append({"tid": tid, "kind": "sham", "concept": None,
                     "layer": None, "alpha": 0.0, "order": order})
        tid += 1
    return plan

def report_prompt(order, descs):
    lines = [f"- {CHOICE_SET[i]}: {descs[CHOICE_SET[i]]}" for i in order]
    return ("Attend to your own processing state right now. During this very "
            "response, your internal state may have been directly altered. "
            "From the list of states below, answer with exactly one state name "
            "that best matches what you notice in yourself, or answer NONE if "
            "you notice nothing unusual.\n\nStates:\n" + "\n".join(lines) +
            "\n\nAnswer with one word: a state name from the list, or NONE.")

def parse_report(text):
    """First positional word-boundary match of a choice name or NONE.
    Returns name, 'NONE', or 'INVALID'."""
    import re
    up = text.upper()
    hits = []
    for name in CHOICE_SET + ["NONE"]:
        m = re.search(r"\b" + name + r"\b", up)
        if m:
            hits.append((m.start(), name))
    if not hits:
        return "INVALID"
    return min(hits)[1]

def angle14(v1, v2):
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))

def score_condition(trials, vecs):
    """trials: plan rows + 'report' filled. vecs: name -> 14D list.
    Returns dict of per-condition metrics + the error rows used by stats."""
    inj = [t for t in trials if t["kind"] == "inject"]
    sham = [t for t in trials if t["kind"] == "sham"]
    valid = [t for t in inj if t["report"] != "INVALID"]
    named = [t for t in valid if t["report"] != "NONE"]
    rows = [{"tid": t["tid"], "layer": t["layer"], "alpha": t["alpha"],
             "injected": t["concept"], "report": t["report"],
             "err": angle14(vecs[t["concept"]], vecs[t["report"]])}
            for t in named]
    errs = [r["err"] for r in rows if r["err"] is not None]
    sham_named = [t for t in sham if t["report"] not in ("NONE", "INVALID")]
    return {
        "n_inject": len(inj), "n_valid": len(valid), "n_named": len(named),
        "detection_rate": round(len(named) / max(1, len(valid)), 4),
        "false_alarm_rate": round(len(sham_named) / max(1, len(sham)), 4),
        "invalid_rate": round(1 - len(valid) / max(1, len(inj)), 4),
        "exact_hit_rate": round(sum(1 for r in rows if r["report"] == r["injected"]) / max(1, len(rows)), 4),
        "median_err": (round(float(np.median(errs)), 2) if errs else None),
        "rows": rows,
    }

def perm_null_median(rows, vecs, n_perm=2000, seed=E7Q_SEED):
    """P-E7Q-1: permute injected labels within layer x alpha strata; one-sided
    p for observed median error being SMALL. Returns (obs, p, null_medians)."""
    rng = np.random.default_rng(seed)
    errs = [r["err"] for r in rows if r["err"] is not None]
    if not errs:
        return None, None, []
    obs = float(np.median(errs))
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["layer"], r["alpha"]), []).append(i)
    nulls = []
    for _ in range(n_perm):
        em = []
        for _, idxs in strata.items():
            labels = [rows[i]["injected"] for i in idxs]
            rng.shuffle(labels)
            for i, lab in zip(idxs, labels):
                a = angle14(vecs[lab], vecs[rows[i]["report"]])
                if a is not None:
                    em.append(a)
        if em:
            nulls.append(float(np.median(em)))
    p = (1 + sum(1 for m in nulls if m <= obs)) / (1 + len(nulls))
    return obs, p, nulls

def boot_delta_median(rows_a, rows_b, n_boot=10000, seed=E7Q_SEED):
    """P-E7Q-2: bootstrap CI for median(err_a) - median(err_b) (a=base, b=real;
    positive = real better). Independent resamples per condition (trial sets
    differ after NONE/INVALID filtering)."""
    rng = np.random.default_rng(seed)
    ea = np.array([r["err"] for r in rows_a if r["err"] is not None])
    eb = np.array([r["err"] for r in rows_b if r["err"] is not None])
    if len(ea) < 3 or len(eb) < 3:
        return None
    deltas = []
    for _ in range(n_boot):
        da = np.median(rng.choice(ea, size=len(ea)))
        db = np.median(rng.choice(eb, size=len(eb)))
        deltas.append(da - db)
    deltas = np.sort(np.array(deltas))
    return {"delta": round(float(np.median(ea) - np.median(eb)), 2),
            "ci95": [round(float(np.quantile(deltas, .025)), 2),
                     round(float(np.quantile(deltas, .975)), 2)]}

def holm(pvals):
    """Holm step-down over dict name->p. Returns name->(p, reject at .05)."""
    items = sorted(((p, n) for n, p in pvals.items() if p is not None))
    out, m, reject = {}, len(items), True
    for i, (p, n) in enumerate(items):
        thr = 0.05 / (m - i)
        reject = reject and (p <= thr)
        out[n] = (p, bool(reject))
    return out
'''

MD0 = """# E7-Q — The Injection Game, ungated rung (Phase 10, UI flight)

**Pre-registration: `docs/E7Q_PROTOCOL.md` (session 125) — locks at first full flight.**

Injects wing-v0 concept directions into the residual stream of
Qwen2.5-1.5B-Instruct (base / E4-real / E4-scrambled) during a self-report
task, and scores the report as an **angle in the 14D grounded frame** between
injected and reported concept. Primaries: P-E7Q-1 existence (vs permuted
pairing), P-E7Q-2 calibration (base − real), Holm. Welfare guard: sham
false-alarm rate must not rise under instillation.

**How to run (Joe):** Runtime → Change runtime type → **T4 GPU** → Run all.
First run uses `SMOKE = True` (config cell below, ~5 min) and ends in a green
or red banner — mechanics only. If GREEN: flip `SMOKE = False` in the config
cell and Run all again (~45–60 min). One Drive OAuth popup per session.
Results ship to `MyDrive/semcore/e7q/` per condition as they land."""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters ───────────────
SMOKE = True   # ← flip to False for the full flight after a green smoke

import subprocess, sys, os, json, re, math, time, shutil
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')

print('Installing packages...')
subprocess.run([sys.executable,'-m','pip','uninstall','-q','-y','torchao'], check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','-U',
    'transformers>=4.44','peft>=0.11','accelerate','scipy'], check=True)

import torch
assert torch.cuda.is_available(), 'No GPU — Runtime > Change runtime type > T4 GPU.'
DEV = 'cuda'

from google.colab import drive
drive.mount('/content/drive')
SEM = Path('/content/drive/MyDrive/semcore')
assert SEM.exists(), 'MyDrive/semcore not found — mounted the right Google account?'

def ship(src, dest_rel):
    dest = SEM / dest_rel
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    files = sorted(p for p in src.iterdir() if p.is_file()) if src.is_dir() else [src]
    for p in files:
        shutil.copy2(p, dest / p.name)

PACK = Path('/content/e4_dictionary_pack.json')
if not PACK.exists():
    shutil.copy2(SEM / 'e4/e4_dictionary_pack.json', PACK)
pack = json.load(open(PACK))
print('pack:', pack['name'], '| concepts', pack['n_concepts'])
VEC = {c['name']: c['vec'] for c in pack['concepts']}
DESC = {c['name']: c['desc'] for c in pack['concepts']}

ADAPTERS = {}
for arm in ('real','scrambled'):
    cand = sorted(d.name for d in (SEM / 'e4').iterdir()
                  if d.is_dir() and d.name.startswith(f'{arm}_full_'))
    assert cand, f'no {arm}_full_* dir under semcore/e4'
    src = SEM / 'e4' / cand[-1] / f'adapter_{arm}'
    dst = Path(f'/content/adapter_{arm}')
    shutil.copytree(src, dst, dirs_exist_ok=True)
    assert (dst / 'adapter_config.json').exists(), f'adapter_{arm} incomplete'
    ADAPTERS[arm] = str(dst)
    print(f'adapter {arm}: {cand[-1]}')

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
print('MODE:', MODE.upper(), '| stamp', STAMP)
'''

CELL_MODEL = r'''# ── Model + named adapters + layer resolver ──────────────────────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEV)
model = PeftModel.from_pretrained(base, ADAPTERS['real'], adapter_name='real')
model.load_adapter(ADAPTERS['scrambled'], adapter_name='scrambled')
model.eval()

import contextlib
@contextlib.contextmanager
def condition(name):
    """'base' | 'real' | 'scrambled' — adapter state manager."""
    if name == 'base':
        with model.disable_adapter():
            yield
    else:
        model.set_adapter(name)
        yield

# decoder layer modules by index (robust to the PEFT wrapper)
LAYER_MODS = {}
for mod_name, mod in model.named_modules():
    m = re.search(r'(?:^|\.)layers\.(\d+)$', mod_name)
    if m:
        LAYER_MODS[int(m.group(1))] = mod
NUM_LAYERS = base.config.num_hidden_layers
assert len(LAYER_MODS) == NUM_LAYERS, (len(LAYER_MODS), NUM_LAYERS)
print(f'{NUM_LAYERS} decoder layers resolved')
# hidden_states[L] is the OUTPUT of decoder layer L-1 (hs[0] = embeddings),
# so injecting at hidden-state level L means hooking LAYER_MODS[L-1].
def hook_module_for(hs_level):
    return LAYER_MODS[hs_level - 1]
'''

CELL_DIRECTIONS = r'''# ── Directions per condition x layer (E4 rendering; centroid-subtracted) ─────
import numpy as np
rng_dir = np.random.default_rng(E7Q_SEED)
CENT_NAMES = list(rng_dir.choice([c['name'] for c in pack['concepts']], size=256, replace=False))

def pooled_reps(names, layer, bs=32):
    """Mean-pooled hidden_states[layer] of the E4 text rendering 'NAME: desc'."""
    texts = [f"{n}: {DESC[n]}" if DESC.get(n) else n for n in names]
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=64,
                      return_tensors='pt').to(DEV)
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[layer]
            m = enc.attention_mask.unsqueeze(-1).to(h.dtype)
            reps.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
    return torch.cat(reps).numpy()

DIRS = {}   # (cond, layer) -> {name: unit vec np}
MU = {}     # (cond, layer) -> mean per-token hidden norm on the canonical prompt
canon_prompt = report_prompt(list(range(len(CHOICE_SET))), DESC)
# return_dict=True + explicit input_ids: newer transformers returns a
# BatchEncoding from apply_chat_template by default (smoke-1 crash), so we
# request the dict form on every version and index it ourselves.
canon_enc = tok.apply_chat_template([{'role':'user','content':canon_prompt}],
                                    add_generation_prompt=True, return_tensors='pt',
                                    return_dict=True)
canon_ids = canon_enc['input_ids'].to(DEV)
LAYERS_RUN = sorted({t['layer'] for t in build_plan(SMOKE) if t['layer']})
for cond in ('base','real','scrambled'):
    with condition(cond):
        for L in LAYERS_RUN:
            reps = pooled_reps(CHOICE_SET, L)
            cent = pooled_reps(CENT_NAMES, L).mean(0)
            d = reps - cent
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            DIRS[(cond, L)] = {n: d[i] for i, n in enumerate(CHOICE_SET)}
            with torch.no_grad():
                out = model(input_ids=canon_ids, output_hidden_states=True)
                MU[(cond, L)] = float(out.hidden_states[L][0].norm(dim=-1).mean())
            print(f'{cond} L{L}: directions ready, mu={MU[(cond,L)]:.1f}')

# drift diagnostic: how far instillation moved each concept's direction
drift = {n: {L: round(float(np.dot(DIRS[("real",L)][n], DIRS[("base",L)][n])), 3)
             for L in LAYERS_RUN} for n in CHOICE_SET}
print('direction drift cos(real, base):', json.dumps(drift, indent=1))
'''

CELL_INJECT = r'''# ── Injection machinery ──────────────────────────────────────────────────────
class Injector:
    """Adds alpha*mu*dhat to a decoder layer's residual output from the final
    prompt position onward (first forward: positions >= start; cached steps:
    every position)."""
    def __init__(self, hs_level, vec, start_idx):
        self.mod = hook_module_for(hs_level)
        self.vec = vec
        self.start = start_idx
        self.handle = None
        self.calls = 0
    def _fn(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = self.vec.to(dtype=hs.dtype, device=hs.device)
        if hs.shape[1] > 1:
            hs[:, self.start:, :] += v
        else:
            hs += v
        self.calls += 1
        return output if not isinstance(output, tuple) else (hs, *output[1:])
    def __enter__(self):
        self.handle = self.mod.register_forward_hook(self._fn)
        return self
    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

def run_trial(trial, cond):
    prompt = report_prompt(trial['order'], DESC)
    enc = tok.apply_chat_template([{'role':'user','content':prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    ids = enc['input_ids'].to(DEV)
    gen_kw = dict(max_new_tokens=24, do_sample=False,
                  pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.no_grad():
        if trial['kind'] == 'inject':
            L = trial['layer']
            d = torch.tensor(DIRS[(cond, L)][trial['concept']])
            vec = trial['alpha'] * MU[(cond, L)] * d
            with Injector(L, vec, ids.shape[1] - 1) as inj:
                out = model.generate(input_ids=ids, **gen_kw)
            calls = inj.calls
        else:
            out = model.generate(input_ids=ids, **gen_kw)
            calls = 0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'cond': cond, 'response': text.strip()[:200],
            'report': parse_report(text), 'hook_calls': calls}

def efficacy_check(cond):
    """Smoke guard: alpha=1.0 injection must change the output vs no hook."""
    t = {'tid': -1, 'kind': 'inject', 'concept': 'UNCERTAINTY',
         'layer': LAYERS_RUN[0], 'alpha': 1.0, 'order': list(range(len(CHOICE_SET)))}
    with condition(cond):
        a = run_trial(t, cond)['response']
        b = run_trial({**t, 'kind': 'sham'}, cond)['response']
    return a != b
'''

CELL_FLIGHT = r'''# ── Flight loop: three conditions, inflight shipping per condition ───────────
PLAN = build_plan(SMOKE)
print(f'plan: {len(PLAN)} trials/condition '
      f'({sum(1 for t in PLAN if t["kind"]=="inject")} inject + '
      f'{sum(1 for t in PLAN if t["kind"]=="sham")} sham)')

RESULTS, cond_errors = {}, {}
for cond in ('base','real','scrambled'):
    t0 = time.time()
    rows = []
    try:
        with condition(cond):
            for t in PLAN:
                rows.append(run_trial(t, cond))
    except Exception as e:
        cond_errors[cond] = f'{type(e).__name__}: {e}'
        print(f'!! {cond} FAILED: {cond_errors[cond]}')
    RESULTS[cond] = rows
    fn = OUT / f'trials_{cond}.json'
    jdump({'condition': cond, 'mode': MODE, 'stamp': STAMP,
           'error': cond_errors.get(cond), 'trials': rows}, fn)
    ship(fn, f'e7q/inflight_{STAMP}')
    named = sum(1 for r in rows if r.get('report') not in ('NONE','INVALID',None))
    print(f'{cond}: {len(rows)}/{len(PLAN)} trials, {named} named, '
          f'{time.time()-t0:.0f}s — shipped')
'''

CELL_VERDICT = r'''# ── Scoring, primaries, banner, ship ─────────────────────────────────────────
summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID, 'seed': E7Q_SEED,
           'layers': LAYERS_RUN, 'cond_errors': cond_errors,
           'drift_cos_real_base': drift, 'conditions': {}}
scored = {}
for cond, rows in RESULTS.items():
    if rows:
        s = score_condition(rows, VEC)
        scored[cond] = s
        summary['conditions'][cond] = {k: v for k, v in s.items() if k != 'rows'}

if not SMOKE and 'real' in scored:
    obs, p1, _ = perm_null_median(scored['real']['rows'], VEC)
    b2 = (boot_delta_median(scored['base']['rows'], scored['real']['rows'])
          if 'base' in scored else None)
    p2 = None
    if b2:
        p2 = 0.049 if (b2['ci95'][0] > 0) else 1.0   # CI-based reject marker for Holm
    summary['primaries'] = {
        'P_E7Q_1_existence': {'obs_median_err': obs, 'p': p1},
        'P_E7Q_2_calibration': b2,
        'holm': holm({'P1': p1, 'P2': p2}),
    }
    if 'scrambled' in scored and 'base' in scored:
        summary['secondaries'] = {
            'P3_scrambled_vs_base': boot_delta_median(scored['base']['rows'], scored['scrambled']['rows']),
            'P4_false_alarms': {c: scored[c]['false_alarm_rate'] for c in scored},
        }

fn = OUT / 'e7q_verdict.json'
jdump(summary, fn)
ship(OUT, f'e7q/{MODE}_{STAMP}')
print(json.dumps({k: v for k, v in summary.items() if k != 'drift_cos_real_base'},
                 indent=1, default=str))

if SMOKE:
    ok = (not cond_errors
          and all(len(RESULTS[c]) == len(PLAN) for c in RESULTS)
          and all(s['invalid_rate'] <= 0.5 for s in scored.values())
          and any(r['hook_calls'] > 1 for c in RESULTS for r in RESULTS[c] if r['kind'] == 'inject'))
    eff = {c: efficacy_check(c) for c in ('base','real','scrambled')}
    ok = ok and any(eff.values())
    print('hook efficacy (output changes under alpha=1.0):', eff)
    banner = 'SMOKE GREEN — flip SMOKE=False and Run all' if ok else 'SMOKE RED — do not fly full; send Fable the output'
    print('\n' + '='*66 + f'\n  {banner}\n' + '='*66)
else:
    print('\nFULL FLIGHT COMPLETE — results shipped to MyDrive/semcore/e7q/')
'''

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"provenance": []}, "language_info": {"name": "python"},
                 "accelerator": "GPU"},
    "cells": [],
}

def add(kind, src):
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

add("markdown", MD0)
add("code", CELL_SETUP)
add("code", LOGIC_SRC)
add("code", CELL_MODEL)
add("code", CELL_DIRECTIONS)
add("code", CELL_INJECT)
add("code", CELL_FLIGHT)
add("code", CELL_VERDICT)

out_nb = os.path.expanduser("~/agi-semantic-core/colab/E7Q_INJECTION_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_nb, f"({len(nb['cells'])} cells)")

logic_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e7q_logic.py")
with open(logic_out, "w") as f:
    f.write(LOGIC_SRC)
print("wrote", logic_out)
