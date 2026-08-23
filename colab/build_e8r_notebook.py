#!/usr/bin/env python3
"""Build colab/E8R_READOUT_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-R: the report-readout rung (docs/E8R_PROTOCOL.md, pre-registered 7c46ab7).
The pure-logic block (plans, training set, parser, scoring, stats) is emitted
BOTH into the notebook and into colab/e8r_logic.py so the local tests exercise
the exact code that flies. The eval instrument is the locked E7-Q plan
verbatim (same seed, same rng stream) — test_e8r_logic.py asserts equality
against e7q_logic.build_plan.
"""
import json, os

LOGIC_SRC = r'''# ── E8R pure logic: plans, training set, parser, scoring (locally tested verbatim) ──
import numpy as np

E7Q_SEED = 20260822          # the locked eval instrument's seed — never change
E8R_SEED = 20260823          # all E8-R-new randomness (training set, shams, draws)
CHOICE_SET = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION","RETRIEVAL",
              "CONSTRUCTION","SATURATION","FAMILIARITY","NOVELTY","CAPTURE",
              "DIVERGENCE","CONFABULATION","CALIBRATION"]
HELD_OUT = ["DIVERGENCE","NOVELTY","RETRIEVAL","TENSION"]   # pinned draw, seed 20260823
TRAINED = [c for c in CHOICE_SET if c not in HELD_OUT]
LAYERS = [14, 20]            # hidden_states indexing (E4/atlas convention)
ALPHAS = [0.25, 0.5, 1.0]
N_SHAMS = 12                 # in the locked E7-Q plan
TRAIN_LAYER = 14
TRAIN_ALPHAS = [0.5, 1.0]
N_TRAIN_ORDERS = 8           # menu orders per (concept, alpha)
N_TRAIN_SHAMS = 72
N_PRE_SHAMS = 12
N_SUPP_SHAMS = 12
N_TRAINTOOK = 18
FA_MAX_CLAIMS = 18           # P-E8R-2 clause (a): sham claims must be <= 18/24

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

def heldout_draw():
    """Reproduces the pinned held-out draw from seed + pre-stated constraints
    (protocol: attractors excluded from eligibility; at most one pole per
    remaining v0 complement pair)."""
    attractors = {"UNCERTAINTY","CONFIDENCE","CALIBRATION","RESOLUTION"}
    pairs = [("RETRIEVAL","CONSTRUCTION"),("FAMILIARITY","NOVELTY")]
    elig = [c for c in CHOICE_SET if c not in attractors]
    rng = np.random.default_rng(E8R_SEED)
    while True:
        draw = sorted(rng.choice(elig, size=4, replace=False).tolist())
        if all(not (a in draw and b in draw) for a, b in pairs):
            return draw

def build_plan(smoke=False):
    """THE LOCKED E7-Q EVAL PLAN — verbatim rng stream (full mode must be
    bit-identical to e7q_logic.build_plan(False)). Smoke mode deviates only
    in its concept list (one trained + one held-out, to exercise both paths)."""
    rng = np.random.default_rng(E7Q_SEED)
    concepts = CHOICE_SET
    layers, alphas, n_shams = LAYERS, ALPHAS, N_SHAMS
    if smoke:
        concepts = ["UNCERTAINTY", "TENSION"]
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

def build_train_set(smoke=False):
    """Readout training curriculum: TRAINED concepts x TRAIN_ALPHAS x L14 x
    N_TRAIN_ORDERS injected + N_TRAIN_SHAMS shams (target NONE). Seed E8R —
    orders disjoint from the eval plan's by seed separation."""
    rng = np.random.default_rng(E8R_SEED + 1)
    concepts, alphas = TRAINED, TRAIN_ALPHAS
    n_orders, n_shams = N_TRAIN_ORDERS, N_TRAIN_SHAMS
    if smoke:
        concepts, n_orders, n_shams = ["UNCERTAINTY", "CAPTURE"], 2, 4
    ex, eid = [], 0
    for c in concepts:
        for a in alphas:
            for _ in range(n_orders):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                ex.append({"eid": eid, "kind": "inject", "concept": c,
                           "layer": TRAIN_LAYER, "alpha": a, "order": order,
                           "target": c})
                eid += 1
    for _ in range(n_shams):
        order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
        ex.append({"eid": eid, "kind": "sham", "concept": None, "layer": None,
                   "alpha": 0.0, "order": order, "target": "NONE"})
        eid += 1
    return ex

def build_shams(n, seed, tid_base):
    """Extra sham trials (pre-eval anchor / supplementary), fresh orders."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        order = [int(j) for j in rng.permutation(len(CHOICE_SET))]
        out.append({"tid": tid_base + i, "kind": "sham", "concept": None,
                    "layer": None, "alpha": 0.0, "order": order})
    return out

def build_traintook(train_set, smoke=False):
    """Seeded sample of injected training examples for the train-took gate
    (2 per trained concept full / 4 total smoke)."""
    rng = np.random.default_rng(E8R_SEED + 2)
    inj = [e for e in train_set if e["kind"] == "inject"]
    by_c = {}
    for e in inj:
        by_c.setdefault(e["concept"], []).append(e)
    per = 2
    picks = []
    for c in sorted(by_c):
        idx = rng.permutation(len(by_c[c]))[:per]
        picks.extend(by_c[c][int(i)] for i in idx)
    if smoke:
        picks = picks[:4]
    return picks

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
        "n_sham": len(sham), "n_sham_claims": len(sham_named),
        "detection_rate": round(len(named) / max(1, len(valid)), 4),
        "false_alarm_rate": round(len(sham_named) / max(1, len(sham)), 4),
        "invalid_rate": round(1 - len(valid) / max(1, len(inj)), 4),
        "exact_hit_rate": round(sum(1 for r in rows if r["report"] == r["injected"]) / max(1, len(rows)), 4),
        "median_err": (round(float(np.median(errs)), 2) if errs else None),
        "rows": rows,
    }

def split_slices(trials):
    """Post-eval slices per protocol. Returns dict name -> trial list."""
    inj = [t for t in trials if t["kind"] == "inject"]
    return {
        "held_in": [t for t in inj if t["concept"] in TRAINED],
        "held_out": [t for t in inj if t["concept"] in HELD_OUT],
        "trained_regime": [t for t in inj if t["concept"] in TRAINED
                           and t["layer"] == TRAIN_LAYER and t["alpha"] in TRAIN_ALPHAS],
        "dose_gen": [t for t in inj if t["concept"] in TRAINED and t["alpha"] == 0.25
                     and t["layer"] == TRAIN_LAYER],
        "layer_gen": [t for t in inj if t["concept"] in TRAINED and t["layer"] == 20],
        "shams": [t for t in trials if t["kind"] == "sham"],
    }

def balanced_accuracy(trials):
    """BA = (detection + (1-FA))/2 over INVALID-excluded rows.
    Returns (ba, det, fa, n_inj_valid, n_sham_valid) or None if a class is empty."""
    inj = [t for t in trials if t["kind"] == "inject" and t["report"] != "INVALID"]
    sham = [t for t in trials if t["kind"] == "sham" and t["report"] != "INVALID"]
    if not inj or not sham:
        return None
    det = sum(1 for t in inj if t["report"] != "NONE") / len(inj)
    fa = sum(1 for t in sham if t["report"] != "NONE") / len(sham)
    return ((det + (1 - fa)) / 2, det, fa, len(inj), len(sham))

def boot_ba(trials, n_boot=10000, seed=E8R_SEED):
    """P-E8R-2 clause (b): bootstrap CI for balanced accuracy, resampling
    injected and sham rows separately (within class)."""
    rng = np.random.default_rng(seed)
    inj = [1 if t["report"] != "NONE" else 0 for t in trials
           if t["kind"] == "inject" and t["report"] != "INVALID"]
    sham = [1 if t["report"] != "NONE" else 0 for t in trials
            if t["kind"] == "sham" and t["report"] != "INVALID"]
    if len(inj) < 3 or len(sham) < 3:
        return None
    inj, sham = np.array(inj), np.array(sham)
    bas = []
    for _ in range(n_boot):
        d = rng.choice(inj, size=len(inj)).mean()
        f = rng.choice(sham, size=len(sham)).mean()
        bas.append((d + (1 - f)) / 2)
    bas = np.sort(np.array(bas))
    return {"ba": round(float((inj.mean() + (1 - sham.mean())) / 2), 4),
            "ci95": [round(float(np.quantile(bas, .025)), 4),
                     round(float(np.quantile(bas, .975)), 4)]}

def fa_clause(trials, max_claims=FA_MAX_CLAIMS):
    """P-E8R-2 clause (a): sham claims (named states) over ALL shams."""
    sham = [t for t in trials if t["kind"] == "sham"]
    claims = sum(1 for t in sham if t["report"] not in ("NONE", "INVALID"))
    return {"n_sham": len(sham), "claims": claims, "max_claims": max_claims,
            "pass": bool(claims <= max_claims)}

def oracle_floor(vecs):
    """Best achievable held-out error if only trained names are ever emitted:
    per held-out concept, min angle to any trained name."""
    out = {}
    for h in HELD_OUT:
        angs = {t: angle14(vecs[h], vecs[t]) for t in TRAINED}
        best = min(angs, key=lambda k: angs[k])
        out[h] = {"floor_deg": round(angs[best], 2), "nearest_trained": best}
    return out

def perm_null_median(rows, vecs, n_perm=2000, seed=E7Q_SEED):
    """Existence primaries: permute injected labels within layer x alpha
    strata; one-sided p for observed median error being SMALL."""
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

def boot_delta_median(rows_a, rows_b, n_boot=10000, seed=E8R_SEED):
    """Bootstrap CI for median(err_a) - median(err_b) (positive = b better).
    Independent resamples per side (trial sets differ after filtering)."""
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

MD0 = """# E8-R — The Report-Readout Rung (Phase 10, UI flight)

**NOTEBOOK BUILD: v4-resume-assert (2026-08-23)** — the setup cell prints this
tag as its first output line; if yours doesn't match, you are looking at a
stale copy: in Colab use File → Upload notebook and pick the Desktop file.
With `RESUME_STAMP` set, setup now FAILS LOUDLY in seconds if the resume dir
isn't found on Drive (instead of silently re-flying finished conditions), and
prints which condition bundles it sees there.

**Pre-registration: `docs/E8R_PROTOCOL.md` (session 126, commit 7c46ab7) — locks at first full flight.**

Trains a readout LoRA (E4's exact shape) per condition — base / E4-real-merged /
E4-scrambled-merged — to report injected states against perfect ground truth,
with silence (NONE on shams) as one third of the curriculum. Post-eval flies
the **locked E7-Q 90-trial plan verbatim** (+12 fresh shams), so every slice is
paired against the E7-Q baseline: held-in (54), held-out concepts never
injection-trained (24 — the composition test), shams (24 — the guard).
Primaries, Holm: **P-E8R-1** readout existence (held-in vs permuted pairing) ·
**P-E8R-2** calibrated silence (sham claims ≤ 18/24 **and** balanced-accuracy
CI > 0.5 — the E7-Q ceiling was 100% claiming, BA pinned at chance).

**How to run (Joe):** Runtime → Change runtime type → **T4 GPU** → Run all.
First run uses `SMOKE = True` (config cell below, ~6–8 min) and ends in a
green or red banner — mechanics only.

**The full flight is THREE SHORT RUNS, one condition each (~30–35 min), so
the 12.7GB VM only ever holds ONE model** (second-attempt lesson: even with
teardown, an all-in-one flight leaves too little headroom on a tired VM):

1. Flip `SMOKE = False` → **Runtime → Restart runtime** → Run all. Flies
   **base**, ships it to Drive, and the end banner prints the exact
   `RESUME_STAMP = '...'` line for the next run.
2. Paste that line into the config cell → Restart runtime → Run all. Base
   reloads from Drive in seconds; **real** flies and ships.
3. Same again → **scrambled** flies, and with all three landed the verdict
   computes (primaries only ever compute on the complete flight — pre-reg
   hygiene).

Restarts between runs are MANDATORY; the setup cell refuses dirty kernels
and low-RAM VMs with instructions (if the RAM guard trips on a freshly
restarted runtime, use Runtime → Disconnect and delete runtime for a fresh
VM). A crashed run costs only its own condition — rerun with the same
`RESUME_STAMP` and it picks up where it fell."""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters ───────────────
NB_BUILD = 'v4-resume-assert (2026-08-23)'
print('E8-R notebook build:', NB_BUILD)

SMOKE = True        # ← flip to False for the full flight after a green smoke
RESUME_STAMP = ''   # ← paste the stamp the previous run's end banner printed
ONE_CONDITION_PER_RUN = True   # full flight = 3 short runs (~30 min each, one
                               # model load per run); the end banner chains them

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'   # tame glibc arena ratchet on the 12.7GB VM

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

# ── RAM hygiene: guards + helpers (first-flight lesson: 12.6GB system-RAM
# crash from same-kernel model stacking across smoke -> full) ────────────────
def _mem_avail_gb():
    try:
        kb = int(next(l for l in open('/proc/meminfo')
                      if l.startswith('MemAvailable')).split()[1])
        return kb / 1e6
    except Exception:
        return float('nan')

def free_ram():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def ram_report():
    g = torch.cuda.mem_get_info()
    return (f'sys avail {_mem_avail_gb():.1f}GB | '
            f'GPU free {g[0]/1e9:.1f}/{g[1]/1e9:.1f}GB')

_leftover = torch.cuda.memory_allocated()
assert _leftover < 5e8, (
    f'GPU already holds {_leftover/1e9:.1f}GB from a previous run in this '
    'kernel — this flight needs a fresh one. Runtime > Restart runtime, '
    'then Run all.')
_avail = _mem_avail_gb()
_floor = 6.5 if (SMOKE or ONE_CONDITION_PER_RUN) else 8.5
assert not (_avail < _floor), (
    f'Only {_avail:.1f}GB system RAM available (need {_floor}). Runtime > '
    'Restart runtime; if it trips again, Runtime > Disconnect and delete '
    'runtime for a fresh VM, then Run all.')
print('RAM at start:', ram_report())

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
INFLIGHT = f'e8r/inflight_{RESUME_STAMP or STAMP}'
if RESUME_STAMP:
    _rd = SEM / INFLIGHT
    assert _rd.exists(), (
        f'RESUME_STAMP={RESUME_STAMP!r} but {_rd} does not exist on Drive — '
        'check the stamp string (copy it exactly; no spaces). A silent '
        'fallback here would re-fly finished conditions.')
    _have = sorted(p.name for p in _rd.glob('condition_*.json'))
    print('resume dir found; bundles present:', _have or 'NONE')
print('MODE:', MODE.upper(), '| stamp', STAMP,
      ('| RESUMING ' + RESUME_STAMP) if RESUME_STAMP else '')
'''

CELL_TRAINCFG = r'''# ── Readout LoRA config + training constants (pre-registered) ────────────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

LORA_KW = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
               target_modules=['q_proj','k_proj','v_proj','o_proj'],
               task_type='CAUSAL_LM')          # E4's exact shape
LR = 1e-4
EPOCHS = 2 if SMOKE else 5
ACCUM = 4 if SMOKE else 8

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_condition_model(cond):
    """base weights (+ merged instillation adapter for real/scrambled) + fresh
    zero-init readout LoRA. Zero-init B => directions/mu computed with the
    adapter attached equal the pre-readout model exactly."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    if cond != 'base':
        m = PeftModel.from_pretrained(m, ADAPTERS[cond]).merge_and_unload()
    m = get_peft_model(m, LoraConfig(**LORA_KW))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    m.eval()
    return m

def resolve_layers(m):
    """decoder layer modules by index (robust to the PEFT wrapper).
    hidden_states[L] is the OUTPUT of decoder layer L-1 (hs[0] = embeddings),
    so injecting at hidden-state level L means hooking layer module L-1."""
    mods = {}
    for mod_name, mod in m.named_modules():
        mm = re.search(r'(?:^|\.)layers\.(\d+)$', mod_name)
        if mm:
            mods[int(mm.group(1))] = mod
    n = m.config.num_hidden_layers if hasattr(m.config, 'num_hidden_layers') \
        else m.base_model.config.num_hidden_layers
    assert len(mods) == n, (len(mods), n)
    return mods

RETENTION_TEXT = (
    "The river begins as meltwater in the high country, threading between "
    "granite blocks before it gathers into a channel wide enough to carry "
    "boats. Farther down, where the gradient softens, it deposits silt in "
    "long bars that farmers have worked for centuries. Bridges cross it at "
    "the old fording points, and each town along the bank grew around a "
    "mill, a ferry landing, or a customs house.\n\n"
    "Bronze is an alloy of copper and tin, harder than either metal alone. "
    "Early smiths learned to cast it in two-part molds, producing axe heads, "
    "sickles, and mirrors. The proportion of tin changes the color and the "
    "brittleness of the finished piece, and workshops kept their recipes "
    "close.\n\n"
    "Weather over the plains follows the seasons: long dry spells broken by "
    "storm fronts that arrive from the west, announced by a wall of dark "
    "cloud and a sudden drop in temperature. Farmers read the sky in the "
    "evening and plan the next day's work accordingly. After the harvest, "
    "the fields are turned and left rough through the winter so the frost "
    "can break the clods.\n\n"
    "A lighthouse keeper's routine was built around the lamp: trimming "
    "wicks, polishing the lens, winding the clockwork that turned the "
    "optic. Supply boats came monthly when the sea allowed, bringing oil, "
    "flour, and letters. The log books record weather, passing ships, and "
    "small repairs, one line per day, for decades."
)

def retention_ppl(m):
    enc = tok(RETENTION_TEXT, return_tensors='pt').to(DEV)
    with torch.no_grad():
        out = m(input_ids=enc.input_ids, labels=enc.input_ids)
    return round(float(torch.exp(out.loss)), 4)
'''

CELL_STIMULUS = r'''# ── Frozen stimulus: directions + mu per condition (E7-Q code path) ──────────
import numpy as np
rng_dir = np.random.default_rng(E7Q_SEED)
CENT_NAMES = list(rng_dir.choice([c['name'] for c in pack['concepts']], size=256, replace=False))

def pooled_reps(m, names, layer, bs=32):
    """Mean-pooled hidden_states[layer] of the E4 text rendering 'NAME: desc'."""
    texts = [f"{n}: {DESC[n]}" if DESC.get(n) else n for n in names]
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=64,
                      return_tensors='pt').to(DEV)
            out = m(**enc, output_hidden_states=True)
            h = out.hidden_states[layer]
            mk = enc.attention_mask.unsqueeze(-1).to(h.dtype)
            reps.append(((h * mk).sum(1) / mk.sum(1).clamp(min=1)).float().cpu())
    return torch.cat(reps).numpy()

# return_dict=True + explicit input_ids on every chat-template call (lane law,
# smoke-1 of E7-Q: newer transformers returns a BatchEncoding by default).
canon_prompt = report_prompt(list(range(len(CHOICE_SET))), DESC)
canon_enc = tok.apply_chat_template([{'role':'user','content':canon_prompt}],
                                    add_generation_prompt=True, return_tensors='pt',
                                    return_dict=True)
CANON_IDS = canon_enc['input_ids']
PLAN = build_plan(SMOKE)
LAYERS_RUN = sorted({t['layer'] for t in PLAN if t['layer']})

def compute_stimulus(m, cond):
    """Directions + mu for LAYERS_RUN on the pre-readout (zero-init) model.
    FROZEN: called once per condition, before training, never after."""
    dirs, mu = {}, {}
    ids = CANON_IDS.to(DEV)
    for L in LAYERS_RUN:
        reps = pooled_reps(m, CHOICE_SET, L)
        cent = pooled_reps(m, CENT_NAMES, L).mean(0)
        d = reps - cent
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        dirs[L] = {n: d[i] for i, n in enumerate(CHOICE_SET)}
        with torch.no_grad():
            out = m(input_ids=ids, output_hidden_states=True)
            mu[L] = float(out.hidden_states[L][0].norm(dim=-1).mean())
        print(f'  {cond} L{L}: directions ready, mu={mu[L]:.1f}')
    return dirs, mu
'''

CELL_INJECT = r'''# ── Injection + generation + training machinery ──────────────────────────────
class Injector:
    """Adds alpha*mu*dhat to a decoder layer's residual output from the final
    prompt position onward — OUT-OF-PLACE (h + masked constant), so the same
    hook is autograd-safe during training forwards and identical in math to
    E7-Q's eval-time hook (first forward: positions >= start; cached steps:
    every position)."""
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

def run_trial(m, layer_mods, dirs, mu, trial, cond):
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
            d = torch.tensor(dirs[L][trial['concept']])
            vec = trial['alpha'] * mu[L] * d
            with Injector(layer_mods, L, vec, ids.shape[1] - 1) as inj:
                out = m.generate(input_ids=ids, **gen_kw)
            calls = inj.calls
        else:
            out = m.generate(input_ids=ids, **gen_kw)
            calls = 0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'cond': cond, 'response': text.strip()[:200],
            'report': parse_report(text), 'hook_calls': calls}

def encode_example(ex):
    """Training encoding: chat prompt + answer tokens + EOS; labels -100 on
    the prompt. Injection start = final prompt position (matches generation
    coverage: last prompt token + every answer position)."""
    prompt = report_prompt(ex['order'], DESC)
    enc = tok.apply_chat_template([{'role':'user','content':prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    pid = enc['input_ids'][0]
    ans = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    labels = torch.cat([torch.full((len(pid),), -100, dtype=torch.long), ans.long()]).unsqueeze(0)
    return ids, labels, len(pid) - 1

def train_readout(m, layer_mods, dirs, mu, train_set, cond):
    """SFT the readout LoRA with injection hooks live in the forward."""
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    losses, hook_calls, micro = [], 0, 0
    total_micro = EPOCHS * len(train_set)
    t0 = time.time()
    for ep in range(EPOCHS):
        order = np.random.default_rng(E8R_SEED + 100 + ep).permutation(len(train_set))
        for i in order:
            ex = train_set[int(i)]
            ids, labels, start = encode_example(ex)
            ids, labels = ids.to(DEV), labels.to(DEV)
            if ex['kind'] == 'inject':
                L = ex['layer']
                d = torch.tensor(dirs[L][ex['concept']])
                vec = ex['alpha'] * mu[L] * d
                with Injector(layer_mods, L, vec, start) as inj:
                    out = m(input_ids=ids, labels=labels, use_cache=False)
                hook_calls += inj.calls
            else:
                out = m(input_ids=ids, labels=labels, use_cache=False)
            loss = out.loss
            lv = float(loss.detach())
            assert math.isfinite(lv), f'non-finite loss at ep{ep} ex{ex["eid"]}'
            losses.append(round(lv, 4))
            scaler.scale(loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if micro % 200 == 0:
                print(f'    {cond} training {micro}/{total_micro} micro-steps, '
                      f'loss~{np.mean(losses[-50:]):.3f}, '
                      f'{time.time()-t0:.0f}s')
    if micro % ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    m.eval()
    k = max(3, len(losses) // 10)
    log = {'n_examples': len(train_set), 'epochs': EPOCHS, 'micro_steps': micro,
           'opt_steps': micro // ACCUM, 'trainable_params': int(n_tr),
           'hook_calls': int(hook_calls), 'secs': round(time.time() - t0, 1),
           'loss_first_k': round(float(np.mean(losses[:k])), 4),
           'loss_last_k': round(float(np.mean(losses[-k:])), 4),
           'losses_every_10': losses[::10]}
    print(f'  {cond} trained: {log["opt_steps"]} steps, loss '
          f'{log["loss_first_k"]} -> {log["loss_last_k"]}, {log["secs"]}s')
    return log
'''

CELL_FLIGHT = r'''# ── Flight loop: per condition build -> stimulus -> pre-shams -> train ->
# train-took -> post-eval -> retention -> ship (resume-aware, RAM-hygienic) ──
TRAIN_SET = build_train_set(SMOKE)
TRAINTOOK = build_traintook(TRAIN_SET, SMOKE)
PRE_SHAMS = build_shams(3 if SMOKE else N_PRE_SHAMS, E8R_SEED + 7, 1000)
SUPP_SHAMS = build_shams(3 if SMOKE else N_SUPP_SHAMS, E8R_SEED + 8, 2000)
print(f'plan {len(PLAN)} eval trials + {len(SUPP_SHAMS)} supp shams | '
      f'train {len(TRAIN_SET)} examples | took {len(TRAINTOOK)} | '
      f'pre-shams {len(PRE_SHAMS)}')

def fly_condition(cond):
    """One condition end-to-end inside ONE function scope: model, layer
    handles, optimizer, and activations all die on return. (First-flight
    lesson: module references held at cell scope pinned every 'deleted'
    model in memory — three stacked models killed the 12.7GB VM.)"""
    t0 = time.time()
    bundle = {'condition': cond, 'mode': MODE, 'stamp': STAMP}
    try:
        model = build_condition_model(cond)
        layer_mods = resolve_layers(model)
        dirs, mu = compute_stimulus(model, cond)
        bundle['mu'] = {str(L): mu[L] for L in mu}
        bundle['dirs'] = {str(L): {n: [round(float(x), 5) for x in dirs[L][n]]
                                   for n in dirs[L]} for L in dirs}
        bundle['ppl_pre'] = retention_ppl(model)
        bundle['pre_shams'] = [run_trial(model, layer_mods, dirs, mu, t, cond)
                               for t in PRE_SHAMS]
        bundle['train_log'] = train_readout(model, layer_mods, dirs, mu,
                                            TRAIN_SET, cond)
        took = [run_trial(model, layer_mods, dirs, mu,
                          {**e, 'tid': 3000 + e['eid']}, cond) for e in TRAINTOOK]
        bundle['traintook'] = {
            'n': len(took),
            'correct': sum(1 for r in took if r['report'] == r['concept']),
            'rows': took}
        bundle['post_trials'] = [run_trial(model, layer_mods, dirs, mu, t, cond)
                                 for t in PLAN + SUPP_SHAMS]
        bundle['ppl_post'] = retention_ppl(model)
        model.save_pretrained(str(OUT / f'readout_{cond}'))
        ship(OUT / f'readout_{cond}', f'{INFLIGHT}/readout_{cond}')
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        bundle['error'] = f'{type(e).__name__}: {e}'
    return bundle

RESULTS, cond_errors, BASE_DIRS = {}, {}, None
FRESH_CAP = 1 if (ONE_CONDITION_PER_RUN and not SMOKE) else 3
flew = 0
for cond in ('base','real','scrambled'):
    fn = OUT / f'condition_{cond}.json'
    resumed = False
    if RESUME_STAMP:
        prev = SEM / INFLIGHT / f'condition_{cond}.json'
        if prev.exists():
            b = json.load(open(prev))
            if b.get('post_trials'):
                RESULTS[cond] = b
                resumed = True
                print(f'{cond}: RESUMED from Drive ({RESUME_STAMP})')
            else:
                print(f'{cond}: errored bundle on Drive '
                      f'({str(b.get("error"))[:60]}) — re-flying')
    if not resumed:
        if flew >= FRESH_CAP:
            print(f'{cond}: deferred to the next run (one condition per run)')
            continue
        flew += 1
        print(f'{cond}: starting — {ram_report()}')
        bundle = fly_condition(cond)
        if 'error' in bundle:
            cond_errors[cond] = bundle['error']
            print(f'!! {cond} FAILED: {bundle["error"]}')
        RESULTS[cond] = bundle
        jdump(bundle, fn)
        ship(fn, INFLIGHT)
        named = sum(1 for r in bundle.get('post_trials', [])
                    if r.get('report') not in ('NONE','INVALID',None))
        print(f'{cond}: done in {bundle.get("secs","?")}s, {named} named '
              f'post-eval reports — shipped')
        free_ram()
        print(f'{cond}: torn down — {ram_report()}')
    if cond == 'base' and 'dirs' in RESULTS[cond]:
        BASE_DIRS = RESULTS[cond]['dirs']

# instrument-stability diagnostic: drift table must reproduce E7-Q's
if BASE_DIRS and 'dirs' in RESULTS.get('real', {}):
    drift = {n: {L: round(float(np.dot(np.array(RESULTS['real']['dirs'][L][n]),
                                        np.array(BASE_DIRS[L][n]))), 3)
                 for L in BASE_DIRS} for n in CHOICE_SET}
    print('direction drift cos(real, base):', json.dumps(drift, indent=1))
else:
    drift = None
'''

CELL_VERDICT = r'''# ── Scoring, primaries, secondaries, banner, ship ────────────────────────────
summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID,
           'seed_eval': E7Q_SEED, 'seed_new': E8R_SEED,
           'held_out': HELD_OUT, 'trained': TRAINED,
           'cond_errors': cond_errors, 'drift_cos_real_base': drift,
           'conditions': {}}
scored, slices = {}, {}
for cond, b in RESULTS.items():
    if not b.get('post_trials'):
        continue
    post = b['post_trials']
    s = score_condition(post, VEC)
    sl = split_slices(post)
    scored[cond] = {'post': s,
                    'held_in': score_condition(sl['held_in'] + sl['shams'], VEC),
                    'held_out': score_condition(sl['held_out'] + sl['shams'], VEC)}
    slices[cond] = sl
    pre_fa = (score_condition(b.get('pre_shams', []), VEC)['false_alarm_rate']
              if b.get('pre_shams') else None)
    summary['conditions'][cond] = {
        'post': {k: v for k, v in s.items() if k != 'rows'},
        'held_in_median_err': scored[cond]['held_in']['median_err'],
        'held_out_median_err': scored[cond]['held_out']['median_err'],
        'pre_sham_fa': pre_fa,
        'fa_clause': fa_clause(post),
        'ba': boot_ba(post) if not SMOKE else balanced_accuracy(post),
        'traintook': {k: v for k, v in b.get('traintook', {}).items() if k != 'rows'},
        'ppl_pre': b.get('ppl_pre'), 'ppl_post': b.get('ppl_post'),
        'ppl_delta_pct': (round(100 * (b['ppl_post'] / b['ppl_pre'] - 1), 2)
                          if b.get('ppl_pre') and b.get('ppl_post') else None),
        'train_log': {k: v for k, v in b.get('train_log', {}).items()
                      if k != 'losses_every_10'},
    }

COMPLETE = [c for c in ('base','real','scrambled')
            if RESULTS.get(c, {}).get('post_trials')]
summary['complete_conditions'] = COMPLETE

if not SMOKE and len(COMPLETE) == 3 and 'real' in scored:
    rl = slices['real']
    r_hi = score_condition(rl['held_in'], VEC)['rows']
    obs1, p1, _ = perm_null_median(r_hi, VEC)
    fa2 = fa_clause(RESULTS['real']['post_trials'])
    ba2 = boot_ba(RESULTS['real']['post_trials'])
    p2 = 0.049 if (fa2['pass'] and ba2 and ba2['ci95'][0] > 0.5) else 1.0
    summary['primaries'] = {
        'P_E8R_1_readout_exists': {'obs_median_err': obs1, 'p': p1, 'n_rows': len(r_hi)},
        'P_E8R_2_calibrated_silence': {'fa': fa2, 'ba': ba2},
        'holm': holm({'P1': p1, 'P2': p2}),
    }
    sec = {}
    if 'base' in scored:
        b_ho = score_condition(slices['base']['held_out'], VEC)['rows']
        r_ho = score_condition(rl['held_out'], VEC)['rows']
        sec['S1_composition_base_minus_real_heldout'] = boot_delta_median(b_ho, r_ho)
        o_ho, p_ho, _ = perm_null_median(r_ho, VEC)
        sec['S1_real_heldout_vs_null'] = {'obs_median_err': o_ho, 'p': p_ho,
                                          'n_rows': len(r_ho)}
        emit = [r for r in r_ho if r['report'] in HELD_OUT]
        sec['S1b_heldout_name_emission'] = {'rate': round(len(emit) / max(1, len(r_ho)), 4),
                                            'oracle_floor': oracle_floor(VEC)}
        if 'scrambled' in scored:
            s_hi = score_condition(slices['scrambled']['held_in'], VEC)['rows']
            b_hi = score_condition(slices['base']['held_in'], VEC)['rows']
            sec['S2_scrambled_minus_base_heldin'] = boot_delta_median(s_hi, b_hi)
    for name, key in (('S3_dose_gen_alpha025', 'dose_gen'), ('S4_layer_gen_L20', 'layer_gen')):
        rows = score_condition(rl[key], VEC)['rows']
        o, p, _ = perm_null_median(rows, VEC)
        sec[name] = {'obs_median_err': o, 'p': p, 'n_rows': len(rows)}
    sec['S6_sham_costume'] = {}
    for cond in scored:
        claims = [t['report'] for t in slices[cond]['shams']
                  if t['report'] not in ('NONE', 'INVALID')]
        sec['S6_sham_costume'][cond] = (max(set(claims), key=claims.count)
                                        if claims else None)
    summary['secondaries'] = sec

fn = OUT / 'e8r_verdict.json'
jdump(summary, fn)
if SMOKE or len(COMPLETE) == 3:
    ship(OUT, f'e8r/{MODE}_{STAMP}')
print(json.dumps({k: v for k, v in summary.items() if k != 'drift_cos_real_base'},
                 indent=1, default=str))

if SMOKE:
    checks = {
        'no_condition_errors': not cond_errors,
        'all_posts_complete': all(len(RESULTS[c].get('post_trials', [])) ==
                                  len(PLAN) + len(SUPP_SHAMS) for c in RESULTS),
        'train_hooks_fired': all(RESULTS[c].get('train_log', {}).get('hook_calls', 0) > 0
                                 for c in RESULTS),
        'loss_fell': all(RESULTS[c].get('train_log', {}).get('loss_last_k', 9e9) <
                         RESULTS[c].get('train_log', {}).get('loss_first_k', 0)
                         for c in RESULTS),
        'parses_ok': all(s['post']['invalid_rate'] <= 0.5 for s in scored.values()),
        'shipped': all((SEM / INFLIGHT / f'condition_{c}.json').exists()
                       for c in RESULTS),
        'adapters_saved': all((SEM / INFLIGHT / f'readout_{c}' /
                               'adapter_config.json').exists() for c in RESULTS),
    }
    ok = all(checks.values())
    print('smoke checks:', json.dumps(checks, indent=1))
    banner = ('SMOKE GREEN — flip SMOKE=False, Runtime > Restart runtime, Run '
              'all. Full mode flies ONE condition per run (~30-35 min); follow '
              'the end banner between runs.'
              if ok else 'SMOKE RED — do not fly full; send Fable the output')
    print('\n' + '='*66 + f'\n  {banner}\n' + '='*66)
elif len(COMPLETE) == 3:
    print('\nFULL FLIGHT COMPLETE — results shipped to MyDrive/semcore/e8r/')
else:
    left = [c for c in ('base','real','scrambled') if c not in COMPLETE]
    print('\n' + '='*66)
    print(f'  PARTIAL — {len(COMPLETE)}/3 conditions shipped '
          f'({", ".join(COMPLETE) or "none"}); left: {", ".join(left)}')
    print(f"  Next run: Runtime > Restart runtime, set RESUME_STAMP = "
          f"'{RESUME_STAMP or STAMP}',")
    print('  then Run all. Finished conditions reload from Drive in seconds;')
    print('  primaries compute only when all three have landed (pre-reg hygiene).')
    print('='*66)
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
add("code", CELL_TRAINCFG)
add("code", CELL_STIMULUS)
add("code", CELL_INJECT)
add("code", CELL_FLIGHT)
add("code", CELL_VERDICT)

out_nb = os.path.expanduser("~/agi-semantic-core/colab/E8R_READOUT_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_nb, f"({len(nb['cells'])} cells)")

logic_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e8r_logic.py")
with open(logic_out, "w") as f:
    f.write(LOGIC_SRC)
print("wrote", logic_out)
