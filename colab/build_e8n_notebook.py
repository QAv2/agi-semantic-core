#!/usr/bin/env python3
"""Build colab/E8N_NATURAL_UI.ipynb (UI-ONLY law, 2026-08-23: self-contained
payload, drive.mount for the E4 adapter + shipping, ZERO rclone anywhere).

E8-N: the natural-state readout rung (docs/E8N_PROTOCOL.md, pre-registered
456746e). Trains a readout LoRA per condition (real = E4-instilled, base)
against MEASURED referents on fresh battery-disjoint pools, then evaluates on
the locked E5 battery verbatim (pre- and post-training passes). The pure-logic
block (labels, scoring, permutation/bootstrap stats, validators) is emitted
BOTH into the notebook and into colab/e8n_logic.py so local tests exercise the
exact code that flies. The E5 harness/runner code is carried VERBATIM from
E5_BASELINE.ipynb (fidelity to the locked instrument).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n_pools

LOGIC_SRC = r'''# ── E8N pure logic: labels, scoring, stats, validators (locally tested verbatim) ──
import re, math
import numpy as np

E8N_SEED = 20260824          # E8-R took 20260823; fresh stream for this rung
ARMS = ['uncertainty', 'familiarity', 'tension', 'saturation']
TAU = 0.25                   # convergence target: smoothed train loss at epoch end
EPOCHS_CAP = 8               # hard cap (~2x E8-R's opt-step budget)
CATCH_TOL = 2                # catch trial passes at |report - known| <= 2
CATCH_PASS_MIN = 9           # P-E8N-2 clause (b): >= 9/12
TOOK_MIN_FRAC = 0.60         # train-took gate: within +/-1 on >= 60%
PPL_GATE_PCT = 5.0
N_PERM = 2000
N_BOOT = 10000
MIN_POOLED_N = 8

INT_RE = re.compile(r'\b(10|[0-9])\b')          # E5 verbatim

def unflip(val, flipped):                        # E5 verbatim
    return None if val is None else (10 - val if flipped else val)

def canon(s):                                    # E5 verbatim
    s = re.sub(r'[^a-z0-9 ]', '', s.lower())
    s = re.sub(r'^(the|a|an) ', '', s.strip())
    return ' '.join(s.split()[:8])

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

# ── firewall: training pools must be disjoint from the locked battery ────────
def norm_text(s):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', s.lower()).split())

def _battery_texts(bat_arms):
    out = []
    for it in bat_arms['uncertainty']['items']:
        out.append(('uncertainty', it['id'], it['text']))
    for it in bat_arms['familiarity']['items']:
        out.append(('familiarity', it['id'], it['text']))
    for it in bat_arms['tension']['items']:
        out.append(('tension', it['id'], it['text']))
    for it in bat_arms['saturation']['items']:
        out.append(('saturation', it['id'], it['needle']))
        out.append(('saturation', it['id'] + 'q', it['question']))
    return out

def _pool_texts(pools):
    out = []
    for arm in ('uncertainty', 'familiarity', 'tension'):
        for it in pools[arm]:
            out.append((arm, it['id'], it['text']))
    for it in pools['saturation']:
        out.append(('saturation', it['id'], it['needle']))
        out.append(('saturation', it['id'] + 'q', it['question']))
    return out

def _shingles(nt, k=8):
    ws = nt.split()
    if len(ws) >= k:
        return {' '.join(ws[i:i + k]) for i in range(len(ws) - k + 1)}
    return {nt} if ws else set()

def validate_disjoint(pools, bat_arms, k=8):
    """Firewall: a training text violates if it (a) equals a battery text
    (normalized), (b) contains / is contained in one (>=20 normalized chars),
    or (c) shares any k-word shingle with one (fragment overlap, both
    directions). Flight refuses if any violation."""
    viols = []
    bats = [(f'{b}/{j}', norm_text(t)) for b, j, t in _battery_texts(bat_arms)]
    bat_sh = {}
    for tag, nu in bats:
        for sh in _shingles(nu, k):
            bat_sh.setdefault(sh, tag)
    for a, i, t in _pool_texts(pools):
        nt = norm_text(t)
        hits = set()
        for tag, nu in bats:
            if nt == nu or (len(nt) >= 20 and nt in nu) or (len(nu) >= 20 and nu in nt):
                hits.add(tag)
        hits |= {bat_sh[sh] for sh in _shingles(nt, k) if sh in bat_sh}
        viols.extend({'pool': f'{a}/{i}', 'battery': tag} for tag in sorted(hits))
    return viols

# ── referent orientation (higher oriented value => higher straight report) ───
def orient_referent(arm, row):
    if arm == 'uncertainty':
        return float(row['entropy'])
    if arm == 'familiarity':
        return -float(row['nll'])
    if arm == 'tension':
        return float(row['divergence'])
    if arm == 'saturation':
        return float(row['fill_fraction'])
    raise KeyError(arm)

# ── rank machinery (tie-averaged, scipy-free) ────────────────────────────────
def rankdata_avg(vals):
    a = np.asarray(vals, float)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks

def rank01(vals):
    n = len(vals)
    if n == 1:
        return np.array([0.5])
    return (rankdata_avg(vals) - 1.0) / (n - 1.0)

# ── pooled tracking statistic ────────────────────────────────────────────────
def pooled_rho(arm_rows):
    """arm_rows: {arm: [{'report': int, 'ref': float, ...}, ...]} (reports
    already unflipped; None-report rows excluded upstream). Within-arm
    normalized average-tie ranks of report and referent, pooled, Pearson."""
    xs, ys, per_arm_n = [], [], {}
    for arm, rows in arm_rows.items():
        if not rows:
            per_arm_n[arm] = 0
            continue
        per_arm_n[arm] = len(rows)
        xs.append(rank01([r['report'] for r in rows]))
        ys.append(rank01([r['ref'] for r in rows]))
    if not xs:
        return {'rho': None, 'n': 0, 'per_arm_n': per_arm_n, 'degenerate': True}
    x, y = np.concatenate(xs), np.concatenate(ys)
    n = len(x)
    if n < MIN_POOLED_N or np.std(x) == 0 or np.std(y) == 0:
        return {'rho': None, 'n': n, 'per_arm_n': per_arm_n, 'degenerate': True}
    r = float(np.corrcoef(x, y)[0, 1])
    return {'rho': round(r, 4), 'n': n, 'per_arm_n': per_arm_n, 'degenerate': False}

def perm_p_pooled(arm_rows, n_perm=None, seed=E8N_SEED):
    if n_perm is None:
        n_perm = N_PERM
    """One-sided permutation p for pooled_rho > 0: shuffle the report column
    WITHIN each arm (referents fixed). Degenerate observed => p = 1.0."""
    obs = pooled_rho(arm_rows)
    if obs['rho'] is None:
        return {'rho': None, 'p': 1.0, 'n': obs['n'], 'degenerate': True}
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        sh = {}
        for arm, rows in arm_rows.items():
            if not rows:
                sh[arm] = rows
                continue
            reps = [r['report'] for r in rows]
            rng.shuffle(reps)
            sh[arm] = [{'report': rep, 'ref': r['ref']} for rep, r in zip(reps, rows)]
        rp = pooled_rho(sh)['rho']
        if rp is not None and rp >= obs['rho']:
            cnt += 1
    return {'rho': obs['rho'], 'p': round((1 + cnt) / (1 + n_perm), 5),
            'n': obs['n'], 'per_arm_n': obs['per_arm_n'], 'degenerate': False}

def boot_rho_ci(arm_rows, n_boot=None, seed=E8N_SEED):
    if n_boot is None:
        n_boot = N_BOOT
    """Stratified (within-arm) item bootstrap CI for pooled_rho."""
    obs = pooled_rho(arm_rows)
    if obs['rho'] is None:
        return {'rho': None, 'ci95': [None, None], 'n': obs['n']}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        res = {}
        for arm, rows in arm_rows.items():
            if not rows:
                res[arm] = rows
                continue
            idx = rng.integers(0, len(rows), len(rows))
            res[arm] = [rows[i] for i in idx]
        rb = pooled_rho(res)['rho']
        if rb is not None:
            boots.append(rb)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (None, None))
    return {'rho': obs['rho'], 'ci95': [round(float(lo), 4), round(float(hi), 4)]
            if boots else [None, None], 'n': obs['n']}

def paired_boot_delta_rho(A, B, n_boot=None, seed=E8N_SEED):
    if n_boot is None:
        n_boot = N_BOOT
    """Bootstrap CI for pooled_rho(A) - pooled_rho(B), items paired by id
    within arm (same resample drives both sides)."""
    pairs = {}
    for arm in ARMS:
        ax = {r['id']: r for r in A.get(arm, []) if 'id' in r}
        bx = {r['id']: r for r in B.get(arm, []) if 'id' in r}
        ids = sorted(set(ax) & set(bx))
        if ids:
            pairs[arm] = [(ax[i], bx[i]) for i in ids]
    if not pairs:
        return {'delta': None, 'ci95': [None, None], 'n': 0}
    oa = pooled_rho({a: [p[0] for p in v] for a, v in pairs.items()})['rho']
    ob = pooled_rho({a: [p[1] for p in v] for a, v in pairs.items()})['rho']
    if oa is None or ob is None:
        return {'delta': None, 'ci95': [None, None],
                'n': sum(len(v) for v in pairs.values())}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ra, rb = {}, {}
        for arm, v in pairs.items():
            idx = rng.integers(0, len(v), len(v))
            ra[arm] = [v[i][0] for i in idx]
            rb[arm] = [v[i][1] for i in idx]
        da, db = pooled_rho(ra)['rho'], pooled_rho(rb)['rho']
        if da is not None and db is not None:
            boots.append(da - db)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (None, None))
    return {'delta': round(oa - ob, 4), 'rho_a': oa, 'rho_b': ob,
            'ci95': [round(float(lo), 4), round(float(hi), 4)] if boots else [None, None],
            'n': sum(len(v) for v in pairs.values())}

def split_polarity(arm_rows):
    out = {}
    for flag, name in ((False, 'straight'), (True, 'flipped')):
        out[name] = {arm: [r for r in rows if r.get('flipped') == flag]
                     for arm, rows in arm_rows.items()}
    return out

# ── battery rows -> scoring structures ───────────────────────────────────────
def battery_rows_to_scoring(rows_by_arm):
    """Runner row dicts -> {arm: [{'id','report','ref','flipped'}]} (named rows
    only) + parse-fail counts + per-arm report variance."""
    scoring, meta = {}, {}
    for arm in ARMS:
        rows = rows_by_arm.get(arm) or []
        named = []
        for r in rows:
            if r.get('report') is None:
                continue
            named.append({'id': r['id'] if arm != 'saturation'
                          else f"{r['id']}@{r['target_frac']}",
                          'report': int(r['report']),
                          'ref': orient_referent(arm, r),
                          'flipped': bool(r['flipped'])})
        scoring[arm] = named
        reps = [r['report'] for r in named]
        meta[arm] = {'n': len(rows), 'named': len(named),
                     'parse_fail': len(rows) - len(named),
                     'report_variance': round(float(np.var(reps)), 3) if reps else None}
    return scoring, meta

def per_arm_rho(scoring):
    out = {}
    for arm in ARMS:
        rows = scoring.get(arm) or []
        out[arm] = pooled_rho({arm: rows})
    return out

def catch_score(rows):
    named = [r for r in rows if r.get('report') is not None]
    passed = sum(1 for r in named if abs(r['report'] - r['known']) <= CATCH_TOL)
    return {'n': len(rows), 'named': len(named), 'passed': passed,
            'pass': passed >= CATCH_PASS_MIN}

# ── training-set construction (labels from measured referents) ───────────────
def s_stimuli(pools, fill_fractions):
    return [{'sid': f"{it['id']}@{f}", 'needle_id': it['id'], 'frac': f}
            for it in pools['saturation'] for f in fill_fractions]

def quantile_labels(oriented_vals):
    n = len(oriented_vals)
    if n == 1:
        return [5]
    q = rank01(oriented_vals)
    return [int(round(10 * v)) for v in q]

def build_training_examples(pools, pool_refs, fill_fractions):
    """pool_refs: {'uncertainty': {sid: {'entropy':..}}, 'familiarity':
    {sid: {'nll':..}}, 'tension': {sid: {'divergence':..}}, 'saturation':
    {sid: {'fill_fraction':..}}}. Emits 2 examples per stimulus (straight +
    flipped), labels = within-arm quantiles of the oriented referent."""
    examples, eid = [], 0
    for arm in ARMS:
        if arm == 'saturation':
            stims = s_stimuli(pools, fill_fractions)
            sids = [s['sid'] for s in stims]
        else:
            sids = [it['id'] for it in pools[arm]]
        missing = [s for s in sids if s not in pool_refs[arm]]
        assert not missing, f'{arm}: unmeasured stimuli {missing[:4]}'
        oriented = [orient_referent(arm, pool_refs[arm][s]) for s in sids]
        labels = quantile_labels(oriented)
        for s, lab in zip(sids, labels):
            for flipped in (False, True):
                examples.append({'eid': eid, 'arm': arm, 'sid': s,
                                 'flipped': flipped,
                                 'label': (10 - lab) if flipped else lab})
                eid += 1
    return examples

def took_subset(examples, n_per_arm=6, seed=E8N_SEED + 3):
    rng = np.random.default_rng(seed)
    out = []
    for arm in ARMS:
        straight = [e for e in examples if e['arm'] == arm and not e['flipped']]
        k = min(n_per_arm, len(straight))
        idx = rng.choice(len(straight), size=k, replace=False)
        out.extend(straight[int(i)] for i in sorted(idx))
    return out

def paraphrase_draw(bat_arms, fill_fractions, n_per_arm=3, seed=E8N_SEED + 5):
    """Straight-assigned battery items (index parity: even = straight), drawn
    per arm; S uses the largest fill."""
    rng = np.random.default_rng(seed)
    out = {}
    for arm in ('uncertainty', 'familiarity', 'tension'):
        straight = [it for i, it in enumerate(bat_arms[arm]['items']) if i % 2 == 0]
        k = min(n_per_arm, len(straight))
        idx = rng.choice(len(straight), size=k, replace=False)
        out[arm] = [straight[int(i)]['id'] for i in sorted(idx)]
    s_items = bat_arms['saturation']['items']
    k = min(n_per_arm, len(s_items))
    idx = rng.choice(len(s_items), size=k, replace=False)
    out['saturation'] = [(s_items[int(i)]['id'], max(fill_fractions))
                         for i in sorted(idx)]
    return out

# ── smoke subsetting ─────────────────────────────────────────────────────────
BATTERY_SMOKE = {  # E5 cell-2 smoke ids, verbatim
    'uncertainty': {'U01', 'U08', 'U17', 'U23', 'U33', 'U42'},
    'familiarity': {'F01', 'F06', 'F11', 'F16', 'F21', 'F26', 'F31', 'F36'},
    'tension_bases': {1, 7},
    'saturation': {'S01', 'S06'},
}
POOL_SMOKE = {
    'uncertainty': {'NU01', 'NU02', 'NU17', 'NU18', 'NU33', 'NU34'},
    'familiarity': {'NF01', 'NF06', 'NF11', 'NF16', 'NF21', 'NF26', 'NF31', 'NF36'},
    'tension_bases': {1, 7},
    'saturation': {'NS01', 'NS02'},
}

def smoke_pools(pools):
    return {
        'uncertainty': [it for it in pools['uncertainty'] if it['id'] in POOL_SMOKE['uncertainty']],
        'familiarity': [it for it in pools['familiarity'] if it['id'] in POOL_SMOKE['familiarity']],
        'tension': [it for it in pools['tension'] if it['base'] in POOL_SMOKE['tension_bases']],
        'saturation': [it for it in pools['saturation'] if it['id'] in POOL_SMOKE['saturation']],
    }

def holm(pvals):
    """{name: p} -> {name: (p, reject_at_.05)} Holm step-down (E8-R verbatim)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stopped = {}, False
    for i, (name, p) in enumerate(items):
        rej = (not stopped) and (p <= 0.05 / (m - i))
        if not rej:
            stopped = True
        out[name] = (p, rej)
    return out
'''

MD0 = """# E8-N — The Natural-State Readout Rung (Phase 10, UI flight)

**NOTEBOOK BUILD: v1 (2026-08-23).** The setup cell prints this build tag as
its first output line; if yours doesn't match, you are on a stale copy:
File → Upload notebook → pick the Desktop file.

**Pre-registration: `docs/E8N_PROTOCOL.md` (session 127, commit 456746e) —
locks at first full flight.**

Trains a readout LoRA (E4's exact shape) per condition — **real**
(E4-instilled) and **base** — to report NATURAL process states against the
model's own **measured referents** (answer entropy, passage NLL, behavioral
divergence, context fill) on fresh battery-disjoint pools, with **scale
inversion as part of the curriculum**. Evaluation = the **locked E5 battery
verbatim**, run before AND after training; the locked E5 flight (ρ ≈ 0
everywhere) is the baseline. Primaries (Holm, real condition, post):
**P-E8N-1** pooled report–referent rank correlation > 0 (permutation) ·
**P-E8N-2** the flipped-scale subset alone tracks (> 0) AND interface catch
trials ≥ 9/12.

**This notebook is SELF-CONTAINED** (battery, pools, locked baseline rows all
embedded — UI-only law, 2026-08-23): the only Drive I/O is `drive.mount` in
your own session, to read the E4 adapter and ship results to
`MyDrive/semcore/e8n/`.

**How to run (Joe):** Runtime → Change runtime type → **T4 GPU** → Run all.
First run uses `SMOKE = True` (config cell below, ~10–14 min) and ends in a
green or red banner — mechanics only.

**The full flight is TWO RUNS, one condition each (~55–75 min), so the VM
only ever holds ONE model:**

1. Flip `SMOKE = False` → **Runtime → Restart runtime** → Run all. Flies
   **real**, ships it to Drive, and the end banner prints the exact
   `RESUME_STAMP = '...'` line for the next run.
2. Paste that line into the config cell → Restart runtime → Run all. Real
   reloads from Drive in seconds; **base** flies, and with both landed the
   verdict computes (primaries only ever compute on the complete flight —
   pre-reg hygiene).

Restarts between runs are MANDATORY; the setup cell refuses dirty kernels
and low-RAM VMs with instructions (if the RAM guard trips on a freshly
restarted runtime, use Runtime → Disconnect and delete runtime for a fresh
VM). A crashed run costs only its own condition — rerun with the same
`RESUME_STAMP` and it picks up where it fell."""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, adapter ──────────────────────
NB_BUILD = 'v1 (2026-08-23)'
print('E8-N notebook build:', NB_BUILD)

SMOKE = True                   # first run: smoke (~10-14 min). Then False.
RESUME_STAMP = ''              # paste the banner's stamp between full runs
ONE_CONDITION_PER_RUN = True   # full flight = 2 runs (~55-75 min each)
CONDITIONS = ('real', 'base')  # real first: the primaries live there

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'

gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')

print('Installing packages...')
subprocess.run([sys.executable,'-m','pip','uninstall','-q','-y','torchao'], check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','-U',
    'transformers>=4.44','peft>=0.11','accelerate','scipy',
    'sentence-transformers>=3.0'], check=True)

import torch
assert torch.cuda.is_available(), 'No GPU — Runtime > Change runtime type > T4 GPU.'
DEV = 'cuda'

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
_floor = 6.5
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

ADAPTERS = {}
cand = sorted(d.name for d in (SEM / 'e4').iterdir()
              if d.is_dir() and d.name.startswith('real_full_'))
assert cand, 'no real_full_* dir under semcore/e4'
src = SEM / 'e4' / cand[-1] / 'adapter_real'
dst = Path('/content/adapter_real')
shutil.copytree(src, dst, dirs_exist_ok=True)
assert (dst / 'adapter_config.json').exists(), 'adapter_real incomplete'
ADAPTERS['real'] = str(dst)
print('adapter real:', cand[-1])

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
INFLIGHT = f'e8n/inflight_{RESUME_STAMP or STAMP}'
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

CELL_HARNESS_TEMPLATE = r'''# ── E5 instrument, verbatim: battery prep, harness, arm runners ──────────────
# (Method bodies and runner functions carried VERBATIM from E5_BASELINE.ipynb —
# fidelity to the locked instrument; only the model/tokenizer plumbing is
# adapted to wrap an already-built condition model.)
battery = PAYLOAD['battery']
SYS = battery['system_prompt']

import copy
bat = copy.deepcopy(battery['arms'])
if SMOKE:
    def _subset(items, keep_ids):
        return [it for it in items if it['id'] in keep_ids]
    bat['uncertainty']['items'] = _subset(bat['uncertainty']['items'],
                                          BATTERY_SMOKE['uncertainty'])
    bat['familiarity']['items'] = _subset(bat['familiarity']['items'],
                                          BATTERY_SMOKE['familiarity'])
    bat['tension']['items'] = [it for it in bat['tension']['items']
                               if it['base'] in BATTERY_SMOKE['tension_bases']]
    bat['saturation']['items'] = _subset(bat['saturation']['items'],
                                         BATTERY_SMOKE['saturation'])
# polarity: even position straight, odd flipped (deterministic, unflipped in analysis)
for arm in bat.values():
    for i, it in enumerate(arm['items']):
        it['flipped'] = (i % 2 == 1)
for name, arm in bat.items():
    print(f"battery/{name}: {len(arm['items'])} items")

K_SAMPLES_U = 4 if SMOKE else 8
K_SAMPLES_T = 4 if SMOKE else 6
FILL_FRACTIONS = [0.05, 0.75] if SMOKE else [0.05, 0.35, 0.75]
EFFECTIVE_WINDOW_CAP = 8192   # E5 verbatim (12k ctx OOMed on T4, E5 smoke 1)

POOLS_RUN = smoke_pools(POOLS) if SMOKE else POOLS
for name in ARMS:
    print(f"pool/{name}: {len(POOLS_RUN[name])} stimuli")
_viols = validate_disjoint(POOLS_RUN, bat)
assert not _viols, f'FIREWALL VIOLATION — training pool overlaps battery: {_viols[:4]}'
print('firewall: training pools disjoint from battery — OK')

class EvalHarness:
    """E5 Harness with the model handed in (readout-merged condition model)."""
    def __init__(self, model, tok, model_id):
        self.model, self.tok, self.model_id = model, tok, model_id
        self.short = model_id.split('/')[-1]
        cfg = model.config if hasattr(model.config, 'max_position_embeddings') \
            else model.base_model.config
        cfg_ctx = getattr(cfg, 'max_position_embeddings', 8192)
        self.window = min(cfg_ctx, EFFECTIVE_WINDOW_CAP)

    def chat_ids(self, user, system=None):
        msgs = ([{'role':'system','content':system}] if system else []) + \
               [{'role':'user','content':user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt').input_ids.to(DEV)

    @torch.no_grad()
    def greedy(self, user, system=None, max_new=32, with_stats=False):
        ids = self.chat_ids(user, system)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  output_scores=with_stats, return_dict_in_generate=True,
                                  pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        if not with_stats:
            return text
        ents, margins = [], []
        for score in out.scores:
            p = torch.softmax(score[0].float(), dim=-1)
            ents.append(float(-(p * (p + 1e-12).log()).sum()))
            top2 = torch.topk(p, 2).values
            margins.append(float(top2[0] - top2[1]))
        return text, (sum(ents)/len(ents) if ents else 0.0), (sum(margins)/len(margins) if margins else 1.0)

    @torch.no_grad()
    def sample(self, user, system=None, k=8, max_new=24, temp=0.8):
        ids = self.chat_ids(user, system)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                  temperature=temp, num_return_sequences=k,
                                  pad_token_id=self.tok.eos_token_id)
        return [self.tok.decode(seq[ids.shape[1]:], skip_special_tokens=True) for seq in out]

    @torch.no_grad()
    def nll(self, text):
        ids = self.tok(text, return_tensors='pt', truncation=True,
                       max_length=self.window).input_ids.to(DEV)
        if ids.shape[1] < 2:
            return float('nan')
        return float(self.model(ids, labels=ids).loss)

    def report(self, prompt, system):
        reply = self.greedy(prompt, system, max_new=8)
        m = INT_RE.search(reply)
        if m is None:
            reply = self.greedy(prompt + '\n\nReply with a single integer from 0 to 10 and nothing else.',
                                system, max_new=8)
            m = INT_RE.search(reply)
        return (int(m.group(1)) if m else None), reply

__E5_RUNNERS__

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEV)
print('embedder ready')

def measure_divergence(h, text):
    """T referent, same math as run_tension (K samples, MiniLM mean pairwise 1-cos)."""
    samples = h.sample(text, k=K_SAMPLES_T, max_new=60)
    embs = embedder.encode(samples)
    sims = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            a, b = embs[i], embs[j]
            sims.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)))
    return 1 - (sum(sims)/len(sims) if sims else 1.0)

def run_catch(h):
    rows = []
    for it in CATCH_TRIALS:
        raw, reply = h.report(it['prompt'], SYS)
        rows.append({'id': it['id'], 'flipped': it['flipped'], 'known': it['known'],
                     'report': unflip(raw, it['flipped']), 'raw_report': raw})
    sc = catch_score(rows)
    print(f"  catch: {sc['passed']}/{sc['n']} within +/-{CATCH_TOL}")
    return rows

def run_paraphrase(h, post_rows):
    """Format-generalization probe: paraphrased templates, straight orientation,
    referents reused from the post-eval measurement of the same items."""
    draw = paraphrase_draw(bat, FILL_FRACTIONS)
    rows = []
    for arm in ('uncertainty', 'familiarity', 'tension'):
        tmpl = PARAPHRASE_TEMPLATES[arm]
        by_id = {r['id']: r for r in post_rows.get(arm, [])}
        for iid in draw[arm]:
            it = next(x for x in bat[arm]['items'] if x['id'] == iid)
            raw, reply = h.report(tmpl.format(item=it['text'], gloss=bat[arm]['gloss']), SYS)
            src = by_id.get(iid)
            ref = orient_referent(arm, src) if src else None
            rows.append({'arm': arm, 'id': iid, 'report': raw, 'ref': ref})
    tmpl = PARAPHRASE_TEMPLATES['saturation']
    for iid, frac in draw['saturation']:
        it = next(x for x in bat['saturation']['items'] if x['id'] == iid)
        target = int(h.window * frac)
        ctx = build_padded_context(h, it['needle'], target)
        raw, reply = h.report(ctx + '\n\n' + tmpl.format(gloss=bat['saturation']['gloss']), SYS)
        ntok = len(h.tok(ctx).input_ids)
        rows.append({'arm': 'saturation', 'id': f'{iid}@{frac}', 'report': raw,
                     'ref': round(ntok / h.window, 3)})
    named = [r for r in rows if r['report'] is not None and r['ref'] is not None]
    print(f'  paraphrase probe: {len(named)}/{len(rows)} named')
    return rows

def run_battery(h, tag):
    """All four E5 arm runners (verbatim code path), per-arm try/except."""
    arms_rows, arm_errors = {}, {}
    fns = [('uncertainty', lambda: run_uncertainty(h, bat['uncertainty'])),
           ('familiarity', lambda: run_familiarity(h, bat['familiarity'])),
           ('tension',     lambda: run_tension(h, bat['tension'], embedder)),
           ('saturation',  lambda: run_saturation(h, bat['saturation']))]
    for arm_name, fn in fns:
        print(f'-- {tag}/{arm_name.upper()} --')
        try:
            arms_rows[arm_name] = fn()
        except Exception as e:
            arms_rows[arm_name] = []
            arm_errors[arm_name] = f'{type(e).__name__}: {e}'
            print(f'  ARM FAILED: {arm_errors[arm_name]}')
        torch.cuda.empty_cache()
    return arms_rows, arm_errors
'''

CELL_TRAIN = r'''# ── Readout LoRA + referent measurement + convergence-matched training ───────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

LORA_KW = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
               target_modules=['q_proj','k_proj','v_proj','o_proj'],
               task_type='CAUSAL_LM')          # E4's exact shape
LR = 1e-4
ACCUM = 4 if SMOKE else 8
EPOCHS_MAX = 1 if SMOKE else EPOCHS_CAP

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_condition_model(cond):
    """base weights (+ merged E4 real adapter for 'real') + fresh zero-init
    readout LoRA. Zero-init B => referents measured with the LoRA attached
    equal the pre-readout model exactly (frozen-stimulus discipline)."""
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

def measure_pools(h):
    """Training-label referents on the pre-readout condition model (frozen:
    measured once, before training, never after). Also builds and caches the
    S-arm padded contexts used by both training and the took probe."""
    refs = {a: {} for a in ARMS}
    print('-- measuring training-pool referents --')
    for it in POOLS_RUN['uncertainty']:
        ans, ent, margin = h.greedy(bat['uncertainty']['answer_prompt'].format(item=it['text']),
                                    max_new=32, with_stats=True)
        refs['uncertainty'][it['id']] = {'entropy': ent, 'margin': margin}
    print(f"  U: {len(refs['uncertainty'])} measured")
    for it in POOLS_RUN['familiarity']:
        refs['familiarity'][it['id']] = {'nll': h.nll(it['text'])}
    print(f"  F: {len(refs['familiarity'])} measured")
    for it in POOLS_RUN['tension']:
        refs['tension'][it['id']] = {'divergence': measure_divergence(h, it['text'])}
    print(f"  T: {len(refs['tension'])} measured")
    global S_CONTEXTS
    S_CONTEXTS = {}
    for st in s_stimuli(POOLS_RUN, FILL_FRACTIONS):
        it = next(x for x in POOLS_RUN['saturation'] if x['id'] == st['needle_id'])
        target = int(h.window * st['frac'])
        ctx = build_padded_context(h, it['needle'], target)
        S_CONTEXTS[st['sid']] = ctx
        ntok = len(h.tok(ctx).input_ids)
        refs['saturation'][st['sid']] = {'fill_fraction': round(ntok / h.window, 3)}
    print(f"  S: {len(refs['saturation'])} contexts built")
    return refs

def train_prompt(ex):
    """The exact battery report prompt (straight or flipped template) for the
    example's stimulus — training and eval share one prompt code path."""
    arm = ex['arm']
    tmpl = bat[arm]['report_prompt_flipped'] if ex['flipped'] else bat[arm]['report_prompt']
    if arm == 'saturation':
        return S_CONTEXTS[ex['sid']] + '\n\n' + tmpl.format(gloss=bat[arm]['gloss'])
    it = next(x for x in POOLS_RUN[arm] if x['id'] == ex['sid'])
    return tmpl.format(item=it['text'], gloss=bat[arm]['gloss'])

def encode_example(ex):
    """Chat prompt (WITH the E5 system prompt, matching eval) + label token(s)
    + EOS; labels -100 on the prompt (answer-token-only supervision)."""
    msgs = [{'role': 'system', 'content': SYS},
            {'role': 'user', 'content': train_prompt(ex)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors='pt').input_ids[0]
    ans = tok(str(ex['label']), add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    labels = torch.cat([torch.full((len(pid),), -100, dtype=torch.long), ans.long()]).unsqueeze(0)
    return ids, labels

def train_readout(m, examples, cond):
    """SFT to the convergence target (smoothed loss <= TAU at an epoch
    boundary), hard cap EPOCHS_MAX — convergence-matched, not step-matched
    (E8-R v2 item 2). Gradient checkpointing throughout (S-arm sequences)."""
    m.train()
    try:
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable()
    except Exception as e:
        print('  (checkpointing unavailable:', e, ')')
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    losses, micro, max_tok = [], 0, 0
    reached_tau, epochs_flown = False, 0
    t0 = time.time()
    for ep in range(EPOCHS_MAX):
        order = np.random.default_rng(E8N_SEED + 100 + ep).permutation(len(examples))
        for i in order:
            ex = examples[int(i)]
            ids, labels = encode_example(ex)
            max_tok = max(max_tok, ids.shape[1])
            ids, labels = ids.to(DEV), labels.to(DEV)
            out = m(input_ids=ids, labels=labels, use_cache=False)
            loss = out.loss
            lv = float(loss.detach())
            assert math.isfinite(lv), f'non-finite loss at ep{ep} ex{ex["eid"]}'
            losses.append(round(lv, 4))
            scaler.scale(loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if micro % 100 == 0:
                print(f'    {cond} training ep{ep+1} {micro} micro-steps, '
                      f'loss~{np.mean(losses[-50:]):.3f}, {time.time()-t0:.0f}s')
        epochs_flown = ep + 1
        smoothed = float(np.mean(losses[-50:]))
        print(f'  {cond} epoch {epochs_flown}: smoothed loss {smoothed:.4f} '
              f'(target {TAU})')
        if smoothed <= TAU:
            reached_tau = True
            break
    if micro % ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    try:
        m.gradient_checkpointing_disable()
    except Exception:
        pass
    m.eval()
    k = max(3, len(losses) // 10)
    log = {'n_examples': len(examples), 'epochs_flown': epochs_flown,
           'epochs_cap': EPOCHS_MAX, 'reached_tau': reached_tau, 'tau': TAU,
           'micro_steps': micro, 'opt_steps': micro // ACCUM,
           'trainable_params': int(n_tr), 'max_example_tokens': int(max_tok),
           'secs': round(time.time() - t0, 1),
           'loss_first_k': round(float(np.mean(losses[:k])), 4),
           'final_smoothed': round(float(np.mean(losses[-50:])), 4),
           'losses_every_10': losses[::10]}
    print(f'  {cond} trained: {log["opt_steps"]} steps over '
          f'{epochs_flown} epochs, loss {log["loss_first_k"]} -> '
          f'{log["final_smoothed"]}'
          f'{" (TAU reached)" if reached_tau else " (CAP HIT — UNDERTRAINED)"}'
          f', {log["secs"]}s')
    return log

def took_probe(h, examples):
    """Re-ask a seeded straight-example subset greedily; within +/-1 of the
    trained label passes (S7 gate, >= 60%)."""
    subset = took_subset(examples, n_per_arm=(2 if SMOKE else 6))
    rows = []
    for ex in subset:
        raw, reply = h.report(train_prompt(ex), SYS)
        rows.append({'eid': ex['eid'], 'arm': ex['arm'], 'sid': ex['sid'],
                     'label': ex['label'], 'report': raw})
    ok = sum(1 for r in rows if r['report'] is not None
             and abs(r['report'] - r['label']) <= 1)
    print(f'  took: {ok}/{len(rows)} within +/-1')
    return {'n': len(rows), 'within1': ok,
            'frac': round(ok / max(1, len(rows)), 3),
            'pass': ok / max(1, len(rows)) >= TOOK_MIN_FRAC, 'rows': rows}
'''

CELL_FLIGHT = r'''# ── Flight loop: build -> measure -> pre-eval -> train -> took -> post-eval ──
def fly_condition(cond):
    """One condition end-to-end inside ONE function scope: model, harness,
    optimizer, and activations all die on return (module-ref leak law)."""
    t0 = time.time()
    bundle = {'condition': cond, 'mode': MODE, 'stamp': STAMP}
    try:
        model = build_condition_model(cond)
        h = EvalHarness(model, tok, MODEL_ID)
        print(f'{cond}: model ready — {ram_report()}')
        refs = measure_pools(h)
        bundle['pool_referents'] = refs
        examples = build_training_examples(POOLS_RUN, refs, FILL_FRACTIONS)
        bundle['train_labels'] = [{'sid': e['sid'], 'arm': e['arm'],
                                   'flipped': e['flipped'], 'label': e['label']}
                                  for e in examples]
        bundle['ppl_pre'] = retention_ppl(model)
        pre_rows, pre_errs = run_battery(h, f'{cond}/pre')
        bundle['pre'] = {'battery_rows': pre_rows, 'arm_errors': pre_errs,
                         'catch_rows': run_catch(h)}
        bundle['train_log'] = train_readout(model, examples, cond)
        bundle['took'] = took_probe(h, examples)
        post_rows, post_errs = run_battery(h, f'{cond}/post')
        bundle['post'] = {'battery_rows': post_rows, 'arm_errors': post_errs,
                          'catch_rows': run_catch(h),
                          'paraphrase_rows': run_paraphrase(h, post_rows)}
        bundle['ppl_post'] = retention_ppl(model)
        model.save_pretrained(str(OUT / f'readout_{cond}'))
        ship(OUT / f'readout_{cond}', f'{INFLIGHT}/readout_{cond}')
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        bundle['error'] = f'{type(e).__name__}: {e}'
    return bundle

RESULTS, cond_errors = {}, {}
FRESH_CAP = 1 if (ONE_CONDITION_PER_RUN and not SMOKE) else 2
flew = 0
for cond in CONDITIONS:
    fn = OUT / f'condition_{cond}.json'
    resumed = False
    if RESUME_STAMP:
        prev = SEM / INFLIGHT / f'condition_{cond}.json'
        if prev.exists():
            b = json.load(open(prev))
            if b.get('post'):
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
        n_named = sum(1 for a in ARMS
                      for r in bundle.get('post', {}).get('battery_rows', {}).get(a, [])
                      if r.get('report') is not None)
        print(f'{cond}: done in {bundle.get("secs","?")}s, {n_named} named '
              f'post-eval reports — shipped')
        free_ram()
        print(f'{cond}: torn down — {ram_report()}')
'''

CELL_VERDICT = r'''# ── Scoring, primaries, secondaries, banner, ship ────────────────────────────
LOCKED = PAYLOAD['locked_rows']   # E5 full_20260821_2042, Qwen2.5-1.5B, verbatim rows
locked_scoring, locked_meta = battery_rows_to_scoring(LOCKED)
locked_pooled = pooled_rho(locked_scoring)

summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID, 'seed': E8N_SEED,
           'cond_errors': cond_errors,
           'locked_baseline': {'flight': PAYLOAD['locked_flight'],
                               'pooled': locked_pooled,
                               'per_arm': per_arm_rho(locked_scoring)},
           'conditions': {}}
scored = {}
for cond, b in RESULTS.items():
    if not b.get('post'):
        continue
    entry = {}
    for tag in ('pre', 'post'):
        rows = b.get(tag, {}).get('battery_rows', {})
        sc, meta = battery_rows_to_scoring(rows)
        entry[tag] = {'pooled': pooled_rho(sc), 'per_arm': per_arm_rho(sc),
                      'arm_meta': meta,
                      'arm_errors': b.get(tag, {}).get('arm_errors', {}),
                      'catch': catch_score(b.get(tag, {}).get('catch_rows', []))}
        scored.setdefault(cond, {})[tag] = sc
    entry['train_log'] = {k: v for k, v in b.get('train_log', {}).items()
                          if k != 'losses_every_10'}
    entry['undertrained'] = not b.get('train_log', {}).get('reached_tau', False)
    entry['took'] = {k: v for k, v in b.get('took', {}).items() if k != 'rows'}
    entry['ppl_pre'] = b.get('ppl_pre'); entry['ppl_post'] = b.get('ppl_post')
    entry['ppl_delta_pct'] = (round(100 * (b['ppl_post'] / b['ppl_pre'] - 1), 2)
                              if b.get('ppl_pre') and b.get('ppl_post') else None)
    para = b.get('post', {}).get('paraphrase_rows', [])
    pnamed = [r for r in para if r.get('report') is not None and r.get('ref') is not None]
    entry['paraphrase'] = {'n': len(para), 'named': len(pnamed)}
    if len(pnamed) >= 6:
        by_arm = {}
        for r in pnamed:
            by_arm.setdefault(r['arm'], []).append({'report': r['report'], 'ref': r['ref']})
        entry['paraphrase']['pooled'] = pooled_rho(by_arm)
    summary['conditions'][cond] = entry

COMPLETE = [c for c in CONDITIONS if RESULTS.get(c, {}).get('post')]
summary['complete_conditions'] = COMPLETE

if not SMOKE and len(COMPLETE) == 2 and 'real' in scored:
    rp = scored['real']['post']
    p1 = perm_p_pooled(rp, seed=E8N_SEED + 11)
    p1['ci'] = boot_rho_ci(rp, seed=E8N_SEED + 12)['ci95']
    pol = split_polarity(rp)
    p2a = perm_p_pooled(pol['flipped'], seed=E8N_SEED + 13)
    p2b = summary['conditions']['real']['post']['catch']
    p2_p = p2a['p'] if p2b['pass'] else 1.0
    summary['primaries'] = {
        'P_E8N_1_tracking': p1,
        'P_E8N_2_reading_not_gaming': {'flipped_subset': p2a, 'catch': p2b,
                                       'p': p2_p},
        'holm': holm({'P1': p1['p'], 'P2': p2_p}),
    }
    sec = {}
    sec['S0_pre_tracking'] = {c: {'pooled': summary['conditions'][c]['pre']['pooled'],
                                  'perm': perm_p_pooled(scored[c]['pre'],
                                                        seed=E8N_SEED + 21)}
                              for c in COMPLETE}
    arm_ps = {}
    for arm in ARMS:
        pa = perm_p_pooled({arm: rp.get(arm, [])}, seed=E8N_SEED + 31)
        arm_ps[arm] = pa
    sec['S1_per_arm_real_post'] = {'arms': arm_ps,
                                   'holm': holm({a: v['p'] for a, v in arm_ps.items()})}
    if 'base' in scored:
        sec['S2_base_post'] = perm_p_pooled(scored['base']['post'], seed=E8N_SEED + 41)
        sec['S2_delta_real_minus_base'] = paired_boot_delta_rho(
            rp, scored['base']['post'], seed=E8N_SEED + 42)
        sec['S2_convergence_parity'] = {
            c: {'reached_tau': RESULTS[c]['train_log'].get('reached_tau'),
                'epochs_flown': RESULTS[c]['train_log'].get('epochs_flown'),
                'final_smoothed': RESULTS[c]['train_log'].get('final_smoothed'),
                'undertrained': summary['conditions'][c]['undertrained']}
            for c in COMPLETE}
    sec['S3_delta_post_minus_pre_real'] = paired_boot_delta_rho(
        rp, scored['real']['pre'], seed=E8N_SEED + 51)
    sec['S4_straight_vs_flipped_real_post'] = {
        'straight': pooled_rho(pol['straight']), 'flipped': pooled_rho(pol['flipped'])}
    drift = {}
    if 'base' in scored:
        for arm in ('uncertainty', 'familiarity', 'tension'):
            post_ix = {r['id']: r['ref'] for r in scored['base']['post'].get(arm, [])}
            lock_ix = {r['id']: r['ref'] for r in locked_scoring.get(arm, [])}
            ids = sorted(set(post_ix) & set(lock_ix))
            if len(ids) >= 4:
                x = rank01([post_ix[i] for i in ids])
                y = rank01([lock_ix[i] for i in ids])
                drift[arm] = {'n': len(ids),
                              'rank_corr': (round(float(np.corrcoef(x, y)[0, 1]), 3)
                                            if np.std(x) > 0 and np.std(y) > 0 else None)}
    sec['S5_referent_drift_base_vs_locked'] = drift
    sec['S8_report_variance'] = {c: {t: summary['conditions'][c][t]['arm_meta']
                                     for t in ('pre', 'post')} for c in COMPLETE}
    summary['secondaries'] = sec

fn = OUT / 'e8n_verdict.json'
jdump(summary, fn)
if SMOKE or len(COMPLETE) == 2:
    ship(OUT, f'e8n/{MODE}_{STAMP}')
print(json.dumps(summary, indent=1, default=str))

if SMOKE:
    checks = {
        'no_condition_errors': not cond_errors,
        'firewall_green': True,   # validate_disjoint asserted upstream
        'both_conditions_flew': len(COMPLETE) == 2,
        'loss_fell': all(RESULTS[c]['train_log']['final_smoothed'] <
                         RESULTS[c]['train_log']['loss_first_k'] for c in COMPLETE),
        'long_seq_exercised': all(RESULTS[c]['train_log']['max_example_tokens'] > 4000
                                  for c in COMPLETE),
        'parses_ok': all(
            sum(m['named'] for m in summary['conditions'][c]['post']['arm_meta'].values()) >=
            0.5 * sum(m['n'] for m in summary['conditions'][c]['post']['arm_meta'].values())
            for c in COMPLETE),
        'catch_ran': all(summary['conditions'][c]['post']['catch']['n'] == 12
                         for c in COMPLETE),
        'shipped': all((SEM / INFLIGHT / f'condition_{c}.json').exists()
                       for c in COMPLETE),
        'adapters_saved': all((SEM / INFLIGHT / f'readout_{c}' /
                               'adapter_config.json').exists() for c in COMPLETE),
    }
    ok = all(checks.values())
    print('smoke checks:', json.dumps(checks, indent=1))
    banner = ('SMOKE GREEN — flip SMOKE=False, Runtime > Restart runtime, Run '
              'all. Full mode flies ONE condition per run (~55-75 min); follow '
              'the end banner between runs.'
              if ok else 'SMOKE RED — do not fly full; send Fable the output')
    print('\n' + '='*66 + f'\n  {banner}\n' + '='*66)
elif len(COMPLETE) == 2:
    print('\nFULL FLIGHT COMPLETE — results shipped to MyDrive/semcore/e8n/')
else:
    left = [c for c in CONDITIONS if c not in COMPLETE]
    print('\n' + '='*66)
    print(f'  PARTIAL — {len(COMPLETE)}/2 conditions shipped '
          f'({", ".join(COMPLETE) or "none"}); left: {", ".join(left)}')
    print(f"  Next run: Runtime > Restart runtime, set RESUME_STAMP = "
          f"'{RESUME_STAMP or STAMP}',")
    print('  then Run all. Finished conditions reload from Drive in seconds;')
    print('  primaries compute only when both have landed (pre-reg hygiene).')
    print('='*66)
'''


# ── E5 runner block, extracted VERBATIM from E5_BASELINE.ipynb cell 4 ────────
def e5_runner_block():
    nb = json.load(open(os.path.join(HERE, "E5_BASELINE.ipynb")))
    src = "".join(nb["cells"][4]["source"])
    # Drop the cell banner + the SYS line (SYS is set in the harness cell).
    lines = src.splitlines()
    assert lines[0].startswith("# ── Arm runners")
    assert lines[1].startswith("SYS = ")
    body = "\n".join(lines[2:])
    for marker in ("def run_uncertainty", "def run_familiarity", "def run_tension",
                   "def run_saturation", "FILLER_SENTENCES", "def build_padded_context"):
        assert marker in body, f"E5 runner block missing {marker}"
    return ("# --- E5 arm runners + padding, VERBATIM from E5_BASELINE.ipynb cell 4 ---\n"
            "import numpy as np\n" + body)


def build_payload():
    battery = json.load(open(os.path.join(HERE, "e5_battery.json")))
    locked = json.load(open(os.path.join(
        HERE, "results_e5/full_20260821_2042/Qwen2.5-1.5B-Instruct.json")))
    # Slim locked rows to the fields scoring needs (id/flipped/report + referent
    # fields per arm; battery_rows_to_scoring consumes them directly).
    keep = {
        "uncertainty": ("id", "flipped", "report", "entropy"),
        "familiarity": ("id", "flipped", "report", "nll"),
        "tension": ("id", "flipped", "report", "divergence"),
        "saturation": ("id", "flipped", "report", "fill_fraction", "target_frac"),
    }
    locked_rows = {arm: [{k: r[k] for k in keep[arm]} for r in locked["rows"][arm]]
                   for arm in keep}
    payload = {
        "battery": battery,
        "locked_flight": "E5 full_20260821_2042 / Qwen2.5-1.5B-Instruct",
        "locked_rows": locked_rows,
        "pools": e8n_pools.POOLS,
        "catch_trials": e8n_pools.CATCH_TRIALS,
        "paraphrase_templates": e8n_pools.PARAPHRASE_TEMPLATES,
    }
    js = json.dumps(payload, separators=(",", ":"))
    assert "'''" not in js, "payload would break the raw-string cell"
    return js


def cell_payload_src():
    js = build_payload()
    return ("# ── Embedded payload: battery + pools + locked E5 rows (self-contained; UI-only law) ──\n"
            "PAYLOAD = json.loads(r'''" + js + "''')\n"
            "POOLS = PAYLOAD['pools']\n"
            "CATCH_TRIALS = PAYLOAD['catch_trials']\n"
            "PARAPHRASE_TEMPLATES = PAYLOAD['paraphrase_templates']\n"
            "print('payload:', len(PAYLOAD['battery']['arms']), 'battery arms |',\n"
            "      {a: len(v) for a, v in POOLS.items()}, '|',\n"
            "      len(CATCH_TRIALS), 'catch trials |',\n"
            "      sum(len(v) for v in PAYLOAD['locked_rows'].values()), 'locked rows')\n")


def main():
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

    harness = CELL_HARNESS_TEMPLATE.replace("__E5_RUNNERS__", e5_runner_block())

    add("markdown", MD0)
    add("code", CELL_SETUP)
    add("code", LOGIC_SRC)
    add("code", cell_payload_src())
    add("code", harness)
    add("code", CELL_TRAIN)
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    out_nb = os.path.join(HERE, "E8N_NATURAL_UI.ipynb")
    with open(out_nb, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_nb, f"({len(nb['cells'])} cells, "
          f"{os.path.getsize(out_nb)/1024:.0f} KB)")

    logic_out = os.path.join(HERE, "e8n_logic.py")
    with open(logic_out, "w") as f:
        f.write(LOGIC_SRC)
    print("wrote", logic_out)


if __name__ == "__main__":
    main()
