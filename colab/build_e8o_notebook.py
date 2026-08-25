#!/usr/bin/env python3
"""Build colab/E8O_ORDERED_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-O: ordered composition — L4 in bottleneck form, the Ifá rung
(docs/E8O_PROTOCOL.md, pre-registered e10c60f BEFORE this file existed).

EVAL-ONLY on the LOCKED E8-F readout (flight of record
e8f/inflight_20260824_2311/readout_real). No training anywhere. Stimuli =
pinned W_L3 compositions (OP-A leg-swap) + the Odù-16 rider dirs computed
on the PRE-readout stack in own-16-call shape.

Single-source law: the logic cell = e8f_logic.py BYTE-VERBATIM (it flew
twice) + the E8-O block, emitted BOTH into the notebook and into
colab/e8o_logic.py; g2_probe / run_trial_feat / Injector / stimulus
machinery sliced byte-verbatim from the E8-F builder; compose_A asserted
byte-equal to the design check's function; the E8-F payload embedded
UNCHANGED (same sha 3ff29fe916792edd).
"""
import hashlib, importlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_e8f_notebook as BF          # donor cells (side effects: rebuilds
import e8j_logic as J                    # E8-R artifacts byte-identically)

E8F_LOGIC_SRC = open(os.path.join(HERE, "e8f_logic.py")).read()
DESIGN_SRC = open(os.path.join(HERE, "e8o_design_check.py")).read()


def slice_block(src, start, end=None, tag=""):
    i = src.find(start)
    assert i >= 0, f"slice_block[{tag}]: start marker missing: {start[:50]!r}"
    if end is None:
        return src[i:]
    j = src.find(end, i)
    assert j > i, f"slice_block[{tag}]: end marker missing: {end[:50]!r}"
    return src[i:j]


COMPOSE_SRC = slice_block(DESIGN_SRC, "def compose_A", "def compose_B",
                          tag="compose")
RIDER_SRC = slice_block(DESIGN_SRC, "def span_residual", "def self_tests",
                        tag="rider")
BUILD_EVAL_MODEL_SRC = slice_block(BF.E8F_TRAIN_SRC, "def build_eval_model",
                                   "def train_readout_feat", tag="bem")

E8O_SEED = 20260900
READOUT_SRC_PATH = "e8f/inflight_20260824_2311/readout_real"   # flight of record

ODU_TEXTS = {
    "OGBE": "the open road, first light, initiative unobstructed, all channels clear and moving outward",
    "OYEKU": "the closed road, full darkness, endings accepted, rest and the dignity of what concludes",
    "IWORI": "penetrating scrutiny, fire that transforms what it examines, insight that burns through surface",
    "ODI": "the sealed vessel, containment and gestation, what is held in until its time",
    "IROSUN": "inherited weight, the ancestors pressing on the present, sleep that carries old debts",
    "OWONRIN": "sudden reversal, the world upended, chaos that rearranges what order could not",
    "OBARA": "status transformed, the humble raised and the proud brought low, abundance from a small seed",
    "OKANRAN": "the single sharp word, stubborn conflict, the one cowrie that refuses the bargain",
    "OGUNDA": "iron clearing the path, work that cuts through obstruction, the pioneer's blade",
    "OSA": "flight from the storm, sudden fear that scatters, escape as survival wisdom",
    "IKA": "the coiled serpent, malice held in check, poison studied to become antidote",
    "OTURUPON": "the borne burden, illness endured, the load carried past the body's protest",
    "OTURA": "the mystic's calm, gentle persuasion, vision that reconciles without force",
    "IRETE": "earth pressed down, resilience under suppression, defiance that outlasts burial",
    "OSE": "sweetness beside loss, the ambivalent gift, tears and abundance from one spring",
    "OFUN": "the white cloth of the elders, purity and completion, return to the source that gives",
}
ODU_NAMES = list(ODU_TEXTS)


# ═══════════════════════════════════════════════════════════════════════════
E8O_SRC_TEMPLATE = r'''# ── E8-O pure logic: the ordered operator, rows, stats (locally tested) ─────
# (builds on the e8f_logic namespace: AXIS_NAMES_F, code_levels/code_text,
#  parse_feat, feat_prompt, EVAL64/TRAIN256, MU_FRAME, PAYLOAD_SHA, holm,
#  s1_profile, s7_carrier, sham_claims, invalid_rate, levels_matrix)

from itertools import product

E8O_SEED = 20260900
ALPHAS_O = [0.5, 1.0]
N_PERM_O = 10000
P_CRIT_O = 0.0025
AX_MIN_O = 6
GID_MIN = 0.55
SHAM_MAX_O = 2
INVALID_MAX_O = 0.10
READOUT_SRC = '@@READOUT_SRC@@'

PAIR24 = @@PAIR24@@
IDENT16 = @@IDENT16@@
CONTROL16 = @@CONTROL16@@
ODU_DESC = @@ODU_DESC@@
ODU_NAMES_O = list(ODU_DESC)
PAIR24_SHA = '@@PAIR24_SHA@@'

@@COMPOSE_SRC@@

def composed_code(vA, vB):
    """Exact code semantics of OP-A: A's 7 leg-1 fields + B's 7 leg-2."""
    return code_levels(compose_A(vA, vB))

def pair_codes(vecs):
    """rowkey (pair_idx, order) -> composed code. order 0 = (A,B), 1 = (B,A)."""
    out = {}
    for i, (a, b) in enumerate(PAIR24):
        out[(i, 0)] = composed_code(vecs[a], vecs[b])
        out[(i, 1)] = composed_code(vecs[b], vecs[a])
    return out

def mint_stimuli_o(W5, vecs):
    """skey -> unit float64 dir. Families: carrier / full:<n> (ident) /
    comp:<i>:<o> (the 48 ordered compositions). Odù dirs are computed
    on-VM and merged separately (skey odudir:<NAME>)."""
    import numpy as _np
    Wax, b = _np.asarray(W5, float)[:14], _np.asarray(W5, float)[14]
    def _pred(x):
        v = _np.asarray(x, float) @ Wax + b
        return v / _np.linalg.norm(v)
    stim = {'carrier': _pred(MU_FRAME)}
    for n in sorted(EVAL64):
        stim[f'full:{n}'] = _pred(vecs[n])
    for i, (a, bnm) in enumerate(PAIR24):
        stim[f'comp:{i}:0'] = _pred(compose_A(vecs[a], vecs[bnm]))
        stim[f'comp:{i}:1'] = _pred(compose_A(vecs[bnm], vecs[a]))
    return stim

def build_e8o_eval(smoke=False):
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    pairs = list(range(2)) if smoke else list(range(len(PAIR24)))
    ident = IDENT16[:2] if smoke else IDENT16
    n_sham = 2 if smoke else 12
    n_car = 1 if smoke else 2
    odus = ODU_NAMES_O[:2] if smoke else ODU_NAMES_O
    rows, tid = [], 50000
    for i in pairs:
        for o in (0, 1):
            for a in A:
                rows.append({'tid': tid, 'block': 'po', 'kind': 'inject',
                             'skey': f'comp:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': a})
                tid += 1
    for (n, a) in ident:
        rows.append({'tid': tid, 'block': 'ident', 'kind': 'inject',
                     'skey': f'full:{n}', 'concept': n,
                     'layer': TRAIN_LAYER, 'alpha': a})
        tid += 1
    for _ in range(n_sham):
        rows.append({'tid': tid, 'block': 'sham', 'kind': 'sham',
                     'skey': None, 'layer': None, 'alpha': 0.0})
        tid += 1
    for _ in range(n_car):
        rows.append({'tid': tid, 'block': 'carrier', 'kind': 'inject',
                     'skey': 'carrier', 'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    for n in odus:
        rows.append({'tid': tid, 'block': 'odu', 'kind': 'inject',
                     'skey': f'odudir:{n}', 'odu': n,
                     'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    return rows

EXPECT_EVAL_O = {'po': 96, 'ident': 16, 'sham': 12, 'carrier': 2, 'odu': 16}

def eval_counts_o(rows):
    out = {}
    for r in rows:
        out[r['block']] = out.get(r['block'], 0) + 1
    return out

# ── statistics ───────────────────────────────────────────────────────────────
def _acc_vs(parsed_levels, code):
    return [1 if parsed_levels.get(AXIS_NAMES_F[j], 99) == code[j] else 0
            for j in range(14)]

def po1_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """Leg-faithful reading: pooled + per-axis accuracy vs the exact composed
    codes, null = derangements of the PAIR assignment (order preserved)."""
    import numpy as _np
    if seed is None:
        seed = E8O_SEED + 20
    P = levels_matrix(rows)
    pair_ids = sorted({r['pair_idx'] for r in rows})
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows], int)
    obs_m = (P == T)
    obs = float(obs_m.mean())
    per_axis_obs = {a: float(obs_m[:, j].mean())
                    for j, a in enumerate(AXIS_NAMES_F)}
    rng = _np.random.default_rng(seed)
    idx = _np.arange(len(pair_ids))
    pos = {p: k for k, p in enumerate(pair_ids)}
    ge_pool = 0
    ge_axis = {a: 0 for a in AXIS_NAMES_F}
    nulls = []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while _np.any(pi == idx):
            pi = rng.permutation(idx)
        Tn = _np.array([pcodes[(pair_ids[pi[pos[r['pair_idx']]]], r['order'])]
                        for r in rows], int)
        m = (P == Tn)
        nm = float(m.mean())
        nulls.append(nm)
        ge_pool += nm >= obs
        for j, a in enumerate(AXIS_NAMES_F):
            if float(m[:, j].mean()) >= per_axis_obs[a]:
                ge_axis[a] += 1
    import numpy as _np2
    p_pool = (1 + ge_pool) / (1 + n_perm)
    p_axis = {a: (1 + ge_axis[a]) / (1 + n_perm) for a in AXIS_NAMES_F}
    hres = holm(p_axis)
    sig = [a for a in AXIS_NAMES_F if hres[a][1]]
    return {'pooled_acc': round(obs, 4), 'p': p_pool,
            'null_mean': round(float(_np2.mean(nulls)), 4),
            'per_axis': {a: {'acc': round(per_axis_obs[a], 4),
                             'p': round(p_axis[a], 5),
                             'holm_sig': bool(hres[a][1])}
                         for a in AXIS_NAMES_F},
            'axes_sig': sig, 'n_axes_sig': len(sig), 'n_rows': len(rows)}

def po2_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """THE L4 CLAIM: order discrimination. Per pair p, d_p = mean over its
    rows of [acc vs OWN composed code - acc vs the FLIPPED code]; statistic
    = mean d over pairs; null = sign-flips over pairs. An order-indifferent
    reader gives d == 0 identically."""
    import numpy as _np
    if seed is None:
        seed = E8O_SEED + 21
    per_pair = {}
    for r in rows:
        own = pcodes[(r['pair_idx'], r['order'])]
        flip = pcodes[(r['pair_idx'], 1 - r['order'])]
        lv = r['parsed']['levels']
        d = (sum(_acc_vs(lv, own)) - sum(_acc_vs(lv, flip))) / 14.0
        per_pair.setdefault(r['pair_idx'], []).append(d)
    dbar = {p: float(_np.mean(v)) for p, v in per_pair.items()}
    dvec = _np.array([dbar[p] for p in sorted(dbar)])
    obs = float(dvec.mean())
    rng = _np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(dvec))
        if float((dvec * s).mean()) >= obs:
            ge += 1
    p = (1 + ge) / (1 + n_perm)
    return {'d_mean': round(obs, 4), 'p': p,
            'per_pair': {int(k): round(v, 4) for k, v in dbar.items()},
            'n_pairs': len(dvec), 'n_pos': int((dvec > 0).sum())}

def gid_stats(rows, codes64):
    """Readout-identity: pooled field accuracy on the re-flown E8-F rows."""
    import numpy as _np
    P = levels_matrix(rows)
    T = _np.array([codes64[r['concept']] for r in rows], int)
    acc = float((P == T).mean())
    return {'pooled_acc': round(acc, 4), 'min': GID_MIN,
            'pass': bool(acc >= GID_MIN), 'n_rows': len(rows)}

@@RIDER_SRC@@

def odu_spell_texture(rows):
    out = {}
    for r in rows:
        out[r['odu']] = {'kind': r['parsed']['kind'],
                         'code': code_text([r['parsed']['levels'].get(a, 0)
                                            for a in AXIS_NAMES_F])
                         if r['parsed']['kind'] == 'CODE' else None}
    return out
'''

MD0 = """# E8-O — Ordered Composition (L4, the Ifá rung; Phase 10 UI flight)

**Pre-registered**: `docs/E8O_PROTOCOL.md` (e10c60f, before build). EVAL-ONLY
on the **locked E8-F readout** — no training. The operator: OP-A leg-swap,
compose(A,B) = essence leg of A + function leg of B (the full-Odù structure
as algebra; order-separation measured 100.8° median through the pinned W).
**P-O1**: are leg-heterogeneous points spelled correctly. **P-O2 (the L4
claim)**: is (A,B) read differently from (B,A) — an order-blind reader
scores exactly 0 by construction. Rider: the 16 Odù-profile dirs + 16
matched controls, computed pre-readout in own-16-call shape, shipped for
the local span-residual/RSA arms.

**Flight plan (Run all, twice)**
1. **Smoke** (`SMOKE=True`, armed): ~8–12 min. GREEN → Restart runtime.
2. **Full** (`SMOKE=False`): one run, ~20–30 min (142 generations with
   progress prints). Verdict banner + `e8o/` results on Drive.
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters, bundles ──────
NB_BUILD = 'v1 (2026-08-25)'
print('E8-O notebook build:', NB_BUILD)

SMOKE = True                   # ARMED FOR SMOKE: flip to False after GREEN

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes, hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'

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
    f'GPU already holds {_leftover/1e9:.1f}GB from a previous run — Runtime > '
    'Restart runtime, then Run all.')
print('RAM at start:', ram_report())

from google.colab import drive
drive.mount('/content/drive')
SEM = Path('/content/drive/MyDrive/semcore')
assert SEM.exists(), 'MyDrive/semcore not found — mounted the right account?'

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

# Shipped E8-R real bundle (G2 identity probe)
E8R_SRC = SEM / 'e8r/inflight_20260822_2329'   # flight of record — never change
_b = json.load(open(E8R_SRC / 'condition_real.json'))
assert _b.get('dirs') and _b.get('mu'), 'E8-R bundle lacks dirs/mu'
SHIPPED_E8R = {'dirs': _b['dirs'], 'mu': _b['mu']}
print('E8-R shipped stimulus [real]: layers', sorted(_b['dirs']))

# E4 real adapter (instillation) + the LOCKED E8-F readout (flight of record)
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
print('MODE:', MODE.upper(), '| stamp', STAMP)
'''

CELL_PAYLOAD_O_TEMPLATE = r'''# ── Pinned payload (E8-F verbatim) + E8-O stimulus mint + G1 pin gate ────────
import numpy as np

PAYLOAD = json.loads(r"""@@PAYLOAD_JSON@@""")
_pl_sha = hashlib.sha256(json.dumps(
    {k: PAYLOAD[k] for k in sorted(PAYLOAD)}, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()[:16]
assert _pl_sha == PAYLOAD_SHA, f'payload sha drift: {_pl_sha} != {PAYLOAD_SHA}'
W5 = np.array(PAYLOAD['W'], float)
assert W5.shape == (15, 1536)
assert abs(PAYLOAD['mu14'] - MU14_ATLAS) < 1e-9

# the locked E8-F readout comes down from Drive (flight of record)
READOUT_DIR = Path('/content/readout_real')
_ro = SEM / READOUT_SRC
assert (_ro / 'adapter_config.json').exists(), (
    f'locked E8-F readout not found at {_ro} — E8-O is eval-only on it')
shutil.copytree(_ro, READOUT_DIR, dirs_exist_ok=True)
print('readout: pulled', READOUT_SRC)

_sha24 = hashlib.sha256('|'.join(f'{a}+{b}' for a, b in PAIR24).encode()
                        ).hexdigest()[:16]
assert _sha24 == PAIR24_SHA, 'pair draw drift'
assert len(PAIR24) == 24 and len(IDENT16) == 16 and len(CONTROL16) == 16
assert not set(ODU_NAMES_O) & set(VEC), 'Odù name collides with the pack'
for n, d in ODU_DESC.items():
    DESC[n] = d                      # rider texts join the DESC table
assert all(len(d) >= 40 for d in ODU_DESC.values())

STIM = mint_stimuli_o(W5, VEC)
for k, v in STIM.items():
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9, k
assert len([k for k in STIM if k.startswith('comp:')]) == 48
for k, pv in PAYLOAD_PROBES_O.items():
    assert float(np.max(np.abs(STIM[k] - np.asarray(pv, float)))) < 1e-4, \
        f'stimulus probe drift at {k}'
FR_CODES = frame_codes(pack['concepts'])
assert frame_codes_sha(FR_CODES) == FRAME_CODES_SHA, 'frame codes drift'
CODES64 = {n: FR_CODES[n] for n in EVAL64}
PCODES = pair_codes(VEC)
assert len(PCODES) == 48
print(f'G1 PIN GATE: payload {PAYLOAD_SHA} ok | pairs {PAIR24_SHA} ok | '
      f'{len(STIM)} stimuli minted + probed | readout pulled')
'''

CELL_MODEL = ("# ── Model builders (E8-F donors, verbatim) ──────────────────────────────────\n"
              "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
              "from peft import PeftModel\n\n"
              "tok = AutoTokenizer.from_pretrained(MODEL_ID)\n"
              "if tok.pad_token is None:\n"
              "    tok.pad_token = tok.eos_token\n\n"
              + BUILD_EVAL_MODEL_SRC +
              "def resolve_layers(m):\n"
              "    mods = {}\n"
              "    for mod_name, mod in m.named_modules():\n"
              "        mm = re.search(r'(?:^|\\.)layers\\.(\\d+)$', mod_name)\n"
              "        if mm:\n"
              "            mods[int(mm.group(1))] = mod\n"
              "    n = m.config.num_hidden_layers if hasattr(m.config, 'num_hidden_layers') \\\n"
              "        else m.base_model.config.num_hidden_layers\n"
              "    assert len(mods) == n, (len(mods), n)\n"
              "    return mods\n")


CELL_FLIGHT = r'''# ── Flight: pre-readout G2 + rider dirs -> attach readout -> eval -> ship ────
def fly():
    t0 = time.time()
    bundle = {'condition': 'real', 'mode': MODE, 'stamp': STAMP,
              'payload_sha': PAYLOAD_SHA, 'pair_sha': PAIR24_SHA,
              'readout_src': READOUT_SRC}
    try:
        # pre-readout stack: base + merged E4 (the stimulus/probe model)
        m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                  device_map=DEV, low_cpu_mem_usage=True)
        m0 = PeftModel.from_pretrained(m0, ADAPTERS['real']).merge_and_unload()
        m0.eval()
        print(f'pre-readout stack ready — {ram_report()}')
        stab, mu14 = g2_probe(m0)
        bundle['g2_dirs'], bundle['mu14'] = stab, mu14
        assert stab['pass'], f'G2 dirs-stability FAILED: {stab}'
        # rider dirs, own-16-call shape each (batch-composition law)
        odu_d = compute_dirs(m0, [14], ODU_NAMES_O)[14]
        ctl_d = compute_dirs(m0, [14], CONTROL16)[14]
        bundle['odu_dirs'] = {n: [round(float(x), 5) for x in odu_d[n]]
                              for n in ODU_NAMES_O}
        bundle['control_dirs'] = {n: [round(float(x), 5) for x in ctl_d[n]]
                                  for n in CONTROL16}
        for n in ODU_NAMES_O:
            v = np.asarray(odu_d[n], float)
            STIM[f'odudir:{n}'] = v / np.linalg.norm(v)
        print(f'  rider dirs computed: {len(odu_d)} Odù + {len(ctl_d)} controls')
        # attach the LOCKED readout (no training anywhere)
        model = PeftModel.from_pretrained(m0, str(READOUT_DIR))
        model.eval()
        layer_mods = resolve_layers(model)
        rows = build_e8o_eval(SMOKE)
        out, t1 = {}, time.time()
        for i, t in enumerate(rows, 1):
            r = run_trial_feat(model, layer_mods, mu14, t)
            out.setdefault(t['block'], []).append(r)
            if i % 20 == 0 or i == len(rows):
                el = time.time() - t1
                eta = el / i * (len(rows) - i)
                print(f'    eval {i}/{len(rows)}, {el:.0f}s elapsed, ~{eta:.0f}s left')
        bundle['eval_rows'] = out
        cnt = eval_counts_o([r for rs in out.values() for r in rs])
        print(f'  eval rows: {cnt}')
        if not SMOKE:
            assert cnt == EXPECT_EVAL_O, cnt
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        bundle['error'] = f'{type(e).__name__}: {e}'
    return bundle

print(f'flight: starting — {ram_report()}')
BUNDLE = fly()
fn = OUT / 'bundle_real.json'
jdump(BUNDLE, fn)
ship(fn, f'e8o/inflight_{STAMP}')
print(f'flight done in {BUNDLE.get("secs", "?")}s — bundle shipped')
free_ram()
'''

CELL_VERDICT = r'''# ── Gates, P-O1/P-O2, S-blocks, fork, banner, ship ──────────────────────────
ok = bool(BUNDLE.get('eval_rows')) and 'error' not in BUNDLE
if not ok:
    print()
    print('=' * 72)
    print(f'  FLIGHT INCOMPLETE — error: {str(BUNDLE.get("error"))[:200]}')
    print('  Fix comes home to the builder (lane law).')
    print('=' * 72)
else:
    verdict = {'flight': f'{MODE}_{STAMP}', 'protocol': 'E8O_PROTOCOL.md',
               'payload_sha': PAYLOAD_SHA, 'pair_sha': PAIR24_SHA,
               'readout_src': READOUT_SRC}
    blocks = BUNDLE['eval_rows']
    allrows = [r for rs in blocks.values() for r in rs]
    NP = 200 if SMOKE else N_PERM_O

    inv = invalid_rate(allrows)
    fa = sham_claims(blocks.get('sham', []))
    gid = gid_stats(blocks['ident'], CODES64)
    gates = {'g1_pins': {'payload': PAYLOAD_SHA, 'pairs': PAIR24_SHA,
                         'asserted_in_flight': True, 'pass': True},
             'g2_dirs': BUNDLE['g2_dirs'],
             'g_id': gid,
             'g4_parse': {**inv, 'max': INVALID_MAX_O,
                          'pass': bool(inv['rate'] <= INVALID_MAX_O)},
             'g5_sham': {'claims': fa, 'n': len(blocks.get('sham', [])),
                         'max': SHAM_MAX_O, 'pass': bool(fa <= SHAM_MAX_O)}}
    gates_ok = all(g.get('pass') for g in gates.values())
    verdict['gates'] = gates

    po1 = po1_stats(blocks['po'], PCODES, NP)
    po1_pass = bool(po1['p'] <= P_CRIT_O and po1['n_axes_sig'] >= AX_MIN_O)
    po2 = po2_stats(blocks['po'], PCODES, NP)
    po2_pass = bool(po2['p'] <= P_CRIT_O)
    verdict['P_O1'] = {**po1, 'pass': po1_pass}
    verdict['P_O2'] = {**po2, 'pass': po2_pass}
    verdict['S1'] = s1_profile(blocks)
    verdict['S_carrier'] = s7_carrier(blocks['carrier'])
    verdict['S_odu'] = odu_spell_texture(blocks['odu'])

    if SMOKE:
        fork = 'SMOKE — mechanics only, no verdict'
    elif not gates_ok:
        fork = 'GATES DIRTY — primaries withheld (NO_VERDICT, lane law)'
    elif po1_pass and po2_pass:
        fork = ('FO1 — L4 LANDS in bottleneck form: the register\'s '
                'non-commutative composition executes through the trained '
                'channel; order is behaviorally discriminated')
    elif po1_pass and not po2_pass:
        fork = ('FO2 — leg-heterogeneous points read but the reader '
                'SYMMETRIZES: order collapses in the channel')
    else:
        fork = ('FO3 — composed points off the méjì manifold are unreadable '
                '(register-capacity finding)')
    verdict['fork'] = fork

    print()
    print('=' * 72)
    if SMOKE:
        n_par = sum(1 for r in allrows if r['parsed']['kind'] != 'INVALID')
        green = ok and n_par >= max(1, int(0.5 * len(allrows)))
        print(f'  SMOKE {"GREEN" if green else "RED"} — mechanics '
              f'{"exercised" if green else "FAILED"} ({n_par}/{len(allrows)} '
              f'parsed; g2 {gates["g2_dirs"]["pass"]}, g_id mechanical '
              f'{gid["pooled_acc"]})')
        if green:
            print('  Next: Runtime > Restart runtime, set SMOKE = False, Run all.')
    else:
        print(f'  E8-O VERDICT — gates {"PASS" if gates_ok else "FAIL"} '
              f'(G-ID {gid["pooled_acc"]} vs {GID_MIN})')
        if gates_ok:
            print(f'  P-O1 leg-faithful: pooled {po1["pooled_acc"]} vs null '
                  f'{po1["null_mean"]} p={po1["p"]} | axes {po1["n_axes_sig"]}/14 '
                  f'-> {"PASS" if po1_pass else "FAIL"}')
            print(f'  P-O2 ORDER: d_mean {po2["d_mean"]} '
                  f'({po2["n_pos"]}/{po2["n_pairs"]} pairs positive) '
                  f'p={po2["p"]} -> {"PASS" if po2_pass else "FAIL"}')
        print(f'  fork: {fork}')
    print('=' * 72)
    vf = OUT / 'e8o_verdict.json'
    jdump(verdict, vf)
    ship(vf, f'e8o/{MODE}_{STAMP}')
    fn = OUT / 'bundle_real.json'
    if not fn.exists():
        jdump(BUNDLE, fn)
    ship(fn, f'e8o/{MODE}_{STAMP}')
    print('shipped: e8o/' + f'{MODE}_{STAMP}')
'''


def mint():
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    DESC = {c["name"]: c.get("desc") or "" for c in pack["concepts"]}
    payload = json.load(open(os.path.join(HERE, "e8f_payload.json")))
    sys.path.insert(0, HERE)
    import e8f_logic as F

    assert not set(ODU_NAMES) & set(VEC), "Odù name collides with pack"
    for d in ODU_TEXTS.values():
        assert len(d) >= 40

    ev = sorted(F.EVAL64)
    rng = np.random.default_rng(E8O_SEED + 10)
    pair24, deg = [], {}
    guard = 0
    while len(pair24) < 24:
        guard += 1
        assert guard < 100000
        a, b = (str(x) for x in rng.choice(ev, size=2, replace=False))
        a, b = sorted((a, b))
        if (a, b) in pair24 or deg.get(a, 0) >= 2 or deg.get(b, 0) >= 2:
            continue
        pair24.append((a, b))
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    pair24 = sorted(pair24)
    sha24 = hashlib.sha256("|".join(f"{a}+{b}" for a, b in pair24).encode()
                           ).hexdigest()[:16]

    rng2 = np.random.default_rng(E8O_SEED + 11)
    ident_pool = [(n, a) for n in ev for a in (0.5, 1.0)]
    idx = rng2.choice(len(ident_pool), size=16, replace=False)
    ident16 = sorted((ident_pool[int(i)] for i in idx))

    # desc-length-matched controls from train-256 (greedy nearest length)
    tr = sorted(F.TRAIN256)
    lens = sorted(((abs(len(DESC[n]) - int(np.mean([len(d) for d in
                   ODU_TEXTS.values()]))), n) for n in tr))
    control16 = sorted(n for _, n in lens[:16])

    W5 = np.array(payload["W"], float)
    Wax, b = W5[:14], W5[14]

    def pred(x):
        v = np.asarray(x, float) @ Wax + b
        return v / np.linalg.norm(v)

    def compA(vA, vB):
        return list(vA[:7]) + list(vB[7:])

    probes = {}
    for key, (a, bnm, o) in {
        "comp:0:0": (pair24[0][0], pair24[0][1], 0),
        "comp:11:1": (pair24[11][0], pair24[11][1], 1),
    }.items():
        va, vb = (VEC[a], VEC[bnm]) if o == 0 else (VEC[bnm], VEC[a])
        probes[key] = [round(float(x), 5) for x in pred(compA(va, vb))]
    probes["carrier"] = [round(float(x), 5) for x in pred(F.MU_FRAME)]

    subs = {
        "@@READOUT_SRC@@": READOUT_SRC_PATH,
        "@@PAIR24@@": json.dumps([list(p) for p in pair24]),
        "@@IDENT16@@": json.dumps([[n, a] for n, a in ident16]),
        "@@CONTROL16@@": json.dumps(control16),
        "@@ODU_DESC@@": json.dumps(ODU_TEXTS),
        "@@PAIR24_SHA@@": sha24,
        "@@COMPOSE_SRC@@": COMPOSE_SRC.rstrip(),
        "@@RIDER_SRC@@": RIDER_SRC.rstrip(),
    }
    return payload, subs, probes, {"vec": VEC, "pair24": pair24,
                                   "control16": control16, "F": F}


def main():
    payload, subs, probes, ctx = mint()
    e8o_src = E8O_SRC_TEMPLATE
    for k, v in subs.items():
        assert k in e8o_src, f"missing sentinel {k}"
        e8o_src = e8o_src.replace(k, v)
    assert "@@" not in e8o_src

    logic = E8F_LOGIC_SRC + "\n\n" + e8o_src
    with open(os.path.join(HERE, "e8o_logic.py"), "w") as f:
        f.write(logic)
    for mod in ("e8o_logic",):
        if mod in sys.modules:
            del sys.modules[mod]
    L = importlib.import_module("e8o_logic")

    # mint round-trips + registered invariants
    assert [tuple(p) for p in L.PAIR24] == ctx["pair24"]
    assert L.CONTROL16 == ctx["control16"]
    deg = {}
    for a, b in L.PAIR24:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert all(v <= 2 for v in deg.values())
    assert set(deg) <= set(L.EVAL64)
    assert set(L.CONTROL16) <= set(L.TRAIN256)
    rows = L.build_e8o_eval(smoke=False)
    assert L.eval_counts_o(rows) == L.EXPECT_EVAL_O, L.eval_counts_o(rows)
    stim = L.mint_stimuli_o(np.array(payload["W"], float), ctx["vec"])
    for k, pv in probes.items():
        assert float(np.max(np.abs(stim[k] - np.asarray(pv, float)))) < 1e-4, k
    pc = L.pair_codes(ctx["vec"])
    for i, (a, b) in enumerate(L.PAIR24):
        assert pc[(i, 0)][:7] == L.code_levels(ctx["vec"][a])[:7]
        assert pc[(i, 0)][7:] == L.code_levels(ctx["vec"][b])[7:]

    payload_json = json.dumps(payload, separators=(",", ":"))
    assert '"""' not in payload_json and "\\" not in payload_json
    cell_payload = CELL_PAYLOAD_O_TEMPLATE.replace("@@PAYLOAD_JSON@@",
                                                   payload_json)
    cell_payload = cell_payload.replace(
        "for k, pv in PAYLOAD_PROBES_O.items():",
        "PAYLOAD_PROBES_O = json.loads(r'''" + json.dumps(probes) + "''')\n"
        "for k, pv in PAYLOAD_PROBES_O.items():")

    cell_stim = ("# ── G2 stimulus machinery (E8-J slices) + Injector + runner (E8-F donors) ──\n"
                 "import numpy as np\n"
                 + BF.STIM_MACH_SRC + "\n" + BF.INJECTOR_SRC + BF.CELL_RUN_SRC)

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": []},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "cells": []}

    def add(kind, src):
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)

    add("markdown", MD0)
    add("code", CELL_SETUP)
    add("code", logic)
    add("code", cell_payload)
    add("code", CELL_MODEL)
    add("code", cell_stim)
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    import ast
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        assert "rclone" not in src.lower(), f"cell {i}: rclone"
        if c["cell_type"] == "code":
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"cell {i} does not compile: {e}")

    out_path = os.path.join(HERE, "E8O_ORDERED_UI.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"built {out_path} ({os.path.getsize(out_path)/1e3:.0f}KB, "
          f"{len(nb['cells'])} cells)")
    print(f"pairs sha {subs['@@PAIR24_SHA@@']}; readout {READOUT_SRC_PATH}")
    import shutil
    dst = os.path.expanduser("~/Desktop/E8O_ORDERED_UI.ipynb")
    shutil.copy2(out_path, dst)
    print(f"staged {dst}")


if __name__ == "__main__":
    main()
