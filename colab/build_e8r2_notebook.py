#!/usr/bin/env python3
"""Build colab/E8R2_FORCEDCHOICE_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-R2: forced-choice micro-flight on the locked E8-R readout adapters
(docs/E8R2_PROTOCOL.md, pre-registered 97c5f50). EVAL-ONLY — no training
anywhere. The E8-R pure-logic block is lifted VERBATIM from colab/e8r_logic.py
(single source: what flew is what scores); the E8-R2 additions are emitted
BOTH into the notebook and into colab/e8r2_logic.py (concatenated after the
E8-R block) so local tests exercise the exact code that flies.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
E8R_LOGIC_SRC = open(os.path.join(HERE, "e8r_logic.py")).read()

E8R2_SRC = r'''# ── E8R2 pure logic: forced-choice plans, scoring rows, stats (locally tested verbatim) ──
# (builds on the e8r_logic namespace: CHOICE_SET, HELD_OUT, TRAINED, build_plan,
#  report_prompt, parse_report, angle14, perm_null_median, boot_delta_median,
#  oracle_floor, holm, jdump)

E8R2_SEED = 20260824            # all E8-R2-new randomness; disjoint from 20260822/20260823
FC_ALPHAS = [0.5, 1.0]          # the trained regime — where the readout speaks
TITRATION_ALPHAS = [0.30, 0.375, 0.45]   # v2 item 3: between locked 0.25 (0/9) and 0.5 (18/18)
N_FC_ORDERS = 12                # 4 concepts x 2 alphas x 12 orders = 96 rows
ANCHOR_MIN_EXACT = 16           # G1: rides the measured 18/18 baseline on the same trials
PPL_TOL_PCT = 0.5               # G1b: reconstructed ppl vs shipped ppl_post
DIRNORM_TOL = 1e-3              # G2: 5dp ship-rounding residual bound
INFLIGHT_SRC = 'e8r/inflight_20260822_2329'   # flight of record — never change
FC_CANDIDATES = list(CHOICE_SET)              # argmax eligibility — NONE excluded
SCORED_SET = FC_CANDIDATES + ['NONE']         # NONE scored as texture, never eligible
N_PERM = 2000
N_BOOT = 10000

def anchor_rows(smoke=False):
    """The locked E7-Q plan's trained-regime rows VERBATIM (tids + menu orders):
    the 18 trials generation scored 18/18 exact on — G1 rides that baseline."""
    rows = [t for t in build_plan(False)
            if t['kind'] == 'inject' and t['concept'] in TRAINED
            and t['layer'] == TRAIN_LAYER and t['alpha'] in TRAIN_ALPHAS]
    assert len(rows) == 18, len(rows)
    if smoke:
        rows = [rows[0], rows[-1]]           # both alphas exercised
    return [{**t, 'block': 'anchor'} for t in rows]

def sham_rows(smoke=False):
    """The locked plan's 12 sham rows verbatim — the forced-choice prior is
    measured on the trials whose generation behavior is known (0 claims)."""
    rows = [t for t in build_plan(False) if t['kind'] == 'sham']
    assert len(rows) == N_SHAMS, len(rows)
    if smoke:
        rows = rows[:2]
    return [{**t, 'block': 'sham'} for t in rows]

def heldout_fc_rows(smoke=False):
    """4 held-out concepts x FC_ALPHAS x L14 x N_FC_ORDERS fresh seeded menu
    orders = 96 rows. All named by construction under forced scoring."""
    rng = np.random.default_rng(E8R2_SEED)
    rows, tid = [], 5000
    for c in HELD_OUT:
        for a in FC_ALPHAS:
            for _ in range(N_FC_ORDERS):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                rows.append({'tid': tid, 'kind': 'inject', 'concept': c,
                             'layer': TRAIN_LAYER, 'alpha': a, 'order': order,
                             'block': 'heldout_fc'})
                tid += 1
    if smoke:
        by = {}
        for r in rows:
            by.setdefault(r['concept'], []).append(r)
        rows = [by[HELD_OUT[0]][0], by[HELD_OUT[1]][N_FC_ORDERS],
                by[HELD_OUT[2]][0], by[HELD_OUT[3]][N_FC_ORDERS]]
    return rows

def titration_rows(smoke=False):
    """9 trained concepts x TITRATION_ALPHAS x L14, free-report path (v2 item 3)."""
    rng = np.random.default_rng(E8R2_SEED + 1)
    rows, tid = [], 6000
    for c in TRAINED:
        for a in TITRATION_ALPHAS:
            order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
            rows.append({'tid': tid, 'kind': 'inject', 'concept': c,
                         'layer': TRAIN_LAYER, 'alpha': a, 'order': order,
                         'block': 'titration'})
            tid += 1
    if smoke:
        rows = rows[:2]
    return rows

def rank_of(name, scores):
    """Competition rank of a name among the 13 candidates (ties share rank)."""
    return 1 + sum(1 for m in FC_CANDIDATES if scores[m] > scores[name])

def fc_row(trial, scores_mean, scores_sum, vecs):
    """One scored row. Forced answer = argmax over the 13 names by MEAN
    per-token logprob (pre-registered primary metric); sum-logprob argmax is
    the pre-named robustness twin. NONE is scored but never eligible."""
    top = max(FC_CANDIDATES, key=lambda n: scores_mean[n])
    top_sum = max(FC_CANDIDATES, key=lambda n: scores_sum[n])
    inj = trial.get('concept')
    row = {'tid': trial['tid'], 'block': trial['block'],
           'layer': trial.get('layer'), 'alpha': trial['alpha'],
           'injected': inj, 'argmax': top, 'argmax_sum': top_sum,
           'agree_sum_mean': bool(top == top_sum),
           'none_top': bool(scores_mean['NONE'] > scores_mean[top]),
           'scores': {n: round(float(scores_mean[n]), 5) for n in SCORED_SET},
           'scores_sum': {n: round(float(scores_sum[n]), 5) for n in SCORED_SET}}
    if inj is not None:
        row['rank'] = rank_of(inj, scores_mean)
        row['exact'] = bool(top == inj)
        row['err'] = angle14(vecs[inj], vecs[top])
        row['err_sum'] = angle14(vecs[inj], vecs[top_sum])
    return row

def as_perm_rows(fcrows):
    """Adapt forced rows to the e8r_logic permutation-null row shape
    (P-E8R2-1 reuses the E8-R P1 machinery verbatim)."""
    return [{'tid': r['tid'], 'layer': r['layer'], 'alpha': r['alpha'],
             'injected': r['injected'], 'report': r['argmax'], 'err': r['err']}
            for r in fcrows]

def perm_null_rank(fcrows, n_perm=2000, seed=None):
    """S2: median rank of the injected name vs injected-label shuffles within
    alpha strata (ranks looked up in each row's own frozen score vector)."""
    if seed is None:
        seed = E8R2_SEED + 2
    rng = np.random.default_rng(seed)
    obs = float(np.median([r['rank'] for r in fcrows]))
    strata = {}
    for i, r in enumerate(fcrows):
        strata.setdefault(r['alpha'], []).append(i)
    nulls = []
    for _ in range(n_perm):
        med = []
        for idxs in strata.values():
            labels = [fcrows[i]['injected'] for i in idxs]
            rng.shuffle(labels)
            med.extend(rank_of(lab, fcrows[i]['scores'])
                       for i, lab in zip(idxs, labels))
        nulls.append(float(np.median(med)))
    p = (1 + sum(1 for m in nulls if m <= obs)) / (1 + len(nulls))
    return obs, p

def binom_tail(k, n, p):
    """Exact one-sided binomial tail P(X >= k)."""
    from math import comb
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))

def gate_g1(anchor_scored):
    """Scoring must reproduce the trained readout on the locked anchor trials."""
    n = len(anchor_scored)
    exact = sum(1 for r in anchor_scored if r.get('exact'))
    return {'n': n, 'exact': exact, 'min_exact': ANCHOR_MIN_EXACT,
            'pass': bool(n == 18 and exact >= ANCHOR_MIN_EXACT)}

def gate_g1b(ppl_recon, ppl_post):
    """Reconstructed weights must be the flight's weights (shipped ppl_post)."""
    d = 100.0 * (ppl_recon / ppl_post - 1.0)
    return {'ppl_recon': round(float(ppl_recon), 4),
            'ppl_post_shipped': round(float(ppl_post), 4),
            'delta_pct': round(float(d), 4), 'tol_pct': PPL_TOL_PCT,
            'pass': bool(abs(d) <= PPL_TOL_PCT)}

def check_bundle(src):
    """G2: shipped bundle integrity — L14 dirs for all 13 concepts at unit norm
    within ship-rounding, mu present, ppl_post present."""
    d14 = src.get('dirs', {}).get('14', {})
    names_ok = set(d14) == set(CHOICE_SET)
    resid = (max(abs(1.0 - float(np.linalg.norm(np.asarray(v, float))))
                 for v in d14.values()) if names_ok else None)
    ok = bool(names_ok and resid is not None and resid < DIRNORM_TOL
              and '14' in src.get('mu', {}) and src.get('ppl_post'))
    return {'names_ok': bool(names_ok),
            'norm_resid_max': (round(resid, 8) if resid is not None else None),
            'mu14': src.get('mu', {}).get('14'), 'pass': ok}

def load_stimulus(src):
    """The flight-of-record frozen stimulus, loaded not recomputed:
    L14 directions re-unit-normalized (5dp ship rounding) + mu."""
    d14 = {n: np.asarray(v, float) for n, v in src['dirs']['14'].items()}
    d14 = {n: v / np.linalg.norm(v) for n, v in d14.items()}
    return d14, float(src['mu']['14'])

def titration_curve(rows):
    """S6: free-report claim/exact rates by alpha."""
    out = {}
    for a in sorted({r['alpha'] for r in rows}):
        rs = [r for r in rows if r['alpha'] == a]
        claims = [r for r in rs if r['report'] not in ('NONE', 'INVALID')]
        out[str(a)] = {'n': len(rs), 'claim_rate': round(len(claims) / max(1, len(rs)), 4),
                       'exact_rate': round(sum(1 for r in claims
                                               if r['report'] == r['concept'])
                                           / max(1, len(rs)), 4),
                       'invalid': sum(1 for r in rs if r['report'] == 'INVALID')}
    return out

def sham_prior(sham_scored):
    """S5: forced-choice prior on shams — argmax distribution + NONE-top rate."""
    dist = {}
    for r in sham_scored:
        dist[r['argmax']] = dist.get(r['argmax'], 0) + 1
    modal = max(dist, key=dist.get) if dist else None
    return {'n': len(sham_scored), 'argmax_dist': dist, 'modal': modal,
            'none_top_rate': round(sum(1 for r in sham_scored if r['none_top'])
                                   / max(1, len(sham_scored)), 4)}
'''

MD0 = """# E8-R2 — Forced-Choice Micro-Flight (Phase 10, UI flight, EVAL-ONLY)

**Pre-registration: `docs/E8R2_PROTOCOL.md` (session 128, commit 97c5f50) — locks at first full flight.**

Probes the **locked E8-R readout adapters** (flight of record
`inflight_20260822_2329`, both verified complete on Drive) with the abstention
channel closed at the measurement layer: every trial is scored as a ranking of
all 13 state names by answer log-probability under the **verbatim trained
prompt** — no NONE escape, no new prompt, **no training anywhere**. Question:
does the readout **know more than it says** on the four held-out concepts
(n=96 vs the locked flight's starved n=6), and is that knowledge carried by
the instilled geometry (real vs convergence-matched scrambled)? Plus the
pre-named α-titration block (speech threshold between 0.25 and 0.5).

**Flow (three Run-alls, ~40 min total):**
1. **Smoke** (fresh runtime, `SMOKE=True` as shipped): Run all → ~8–12 min →
   GREEN/RED banner. Smoke loads BOTH conditions once each and fully exercises
   bundle load, adapter reconstruction (ppl gate), scoring, generation,
   shipping.
2. Runtime > **Restart runtime** → set `SMOKE=False` → Run all → flies **real**
   (~12–18 min) → the end banner prints a `RESUME_STAMP` line.
3. Runtime > **Restart runtime** → paste the stamp into `RESUME_STAMP` →
   Run all → flies **scrambled** + computes and ships the verdict.

If a RAM guard trips twice, Runtime > **Disconnect and delete runtime** for a
fresh VM, then Run all again. Results land in `MyDrive/semcore/e8r2/` and are
pulled session-side via the Drive integration. The setup cell prints the build
tag as its first line; if yours differs, you're on a stale copy — File >
Upload notebook > pick the Desktop file.
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters, source bundles ─
NB_BUILD = 'v1 (2026-08-23)'
print('E8-R2 notebook build:', NB_BUILD)

SMOKE = True                   # first run: smoke. Then False for the full flights.
RESUME_STAMP = ''              # paste the banner's stamp between full runs
ONE_CONDITION_PER_RUN = True   # lane law: full mode = one model load per VM session

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
    'transformers>=4.44','peft>=0.11','accelerate'], check=True)

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
assert not (_avail < 6.5), (
    f'Only {_avail:.1f}GB system RAM available (need 6.5). Runtime > Restart '
    'runtime; if it trips again, Runtime > Disconnect and delete runtime for '
    'a fresh VM, then Run all.')
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

CONDITIONS = ('real', 'scrambled')     # base excluded — pre-reg: unconverged readout

# E4 instillation adapters (E8-R resolution code verbatim)
ADAPTERS = {}
for arm in CONDITIONS:
    cand = sorted(d.name for d in (SEM / 'e4').iterdir()
                  if d.is_dir() and d.name.startswith(f'{arm}_full_'))
    assert cand, f'no {arm}_full_* dir under semcore/e4'
    src = SEM / 'e4' / cand[-1] / f'adapter_{arm}'
    dst = Path(f'/content/adapter_{arm}')
    shutil.copytree(src, dst, dirs_exist_ok=True)
    assert (dst / 'adapter_config.json').exists(), f'adapter_{arm} incomplete'
    ADAPTERS[arm] = str(dst)
    print(f'instillation adapter {arm}: {cand[-1]}')

# The flight of record: shipped condition bundles + readout adapters.
# (E8-R's smoke check pinned only adapter_config.json; the safetensors are
# asserted here explicitly — the whole rung rides on them.)
SRC_DIR = SEM / 'e8r/inflight_20260822_2329'
assert SRC_DIR.exists(), f'{SRC_DIR} not on Drive — wrong Google account?'
SRC_BUNDLE, READOUTS = {}, {}
for cond in CONDITIONS:
    bj = SRC_DIR / f'condition_{cond}.json'
    assert bj.exists(), f'missing {bj}'
    SRC_BUNDLE[cond] = json.load(open(bj))
    assert SRC_BUNDLE[cond].get('post_trials'), f'{cond} source bundle incomplete'
    rsrc = SRC_DIR / f'readout_{cond}'
    for fname in ('adapter_config.json', 'adapter_model.safetensors'):
        assert (rsrc / fname).exists(), f'missing {rsrc / fname}'
    rcfg = json.load(open(rsrc / 'adapter_config.json'))
    assert rcfg.get('r') == 16 and rcfg.get('lora_alpha') == 32, rcfg
    rdst = Path(f'/content/readout_{cond}')
    shutil.copytree(rsrc, rdst, dirs_exist_ok=True)
    READOUTS[cond] = str(rdst)
    print(f'readout adapter {cond}: loaded from flight of record '
          f'({(rsrc / "adapter_model.safetensors").stat().st_size} bytes)')

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
INFLIGHT = f'e8r2/inflight_{RESUME_STAMP or STAMP}'
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

CELL_MODEL = r'''# ── Model reconstruction + scoring/generation machinery (EVAL-ONLY) ──────────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def build_condition_model(cond):
    """Exactly as flown: base fp16 + merged E4 instillation adapter + the
    shipped readout LoRA ATTACHED (not merged — matching flight state).
    No get_peft_model, no fresh LoRA, nothing trainable."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(m, ADAPTERS[cond]).merge_and_unload()
    m = PeftModel.from_pretrained(m, READOUTS[cond])
    m.eval()
    assert not any(p.requires_grad for p in m.parameters()), 'eval-only flight'
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

class Injector:
    """Adds alpha*mu*dhat to a decoder layer's residual output from the final
    prompt position onward — OUT-OF-PLACE, identical math to the E8-R hook
    (training-forward coverage: positions >= start on the full forward)."""
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

# Candidate answer encodings — the exact trained answer form: name tokens + EOS.
CAND_IDS = {name: tok(name, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            for name in SCORED_SET}
print('candidate token counts:',
      {n: len(ids) for n, ids in sorted(CAND_IDS.items())})

def encode_prompt(order):
    prompt = report_prompt(order, DESC)
    enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    return enc['input_ids'][0]

def score_trial(m, layer_mods, dirs14, mu14, trial, cond):
    """Forced-choice scoring: ONE batched decoder forward over
    [prompt + candidate answer] for all 14 candidates (13 names + NONE),
    injection hook live from the final prompt position (training coverage),
    then answer-sliced head — full-sequence logits never materialized
    (E8-N v2 memory law). Returns the fc_row + ops fields."""
    pid = encode_prompt(trial['order'])
    plen = int(pid.shape[0])
    seqs = [torch.cat([pid, torch.tensor(CAND_IDS[n], dtype=pid.dtype)])
            for n in SCORED_SET]
    maxlen = max(int(s.shape[0]) for s in seqs)
    ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    ids, mask = ids.to(DEV), mask.to(DEV)
    _cm = m.base_model.model
    DECODER, HEAD = _cm.model, _cm.get_output_embeddings()
    with torch.no_grad():
        if trial['kind'] == 'inject':
            d = torch.tensor(dirs14[trial['concept']])
            vec = trial['alpha'] * mu14 * d
            with Injector(layer_mods, TRAIN_LAYER, vec, plen - 1) as inj:
                hid = DECODER(input_ids=ids, attention_mask=mask,
                              use_cache=False).last_hidden_state
            calls = inj.calls
            assert calls >= 1, 'injection hook never fired during scoring'
        else:
            hid = DECODER(input_ids=ids, attention_mask=mask,
                          use_cache=False).last_hidden_state
            calls = 0
        lo = plen - 1                      # position p predicts token p+1
        sm, ss = {}, {}
        for i, name in enumerate(SCORED_SET):
            n_ans = len(CAND_IDS[name])
            logits = HEAD(hid[i, lo:lo + n_ans, :]).float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(CAND_IDS[name], device=lp.device)
            tlp = lp[torch.arange(n_ans, device=lp.device), tgt]
            sm[name] = float(tlp.mean())
            ss[name] = float(tlp.sum())
    row = fc_row(trial, sm, ss, VEC)
    row.update({'cond': cond, 'hook_calls': calls})
    return row

def run_trial_free(m, layer_mods, dirs14, mu14, trial, cond):
    """Free-report generation path for the titration block — E8-R run_trial
    verbatim semantics (greedy, 24 new tokens, parse_report)."""
    prompt = report_prompt(trial['order'], DESC)
    enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    ids = enc['input_ids'].to(DEV)
    gen_kw = dict(max_new_tokens=24, do_sample=False,
                  pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.no_grad():
        d = torch.tensor(dirs14[trial['concept']])
        vec = trial['alpha'] * mu14 * d
        with Injector(layer_mods, TRAIN_LAYER, vec, ids.shape[1] - 1) as inj:
            out = m.generate(input_ids=ids, **gen_kw)
        calls = inj.calls
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'cond': cond, 'response': text.strip()[:200],
            'report': parse_report(text), 'hook_calls': calls}
'''

CELL_FLIGHT = r'''# ── Flight loop: per condition reconstruct -> gates -> score blocks -> ship ──
ANCHOR = anchor_rows(SMOKE)
HELDOUT = heldout_fc_rows(SMOKE)
SHAMS = sham_rows(SMOKE)
TITR = titration_rows(SMOKE)
print(f'blocks per condition: anchor {len(ANCHOR)} | heldout_fc {len(HELDOUT)} | '
      f'shams {len(SHAMS)} | titration {len(TITR)}')

def fly_condition(cond):
    """One condition end-to-end inside ONE function scope (lane law: model,
    layer handles, activations all die on return). G2/G1b failures abort the
    condition loudly — scoring a mis-reconstructed model is worthless."""
    t0 = time.time()
    bundle = {'condition': cond, 'mode': MODE, 'stamp': STAMP,
              'src_flight': 'inflight_20260822_2329'}
    try:
        src = SRC_BUNDLE[cond]
        bundle['g2'] = check_bundle(src)
        assert bundle['g2']['pass'], f'G2 bundle integrity failed: {bundle["g2"]}'
        dirs14, mu14 = load_stimulus(src)
        model = build_condition_model(cond)
        layer_mods = resolve_layers(model)
        bundle['g1b'] = gate_g1b(retention_ppl(model), src['ppl_post'])
        assert bundle['g1b']['pass'], (
            f'G1b reconstruction failed: {bundle["g1b"]} — the rebuilt model '
            'is not the flight\'s model')
        print(f'  {cond}: G2 + G1b PASS (ppl delta '
              f'{bundle["g1b"]["delta_pct"]:+.3f}%) — {ram_report()}')
        def run_block(name, rows, fn):
            out = []
            for i, t in enumerate(rows):
                out.append(fn(model, layer_mods, dirs14, mu14, t, cond))
                if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                    print(f'    {cond} {name}: {i + 1}/{len(rows)} '
                          f'({time.time() - t0:.0f}s)')
            return out
        bundle['anchor'] = run_block('anchor', ANCHOR, score_trial)
        bundle['g1'] = gate_g1(bundle['anchor'])
        print(f'  {cond}: G1 anchor exact {bundle["g1"]["exact"]}/{bundle["g1"]["n"]}'
              + ('' if SMOKE else f' (gate >= {ANCHOR_MIN_EXACT})'))
        bundle['heldout'] = run_block('heldout_fc', HELDOUT, score_trial)
        bundle['shams'] = run_block('shams', SHAMS, score_trial)
        bundle['titration'] = run_block('titration', TITR, run_trial_free)
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
            if b.get('titration'):
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
        print(f'{cond}: done in {bundle.get("secs", "?")}s — shipped')
        free_ram()
        print(f'{cond}: torn down — {ram_report()}')
'''

CELL_VERDICT = r'''# ── Gates, primaries, secondaries, banner, ship ──────────────────────────────
summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID,
           'src_flight': 'inflight_20260822_2329',
           'seed_new': E8R2_SEED, 'held_out': HELD_OUT,
           'cond_errors': cond_errors, 'conditions': {}}

COMPLETE = [c for c in CONDITIONS if RESULTS.get(c, {}).get('titration')]
summary['complete_conditions'] = COMPLETE

for cond in COMPLETE:
    b = RESULTS[cond]
    ho = b['heldout']
    summary['conditions'][cond] = {
        'gates': {'g1': b.get('g1'), 'g1b': b.get('g1b'), 'g2': b.get('g2')},
        'n_heldout': len(ho),
        'heldout_median_err': (round(float(np.median([r['err'] for r in ho])), 2)
                               if ho else None),
        'heldout_exact': sum(1 for r in ho if r['exact']),
        'heldout_exact_rate': round(sum(1 for r in ho if r['exact']) / max(1, len(ho)), 4),
        'heldout_median_rank': (round(float(np.median([r['rank'] for r in ho])), 1)
                                if ho else None),
        'none_top_rate': round(sum(1 for r in ho if r['none_top']) / max(1, len(ho)), 4),
        'heldout_name_argmax_rate': round(sum(1 for r in ho if r['argmax'] in HELD_OUT)
                                          / max(1, len(ho)), 4),
        'sum_mean_agreement': round(sum(1 for r in ho if r['agree_sum_mean'])
                                    / max(1, len(ho)), 4),
        'sham_prior': sham_prior(b['shams']),
        'titration': titration_curve(b['titration']),
        'secs': b.get('secs'),
    }

GATES_ALL = bool(COMPLETE) and all(
    RESULTS[c].get(g, {}).get('pass') for c in COMPLETE for g in ('g1', 'g1b', 'g2'))
summary['gates_all_pass'] = GATES_ALL

if not SMOKE and len(COMPLETE) == 2 and GATES_ALL:
    r_ho, s_ho = RESULTS['real']['heldout'], RESULTS['scrambled']['heldout']
    # P-E8R2-1: real held-out forced-choice vs permuted-pairing null
    obs1, p1, _ = perm_null_median(as_perm_rows(r_ho), VEC, n_perm=N_PERM)
    # P-E8R2-2: geometry-specificity — delta median (scrambled - real), CI > 0
    d2 = boot_delta_median(as_perm_rows(s_ho), as_perm_rows(r_ho), n_boot=N_BOOT)
    p2 = 0.049 if (d2 and d2['ci95'][0] > 0) else 1.0
    summary['primaries'] = {
        'P_E8R2_1_heldout_comprehension': {'obs_median_err': obs1, 'p': p1,
                                           'n_rows': len(r_ho)},
        'P_E8R2_2_geometry_specificity': {'delta_scrambled_minus_real': d2,
                                          'pass_ci': bool(d2 and d2['ci95'][0] > 0)},
        'holm': holm({'P1': p1, 'P2': p2}),
    }
    sec = {}
    hits = sum(1 for r in r_ho if r['exact'])
    sec['S1_exact_hits_real'] = {'hits': hits, 'n': len(r_ho),
                                 'chance': round(1 / len(FC_CANDIDATES), 4),
                                 'p_binom': binom_tail(hits, len(r_ho),
                                                       1 / len(FC_CANDIDATES))}
    obs_rk, p_rk = perm_null_rank(r_ho, n_perm=N_PERM)
    sec['S2_rank_real'] = {'obs_median_rank': obs_rk, 'p': p_rk,
                           'mrr': round(float(np.mean([1 / r['rank'] for r in r_ho])), 4)}
    sec['S3_none_top'] = {c: summary['conditions'][c]['none_top_rate'] for c in COMPLETE}
    sec['S4_heldout_name_argmax'] = {
        c: {'rate': summary['conditions'][c]['heldout_name_argmax_rate'],
            'correct_subset': sum(1 for r in RESULTS[c]['heldout']
                                  if r['argmax'] in HELD_OUT and r['exact'])}
        for c in COMPLETE}
    sec['S5_sham_prior'] = {c: summary['conditions'][c]['sham_prior'] for c in COMPLETE}
    sec['S6_titration'] = {c: summary['conditions'][c]['titration'] for c in COMPLETE}
    med_sum = float(np.median([r['err_sum'] for r in r_ho]))
    sec['S7_robustness_sum_logprob'] = {
        'agreement_rate': summary['conditions']['real']['sum_mean_agreement'],
        'real_heldout_median_err_sum': round(med_sum, 2)}
    ofl = oracle_floor(VEC)
    below = sum(1 for r in r_ho if r['err'] is not None
                and r['err'] < ofl[r['injected']]['floor_deg'])
    sec['S8_oracle_floor'] = {'floors': ofl,
                              'real_rows_below_floor': below,
                              'rate_below_floor': round(below / max(1, len(r_ho)), 4)}
    sec['X_per_concept_real'] = {
        c: {'n': len([r for r in r_ho if r['injected'] == c]),
            'median_err': round(float(np.median([r['err'] for r in r_ho
                                                 if r['injected'] == c])), 2),
            'exact': sum(1 for r in r_ho if r['injected'] == c and r['exact']),
            'modal_argmax': max(set(x['argmax'] for x in r_ho if x['injected'] == c),
                                key=[x['argmax'] for x in r_ho
                                     if x['injected'] == c].count)}
        for c in HELD_OUT}
    sec['X_per_alpha_real'] = {
        str(a): {'n': len([r for r in r_ho if r['alpha'] == a]),
                 'median_err': round(float(np.median([r['err'] for r in r_ho
                                                      if r['alpha'] == a])), 2),
                 'exact': sum(1 for r in r_ho if r['alpha'] == a and r['exact'])}
        for a in FC_ALPHAS}
    summary['secondaries'] = sec

fn = OUT / 'e8r2_verdict.json'
jdump(summary, fn)
if SMOKE or len(COMPLETE) == 2:
    ship(OUT, f'e8r2/{MODE}_{STAMP}')

def _no_rows(o):
    if isinstance(o, dict):
        return {k: _no_rows(v) for k, v in o.items()
                if k not in ('scores', 'scores_sum', 'argmax_dist')}
    if isinstance(o, list):
        return [_no_rows(x) for x in o]
    return o
print(json.dumps(_no_rows(summary), indent=1, default=str))

if SMOKE:
    checks = {
        'no_condition_errors': not cond_errors,
        'both_conditions_flew': len(COMPLETE) == 2,
        'g2_pass_both': all(RESULTS[c].get('g2', {}).get('pass') for c in COMPLETE),
        'g1b_pass_both': all(RESULTS[c].get('g1b', {}).get('pass') for c in COMPLETE),
        'blocks_complete': all(
            len(RESULTS[c].get('anchor', [])) == len(ANCHOR)
            and len(RESULTS[c].get('heldout', [])) == len(HELDOUT)
            and len(RESULTS[c].get('shams', [])) == len(SHAMS)
            and len(RESULTS[c].get('titration', [])) == len(TITR)
            for c in COMPLETE),
        'hooks_fired_on_scored_injects': all(
            r['hook_calls'] >= 1
            for c in COMPLETE for r in RESULTS[c].get('anchor', [])
            + RESULTS[c].get('heldout', [])),
        'shams_uninjected': all(
            r['hook_calls'] == 0 for c in COMPLETE
            for r in RESULTS[c].get('shams', [])),
        'score_vectors_complete': all(
            set(r['scores']) == set(SCORED_SET)
            for c in COMPLETE for r in RESULTS[c].get('heldout', [])),
        'titration_parses_ok': all(
            sum(1 for r in RESULTS[c].get('titration', [])
                if r['report'] == 'INVALID') <= len(TITR) // 2
            for c in COMPLETE),
        'shipped': all((SEM / INFLIGHT / f'condition_{c}.json').exists()
                       for c in COMPLETE),
    }
    ok = all(checks.values())
    print('smoke checks:', json.dumps(checks, indent=1))
    banner = ('SMOKE GREEN — set SMOKE=False, Runtime > Restart runtime, Run '
              'all. Full mode flies ONE condition per run (~12-18 min); follow '
              'the end banner between runs.'
              if ok else 'SMOKE RED — do not fly full; send Fable the output')
    print('\n' + '=' * 66 + f'\n  {banner}\n' + '=' * 66)
elif len(COMPLETE) == 2:
    if GATES_ALL:
        p = summary['primaries']
        print('\n' + '=' * 66)
        print('  FULL FLIGHT COMPLETE 2/2 — verdict shipped to MyDrive/semcore/e8r2/')
        print(f"  P1 held-out comprehension: median "
              f"{p['P_E8R2_1_heldout_comprehension']['obs_median_err']:.1f} deg, "
              f"p={p['P_E8R2_1_heldout_comprehension']['p']:.4f}")
        d = p['P_E8R2_2_geometry_specificity']['delta_scrambled_minus_real']
        print(f"  P2 geometry-specificity: delta {d['delta']:+.1f} deg, "
              f"CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]")
        print('=' * 66)
    else:
        print('\n' + '=' * 66)
        print('  GATE FAILURE — primaries withheld per pre-registration.')
        print('  One engineering re-fly is pre-authorized after the instrument fix.')
        print('  Send Fable the verdict json (gates block has the diagnosis).')
        print('=' * 66)
else:
    left = [c for c in CONDITIONS if c not in COMPLETE]
    print('\n' + '=' * 66)
    print(f'  PARTIAL — {len(COMPLETE)}/2 conditions shipped '
          f'({", ".join(COMPLETE) or "none"}); left: {", ".join(left)}')
    print(f"  Next run: Runtime > Restart runtime, set RESUME_STAMP = "
          f"'{RESUME_STAMP or STAMP}',")
    print('  then Run all. Finished conditions reload from Drive in seconds;')
    print('  primaries compute only when both have landed (pre-reg hygiene).')
    print('=' * 66)
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
add("code", E8R_LOGIC_SRC)
add("code", E8R2_SRC)
add("code", CELL_MODEL)
add("code", CELL_FLIGHT)
add("code", CELL_VERDICT)

out_nb = os.path.join(HERE, "E8R2_FORCEDCHOICE_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_nb, f"({len(nb['cells'])} cells)")

logic_out = os.path.join(HERE, "e8r2_logic.py")
with open(logic_out, "w") as f:
    f.write(E8R_LOGIC_SRC + "\n\n" + E8R2_SRC)
print("wrote", logic_out)
