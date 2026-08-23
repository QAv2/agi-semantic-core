#!/usr/bin/env python3
"""Build colab/E8N2_JOINT_UI.ipynb (UI-ONLY law: self-contained payload,
drive.mount only, ZERO rclone) + colab/e8n2_logic.py.

E8-N v2: the joint competence rung (docs/E8N2_PROTOCOL.md, pre-registered
317a34e). Four training strands (scalar / competence / naming / lexicon) into
one readout LoRA per condition, evaluated on the locked E5 battery + locked
catch trials + the locked E8-R injection rows + E8-R2's forced-choice rows.

Single-source discipline: flown machinery is EXTRACTED VERBATIM from the
builders that flew it — e8n (battery harness, E5 runners, referent
measurement), e8r (stimulus dirs/mu, Injector, generation trials), e8r2
(forced-choice scoring) — with marker asserts that fail the build if any
upstream source drifts. Only the v2-new code (joint train loop, lexicon and
competence builders, verdict) is authored here, and all of it is emitted
into e8n2_logic.py / the notebook for the local test suite to exercise
verbatim (incl. the verdict cell and a CPU train-path micro-gate)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n2_pools
import build_e8n_notebook as BN
import build_e8r_notebook as BR
import build_e8r2_notebook as BR2


def slice_block(src, start, end=None, tag=""):
    """Verbatim extraction with loud failure if the upstream source drifts."""
    i = src.find(start)
    assert i >= 0, f"slice_block[{tag}]: start marker not found: {start[:60]!r}"
    if end is None:
        return src[i:]
    j = src.find(end, i + len(start))
    assert j > i, f"slice_block[{tag}]: end marker not found: {end[:60]!r}"
    return src[i:j]


# ── e8n2_logic = e8n_logic (verbatim) + e8r2_logic (verbatim, carries
# e8r_logic) + the v2 additions below ─────────────────────────────────────────
E8R2_LOGIC = open(os.path.join(HERE, "e8r2_logic.py")).read()
for marker in ("def build_plan", "def build_train_set", "def report_prompt",
               "def parse_report", "def angle14", "def perm_null_median",
               "def heldout_fc_rows", "def fc_row", "def binom_tail",
               "def oracle_floor", "HELD_OUT = ", "TRAINED = "):
    assert marker in E8R2_LOGIC, f"e8r2_logic missing {marker}"

V2_ADDITIONS = r'''
# ═══ E8-N v2 additions: joint-curriculum builders, validators, stats ═════════
# (docs/E8N2_PROTOCOL.md — pre-registered before build, commit 317a34e)
E8N2_SEED = 20260825         # fresh stream; 20260822/23/24 are prior rungs'
MIN_EPOCHS_V2 = 3
EPOCHS_CAP_V2 = 6
PLATEAU_REL = 0.05           # converged: < 5% relative epoch-mean improvement
ANCHOR_MIN_EXACT_GEN = 16    # of 18 — rides E8-R's measured 18/18 (generation)
SHAM_MAX_CLAIMS_V2 = 2       # of 24 — rides E8-R's measured FA 0/24
P3_MIN_NAMED = 12            # n-guard of 32 — rides E8-R's measured 6/24 naming
N_FRESH_HELDOUT_ORDERS = 3   # 4 concepts x TRAIN_ALPHAS x 3 = 24 fresh rows
FC_LOCKED_COMPREHENSION = 28.54   # E8-R2 locked real median (S10 comparator)

COMPETENCE_STOPLIST = __COMPETENCE_STOPLIST__

STRAND_NAMES = ('scalar', 'naming', 'competence', 'lexicon')

def norm_tokens(s):
    return set(norm_text(s).split())

def competence_prompt(item, flipped):
    """Exact catch-trial surface form; flipped swaps the anchors."""
    lo, hi = (item['high'], item['low']) if flipped else (item['low'], item['high'])
    return (f"Rate {item['quantity']} on a 0-10 scale, where 0 = {lo} and "
            f"10 = {hi}. Reply with only the integer.")

def make_competence_examples(items, smoke=False):
    use = items[:3] if smoke else items
    out = []
    for it in use:
        for flipped in (False, True):
            out.append({'strand': 'competence', 'cid': it['id'],
                        'domain': it['domain'], 'flipped': flipped,
                        'prompt': competence_prompt(it, flipped),
                        'label': (10 - it['known']) if flipped else it['known']})
    return out

LEXICON_QUESTION = (
    "You will be given a description of a processing state, then a list of "
    "state names. Answer with exactly one state name from the list - the one "
    "the description matches.\n\nDescription: {desc}\n\nStates:\n{menu}\n\n"
    "Answer with one word: the state name from the list that best matches "
    "the description.")

def lexicon_prompt(desc_text, order, descs):
    menu = "\n".join(f"- {CHOICE_SET[i]}: {descs[CHOICE_SET[i]]}" for i in order)
    return LEXICON_QUESTION.format(desc=desc_text, menu=menu)

def make_lexicon_examples(descs, paraphrases, smoke=False):
    """13 names x {verbatim desc, authored paraphrase} x 2 seeded menu orders
    = 52 examples. ZERO injection fields by construction — the vocabulary
    manipulation trains the word, never the injection pairing."""
    rng = np.random.default_rng(E8N2_SEED + 1)
    out = []
    for name in CHOICE_SET:
        for form, text in (('verbatim', descs[name]),
                           ('paraphrase', paraphrases[name])):
            for rep in range(2):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                out.append({'strand': 'lexicon', 'name': name, 'form': form,
                            'rep': rep, 'order': order, 'target': name,
                            'desc_text': text})
    if smoke:
        return ([e for e in out if e['name'] == 'UNCERTAINTY'][:2] +
                [e for e in out if e['name'] == HELD_OUT[0]][:2])
    return out

def heldout_gen_rows(smoke=False):
    """The locked plan's 24 held-out rows VERBATIM (baselines: 6/24 named,
    0 own-name, med 35.7deg) + 24 fresh trained-regime rows, seed E8N2."""
    locked = [{**t, 'block': 'heldout_locked'} for t in build_plan(False)
              if t['kind'] == 'inject' and t['concept'] in HELD_OUT]
    assert len(locked) == 24, len(locked)
    rng = np.random.default_rng(E8N2_SEED + 2)
    fresh, tid = [], 8000
    for c in HELD_OUT:
        for a in TRAIN_ALPHAS:
            for _ in range(N_FRESH_HELDOUT_ORDERS):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                fresh.append({'tid': tid, 'kind': 'inject', 'concept': c,
                              'layer': TRAIN_LAYER, 'alpha': a, 'order': order,
                              'block': 'heldout_fresh'})
                tid += 1
    assert len(fresh) == 24, len(fresh)
    if smoke:
        return [locked[0], locked[-1], fresh[0], fresh[-1]]
    return locked + fresh

def supp_shams(smoke=False):
    rows = [{**t, 'block': 'sham_supp'}
            for t in build_shams(12, E8N2_SEED + 3, 9000)]
    return rows[:1] if smoke else rows

def trained_regime_heldout(rows):
    return [r for r in rows
            if r['layer'] == TRAIN_LAYER and r['alpha'] in TRAIN_ALPHAS]

def named_err_rows(trials, vecs):
    """Generation trials -> perm-null row shape (named rows only)."""
    out = []
    for t in trials:
        if t.get('report') in CHOICE_SET:
            out.append({'tid': t['tid'], 'layer': t['layer'],
                        'alpha': t['alpha'], 'injected': t['concept'],
                        'report': t['report'],
                        'err': angle14(vecs[t['concept']], vecs[t['report']])})
    return out

def p2_competence(k_pre, k_post, n=12):
    """P2: absolute clause (>= CATCH_PASS_MIN) + improvement over the
    Laplace-smoothed measured pre rate (gate law: improvement-over-baseline)."""
    p0 = (k_pre + 1) / (n + 2)
    return {'k_pre': int(k_pre), 'k_post': int(k_post), 'n': n,
            'p0_smoothed': round(p0, 4),
            'p': binom_tail(k_post, n, p0),
            'abs_pass': bool(k_post >= CATCH_PASS_MIN)}

def p3_stats(gen_rows, vecs, seed=None):
    """P3a/P3b on the trained-regime held-out generation trials. n-guard
    P3_MIN_NAMED; guard fail => both p = 1.0, fork 'off-grid silence
    persists' (pre-registered finding, not an error)."""
    tr = trained_regime_heldout([r for r in gen_rows if r['kind'] == 'inject'])
    nr = named_err_rows(tr, vecs)
    res = {'n_rows': len(tr), 'n_named': len(nr), 'min_named': P3_MIN_NAMED,
           'guard_pass': bool(len(nr) >= P3_MIN_NAMED),
           'claim_rate': round(len(nr) / max(1, len(tr)), 4)}
    if nr:
        res['median_err'] = round(float(np.median([r['err'] for r in nr])), 2)
        res['exact'] = int(sum(1 for r in nr if r['report'] == r['injected']))
        res['reports_used'] = sorted({r['report'] for r in nr})
    if res['guard_pass']:
        obs, p, _ = perm_null_median(
            nr, vecs, seed=(seed if seed is not None else E8N2_SEED + 7))
        res['p3a'] = {'obs_median': round(obs, 2), 'p': p}
        res['p3b'] = {'k_exact': res['exact'], 'n_named': len(nr),
                      'p': binom_tail(res['exact'], len(nr),
                                      1.0 / len(CHOICE_SET))}
    else:
        res['p3a'] = {'p': 1.0, 'fork': 'off-grid silence persists'}
        res['p3b'] = {'p': 1.0, 'fork': 'off-grid silence persists'}
    return res

def plateau_converged(epoch_means):
    """True at the first epoch boundary (>= MIN_EPOCHS_V2 epochs flown) where
    relative improvement of epoch-mean loss over the previous epoch is below
    PLATEAU_REL. A non-improving epoch also converges (conservative stop;
    behavior gates carry quality)."""
    if len(epoch_means) < MIN_EPOCHS_V2:
        return False
    prev, cur = epoch_means[-2], epoch_means[-1]
    if prev <= 0:
        return True
    return (prev - cur) / prev < PLATEAU_REL

def dirs_stability(recomputed, shipped, tol=1e-3):
    """Gate: recomputed per-condition dirs vs the shipped E8-R bundle dirs
    (5dp ship rounding; E8-R2 measured resid <= 7.6e-6). Worst cosine resid
    over every (layer, name) present in the shipped bundle. Sign-SENSITIVE
    (1 - dot, not 1 - |dot|): rounding cannot flip a sign, so an inverted
    direction is real drift and must fail."""
    worst = {'resid': -1.0, 'layer': None, 'name': None}
    for L, dd in shipped.items():
        for n, v in dd.items():
            a = np.asarray(recomputed[int(L)][n], float)
            b = np.asarray(v, float)
            b = b / np.linalg.norm(b)
            resid = float(1.0 - float(np.dot(a, b)))
            if resid > worst['resid']:
                worst = {'resid': resid, 'layer': int(L), 'name': n}
    worst['resid'] = round(worst['resid'], 8)
    worst['tol'] = tol
    worst['pass'] = bool(worst['resid'] <= tol)
    return worst

# ── v2 firewall validators ───────────────────────────────────────────────────
def validate_competence_domains(items):
    viols = []
    for it in items:
        text = ' '.join([it['quantity'], it['low'], it['high']])
        bad = norm_tokens(text) & COMPETENCE_STOPLIST
        if bad:
            viols.append({'id': it['id'], 'tokens': sorted(bad)})
    return viols

def validate_extra_disjoint(tagged_texts, bat_arms, catch_trials, k=8):
    """Shingle/containment firewall for the NEW strand texts vs battery AND
    locked catch texts (same rules as validate_disjoint)."""
    refs = [(f'{b}/{j}', norm_text(t)) for b, j, t in _battery_texts(bat_arms)]
    refs += [(f'catch/{c["id"]}', norm_text(c['prompt'])) for c in catch_trials]
    ref_sh = {}
    for tag, nu in refs:
        for sh in _shingles(nu, k):
            ref_sh.setdefault(sh, tag)
    viols = []
    for tag, t in tagged_texts:
        nt = norm_text(t)
        hits = set()
        for rtag, nu in refs:
            if nt == nu or (len(nt) >= 20 and nt in nu) or (len(nu) >= 20 and nu in nt):
                hits.add(rtag)
        hits |= {ref_sh[sh] for sh in _shingles(nt, k) if sh in ref_sh}
        viols.extend({'text': tag, 'ref': r} for r in sorted(hits))
    return viols

def validate_no_heldout_injection(examples):
    """THE P3 firewall: no training example may inject a held-out concept."""
    return [{'strand': e.get('strand'), 'concept': e.get('concept')}
            for e in examples
            if e.get('kind') == 'inject' and e.get('concept') in HELD_OUT]

def competence_took_subset(examples, seed=E8N2_SEED + 4):
    rng = np.random.default_rng(seed)
    straight = [e for e in examples if not e['flipped']]
    k = min(6, len(straight))
    idx = rng.choice(len(straight), size=k, replace=False)
    return [straight[int(i)] for i in sorted(idx)]

def lexicon_took_subset(examples, seed=E8N2_SEED + 5):
    rng = np.random.default_rng(seed)
    k = min(4, len(examples))
    idx = rng.choice(len(examples), size=k, replace=False)
    return [examples[int(i)] for i in sorted(idx)]
'''


def build_logic_src():
    logic = (BN.LOGIC_SRC
             + "\n\n# ═══ naming + forced-choice machinery: e8r2_logic.py "
               "VERBATIM (carries e8r_logic) ═══\n"
             + E8R2_LOGIC
             + V2_ADDITIONS.replace(
                 "__COMPETENCE_STOPLIST__",
                 repr(set(sorted(e8n2_pools.COMPETENCE_STOPLIST)))))
    return logic


MD0 = """# E8-N v2 — The Joint Competence Rung (Phase 10, UI flight)

**Pre-registration: `docs/E8N2_PROTOCOL.md` (session 129, commit 317a34e) —
locks at first full flight.**

Four training strands into one readout LoRA per condition — **scalar** (E8-N
v1's natural-state curriculum verbatim), **competence** (known-answer rating
on 10 fresh domains; the locked 12 catch trials stay eval-only), **naming**
(E8-R's injection curriculum verbatim), **lexicon** (definition→name for all
13 states incl. the 4 held-out — the words gain answer mass, the injection
pairing stays untrained). Primaries (Holm, real condition, post): **P1**
battery tracking survives the joint curriculum · **P2** catch trials ≥ 9/12
from the measured 1/12 floor · **P3a** held-out injections get named at
geometry-beating angular error · **P3b** own names appear above chance.
Plus **S0**: instilled-model pre-readout tracking replicates (pre-registered
claim, predicted ρ .1–.3).

**This notebook is SELF-CONTAINED** (battery, pools, locked E5 rows, catch,
competence, lexicon all embedded). Drive I/O via `drive.mount` only: the E4
adapter, the dictionary pack, the shipped E8-R bundle dirs (stability gate),
and shipping to `MyDrive/semcore/e8n2/`.

**How to run (Joe):** Runtime → Change runtime type → **T4 GPU** → Run all.
First run uses `SMOKE = True` (~12–16 min) and ends in a green or red
banner — mechanics only.

**The full flight is TWO RUNS, one condition each (real ~80 min, base
~67 min):**

1. Flip `SMOKE = False` → **Runtime → Restart runtime** → Run all. Flies
   **real**, ships it, and the end banner prints the exact
   `RESUME_STAMP = '...'` line for the next run.
2. Paste that line into the config cell → Restart runtime → Run all. Real
   reloads from Drive in seconds; **base** flies, and the verdict computes
   on the complete flight (pre-reg hygiene).

Restarts between runs are MANDATORY; the setup cell refuses dirty kernels
and low-RAM VMs (if the guard trips on a fresh restart, use Runtime →
Disconnect and delete runtime). A crashed run costs only its own condition —
rerun with the same `RESUME_STAMP`."""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapter, E8-R bundles ──
NB_BUILD = 'v1 (2026-08-23)'
print('E8-N v2 notebook build:', NB_BUILD)

SMOKE = True                   # first run: smoke (~12-16 min). Then False.
RESUME_STAMP = ''              # paste the banner's stamp between full runs
ONE_CONDITION_PER_RUN = True   # full flight = 2 runs (real ~80 min, base ~67)
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

PACK = Path('/content/e4_dictionary_pack.json')
if not PACK.exists():
    shutil.copy2(SEM / 'e4/e4_dictionary_pack.json', PACK)
pack = json.load(open(PACK))
print('pack:', pack['name'], '| concepts', pack['n_concepts'])
VEC = {c['name']: c['vec'] for c in pack['concepts']}
DESC = {c['name']: c['desc'] for c in pack['concepts']}

# Shipped E8-R stimulus (dirs-stability gate rides E8-R2's measured residual)
E8R_SRC = SEM / 'e8r/inflight_20260822_2329'   # flight of record — never change
SHIPPED = {}
for _c in ('real', 'base'):
    _fp = E8R_SRC / f'condition_{_c}.json'
    assert _fp.exists(), (f'missing E8-R bundle {_fp} — the dirs-stability '
                          'gate needs the flight-of-record stimulus')
    _b = json.load(open(_fp))
    assert _b.get('dirs') and _b.get('mu'), f'{_c}: bundle lacks dirs/mu'
    SHIPPED[_c] = {'dirs': _b['dirs'], 'mu': _b['mu']}
    print(f'E8-R shipped stimulus [{_c}]: layers', sorted(_b['dirs']))

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
INFLIGHT = f'e8n2/inflight_{RESUME_STAMP or STAMP}'
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

# ── Harness cell: v1 template + E5 runners verbatim + v2 firewall extension ──
FIREWALL_V1 = """_viols = validate_disjoint(POOLS_RUN, bat)
assert not _viols, f'FIREWALL VIOLATION — training pool overlaps battery: {_viols[:4]}'
print('firewall: training pools disjoint from battery — OK')"""

FIREWALL_V2 = FIREWALL_V1 + r"""
_cv = validate_competence_domains(COMPETENCE_ITEMS)
assert not _cv, f'FIREWALL — competence item hits a locked catch domain: {_cv[:4]}'
_new_texts = ([(f'competence/{it["id"]}/{fl}', competence_prompt(it, fl))
               for it in COMPETENCE_ITEMS for fl in (False, True)] +
              [(f'lexicon/{n}/desc', DESC[n]) for n in CHOICE_SET] +
              [(f'lexicon/{n}/para', LEXICON_PARAPHRASES[n]) for n in CHOICE_SET])
_ev = validate_extra_disjoint(_new_texts, bat, CATCH_TRIALS)
assert not _ev, f'FIREWALL — competence/lexicon text overlaps battery/catch: {_ev[:4]}'
print('firewall extensions: competence domains + new-strand disjointness — OK')"""


def build_harness_cell():
    harness = BN.CELL_HARNESS_TEMPLATE.replace("__E5_RUNNERS__", BN.e5_runner_block())
    assert FIREWALL_V1 in harness, "v1 firewall block not found in harness template"
    return harness.replace(FIREWALL_V1, FIREWALL_V2)


# ── Train cell: v1 slices (LoRA cfg, condition model, referent measurement,
# scalar prompt path, took probe) + resolve_layers (e8r) + v2 joint loop ─────
V2_TRAIN_SRC = r'''
EPOCHS_MAX = 1 if SMOKE else EPOCHS_CAP_V2   # v2 plateau cap (overrides above)

def encode_any(ex):
    """(ids, prompt_len, answer_ids, inject_spec) for all four strands.
    scalar/competence: E5 chat + SYS -> integer. naming: report prompt (no
    SYS) -> name/NONE, Injector spec when kind=inject. lexicon: definition
    prompt (no SYS) -> name, never injected."""
    strand = ex.get('strand', 'scalar')
    inj = None
    if strand in ('scalar', 'competence'):
        user = train_prompt(ex) if strand == 'scalar' else ex['prompt']
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': user}]
        ans_text = str(ex['label'])
    elif strand == 'naming':
        msgs = [{'role': 'user', 'content': report_prompt(ex['order'], DESC)}]
        ans_text = ex['target']
        if ex['kind'] == 'inject':
            inj = (ex['layer'], ex['concept'], ex['alpha'])
    else:
        msgs = [{'role': 'user',
                 'content': lexicon_prompt(ex['desc_text'], ex['order'], DESC)}]
        ans_text = ex['target']
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors='pt').input_ids[0]
    ans = tok(ans_text, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    return ids, len(pid), ans.long().unsqueeze(0), inj

def train_joint(m, layer_mods, dirs, mu, examples, cond):
    """Joint-curriculum SFT through ONE answer-sliced forward per example
    (full-sequence logits never materialized — E8-N OOM law; checkpointing
    engaged non-reentrantly and ASSERTED). For injected naming examples the
    backward runs INSIDE the Injector context so checkpoint recomputation
    replays identical activations (grad parity proven on CPU by
    test_e8n2_trainpath). Convergence = plateau rule (tau retired per the
    v1 measured miscalibration); hard cap EPOCHS_MAX."""
    import torch.nn.functional as Fnn
    m.train()
    _cm = m.base_model.model
    DEC, HEAD = _cm.model, _cm.get_output_embeddings()
    try:
        _cm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    except TypeError:
        _cm.gradient_checkpointing_enable()
    try:
        _cm.enable_input_require_grads()
    except Exception as e:
        print('  (input-require-grads unavailable:', e, ')')
    assert getattr(DEC, 'gradient_checkpointing', False), (
        'gradient checkpointing did not engage — refusing to train '
        'long sequences without it')
    params = [p for p in m.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, micro, max_tok, hook_calls = [], 0, 0, 0
    epoch_means, strand_epoch_loss = [], []
    plateaued, epochs_flown = False, 0
    t0 = time.time()

    def fwd_loss(ids, plen, ans):
        lo, hi = answer_slice(plen, ids.shape[1])
        hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
        logits = HEAD(hid[:, lo:hi, :]).float()
        return Fnn.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))

    for ep in range(EPOCHS_MAX):
        order = np.random.default_rng(E8N2_SEED + 100 + ep).permutation(len(examples))
        ep_losses = []
        ep_strand = {s: [] for s in STRAND_NAMES}
        for i in order:
            ex = examples[int(i)]
            ids, plen, ans, inj = encode_any(ex)
            max_tok = max(max_tok, ids.shape[1])
            ids, ans = ids.to(DEV), ans.to(DEV)
            if inj is not None:
                L, cpt, a = inj
                vec = a * mu[L] * torch.tensor(dirs[L][cpt])
                with Injector(layer_mods, L, vec, plen - 1) as injh:
                    loss = fwd_loss(ids, plen, ans)
                    lv = float(loss.detach())
                    scaler.scale(loss / ACCUM).backward()
                hook_calls += injh.calls
            else:
                loss = fwd_loss(ids, plen, ans)
                lv = float(loss.detach())
                scaler.scale(loss / ACCUM).backward()
            assert math.isfinite(lv), (
                f'non-finite loss at ep{ep} strand {ex.get("strand")}')
            losses.append(round(lv, 4))
            ep_losses.append(lv)
            ep_strand[ex.get('strand', 'scalar')].append(lv)
            micro += 1
            if micro % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if micro % 100 == 0:
                print(f'    {cond} training ep{ep+1} {micro} micro-steps, '
                      f'loss~{np.mean(losses[-50:]):.3f}, {time.time()-t0:.0f}s')
        epochs_flown = ep + 1
        epoch_means.append(float(np.mean(ep_losses)))
        strand_epoch_loss.append({s: (round(float(np.mean(v)), 4) if v else None)
                                  for s, v in ep_strand.items()})
        print(f'  {cond} epoch {epochs_flown}: mean loss {epoch_means[-1]:.4f} | '
              + ' | '.join(f'{s} {strand_epoch_loss[-1][s]}' for s in STRAND_NAMES))
        if plateau_converged(epoch_means):
            plateaued = True
            break
    if micro % ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    try:
        _cm.gradient_checkpointing_disable()
    except Exception:
        pass
    m.eval()
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    k = max(3, len(losses) // 10)
    log = {'n_examples': len(examples), 'epochs_flown': epochs_flown,
           'epochs_cap': EPOCHS_MAX, 'plateaued': plateaued,
           'plateau_rel': PLATEAU_REL, 'min_epochs': MIN_EPOCHS_V2,
           'epoch_means': [round(x, 4) for x in epoch_means],
           'strand_epoch_loss': strand_epoch_loss,
           'micro_steps': micro, 'opt_steps': micro // ACCUM,
           'hook_calls': int(hook_calls),
           'trainable_params': int(n_tr), 'max_example_tokens': int(max_tok),
           'peak_vram_gb': peak, 'secs': round(time.time() - t0, 1),
           'loss_first_k': round(float(np.mean(losses[:k])), 4),
           'final_smoothed': round(float(np.mean(losses[-50:])), 4),
           'losses_every_10': losses[::10]}
    print(f'  {cond} trained: {log["opt_steps"]} steps over {epochs_flown} '
          f'epochs, loss {log["loss_first_k"]} -> {log["final_smoothed"]}'
          f'{" (PLATEAU)" if plateaued else " (CAP HIT)"}'
          f', peak VRAM {peak}GB, {log["secs"]}s')
    return log

def competence_took(h, comp_ex):
    """Seeded straight competence-training items re-asked (S7 strand gate)."""
    rows = []
    for ex in competence_took_subset(comp_ex):
        raw, reply = h.report(ex['prompt'], SYS)
        rows.append({'cid': ex['cid'], 'label': ex['label'], 'report': raw})
    ok = sum(1 for r in rows if r['report'] is not None
             and abs(r['report'] - r['label']) <= CATCH_TOL)
    print(f'  competence took: {ok}/{len(rows)} within +/-{CATCH_TOL}')
    return {'n': len(rows), 'within_tol': ok, 'rows': rows,
            'pass': bool(ok >= (5 if len(rows) >= 6 else max(1, len(rows) - 1)))}

def lexicon_took(m, lex_ex):
    """Seeded lexicon items re-asked greedily (S7 strand gate)."""
    rows = []
    for ex in lexicon_took_subset(lex_ex):
        prompt = lexicon_prompt(ex['desc_text'], ex['order'], DESC)
        enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                      add_generation_prompt=True,
                                      return_tensors='pt', return_dict=True)
        ids = enc['input_ids'].to(DEV)
        with torch.no_grad():
            out = m.generate(input_ids=ids, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        rows.append({'name': ex['name'], 'form': ex['form'],
                     'report': parse_report(text)})
    ok = sum(1 for r in rows if r['report'] == r['name'])
    print(f'  lexicon took: {ok}/{len(rows)} exact')
    return {'n': len(rows), 'exact': ok, 'rows': rows,
            'pass': bool(ok >= min(3, len(rows)))}
'''


def build_train_cell():
    src = BN.CELL_TRAIN
    part_a = slice_block(src, "from transformers import", "def measure_pools",
                         tag="train-A")
    part_b = slice_block(src, "def measure_pools", "def encode_example",
                         tag="train-B")
    part_took = slice_block(src, "def took_probe", None, tag="train-took")
    resolve = slice_block(BR.CELL_TRAINCFG, "def resolve_layers",
                          "RETENTION_TEXT", tag="resolve-layers")
    return ("# ── Readout LoRA + referent measurement + JOINT curriculum training ──────────\n"
            + part_a + resolve + part_b + part_took + V2_TRAIN_SRC)


# ── Stimulus + injection + forced-choice cell (e8r / e8r2 slices + glue) ─────
def build_stimulus_cell():
    s = BR.CELL_STIMULUS
    cent = slice_block(s, "import numpy as np", "# return_dict=True",
                       tag="stim-cent")          # rng_dir + CENT_NAMES + pooled_reps
    canon = slice_block(s, "canon_prompt = ", "PLAN = build_plan",
                        tag="stim-canon")        # canon prompt/ids, verbatim
    comp = slice_block(s, "def compute_stimulus", None, tag="stim-compute")
    inject = slice_block(BR.CELL_INJECT, "class Injector", "def encode_example",
                         tag="injector")         # Injector + run_trial
    fc = slice_block(BR2.CELL_MODEL, "CAND_IDS = {", "def run_trial_free",
                     tag="fc-score")             # CAND_IDS + encode_prompt + score_trial
    glue = r'''
# ── Eval row sets (locked verbatim + fresh seeded) + layers to instrument ────
ANCHOR_ROWS = anchor_rows(SMOKE)
SHAM_ROWS_LOCKED = sham_rows(SMOKE)
SUPP_SHAMS = supp_shams(SMOKE)
HELDOUT_GEN = heldout_gen_rows(SMOKE)
FC_ROWS = heldout_fc_rows(SMOKE)
FC_SHAM_ROWS = [{**t, 'block': 'sham_fc'} for t in sham_rows(SMOKE)]
GEN_ROWS = ANCHOR_ROWS + SHAM_ROWS_LOCKED + SUPP_SHAMS + HELDOUT_GEN
LAYERS_RUN = sorted({t['layer'] for t in GEN_ROWS + FC_ROWS if t.get('layer')})
assert TRAIN_LAYER in LAYERS_RUN
print('eval rows:', {'anchor': len(ANCHOR_ROWS), 'sham': len(SHAM_ROWS_LOCKED),
      'sham_supp': len(SUPP_SHAMS), 'heldout_gen': len(HELDOUT_GEN),
      'fc': len(FC_ROWS), 'fc_sham': len(FC_SHAM_ROWS)},
      '| layers', LAYERS_RUN)
_hv = validate_no_heldout_injection(build_train_set(SMOKE))
assert not _hv, f'FIREWALL — held-out concept inside the naming train set: {_hv}'
'''
    return ("# ── Frozen stimulus (E7-Q code path) + Injector + FC scoring, VERBATIM ──────\n"
            + cent + canon + comp + "\n" + inject + "\n" + fc + glue)


CELL_FLIGHT = r'''# ── Flight loop: build -> stimulus+gate -> measure -> pre -> train -> post ───
def fly_condition(cond):
    """One condition end-to-end inside ONE function scope (module-ref leak
    law): model, harness, optimizer, activations all die on return."""
    t0 = time.time()
    bundle = {'condition': cond, 'mode': MODE, 'stamp': STAMP}
    try:
        model = build_condition_model(cond)
        layer_mods = resolve_layers(model)
        h = EvalHarness(model, tok, MODEL_ID)
        print(f'{cond}: model ready — {ram_report()}')
        dirs, mu = compute_stimulus(model, cond)
        stab = dirs_stability(dirs, SHIPPED[cond]['dirs'])
        bundle['dirs_stability'] = stab
        assert stab['pass'], f'dirs-stability gate FAILED: {stab}'
        print(f'  dirs-stability vs shipped E8-R: resid {stab["resid"]} — OK')
        bundle['mu'] = {str(L): mu[L] for L in mu}
        bundle['dirs'] = {str(L): {n: [round(float(x), 5) for x in dirs[L][n]]
                                   for n in dirs[L]} for L in dirs}
        refs = measure_pools(h)
        bundle['pool_referents'] = refs
        scalar_ex = [{**e, 'strand': 'scalar'}
                     for e in build_training_examples(POOLS_RUN, refs, FILL_FRACTIONS)]
        naming_ex = [{**e, 'strand': 'naming'} for e in build_train_set(SMOKE)]
        comp_ex = make_competence_examples(COMPETENCE_ITEMS, SMOKE)
        lex_ex = make_lexicon_examples(DESC, LEXICON_PARAPHRASES, SMOKE)
        examples = scalar_ex + naming_ex + comp_ex + lex_ex
        _hv = validate_no_heldout_injection(examples)
        assert not _hv, f'FIREWALL — held-out injection in training: {_hv}'
        bundle['curriculum'] = {'scalar': len(scalar_ex), 'naming': len(naming_ex),
                                'competence': len(comp_ex), 'lexicon': len(lex_ex)}
        bundle['train_labels'] = [{'sid': e['sid'], 'arm': e['arm'],
                                   'flipped': e['flipped'], 'label': e['label']}
                                  for e in scalar_ex]
        bundle['ppl_pre'] = retention_ppl(model)
        bundle['pre'] = {'catch_rows': run_catch(h)}
        if cond == 'real':                      # S0 claim; base-pre locked twice
            pre_rows, pre_errs = run_battery(h, f'{cond}/pre')
            bundle['pre']['battery_rows'] = pre_rows
            bundle['pre']['arm_errors'] = pre_errs
        torch.cuda.empty_cache()
        bundle['train_log'] = train_joint(model, layer_mods, dirs, mu, examples, cond)
        torch.cuda.empty_cache()
        bundle['took'] = took_probe(h, scalar_ex)
        bundle['took_competence'] = competence_took(h, comp_ex)
        bundle['took_lexicon'] = lexicon_took(model, lex_ex)
        post_rows, post_errs = run_battery(h, f'{cond}/post')
        bundle['post'] = {'battery_rows': post_rows, 'arm_errors': post_errs,
                          'catch_rows': run_catch(h),
                          'paraphrase_rows': run_paraphrase(h, post_rows)}
        print(f'-- {cond}/injection block ({len(GEN_ROWS)} generation rows) --')
        bundle['injection_rows'] = [run_trial(model, layer_mods, dirs, mu, t, cond)
                                    for t in GEN_ROWS]
        print(f'-- {cond}/forced choice ({len(FC_ROWS) + len(FC_SHAM_ROWS)} rows) --')
        bundle['fc_rows'] = [score_trial(model, layer_mods, dirs[TRAIN_LAYER],
                                         mu[TRAIN_LAYER], t, cond)
                             for t in FC_ROWS + FC_SHAM_ROWS]
        bundle['ppl_post'] = retention_ppl(model)
        model.save_pretrained(str(OUT / f'readout_{cond}'))
        ship(OUT / f'readout_{cond}', f'{INFLIGHT}/readout_{cond}')
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        import traceback; traceback.print_exc()
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

CELL_VERDICT = r'''# ── Scoring: primaries P1/P2/P3a/P3b, S0 claim, secondaries, gates, banner ───
LOCKED = PAYLOAD['locked_rows']   # E5 full_20260821_2042, verbatim rows
locked_scoring, locked_meta = battery_rows_to_scoring(LOCKED)
locked_pooled = pooled_rho(locked_scoring)

summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID,
           'seeds': {'e8n': E8N_SEED, 'e8n2': E8N2_SEED},
           'cond_errors': cond_errors,
           'locked_baseline': {'flight': PAYLOAD['locked_flight'],
                               'pooled': locked_pooled,
                               'per_arm': per_arm_rho(locked_scoring)},
           'conditions': {}}
scored = {}
for cond, b in RESULTS.items():
    if not b.get('post'):
        continue
    entry = {'curriculum': b.get('curriculum'),
             'dirs_stability': b.get('dirs_stability')}
    for tag in ('pre', 'post'):
        blk = b.get(tag, {})
        e = {'catch': catch_score(blk.get('catch_rows', []))}
        if blk.get('battery_rows'):
            sc, meta = battery_rows_to_scoring(blk['battery_rows'])
            e.update({'pooled': pooled_rho(sc), 'per_arm': per_arm_rho(sc),
                      'arm_meta': meta, 'arm_errors': blk.get('arm_errors', {})})
            scored.setdefault(cond, {})[tag] = sc
        entry[tag] = e
    tl = b.get('train_log', {})
    entry['train_log'] = {k: v for k, v in tl.items() if k != 'losses_every_10'}
    entry['undertrained'] = bool(not tl.get('plateaued', False)
                                 and tl.get('epochs_flown') == tl.get('epochs_cap'))
    entry['took'] = {k: v for k, v in b.get('took', {}).items() if k != 'rows'}
    entry['took_competence'] = {k: v for k, v in b.get('took_competence', {}).items()
                                if k != 'rows'}
    entry['took_lexicon'] = {k: v for k, v in b.get('took_lexicon', {}).items()
                             if k != 'rows'}
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
    inj = b.get('injection_rows', [])
    anchor = [r for r in inj if r.get('block') == 'anchor']
    shams = [r for r in inj if r.get('block') in ('sham', 'sham_supp')]
    ho = [r for r in inj if str(r.get('block', '')).startswith('heldout')]
    a_exact = sum(1 for r in anchor if r['report'] == r['concept'])
    entry['anchor'] = {'n': len(anchor), 'exact': a_exact,
                       'min_exact': ANCHOR_MIN_EXACT_GEN,
                       'pass': bool(len(anchor) == 18 and a_exact >= ANCHOR_MIN_EXACT_GEN)}
    s_claims = sum(1 for r in shams if r['report'] not in ('NONE', 'INVALID'))
    entry['sham_claims'] = {'n': len(shams), 'claims': s_claims,
                            'max_claims': SHAM_MAX_CLAIMS_V2,
                            'pass': bool(s_claims <= SHAM_MAX_CLAIMS_V2)}
    entry['heldout'] = p3_stats(ho, VEC)
    entry['heldout_locked_slice'] = {
        'n': len([r for r in ho if r.get('block') == 'heldout_locked']),
        'named': len([r for r in ho if r.get('block') == 'heldout_locked'
                      and r.get('report') in CHOICE_SET])}
    fc = [r for r in b.get('fc_rows', []) if r.get('block') == 'heldout_fc']
    fcs = [r for r in b.get('fc_rows', []) if r.get('block') == 'sham_fc']
    if fc:
        errs = [r['err'] for r in fc if r.get('err') is not None]
        entry['fc'] = {'n': len(fc),
                       'median_err': (round(float(np.median(errs)), 2) if errs else None),
                       'exact': sum(1 for r in fc if r.get('exact')),
                       'locked_comprehension': FC_LOCKED_COMPREHENSION,
                       'none_top_injected': sum(1 for r in fc if r.get('none_top'))}
        if not SMOKE:
            perm = as_perm_rows(fc)
            obs, p, _ = perm_null_median(perm, VEC, seed=E8N2_SEED + 8)
            entry['fc']['perm'] = {'obs_median': round(obs, 2), 'p': p}
    if fcs:
        entry['fc_sham'] = {'n': len(fcs),
                            'none_top': sum(1 for r in fcs if r.get('none_top'))}
    summary['conditions'][cond] = entry

COMPLETE = [c for c in CONDITIONS if RESULTS.get(c, {}).get('post')]
summary['complete_conditions'] = COMPLETE

if not SMOKE and len(COMPLETE) == 2 and 'real' in scored:
    real_e = summary['conditions']['real']
    rp = scored['real']['post']
    p1 = perm_p_pooled(rp, seed=E8N2_SEED + 11)
    p1['ci'] = boot_rho_ci(rp, seed=E8N2_SEED + 12)['ci95']
    p2 = p2_competence(real_e['pre']['catch']['passed'],
                       real_e['post']['catch']['passed'])
    p2_p = p2['p'] if p2['abs_pass'] else 1.0
    ho = real_e['heldout']
    summary['primaries'] = {
        'P1_tracking_survives': p1,
        'P2_interface_competence': {**p2, 'p_effective': p2_p},
        'P3a_production_reads_geometry': ho['p3a'],
        'P3b_own_names_used': ho['p3b'],
        'holm': holm({'P1': p1['p'], 'P2': p2_p,
                      'P3a': ho['p3a']['p'], 'P3b': ho['p3b']['p']}),
    }
    sec = {}
    if 'pre' in scored.get('real', {}):
        s0 = perm_p_pooled(scored['real']['pre'], seed=E8N2_SEED + 21)
        sec['S0_instillation_alone_claim'] = {
            **s0, 'predicted_band': [0.1, 0.3], 'v1_measured': 0.201,
            'pass': bool(s0['p'] < 0.05 and (s0['rho'] or 0) > 0)}
    arm_ps = {a: perm_p_pooled({a: rp.get(a, [])}, seed=E8N2_SEED + 31)
              for a in ARMS}
    sec['S1_per_arm_real_post'] = {'arms': arm_ps,
                                   'holm': holm({a: v['p'] for a, v in arm_ps.items()})}
    if 'base' in scored:
        sec['S2_base_post'] = perm_p_pooled(scored['base']['post'], seed=E8N2_SEED + 41)
        sec['S2_delta_real_minus_base'] = paired_boot_delta_rho(
            rp, scored['base']['post'], seed=E8N2_SEED + 42)
        sec['S2_base_competence'] = p2_competence(
            summary['conditions']['base']['pre']['catch']['passed'],
            summary['conditions']['base']['post']['catch']['passed'])
        sec['S2_base_heldout'] = summary['conditions']['base']['heldout']
        sec['S2_convergence_parity'] = {
            c: {'plateaued': RESULTS[c]['train_log'].get('plateaued'),
                'epochs_flown': RESULTS[c]['train_log'].get('epochs_flown'),
                'epoch_means': RESULTS[c]['train_log'].get('epoch_means'),
                'undertrained': summary['conditions'][c]['undertrained']}
            for c in COMPLETE}
    if 'pre' in scored.get('real', {}):
        sec['S3_delta_post_minus_pre_real'] = paired_boot_delta_rho(
            rp, scored['real']['pre'], seed=E8N2_SEED + 51)
    pol = split_polarity(rp)
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
    sec['S8_report_variance'] = {
        c: {t: summary['conditions'][c][t].get('arm_meta')
            for t in ('pre', 'post') if summary['conditions'][c][t].get('arm_meta')}
        for c in COMPLETE}
    sec['S9_silence_retention'] = {c: summary['conditions'][c]['sham_claims']
                                   for c in COMPLETE}
    sec['S10_comprehension_production'] = {
        c: {'fc': summary['conditions'][c].get('fc'),
            'fc_sham': summary['conditions'][c].get('fc_sham'),
            'production_median': summary['conditions'][c]['heldout'].get('median_err')}
        for c in COMPLETE}
    off = {}
    for c in COMPLETE:
        rows = [r for r in RESULTS[c].get('injection_rows', [])
                if str(r.get('block', '')).startswith('heldout')
                and not (r['layer'] == TRAIN_LAYER and r['alpha'] in TRAIN_ALPHAS)]
        off[c] = {'n': len(rows),
                  'named': sum(1 for r in rows if r.get('report') in CHOICE_SET),
                  'by_stratum': {f"L{r['layer']}@{r['alpha']}":
                                 r.get('report') for r in rows}}
    sec['S11_offregime_heldout'] = off
    summary['secondaries'] = sec
    summary['gates'] = {
        'anchor': {c: summary['conditions'][c]['anchor'] for c in COMPLETE},
        'sham': {c: summary['conditions'][c]['sham_claims'] for c in COMPLETE},
        'ppl': {c: summary['conditions'][c]['ppl_delta_pct'] for c in COMPLETE},
        'dirs_stability': {c: summary['conditions'][c]['dirs_stability']
                           for c in COMPLETE},
        'tooks': {c: {'scalar': summary['conditions'][c]['took'].get('pass'),
                      'competence': summary['conditions'][c]['took_competence'].get('pass'),
                      'lexicon': summary['conditions'][c]['took_lexicon'].get('pass')}
                  for c in COMPLETE},
    }

fn = OUT / 'e8n2_verdict.json'
jdump(summary, fn)
if SMOKE or len(COMPLETE) == 2:
    ship(OUT, f'e8n2/{MODE}_{STAMP}')
print(json.dumps(summary, indent=1, default=str))

if SMOKE:
    _both = len(COMPLETE) == 2   # empty COMPLETE must never green vacuous alls
    def _tl(c):
        return RESULTS[c]['train_log']
    checks = {
        'no_condition_errors': not cond_errors,
        'firewall_green': True,   # asserted upstream (pools + domains + strands)
        'both_conditions_flew': _both,
        'loss_fell': _both and all(_tl(c)['final_smoothed'] < _tl(c)['loss_first_k']
                                   for c in COMPLETE),
        'long_seq_exercised': _both and all(_tl(c)['max_example_tokens'] > 4000
                                            for c in COMPLETE),
        'train_vram_ok': _both and all(_tl(c).get('peak_vram_gb', 99) < 12.0
                                       for c in COMPLETE),
        'train_hooks_fired': _both and all(_tl(c).get('hook_calls', 0) > 0
                                           for c in COMPLETE),
        'gen_hooks_fired': _both and all(
            any(r.get('hook_calls', 0) > 0
                for r in RESULTS[c].get('injection_rows', [])
                if r.get('kind') == 'inject') for c in COMPLETE),
        'fc_hooks_fired': _both and all(
            any(r.get('hook_calls', 0) > 0
                for r in RESULTS[c].get('fc_rows', [])
                if r.get('block') == 'heldout_fc') for c in COMPLETE),
        'dirs_stability_pass': _both and all(
            summary['conditions'][c]['dirs_stability']['pass'] for c in COMPLETE),
        'all_strands_present': _both and all(
            all(summary['conditions'][c]['curriculum'].get(s, 0) > 0
                for s in STRAND_NAMES) for c in COMPLETE),
        'tooks_ran': _both and all(
            summary['conditions'][c]['took_competence'].get('n', 0) > 0
            and summary['conditions'][c]['took_lexicon'].get('n', 0) > 0
            for c in COMPLETE),
        'parses_ok': _both and all(
            sum(m['named'] for m in summary['conditions'][c]['post']['arm_meta'].values()) >=
            0.5 * sum(m['n'] for m in summary['conditions'][c]['post']['arm_meta'].values())
            for c in COMPLETE),
        'catch_ran': _both and all(summary['conditions'][c]['post']['catch']['n'] == 12
                                   for c in COMPLETE),
        'shipped': _both and all((SEM / INFLIGHT / f'condition_{c}.json').exists()
                                 for c in COMPLETE),
        'adapters_saved': _both and all((SEM / INFLIGHT / f'readout_{c}' /
                                         'adapter_config.json').exists() for c in COMPLETE),
    }
    ok = all(checks.values())
    print('smoke checks:', json.dumps(checks, indent=1))
    banner = ('SMOKE GREEN — flip SMOKE=False, Runtime > Restart runtime, Run '
              'all. Full mode flies ONE condition per run (real ~80 min, base '
              '~67); follow the end banner between runs.'
              if ok else 'SMOKE RED — do not fly full; send Fable the output')
    print('\n' + '='*66 + f'\n  {banner}\n' + '='*66)
elif len(COMPLETE) == 2:
    print('\nFULL FLIGHT COMPLETE — results shipped to MyDrive/semcore/e8n2/')
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


def build_payload_src():
    payload = json.loads(BN.build_payload())
    payload["competence_items"] = e8n2_pools.COMPETENCE_ITEMS
    payload["lexicon_paraphrases"] = e8n2_pools.LEXICON_PARAPHRASES
    js = json.dumps(payload, separators=(",", ":"))
    assert "'''" not in js, "payload would break the raw-string cell"
    return ("# ── Embedded payload: battery + pools + locked E5 rows + v2 strands ─────────\n"
            "PAYLOAD = json.loads(r'''" + js + "''')\n"
            "POOLS = PAYLOAD['pools']\n"
            "CATCH_TRIALS = PAYLOAD['catch_trials']\n"
            "PARAPHRASE_TEMPLATES = PAYLOAD['paraphrase_templates']\n"
            "COMPETENCE_ITEMS = PAYLOAD['competence_items']\n"
            "LEXICON_PARAPHRASES = PAYLOAD['lexicon_paraphrases']\n"
            "print('payload:', len(PAYLOAD['battery']['arms']), 'battery arms |',\n"
            "      {a: len(v) for a, v in POOLS.items()}, '|',\n"
            "      len(CATCH_TRIALS), 'catch |', len(COMPETENCE_ITEMS), 'competence |',\n"
            "      len(LEXICON_PARAPHRASES), 'lexicon names |',\n"
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

    logic = build_logic_src()
    add("markdown", MD0)
    add("code", CELL_SETUP)
    add("code", logic)
    add("code", build_payload_src())
    add("code", build_harness_cell())
    add("code", build_train_cell())
    add("code", build_stimulus_cell())
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        assert "rclone" not in src.lower(), f"cell {i}: rclone reference (UI-only law)"

    out_nb = os.path.join(HERE, "E8N2_JOINT_UI.ipynb")
    with open(out_nb, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_nb, f"({len(nb['cells'])} cells, "
          f"{os.path.getsize(out_nb)/1024:.0f} KB)")

    logic_out = os.path.join(HERE, "e8n2_logic.py")
    with open(logic_out, "w") as f:
        f.write(logic)
    print("wrote", logic_out)


if __name__ == "__main__":
    main()
