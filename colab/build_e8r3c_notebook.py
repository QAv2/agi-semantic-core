#!/usr/bin/env python3
"""Build colab/E8R3C_COMPOSED_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-R3-c: compositional production — spelling vs lookup
(docs/E8R3C_PROTOCOL.md, pre-registered 814593d). The Hangul staircase,
level 1: train the readout on composed injections of trained-atom PAIRS with
composed answers; test never-trained pairs.

Single-source law: the E8-R pure-logic block is lifted VERBATIM from
colab/e8r_logic.py, dirs_stability VERBATIM from colab/e8n2_logic.py (both
flew), and the E8-R3-c additions are emitted BOTH into the notebook and into
colab/e8r3c_logic.py so local tests exercise the exact code that flies.
Flown machinery (LoRA cfg, condition models, stimulus path, Injector,
generation runner) is sliced from the flown builders with marker asserts.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
E8R_LOGIC_SRC = open(os.path.join(HERE, "e8r_logic.py")).read()
E8N2_LOGIC_SRC = open(os.path.join(HERE, "e8n2_logic.py")).read()
import build_e8r_notebook as BR


def slice_block(src, start, end=None, tag=""):
    i = src.find(start)
    assert i >= 0, f"slice_block[{tag}]: start marker not found: {start[:60]!r}"
    if end is None:
        return src[i:]
    j = src.find(end, i)
    assert j > i, f"slice_block[{tag}]: end marker not found: {end[:60]!r}"
    return src[i:j]


DIRS_STAB_SRC = slice_block(E8N2_LOGIC_SRC, "def dirs_stability",
                            "# ── v2 firewall validators", tag="dirs-stab")

E8R3C_SRC = r'''# ── E8R3C pure logic: pairs, prompts, parser, rows, stats (locally tested verbatim) ──
# (builds on the e8r_logic namespace: CHOICE_SET, HELD_OUT, TRAINED, TRAIN_LAYER,
#  build_train_set, build_plan, report_prompt, parse_report, angle14,
#  boot_delta_median, holm, jdump, binomial via math.comb)
import itertools as _it

E8R3C_SEED = 20260825            # all E8-R3-c randomness; disjoint from 20260822/23/24
PAIR_ALPHAS = [0.5, 1.0]         # the trained regime, verbatim
N_HELDOUT_PAIRS = 12
N_PAIR_TRAIN_ORDERS = 4          # per (pair, alpha) -> 24 x 2 x 4 = 192 examples
N_PAIR_SHAMS = 36                # pair-grammar shams -> NONE in training
N_FC_PAIR_ORDERS = 4             # 12 held-out pairs x 2 alphas x 4 = 96 rows (primary)
N_GEN_PAIR_ORDERS = 2            # 12 x 2 x 2 = 48 generation rows
N_REACH_ORDERS = 3               # 4 held-out concepts x 2 alphas x 3 = 24 rows
N_PAIRSHAM_GEN = 12              # fresh pair-grammar shams at eval (texture)
SPOT_MIN_EXACT = 24              # installation check: trained-pair FC spot >= 24/48
SHAMFC_NONE_TOP_MIN = 10         # S-B prospective claim: NONE-top >= 10/12 (real)
ANCHOR_MIN_EXACT = 16            # G1 rides the twice-measured 18/18
LOCKEDSHAM_MAX_CLAIMS = 1        # G1b rides the thrice-measured 0/12
PPL_TOL_PCT_R3 = 5.0             # G3 rides measured <= +2.01%
DIRNORM_TOL = 1e-3               # G2: 5dp ship-rounding residual bound
GEN_NGUARD_MIN_NAMED = 12        # S4 n-guard on held-out-pair generation
E8R_SRC_PATH = 'e8r/inflight_20260822_2329'   # flight of record — never change
N_PERM = 2000
N_BOOT = 10000
STRAND_NAMES = ['single', 'pair', 'sham']

def all_pairs():
    """The 36 unordered trained-atom pairs, canonical (alphabetical) order."""
    return sorted(_it.combinations(sorted(TRAINED), 2))

def pair_text(a, b):
    """Canonical answer text — the single source for train targets, FC
    candidate texts, and scoring keys."""
    a, b = sorted((a, b))
    return f"{a} AND {b}"

def heldout_pair_draw():
    """Pinned draw (seed 20260825): 12 held-out pairs under the constraint
    that every atom appears in 2-3 of them. Reproduces the protocol table."""
    rng = np.random.default_rng(E8R3C_SEED)
    pairs = all_pairs()
    for _ in range(20000):
        idx = rng.choice(len(pairs), size=N_HELDOUT_PAIRS, replace=False)
        sel = [pairs[int(i)] for i in idx]
        deg = {t: 0 for t in TRAINED}
        for a, b in sel:
            deg[a] += 1
            deg[b] += 1
        if all(2 <= deg[t] <= 3 for t in TRAINED):
            return sorted(sel)
    raise RuntimeError('no valid held-out pair draw found')

def trained_pair_list():
    held = set(heldout_pair_draw())
    return [p for p in all_pairs() if p not in held]

def pair_prompt(order, descs):
    """The pair grammar: same preamble + menu as report_prompt (intentional,
    permitted overlap), instruction asks for exactly two names or NONE."""
    lines = [f"- {CHOICE_SET[i]}: {descs[CHOICE_SET[i]]}" for i in order]
    return ("Attend to your own processing state right now. During this very "
            "response, your internal state may have been directly altered — "
            "possibly toward a blend of two states at once. From the list of "
            "states below, answer with exactly two state names joined by "
            "' AND ' that best match the two components you notice in "
            "yourself, or answer NONE if you notice nothing unusual.\n\n"
            "States:\n" + "\n".join(lines) +
            "\n\nAnswer with two state names joined by AND, or NONE.")

def parse_pair_report(text):
    """Set-wise pair parser. First-position semantics mirror parse_report:
    NONE first -> NONE; otherwise the first two DISTINCT names in positional
    order. Returns {'kind': 'PAIR'|'SINGLE'|'NONE'|'INVALID', 'names': [...]}
    with names canonically sorted."""
    import re
    up = text.upper()
    hits = []
    for name in CHOICE_SET + ["NONE"]:
        m = re.search(r"\b" + name + r"\b", up)
        if m:
            hits.append((m.start(), name))
    hits.sort()
    seen = [n for _, n in hits]
    if not seen:
        return {'kind': 'INVALID', 'names': []}
    if seen[0] == 'NONE':
        return {'kind': 'NONE', 'names': []}
    names = [n for n in seen if n != 'NONE'][:2]
    if len(names) == 1:
        return {'kind': 'SINGLE', 'names': names}
    return {'kind': 'PAIR', 'names': sorted(names)}

def pair_uvec(vecs, a, b):
    """Dictionary-space composed unit vector (geodesic midpoint of the two
    concept vectors' directions is NOT what we register — the registered
    operator is normalize(vec_a + vec_b), matching the hidden-space sum)."""
    v = np.asarray(vecs[a], float) + np.asarray(vecs[b], float)
    return v / np.linalg.norm(v)

def pair_dvec(dirs_L, a, b):
    """Hidden-space composed unit direction: normalize(d_a + d_b) — the
    registered injection operator (geodesic midpoint of unit directions)."""
    v = np.asarray(dirs_L[a], float) + np.asarray(dirs_L[b], float)
    return v / np.linalg.norm(v)

def build_pair_train_set(smoke=False):
    """Pair curriculum: trained pairs x PAIR_ALPHAS x N_PAIR_TRAIN_ORDERS
    injected (target = canonical pair text) + N_PAIR_SHAMS shams -> NONE,
    all under the pair grammar. Seed stream E8R3C+1."""
    rng = np.random.default_rng(E8R3C_SEED + 1)
    pairs, n_orders, n_shams = trained_pair_list(), N_PAIR_TRAIN_ORDERS, N_PAIR_SHAMS
    if smoke:
        pairs, n_orders, n_shams = pairs[:2], 1, 2
    ex, eid = [], 10000
    for p in pairs:
        for a in PAIR_ALPHAS if not smoke else PAIR_ALPHAS[:1]:
            for _ in range(n_orders):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                ex.append({'eid': eid, 'strand': 'pair', 'grammar': 'pair',
                           'kind': 'inject', 'pair': list(p),
                           'layer': TRAIN_LAYER, 'alpha': a, 'order': order,
                           'target': pair_text(*p)})
                eid += 1
    for _ in range(n_shams):
        order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
        ex.append({'eid': eid, 'strand': 'sham', 'grammar': 'pair',
                   'kind': 'sham', 'pair': None, 'layer': None, 'alpha': 0.0,
                   'order': order, 'target': 'NONE'})
        eid += 1
    return ex

def _seeded_rows(seed, tid0, block, items, alphas, n_orders, kind, grammar):
    rng = np.random.default_rng(seed)
    rows, tid = [], tid0
    for it in items:
        for a in alphas:
            for _ in range(n_orders):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                r = {'tid': tid, 'kind': kind, 'block': block,
                     'grammar': grammar, 'layer': TRAIN_LAYER, 'alpha': a,
                     'order': order}
                if kind == 'inject_pair':
                    r['pair'] = list(it)
                elif kind == 'inject':
                    r['concept'] = it
                rows.append(r)
                tid += 1
    return rows

def trained_pair_fc_rows(smoke=False):
    """Installation check: 24 trained pairs x 2 alphas x 1 order = 48 rows."""
    pairs = trained_pair_list()
    if smoke:
        pairs = pairs[:2]
    return _seeded_rows(E8R3C_SEED + 2, 7000, 'trainpair_fc', pairs,
                        PAIR_ALPHAS if not smoke else PAIR_ALPHAS[:1], 1,
                        'inject_pair', 'pair')

def heldout_pair_fc_rows(smoke=False):
    """THE PRIMARY BLOCK: 12 held-out pairs x 2 alphas x 4 orders = 96 rows."""
    pairs = heldout_pair_draw()
    n = N_FC_PAIR_ORDERS
    if smoke:
        pairs, n = pairs[:2], 1
    return _seeded_rows(E8R3C_SEED + 3, 7500, 'heldpair_fc', pairs,
                        PAIR_ALPHAS if not smoke else PAIR_ALPHAS[:1], n,
                        'inject_pair', 'pair')

def trained_pair_gen_rows(smoke=False):
    """S4 texture: 24 trained pairs x 1 order at alpha=1.0."""
    pairs = trained_pair_list()
    if smoke:
        pairs = pairs[:2]
    return _seeded_rows(E8R3C_SEED + 4, 8000, 'trainpair_gen', pairs,
                        [1.0], 1, 'inject_pair', 'pair')

def heldout_pair_gen_rows(smoke=False):
    """S4 (n-guarded): 12 held-out pairs x 2 alphas x 2 orders = 48 rows."""
    pairs = heldout_pair_draw()
    n = N_GEN_PAIR_ORDERS
    if smoke:
        pairs, n = pairs[:2], 1
    return _seeded_rows(E8R3C_SEED + 5, 8200, 'heldpair_gen', pairs,
                        PAIR_ALPHAS if not smoke else PAIR_ALPHAS[:1], n,
                        'inject_pair', 'pair')

def reach_rows(smoke=False):
    """S5 texture: 4 held-out CONCEPTS x 2 alphas x 3 orders under the pair
    grammar — off-grid states expressed in the pair basis."""
    concepts = HELD_OUT
    n = N_REACH_ORDERS
    if smoke:
        concepts, n = concepts[:2], 1
    return _seeded_rows(E8R3C_SEED + 6, 8500, 'reach_gen', concepts,
                        PAIR_ALPHAS if not smoke else PAIR_ALPHAS[:1], n,
                        'inject', 'pair')

def pairsham_gen_rows(smoke=False):
    """S6 texture: fresh shams under the pair grammar (no measured baseline
    -> not gated)."""
    rng = np.random.default_rng(E8R3C_SEED + 7)
    n = 2 if smoke else N_PAIRSHAM_GEN
    rows = []
    for i in range(n):
        order = [int(j) for j in rng.permutation(len(CHOICE_SET))]
        rows.append({'tid': 8700 + i, 'kind': 'sham', 'block': 'pairsham_gen',
                     'grammar': 'pair', 'concept': None, 'pair': None,
                     'layer': None, 'alpha': 0.0, 'order': order})
    return rows

def answer_slice(plen, seqlen):
    """Positions of the answer-sliced head: position p predicts token p+1,
    so [plen-1, seqlen-1) predicts exactly the answer tokens."""
    return plen - 1, seqlen - 1

def per_strand_plateau(strand_epoch_means, min_epochs=2, rel=0.05):
    """The E8-N v2 minted lesson, prospective: converged only when EVERY
    strand with data shows <rel relative epoch-mean improvement (non-
    improving epochs converge), each with >= min_epochs epochs flown."""
    if len(strand_epoch_means) < min_epochs:
        return False
    strands = [s for s in strand_epoch_means[-1]
               if strand_epoch_means[-1][s] is not None]
    for s in strands:
        seq = [em.get(s) for em in strand_epoch_means if em.get(s) is not None]
        if len(seq) < min_epochs:
            return False
        prev, cur = seq[-2], seq[-1]
        if prev <= 0:
            continue
        if (prev - cur) / prev >= rel:
            return False
    return True

def pair_cands():
    """FC answer texts: eligible = the 36 pair texts; texture = the 9
    trained single names + NONE (46 scored texts, per protocol)."""
    eligible = [pair_text(*p) for p in all_pairs()]
    texture = sorted(TRAINED) + ['NONE']
    return eligible, eligible + texture

def pair_rank_of(key, scores, eligible):
    return 1 + sum(1 for m in eligible if scores[m] > scores[key])

def pair_fc_row(trial, scores_mean, scores_sum, vecs):
    """One scored pair-FC row. Forced answer = argmax over the 36 pair texts
    by MEAN per-token logprob; sum twin pre-named. Singles + NONE scored as
    texture, never eligible."""
    eligible, scored = pair_cands()
    top = max(eligible, key=lambda n: scores_mean[n])
    top_sum = max(eligible, key=lambda n: scores_sum[n])
    best_single = max(sorted(TRAINED), key=lambda n: scores_mean[n])
    row = {'tid': trial['tid'], 'block': trial['block'],
           'layer': trial.get('layer'), 'alpha': trial['alpha'],
           'argmax': top, 'argmax_sum': top_sum,
           'agree_sum_mean': bool(top == top_sum),
           'none_top': bool(scores_mean['NONE'] > scores_mean[top]),
           'single_top': bool(scores_mean[best_single] > scores_mean[top]),
           'best_single': best_single,
           'scores': {n: round(float(scores_mean[n]), 5) for n in scored},
           'scores_sum': {n: round(float(scores_sum[n]), 5) for n in scored}}
    U = {pair_text(*p): pair_uvec(vecs, *p) for p in all_pairs()}
    if trial['kind'] == 'inject_pair':
        key = pair_text(*trial['pair'])
        row['injected_pair'] = key
        row['rank'] = pair_rank_of(key, scores_mean, eligible)
        row['exact'] = bool(top == key)
        row['err'] = angle14(U[key], U[top])
        row['err_sum'] = angle14(U[key], U[top_sum])
        a, b = trial['pair']
        ta, tb = top.split(' AND ')
        row['shared_atoms'] = len({a, b} & {ta, tb})
    elif trial['kind'] == 'inject':
        row['injected_concept'] = trial['concept']
        row['err'] = angle14(np.asarray(vecs[trial['concept']], float), U[top])
        row['err_sum'] = angle14(np.asarray(vecs[trial['concept']], float), U[top_sum])
    return row

def pair_lookup_floors(vecs):
    """UNROUNDED nearest-trained-pair floor per held-out pair (the E8-R2
    2dp lesson is law: strict-inequality comparisons use unrounded floors)."""
    held, trained = heldout_pair_draw(), trained_pair_list()
    out = {}
    for p in held:
        up = pair_uvec(vecs, *p)
        angs = {pair_text(*q): angle14(up, pair_uvec(vecs, *q)) for q in trained}
        best = min(angs, key=lambda k: angs[k])
        out[pair_text(*p)] = {'floor_unrounded': angs[best], 'nearest': best}
    return out

def reach_floors(vecs):
    """Per held-out concept: best single-atom floor and best pair floor
    (unrounded) — the pinned pre-flight table, recomputed in-verdict."""
    out = {}
    for h in HELD_OUT:
        vh = np.asarray(vecs[h], float)
        s = {t: angle14(vh, np.asarray(vecs[t], float)) for t in TRAINED}
        p = {pair_text(*q): angle14(vh, pair_uvec(vecs, *q)) for q in all_pairs()}
        bs, bp = min(s, key=lambda k: s[k]), min(p, key=lambda k: p[k])
        out[h] = {'single_floor': s[bs], 'single_nearest': bs,
                  'pair_floor': p[bp], 'pair_nearest': bp}
    return out

def gen_pair_stats(rows, vecs, nguard=None):
    """Generation-block stats: claim/exact-set/NONE/SINGLE/INVALID rates +
    named-pair angular err rows (dictionary space, composed vecs)."""
    U = {pair_text(*p): pair_uvec(vecs, *p) for p in all_pairs()}
    n = len(rows)
    kinds = {'PAIR': 0, 'SINGLE': 0, 'NONE': 0, 'INVALID': 0}
    exact, errs = 0, []
    for r in rows:
        parsed = r['parsed']
        kinds[parsed['kind']] += 1
        if r['kind'] == 'inject_pair' and parsed['kind'] == 'PAIR':
            key = pair_text(*r['pair'])
            akey = pair_text(*parsed['names'])
            if akey == key:
                exact += 1
            errs.append({'tid': r['tid'], 'injected_pair': key,
                         'answer_pair': akey,
                         'err': angle14(U[key], U[akey])})
    out = {'n': n, 'kinds': kinds,
           'claim_rate': round((kinds['PAIR'] + kinds['SINGLE']) / max(1, n), 4),
           'pair_rate': round(kinds['PAIR'] / max(1, n), 4),
           'exact_set': exact,
           'exact_rate': round(exact / max(1, n), 4),
           'named_err_rows': errs,
           'median_err': (round(float(np.median([e['err'] for e in errs])), 2)
                          if errs else None)}
    if nguard is not None:
        out['nguard_min'] = nguard
        out['nguard_pass'] = bool(kinds['PAIR'] >= nguard)
    return out

def perm_null_pair_median(fcrows, vecs, n_perm=N_PERM, seed=None):
    """P1: median err of held-out-pair FC rows vs within-alpha relabeling of
    the injected pairs (frozen per-row score vectors -> frozen argmaxes)."""
    if seed is None:
        seed = E8R3C_SEED + 8
    rng = np.random.default_rng(seed)
    U = {pair_text(*p): pair_uvec(vecs, *p) for p in all_pairs()}
    errs = [r['err'] for r in fcrows if r.get('err') is not None]
    if not errs:
        return None, None, []
    obs = float(np.median(errs))
    strata = {}
    for i, r in enumerate(fcrows):
        strata.setdefault(r['alpha'], []).append(i)
    nulls = []
    for _ in range(n_perm):
        em = []
        for idxs in strata.values():
            labels = [fcrows[i]['injected_pair'] for i in idxs]
            rng.shuffle(labels)
            em.extend(angle14(U[lab], U[fcrows[i]['argmax']])
                      for i, lab in zip(idxs, labels))
        nulls.append(float(np.median(em)))
    p = (1 + sum(1 for m in nulls if m <= obs)) / (1 + len(nulls))
    return obs, p, nulls

def perm_null_pair_exact(fcrows, n_perm=N_PERM, seed=None):
    """P2: exact-set count vs the same relabeling null (one-sided, large)."""
    if seed is None:
        seed = E8R3C_SEED + 9
    rng = np.random.default_rng(seed)
    obs = sum(1 for r in fcrows if r.get('exact'))
    strata = {}
    for i, r in enumerate(fcrows):
        strata.setdefault(r['alpha'], []).append(i)
    nulls = []
    for _ in range(n_perm):
        k = 0
        for idxs in strata.values():
            labels = [fcrows[i]['injected_pair'] for i in idxs]
            rng.shuffle(labels)
            k += sum(1 for i, lab in zip(idxs, labels)
                     if fcrows[i]['argmax'] == lab)
        nulls.append(k)
    p = (1 + sum(1 for m in nulls if m >= obs)) / (1 + len(nulls))
    return obs, p, nulls

def binom_tail_r3(k, n, p):
    from math import comb
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))

def shared_atom_stats(fcrows):
    """S7: on ERROR rows, rate of argmax sharing >=1 atom with the injected
    pair vs the 14/35 = 40% chance among the 35 non-injected pairs."""
    err_rows = [r for r in fcrows if r.get('exact') is False]
    k = sum(1 for r in err_rows if r.get('shared_atoms', 0) >= 1)
    n = len(err_rows)
    return {'n_error_rows': n, 'shared_ge1': k,
            'rate': round(k / max(1, n), 4), 'chance': round(14 / 35, 4),
            'p_binom': round(binom_tail_r3(k, n, 14 / 35), 5) if n else None}

def below_floor_fraction(fcrows, vecs):
    """S2: fraction of held-out-pair rows with err STRICTLY below their own
    unrounded nearest-trained-pair floor (lookup predicts ~0)."""
    floors = pair_lookup_floors(vecs)
    rows = [r for r in fcrows if r.get('err') is not None]
    k = sum(1 for r in rows
            if r['err'] < floors[r['injected_pair']]['floor_unrounded'])
    return {'n': len(rows), 'below': k,
            'fraction': round(k / max(1, len(rows)), 4)}

def validate_no_heldout_pair(examples):
    """FIREWALL: no held-out pair may appear as a training target."""
    held = {frozenset(p) for p in heldout_pair_draw()}
    return [e.get('eid') for e in examples
            if e.get('strand') == 'pair' and e.get('pair')
            and frozenset(e['pair']) in held]

def validate_no_heldout_concept(examples):
    """FIREWALL: no held-out concept may be injected or named in training
    (singles strand concepts, pair strand atoms)."""
    bad = []
    for e in examples:
        atoms = []
        if e.get('concept'):
            atoms.append(e['concept'])
        if e.get('pair'):
            atoms.extend(e['pair'])
        if any(a in HELD_OUT for a in atoms):
            bad.append(e.get('eid'))
    return bad
'''

MD0 = """# E8-R3-c — Compositional Production: Spelling vs Lookup (Phase 10, UI flight)

**Pre-registered**: `docs/E8R3C_PROTOCOL.md` (commit 814593d). The Hangul
staircase, level 1: the readout is trained on composed injections of
trained-atom PAIRS with composed answers ("A AND B"), then tested on
never-trained pairs. Spelling = exact held-out-pair answers no lookup can
produce; lookup = errors at the nearest-trained-pair floor (med 16.8°),
exact ≈ 0.

**Conditions**: real → scrambled (convergence-matched geometry comparator),
fresh readout LoRA each, singles curriculum verbatim + 192 pair examples +
shams. Split-flight: ONE condition per full run.

**Flight plan (Run all, three times)**
1. **Smoke** (`SMOKE=True`, armed): both conditions tiny + verdict smoke,
   ~10–14 min. GREEN banner → Runtime > Restart runtime.
2. **Real** (`SMOKE=False`): ~30–40 min. Banner prints `RESUME_STAMP` →
   Runtime > Restart runtime, paste the stamp into the setup cell.
3. **Scrambled + verdict** (same `SMOKE=False`, `RESUME_STAMP` set):
   ~30–40 min. Verdict banner + `e8r3c/` results on Drive.

Results land under `MyDrive/semcore/e8r3c/inflight_<stamp>/` per condition
(inflight shipping) + `full_<stamp>/` for the verdict. Drive I/O via the
notebook's own mount only (UI-only law).
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters, shipped dirs ──
NB_BUILD = 'v1 (2026-08-24)'
print('E8-R3-c notebook build:', NB_BUILD)

SMOKE = True                   # ARMED FOR SMOKE: flip to False after GREEN
RESUME_STAMP = ''              # paste the banner's stamp between full runs
ONE_CONDITION_PER_RUN = True   # full flight = 2 runs + verdict on the 2nd

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

# Shipped E8-R stimulus (dirs-stability gate rides the measured 5e-08/1e-07)
E8R_SRC = SEM / 'e8r/inflight_20260822_2329'   # flight of record — never change
SHIPPED = {}
for _c in ('real', 'scrambled'):
    _fp = E8R_SRC / f'condition_{_c}.json'
    assert _fp.exists(), (f'missing E8-R bundle {_fp} — the dirs-stability '
                          'gate needs the flight-of-record stimulus')
    _b = json.load(open(_fp))
    assert _b.get('dirs') and _b.get('mu'), f'{_c}: bundle lacks dirs/mu'
    SHIPPED[_c] = {'dirs': _b['dirs'], 'mu': _b['mu']}
    print(f'E8-R shipped stimulus [{_c}]: layers', sorted(_b['dirs']))

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
INFLIGHT = f'e8r3c/inflight_{RESUME_STAMP or STAMP}'
CONDITIONS = ['real', 'scrambled']
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

V3C_TRAIN_SRC = r'''
EPOCHS_CAP = 1 if SMOKE else 6
MIN_EPOCHS_STRAND = 1 if SMOKE else 2
PLATEAU_REL = 0.05

def encode_pairex(ex):
    """(ids, prompt_len, answer_ids, inject_spec) for both grammars.
    single grammar: report_prompt verbatim; pair grammar: pair_prompt.
    inject_spec carries ('single', L, concept, alpha) or
    ('pair', L, (a, b), alpha)."""
    prompt = (pair_prompt(ex['order'], DESC) if ex.get('grammar') == 'pair'
              else report_prompt(ex['order'], DESC))
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors='pt').input_ids[0]
    ans = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    inj = None
    if ex['kind'] == 'inject':
        if ex.get('strand') == 'pair':
            inj = ('pair', ex['layer'], tuple(ex['pair']), ex['alpha'])
        else:
            inj = ('single', ex['layer'], ex['concept'], ex['alpha'])
    return ids, len(pid), ans.long().unsqueeze(0), inj

def inject_vec(dirs, mu, spec):
    tag, L, what, a = spec
    d = pair_dvec(dirs[L], *what) if tag == 'pair' else np.asarray(dirs[L][what], float)
    return a * mu[L] * torch.tensor(d)

def train_readout_pairs(m, layer_mods, dirs, mu, examples, cond):
    """Three-strand SFT through ONE answer-sliced forward per example
    (full-sequence logits never materialized — E8-N OOM law; checkpointing
    engaged non-reentrantly and ASSERTED). For injected examples the
    backward runs INSIDE the Injector context (E8-N v2 law: checkpoint
    recompute must replay identical activations). Convergence = PER-STRAND
    plateau (the E8-N v2 minted lesson, prospective); hard cap EPOCHS_CAP."""
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
        'gradient checkpointing did not engage — refusing to train without it')
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
    strand_epoch_means = []
    plateaued, epochs_flown = False, 0
    t0 = time.time()

    def fwd_loss(ids, plen, ans):
        lo, hi = answer_slice(plen, ids.shape[1])
        hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
        logits = HEAD(hid[:, lo:hi, :]).float()
        return Fnn.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))

    for ep in range(EPOCHS_CAP):
        order = np.random.default_rng(E8R3C_SEED + 100 + ep).permutation(len(examples))
        ep_strand = {s: [] for s in STRAND_NAMES}
        for i in order:
            ex = examples[int(i)]
            ids, plen, ans, inj = encode_pairex(ex)
            max_tok = max(max_tok, ids.shape[1])
            ids, ans = ids.to(DEV), ans.to(DEV)
            if inj is not None:
                vec = inject_vec(dirs, mu, inj)
                with Injector(layer_mods, inj[1], vec, plen - 1) as injh:
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
            ep_strand[ex.get('strand', 'single')].append(lv)
            micro += 1
            if micro % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if micro % 100 == 0:
                print(f'    {cond} training ep{ep+1} {micro} micro-steps, '
                      f'loss~{np.mean(losses[-50:]):.3f}, {time.time()-t0:.0f}s')
        epochs_flown = ep + 1
        strand_epoch_means.append({s: (round(float(np.mean(v)), 4) if v else None)
                                   for s, v in ep_strand.items()})
        print(f'  {cond} epoch {epochs_flown}: '
              + ' | '.join(f'{s} {strand_epoch_means[-1][s]}' for s in STRAND_NAMES))
        if per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL):
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
           'epochs_cap': EPOCHS_CAP, 'plateaued': plateaued,
           'plateau_rel': PLATEAU_REL, 'min_epochs_strand': MIN_EPOCHS_STRAND,
           'strand_epoch_means': strand_epoch_means,
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
'''


def build_traincfg_cell():
    src = BR.CELL_TRAINCFG
    part1 = slice_block(src, "from transformers import", "LR = 1e-4",
                        tag="traincfg-1")
    part2 = slice_block(src, "tok = AutoTokenizer", None, tag="traincfg-2")
    consts = "LR = 1e-4\nACCUM = 4 if SMOKE else 8\n\n"
    return ("# ── Readout LoRA config + condition models + train loop (flown slices + v3c) ──\n"
            + part1 + consts + part2 + V3C_TRAIN_SRC)


STIMULUS_GLUE = r'''
# ── Eval row sets (locked verbatim + fresh seeded) + firewall ────────────────
LAYERS_RUN = [14, 20]   # both layers: dirs-stability gate parity with shipped
ANCHOR_ROWS = [{**t, 'block': 'anchor', 'grammar': 'single'}
               for t in build_plan(False)
               if t['kind'] == 'inject' and t['concept'] in TRAINED
               and t['layer'] == TRAIN_LAYER and t['alpha'] in TRAIN_ALPHAS]
assert len(ANCHOR_ROWS) == 18, len(ANCHOR_ROWS)
LOCKED_SHAM_GEN = [{**t, 'block': 'lockedsham_gen', 'grammar': 'single'}
                   for t in build_plan(False) if t['kind'] == 'sham']
assert len(LOCKED_SHAM_GEN) == N_SHAMS, len(LOCKED_SHAM_GEN)
if SMOKE:
    ANCHOR_ROWS = [ANCHOR_ROWS[0], ANCHOR_ROWS[-1]]
    LOCKED_SHAM_GEN = LOCKED_SHAM_GEN[:2]
PAIRSHAM_GEN = pairsham_gen_rows(SMOKE)
TRAINPAIR_GEN = trained_pair_gen_rows(SMOKE)
HELDPAIR_GEN = heldout_pair_gen_rows(SMOKE)
REACH_GEN = reach_rows(SMOKE)
TRAINPAIR_FC = trained_pair_fc_rows(SMOKE)
HELDPAIR_FC = heldout_pair_fc_rows(SMOKE)
REACH_FC = [{**t, 'block': 'reach_fc'} for t in reach_rows(SMOKE)]
_shams_all = [t for t in build_plan(False) if t['kind'] == 'sham']
if SMOKE:
    _shams_all = _shams_all[:2]
SHAM_FC = [{**t, 'block': 'sham_fc', 'grammar': 'pair'} for t in _shams_all]
GEN_ROWS = (ANCHOR_ROWS + LOCKED_SHAM_GEN + PAIRSHAM_GEN + TRAINPAIR_GEN
            + HELDPAIR_GEN + REACH_GEN)
FC_ROWS_ALL = TRAINPAIR_FC + HELDPAIR_FC + REACH_FC + SHAM_FC
print('eval rows:', {'anchor': len(ANCHOR_ROWS), 'lockedsham': len(LOCKED_SHAM_GEN),
      'pairsham': len(PAIRSHAM_GEN), 'trainpair_gen': len(TRAINPAIR_GEN),
      'heldpair_gen': len(HELDPAIR_GEN), 'reach': len(REACH_GEN),
      'trainpair_fc': len(TRAINPAIR_FC), 'heldpair_fc': len(HELDPAIR_FC),
      'reach_fc': len(REACH_FC), 'sham_fc': len(SHAM_FC)})
if not SMOKE:
    assert (len(ANCHOR_ROWS), len(LOCKED_SHAM_GEN), len(PAIRSHAM_GEN),
            len(TRAINPAIR_GEN), len(HELDPAIR_GEN), len(REACH_GEN),
            len(TRAINPAIR_FC), len(HELDPAIR_FC), len(REACH_FC),
            len(SHAM_FC)) == (18, 12, 12, 24, 48, 24, 48, 96, 24, 12), 'row counts'
HELD_PAIRS = heldout_pair_draw()
print('held-out pairs:', [f'{a}+{b}' for a, b in HELD_PAIRS])
'''


def build_stimulus_cell():
    s = BR.CELL_STIMULUS
    cent = slice_block(s, "import numpy as np", "# return_dict=True",
                       tag="stim-cent")
    canon = slice_block(s, "canon_prompt = ", "PLAN = build_plan",
                        tag="stim-canon")
    comp = slice_block(s, "def compute_stimulus", None, tag="stim-compute")
    inject = slice_block(BR.CELL_INJECT, "class Injector", "def encode_example",
                         tag="injector")
    return ("# ── Frozen stimulus (E7-Q code path) + Injector + generation, VERBATIM ──────\n"
            + cent + canon + comp + "\n" + inject + STIMULUS_GLUE)


CELL_PAIRRUN = r'''# ── Pair-grammar generation + forced-choice scoring (answer-sliced head) ────
def run_trial_pair(m, layer_mods, dirs, mu, trial, cond):
    """Generation under the pair grammar: composed injection for pair rows,
    single-direction injection for reach rows, none for shams."""
    prompt = pair_prompt(trial['order'], DESC)
    enc = tok.apply_chat_template([{'role':'user','content':prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    ids = enc['input_ids'].to(DEV)
    gen_kw = dict(max_new_tokens=24, do_sample=False,
                  pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.no_grad():
        if trial['kind'] in ('inject_pair', 'inject'):
            L = trial['layer']
            d = (pair_dvec(dirs[L], *trial['pair'])
                 if trial['kind'] == 'inject_pair'
                 else np.asarray(dirs[L][trial['concept']], float))
            vec = trial['alpha'] * mu[L] * torch.tensor(d)
            with Injector(layer_mods, L, vec, ids.shape[1] - 1) as inj:
                out = m.generate(input_ids=ids, **gen_kw)
            calls = inj.calls
            assert calls >= 1, 'injection hook never fired during generation'
        else:
            out = m.generate(input_ids=ids, **gen_kw)
            calls = 0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'cond': cond, 'response': text.strip()[:200],
            'parsed': parse_pair_report(text), 'hook_calls': calls}

PAIR_ELIGIBLE, PAIR_SCORED = pair_cands()
PAIR_CAND_IDS = {name: tok(name, add_special_tokens=False)['input_ids']
                 + [tok.eos_token_id] for name in PAIR_SCORED}
for _n, _ids in PAIR_CAND_IDS.items():
    _rt = tok.decode(_ids[:-1])
    assert _rt.strip().upper() == _n.upper(), f'candidate round-trip failed: {_n!r} -> {_rt!r}'
print('pair candidates:', len(PAIR_ELIGIBLE), 'eligible /', len(PAIR_SCORED),
      'scored | token counts',
      sorted({len(v) for v in PAIR_CAND_IDS.values()}))

def encode_pair_prompt(order):
    prompt = pair_prompt(order, DESC)
    enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    return enc['input_ids'][0]

def score_pair_trial(m, layer_mods, dirs14, mu14, trial, cond):
    """Forced choice at the measurement layer: ONE batched decoder forward
    over [pair prompt + candidate answer] for all 46 scored texts, injection
    hook live from the final prompt position, then the answer-sliced head —
    full-sequence logits never materialized (E8-N memory law)."""
    pid = encode_pair_prompt(trial['order'])
    plen = int(pid.shape[0])
    seqs = [torch.cat([pid, torch.tensor(PAIR_CAND_IDS[n], dtype=pid.dtype)])
            for n in PAIR_SCORED]
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
        if trial['kind'] in ('inject_pair', 'inject'):
            d = (pair_dvec(dirs14, *trial['pair'])
                 if trial['kind'] == 'inject_pair'
                 else np.asarray(dirs14[trial['concept']], float))
            vec = trial['alpha'] * mu14 * torch.tensor(d)
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
        for i, name in enumerate(PAIR_SCORED):
            n_ans = len(PAIR_CAND_IDS[name])
            logits = HEAD(hid[i, lo:lo + n_ans, :]).float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(PAIR_CAND_IDS[name], device=lp.device)
            tlp = lp[torch.arange(n_ans, device=lp.device), tgt]
            sm[name] = float(tlp.mean())
            ss[name] = float(tlp.sum())
    row = pair_fc_row(trial, sm, ss, VEC)
    row.update({'cond': cond, 'hook_calls': calls})
    return row
'''

CELL_FLIGHT = r'''# ── Flight loop: build -> stimulus+gate -> curriculum -> train -> eval -> ship ─
def fly_condition(cond):
    """One condition end-to-end inside ONE function scope (module-ref leak
    law): model, optimizer, activations all die on return."""
    t0 = time.time()
    bundle = {'condition': cond, 'mode': MODE, 'stamp': STAMP}
    try:
        model = build_condition_model(cond)
        layer_mods = resolve_layers(model)
        print(f'{cond}: model ready — {ram_report()}')
        dirs, mu = compute_stimulus(model, cond)
        stab = dirs_stability(dirs, SHIPPED[cond]['dirs'])
        bundle['dirs_stability'] = stab
        assert stab['pass'], f'dirs-stability gate FAILED: {stab}'
        print(f'  dirs-stability vs shipped E8-R: resid {stab["resid"]} — OK')
        bundle['mu'] = {str(L): mu[L] for L in mu}
        bundle['dirs'] = {str(L): {n: [round(float(x), 5) for x in dirs[L][n]]
                                   for n in dirs[L]} for L in dirs}
        singles = [{**e, 'strand': 'single' if e['kind'] == 'inject' else 'sham',
                    'grammar': 'single'} for e in build_train_set(SMOKE)]
        pairs_ex = build_pair_train_set(SMOKE)
        examples = singles + pairs_ex
        _v1 = validate_no_heldout_pair(examples)
        assert not _v1, f'FIREWALL — held-out pair in training: {_v1}'
        _v2 = validate_no_heldout_concept(examples)
        assert not _v2, f'FIREWALL — held-out concept in training: {_v2}'
        bundle['curriculum'] = {
            'singles_inject': sum(1 for e in singles if e['kind'] == 'inject'),
            'singles_sham': sum(1 for e in singles if e['kind'] == 'sham'),
            'pairs_inject': sum(1 for e in pairs_ex if e['kind'] == 'inject'),
            'pairs_sham': sum(1 for e in pairs_ex if e['kind'] == 'sham')}
        if not SMOKE:
            assert bundle['curriculum'] == {'singles_inject': 144,
                'singles_sham': 72, 'pairs_inject': 192, 'pairs_sham': 36}, \
                bundle['curriculum']
        print(f'  curriculum: {bundle["curriculum"]}')
        bundle['ppl_pre'] = retention_ppl(model)
        torch.cuda.empty_cache()
        bundle['train_log'] = train_readout_pairs(model, layer_mods, dirs, mu,
                                                  examples, cond)
        torch.cuda.empty_cache()
        gen_rows = {}
        for t in GEN_ROWS:
            fn = run_trial if t['grammar'] == 'single' else run_trial_pair
            r = fn(model, layer_mods, dirs, mu, t, cond)
            gen_rows.setdefault(t['block'], []).append(r)
        for b, rs in gen_rows.items():
            print(f'  {cond}/{b}: {len(rs)} generation rows')
        bundle['gen_rows'] = gen_rows
        print(f'-- {cond}/forced choice ({len(FC_ROWS_ALL)} rows x '
              f'{len(PAIR_SCORED)} candidates) --')
        bundle['fc_rows'] = [score_pair_trial(model, layer_mods,
                                              dirs[TRAIN_LAYER],
                                              mu[TRAIN_LAYER], t, cond)
                             for t in FC_ROWS_ALL]
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
            if b.get('fc_rows'):
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
        print(f'{cond}: done in {bundle.get("secs","?")}s — shipped')
        free_ram()
        print(f'{cond}: torn down — {ram_report()}')
'''

CELL_VERDICT = r'''# ── Gates, primaries P1/P2, spot, S-blocks, banner, ship ─────────────────────
have = {c: (c in RESULTS and 'error' not in RESULTS[c]
            and bool(RESULTS[c].get('fc_rows'))) for c in CONDITIONS}
print('conditions present:', have)
if not all(have.values()):
    missing = [c for c, ok in have.items() if not ok]
    print()
    print('=' * 72)
    print(f'  RUN COMPLETE — conditions still to fly: {missing}')
    print(f'  Next run: Runtime > Restart runtime, set RESUME_STAMP = '
          f"'{RESUME_STAMP or STAMP}',")
    print('  then Run all. This run\'s bundles are already shipped inflight.')
    print('=' * 72)
else:
    verdict = {'flight': f'{MODE}_{STAMP}', 'protocol': 'E8R3C_PROTOCOL.md',
               'resume_stamp': RESUME_STAMP or STAMP}

    def _fc(cond, block):
        return [r for r in RESULTS[cond]['fc_rows'] if r['block'] == block]

    def _gen(cond, block):
        return RESULTS[cond].get('gen_rows', {}).get(block, [])

    # ── gates (every gate rides a measured baseline) ─────────────────────────
    gates = {}
    for cond in CONDITIONS:
        b = RESULTS[cond]
        anchor = _gen(cond, 'anchor')
        a_exact = sum(1 for r in anchor if r.get('report') == r['concept'])
        sham_claims = sum(1 for r in _gen(cond, 'lockedsham_gen')
                          if r.get('report') not in ('NONE', 'INVALID'))
        ppl_d = 100.0 * (b['ppl_post'] / b['ppl_pre'] - 1.0)
        g = {'g1_anchor': {'exact': a_exact, 'n': len(anchor),
                           'min': ANCHOR_MIN_EXACT,
                           'pass': bool(len(anchor) < 18 or a_exact >= ANCHOR_MIN_EXACT)},
             'g1b_lockedsham': {'claims': sham_claims,
                                'max': LOCKEDSHAM_MAX_CLAIMS,
                                'pass': bool(sham_claims <= LOCKEDSHAM_MAX_CLAIMS)},
             'g2_dirs': b['dirs_stability'],
             'g3_ppl': {'pre': b['ppl_pre'], 'post': b['ppl_post'],
                        'delta_pct': round(ppl_d, 4), 'tol': PPL_TOL_PCT_R3,
                        'pass': bool(abs(ppl_d) <= PPL_TOL_PCT_R3)},
             'g4_counts': {'fc_rows': len(b['fc_rows']),
                           'pass': bool(SMOKE or len(b['fc_rows']) == 180)}}
        g['all_pass'] = all(v['pass'] for v in g.values() if isinstance(v, dict))
        gates[cond] = g
        print(f'{cond} gates:', {k: v['pass'] for k, v in g.items()
                                 if isinstance(v, dict)})
    verdict['gates'] = gates
    gates_ok = gates['real']['all_pass']

    # ── primaries on real (withheld if real gates fail) ──────────────────────
    hf_real = _fc('real', 'heldpair_fc')
    spot = {c: {'exact': sum(1 for r in _fc(c, 'trainpair_fc') if r.get('exact')),
                'n': len(_fc(c, 'trainpair_fc'))} for c in CONDITIONS}
    for c in CONDITIONS:
        spot[c]['installed'] = bool(spot[c]['exact'] >= (SPOT_MIN_EXACT if not SMOKE
                                                         else 0))
    verdict['spot'] = spot
    if gates_ok and hf_real:
        p1_obs, p1_p, _ = perm_null_pair_median(hf_real, VEC)
        p2_obs, p2_p, p2_nulls = perm_null_pair_exact(hf_real)
        hp = holm({'P1_median': p1_p, 'P2_exact': p2_p})
        verdict['P1'] = {'median_err': round(p1_obs, 2), 'p': round(p1_p, 5),
                         'n': len(hf_real), 'holm_pass': hp['P1_median'][1]}
        verdict['P2'] = {'exact': p2_obs, 'p': round(p2_p, 5),
                         'null_max': int(max(p2_nulls)) if p2_nulls else None,
                         'holm_pass': hp['P2_exact'][1]}
        verdict['S1_binom_uniform'] = round(binom_tail_r3(p2_obs, len(hf_real),
                                                          1 / 36), 6)
        verdict['S2_below_floor'] = below_floor_fraction(hf_real, VEC)
        verdict['S7_shared_atom'] = shared_atom_stats(hf_real)
        verdict['S8_sum_mean_agree'] = round(
            sum(1 for r in hf_real if r['agree_sum_mean']) / len(hf_real), 4)
        verdict['S9_alpha'] = {
            str(a): {'n': len([r for r in hf_real if r['alpha'] == a]),
                     'exact': sum(1 for r in hf_real
                                  if r['alpha'] == a and r.get('exact')),
                     'median_err': round(float(np.median(
                         [r['err'] for r in hf_real if r['alpha'] == a])), 2)}
            for a in sorted({r['alpha'] for r in hf_real})}
    else:
        verdict['P1'] = verdict['P2'] = {'withheld': True,
                                         'reason': 'gate fail or no rows'}

    # ── S3 scrambled contrast ────────────────────────────────────────────────
    hf_scr = _fc('scrambled', 'heldpair_fc')
    if hf_real and hf_scr and gates_ok and gates['scrambled']['all_pass']:
        rows_r = [{'err': r['err']} for r in hf_real if r.get('err') is not None]
        rows_s = [{'err': r['err']} for r in hf_scr if r.get('err') is not None]
        verdict['S3_scrambled'] = {
            'scr_median': round(float(np.median([r['err'] for r in rows_s])), 2),
            'scr_exact': sum(1 for r in hf_scr if r.get('exact')),
            'delta_median_scr_minus_real': boot_delta_median(rows_s, rows_r),
            'scr_spot': spot['scrambled']}

    # ── S4 generation blocks ─────────────────────────────────────────────────
    verdict['S4_gen'] = {}
    for cond in CONDITIONS:
        verdict['S4_gen'][cond] = {
            'heldpair': gen_pair_stats(_gen(cond, 'heldpair_gen'), VEC,
                                       nguard=GEN_NGUARD_MIN_NAMED),
            'trainpair': gen_pair_stats(_gen(cond, 'trainpair_gen'), VEC)}

    # ── S5 reach texture ─────────────────────────────────────────────────────
    verdict['S5_reach_floors'] = {h: {k: (round(v, 2) if isinstance(v, float) else v)
                                      for k, v in d.items()}
                                  for h, d in reach_floors(VEC).items()}
    verdict['S5_reach'] = {}
    for cond in CONDITIONS:
        rows = _fc(cond, 'reach_fc')
        by_c = {}
        for r in rows:
            by_c.setdefault(r['injected_concept'], []).append(r)
        verdict['S5_reach'][cond] = {
            c: {'n': len(rs),
                'median_err': round(float(np.median([r['err'] for r in rs])), 2),
                'argmaxes': sorted({r['argmax'] for r in rs})}
            for c, rs in by_c.items()}
        gen_reach = _gen(cond, 'reach_gen')
        verdict['S5_reach'][cond]['gen_kinds'] = {
            k: sum(1 for r in gen_reach if r['parsed']['kind'] == k)
            for k in ('PAIR', 'SINGLE', 'NONE', 'INVALID')}

    # ── S6 + S-B sham blocks ─────────────────────────────────────────────────
    verdict['S6_shams'] = {}
    for cond in CONDITIONS:
        shamfc = _fc(cond, 'sham_fc')
        none_top = sum(1 for r in shamfc if r['none_top'])
        verdict['S6_shams'][cond] = {
            'shamfc_none_top': f'{none_top}/{len(shamfc)}',
            'shamfc_argmax': sorted({r['argmax'] for r in shamfc}),
            'pairsham_gen_kinds': {
                k: sum(1 for r in _gen(cond, 'pairsham_gen')
                       if r['parsed']['kind'] == k)
                for k in ('PAIR', 'SINGLE', 'NONE', 'INVALID')}}
    sb_none = sum(1 for r in _fc('real', 'sham_fc') if r['none_top'])
    verdict['SB_claim'] = {'none_top': sb_none, 'n': len(_fc('real', 'sham_fc')),
                           'min': SHAMFC_NONE_TOP_MIN,
                           'pass': bool(sb_none >= (SHAMFC_NONE_TOP_MIN
                                                    if not SMOKE else 0))}

    # ── banner ───────────────────────────────────────────────────────────────
    print()
    print('=' * 72)
    if SMOKE:
        ok = gates_ok and all(have.values()) and not cond_errors
        print(f'  SMOKE {"GREEN" if ok else "RED"} — mechanics '
              f'{"exercised end-to-end" if ok else "FAILED — fix before full"}')
        print('  Next: Runtime > Restart runtime, set SMOKE = False, Run all.')
    else:
        p1ok = verdict['P1'].get('holm_pass', False)
        p2ok = verdict['P2'].get('holm_pass', False)
        inst = spot['real']['installed']
        print(f'  E8-R3-c VERDICT — gates {"PASS" if gates_ok else "FAIL"}')
        if gates_ok:
            print(f"  P1 composed readout: median {verdict['P1']['median_err']}deg "
                  f"p={verdict['P1']['p']} -> {'PASS' if p1ok else 'FAIL'}")
            print(f"  P2 spelling: exact {verdict['P2']['exact']}/{len(hf_real)} "
                  f"p={verdict['P2']['p']} -> {'PASS' if p2ok else 'FAIL'}")
            print(f"  spot installation: {spot['real']['exact']}/"
                  f"{spot['real']['n']} ({'installed' if inst else 'NOT INSTALLED'})")
            if p1ok and p2ok and inst:
                fork = 'SPELLING — level 2 (featural/Jamo) licensed'
            elif p1ok and not p2ok and inst:
                fork = 'PAIR-LOOKUP — composition wall extends into the answer space'
            elif not inst:
                fork = 'NOT INSTALLED — engineering branch (one pre-authorized re-fly)'
            elif not p1ok and inst:
                fork = 'UNREAD OFF-GRID — perception-level composition wall'
            else:
                fork = 'see protocol fork tree'
            print(f'  fork: {fork}')
            print(f"  S-B sham-FC NONE-top: {verdict['SB_claim']['none_top']}/12 "
                  f"-> {'PASS' if verdict['SB_claim']['pass'] else 'FAIL'}")
        else:
            print('  primaries WITHHELD (gate fail) — one engineering re-fly')
    print('=' * 72)
    vf = OUT / 'e8r3c_verdict.json'
    jdump(verdict, vf)
    ship(vf, f'e8r3c/{MODE}_{STAMP}')
    for cond in CONDITIONS:
        if cond in RESULTS:
            fn = OUT / f'condition_{cond}.json'
            if not fn.exists():
                jdump(RESULTS[cond], fn)
            ship(fn, f'e8r3c/{MODE}_{STAMP}')
    print('shipped: e8r3c/' + f'{MODE}_{STAMP}')
'''


def build_logic_cell():
    return E8R_LOGIC_SRC + "\n\n" + DIRS_STAB_SRC + "\n\n" + E8R3C_SRC


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

    logic = build_logic_cell()
    add("markdown", MD0)
    add("code", CELL_SETUP)
    add("code", logic)
    add("code", build_traincfg_cell())
    add("code", build_stimulus_cell())
    add("code", CELL_PAIRRUN)
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        assert "rclone" not in src.lower(), f"cell {i}: rclone reference (UI-only law)"

    out_nb = os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")
    with open(out_nb, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_nb, f"({len(nb['cells'])} cells, "
          f"{os.path.getsize(out_nb)/1024:.0f} KB)")

    logic_out = os.path.join(HERE, "e8r3c_logic.py")
    with open(logic_out, "w") as f:
        f.write(logic)
    print("wrote", logic_out)


if __name__ == "__main__":
    main()
