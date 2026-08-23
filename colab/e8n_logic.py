# ── E8N pure logic: labels, scoring, stats, validators (locally tested verbatim) ──
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
