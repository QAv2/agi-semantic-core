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

def answer_slice(prompt_len, total_len):
    """Hidden-state index range whose logits predict the answer tokens:
    position p predicts token p+1, so predicting ids[prompt_len:total_len]
    takes hidden[prompt_len-1 : total_len-1]. (v2 memory law, smoke-1 OOM:
    training must never materialize full-sequence logits — slice the head.)"""
    assert 0 < prompt_len < total_len, (prompt_len, total_len)
    return prompt_len - 1, total_len - 1

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


# ═══ naming + forced-choice machinery: e8r2_logic.py VERBATIM (carries e8r_logic) ═══
# ── E8R pure logic: plans, training set, parser, scoring (locally tested verbatim) ──
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


# ── E8R2 pure logic: forced-choice plans, scoring rows, stats (locally tested verbatim) ──
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

COMPETENCE_STOPLIST = set(['big', 'blinding', 'blindingly', 'boiling', 'bright', 'brightness', 'cold', 'dark', 'deafening', 'dry', 'fast', 'freezing', 'hot', 'huge', 'large', 'largest', 'light', 'loud', 'loudness', 'moonless', 'parked', 'quiet', 'silence', 'silent', 'size', 'slow', 'small', 'smallest', 'soaked', 'speed', 'still', 'temperature', 'tiny', 'wet', 'wetness', 'whisper'])

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

# ═══ E8-N v3 additions (docs/E8N3_PROTOCOL.md — pre-registered 26f17b0) ══════
E8N3_SEED = 20260827          # fresh stream; 20260822..26 are prior rungs'
EPOCHS_CAP_V3 = 12            # design check: catch-level ~ep 11, plateau ~15
S10_RECOVERED_MEDIAN = 35.0   # RECOVERED reading needs p<=.05 AND median<=this
CATCH_PRE_SANITY_MAX = 4      # pre catch > this => engineering NO_VERDICT
FC_BASELINE_SHA = '02ae2e7a0374a249'
FC_BASELINE_V2 = {5000: 63.274754, 5001: 24.176505, 5002: 24.176505, 5003: 24.176505, 5004: 24.176505, 5005: 24.176505, 5006: 63.274754, 5007: 24.176505, 5008: 24.176505, 5009: 24.176505, 5010: 24.176505, 5011: 63.274754, 5012: 44.379792, 5013: 24.625776, 5014: 24.176505, 5015: 24.176505, 5016: 24.176505, 5017: 24.176505, 5018: 24.625776, 5019: 44.379792, 5020: 24.176505, 5021: 24.176505, 5022: 24.176505, 5023: 24.176505, 5024: 60.793234, 5025: 52.511227, 5026: 52.511227, 5027: 60.793234, 5028: 52.511227, 5029: 71.337523, 5030: 61.330281, 5031: 60.793234, 5032: 71.337523, 5033: 60.793234, 5034: 60.793234, 5035: 52.511227, 5036: 59.3566, 5037: 48.715673, 5038: 48.715673, 5039: 52.511227, 5040: 0.0, 5041: 48.715673, 5042: 0.0, 5043: 59.3566, 5044: 59.3566, 5045: 60.793234, 5046: 28.536044, 5047: 60.793234, 5048: 71.008537, 5049: 13.943669, 5050: 13.943669, 5051: 13.943669, 5052: 71.008537, 5053: 71.008537, 5054: 13.943669, 5055: 71.008537, 5056: 13.943669, 5057: 13.943669, 5058: 71.008537, 5059: 13.943669, 5060: 43.820609, 5061: 52.879614, 5062: 52.879614, 5063: 65.060121, 5064: 43.820609, 5065: 71.008537, 5066: 13.943669, 5067: 52.879614, 5068: 13.943669, 5069: 65.060121, 5070: 71.008537, 5071: 58.571629, 5072: 54.051292, 5073: 76.388538, 5074: 54.051292, 5075: 54.051292, 5076: 76.388538, 5077: 54.051292, 5078: 76.388538, 5079: 63.166957, 5080: 18.37438, 5081: 54.051292, 5082: 54.051292, 5083: 18.37438, 5084: 46.981005, 5085: 30.835377, 5086: 46.981005, 5087: 46.981005, 5088: 46.981005, 5089: 30.835377, 5090: 30.835377, 5091: 46.981005, 5092: 46.981005, 5093: 46.981005, 5094: 30.835377, 5095: 46.981005}
EXPECT_CURRICULUM_V3 = {'scalar': 272, 'naming': 216,
                        'competence': 60, 'lexicon': 52}

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
        if len(seq) < max(2, min_epochs):   # a comparison needs two epochs
            return False                    # (smoke-1: min_epochs=1 crashed here)
        prev, cur = seq[-2], seq[-1]
        if prev <= 0:
            continue
        if (prev - cur) / prev >= rel:
            return False
    return True

def fc_baseline_sha_check():
    import hashlib as _h, json as _j
    js = _j.dumps({str(k): round(v, 6) for k, v in
                   sorted(FC_BASELINE_V2.items())}, sort_keys=True)
    return _h.sha256(js.encode()).hexdigest()[:16]

def s10_paired(fc_rows, baseline=None, n_perm=N_PERM, seed=None):
    """S10' — FC-interference recovery: per-tid paired sign-flip test of v3
    errors against the PINNED v2 error vector (one-sided, v3 < v2). An
    unchanged reader gives d ~ 0; recovery gives negative deltas."""
    if baseline is None:
        baseline = FC_BASELINE_V2
    if seed is None:
        seed = E8N3_SEED + 11
    pairs = [(float(r['err']), baseline[int(r['tid'])]) for r in fc_rows
             if r.get('err') is not None and int(r['tid']) in baseline]
    assert len(pairs) >= 2, f's10_paired needs >=2 paired rows, got {len(pairs)}'
    d = np.array([a - b for a, b in pairs])
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        if float((d * s).mean()) <= obs:      # one-sided: recovery = negative
            ge += 1
    med_v3 = float(np.median([a for a, _ in pairs]))
    med_v2 = float(np.median([b for _, b in pairs]))
    return {'n_paired': len(pairs), 'd_mean': round(obs, 4),
            'p': (1 + ge) / (1 + n_perm),
            'median_v3': round(med_v3, 2), 'median_v2_pinned': round(med_v2, 2),
            'n_improved': int((d < 0).sum())}

def s10_reading(s10):
    """The three pre-stated readings (protocol, verbatim)."""
    if s10['p'] <= 0.05 and s10['median_v3'] <= S10_RECOVERED_MEDIAN:
        return ('RECOVERED — interference was epoch-starvation; slate item 3 '
                'CLOSED by the same cure, no recovery probe flies')
    if s10['p'] <= 0.05:
        return ('PARTIAL — budget helps, a budget-independent component '
                'remains; naming-only recovery probe stays queued, target '
                'sharpened to the residual')
    return ('PERSISTS — multi-task interference is budget-independent; the '
            'naming-only continued-training probe is the queued next rung')

def calib_stats(rows_by_arm):
    scoring, meta = battery_rows_to_scoring(rows_by_arm)
    out = {}
    pooled_pairs = []
    for arm in ARMS:
        rows = scoring.get(arm) or []
        if len(rows) < MIN_POOLED_N:
            out[arm] = {"n": len(rows), "degenerate": True}
            continue
        rep = np.array([r["report"] for r in rows], float)
        ref = np.array([r["ref"] for r in rows], float)
        q10 = 10.0 * np.array(rank01(list(ref)))     # the trained target
        rho_rank = pooled_rho({arm: rows})["rho"]    # flown statistic
        refz = (ref - ref.mean()) / (ref.std() or 1.0)
        r_int = float(np.corrcoef(rep, refz)[0, 1])    # interval fidelity
        b_ = float(np.polyfit(q10, rep, 1)[0])
        a_ = float(np.polyfit(q10, rep, 1)[1])
        mae = float(np.mean(np.abs(rep - q10)))
        out[arm] = {"n": len(rows), "rho_rank": rho_rank,
                    "pearson_raw": round(r_int, 4),
                    "slope": round(b_, 4), "intercept": round(a_, 4),
                    "mae_vs_target": round(mae, 4),
                    "report_min": int(rep.min()), "report_max": int(rep.max()),
                    "report_var": round(float(rep.var()), 3)}
        pooled_pairs.append((q10, rep, arm))
    # cross-arm scale invariance: LOO arm transfer of the calibration line
    loo = {}
    for q10, rep, arm in pooled_pairs:
        tr_q = np.concatenate([q for q, r_, a in pooled_pairs if a != arm])
        tr_r = np.concatenate([r_ for q, r_, a in pooled_pairs if a != arm])
        bb, aa = np.polyfit(tr_q, tr_r, 1)
        pred = aa + bb * q10
        mae_t = float(np.mean(np.abs(pred - rep)))
        own_b, own_a = np.polyfit(q10, rep, 1)
        mae_o = float(np.mean(np.abs(own_a + own_b * q10 - rep)))
        loo[arm] = {"mae_transfer": round(mae_t, 4),
                    "mae_own": round(mae_o, 4),
                    "transfer_penalty": round(mae_t - mae_o, 4)}
    return {'per_arm': out, 'loo': loo}
