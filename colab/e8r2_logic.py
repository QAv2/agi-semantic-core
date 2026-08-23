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
