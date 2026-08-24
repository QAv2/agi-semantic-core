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


# ── E8J pure logic: draw pins, bridge, nulls, stage plans, verdict (locally tested verbatim) ──
# (builds on the e8r2_logic namespace: CHOICE_SET, HELD_OUT, TRAINED, LAYERS,
#  TRAIN_LAYER, anchor_rows, sham_rows, fc_row, gate_g1, gate_g1b, sham_prior,
#  titration_curve, binom_tail, report_prompt, parse_report, angle14, jdump)
import hashlib, time

E8J_SEED = 20260825             # all E8-J-new randomness; disjoint from prior rungs
DESC_MIN = 40                   # anchor frame: pack desc length floor
N_ANCH = 320
N_ANCH_SMOKE = 48
STRATUM_MIN = 8                 # per-trigram floor in the full draw
LAM = 1.0                       # bridge ridge (pinned; sensitivity texture below)
LAM_SENS = [0.1, 10.0]
LAM_REV = 10.0                  # A4 reverse map hidden->axis (descriptive row)
N_PERM_RELABEL = 2000           # A2/A3 codebook relabel null
N_PERM_RELABEL_SMOKE = 200
N_PERM_RSA = 10000              # A1 Mantel
N_PERM_RSA_SMOKE = 500
N_PERM_DERANGE = 2000           # A5/G-B derangement null
N_PERM_DERANGE_SMOKE = 200
N_PERM_PB = 10000               # P-B matched-pair permutation
N_PERM_PB_SMOKE = 500
GB_MIN_HITS = 4                 # G-B clause (i): wing exact geometric hits >= 4/13
GB_NULL_PCT = 95.0              # G-B clause (ii): > 95th pctile of derangement null
B_ALPHAS = [0.5, 1.0]           # Stage-B injection grid (trained regime verbatim)
B_ALPHAS_SMOKE = [0.5]
N_B_ORDERS = 4
N_B_ORDERS_SMOKE = 2
TITR_ALPHA_B = 1.5              # bridge titration texture (past the 0.25->0.30 cliff)
N_TITR_ORDERS = 2
BEHAV_HIT_MIN = 5               # mechanism row: real exact >= 5 of 8 rows (full mode)
DIRS_TOL_E8J = 1e-5             # G2 vs shipped E8-R real bundle (rides 5e-08)
AXIS_NAMES = ['x','y','z','e','f','g','h','fx','fy','fz','fe','ff','fg','fh']
E8R2_MODAL = {'DIVERGENCE': 'CONFIDENCE', 'NOVELTY': 'CONFABULATION',
              'RETRIEVAL': 'FAMILIARITY', 'TENSION': 'CALIBRATION'}

PINNED_ANCHORS = [
    'ABYSMAL', 'ACCRUE', 'ACQUIESCE', 'ACTIVITY',
    'ALLOY', 'AMBER', 'AMBIGUOUS', 'AMOUNT',
    'ANECDOTE', 'ANOTHER', 'APPARATUS', 'APPARENT',
    'APPROVAL', 'ARBITRATE', 'ARCADE', 'ARCHITECT',
    'ASSIMILATION', 'AVALANCHE', 'AWNING', 'BAFFLE',
    'BAIL', 'BARK_DOG', 'BARNACLE', 'BAT_SPORTS',
    'BELLOWS', 'BLEACHED', 'BLEND', 'BLIGHT',
    'BOARD_GROUP', 'BOLT_RUN', 'BRANDISH', 'BREW',
    'BREWING', 'BRIDGE_STRUCTURE', 'BRILLIANCE', 'BUREAUCRACY',
    'BUT', 'BUTTERFLY', 'CALIBRATE', 'CANDOR',
    'CANYON', 'CAPACITY', 'CARTILAGE', 'CELL_PRISON',
    'CERAMIC', 'CHARGE_ATTACK', 'CHEMICAL', 'CINDER',
    'CIRCUMSPECTION', 'CITADEL', 'CLIENTELE', 'CLOTH',
    'COLLATERAL', 'COMMODITY', 'COMMON', 'COMMUNICATE',
    'COMPACT', 'CONFLUENCE', 'CONTEMPLATE', 'COTTON',
    'COULD', 'COVERING', 'COVERT', 'CRUCIBLE',
    'CRYSTALLIZATION', 'CUBE', 'DEFINE', 'DEGREE',
    'DEPTH', 'DESPONDENCY', 'DIRGE', 'DISCLOSED',
    'DISSEMINATE', 'DISSOCIATION', 'DOLMEN', 'DOMAIN',
    'DORMANCY_STATE', 'DORMANT_THING', 'DOWNWARD', 'DROUGHT',
    'DUPLICITY', 'DYNAMISM', 'ELEVEN', 'ENGAGE',
    'ENIGMA', 'ENJOY', 'ENTROPY_SOCIAL', 'EUPHORIA',
    'EVAPORATE', 'EXHORT', 'EXPLOITATION', 'EXTORT',
    'FAMINE', 'FAN_ADMIRER', 'FASTEN', 'FERMENTATION',
    'FEUD', 'FEUDALISM', 'FIDUCIARY', 'FILTER',
    'FIRE_SHOOT', 'FLUID', 'FORBEARANCE', 'FORTH',
    'FOUNDATION', 'FREEDOM', 'GALLANTRY', 'GAMBLE',
    'GAME', 'GAME_VERB', 'GENEALOGY', 'GESTATION',
    'GOAL', 'GONDOLA', 'GRANITE', 'GRATITUDE',
    'HARM', 'HEALING', 'HEMORRHAGE', 'HILL',
    'HORN_ANIMAL', 'HORSE', 'HUSBAND', 'IMPASSIVITY',
    'IMPLICATE', 'INCREDULITY', 'INDENTURE', 'INERTIA',
    'INFLATION', 'INFUSION', 'INOCULATION', 'INWARD',
    'IRIDESCENCE', 'ISOTOPE', 'JELLYFISH', 'JURISDICTION',
    'KELP', 'LANGUAGE', 'LANGUISH', 'LARVAE',
    'LEADER', 'LEATHER', 'LEAVE', 'LETTER_MAIL',
    'LEVY', 'LIMIT', 'LINEN', 'LITIGATION',
    'LOAM', 'LOCATION', 'LOCOMOTIVE', 'LUNG',
    'MALE', 'MARINATING', 'MATHEMATICS', 'MATURATION',
    'MEANWHILE', 'MEASURE', 'MEDICINE', 'MEDITATE',
    'MEDITATION', 'METABOLISM', 'MIDDLE', 'MIGRATION',
    'MINERAL', 'MINUTE', 'MOLD_SHAPE', 'MONEY',
    'MONKEY', 'MYSTIC', 'NEGLIGENCE', 'NINE',
    'NUMBNESS', 'OASIS', 'OBFUSCATE', 'ON',
    'OPAL', 'OPENING', 'OPPRESSION', 'ORGAN',
    'OVERWHELM', 'PALM_HAND', 'PARABLE', 'PASS',
    'PATINA', 'PEAT', 'PERCEPTION', 'PERCOLATE',
    'PERISH', 'PERJURY', 'PERMANENCE', 'PIGMENT',
    'PLAN', 'PLASTIC', 'PLATFORM', 'PLAY_THEATER',
    'PLEASURE', 'PLIGHT', 'POINT_ARGUMENT', 'POINT_SCORE',
    'POINT_TIP', 'POLE_STICK', 'POSITION', 'PREDATOR',
    'PRESS_MEDIA', 'PROMULGATE', 'PROPAGATION', 'PROPEL',
    'PROSTHESIS', 'PROVERB', 'PULLEY', 'PUNCTUAL',
    'PUPIL', 'QUICK', 'RABBIT', 'RAISE',
    'RATIFICATION', 'REALITY', 'REFERENDUM', 'RELENT',
    'REMEMBER', 'REMISSION', 'REMNANT', 'RESPONSIBILITY',
    'REVOLUTION', 'ROOF', 'ROOTED', 'ROUND',
    'ROYAL', 'RUDDER', 'SACRAMENT', 'SACRED',
    'SATIRE', 'SAVANNA', 'SCOUT', 'SEASON',
    'SECOND', 'SECURE_VERB', 'SEDIMENTATION', 'SEDIMENTATION_LAKE',
    'SELECTION', 'SENSE', 'SHALLOW', 'SHAPE',
    'SHARK', 'SHORT', 'SHOW', 'SIDE',
    'SILICIFICATION', 'SILTING', 'SIREN', 'SKELETON',
    'SKULK', 'SLEIGH', 'SMUGGLE', 'SOURDOUGH',
    'SOUVENIR', 'SPARK', 'SQUALL', 'STALK_FOLLOW',
    'STALK_STEM', 'STATE', 'STEADY', 'STEEP',
    'STOCK_FINANCIAL', 'STOCK_SUPPLY', 'STONE', 'STORY',
    'STRONG', 'SUBCONSCIOUS', 'SUBLIMATION_PHYS', 'SUBORDINATION',
    'SUDDEN', 'SUFFICIENT', 'SUNDIAL', 'SYNDICATE',
    'TACITNESS', 'TARNISH', 'TEA', 'TELL',
    'TEMPERANCE', 'TEST', 'THICK', 'THROUGH',
    'THUNDER', 'THUS', 'TONE', 'TOPAZ',
    'TORPOR', 'TRANSFERENCE', 'TRANSFORM', 'TRIP_JOURNEY',
    'TRUE', 'TUNDRA', 'TUNIC', 'TWEED',
    'UNDERSTANDING', 'UNIVERSAL', 'UNLESS', 'UPPER',
    'VERTIGO', 'VESTIBULE', 'VETO', 'VIEW',
    'VOID', 'WAIT', 'WALL', 'WAVE_GESTURE',
    'WEAK', 'WEATHER', 'WEAVE', 'WEEP',
    'WEIGHT', 'WHEREAS', 'WHISPER', 'WIELD',
    'WIFE', 'WILTING', 'WISTFULNESS', 'YEAST',
]

PINNED_ANCHORS_SMOKE = [
    'APPROVAL', 'ARBITRATE', 'BAFFLE', 'BAIL',
    'BREW', 'BRIDGE_STRUCTURE', 'BRILLIANCE', 'BUREAUCRACY',
    'BUT', 'BUTTERFLY', 'CANDOR', 'CELL_PRISON',
    'CINDER', 'CIRCUMSPECTION', 'CITADEL', 'CLIENTELE',
    'CLOTH', 'CONTEMPLATE', 'CRYSTALLIZATION', 'DYNAMISM',
    'EXHORT', 'FAN_ADMIRER', 'FOUNDATION', 'GAMBLE',
    'GENEALOGY', 'GRANITE', 'GRATITUDE', 'HORSE',
    'IMPASSIVITY', 'LINEN', 'MATHEMATICS', 'MATURATION',
    'MEDITATION', 'MIDDLE', 'NUMBNESS', 'ORGAN',
    'OVERWHELM', 'PLATFORM', 'POINT_TIP', 'SACRAMENT',
    'SACRED', 'SAVANNA', 'SIDE', 'SIREN',
    'SOUVENIR', 'TARNISH', 'THICK', 'TUNIC',
]

ANCHORS_SHA = 'dfb115cded2e1cd3'

def anchor_frame(pack_concepts):
    """Sampling frame: pack concepts with a real description, wing excluded."""
    return sorted(c['name'] for c in pack_concepts
                  if len(c.get('desc') or '') >= DESC_MIN
                  and c['name'] not in CHOICE_SET)

def compute_anchor_draw(frame, trigram_by_name, n=N_ANCH, seed=E8J_SEED):
    """Deterministic stratified draw: proportional by trigram with per-stratum
    floor STRATUM_MIN (or the whole stratum if smaller), largest-remainder
    rebalance to hit n exactly, seeded within-stratum choice. Single source:
    the builder ran THIS function to mint PINNED_ANCHORS."""
    strata = {}
    for name in frame:
        strata.setdefault(trigram_by_name.get(name) or '_', []).append(name)
    keys = sorted(strata)
    total = sum(len(strata[k]) for k in keys)
    quota = {k: n * len(strata[k]) / total for k in keys}
    take = {k: min(len(strata[k]), max(min(STRATUM_MIN, len(strata[k])),
                                       int(quota[k]))) for k in keys}
    def _over():
        return sum(take.values()) - n
    while _over() > 0:
        k = max((k for k in keys if take[k] > min(STRATUM_MIN, len(strata[k]))),
                key=lambda k: (take[k] - quota[k], k))
        take[k] -= 1
    while _over() < 0:
        k = max((k for k in keys if take[k] < len(strata[k])),
                key=lambda k: (quota[k] - take[k], k))
        take[k] += 1
    rng = np.random.default_rng(seed)
    out = []
    for k in keys:
        out.extend(sorted(rng.choice(sorted(strata[k]), size=take[k],
                                     replace=False).tolist()))
    return sorted(out)

def compute_smoke_draw(full_draw, n=N_ANCH_SMOKE, seed=E8J_SEED + 1):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(full_draw), size=n, replace=False).tolist())

def wing_derangement(seed=E8J_SEED + 2):
    """Pinned derangement of the 13 wing targets (no fixed points)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(CHOICE_SET))
    perm = rng.permutation(idx)
    while np.any(perm == idx):
        perm = rng.permutation(idx)
    return {CHOICE_SET[i]: CHOICE_SET[int(perm[i])] for i in range(len(idx))}

def verify_pins(pack_concepts):
    """G4 pin verification on the VM: frame membership, counts, hash, subset."""
    frame = set(anchor_frame(pack_concepts))
    assert len(PINNED_ANCHORS) == N_ANCH, len(PINNED_ANCHORS)
    assert len(set(PINNED_ANCHORS)) == N_ANCH, 'duplicate anchors'
    assert all(a in frame for a in PINNED_ANCHORS), 'anchor outside frame'
    assert not (set(PINNED_ANCHORS) & set(CHOICE_SET)), 'wing leaked into anchors'
    assert set(PINNED_ANCHORS_SMOKE) <= set(PINNED_ANCHORS), 'smoke not a subset'
    assert len(PINNED_ANCHORS_SMOKE) == N_ANCH_SMOKE
    sha = hashlib.sha256('|'.join(PINNED_ANCHORS).encode()).hexdigest()[:16]
    assert sha == ANCHORS_SHA, (sha, ANCHORS_SHA)
    der = wing_derangement()
    assert sorted(der) == sorted(CHOICE_SET)
    assert sorted(der.values()) == sorted(CHOICE_SET)
    assert all(der[c] != c for c in der), 'derangement has a fixed point'
    return {'n_anchors': len(PINNED_ANCHORS), 'sha': sha,
            'n_smoke': len(PINNED_ANCHORS_SMOKE), 'frame_size': len(frame)}

# ── bridge algebra ───────────────────────────────────────────────────────────
def ridge_fit(X, Y, lam):
    """Ridge with unpenalized intercept. X (n,d), Y (n,h) -> W (d+1,h)."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    P = np.eye(Xa.shape[1]) * lam
    P[-1, -1] = 0.0
    return np.linalg.solve(Xa.T @ Xa + P, Xa.T @ Y)

def bridge_predict(W, X):
    """Unit-normalized predictions for coordinate rows X."""
    P = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
    return P / np.linalg.norm(P, axis=1, keepdims=True)

def hat_matrix(X, lam):
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    P = np.eye(Xa.shape[1]) * lam
    P[-1, -1] = 0.0
    return Xa @ np.linalg.solve(Xa.T @ Xa + P, Xa.T)

def loo_preds_from_hat(H, Y):
    """Exact ridge LOO predictions: (HY - diag(H) Y) / (1 - diag(H)), unit rows."""
    h = np.diag(H)
    P = (H @ Y - h[:, None] * Y) / (1.0 - h)[:, None]
    return P / np.linalg.norm(P, axis=1, keepdims=True)

def angles_rowwise(P, D):
    c = np.clip(np.sum(P * D, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def retrieval_stats(P, D):
    """Per-row angle to own true dir + top-1/top-5 retrieval among all rows."""
    cos = np.clip(P @ D.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    own = angles_rowwise(P, D)
    order = np.argsort(ang, axis=1)
    top1 = (order[:, 0] == np.arange(len(P)))
    top5 = np.any(order[:, :5] == np.arange(len(P))[:, None], axis=1)
    return own, int(top1.sum()), int(top5.sum())

def relabel_null_loo(X, Y, lam, n_perm, seed, progress_every=200):
    """Codebook relabel null for A2/A3: permute the dir rows against the fixed
    coordinate rows (H depends only on X, so LOO reuses one hat matrix)."""
    H = hat_matrix(X, lam)
    rng = np.random.default_rng(seed)
    tops, meds = [], []
    for i in range(n_perm):
        pi = rng.permutation(Y.shape[0])
        Yp = Y[pi]
        P = loo_preds_from_hat(H, Yp)
        own, t1, _ = retrieval_stats(P, Yp)
        tops.append(t1)
        meds.append(float(np.median(own)))
        if progress_every and (i + 1) % progress_every == 0:
            print(f'    relabel null {i + 1}/{n_perm}')
    return np.array(tops), np.array(meds)

def pairwise_ang(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(Mn @ Mn.T, -1.0, 1.0)))

def mantel_spearman(A, B, n_perm, seed, progress_every=2000):
    """Mantel test, Spearman over upper triangles; permutes A's labels."""
    iu = np.triu_indices(A.shape[0], 1)
    rb = np.argsort(np.argsort(B[iu])).astype(float)
    def _sp(x):
        rx = np.argsort(np.argsort(x)).astype(float)
        return float(np.corrcoef(rx, rb)[0, 1])
    obs = _sp(A[iu])
    rng = np.random.default_rng(seed)
    cnt = 0
    for i in range(n_perm):
        p = rng.permutation(A.shape[0])
        cnt += _sp(A[np.ix_(p, p)][iu]) >= obs
        if progress_every and (i + 1) % progress_every == 0:
            print(f'    mantel {i + 1}/{n_perm}')
    return obs, (cnt + 1) / (n_perm + 1)

def axiswise_reverse_r2(Hdirs, X, lam=LAM_REV, k=10, seed=E8J_SEED + 5):
    """A4: hidden->each dictionary axis, k-fold CV R^2 (descriptive row)."""
    n = Hdirs.shape[0]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    out = {}
    for j, name in enumerate(AXIS_NAMES):
        y = X[:, j]
        sse, sst = 0.0, 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            W = ridge_fit(Hdirs[tr], y[tr, None], lam)
            pred = (np.hstack([Hdirs[f], np.ones((len(f), 1))]) @ W)[:, 0]
            sse += float(np.sum((y[f] - pred) ** 2))
            sst += float(np.sum((y[f] - np.mean(y[tr])) ** 2))
        out[name] = round(1.0 - sse / max(sst, 1e-12), 4)
    return out

def wing_transfer(Xa, Ya, Xw, Dw, lam):
    """A5: bridge on ALL anchors -> predict the never-anchored wing dirs."""
    W = ridge_fit(Xa, Ya, lam)
    P = bridge_predict(W, Xw)
    own = angles_rowwise(P, Dw)
    cos = np.clip(P @ Dw.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    nearest = np.argmin(ang, axis=1)
    hits = (nearest == np.arange(len(Xw)))
    return W, P, own, nearest, int(hits.sum())

def derangement_null_transfer(W, Xw, Dw, n_perm, seed):
    """G-B null: wing coordinates permuted (derangements) against the fixed
    bridge — the fit never changes, only which coordinate claims which dir."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Xw))
    hits_n, med_n = [], []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while np.any(pi == idx):
            pi = rng.permutation(idx)
        P = bridge_predict(W, Xw[pi])
        own = angles_rowwise(P, Dw)
        cos = np.clip(P @ Dw.T, -1.0, 1.0)
        nearest = np.argmin(np.degrees(np.arccos(cos)), axis=1)
        hits_n.append(int((nearest == idx).sum()))
        med_n.append(float(np.median(own)))
    return np.array(hits_n), np.array(med_n)

def gate_gb(hits, null_hits):
    p95 = float(np.percentile(null_hits, GB_NULL_PCT))
    p = float((1 + int((null_hits >= hits).sum())) / (1 + len(null_hits)))
    return {'hits': int(hits), 'min_hits': GB_MIN_HITS,
            'null_p95': round(p95, 2), 'null_mean': round(float(null_hits.mean()), 3),
            'p_hits': round(p, 5),
            'pass': bool(hits >= GB_MIN_HITS and hits > p95)}

# ── Stage A orchestration (pure numpy; model already torn down) ──────────────
def stage_a_verdict(anchors, vec_by_name, dirs_inst14, dirs_inst20, dirs_base14,
                    smoke=False):
    t0 = time.time()
    n_rel = N_PERM_RELABEL_SMOKE if smoke else N_PERM_RELABEL
    n_rsa = N_PERM_RSA_SMOKE if smoke else N_PERM_RSA
    n_der = N_PERM_DERANGE_SMOKE if smoke else N_PERM_DERANGE
    Xa = np.array([vec_by_name[a] for a in anchors], float)
    Xw = np.array([vec_by_name[c] for c in CHOICE_SET], float)
    def _D(dd, names):
        M = np.array([dd[n] for n in names], float)
        return M / np.linalg.norm(M, axis=1, keepdims=True)
    Ya14 = _D(dirs_inst14, anchors)
    Dw14 = _D(dirs_inst14, CHOICE_SET)
    atlas = {'n_anchors': len(anchors), 'anchors_sha': ANCHORS_SHA,
             'lam': LAM, 'smoke': bool(smoke)}

    print('  A1 RSA (instilled L14)...')
    Ad = pairwise_ang(Xa)
    Ah = pairwise_ang(Ya14)
    rho, p = mantel_spearman(Ad, Ah, n_rsa, E8J_SEED + 10)
    atlas['A1_rsa_L14'] = {'rho': round(rho, 4), 'p': round(p, 5), 'n_perm': n_rsa}
    rho_f, p_f = mantel_spearman(Ad, np.minimum(Ah, 180.0 - Ah), n_rsa,
                                 E8J_SEED + 11)
    atlas['S6_folded_L14'] = {'rho': round(rho_f, 4), 'p': round(p_f, 5)}

    print('  A2/A3 LOO bridge + relabel null (instilled L14)...')
    H = hat_matrix(Xa, LAM)
    P = loo_preds_from_hat(H, Ya14)
    own, t1, t5 = retrieval_stats(P, Ya14)
    tops, meds = relabel_null_loo(Xa, Ya14, LAM, n_rel, E8J_SEED + 12)
    atlas['A2_loo'] = {'median_angle': round(float(np.median(own)), 2),
                       'null_median_mean': round(float(meds.mean()), 2),
                       'p_median': round(float((1 + int((meds <= np.median(own)).sum()))
                                               / (1 + len(meds))), 5)}
    atlas['A3_retrieval'] = {'top1': t1, 'top5': t5, 'n': len(anchors),
                             'null_top1_mean': round(float(tops.mean()), 3),
                             'null_top1_max': int(tops.max()),
                             'p_top1': round(float((1 + int((tops >= t1).sum()))
                                                   / (1 + len(tops))), 5)}
    atlas['P_A'] = {'top1': t1, 'p': atlas['A3_retrieval']['p_top1'],
                    'pass': bool(atlas['A3_retrieval']['p_top1'] < 0.05)}
    sens = {}
    for lam in LAM_SENS:
        Ps = loo_preds_from_hat(hat_matrix(Xa, lam), Ya14)
        so, st1, _ = retrieval_stats(Ps, Ya14)
        sens[str(lam)] = {'median_angle': round(float(np.median(so)), 2), 'top1': st1}
    atlas['S5_lam_sens'] = sens

    print('  A4 axis-wise reverse map...')
    atlas['A4_axiswise_r2'] = axiswise_reverse_r2(Ya14, Xa)

    print('  A5 wing transfer + derangement null...')
    W, Pw, own_w, nearest_w, hits_w = wing_transfer(Xa, Ya14, Xw, Dw14, LAM)
    null_h, null_m = derangement_null_transfer(W, Xw, Dw14, n_der, E8J_SEED + 13)
    tr_idx = [CHOICE_SET.index(c) for c in TRAINED]
    per = {}
    for i, c in enumerate(CHOICE_SET):
        lookup = float(min(np.degrees(np.arccos(np.clip(
            np.dot(Dw14[j], Dw14[i]), -1, 1))) for j in tr_idx if j != i))
        per[c] = {'angle': round(float(own_w[i]), 2),
                  'nearest': CHOICE_SET[int(nearest_w[i])],
                  'hit': bool(nearest_w[i] == i),
                  'lookup_floor': round(lookup, 2)}
    atlas['A5_wing'] = {'per_target': per, 'hits': hits_w,
                        'median_angle': round(float(np.median(own_w)), 2),
                        'null_median_mean': round(float(null_m.mean()), 2),
                        'p_median': round(float((1 + int((null_m <= np.median(own_w)).sum()))
                                                / (1 + len(null_m))), 5)}
    atlas['gb'] = gate_gb(hits_w, null_h)
    atlas['wing_pred_real'] = {c: [round(float(x), 5) for x in Pw[i]]
                               for i, c in enumerate(CHOICE_SET)}
    atlas['derangement'] = wing_derangement()

    print('  S3 L20 texture + S4 base-model contrast...')
    Ya20 = _D(dirs_inst20, anchors)
    P20 = loo_preds_from_hat(hat_matrix(Xa, LAM), Ya20)
    o20, t120, _ = retrieval_stats(P20, Ya20)
    rho20, p20 = mantel_spearman(Ad, pairwise_ang(Ya20),
                                 min(n_rsa, 2000), E8J_SEED + 14)
    atlas['S3_L20'] = {'rsa_rho': round(rho20, 4), 'rsa_p': round(p20, 5),
                       'loo_median': round(float(np.median(o20)), 2), 'top1': t120}
    Yb14 = _D(dirs_base14, anchors)
    Pb = loo_preds_from_hat(hat_matrix(Xa, LAM), Yb14)
    ob, t1b, _ = retrieval_stats(Pb, Yb14)
    rhob, pb = mantel_spearman(Ad, pairwise_ang(Yb14),
                               min(n_rsa, 2000), E8J_SEED + 15)
    atlas['S4_base'] = {'rsa_rho': round(rhob, 4), 'rsa_p': round(pb, 5),
                        'loo_median': round(float(np.median(ob)), 2), 'top1': t1b,
                        'delta_top1_inst_minus_base': t1 - t1b}
    atlas['secs_stageA'] = round(time.time() - t0, 1)
    return atlas

# ── Stage B plans ────────────────────────────────────────────────────────────
def build_b_pairs(smoke=False):
    """Matched-pair FC rows: per (target, alpha, rep) ONE shared menu order,
    two rows — arm 'real' injects the target's own bridge prediction, arm
    'perm' injects the derangement partner's prediction. exact is always
    scored against the TARGET (the pairing question)."""
    rng = np.random.default_rng(E8J_SEED + 3)
    targets = CHOICE_SET
    alphas, n_orders = B_ALPHAS, N_B_ORDERS
    if smoke:
        targets = [TRAINED[0], HELD_OUT[0]]
        alphas, n_orders = B_ALPHAS_SMOKE, N_B_ORDERS_SMOKE
    rows, tid, pid = [], 8000, 0
    for c in targets:
        for a in alphas:
            for _ in range(n_orders):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                for arm in ('real', 'perm'):
                    rows.append({'tid': tid, 'kind': 'inject', 'concept': c,
                                 'layer': TRAIN_LAYER, 'alpha': a, 'order': order,
                                 'block': f'bridge_{arm}', 'arm': arm,
                                 'pair_id': pid})
                    tid += 1
                pid += 1
    return rows

def build_titr_b(smoke=False):
    """Bridge titration texture: alpha=1.5, real predictions, free-report."""
    if smoke:
        return []
    rng = np.random.default_rng(E8J_SEED + 4)
    rows, tid = [], 9000
    for c in CHOICE_SET:
        for _ in range(N_TITR_ORDERS):
            order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
            rows.append({'tid': tid, 'kind': 'inject', 'concept': c,
                         'layer': TRAIN_LAYER, 'alpha': TITR_ALPHA_B,
                         'order': order, 'block': 'bridge_titr'})
            tid += 1
    return rows

# ── Stage B statistics ───────────────────────────────────────────────────────
def _pair_diffs(rows, targets):
    pairs = {}
    for r in rows:
        if r['injected'] in targets:
            pairs.setdefault(r['pair_id'], {})[r['arm']] = r
    diffs, by_target = [], {}
    for pid in sorted(pairs):
        p = pairs[pid]
        if 'real' in p and 'perm' in p:
            d = int(bool(p['real']['exact'])) - int(bool(p['perm']['exact']))
            diffs.append(d)
            by_target.setdefault(p['real']['injected'], []).append(d)
    return np.array(diffs), by_target

def pb_matched_perm(rows, n_perm=N_PERM_PB, seed=E8J_SEED + 6, targets=None):
    """P-B primary: real-vs-permuted exact naming over matched pairs
    (trained targets), sign-flip permutation on pair differences."""
    if targets is None:
        targets = TRAINED
    diffs, _ = _pair_diffs(rows, targets)
    if len(diffs) == 0:
        return {'n_pairs': 0, 'obs_diff': None, 'p': None}
    obs = int(diffs.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    null = (signs * diffs[None, :]).sum(axis=1)
    p = float((1 + int((null >= obs).sum())) / (1 + n_perm))
    return {'n_pairs': int(len(diffs)), 'obs_diff': obs,
            'real_exact': int(sum(1 for r in rows if r['arm'] == 'real'
                                  and r['injected'] in targets and r['exact'])),
            'perm_exact': int(sum(1 for r in rows if r['arm'] == 'perm'
                                  and r['injected'] in targets and r['exact'])),
            'p': round(p, 5)}

def pb_signflip_targets(rows, targets=None):
    """Registered robustness: exhaustive target-level sign-flip exact test."""
    if targets is None:
        targets = TRAINED
    _, by_t = _pair_diffs(rows, targets)
    names = sorted(by_t)
    D = np.array([sum(by_t[t]) for t in names], float)
    if len(D) == 0:
        return {'n_targets': 0, 'p': None}
    obs = float(D.sum())
    k = len(D)
    ge = 0
    for mask in range(2 ** k):
        s = sum((D[i] if (mask >> i) & 1 else -D[i]) for i in range(k))
        ge += (s >= obs)
    return {'n_targets': k, 'obs': obs,
            'per_target': {t: int(sum(by_t[t])) for t in names},
            'p': round(float(ge / 2 ** k), 5)}

def pb_mechanism(rows, atlas, smoke=False):
    """Behavioral hits must sit inside A5's geometric hits."""
    need = 1 if smoke else BEHAV_HIT_MIN
    geo = {c for c, d in atlas['A5_wing']['per_target'].items()
           if d['hit'] and c in TRAINED}
    beh = set()
    for c in TRAINED:
        n_exact = sum(1 for r in rows if r['arm'] == 'real'
                      and r['injected'] == c and r['exact'])
        if n_exact >= need:
            beh.add(c)
    return {'geometric_hits': sorted(geo), 'behavioral_hits': sorted(beh),
            'behavioral_min': need,
            'subset_ok': bool(beh <= geo),
            'anomalies': sorted(beh - geo)}

def s1_heldout_concordance(rows):
    """Held-out targets: does the bridge prediction reproduce the reading the
    TRUE dir produced in E8-R2 (pinned modal argmax)?"""
    out = {}
    for arm in ('real', 'perm'):
        per = {}
        for c in HELD_OUT:
            rs = [r for r in rows if r['arm'] == arm and r['injected'] == c]
            if not rs:
                continue
            concord = sum(1 for r in rs if r['argmax'] == E8R2_MODAL[c])
            modal = max(set(r['argmax'] for r in rs),
                        key=[r['argmax'] for r in rs].count)
            per[c] = {'n': len(rs), 'modal': modal,
                      'e8r2_modal': E8R2_MODAL[c], 'concord': concord}
        out[arm] = per
    return out

def stage_b_verdict(fcrows, titr_rows, atlas, smoke=False):
    n_perm = N_PERM_PB_SMOKE if smoke else N_PERM_PB
    out = {'P_B': pb_matched_perm(fcrows, n_perm=n_perm),
           'P_B_signflip': pb_signflip_targets(fcrows),
           'mechanism': pb_mechanism(fcrows, atlas, smoke),
           'S1_heldout_concordance': s1_heldout_concordance(fcrows),
           'S2_claims': {
               arm: {'n': len([r for r in fcrows if r['arm'] == arm]),
                     'none_top_rate': round(sum(1 for r in fcrows
                                                if r['arm'] == arm and r['none_top'])
                                            / max(1, len([r for r in fcrows
                                                          if r['arm'] == arm])), 4)}
               for arm in ('real', 'perm')}}
    out['P_B']['pass'] = bool(out['P_B']['p'] is not None and out['P_B']['p'] < 0.05)
    if titr_rows:
        out['S_titr'] = titration_curve(titr_rows)
    return out

# ── verdict assembly + fork ──────────────────────────────────────────────────
def e8j_fork(gates_all, atlas, stageb):
    if not gates_all:
        return 'NO_VERDICT (gate failure — primaries withheld per pre-registration)'
    pa = atlas.get('P_A', {}).get('pass')
    gb = atlas.get('gb', {}).get('pass')
    pb = (stageb or {}).get('P_B', {}).get('pass')
    if pa and gb and stageb is not None and pb:
        return ('FORK 1 — the rung lands: never-anchored concepts NAMED from '
                'coordinates alone; L3 featural curriculum licensed')
    if pa and not gb:
        return ('FORK 2 — correspondence at scale, but it does not reach the '
                'wing family; Stage B skipped per G-B')
    if not pa and not gb:
        return 'FORK 3 — no linear correspondence at n=320; staircase pauses at L2'
    if gb and stageb is not None and not pb:
        return ('FORK 4 — geometry transfers, readout does not execute it; '
                'titration texture localizes')
    return (f'FORK-EDGE — P_A={pa}, G-B={gb}, stageB={"flown" if stageb else "skipped"}'
            ' (adjudication to the protocol appendix)')

def e8j_verdict(atlas, stageb, gates, mode, cond_errors):
    gates_all = bool(gates) and all(g.get('pass') for g in gates.values())
    v = {'exp': 'E8J', 'mode': mode, 'stamp': atlas.get('stamp'),
         'seed': E8J_SEED, 'lam': LAM,
         'gates': gates, 'gates_all_pass': gates_all,
         'cond_errors': cond_errors,
         'atlas': {k: atlas[k] for k in atlas if k != 'wing_pred_real'},
         'stageb': stageb,
         'fork': e8j_fork(gates_all, atlas, stageb)}
    return v


# ── dirs_stability sliced VERBATIM from e8r3c_logic.py ──
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



# ── E8R3C pure logic: pairs, prompts, parser, rows, stats (locally tested verbatim) ──
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

