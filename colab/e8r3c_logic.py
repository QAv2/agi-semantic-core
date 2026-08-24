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
