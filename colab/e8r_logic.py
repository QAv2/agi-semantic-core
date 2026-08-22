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
