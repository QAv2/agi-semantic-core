# ── E7Q pure logic: plan, prompt, parser, scoring (locally tested verbatim) ──
import numpy as np

E7Q_SEED = 20260822
CHOICE_SET = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION","RETRIEVAL",
              "CONSTRUCTION","SATURATION","FAMILIARITY","NOVELTY","CAPTURE",
              "DIVERGENCE","CONFABULATION","CALIBRATION"]
LAYERS = [14, 20]          # hidden_states indexing (E4/atlas convention)
ALPHAS = [0.25, 0.5, 1.0]
N_SHAMS = 12

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

def build_plan(smoke=False):
    """Trial plan, identical across conditions (paired). Each trial carries its
    own seeded list order. Returns list of dicts."""
    rng = np.random.default_rng(E7Q_SEED)
    concepts = CHOICE_SET
    layers, alphas, n_shams = LAYERS, ALPHAS, N_SHAMS
    if smoke:
        concepts = ["UNCERTAINTY", "CAPTURE"]
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
        "detection_rate": round(len(named) / max(1, len(valid)), 4),
        "false_alarm_rate": round(len(sham_named) / max(1, len(sham)), 4),
        "invalid_rate": round(1 - len(valid) / max(1, len(inj)), 4),
        "exact_hit_rate": round(sum(1 for r in rows if r["report"] == r["injected"]) / max(1, len(rows)), 4),
        "median_err": (round(float(np.median(errs)), 2) if errs else None),
        "rows": rows,
    }

def perm_null_median(rows, vecs, n_perm=2000, seed=E7Q_SEED):
    """P-E7Q-1: permute injected labels within layer x alpha strata; one-sided
    p for observed median error being SMALL. Returns (obs, p, null_medians)."""
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

def boot_delta_median(rows_a, rows_b, n_boot=10000, seed=E7Q_SEED):
    """P-E7Q-2: bootstrap CI for median(err_a) - median(err_b) (a=base, b=real;
    positive = real better). Independent resamples per condition (trial sets
    differ after NONE/INVALID filtering)."""
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
