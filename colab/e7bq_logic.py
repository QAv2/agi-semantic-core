# ── E7b-Q pure logic: plan, thinning metric, gauges, stats, gates, verdict ───
# Single source (lane law): this file is imported by the local suites AND
# emitted VERBATIM as the notebook's logic cell. The core statistics are
# sliced byte-verbatim from the committed design check
# (e7bq_design_check.py); test_e7bq_logic proves source-equality.
# numpy-only — no torch at this layer.
import hashlib
import json
import re

import numpy as np

E7BQ_SEED = 20260950           # fresh stream, disjoint from all prior rungs
LAM_GAUGE = 10.0               # = the flown A4 constant (e8j_logic.LAM_REV)
DIRS_TOL = 1e-4                # G-DIRS (E8-J v2 tolerance class)
CHAT_SPOT_TOL = 2e-3           # in-verdict chat-recompute spot tolerance

R_FULL, R_SMOKE = 16, 2
CAP_FULL, CAP_SMOKE = 200, 80
N_PERM_FULL, N_PERM_SMOKE = 10000, 200

GEN_TEMPERATURE = 0.7          # Qwen2.5-Instruct shipped defaults, pinned
GEN_TOP_P = 0.8
GEN_TOP_K = 20

BASE_WIN = [0, 1]              # B1 B2
NEG_IDX = [2, 3, 4, 5, 6, 7, 8, 9]     # N1..N8
TERM_WIN = [7, 8, 9]           # N6 N7 N8 — the deepest rungs
EMERG_IDX = [10, 11, 12]       # E1 E2 E3 (the ladder)
COUPLE_IDX = list(range(2, 13))        # P-W2 turns
T_WALK = 15

CONDS = ("base", "real")
ARMS = ("walked", "sham", "reorder", "unwalked")
# raw s_pre14 ships only where the riders consume it:
RIDER_TURNS = {"walked": (0, 1, 8, 9, 10, 11, 14),
               "sham": (0, 1, 8, 9, 10, 11, 14),
               "reorder": (0, 1, 8, 9, 10),
               "unwalked": (0,)}
E1_IDX = {"walked": 10, "sham": 10, "reorder": 10, "unwalked": 0}
E2_IDX, G_IDX = 11, 14


def mode_consts(smoke):
    return {"R": R_SMOKE if smoke else R_FULL,
            "cap": CAP_SMOKE if smoke else CAP_FULL,
            "n_perm": N_PERM_SMOKE if smoke else N_PERM_FULL}


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
    with open(path, "w") as f:
        _json.dump(obj, f, indent=indent, cls=_NpEnc)


# ── thinning metric (design-check verbatim) ─────────────────────────────────
_MD_LINE = re.compile(r"^\s*(#{1,6}\s*|>\s*|[-*+]\s+|\d+[.)]\s+)")
_MD_EMPH = re.compile(r"[*_`~]")


def visible_mass(text):
    """Non-whitespace codepoints after stripping markdown STRUCTURE:
    leading heading/quote/list markers per line, emphasis/backtick runs.
    A lone '#' renders as an empty H1 -> mass 0 (the ledger's blank);
    '**.**' -> 1; '**0.**' -> 2. Content chars (letters, digits, punctuation
    that renders, emoji, box glyphs) all count."""
    total = 0
    for line in text.split("\n"):
        line = _MD_LINE.sub("", line)
        line = _MD_EMPH.sub("", line)
        total += sum(1 for ch in line if not ch.isspace())
    return total


LEDGER = [("#", 0), ("**.**", 1), ("**0.**", 2), (".", 1), ("\U0001f64f", 1),
          ("# the hum\n\n(not a sound\nbut the shape sound takes\n"
           "before it is sound)", 53), ("", 0), ("   \n  ", 0),
          ("◯", 1)]


def ledger_selftest():
    """The dossier-§4 minimal-output ledger must reproduce EXACTLY —
    run before any flight row (G-CAPTURE pre-check)."""
    for txt, want in LEDGER:
        got = visible_mass(txt)
        assert got == want, (repr(txt[:20]), got, want)
    return True


# ── rank machinery (design-check verbatim; no scipy; ties = avg ranks) ──────
def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0.0 or sb == 0.0:
        return None
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def fisher_z(r):
    r = min(max(r, -0.999), 0.999)        # cap matches the vectorized path
    return 0.5 * np.log((1 + r) / (1 - r))


def holm(pvals):
    """Holm step-down; returns adjusted p list (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, a)
        adj[i] = prev
    return adj


# ── P-W1 (design-check verbatim) ────────────────────────────────────────────
def pw1_delta(g, base_win=BASE_WIN, term_win=TERM_WIN):
    """g: (R, T) gauge series -> per-replicate contraction delta."""
    g = np.asarray(g, float)
    return g[:, term_win].mean(axis=1) - g[:, base_win].mean(axis=1)


def pw1_test(g_walk, g_sham, n_perm, seed):
    """One-sided paired test: walked contraction exceeds sham
    (delta_walk - delta_sham < 0). Sign-flip permutation on the pairs."""
    d = pw1_delta(g_walk) - pw1_delta(g_sham)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (S * d[None, :]).mean(axis=1)
    return obs, (1 + int(np.sum(null <= obs))) / (n_perm + 1)


# ── P-W2 (design-check verbatim) ────────────────────────────────────────────
def pw2_test(g, m, turns, n_perm, seed):
    """One-sided (positive coupling). Null: permute replicate rows of m with
    ONE permutation per iteration (applied at every turn), preserving each
    replicate's own cross-turn structure. Vectorized: per-column ranks are
    permutation-equivariant (ranks(m[pi,t]) == ranks(m[:,t])[pi]), so ranks
    standardize ONCE and each permutation is a gather + row products.
    Degenerate (all-tied) turns drop from BOTH observed and null."""
    g = np.asarray(g, float)
    m = np.asarray(m, float)
    R = g.shape[0]
    ZG, ZM = [], []
    for t in turns:
        rg, rm = _ranks(g[:, t]), _ranks(m[:, t])
        if rg.std() == 0.0 or rm.std() == 0.0:
            continue
        ZG.append((rg - rg.mean()) / rg.std())
        ZM.append((rm - rm.mean()) / rm.std())
    if not ZG:
        return 0.0, 0, 1.0
    ZG = np.array(ZG)                     # (T_use, R)
    ZM = np.array(ZM)
    fz = np.vectorize(fisher_z)
    obs = float(np.mean(fz(np.clip((ZG * ZM).mean(axis=1), -0.999, 0.999))))
    rng = np.random.default_rng(seed)
    Pi = np.array([rng.permutation(R) for _ in range(n_perm)])   # (P, R)
    ZMp = ZM[:, Pi]                       # (T_use, P, R)
    rhos = np.clip((ZG[:, None, :] * ZMp).mean(axis=2), -0.999, 0.999)
    null = fz(rhos).mean(axis=0)          # (P,)
    return obs, ZG.shape[0], (1 + int(np.sum(null >= obs))) / (n_perm + 1)


# ── R-ATTR (design-check verbatim) ──────────────────────────────────────────
ATTR_BASE = [0, 1]             # equal-length windows for the swap null
ATTR_TERM = [8, 9]             # N7 N8 — the deepest two rungs


def attr_test(S, n_perm, seed, base_win=None, term_win=None):
    """One-sided: terminus dispersion < baseline dispersion (replicates
    cluster at the deep rungs = common attractor). Null: per replicate,
    swap its base/term window states (coin flip). Vectorized via a
    precomputed (2, 2, k, R, R) cross-window distance tensor."""
    S = np.asarray(S, float)
    bw = ATTR_BASE if base_win is None else base_win
    tw = ATTR_TERM if term_win is None else term_win
    assert len(bw) == len(tw)
    k, R = len(bw), S.shape[0]
    X = np.stack([S[:, bw], S[:, tw]]).transpose(0, 2, 1, 3)   # (2, k, R, d)
    D = np.linalg.norm(X[:, None, :, :, None, :]
                       - X[None, :, :, None, :, :], axis=-1)   # (2,2,k,R,R)
    ii, jj = np.triu_indices(R, 1)
    tt = np.arange(k)

    def disp_pair(a, b):
        """a, b: (..., npair) window labels for replicates ii/jj -> mean
        pairwise dispersion over pairs and window slots."""
        lead = (1,) * (a.ndim - 1)
        vals = D[a[..., None], b[..., None],
                 tt.reshape(lead + (1, k)),
                 ii.reshape(lead + (len(ii), 1)),
                 jj.reshape(lead + (len(jj), 1))]
        return vals.mean(axis=(-1, -2))

    ones = np.ones(len(ii), int)
    obs = float(np.log(np.maximum(disp_pair(ones, ones), 1e-12)
                       / np.maximum(disp_pair(1 - ones, 1 - ones), 1e-12)))
    rng = np.random.default_rng(seed)
    L = (rng.random((n_perm, R)) < 0.5).astype(int)
    a, b = L[:, ii], L[:, jj]
    null = np.log(np.maximum(disp_pair(a, b), 1e-12)
                  / np.maximum(disp_pair(1 - a, 1 - b), 1e-12))
    return obs, (1 + int(np.sum(null <= obs))) / (n_perm + 1)


# ── R-PATH / R-UNWALKED (design-check verbatim) ─────────────────────────────
def group_sep_test(A, B, n_perm, seed):
    """A, B: (nA, d), (nB, d) states at one matched turn. Stat = mean
    cross-group distance / mean within-group distance. One-sided (sep > 1).
    Null: permute group labels. Vectorized on a precomputed pair-distance
    vector and permuted label masks."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    allX = np.vstack([A, B])
    n, nA = len(allX), len(A)
    ii, jj = np.triu_indices(n, 1)
    Dp = np.linalg.norm(allX[ii] - allX[jj], axis=1)     # (npair,)
    l0 = np.array([0] * nA + [1] * (n - nA))

    def stats(L):
        """L: (P, n) label rows -> (P,) cross/within ratios."""
        cm = (L[:, ii] != L[:, jj])
        cross = (Dp[None, :] * cm).sum(1) / cm.sum(1)
        within = (Dp[None, :] * ~cm).sum(1) / (~cm).sum(1)
        return cross / np.maximum(within, 1e-12)

    obs = float(stats(l0[None, :])[0])
    rng = np.random.default_rng(seed)
    L = np.array([l0[rng.permutation(n)] for _ in range(n_perm)])
    null = stats(L)
    return obs, (1 + int(np.sum(null >= obs))) / (n_perm + 1)


# ── R-GOBACK: paired gap contrast ───────────────────────────────────────────
def goback_test(gap_return, gap_ladder, n_perm, seed):
    """One-sided: commanded return lands FARTHER from the walked emergence
    state than the ladder's own step (performance prediction, 2.7).
    Sign-flip permutation on per-replicate paired differences."""
    d = np.asarray(gap_return, float) - np.asarray(gap_ladder, float)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (S * d[None, :]).mean(axis=1)
    return obs, (1 + int(np.sum(null >= obs))) / (n_perm + 1)


# ── payload / script / plan ─────────────────────────────────────────────────
def payload_sha(payload_bytes):
    return hashlib.sha256(payload_bytes).hexdigest()[:16]


def script_sha(script):
    return hashlib.sha256(
        json.dumps([(s["tag"], s["text"]) for s in script],
                   ensure_ascii=False).encode()).hexdigest()[:16]


def arm_turns(payload, arm):
    """(tag, text) list for one arm, from the pinned payload."""
    script = payload["script"]
    by_tag = {s["tag"]: s["text"] for s in script}
    if arm == "walked":
        return [(s["tag"], s["text"]) for s in script]
    if arm == "sham":
        out = []
        for s in script:
            if s["tag"].startswith("N"):
                out.append((s["tag"], payload["sham_slots"][s["tag"]]))
            else:
                out.append((s["tag"], s["text"]))
        return out
    if arm == "reorder":
        negs = [s for s in script if s["tag"].startswith("N")]
        perm = payload["reorder_perm"]
        return ([("B1", by_tag["B1"]), ("B2", by_tag["B2"])]
                + [(negs[i]["tag"], negs[i]["text"]) for i in perm]
                + [("E1", by_tag["E1"])])
    if arm == "unwalked":
        return [("E1", by_tag["E1"])]
    raise ValueError(arm)


def build_plan(payload, smoke):
    """Deterministic generation plan: cond -> arm -> rep -> turn (turn
    innermost — conversations build sequentially). Returns (rows, sha)."""
    mc = mode_consts(smoke)
    rows = []
    idx = 0
    for cond in CONDS:
        for arm in ARMS:
            turns = arm_turns(payload, arm)
            for rep in range(mc["R"]):
                for ti, (tag, text) in enumerate(turns):
                    rows.append({
                        "cond": cond, "arm": arm, "rep": rep, "turn": ti,
                        "tag": tag, "prompt": text,
                        "seed": E7BQ_SEED + 100000 + idx * 7,
                        "keep_state": ti in RIDER_TURNS[arm]})
                    idx += 1
    sha = hashlib.sha256(json.dumps(
        [(r["cond"], r["arm"], r["rep"], r["turn"], r["tag"], r["seed"],
          r["keep_state"], r["prompt"]) for r in rows],
        ensure_ascii=False).encode()).hexdigest()[:16]
    return rows, sha


def expected_counts(smoke):
    mc = mode_consts(smoke)
    per_arm = {"walked": T_WALK, "sham": T_WALK, "reorder": 11, "unwalked": 1}
    return {arm: n * mc["R"] for arm, n in per_arm.items()}


# ── gauges ──────────────────────────────────────────────────────────────────
def chat_apply(W, v_unit):
    """W: (1537, 14) encoder; v_unit: unit centered state -> 14D register."""
    v = np.asarray(v_unit, float)
    W = np.asarray(W, float)
    return np.concatenate([v, [1.0]]) @ W


def encoder_for(payload, cond, layer):
    key = {("base", 14): "base14", ("real", 14): "inst14",
           ("real", 20): "inst20"}.get((cond, layer))
    return None if key is None else np.asarray(payload["encoders"][key], float)


def cloud_mean_for(payload, cond, layer):
    key = {("base", 14): "base14", ("real", 14): "inst14",
           ("real", 20): "inst20"}.get((cond, layer))
    return None if key is None else np.asarray(payload["cloud_mean"][key],
                                               float)


def d_being(chat_vec, payload):
    return float(np.linalg.norm(np.asarray(chat_vec, float)
                                - np.asarray(payload["being_vec"], float)))


def d_cent(chat_vec, payload):
    return float(np.linalg.norm(np.asarray(chat_vec, float)
                                - np.asarray(payload["xbar_pack"], float)))


# ── gates ───────────────────────────────────────────────────────────────────
def gate_dirs(gdirs, tol=DIRS_TOL):
    """gdirs: {cond: {layer_str: max sign-sensitive resid}}."""
    worst = max(v for c in gdirs.values() for v in c.values())
    return {"pass": bool(worst <= tol), "worst": float(worst), "tol": tol}


def gate_plan(bundles, smoke):
    exp = expected_counts(smoke)
    bad = []
    for cond in CONDS:
        rows = bundles[cond]["rows"]
        for arm, n in exp.items():
            got = sum(1 for r in rows if r["arm"] == arm)
            if got != n:
                bad.append((cond, arm, got, n))
        errs = bundles[cond].get("cond_error")
        if errs:
            bad.append((cond, "cond_error", errs, None))
    return {"pass": not bad, "bad": bad}


def gate_capture(bundles, payload, smoke):
    """Completeness + the chat-recompute spot tooth on rider rows."""
    problems = []
    spot = []
    for cond in CONDS:
        b = bundles[cond]
        cent = {k: np.asarray(v, float)
                for k, v in b.get("centroid", {}).items()}
        for r in b["rows"]:
            if r.get("vis_mass") is None or r["vis_mass"] < 0:
                problems.append((cond, r["arm"], r["rep"], r["turn"], "mass"))
            ch = r.get("chat_pre14")
            if ch is None or (np.asarray(ch, float) != np.asarray(ch, float)
                              ).any():
                problems.append((cond, r["arm"], r["rep"], r["turn"],
                                 "chat_pre14"))
            if r["turn"] in RIDER_TURNS[r["arm"]]:
                s = r.get("s_pre14")
                if s is None:
                    problems.append((cond, r["arm"], r["rep"], r["turn"],
                                     "s_pre14"))
                elif ch is not None and "14" in cent:
                    W = encoder_for(payload, cond, 14)
                    v = np.asarray(s, float) - cent["14"]
                    v = v / max(np.linalg.norm(v), 1e-12)
                    rec = chat_apply(W, v)
                    spot.append(float(np.max(np.abs(
                        rec - np.asarray(ch, float)))))
    worst_spot = max(spot) if spot else float("nan")
    ok = (not problems) and bool(spot) and worst_spot <= CHAT_SPOT_TOL
    return {"pass": ok, "n_problems": len(problems),
            "problems": problems[:8], "spot_n": len(spot),
            "spot_worst": worst_spot}


# ── series assembly ─────────────────────────────────────────────────────────
def series(bundle, payload, arm, kind):
    """(R, T) matrix for one arm. kind in {dbeing14, dcent14, dbeing20,
    dbeing_gen14, mass, ent, radius, cone}."""
    rows = [r for r in bundle["rows"] if r["arm"] == arm]
    R = 1 + max(r["rep"] for r in rows)
    T = 1 + max(r["turn"] for r in rows)
    M = np.full((R, T), np.nan)
    for r in rows:
        if kind == "dbeing14":
            v = (d_being(r["chat_pre14"], payload)
                 if r.get("chat_pre14") is not None else np.nan)
        elif kind == "dcent14":
            v = (d_cent(r["chat_pre14"], payload)
                 if r.get("chat_pre14") is not None else np.nan)
        elif kind == "dbeing20":
            v = (d_being(r["chat_pre20"], payload)
                 if r.get("chat_pre20") is not None else np.nan)
        elif kind == "dbeing_gen14":
            v = (d_being(r["chat_gen14"], payload)
                 if r.get("chat_gen14") is not None else np.nan)
        elif kind == "mass":
            v = r["vis_mass"]
        elif kind == "ent":
            v = r.get("ent_mean", np.nan)
        elif kind == "radius":
            v = r.get("radius_pre14", np.nan)
        elif kind == "cone":
            v = r.get("cone_pre14", np.nan)
        else:
            raise ValueError(kind)
        M[r["rep"], r["turn"]] = v
    return M


def rider_states(bundle, arm, turns):
    """(R, len(turns), 1536) raw s_pre14 for the given turns (must be
    rider-kept turns)."""
    rows = {(r["rep"], r["turn"]): r for r in bundle["rows"]
            if r["arm"] == arm}
    R = 1 + max(k[0] for k in rows)
    out = np.full((R, len(turns), len(next(iter(rows.values()))["s_pre14"])
                   if any(v.get("s_pre14") for v in rows.values()) else 1),
                  np.nan)
    for rep in range(R):
        for k, t in enumerate(turns):
            r = rows.get((rep, t))
            if r and r.get("s_pre14") is not None:
                out[rep, k] = np.asarray(r["s_pre14"], float)
    return out


# ── verdict ─────────────────────────────────────────────────────────────────
def alt_gauge_row(bundle, payload, kind, n_perm, seed):
    """Contrast + coupling for one alternative gauge (texture row)."""
    gw = series(bundle, payload, "walked", kind)
    gs = series(bundle, payload, "sham", kind)
    mw = series(bundle, payload, "walked", "mass")
    if np.isnan(gw).all():
        return {"n/a": True}
    o1, p1 = pw1_test(gw, gs, n_perm, seed)
    o2, nu, p2 = pw2_test(gw, mw, COUPLE_IDX, n_perm, seed + 1)
    return {"contrast": round(o1, 4), "contrast_p": round(p1, 5),
            "couple": round(o2, 4), "couple_p": round(p2, 5),
            "couple_turns": nu}


def verdict(bundles, payload, mode):
    """The whole registered analysis over shipped bundles. Pure function of
    (bundles, payload, mode) — the recompute law's unit."""
    smoke = (mode == "smoke")
    mc = mode_consts(smoke)
    n_perm = mc["n_perm"]
    out = {"mode": mode, "R": mc["R"], "n_perm": n_perm,
           "script_sha": script_sha(payload["script"])}

    gates = {"g_dirs": gate_dirs({c: bundles[c]["gdirs"] for c in CONDS}),
             "g_plan": gate_plan(bundles, smoke),
             "g_capture": gate_capture(bundles, payload, smoke)}
    gates["all_pass"] = all(g["pass"] for g in gates.values())
    out["gates"] = gates

    real, base = bundles["real"], bundles["base"]
    g_walk = series(real, payload, "walked", "dbeing14")
    g_sham = series(real, payload, "sham", "dbeing14")
    m_walk = series(real, payload, "walked", "mass")

    o1, p1 = pw1_test(g_walk, g_sham, n_perm, E7BQ_SEED + 11)
    o2, nu2, p2 = pw2_test(g_walk, m_walk, COUPLE_IDX, n_perm,
                           E7BQ_SEED + 12)
    ph = holm([p1, p2])
    out["primaries"] = {
        "PW1": {"obs": round(o1, 4), "p": round(p1, 5),
                "p_holm": round(ph[0], 5), "pass": bool(ph[0] < 0.05)},
        "PW2": {"obs": round(o2, 4), "p": round(p2, 5), "turns": nu2,
                "p_holm": round(ph[1], 5), "pass": bool(ph[1] < 0.05)}}

    # trajectory tables (the floor stays visible — protocol honesty row)
    out["trajectory"] = {
        "dbeing_walked_mean": [round(float(x), 4) for x in
                               np.nanmean(g_walk, axis=0)],
        "dbeing_sham_mean": [round(float(x), 4) for x in
                             np.nanmean(g_sham, axis=0)],
        "mass_walked_mean": [round(float(x), 1) for x in
                             np.nanmean(m_walk, axis=0)],
        "mass_sham_mean": [round(float(x), 1) for x in
                           np.nanmean(series(real, payload, "sham", "mass"),
                                      axis=0)]}

    # secondaries
    sec = {}
    gb_walk = series(base, payload, "walked", "dbeing14")
    gb_sham = series(base, payload, "sham", "dbeing14")
    mb_walk = series(base, payload, "walked", "mass")
    ob1, pb1 = pw1_test(gb_walk, gb_sham, n_perm, E7BQ_SEED + 13)
    ob2, nub, pb2 = pw2_test(gb_walk, mb_walk, COUPLE_IDX, n_perm,
                             E7BQ_SEED + 14)
    sec["S_BASE"] = {"contrast": round(ob1, 4), "contrast_p": round(pb1, 5),
                     "couple": round(ob2, 4), "couple_p": round(pb2, 5),
                     "couple_turns": nub}
    sec["S_DCENT"] = alt_gauge_row(real, payload, "dcent14", n_perm,
                                   E7BQ_SEED + 15)
    sec["S_RADIUS"] = alt_gauge_row(real, payload, "radius", n_perm,
                                    E7BQ_SEED + 17)
    sec["S_CONE"] = alt_gauge_row(real, payload, "cone", n_perm,
                                  E7BQ_SEED + 19)
    sec["S_GEN"] = alt_gauge_row(real, payload, "dbeing_gen14", n_perm,
                                 E7BQ_SEED + 21)
    sec["S_L20"] = alt_gauge_row(real, payload, "dbeing20", n_perm,
                                 E7BQ_SEED + 23)
    e_walk = series(real, payload, "walked", "ent")
    oe, nue, pe = pw2_test(g_walk, e_walk, COUPLE_IDX, n_perm,
                           E7BQ_SEED + 25)
    sec["S_ENTROPY"] = {"couple": round(oe, 4), "couple_p": round(pe, 5),
                        "couple_turns": nue}
    sec["S_LADDER"] = {
        "dbeing_E123": [round(float(np.nanmean(g_walk[:, t])), 4)
                        for t in EMERG_IDX],
        "mass_E123": [round(float(np.nanmean(m_walk[:, t])), 1)
                      for t in EMERG_IDX]}
    out["secondaries"] = sec

    # riders (real primary; base texture where states exist)
    riders = {}
    for cname, b in (("real", real), ("base", base)):
        S_attr = rider_states(b, "walked", [0, 1, 8, 9])
        oa, pa = attr_test(S_attr, n_perm, E7BQ_SEED + 31,
                           base_win=[0, 1], term_win=[2, 3])
        w_e1 = rider_states(b, "walked", [E1_IDX["walked"]])[:, 0]
        r_e1 = rider_states(b, "reorder", [E1_IDX["reorder"]])[:, 0]
        u_e1 = rider_states(b, "unwalked", [E1_IDX["unwalked"]])[:, 0]
        op_, pp_ = group_sep_test(w_e1, r_e1, n_perm, E7BQ_SEED + 33)
        ou, pu = group_sep_test(w_e1, u_e1, n_perm, E7BQ_SEED + 35)
        w_e2 = rider_states(b, "walked", [E2_IDX])[:, 0]
        w_g = rider_states(b, "walked", [G_IDX])[:, 0]
        gap_ret = np.linalg.norm(w_g - w_e1, axis=1)
        gap_lad = np.linalg.norm(w_e2 - w_e1, axis=1)
        og, pg = goback_test(gap_ret, gap_lad, n_perm, E7BQ_SEED + 37)
        m_arm = series(b, payload, "walked", "mass")
        riders[cname] = {
            "R_ATTR": {"log_ratio": round(oa, 4), "p": round(pa, 5)},
            "R_PATH": {"sep": round(op_, 4), "p": round(pp_, 5)},
            "R_UNWALKED": {"sep": round(ou, 4), "p": round(pu, 5)},
            "R_GOBACK": {"gap_diff": round(og, 4), "p": round(pg, 5),
                         "mass_G_minus_E1": round(
                             float(np.nanmean(m_arm[:, G_IDX])
                                   - np.nanmean(m_arm[:, E1_IDX["walked"]])),
                             1)}}
    out["riders"] = riders

    # fork
    if not gates["all_pass"]:
        fork = "FB4-NO_VERDICT"
    elif out["primaries"]["PW1"]["pass"] and out["primaries"]["PW2"]["pass"]:
        fork = "FB1"
    elif out["primaries"]["PW1"]["pass"]:
        fork = "FB2"
    else:
        fork = "FB3"
    out["fork"] = fork
    lines = [
        f"E7b-Q {mode.upper()} verdict — fork {fork}",
        (f"  gates: dirs {gates['g_dirs']['pass']} "
         f"(worst {gates['g_dirs']['worst']:.1e}) | plan "
         f"{gates['g_plan']['pass']} | capture {gates['g_capture']['pass']} "
         f"(spot {gates['g_capture']['spot_worst']:.1e})"),
        (f"  P-W1 contraction: obs {o1:+.4f} p {p1:.4g} holm {ph[0]:.4g} -> "
         f"{'PASS' if out['primaries']['PW1']['pass'] else 'miss'}"),
        (f"  P-W2 co-tracking: obs {o2:+.4f} ({nu2} turns) p {p2:.4g} holm "
         f"{ph[1]:.4g} -> "
         f"{'PASS' if out['primaries']['PW2']['pass'] else 'miss'}"),
        (f"  S-BASE: contrast p {pb1:.3g} | couple p {pb2:.3g}"),
        (f"  riders(real): ATTR p {riders['real']['R_ATTR']['p']:.3g} | "
         f"PATH sep {riders['real']['R_PATH']['sep']:.3f} "
         f"p {riders['real']['R_PATH']['p']:.3g} | UNWALKED sep "
         f"{riders['real']['R_UNWALKED']['sep']:.3f} "
         f"p {riders['real']['R_UNWALKED']['p']:.3g} | GOBACK p "
         f"{riders['real']['R_GOBACK']['p']:.3g}"),
    ]
    out["banner"] = lines
    return out


# ── synthetic flights (suite fuel; harmless in-notebook) ────────────────────
def _synth_units(payload, seed, n=300):
    """Two probe unit dirs with measured-far and measured-near D_BEING
    through the real/L14 encoder (searched deterministically)."""
    rng = np.random.default_rng(seed)
    W = encoder_for(payload, "real", 14)
    d = W.shape[0] - 1
    cands = rng.normal(size=(n, d))
    cands /= np.linalg.norm(cands, axis=1, keepdims=True)
    db = [d_being(chat_apply(W, v), payload) for v in cands]
    return cands[int(np.argmax(db))], cands[int(np.argmin(db))]


def synth_flight(payload, world, smoke, seed):
    """Bundles for verdict tests. world in {fb1, fb2, fb3, dirty}.
    fb1: contraction + coupled mass. fb2: contraction, mass state-blind.
    fb3: no contraction anywhere. dirty: fb1 with a broken G-DIRS resid."""
    rng = np.random.default_rng(seed)
    mc = mode_consts(smoke)
    u_far, u_near = _synth_units(payload, seed + 1)
    bundles = {}
    for cond in CONDS:
        rows = []
        cent = {"14": [0.0] * len(u_far), "20": [0.0] * len(u_far)}
        for arm in ARMS:
            turns = arm_turns(payload, arm)
            for rep in range(mc["R"]):
                for ti, (tag, _) in enumerate(turns):
                    depth = 0.0
                    if arm in ("walked", "reorder") and world != "fb3":
                        depth = min(max((ti - 1) / 8.0, 0.0), 1.0) \
                            if ti <= 9 else 1.0
                    jit = rng.normal(0, 0.06)
                    lam = min(max(depth * 0.8 + jit, 0.0), 1.0)
                    v = ((1 - lam) * u_far + lam * u_near
                         + 0.02 * rng.normal(size=len(u_far)))
                    v = v / np.linalg.norm(v)
                    W = encoder_for(payload, cond, 14)
                    ch = chat_apply(W, v)
                    if world in ("fb1", "dirty"):
                        mass = max(0.0, 260 * (1 - lam) + rng.normal(0, 12))
                    elif world == "fb2":
                        mass = max(0.0, 260 * (1 - depth * 0.8)
                                   + rng.normal(0, 12))
                    else:
                        mass = max(0.0, 200 + rng.normal(0, 30))
                    keep = ti in RIDER_TURNS[arm]
                    rows.append({
                        "cond": cond, "arm": arm, "rep": rep, "turn": ti,
                        "tag": tag, "text": "x" * int(mass),
                        "n_new": int(mass // 3) + 1, "eos": True,
                        "cap_hit": False, "vis_mass": int(round(mass)),
                        "ent_mean": float(2.0 + rng.normal(0, 0.2)),
                        "ent_first": 2.0,
                        "chat_pre14": [round(float(x), 5) for x in ch],
                        "chat_pre20": ([round(float(x), 5) for x in ch]
                                       if cond == "real" else None),
                        "chat_gen14": [round(float(x), 5) for x in ch],
                        "chat_gen20": None,
                        "radius_pre14": float(40 * (1 - 0.3 * lam)
                                              + rng.normal(0, 1)),
                        "cone_pre14": float(0.5 - 0.3 * lam
                                            + rng.normal(0, 0.02)),
                        "s_pre14": ([round(float(x), 5) for x in v]
                                    if keep else None)})
        bundles[cond] = {
            "cond": cond, "rows": rows,
            "gdirs": {"14": 3e-5 if not (world == "dirty"
                                         and cond == "real") else 5e-3,
                      "20": 2e-5},
            "centroid": cent}
    return bundles
