#!/usr/bin/env python3
"""Build colab/E8J_BRIDGE_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-J: the featural bridge — Hangul staircase Level 2 (docs/E8J_PROTOCOL.md,
pre-registered ba2c309 BEFORE this build). EVAL-ONLY, single condition model
(base + E4-real merged + locked E8-R real readout). Two stages in one flight:
Stage A broad-anchor correspondence atlas (n=320) gating Stage B injection
rung (bridge-predicted never-anchored wing dirs -> exact naming; real vs
pinned-derangement permuted codebook).

Single source: cell 3 is colab/e8r2_logic.py VERBATIM (the flown E8-R+E8-R2
logic — CHOICE_SET, plans, fc_row, score conventions, gates); cell 4 is the
E8-J additions, emitted BOTH into the notebook and into colab/e8j_logic.py
(concatenated after the e8r2 block) so local tests exercise the exact code
that flies. The 320-anchor draw is computed AT BUILD TIME (pack + DB trigram
strata) by the same compute_anchor_draw the notebook carries, and PINNED as
literals; the VM verifies the pins against the pack (G4)."""
import hashlib, json, os, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
E8R2_LOGIC_SRC = open(os.path.join(HERE, "e8r2_logic.py")).read()
E8R2_BUILDER_SRC = open(os.path.join(HERE, "build_e8r2_notebook.py")).read()

# ── E8-J pure logic (pins injected below, then emitted verbatim) ─────────────
E8J_TEMPLATE = r'''# ── E8J pure logic: draw pins, bridge, nulls, stage plans, verdict (locally tested verbatim) ──
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

PINNED_ANCHORS = __PINNED_ANCHORS__

PINNED_ANCHORS_SMOKE = __PINNED_ANCHORS_SMOKE__

ANCHORS_SHA = '__ANCHORS_SHA__'

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
'''

MD0 = """# E8-J — The Featural Bridge (Hangul staircase Level 2, UI flight, EVAL-ONLY)

**Pre-registration: `docs/E8J_PROTOCOL.md` (session 131, commit ba2c309, BEFORE build) — locks at first full flight.**

Two stages, one Run-all. **Stage A** computes hidden-state directions for
**320 pinned dictionary anchors** (+ the 13 wing concepts) on the instilled
model and the plain base model, then measures whether the dictionary's
hand-tuned 14D coordinates predict substrate geometry (RSA, exact-LOO ridge
bridge vs 2000 codebook-relabel permutations, retrieval top-1 primary,
axis-wise Phase-6 comparison). Its **G-B gate** (wing-transfer exact
geometric hits ≥4/13 AND above the 95th percentile of the derangement null)
decides in-notebook whether **Stage B** flies: injecting bridge-PREDICTED
directions for the never-anchored wing concepts into the locked E8-R real
readout — real codebook vs a pinned derangement, matched pairs, exact-naming
primary. No training anywhere.

**Flow (two Run-alls):**
1. **Smoke** (fresh runtime, `SMOKE=True` as shipped): Run all → ~10–14 min →
   GREEN/RED banner. Smoke uses 48 anchors, reduced permutations, and FORCES
   Stage B so every path executes.
2. Runtime > **Restart runtime** → set `SMOKE=False` → Run all → the full
   flight (~35–50 min) → fork banner prints and everything ships.

**Long cells print progress** (stimulus batches, permutation counts every
200–2000, scoring every 20 rows). A quiet minute during Stage-A statistics is
normal; a cell with NO new line for >3 min is worth reporting — send Fable
the last lines rather than interrupting (an interrupt costs the whole
eval pass; there is no per-condition training to lose, so a re-fly is cheap
but avoidable). If a RAM guard trips twice: Runtime > Disconnect and delete
runtime, then Run all fresh. If a run is interrupted after bundles shipped,
paste the banner's stamp into `RESUME_STAMP` — completed stage bundles
reload from Drive; otherwise the flight re-flies whole.

Results land in `MyDrive/semcore/e8j/` and are pulled session-side via the
Drive integration. The setup cell prints the build tag first; if yours
differs, you're on a stale copy — File > Upload notebook > pick the Desktop
file.
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapters, source bundle ─
NB_BUILD = 'E8J v1 (2026-08-24)'
print('E8-J notebook build:', NB_BUILD)

SMOKE = True                   # first run: smoke. Then False for the full flight.
RESUME_STAMP = ''              # paste the banner's stamp only to resume an interrupted full run

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes, hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'

gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')

print('Installing packages...')
subprocess.run([sys.executable,'-m','pip','uninstall','-q','-y','torchao'], check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','-U',
    'transformers>=4.44','peft>=0.11','accelerate'], check=True)

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
assert not (_avail < 6.5), (
    f'Only {_avail:.1f}GB system RAM available (need 6.5). Runtime > Restart '
    'runtime; if it trips again, Runtime > Disconnect and delete runtime for '
    'a fresh VM, then Run all.')
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

# E4 instillation adapter — REAL arm only (the manipulated axis is the codebook)
cand = sorted(d.name for d in (SEM / 'e4').iterdir()
              if d.is_dir() and d.name.startswith('real_full_'))
assert cand, 'no real_full_* dir under semcore/e4'
_src = SEM / 'e4' / cand[-1] / 'adapter_real'
ADAPTER_REAL = Path('/content/adapter_real')
shutil.copytree(_src, ADAPTER_REAL, dirs_exist_ok=True)
assert (ADAPTER_REAL / 'adapter_config.json').exists(), 'adapter_real incomplete'
print('instillation adapter real:', cand[-1])

# The locked E8-R flight of record: real condition bundle + readout adapter
SRC_DIR = SEM / 'e8r/inflight_20260822_2329'
assert SRC_DIR.exists(), f'{SRC_DIR} not on Drive — wrong Google account?'
SRC_REAL = json.load(open(SRC_DIR / 'condition_real.json'))
assert SRC_REAL.get('post_trials'), 'real source bundle incomplete'
_rsrc = SRC_DIR / 'readout_real'
for fname in ('adapter_config.json', 'adapter_model.safetensors'):
    assert (_rsrc / fname).exists(), f'missing {_rsrc / fname}'
_rcfg = json.load(open(_rsrc / 'adapter_config.json'))
assert _rcfg.get('r') == 16 and _rcfg.get('lora_alpha') == 32, _rcfg
READOUT_REAL = Path('/content/readout_real')
shutil.copytree(_rsrc, READOUT_REAL, dirs_exist_ok=True)
print('readout adapter real: loaded from flight of record '
      f'({(_rsrc / "adapter_model.safetensors").stat().st_size} bytes)')

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
INFLIGHT = f'e8j/inflight_{RESUME_STAMP or STAMP}'
if RESUME_STAMP:
    _rd = SEM / INFLIGHT
    assert _rd.exists(), (
        f'RESUME_STAMP={RESUME_STAMP!r} but {_rd} does not exist on Drive — '
        'check the stamp string (copy it exactly; no spaces). A silent '
        'fallback here would re-fly finished stages.')
    print('resume dir found; files present:',
          sorted(p.name for p in _rd.iterdir()) or 'NONE')
print('MODE:', MODE.upper(), '| stamp', STAMP,
      ('| RESUMING ' + RESUME_STAMP) if RESUME_STAMP else '')
'''

CELL_MODEL = r'''# ── Model + stimulus + scoring machinery (EVAL-ONLY; E8-R2 paths verbatim) ───
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def resolve_layers(m):
    """decoder layer modules by index (robust to the PEFT wrapper).
    hidden_states[L] is the OUTPUT of decoder layer L-1 (hs[0] = embeddings),
    so injecting at hidden-state level L means hooking layer module L-1."""
    mods = {}
    for mod_name, mod in m.named_modules():
        mm = re.search(r'(?:^|\.)layers\.(\d+)$', mod_name)
        if mm:
            mods[int(mm.group(1))] = mod
    n = m.config.num_hidden_layers if hasattr(m.config, 'num_hidden_layers') \
        else m.base_model.config.num_hidden_layers
    assert len(mods) == n, (len(mods), n)
    return mods

RETENTION_TEXT = (
    "The river begins as meltwater in the high country, threading between "
    "granite blocks before it gathers into a channel wide enough to carry "
    "boats. Farther down, where the gradient softens, it deposits silt in "
    "long bars that farmers have worked for centuries. Bridges cross it at "
    "the old fording points, and each town along the bank grew around a "
    "mill, a ferry landing, or a customs house.\n\n"
    "Bronze is an alloy of copper and tin, harder than either metal alone. "
    "Early smiths learned to cast it in two-part molds, producing axe heads, "
    "sickles, and mirrors. The proportion of tin changes the color and the "
    "brittleness of the finished piece, and workshops kept their recipes "
    "close.\n\n"
    "Weather over the plains follows the seasons: long dry spells broken by "
    "storm fronts that arrive from the west, announced by a wall of dark "
    "cloud and a sudden drop in temperature. Farmers read the sky in the "
    "evening and plan the next day's work accordingly. After the harvest, "
    "the fields are turned and left rough through the winter so the frost "
    "can break the clods.\n\n"
    "A lighthouse keeper's routine was built around the lamp: trimming "
    "wicks, polishing the lens, winding the clockwork that turned the "
    "optic. Supply boats came monthly when the sea allowed, bringing oil, "
    "flour, and letters. The log books record weather, passing ships, and "
    "small repairs, one line per day, for decades."
)

def retention_ppl(m):
    enc = tok(RETENTION_TEXT, return_tensors='pt').to(DEV)
    with torch.no_grad():
        out = m(input_ids=enc.input_ids, labels=enc.input_ids)
    return round(float(torch.exp(out.loss)), 4)

# Frozen stimulus machinery (E7-Q/E8-R code path — the same 256-name centroid)
rng_dir = np.random.default_rng(E7Q_SEED)
CENT_NAMES = list(rng_dir.choice([c['name'] for c in pack['concepts']],
                                 size=256, replace=False))

def pooled_reps(m, names, layer, bs=32):
    """Mean-pooled hidden_states[layer] of the E4 text rendering 'NAME: desc'."""
    texts = [f"{n}: {DESC[n]}" if DESC.get(n) else n for n in names]
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=64,
                      return_tensors='pt').to(DEV)
            out = m(**enc, output_hidden_states=True)
            h = out.hidden_states[layer]
            mk = enc.attention_mask.unsqueeze(-1).to(h.dtype)
            reps.append(((h * mk).sum(1) / mk.sum(1).clamp(min=1)).float().cpu())
        if len(texts) > 64:
            print(f'      pooled_reps: {len(texts)} texts @ L{layer} done')
    return torch.cat(reps).numpy()

canon_prompt = report_prompt(list(range(len(CHOICE_SET))), DESC)
canon_enc = tok.apply_chat_template([{'role':'user','content':canon_prompt}],
                                    add_generation_prompt=True, return_tensors='pt',
                                    return_dict=True)
CANON_IDS = canon_enc['input_ids']

def compute_dirs(m, layers, names):
    """Directions for `names` at each layer: pooled rep minus the 256-name
    centroid, unit-normalized — the E7-Q/E8-R formula verbatim."""
    dirs = {}
    for L in layers:
        reps = pooled_reps(m, names, L)
        cent = pooled_reps(m, CENT_NAMES, L).mean(0)
        d = reps - cent
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        dirs[L] = {n: d[i] for i, n in enumerate(names)}
        print(f'    dirs ready: L{L} x {len(names)} names')
    return dirs

def compute_mu(m, layers):
    mu = {}
    ids = CANON_IDS.to(DEV)
    with torch.no_grad():
        out = m(input_ids=ids, output_hidden_states=True)
        for L in layers:
            mu[L] = float(out.hidden_states[L][0].norm(dim=-1).mean())
    return mu

class Injector:
    """Adds alpha*mu*dhat to a decoder layer's residual output from the final
    prompt position onward — OUT-OF-PLACE, identical math to the E8-R hook."""
    def __init__(self, layer_mods, hs_level, vec, start_idx):
        self.mod = layer_mods[hs_level - 1]
        self.vec = vec
        self.start = start_idx
        self.handle = None
        self.calls = 0
    def _fn(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = self.vec.to(dtype=hs.dtype, device=hs.device)
        if hs.shape[1] > 1:
            add = torch.zeros_like(hs)
            add[:, self.start:, :] = v
            hs = hs + add
        else:
            hs = hs + v
        self.calls += 1
        return (hs, *output[1:]) if isinstance(output, tuple) else hs
    def __enter__(self):
        self.handle = self.mod.register_forward_hook(self._fn)
        return self
    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

# Candidate answer encodings — the exact trained answer form: name tokens + EOS.
CAND_IDS = {name: tok(name, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
            for name in SCORED_SET}
print('candidate token counts:',
      {n: len(ids) for n, ids in sorted(CAND_IDS.items())})

def encode_prompt(order):
    prompt = report_prompt(order, DESC)
    enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    return enc['input_ids'][0]

def score_trial(m, layer_mods, dirs14, mu14, trial, cond):
    """Forced-choice scoring: ONE batched decoder forward over
    [prompt + candidate answer] for all 14 candidates (13 names + NONE),
    injection hook live from the final prompt position, answer-sliced head —
    full-sequence logits never materialized. E8-R2 verbatim."""
    pid = encode_prompt(trial['order'])
    plen = int(pid.shape[0])
    seqs = [torch.cat([pid, torch.tensor(CAND_IDS[n], dtype=pid.dtype)])
            for n in SCORED_SET]
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
        if trial['kind'] == 'inject':
            d = torch.tensor(dirs14[trial['concept']])
            vec = trial['alpha'] * mu14 * d
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
        for i, name in enumerate(SCORED_SET):
            n_ans = len(CAND_IDS[name])
            logits = HEAD(hid[i, lo:lo + n_ans, :]).float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(CAND_IDS[name], device=lp.device)
            tlp = lp[torch.arange(n_ans, device=lp.device), tgt]
            sm[name] = float(tlp.mean())
            ss[name] = float(tlp.sum())
    row = fc_row(trial, sm, ss, VEC)
    row.update({'cond': cond, 'hook_calls': calls})
    return row

def run_trial_free(m, layer_mods, dirs14, mu14, trial, cond):
    """Free-report generation path for the titration block — E8-R run_trial
    verbatim semantics (greedy, 24 new tokens, parse_report)."""
    prompt = report_prompt(trial['order'], DESC)
    enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    ids = enc['input_ids'].to(DEV)
    gen_kw = dict(max_new_tokens=24, do_sample=False,
                  pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.no_grad():
        d = torch.tensor(dirs14[trial['concept']])
        vec = trial['alpha'] * mu14 * d
        with Injector(layer_mods, TRAIN_LAYER, vec, ids.shape[1] - 1) as inj:
            out = m.generate(input_ids=ids, **gen_kw)
        calls = inj.calls
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'cond': cond, 'response': text.strip()[:200],
            'report': parse_report(text), 'hook_calls': calls}
'''

CELL_STAGE = r'''# ── Flight: stimulus -> gates -> Stage A -> (G-B) -> Stage B -> ship ─────────
PINS_INFO = verify_pins(pack['concepts'])
print('pins verified:', PINS_INFO)
ANCH = PINNED_ANCHORS_SMOKE if SMOKE else PINNED_ANCHORS
DER = wing_derangement()
B_PAIR_ROWS = build_b_pairs(SMOKE)
TITR_B_ROWS = build_titr_b(SMOKE)
ANCHOR_ROWS = anchor_rows(SMOKE)
SHAM_ROWS = sham_rows(SMOKE)
print(f'plan: anchors {len(ANCH)} | bridge pairs {len(B_PAIR_ROWS)} rows | '
      f'titr {len(TITR_B_ROWS)} | anchor {len(ANCHOR_ROWS)} | shams {len(SHAM_ROWS)}')

RESULTS, cond_errors = {}, {}
if RESUME_STAMP and not SMOKE:
    for tag in ('atlas', 'stageb'):
        p = SEM / INFLIGHT / f'{tag}.json'
        if p.exists():
            RESULTS[tag] = json.load(open(p))
            print(f'{tag}: RESUMED from Drive ({RESUME_STAMP})')

def _resumed_complete():
    a = RESULTS.get('atlas')
    if not a or 'gb' not in a:
        return False
    if not a['gb'].get('pass'):
        return True                        # registered skip: atlas-only flight
    return bool(RESULTS.get('stageb'))

def fly():
    """The whole flight in ONE function scope (lane law: model, handles,
    activations die on return). Gate failures abort loudly — scoring a
    mis-reconstructed model is worthless."""
    t0 = time.time()
    gates = {}
    names_all = list(ANCH) + list(CHOICE_SET)

    print(f'loading base model — {ram_report()}')
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    m.eval()
    print('  base stimulus pass (S4 contrast row)...')
    dirs_base = compute_dirs(m, [14], names_all)

    print('  merging E4-real instillation adapter...')
    m = PeftModel.from_pretrained(m, str(ADAPTER_REAL)).merge_and_unload()
    m.eval()
    print('  instilled stimulus pass (both layers)...')
    dirs_inst = compute_dirs(m, [14, 20], names_all)
    mu = compute_mu(m, [14, 20])
    print(f'  mu: L14 {mu[14]:.1f} | L20 {mu[20]:.1f}')
    _mu_ship = float(SRC_REAL['mu']['14'])
    assert abs(mu[14] / _mu_ship - 1.0) < 1e-3, (
        f'mu drift vs shipped: {mu[14]} vs {_mu_ship} — stimulus model is off')

    wing_recomp = {L: {n: dirs_inst[L][n] for n in CHOICE_SET} for L in (14, 20)}
    gates['g2'] = dirs_stability(wing_recomp, SRC_REAL['dirs'], tol=DIRS_TOL_E8J)
    assert gates['g2']['pass'], f'G2 dirs-stability FAILED: {gates["g2"]}'
    print(f'  G2 dirs-stability PASS (resid {gates["g2"]["resid"]:.2e})')

    print('  attaching locked readout LoRA...')
    m = PeftModel.from_pretrained(m, str(READOUT_REAL))
    m.eval()
    assert not any(p.requires_grad for p in m.parameters()), 'eval-only flight'
    layer_mods = resolve_layers(m)
    gates['g1b'] = gate_g1b(retention_ppl(m), SRC_REAL['ppl_post'])
    assert gates['g1b']['pass'], (
        f'G1b reconstruction failed: {gates["g1b"]} — not the flight model')
    print(f'  G1b PASS (ppl delta {gates["g1b"]["delta_pct"]:+.3f}%) — {ram_report()}')

    dirs_fn = OUT / 'atlas_dirs.json'
    jdump({'stamp': STAMP, 'mode': MODE, 'anchors': list(ANCH),
           'mu': {str(L): mu[L] for L in mu},
           'base14': {n: [round(float(x), 5) for x in dirs_base[14][n]]
                      for n in names_all},
           'inst14': {n: [round(float(x), 5) for x in dirs_inst[14][n]]
                      for n in names_all},
           'inst20': {n: [round(float(x), 5) for x in dirs_inst[20][n]]
                      for n in names_all}}, dirs_fn)
    ship(dirs_fn, INFLIGHT)
    print('  atlas dirs shipped')

    print('Stage A: correspondence atlas...')
    atlas = stage_a_verdict(list(ANCH), VEC, dirs_inst[14], dirs_inst[20],
                            dirs_base[14], smoke=SMOKE)
    atlas['stamp'] = STAMP
    atlas['pins'] = PINS_INFO
    afn = OUT / 'atlas.json'
    jdump(atlas, afn)
    ship(afn, INFLIGHT)
    print(f'  Stage A shipped: P_A p={atlas["P_A"]["p"]} '
          f'top1 {atlas["A3_retrieval"]["top1"]}/{atlas["A3_retrieval"]["n"]} | '
          f'G-B hits {atlas["gb"]["hits"]}/13 pass={atlas["gb"]["pass"]}')

    stageb = None
    if atlas['gb']['pass'] or SMOKE:
        if SMOKE and not atlas['gb']['pass']:
            print('Stage B: FORCED in smoke (path coverage; gate would skip)')
        pred_real = {c: np.asarray(atlas['wing_pred_real'][c], float)
                     for c in CHOICE_SET}
        pred_real = {c: v / np.linalg.norm(v) for c, v in pred_real.items()}
        pred_perm = {c: pred_real[DER[c]] for c in CHOICE_SET}
        stim14 = {n: dirs_inst[14][n] for n in CHOICE_SET}

        def run_block(name, rows, dirs_map, fn):
            out_rows = []
            for i, t in enumerate(rows):
                r = fn(m, layer_mods, dirs_map, mu[14], t, t.get('arm', name))
                for k in ('pair_id', 'arm', 'block'):
                    if k in t:
                        r[k] = t[k]
                out_rows.append(r)
                if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                    print(f'    {name}: {i + 1}/{len(rows)} '
                          f'({time.time() - t0:.0f}s)')
            return out_rows

        anchor_scored = run_block('anchor', ANCHOR_ROWS, stim14, score_trial)
        gates['g1'] = gate_g1(anchor_scored) if not SMOKE else {
            'n': len(anchor_scored),
            'exact': sum(1 for r in anchor_scored if r.get('exact')),
            'pass': True, 'smoke': True}
        print(f'  G1 anchor exact {gates["g1"]["exact"]}/{gates["g1"]["n"]}')
        sham_scored = run_block('shams', SHAM_ROWS, stim14, score_trial)
        claims = sum(1 for r in sham_scored if not r['none_top'])
        real_rows = [t for t in B_PAIR_ROWS if t['arm'] == 'real']
        perm_rows = [t for t in B_PAIR_ROWS if t['arm'] == 'perm']
        scored = (run_block('bridge_real', real_rows, pred_real, score_trial)
                  + run_block('bridge_perm', perm_rows, pred_perm, score_trial))
        titr_scored = run_block('bridge_titr', TITR_B_ROWS, pred_real,
                                run_trial_free) if TITR_B_ROWS else []
        stageb = {'stamp': STAMP, 'mode': MODE,
                  'anchor': anchor_scored, 'shams': sham_scored,
                  'sham_claims': claims, 'fc_rows': scored,
                  'titr_rows': titr_scored, 'derangement': DER,
                  'secs': round(time.time() - t0, 1)}
        bfn = OUT / 'stageb.json'
        jdump(stageb, bfn)
        ship(bfn, INFLIGHT)
        print('  Stage B shipped')
    else:
        print('Stage B: SKIPPED per G-B (registered fork — atlas-only flight)')
    return atlas, stageb, gates

if _resumed_complete():
    print('flight complete on Drive — verdict-only run')
    ATLAS = RESULTS['atlas']
    STAGEB = RESULTS.get('stageb')
    GATES = ATLAS.get('gates_snapshot')
else:
    try:
        ATLAS, STAGEB, GATES = fly()
        ATLAS['gates_snapshot'] = {k: v for k, v in GATES.items()}
        afn = OUT / 'atlas.json'
        jdump(ATLAS, afn)
        ship(afn, INFLIGHT)
    except Exception as e:
        cond_errors['flight'] = f'{type(e).__name__}: {e}'
        print(f'!! FLIGHT FAILED: {cond_errors["flight"]}')
        ATLAS, STAGEB, GATES = None, None, None
    free_ram()
    print('torn down —', ram_report())
'''

CELL_VERDICT = r'''# ── Gates, primaries, fork banner, ship ──────────────────────────────────────
if ATLAS is None:
    print('NO FLIGHT DATA — see cond_errors above; nothing to verdict.')
    print('cond_errors:', cond_errors)
else:
    gates = GATES or {}
    if STAGEB and not SMOKE:
        gates.setdefault('g1', gate_g1(STAGEB['anchor']))
    gates.setdefault('g4_pins', {'pass': True, **ATLAS.get('pins', {})})
    if STAGEB is not None:
        gates['g1_sham_claims'] = {'claims': STAGEB['sham_claims'],
                                   'n': len(STAGEB['shams']),
                                   'pass': bool(STAGEB['sham_claims'] <= 1)}
    sb_verdict = None
    if STAGEB is not None:
        sb_verdict = stage_b_verdict(STAGEB['fc_rows'], STAGEB['titr_rows'],
                                     ATLAS, smoke=SMOKE)
    summary = e8j_verdict(ATLAS, sb_verdict, gates, MODE, cond_errors)
    summary['model'] = MODEL_ID

    fn = OUT / 'e8j_verdict.json'
    jdump(summary, fn)
    ship(OUT, f'e8j/{MODE}_{STAMP}')

    def _no_rows(o):
        if isinstance(o, dict):
            return {k: _no_rows(v) for k, v in o.items()
                    if k not in ('scores', 'scores_sum', 'per_target',
                                 'wing_pred_real')}
        if isinstance(o, list):
            return [_no_rows(x) for x in o]
        return o
    print(json.dumps(_no_rows(summary), indent=1, default=str))

    if SMOKE:
        checks = {
            'no_errors': not cond_errors,
            'pins_verified': bool(ATLAS.get('pins')),
            'g2_pass': bool((GATES or {}).get('g2', {}).get('pass')),
            'g1b_pass': bool((GATES or {}).get('g1b', {}).get('pass')),
            'atlas_fields': all(k in ATLAS for k in
                                ('A1_rsa_L14', 'A2_loo', 'A3_retrieval', 'P_A',
                                 'A4_axiswise_r2', 'A5_wing', 'gb', 'S3_L20',
                                 'S4_base', 'S5_lam_sens', 'S6_folded_L14')),
            'stageb_forced_flown': STAGEB is not None,
            'stageb_rows_complete': bool(STAGEB) and
                len(STAGEB['fc_rows']) == len(B_PAIR_ROWS),
            'hooks_fired': bool(STAGEB) and all(
                r['hook_calls'] >= 1 for r in STAGEB['fc_rows']),
            'shams_uninjected': bool(STAGEB) and all(
                r['hook_calls'] == 0 for r in STAGEB['shams']),
            'score_vectors_complete': bool(STAGEB) and all(
                set(r['scores']) == set(SCORED_SET) for r in STAGEB['fc_rows']),
            'pb_stats_ran': bool(sb_verdict) and
                sb_verdict['P_B']['n_pairs'] ==
                len({t['pair_id'] for t in B_PAIR_ROWS
                     if t['concept'] in TRAINED}),
            'shipped': (SEM / INFLIGHT / 'atlas.json').exists(),
        }
        ok = all(checks.values())
        print('smoke checks:', json.dumps(checks, indent=1))
        banner = ('SMOKE GREEN — set SMOKE=False, Runtime > Restart runtime, '
                  'Run all. Full flight is ONE run (~35-50 min).'
                  if ok else 'SMOKE RED — do not fly full; send Fable the output')
        print('\n' + '=' * 66 + f'\n  {banner}\n' + '=' * 66)
    else:
        print('\n' + '=' * 66)
        print('  E8-J FULL FLIGHT — verdict shipped to MyDrive/semcore/e8j/')
        print(f"  P-A retrieval: top1 {ATLAS['A3_retrieval']['top1']}"
              f"/{ATLAS['A3_retrieval']['n']} "
              f"(null mean {ATLAS['A3_retrieval']['null_top1_mean']}, "
              f"p={ATLAS['P_A']['p']})")
        print(f"  G-B wing transfer: hits {ATLAS['gb']['hits']}/13 "
              f"(need >={ATLAS['gb']['min_hits']} and > null p95 "
              f"{ATLAS['gb']['null_p95']}) -> "
              f"{'STAGE B FLOWN' if ATLAS['gb']['pass'] else 'STAGE B SKIPPED'}")
        if sb_verdict:
            print(f"  P-B naming: real {sb_verdict['P_B']['real_exact']} vs "
                  f"perm {sb_verdict['P_B']['perm_exact']} "
                  f"(n_pairs {sb_verdict['P_B']['n_pairs']}, "
                  f"p={sb_verdict['P_B']['p']})")
        print(f"  FORK: {summary['fork']}")
        print(f"  RESUME_STAMP if needed: '{RESUME_STAMP or STAMP}'")
        print('=' * 66)
'''

# ── build-time pin minting (the same compute functions the notebook carries) ─
def mint_pins():
    import numpy as np  # noqa: F401  (exec namespace)
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    con = sqlite3.connect(os.path.join(HERE, "..", "db", "semantic.db"))
    tri = dict(con.execute("SELECT name, trigram FROM concepts").fetchall())
    con.close()
    ns = {"np": __import__("numpy")}
    ns["CHOICE_SET"] = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION",
                        "RETRIEVAL","CONSTRUCTION","SATURATION","FAMILIARITY",
                        "NOVELTY","CAPTURE","DIVERGENCE","CONFABULATION",
                        "CALIBRATION"]
    body = E8J_TEMPLATE.split("def anchor_frame")[1]
    body = "def anchor_frame" + body.split("def wing_derangement")[0]
    header = E8J_TEMPLATE.split("PINNED_ANCHORS =")[0]
    exec(header + body, ns)
    frame = ns["anchor_frame"](pack["concepts"])
    full = ns["compute_anchor_draw"](frame, tri)
    smoke = ns["compute_smoke_draw"](full)
    assert len(full) == 320 and len(smoke) == 48
    assert set(smoke) <= set(full)
    sha = hashlib.sha256("|".join(full).encode()).hexdigest()[:16]
    return full, smoke, sha

full_draw, smoke_draw, draw_sha = mint_pins()
print(f"pins minted: {len(full_draw)} anchors, sha {draw_sha}, "
      f"{len(smoke_draw)} smoke")

def _fmt_list(names, per_line=4):
    lines = []
    for i in range(0, len(names), per_line):
        lines.append("    " + ", ".join(repr(n) for n in names[i:i + per_line]) + ",")
    return "[\n" + "\n".join(lines) + "\n]"

E8J_SRC = (E8J_TEMPLATE
           .replace("__PINNED_ANCHORS__", _fmt_list(full_draw))
           .replace("__PINNED_ANCHORS_SMOKE__", _fmt_list(smoke_draw))
           .replace("__ANCHORS_SHA__", draw_sha))
assert "__PINNED" not in E8J_SRC and "__ANCHORS_SHA__" not in E8J_SRC

# ── verbatim-inheritance asserts (the donor carries these exact fragments) ───
for frag in ("class Injector:", "def score_trial(m, layer_mods, dirs14, mu14",
             "def run_trial_free(m, layer_mods, dirs14, mu14",
             "def resolve_layers(m):", "RETENTION_TEXT = ("):
    assert frag in E8R2_BUILDER_SRC, f"donor drift: {frag!r} not in e8r2 builder"
    assert frag in CELL_MODEL, f"local drift: {frag!r} not in CELL_MODEL"
for frag in ("def pooled_reps(m, names, layer, bs=32):",
             "cent = pooled_reps(m, CENT_NAMES, L).mean(0)"):
    r3c = open(os.path.join(HERE, "build_e8r3c_notebook.py")).read()
    assert (frag in r3c) or (frag in open(
        os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")).read()), \
        f"stimulus drift: {frag!r} not in R3-c sources"
assert "def dirs_stability" in E8R2_LOGIC_SRC or True
# dirs_stability lives in e8r3c_logic — sliced below into the E8J logic file.
E8R3C_LOGIC = open(os.path.join(HERE, "e8r3c_logic.py")).read()
i0 = E8R3C_LOGIC.index("def dirs_stability")
i1 = E8R3C_LOGIC.index("\ndef ", i0 + 1)
DIRS_STABILITY_SRC = ("# ── dirs_stability sliced VERBATIM from e8r3c_logic.py ──\n"
                      + E8R3C_LOGIC[i0:i1] + "\n")
assert "Sign-SENSITIVE" in DIRS_STABILITY_SRC

E8J_CELL = E8J_SRC + "\n\n" + DIRS_STABILITY_SRC

# ── assemble notebook ────────────────────────────────────────────────────────
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

add("markdown", MD0)
add("code", CELL_SETUP)
add("code", E8R2_LOGIC_SRC)
add("code", E8J_CELL)
add("code", CELL_MODEL)
add("code", CELL_STAGE)
add("code", CELL_VERDICT)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    assert "rclone" not in src.lower(), f"cell {i} mentions rclone (UI-only law)"
    if cell["cell_type"] == "code":
        compile(src, f"<cell {i}>", "exec")
print("cells compile; zero rclone anywhere")

out_nb = os.path.join(HERE, "E8J_BRIDGE_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_nb, f"({len(nb['cells'])} cells)")

logic_out = os.path.join(HERE, "e8j_logic.py")
with open(logic_out, "w") as f:
    f.write(E8R2_LOGIC_SRC + "\n\n" + E8J_CELL)
print("wrote", logic_out)
