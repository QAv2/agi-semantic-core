"""Statistics for the J1/J2 analyses (protocol §5.4, §6.3, §7). numpy + scipy only."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, rankdata


def auroc(score, y):
    """P(score of a positive > score of a negative); ties count half (Mann-Whitney)."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _design(X):
    return np.column_stack([np.ones(len(X)), X])


def logistic_fit(X, y, w0=None, iters=60, tol=1e-10, ridge=1e-6):
    """Newton-IRLS with a tiny ridge (identifiability under separation)."""
    A = _design(X)
    w = np.zeros(A.shape[1]) if w0 is None else w0.copy()
    R = ridge * np.eye(A.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(A @ w, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-12
        g = A.T @ (y - p) - R @ w
        H = A.T @ (A * W[:, None]) + R
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < tol:
            break
    return w


def logistic_predict(X, w):
    return 1 / (1 + np.exp(-np.clip(_design(X) @ w, -35, 35)))


def group_folds(groups, k, rng):
    """Fold index per row; all rows of a group share a fold (seeded)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    fold_of_group = rng.permutation(len(uniq)) % k
    return fold_of_group[inv]


def cv_oof(X, y, folds, warm=None):
    """Out-of-fold predictions; returns (pred, weights per fold) for warm starts."""
    pred = np.empty(len(y))
    ws = []
    for f in np.unique(folds):
        te = folds == f
        w = logistic_fit(X[~te], y[~te], w0=None if warm is None else warm[len(ws)])
        pred[te] = logistic_predict(X[te], w)
        ws.append(w)
    return pred, ws


def incremental_auroc_test(X0, extra, y, folds, strata, n_perm, rng):
    """ΔAUROC = AUROC(oof, X0 + extra) − AUROC(oof, X0); permutation of `extra`
    within strata. Returns dict(delta, auc0, auc1, p, null_mean, null_sd)."""
    p0, _ = cv_oof(X0, y, folds)
    auc0 = auroc(p0, y)
    X1 = np.column_stack([X0, extra])
    p1, w1 = cv_oof(X1, y, folds)
    auc1 = auroc(p1, y)
    delta = auc1 - auc0
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        e = extra.copy()
        for g in groups:
            e[g] = extra[rng.permutation(g)]
        pb, _ = cv_oof(np.column_stack([X0, e]), y, folds, warm=w1)
        null[b] = auroc(pb, y) - auc0
    p = (1 + np.sum(null >= delta)) / (1 + n_perm)
    return {"delta": float(delta), "auc0": float(auc0), "auc1": float(auc1), "p": float(p),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "n_perm": n_perm}


def holm(pvals, alpha):
    """Holm step-down: dict name -> (p, rejected)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, still = len(items), {}, True
    for i, (name, p) in enumerate(items):
        ok = still and p <= alpha / (m - i)
        still = ok
        out[name] = {"p": p, "threshold": alpha / (m - i), "rejected": bool(ok)}
    return out


def boot_mean_strat(values, strata, B, rng):
    """Cluster bootstrap of a mean of per-cluster values, resampled within strata."""
    values, strata = np.asarray(values, float), np.asarray(strata)
    idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    tot = sum(len(g) for g in idx)
    out = np.empty(B)
    for b in range(B):
        acc = 0.0
        for g in idx:
            acc += values[g[rng.integers(0, len(g), len(g))]].sum()
        out[b] = acc / tot
    return out


def ci(samples, level=0.95):
    a = (1 - level) / 2
    return [float(np.percentile(samples, 100 * a)), float(np.percentile(samples, 100 * (1 - a)))]


def sign_flip_p(diffs, B, rng):
    """One-sided p for mean(diffs) > 0 under random sign flips."""
    d = np.asarray(diffs, float)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = (signs * d).mean(1)
    return float((1 + np.sum(null >= obs)) / (1 + B)), float(obs)


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def norm_entropy(p):
    p = np.asarray(p, float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(p)))


def ece_equal_mass(c, y, bins):
    c, y = np.asarray(c, float), np.asarray(y, float)
    o = np.argsort(c, kind="mergesort")
    parts = np.array_split(o, bins)
    return float(sum(len(g) / len(c) * abs(c[g].mean() - y[g].mean()) for g in parts if len(g)))


def brier_murphy(c, y):
    """Binary Brier on (confidence, correct) + Murphy (1973) decomposition over the
    returned 0.01 levels: Brier = REL − RES + UNC (exact for constant-in-bin forecasts)."""
    c, y = np.asarray(c, float), np.asarray(y, float)
    ob = y.mean()
    rel = res = 0.0
    for v in np.unique(c):
        m = c == v
        w = m.mean()
        rel += w * (v - y[m].mean()) ** 2
        res += w * (y[m].mean() - ob) ** 2
    return {"brier": float(((c - y) ** 2).mean()), "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(ob * (1 - ob))}


def meta_i_bits(c, y, bins=10):
    """Plug-in mutual information I(correct; confidence bin), bits (Dayan 2023's
    meta-I, with confidence in equal-mass bins)."""
    c, y = np.asarray(c, float), np.asarray(y).astype(int)
    edges = np.unique(np.quantile(c, np.linspace(0, 1, bins + 1)[1:-1]))
    b = np.searchsorted(edges, c, side="right")
    mi = 0.0
    for bi in np.unique(b):
        for yi in (0, 1):
            pj = np.mean((b == bi) & (y == yi))
            if pj > 0:
                mi += pj * np.log2(pj / (np.mean(b == bi) * np.mean(y == yi)))
    return float(mi)


def meta_d_mle(stim, resp, conf, n_ratings):
    """Single-subject MLE meta-d' (Maniscalco & Lau 2012) for a 2-choice task.
    stim, resp in {0,1} (0 = 'S1'); conf in 1..n_ratings. Padding 1/(2 n_ratings)."""
    stim, resp, conf = map(np.asarray, (stim, resp, conf))
    nR = n_ratings
    pad = 1 / (2 * nR)

    def counts(s):
        # [resp S1 conf nR..1, resp S2 conf 1..nR]
        a = [np.sum((stim == s) & (resp == 0) & (conf == j)) for j in range(nR, 0, -1)]
        b = [np.sum((stim == s) & (resp == 1) & (conf == j)) for j in range(1, nR + 1)]
        return np.array(a + b, float) + pad

    n1, n2 = counts(0), counts(1)
    hr = n2[nR:].sum() / n2.sum()
    far = n1[nR:].sum() / n1.sum()
    d1 = norm.ppf(hr) - norm.ppf(far)
    c1 = -0.5 * (norm.ppf(hr) + norm.ppf(far))

    def nll(theta):
        md = theta[0]
        mc = c1 * md / d1 if d1 != 0 else c1
        inc = np.exp(theta[1:])
        t_s1 = mc - np.cumsum(inc[: nR - 1])        # response S1 criteria (descending)
        t_s2 = mc + np.cumsum(inc[nR - 1:])        # response S2 criteria (ascending)
        ll = 0.0
        for s, n in ((0, n1), (1, n2)):
            mu = -md / 2 if s == 0 else md / 2
            # S1 responses: conf nR (x < t_{nR-1}) ... conf 1 (t_1 < x < mc)
            edges1 = np.concatenate([[-np.inf], t_s1[::-1], [mc]])
            p1 = np.diff(norm.cdf(edges1 - mu)) / max(norm.cdf(mc - mu), 1e-300)
            edges2 = np.concatenate([[mc], t_s2, [np.inf]])
            p2 = np.diff(norm.cdf(edges2 - mu)) / max(1 - norm.cdf(mc - mu), 1e-300)
            ll += (n[:nR] * np.log(np.clip(p1, 1e-300, None))).sum()
            ll += (n[nR:] * np.log(np.clip(p2, 1e-300, None))).sum()
        return -ll

    x0 = np.concatenate([[d1], np.full(2 * (nR - 1), np.log(0.5))])
    fit = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
    md = float(fit.x[0])
    return {"d_prime": float(d1), "meta_d": md, "m_ratio": float(md / d1) if d1 else np.nan,
            "converged": bool(fit.success)}
