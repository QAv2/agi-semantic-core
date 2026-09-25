"""J1-P1 power + null calibration: flexible input-only null (strength-decile x arm bins), grouped CV,
stratified permutation, z-approximation. Output: power_p1b_output.txt. Run single-threaded:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python power_p1b.py"""
import numpy as np
from scipy.stats import rankdata
rng = np.random.default_rng(20260927); R = 3
def logit_fit(Xtr, ytr, Xte, it=20):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-A@w)); W = p*(1-p)+1e-9
        w += np.linalg.solve(A.T@(A*W[:,None]) + 1e-4*np.eye(len(w)), A.T@(ytr-p))
    return 1/(1+np.exp(-np.column_stack([np.ones(len(Xte)), Xte])@w))
def auroc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)
def cv_pred(X, y, grp, k=5):
    out = np.empty(len(y)); fold = grp % k
    for f in range(k):
        te = fold == f; out[te] = logit_fit(X[~te], y[~te], X[te])
    return out
def design(z, arm):
    edges = np.quantile(z, np.linspace(.1, .9, 9)); dec = np.digitize(z, edges)
    cols = [z*(arm == a) for a in range(3)] + [(arm == a).astype(float) for a in (1, 2)]
    cols += [((dec == d) & (arm == a)).astype(float) for a in range(3) for d in range(1, 10)]
    return np.column_stack(cols), arm*10 + dec
def sim(n_stim, sig, sims, perms=60):
    zs = []; ds = []
    for _ in range(sims):
        stim = np.repeat(np.arange(n_stim*3), R)
        z = np.repeat(rng.normal(0, 1, n_stim*3), R); arm = np.repeat(np.tile(np.arange(3), n_stim), R)
        base = 1.1 + 0.9*z - 0.6*(arm == 1); u = rng.normal(0, 1, len(z))
        err = (rng.random(len(z)) > 1/(1+np.exp(-(base + 0.8*u)))).astype(int)
        lc = base*0.9 + sig*u + rng.normal(0, .5, len(z))
        c = np.round(1/(1+np.exp(-lc)), 2); lc = np.log(np.clip(c,.005,.995)/(1-np.clip(c,.005,.995)))
        Xin, strata = design(z, arm); grp = rng.permutation(n_stim*3)[stim]
        a0 = auroc(cv_pred(Xin, err, grp), err)
        d = auroc(cv_pred(np.column_stack([Xin, lc]), err, grp), err) - a0; null = []
        for _ in range(perms):
            lp = lc.copy()
            for s in np.unique(strata):
                m = strata == s; lp[m] = rng.permutation(lp[m])
            null.append(auroc(cv_pred(np.column_stack([Xin, lp]), err, grp), err) - a0)
        null = np.array(null); zs.append((d - null.mean())/null.std()); ds.append(d); print(f'  sig={sig} d={d:.4f} z={zs[-1]:.2f} p_emp={(1+np.sum(null>=d))/(len(null)+1):.3f}', flush=True)
    zs = np.array(zs); return float(np.mean(ds)), float(np.mean(zs > 2.72)), float(np.median(zs))
for sig, n in ((0.0, 25), (0.15, 6)):
    d, pw, zm = sim(400, sig, n); print(f"P1-flex sig={sig}: dAUROC {d:.4f} median z {zm:.2f} rate(z>2.72) {pw:.2f}", flush=True)
