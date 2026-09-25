"""SUPERSEDED null (kept for the record): LINEAR in-sample input-only null, stratified permutation,
z-approximation. Showed a 3/30 false-positive rate at z > 2.72 with no self-signal -> protocol uses
the flexible null (power_p1b.py). Output: power_p1_linear_output.txt."""
import numpy as np
from scipy.stats import rankdata
rng = np.random.default_rng(20260926); R = 3
def logit_fit(X, y, it=20):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-X@w)); W = p*(1-p)+1e-9
        w += np.linalg.solve(X.T@(X*W[:,None]) + 1e-6*np.eye(len(w)), X.T@(y-p))
    return 1/(1+np.exp(-X@w))
def auroc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)
def sim(n_stim, sig, sims=30, perms=150):
    zs = []; ds = []
    for _ in range(sims):
        z = np.repeat(rng.normal(0, 1, n_stim*3), R); arm = np.repeat(np.tile(np.arange(3), n_stim), R)
        base = 1.1 + 0.9*z - 0.6*(arm == 1); u = rng.normal(0, 1, len(z))
        err = (rng.random(len(z)) > 1/(1+np.exp(-(base + 0.8*u)))).astype(int)
        lc = base*0.9 + sig*u + rng.normal(0, .5, len(z))
        c = np.round(1/(1+np.exp(-lc)), 2); lc = np.log(np.clip(c,.005,.995)/(1-np.clip(c,.005,.995)))
        dec = np.digitize(z, np.quantile(z, np.linspace(.1, .9, 9))); strata = arm*10 + dec
        Xin = np.column_stack([z, arm == 1, arm == 2]); a0 = auroc(logit_fit(Xin, err), err)
        d = auroc(logit_fit(np.column_stack([Xin, lc]), err), err) - a0; null = []
        for _ in range(perms):
            lp = lc.copy()
            for s in np.unique(strata):
                m = strata == s; lp[m] = rng.permutation(lp[m])
            null.append(auroc(logit_fit(np.column_stack([Xin, lp]), err), err) - a0)
        null = np.array(null); zs.append((d - null.mean())/null.std()); ds.append(d)
    zs = np.array(zs); return float(np.mean(ds)), float(np.mean(zs > 2.72)), float(np.median(zs))
for sig in (0.0, 0.15, 0.3):
    d, pw, zm = sim(400, sig); print(f"P1 400/family sig={sig}: dAUROC {d:.3f}  median z {zm:.2f}  power(z>2.72 ~ p<.0033) {pw:.2f}")
