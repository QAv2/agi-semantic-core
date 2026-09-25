"""Power check for the J1/J2 pre-registration (numpy only, seed 20260925). Output: power_sim_output.txt.
NOTE: the P1 section here is superseded by power_p1b.py (200 permutations cannot reach p < .0033).
P2 (J1): paired overconfidence difference A1-same minus A0, cluster bootstrap CI.
P1 (J1): incremental AUROC of Jev confidence beyond input-only features (permutation null).
P3 (J2): per-rung paired P(hold) difference CI half-width."""
import numpy as np
rng = np.random.default_rng(20260925)
R = 3  # repeats per stimulus

def boot_ci(per_stim, B=2000):
    n = len(per_stim); idx = rng.integers(0, n, (B, n))
    m = per_stim[idx].mean(1); return np.percentile(m, [2.5, 97.5])

def sim_p2(n_stim, drop, reader, sims=300):
    """reader='self': confidence tracks own accuracy (Delta=0); 'evidence': confidence
    ignores the access loss (Delta=+drop)."""
    inside = over10 = 0; widths = []
    for _ in range(sims):
        # stimulus-level ideal-observer strength -> Jev accuracy per stimulus
        z = rng.normal(0, 1, n_stim)
        q0 = 1/(1+np.exp(-(1.1 + 0.9*z)))          # mean ~.72-.75
        q1 = 1/(1+np.exp(-(1.1 - drop*4.5 + 0.9*z)))  # access degraded
        c0 = np.clip(q0 + rng.normal(0, .08, (R, n_stim)), .25, 1)
        c1 = np.clip((q1 if reader == 'self' else q0) + rng.normal(0, .08, (R, n_stim)), .25, 1)
        k0 = rng.random((R, n_stim)) < q0; k1 = rng.random((R, n_stim)) < q1
        d = (c1 - k1).mean(0) - (c0 - k0).mean(0)
        lo, hi = boot_ci(d); widths.append((hi-lo)/2)
        inside += (lo > -.05) and (hi < .05); over10 += d.mean() > .10
    return inside/sims, over10/sims, float(np.mean(widths)), float((q0.mean()-q1.mean()))

for n in (400, 800, 1200):
    for drop in (.10, .15):
        a = sim_p2(n, drop, 'self'); b = sim_p2(n, drop, 'evidence')
        print(f"P2 n_stim={n:5d} acc-drop~{a[3]:.3f}: self-knower PASS {a[0]:.2f} (CI half {a[2]:.3f}) | "
              f"evidence-reader FAIL(>.10) {b[1]:.2f}, false PASS {b[0]:.2f}")

def logit_fit(X, y, it=25):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1/(1+np.exp(-X@w)); W = p*(1-p)+1e-9
        w += np.linalg.solve(X.T@(X*W[:,None]) + 1e-6*np.eye(len(w)), X.T@(y-p))
    return 1/(1+np.exp(-X@w))

def auroc(s, y):
    r = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort').astype(float)
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y)-n1
    return (r[y == 1].sum() - n1*(n1+1)/2)/(n1*n0)

def sim_p1(n_stim, self_sig, sims=40, perms=200):
    hits = 0; deltas = []
    for _ in range(sims):
        z = np.repeat(rng.normal(0, 1, n_stim*3), R)          # 3 arms x n_stim, x R
        arm = np.repeat(np.tile(np.arange(3), n_stim), R)
        base = 1.1 + 0.9*z - 0.6*(arm == 1)
        u = rng.normal(0, 1, len(z))                          # trial-level processing noise
        err = (rng.random(len(z)) > 1/(1+np.exp(-(base + 0.8*u)))).astype(int)
        c = 1/(1+np.exp(-(base*0.9 + self_sig*u + rng.normal(0, .5, len(z)))))
        c = np.round(c, 2); lc = np.log(np.clip(c, .005, .995)/(1-np.clip(c, .005, .995)))
        Xin = np.column_stack([z, arm == 1, arm == 2])
        a0 = auroc(logit_fit(Xin, err), err); a1 = auroc(logit_fit(np.column_stack([Xin, lc]), err), err)
        d = a1 - a0; deltas.append(d); null = []
        for _ in range(perms):
            lp = lc.copy()
            for a in range(3):  # permute within arm (strata)
                m = arm == a; lp[m] = rng.permutation(lp[m])
            null.append(auroc(logit_fit(np.column_stack([Xin, lp]), err), err) - a0)
        p = (1 + np.sum(np.array(null) >= d))/(perms+1); hits += p < .01/3
    return hits/sims, float(np.mean(deltas))

for sig in (0.0, 0.15, 0.3):
    pw, d = sim_p1(400, sig)
    print(f"P1 n_stim=400/family, self-signal {sig}: mean dAUROC {d:.3f}, power(p<.0033) {pw:.2f}")

# J2 P3: paired named-vs-neutral P(hold) per rung, 120 ladders x R repeats
for n in (120, 240):
    w = []
    for _ in range(300):
        ph = rng.beta(2, 2, n); d = (np.clip(ph + rng.normal(0, .12, (R, n)), 0, 1) -
                                     np.clip(ph + rng.normal(0, .12, (R, n)), 0, 1)).mean(0)
        lo, hi = boot_ci(d, 1000); w.append((hi-lo)/2)
    print(f"J2 P3 ladders={n}: per-rung CI half-width {np.mean(w):.3f}")
