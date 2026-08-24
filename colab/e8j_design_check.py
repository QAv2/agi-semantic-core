# E8-J design check — bridge feasibility BEFORE pre-reg (lane law).
# Question: do the dictionary's hand-tuned 14D coordinates predict the substrate's
# hidden-state concept directions, beyond (a) an arbitrary codebook pairing and
# (b) nearest-trained lookup?
import json, itertools
import numpy as np

RES = '/home/joseph/agi-semantic-core/colab/results_e8r3c/full_20260824_0505/condition_real.json'
PACK = '/home/joseph/agi-semantic-core/colab/e4_dictionary_pack.json'
CHOICE_SET = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION","RETRIEVAL",
              "CONSTRUCTION","SATURATION","FAMILIARITY","NOVELTY","CAPTURE",
              "DIVERGENCE","CONFABULATION","CALIBRATION"]
HELD_OUT = ["DIVERGENCE","NOVELTY","RETRIEVAL","TENSION"]
TRAINED = [c for c in CHOICE_SET if c not in HELD_OUT]
SEED = 20260824
N_PERM = 2000

b = json.load(open(RES))
pack = {c['name']: np.array(c['vec'], float) for c in json.load(open(PACK))['concepts']
        if c['name'] in CHOICE_SET}
D = {L: np.array([b['dirs'][L][n] for n in CHOICE_SET]) for L in ('14', '20')}
for L in D:
    D[L] = D[L] / np.linalg.norm(D[L], axis=1, keepdims=True)
V = np.array([pack[n] for n in CHOICE_SET])          # 13 x 14 dictionary coords
n, d = V.shape

def ang(a, bb):
    c = np.clip(a @ bb / (np.linalg.norm(a) * np.linalg.norm(bb)), -1, 1)
    return np.degrees(np.arccos(c))

# ── DC-0: anchor-cloud conditioning ──────────────────────────────────────────
sv = np.linalg.svd(V - V.mean(0), compute_uv=False)
er = (sv.sum())**2 / (sv**2).sum()  # participation-ratio effective rank
print(f"DC-0 coordinate cloud: 13x14, singular values {np.round(sv,2)}")
print(f"     effective rank (participation ratio) = {er:.2f}")
hidden_sv = np.linalg.svd(D['14'] - D['14'].mean(0), compute_uv=False)
her = (hidden_sv.sum())**2 / (hidden_sv**2).sum()
print(f"     hidden-dir cloud L14 effective rank = {her:.2f}\n")

# ── DC-1: RSA — dictionary pairwise angles vs hidden pairwise angles ─────────
def pairwise_ang(M):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    G = np.clip(Mn @ Mn.T, -1, 1)
    return np.degrees(np.arccos(G))

def mantel(A, B, n_perm=100000, seed=SEED):
    iu = np.triu_indices(A.shape[0], 1)
    a, bb = A[iu], B[iu]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(bb)).astype(float)
    def sp(x, y):
        return np.corrcoef(x, y)[0, 1]
    obs = sp(ra, rb)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(A.shape[0])
        cnt += sp(np.argsort(np.argsort(A[np.ix_(p, p)][iu])).astype(float), rb) >= obs
    return obs, (cnt + 1) / (n_perm + 1)

layers = {
    'full14':    V,
    'essence7':  V[:, :7],
    'function7': V[:, 7:],
    'core3':     V[:, :3],
    'domain4':   V[:, 3:7],
}
print("DC-1 RSA (Mantel, Spearman over 78 pairs, 100k label perms):")
for hl in ('14', '20'):
    H = pairwise_ang(D[hl])
    for name, M in layers.items():
        rho, p = mantel(pairwise_ang(M), H, n_perm=20000)
        print(f"     L{hl} vs {name:10s}: rho={rho:+.3f}  p={p:.4f}")
print()

# ── DC-2/3: LOO ridge bridge + exact-hit geometry + relabel null ─────────────
def ridge_fit(A, Y, lam):
    X = np.hstack([A, np.ones((A.shape[0], 1))])
    P = np.eye(X.shape[1]) * lam
    P[-1, -1] = 0.0                      # intercept unpenalized
    return np.linalg.solve(X.T @ X + P, X.T @ Y)

def loo_predict(Vm, Dm, lam):
    """For each i: fit on the other 12, predict dir_i from coords_i."""
    preds = np.zeros_like(Dm)
    for i in range(Vm.shape[0]):
        tr = np.arange(Vm.shape[0]) != i
        W = ridge_fit(Vm[tr], Dm[tr], lam)
        p = np.append(Vm[i], 1.0) @ W
        preds[i] = p / np.linalg.norm(p)
    return preds

def eval_preds(preds, Dm):
    angs = np.array([ang(preds[i], Dm[i]) for i in range(len(preds))])
    near = np.array([np.argmin([ang(preds[i], Dm[j]) for j in range(len(Dm))])
                     for i in range(len(preds))])
    hits = int((near == np.arange(len(preds))).sum())
    return angs, hits

Dm = D['14']
print("DC-2 LOO ridge bridge (fit 12 -> predict 13th), lambda sweep:")
lam_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
for lam in lam_grid:
    preds = loo_predict(V, Dm, lam)
    angs, hits = eval_preds(preds, Dm)
    print(f"     lam={lam:5.2f}: median LOO angle={np.median(angs):5.2f}  "
          f"exact geometric hits={hits}/13")
mu_base = Dm.mean(0); mu_base /= np.linalg.norm(mu_base)
mu_angs = np.array([ang(mu_base, Dm[i]) for i in range(n)])
print(f"     mu-baseline (mean dir): median angle={np.median(mu_angs):5.2f}")
pair_angs = pairwise_ang(Dm)
np.fill_diagonal(pair_angs, np.inf)
print(f"     nearest-OTHER-dir floor: median={np.median(pair_angs.min(1)):5.2f} "
      f"(min {pair_angs.min():.2f})\n")

LAM = 1.0   # provisional pin; sensitivity shown above, null shares lam
preds = loo_predict(V, Dm, LAM)
angs, hits = eval_preds(preds, Dm)
print(f"DC-3 exact-hit + relabel null at lam={LAM}:")
print(f"     REAL: median angle={np.median(angs):.2f}, exact hits={hits}/13")
for nm, a, h in [(CHOICE_SET[i], angs[i],
                  int(np.argmin([ang(preds[i], Dm[j]) for j in range(n)]) == i))
                 for i in range(n)]:
    pass
rng = np.random.default_rng(SEED)
null_hits, null_med = [], []
for _ in range(N_PERM):
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):          # derangements only
        perm = rng.permutation(n)
    Vp = V[perm]
    pp = loo_predict(Vp, Dm, LAM)
    a_, h_ = eval_preds(pp, Dm)
    null_hits.append(h_); null_med.append(np.median(a_))
null_hits = np.array(null_hits); null_med = np.array(null_med)
p_hits = (1 + (null_hits >= hits).sum()) / (N_PERM + 1)
p_med = (1 + (null_med <= np.median(angs)).sum()) / (N_PERM + 1)
print(f"     NULL (codebook relabel, {N_PERM} derangements): "
      f"hits max={null_hits.max()}, mean={null_hits.mean():.2f}; "
      f"median-angle mean={null_med.mean():.2f}")
print(f"     p(exact hits)={p_hits:.4f}   p(median angle)={p_med:.4f}\n")

print("     per-concept (real, lam=1):")
for i, nm in enumerate(CHOICE_SET):
    nearest = CHOICE_SET[int(np.argmin([ang(preds[i], Dm[j]) for j in range(n)]))]
    tag = 'HELD' if nm in HELD_OUT else 'trn '
    print(f"       {tag} {nm:14s} angle={angs[i]:5.1f}  nearest={nearest}"
          f"{'  <-- HIT' if nearest == nm else ''}")
print()

# ── DC-4: flight configuration — fit on 9 TRAINED, predict 4 HELD-OUT ────────
tr_idx = [CHOICE_SET.index(c) for c in TRAINED]
ho_idx = [CHOICE_SET.index(c) for c in HELD_OUT]
W9 = ridge_fit(V[tr_idx], Dm[tr_idx], LAM)
print("DC-4 flight config (bridge on 9 trained anchors -> 4 held-out):")
for i in ho_idx:
    p_ = np.append(V[i], 1.0) @ W9
    p_ /= np.linalg.norm(p_)
    a_true = ang(p_, Dm[i])
    lookup = min(ang(Dm[j], Dm[i]) for j in tr_idx)
    near_all = CHOICE_SET[int(np.argmin([ang(p_, Dm[j]) for j in range(n)]))]
    print(f"     {CHOICE_SET[i]:11s} angle(pred,true)={a_true:5.1f}  "
          f"nearest-trained-dir floor={lookup:5.1f}  pred-nearest={near_all}")
print()

# ── DC-5: feature-axis separability (B columns) + span ───────────────────────
B14 = W9[:-1]                                    # 14 x 1536 feature dirs
Bn = B14 / np.linalg.norm(B14, axis=1, keepdims=True)
fa = np.degrees(np.arccos(np.clip(Bn @ Bn.T, -1, 1)))
iu = np.triu_indices(14, 1)
print(f"DC-5 feature axes (9-anchor bridge): pairwise angles "
      f"median={np.median(fa[iu]):.1f}, min={fa[iu].min():.1f}, max={fa[iu].max():.1f}")
bsv = np.linalg.svd(B14, compute_uv=False)
print(f"     B column-space singular values: {np.round(bsv, 3)}")

# ── DC-6: L20 texture ────────────────────────────────────────────────────────
Dm20 = D['20']
p20 = loo_predict(V, Dm20, LAM)
a20, h20 = eval_preds(p20, Dm20)
print(f"\nDC-6 L20 texture: median LOO angle={np.median(a20):.2f}, hits={h20}/13")
