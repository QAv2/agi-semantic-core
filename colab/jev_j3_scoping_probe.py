"""J3 scoping probe (NOT a result): can recorded L14 states + synthetic
additive injections be told apart through (a) the 14-coord register and
(b) projections onto the 13 machine-state directions, by a trivial decoder?"""
import json, numpy as np
C = '/home/joseph/agi-semantic-core/colab/'
pay = json.load(open(C + 'e7bq_payload.json'))
W = np.asarray(pay['encoders']['inst14'], float)            # (1537,14)
d = json.load(open(C + 'results_e7bq/full_20260826_1839/condition_real.json'))
cent = np.asarray(d['centroid']['14'], float)
rows = [r for r in d['rows'] if isinstance(r.get('s_pre14'), list)]
S = np.asarray([r['s_pre14'] for r in rows], float)
arms = [r['arm'] for r in rows]
del d
n2 = json.load(open(C + 'results_e8n2/full_20260824_0050/condition_real.json'))
names = list(n2['dirs']['14'].keys())
D = np.asarray([n2['dirs']['14'][k] for k in names], float) # (13,1536)
mu = float(n2['mu']['14']); del n2
D = D / np.linalg.norm(D, axis=1, keepdims=True)
print('states', S.shape, 'arms', sorted(set(arms)), 'mu14', mu, 'concepts', len(names))
print('radius |s-c| mean %.1f' % np.linalg.norm(S - cent, axis=1).mean())
def reg14(X):
    V = X - cent; V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return np.hstack([V, np.ones((len(V), 1))]) @ W
def proj13(X):
    V = X - cent; V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V @ D.T
rng = np.random.default_rng(20260924)
idx = rng.permutation(len(S)); A, B = idx[:len(S)//2], idx[len(S)//2:]   # fit / test split
G = D @ D.T; off = G[~np.eye(13, dtype=bool)]
print('\ndirection cosines among the 13: mean %.3f  max %.3f' % (off.mean(), off.max()))
for feat_name, feat in [('14-coord register', reg14), ('13-dir projection', proj13)]:
    print('\n==', feat_name)
    for alpha in (0.25, 0.5, 1.0):
        # signatures fitted on split A: mean feature shift per concept
        base_A = feat(S[A]); m0 = base_A.mean(0)
        sig = np.stack([(feat(S[A] + alpha * mu * D[k]) - base_A).mean(0) for k in range(13)])
        # test on split B
        Xs = feat(S[B]) - m0
        hits = 0; tot = 0; scores_inj = []
        for k in range(13):
            Xi = feat(S[B] + alpha * mu * D[k]) - m0
            # nearest signature by cosine
            cs = (Xi @ sig.T) / (np.linalg.norm(Xi, axis=1, keepdims=True) * np.linalg.norm(sig, axis=1) + 1e-12)
            hits += (cs.argmax(1) == k).sum(); tot += len(Xi)
            scores_inj.append(np.linalg.norm(Xi, axis=1))
        scores_inj = np.concatenate(scores_inj); scores_sham = np.linalg.norm(Xs, axis=1)
        # detection AUROC: injected vs sham by shift magnitude
        allv = np.concatenate([scores_inj, scores_sham]); lab = np.r_[np.ones(len(scores_inj)), np.zeros(len(scores_sham))]
        order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
        auc = (ranks[lab == 1].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (1 - lab).sum())
        print('  alpha %.2f: 13-way id acc %.3f (chance .077) | detect AUROC inj-vs-sham %.3f | sig norm mean %.3f vs sham spread %.3f'
              % (alpha, hits / tot, auc, np.linalg.norm(sig, axis=1).mean(), scores_sham.mean()))

# instrument sanity: does reg14 reproduce the stored chat_pre14?
st = np.asarray([r['chat_pre14'] for r in rows], float)
print('\nmax |reg14 - stored chat_pre14| = %.2e' % np.abs(reg14(S) - st).max())
iu = np.triu_indices(13, 1); top = np.argsort(G[iu])[::-1][:3]
for t in top: print('closest pair: %s ~ %s  cos %.3f' % (names[iu[0][t]], names[iu[1][t]], G[iu][t]))
# titration on the 14-coord register: where does the trivial decoder sit mid-range?
for alpha in (0.03, 0.06, 0.1, 0.15):
    base_A = reg14(S[A]); m0 = base_A.mean(0)
    sig = np.stack([(reg14(S[A] + alpha * mu * D[k]) - base_A).mean(0) for k in range(13)])
    hits = tot = 0; si = []
    for k in range(13):
        Xi = reg14(S[B] + alpha * mu * D[k]) - m0
        cs = (Xi @ sig.T) / (np.linalg.norm(Xi, axis=1, keepdims=True) * np.linalg.norm(sig, axis=1) + 1e-12)
        hits += (cs.argmax(1) == k).sum(); tot += len(Xi); si.append(np.linalg.norm(Xi, axis=1))
    si = np.concatenate(si); ss = np.linalg.norm(reg14(S[B]) - m0, axis=1)
    allv = np.r_[si, ss]; lab = np.r_[np.ones(len(si)), np.zeros(len(ss))]
    o = allv.argsort(); rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
    auc = (rk[lab == 1].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (1 - lab).sum())
    print('  register alpha %.2f: id acc %.3f | detect AUROC %.3f' % (alpha, hits / tot, auc))
