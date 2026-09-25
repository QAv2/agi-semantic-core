"""Ideal-observer accuracy on the J1 titration grids (protocol §4.2). numpy + sqlite3, seed 1."""
import numpy as np, sqlite3
from pathlib import Path
rng = np.random.default_rng(1); N = 20000; K = 4
# F1: one best machine (+Delta), n=6 readings, sigma=1 units of Delta/sigma = d
for d in [0.18, 0.25, 0.35, 0.5, 0.71, 1.0, 1.41]:
    S = rng.normal(0, 1, (N, K, 6)).sum(2); S[:, 0] += 6*d   # ideal: argmax of sums
    print(f"F1 d={d:.2f} ideal acc {np.mean(S.argmax(1) == 0):.3f}")
for d in [0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0]:
    S = rng.normal(0, 1, (N, 2, 6)).sum(2); S[:, 0] += 6*d
    print(f"F1-K2 d={d:.2f} ideal acc {np.mean(S.argmax(1) == 0):.3f}")
# F2: 14 coords of dictionary concepts
db = sqlite3.connect(Path(__file__).resolve().parents[2] / 'db' / 'semantic.db')
C = np.array(db.execute("select x,y,z,e,f,g,h,fx,fy,fz,fe,ff,fg,fh from concepts").fetchall())
idx = np.array([rng.choice(len(C), K, replace=False) for _ in range(4000)])
cand = C[idx]; nn = np.linalg.norm(cand[:, 1:] - cand[:, :1], axis=2).min(1)
m = np.median(nn); print("F2 median nearest-distractor distance", round(m, 3), "frac exact-dup", np.mean(nn < 1e-9))
for s in [0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0]:
    sig = s*m/np.sqrt(14); x = cand[:, 0] + rng.normal(0, sig, (4000, 14))
    dist = np.linalg.norm(cand - x[:, None], axis=2); print(f"F2 s={s:.2f} ideal acc {np.mean(dist.argmin(1) == 0):.3f}")
# F3: M=5 sensors, reliability r_j = rbar +- U(.08), report truth w.p. r else uniform other lane
for rb in [0.31, 0.36, 0.42, 0.49, 0.57, 0.66, 0.76]:
    r = np.clip(rbar := rb + rng.uniform(-.08, .08, (N, 5)), .26, .95)
    ok = rng.random((N, 5)) < r; rep = np.where(ok, 0, rng.integers(1, K, (N, 5)))
    L = np.zeros((N, K))
    for k in range(K): L[:, k] = np.sum(np.where(rep == k, np.log(r), np.log((1-r)/3)), 1)
    print(f"F3 rbar={rb:.2f} ideal acc {np.mean(L.argmax(1) == 0):.3f}")
