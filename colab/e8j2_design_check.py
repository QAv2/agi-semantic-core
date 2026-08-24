#!/usr/bin/env python3
"""E8-J v2 design check (lane law: BEFORE registration, BEFORE build).

Fork (a) from the E8-J lock: reverse-bridge wing re-encoding. This check is
ANCHOR-SIDE ONLY — it establishes whether the reverse bridge (hidden dirs ->
14D coordinates) is a usable instrument, what its noise floor is, and what a
healthy in-register 13-concept round-trip looks like. It NEVER computes a
reverse estimate for a wing dir: the wing reads are the registered unread
measurement. Wing HAND coordinates (public, in the pack) are used only for
the region-hardness control (§7).

Run:  python3 colab/e8j2_design_check.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

RES = os.path.join(HERE, "results_e8j", "full_20260824_1827")
LAM_GRID = [1.0, 10.0, 100.0]
SEED = L.E8J_SEED + 40          # fresh stream, disjoint from flown offsets


# ── reverse-map primitives (raw affine — coords are NOT unit vectors) ────────
def reverse_fit(D, X, lam):
    """Ridge dirs->coords, unpenalized intercept. Reuses L.ridge_fit verbatim."""
    return L.ridge_fit(D, X, lam)

def reverse_predict(W, D):
    """RAW affine prediction (no row normalization — coordinate space)."""
    return np.hstack([D, np.ones((D.shape[0], 1))]) @ W

def loo_raw_from_hat(H, Y):
    """Exact ridge LOO predictions WITHOUT unit normalization."""
    h = np.diag(H)
    return (H @ Y - h[:, None] * Y) / (1.0 - h)[:, None]


# ── §0 self-tests: exact machinery + planted teeth ───────────────────────────
def self_tests():
    rng = np.random.default_rng(SEED)
    n, d, k = 60, 40, 14
    D = rng.normal(size=(n, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = rng.normal(size=(n, k))
    # exact LOO == naive refit, all grid λ
    for lam in LAM_GRID:
        H = L.hat_matrix(D, lam)
        P = loo_raw_from_hat(H, X)
        for i in [0, 17, n - 1]:
            m = np.ones(n, bool); m[i] = False
            W = reverse_fit(D[m], X[m], lam)
            naive = reverse_predict(W, D[i:i + 1])[0]
            assert np.allclose(P[i], naive, atol=1e-8), f"LOO!=refit λ={lam} i={i}"
    # reverse_predict == naive affine
    W = reverse_fit(D, X, 10.0)
    assert np.allclose(reverse_predict(W, D),
                       np.hstack([D, np.ones((n, 1))]) @ W, atol=1e-12)
    # planted teeth: an in-register world separates from its relabel null…
    B = rng.normal(size=(k, d))
    Dp = X @ B + 0.3 * rng.normal(size=(n, d))
    Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
    H = L.hat_matrix(Dp, 10.0)
    err_true = np.linalg.norm(loo_raw_from_hat(H, X) - X, axis=1)
    meds = []
    for _ in range(200):
        Xp = X[rng.permutation(n)]
        meds.append(np.median(np.linalg.norm(loo_raw_from_hat(H, Xp) - Xp, axis=1)))
    p_plant = (1 + sum(m <= np.median(err_true) for m in meds)) / 201
    assert p_plant < 0.05, f"planted in-register world failed (p={p_plant})"
    # …and a permuted world does NOT
    Xsh = X[rng.permutation(n)]
    err_sh = np.linalg.norm(loo_raw_from_hat(H, Xsh) - Xsh, axis=1)
    meds_sh = []
    for _ in range(200):
        Xp = Xsh[rng.permutation(n)]
        meds_sh.append(np.median(np.linalg.norm(loo_raw_from_hat(H, Xp) - Xp, axis=1)))
    p_perm = (1 + sum(m <= np.median(err_sh) for m in meds_sh)) / 201
    assert p_perm > 0.05, f"permuted world passed (p={p_perm}) — teeth broken"
    print(f"  §0 self-tests GREEN (planted p={p_plant:.4f}, permuted p={p_perm:.3f})")


def main():
    t0 = time.time()
    print("E8-J v2 DESIGN CHECK — reverse bridge, anchor-side only")
    print("=" * 66)
    self_tests()

    # ── §1 load + asserts ────────────────────────────────────────────────────
    dirs = json.load(open(os.path.join(RES, "atlas_dirs.json")))
    shipped = json.load(open(os.path.join(RES, "atlas.json")))
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    anchors = dirs["anchors"]
    assert anchors == list(L.PINNED_ANCHORS), "anchor list drifted vs pins"
    assert shipped["anchors_sha"] == L.ANCHORS_SHA
    for w in L.CHOICE_SET:
        assert w not in anchors, f"{w} leaked into anchors"
        assert w in dirs["inst14"], f"{w} dir missing from atlas"   # presence only
    Xa = np.array([VEC[a] for a in anchors], float)                 # 320×14
    def _D(dd):
        M = np.array([dd[a] for a in anchors], float)
        return M / np.linalg.norm(M, axis=1, keepdims=True)
    Ya14, Ya20, Yb14 = _D(dirs["inst14"]), _D(dirs["inst20"]), _D(dirs["base14"])
    sig = Xa.std(0, ddof=1)                                         # register scale
    print(f"  §1 atlas loaded: {len(anchors)} anchors, dirs {Ya14.shape[1]}D; "
          f"wing-13 present, never-anchored")

    # ── §2 coordinate-block structure (function ≈ scaled essence?) ───────────
    print("  §2 register structure:")
    for e_i, f_i in [(0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13)]:
        r = np.corrcoef(Xa[:, e_i], Xa[:, f_i])[0, 1]
        ratio = (Xa[:, f_i] / np.where(np.abs(Xa[:, e_i]) > 1e-9, Xa[:, e_i], np.nan))
        print(f"      {L.AXIS_NAMES[e_i]:>2s}↔{L.AXIS_NAMES[f_i]:<3s} corr {r:+.4f}  "
              f"median f/e ratio {np.nanmedian(ratio):+.3f}")
    sv = np.linalg.svd(Xa - Xa.mean(0), compute_uv=False)
    er_coord = sv.sum() ** 2 / (sv ** 2).sum()
    print(f"      anchor coord cloud eff-rank {er_coord:.2f}/14")

    # ── §3 reverse LOO across λ grid ─────────────────────────────────────────
    print("  §3 reverse LOO (instilled L14), λ grid:")
    best = None
    loo_by_lam = {}
    for lam in LAM_GRID:
        H = L.hat_matrix(Ya14, lam)
        P = loo_raw_from_hat(H, Xa)
        dz = np.linalg.norm((P - Xa) / sig, axis=1)                 # standardized Δ
        dr = np.linalg.norm(P - Xa, axis=1)                         # raw Δ
        shrink = P.std(0, ddof=1) / sig                             # per-axis
        r2 = 1 - ((P - Xa) ** 2).sum(0) / ((Xa - Xa.mean(0)) ** 2).sum(0)
        loo_by_lam[lam] = (P, dz)
        print(f"      λ={lam:<6} Δstd med {np.median(dz):.3f} "
              f"(q25 {np.percentile(dz,25):.3f} q75 {np.percentile(dz,75):.3f}) | "
              f"Δraw med {np.median(dr):.3f} | shrink med {np.median(shrink):.3f} | "
              f"axis R² med {np.median(r2):.3f} max {r2.max():.3f}")
        if best is None or np.median(dz) < best[1]:
            best = (lam, np.median(dz))
    lam_rev = best[0]
    P_loo, dz_loo = loo_by_lam[lam_rev]
    r2_loo = 1 - ((P_loo - Xa) ** 2).sum(0) / ((Xa - Xa.mean(0)) ** 2).sum(0)
    print(f"      → λ_rev pick: {lam_rev} (best median standardized Δ)")
    print("      per-axis LOO R² at pick: " + "  ".join(
        f"{n}={v:.3f}" for n, v in zip(L.AXIS_NAMES, r2_loo)))

    # ── §4 relabel null at pinned λ (G-M1, the instrument gate) ──────────────
    print(f"  §4 relabel null (2000 perms, λ={lam_rev}):")
    H = L.hat_matrix(Ya14, lam_rev)
    rng = np.random.default_rng(SEED + 1)
    null_meds = []
    for i in range(2000):
        pi = rng.permutation(len(anchors))
        Xp = Xa[pi]
        Pp = loo_raw_from_hat(H, Xp)
        null_meds.append(float(np.median(np.linalg.norm((Pp - Xp) / sig, axis=1))))
        if (i + 1) % 500 == 0:
            print(f"      null {i + 1}/2000")
    null_meds = np.array(null_meds)
    obs = float(np.median(dz_loo))
    p_gm1 = float((1 + int((null_meds <= obs).sum())) / (1 + len(null_meds)))
    print(f"      G-M1: obs Δstd med {obs:.3f} vs null {null_meds.mean():.3f} "
          f"(min {null_meds.min():.3f}) → p={p_gm1:.5f} "
          f"[{'PASS' if p_gm1 < 0.05 else 'FAIL'}]")

    # ── §5 A4 cross-check: flown axiswise row reproduces VERBATIM ────────────
    a4 = L.axiswise_reverse_r2(Ya14, Xa)
    ship_a4 = shipped["A4_axiswise_r2"]
    same = all(abs(a4[k] - ship_a4[k]) < 5e-4 for k in a4)
    print(f"  §5 A4 cross-check vs flown atlas: {'EXACT' if same else 'DIFFERS!'}")
    assert same, "A4 recompute drifted from the flown row"

    # ── §6 double-LOO round trip (forward∘reverse) + 13-set calibration ──────
    print("  §6 double-LOO round trip on anchors:")
    n = len(anchors)
    RT = np.zeros((n, Ya14.shape[1]))
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        Wf = L.ridge_fit(Xa[m], Ya14[m], L.LAM)      # forward, target held out
        RT[i] = L.bridge_predict(Wf, P_loo[i:i + 1])[0]
        if (i + 1) % 80 == 0:
            print(f"      round-trip {i + 1}/{n}")
    rt_ang = L.angles_rowwise(RT, Ya14)
    cosM = np.clip(RT @ Ya14.T, -1, 1)
    angM = np.degrees(np.arccos(cosM))
    nearest = np.argmin(angM, axis=1)
    top1 = int((nearest == np.arange(n)).sum())
    print(f"      own-angle med {np.median(rt_ang):.1f}° "
          f"(q25 {np.percentile(rt_ang,25):.1f} q75 {np.percentile(rt_ang,75):.1f}) | "
          f"retrieval top-1 {top1}/{n}")
    rng = np.random.default_rng(SEED + 2)
    hits13 = []
    for _ in range(2000):
        sub = rng.choice(n, 13, replace=False)
        sub_near = np.argmin(angM[np.ix_(sub, sub)], axis=1)
        hits13.append(int((sub_near == np.arange(13)).sum()))
    hits13 = np.array(hits13)
    print(f"      13-set round-trip retrieval calibration: "
          f"med {np.median(hits13):.0f}/13, q25 {np.percentile(hits13,25):.0f} "
          f"q75 {np.percentile(hits13,75):.0f}, P(hits≥4)={np.mean(hits13 >= 4):.3f}")

    # ── §7 wing-region hardness control (hand coords only — public) ──────────
    Xw_hand = np.array([VEC[c] for c in L.CHOICE_SET], float)
    cen = Xw_hand.mean(0)
    d_wing = np.linalg.norm((Xa - cen) / sig, axis=1)
    near = d_wing <= np.percentile(d_wing, 20)
    rho = np.corrcoef(np.argsort(np.argsort(d_wing)),
                      np.argsort(np.argsort(dz_loo)))[0, 1]
    print(f"  §7 region-hardness: corr(rank dist-to-wing-centroid, rank Δstd) "
          f"= {rho:+.3f}; near-wing (bottom 20%) Δstd med "
          f"{np.median(dz_loo[near]):.3f} vs all {np.median(dz_loo):.3f}")

    # ── §8 secondary instruments on standing dirs (anchor-side) ──────────────
    # L20 replication + base degeneracy expectation, reverse direction
    for label, Y in [("inst L20", Ya20), ("base L14", Yb14)]:
        Hx = L.hat_matrix(Y, lam_rev)
        Px = loo_raw_from_hat(Hx, Xa)
        dzx = np.linalg.norm((Px - Xa) / sig, axis=1)
        shr = float(np.median(Px.std(0, ddof=1) / sig))
        print(f"  §8 {label}: reverse-LOO Δstd med {np.median(dzx):.3f} | "
              f"shrink med {shr:.3f}")

    out = {
        "lam_rev": lam_rev, "gm1_obs": round(obs, 4), "gm1_p": round(p_gm1, 5),
        "null_med_mean": round(float(null_meds.mean()), 4),
        "anchor_dz_quartiles": [round(float(np.percentile(dz_loo, q)), 4)
                                for q in (25, 50, 75)],
        "axis_r2": {k: round(float(v), 4) for k, v in zip(L.AXIS_NAMES, r2_loo)},
        "roundtrip_top1_320": top1,
        "hits13_median": float(np.median(hits13)),
        "hits13_p_ge4": round(float(np.mean(hits13 >= 4)), 4),
        "region_rho": round(float(rho), 4),
        "coord_effrank": round(float(er_coord), 2),
    }
    path = os.path.join(HERE, "results_e8j2_design_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  summary → {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
