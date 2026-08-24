#!/usr/bin/env python3
"""E8-J v2 measurement (registered: docs/E8J2_PROTOCOL.md §2, commit 28ad28c).

The first wing reverse-read in the program: estimate the wing-13's
coordinates FROM their substrate dirs through the broad reverse bridge,
measure the delta against the session-125 hand coordinates vs the bridge's
own noise floor, and (G-M2) test whether re-encoded coordinates round-trip
through the forward bridge geometrically well enough to stage the Part-2
flight. Local, pure numpy, standing atlas only.

Run:  python3 colab/e8j2_measure.py
"""
import hashlib, json, math, os, sqlite3, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L
from e8j2_design_check import reverse_fit, reverse_predict, loo_raw_from_hat

RES = os.path.join(HERE, "results_e8j", "full_20260824_1827")
OUT = os.path.join(HERE, "results_e8j2")
LAM_REV = 1.0                                   # pinned (prereg §1)
S_GM1, S_GM2, S_PROC, S_RSA_H, S_RSA_E, S_PM = (L.E8J_SEED + k
                                                for k in (50, 51, 52, 53, 54, 55))


def effrank(M):
    sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
    return float(sv.sum() ** 2 / (sv ** 2).sum())


def ranksum_perm(a, b, n_perm, seed):
    """One-sided (a larger): rank-sum of a in pooled ranks, label permutation."""
    pool = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(pool)).astype(float)
    obs = float(ranks[: len(a)].sum())
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        pi = rng.permutation(len(pool))
        cnt += ranks[pi[: len(a)]].sum() >= obs
    return obs, float((1 + cnt) / (1 + n_perm))


def procrustes_resid(A, B):
    """Rotation-only Procrustes residual mapping centered A onto centered B."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    R = U @ Vt
    return float(np.linalg.norm(Ac @ R - Bc) / np.linalg.norm(Bc))


def sha_vecs(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    V = {}                                       # verdict accumulator

    # ── §0 load + asserts ────────────────────────────────────────────────────
    dirs = json.load(open(os.path.join(RES, "atlas_dirs.json")))
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    anchors = dirs["anchors"]
    assert anchors == list(L.PINNED_ANCHORS)
    for w in L.CHOICE_SET:
        assert w not in anchors
    Xa = np.array([VEC[a] for a in anchors], float)
    Xw_hand = np.array([VEC[c] for c in L.CHOICE_SET], float)
    def _D(dd, names):
        M = np.array([dd[n] for n in names], float)
        return M / np.linalg.norm(M, axis=1, keepdims=True)
    Ya14 = _D(dirs["inst14"], anchors)
    Ya20 = _D(dirs["inst20"], anchors)
    Yb14 = _D(dirs["base14"], anchors)
    Dw14 = _D(dirs["inst14"], L.CHOICE_SET)
    Dw20 = _D(dirs["inst20"], L.CHOICE_SET)
    Dwb14 = _D(dirs["base14"], L.CHOICE_SET)
    sig = Xa.std(0, ddof=1)
    xbar = Xa.mean(0)
    # machinery-identity assert from the prereg (predict∘permute == permute∘predict)
    _W = L.ridge_fit(Xa[:20], Ya14[:20], L.LAM)
    _pi = np.array([3, 0, 2, 1])
    assert np.allclose(L.bridge_predict(_W, Xw_hand[:4][_pi]),
                       L.bridge_predict(_W, Xw_hand[:4])[_pi], atol=1e-12)
    # planted tooth for the P-M machinery (separation detected, null calibrated)
    rng_t = np.random.default_rng(1)
    _, p_sep = ranksum_perm(rng_t.normal(3, 1, 13), rng_t.normal(0, 1, 64), 2000, 2)
    _, p_nul = ranksum_perm(rng_t.normal(0, 1, 13), rng_t.normal(0, 1, 64), 2000, 3)
    assert p_sep < 0.01 and p_nul > 0.05, "P-M machinery teeth failed"
    print("E8-J v2 MEASUREMENT — first wing reverse-read in the program")
    print("=" * 66)
    print(f"  §0 loaded; asserts green (identity, teeth p={p_sep:.4f}/{p_nul:.3f})")

    # ── §1 G-M1 re-verify (relabel null, fresh seed) ─────────────────────────
    H = L.hat_matrix(Ya14, LAM_REV)
    P_loo = loo_raw_from_hat(H, Xa)
    dz_anchor = np.linalg.norm((P_loo - Xa) / sig, axis=1)
    obs_med = float(np.median(dz_anchor))
    rng = np.random.default_rng(S_GM1)
    null_meds = []
    for i in range(2000):
        pi = rng.permutation(len(anchors))
        Xp = Xa[pi]
        Pp = loo_raw_from_hat(H, Xp)
        null_meds.append(float(np.median(np.linalg.norm((Pp - Xp) / sig, axis=1))))
        if (i + 1) % 500 == 0:
            print(f"      G-M1 null {i + 1}/2000")
    p_gm1 = float((1 + sum(m <= obs_med for m in null_meds)) / 2001)
    V["G_M1"] = {"obs": round(obs_med, 4), "null_mean": round(float(np.mean(null_meds)), 4),
                 "p": round(p_gm1, 5), "pass": bool(p_gm1 < 0.05)}
    print(f"  §1 G-M1 re-verify: obs {obs_med:.3f} vs null {np.mean(null_meds):.3f} "
          f"→ p={p_gm1:.5f} [{'PASS' if p_gm1 < .05 else 'FAIL — NO_VERDICT'}]")

    # ── §2 M1: the wing reverse estimates ────────────────────────────────────
    W_rev = reverse_fit(Ya14, Xa, LAM_REV)
    Xw_hat = reverse_predict(W_rev, Dw14)
    Xw_hat20 = reverse_predict(reverse_fit(Ya20, Xa, LAM_REV), Dw20)
    Xw_hatB = reverse_predict(reverse_fit(Yb14, Xa, LAM_REV), Dwb14)
    print("  §2 M1 estimates (instilled L14, λ=1.0, full-320 fit):")
    dz_wing = np.linalg.norm((Xw_hat - Xw_hand) / sig, axis=1)
    for c, d in zip(L.CHOICE_SET, dz_wing):
        pct = float(np.mean(dz_anchor <= d))
        print(f"      {c:<14s} Δstd {d:6.3f}  (anchor-LOO pctile {pct:5.1%})")

    # ── §3 P-M primary + secondary floor ─────────────────────────────────────
    cen = Xw_hand.mean(0)
    d_region = np.linalg.norm((Xa - cen) / sig, axis=1)
    near = d_region <= np.percentile(d_region, 20)
    floor_near = dz_anchor[near]
    obs_rs, p_pm = ranksum_perm(dz_wing, floor_near, 10000, S_PM)
    k_above = int((dz_wing > np.median(dz_anchor)).sum())
    p_binom = float(sum(math.comb(13, i) for i in range(k_above, 14)) / 2 ** 13)
    V["P_M"] = {"wing_med": round(float(np.median(dz_wing)), 4),
                "floor_near_med": round(float(np.median(floor_near)), 4),
                "floor_full_med": round(float(np.median(dz_anchor)), 4),
                "n_near": int(near.sum()), "p_primary": round(p_pm, 5),
                "primary_pass": bool(p_pm < 0.05),
                "k_above_full_median": k_above, "p_binomial": round(p_binom, 5),
                "secondary_pass": bool(k_above >= 10)}
    print(f"  §3 P-M: wing Δstd med {np.median(dz_wing):.3f} vs region floor "
          f"{np.median(floor_near):.3f} (n={near.sum()}) → p={p_pm:.5f} "
          f"[{'MIS-ENCODING' if p_pm < .05 else 'within noise'}]")
    print(f"      secondary (full floor {np.median(dz_anchor):.3f}): k={k_above}/13 "
          f"above median, binomial p={p_binom:.4f} "
          f"[{'pass' if k_above >= 10 else 'fail'}]")

    # ── §4 S-M1 axis profile ─────────────────────────────────────────────────
    Z = (Xw_hat - Xw_hand) / sig
    V["S_M1"] = {n: {"mean_z": round(float(Z[:, j].mean()), 3),
                     "mean_abs_z": round(float(np.abs(Z[:, j]).mean()), 3)}
                 for j, n in enumerate(L.AXIS_NAMES)}
    prof = sorted(V["S_M1"].items(), key=lambda kv: -kv[1]["mean_abs_z"])
    print("  §4 S-M1 axis profile (top |z|): " + "  ".join(
        f"{n}:{d['mean_abs_z']:.2f}({d['mean_z']:+.2f})" for n, d in prof[:6]))

    # ── §5 S-M2 Procrustes systematicity ─────────────────────────────────────
    obs_proc = procrustes_resid(Xw_hat / sig, Xw_hand / sig)
    rng = np.random.default_rng(S_PROC)
    null_proc = [procrustes_resid((Xw_hat / sig)[rng.permutation(13)],
                                  Xw_hand / sig) for _ in range(2000)]
    p_proc = float((1 + sum(r <= obs_proc for r in null_proc)) / 2001)
    V["S_M2"] = {"resid": round(obs_proc, 4),
                 "null_mean": round(float(np.mean(null_proc)), 4),
                 "p": round(p_proc, 5)}
    print(f"  §5 S-M2 Procrustes: resid {obs_proc:.3f} vs null "
          f"{np.mean(null_proc):.3f} → p={p_proc:.4f} "
          f"[{'coherent transform' if p_proc < .05 else 'incoherent'}]")

    # ── §6 S-M3 cloud diagnostics + RSA ──────────────────────────────────────
    er_hat, er_hand = effrank(Xw_hat / sig), effrank(Xw_hand / sig)
    Dw_ang = L.pairwise_ang(Dw14)
    rho_h, p_h = L.mantel_spearman(L.pairwise_ang(Xw_hand), Dw_ang, 2000, S_RSA_H,
                                   progress_every=0)
    rho_e, p_e = L.mantel_spearman(L.pairwise_ang(Xw_hat), Dw_ang, 2000, S_RSA_E,
                                   progress_every=0)
    V["S_M3"] = {"effrank_hat": round(er_hat, 2), "effrank_hand": round(er_hand, 2),
                 "rsa_hand_dirs": {"rho": round(rho_h, 4), "p": round(p_h, 5)},
                 "rsa_hat_dirs_TEXTURE": {"rho": round(rho_e, 4), "p": round(p_e, 5)}}
    print(f"  §6 S-M3: eff-rank hat {er_hat:.2f} vs hand {er_hand:.2f} | "
          f"RSA hand↔dirs ρ={rho_h:+.3f} p={p_h:.3f} | "
          f"hat↔dirs ρ={rho_e:+.3f} p={p_e:.3f} (texture, by construction)")

    # ── §7 S-M4 depth replication ────────────────────────────────────────────
    A14, A20 = (Xw_hat - xbar) / sig, (Xw_hat20 - xbar) / sig
    cos_dep = np.sum(A14 * A20, 1) / (np.linalg.norm(A14, axis=1)
                                      * np.linalg.norm(A20, axis=1))
    ax_r = [float(np.corrcoef(Xw_hat[:, j], Xw_hat20[:, j])[0, 1]) for j in range(14)]
    V["S_M4"] = {"cos_median": round(float(np.median(cos_dep)), 3),
                 "axis_r_median": round(float(np.median(ax_r)), 3)}
    print(f"  §7 S-M4 L20 replication: per-concept cos med "
          f"{np.median(cos_dep):.3f} | per-axis r med {np.median(ax_r):.3f}")

    # ── §8 S-M5 base negative control ────────────────────────────────────────
    spread_b = float(np.median(Xw_hatB.std(0, ddof=1) / sig))
    spread_i = float(np.median(Xw_hat.std(0, ddof=1) / sig))
    V["S_M5"] = {"spread_base": round(spread_b, 3), "spread_inst": round(spread_i, 3)}
    print(f"  §8 S-M5 base control: spread ratio base {spread_b:.3f} "
          f"vs instilled {spread_i:.3f}")

    # ── §10 G-M2 (stage gate) — A5 verbatim on re-encoded coords ─────────────
    Wf, Pw, own_w, nearest_w, hits_w = L.wing_transfer(Xa, Ya14, Xw_hat, Dw14, L.LAM)
    null_h, _ = L.derangement_null_transfer(Wf, Xw_hat, Dw14, 2000, S_GM2)
    gm2 = L.gate_gb(hits_w, null_h)
    per_target = {c: {"angle": round(float(own_w[i]), 2),
                      "nearest": L.CHOICE_SET[int(nearest_w[i])],
                      "hit": bool(nearest_w[i] == i)}
                  for i, c in enumerate(L.CHOICE_SET)}
    V["G_M2"] = {**gm2, "median_angle": round(float(np.median(own_w)), 2),
                 "per_target": per_target,
                 "flown_A5_hand_coords": {"hits": 0, "median_angle": 89.8}}
    print(f"  §10 G-M2: hits {gm2['hits']}/13 (bar ≥4 AND >null p95 "
          f"{gm2['null_p95']}), median angle {np.median(own_w):.1f}° "
          f"(hand coords flew 0/13 @ 89.8°) → "
          f"[{'PASS — Part 2 stages' if gm2['pass'] else 'FAIL — FM2, no flight'}]")

    # ── §9 S-M6 shrinkage sensitivity (after G-M2 so it can recompute it) ────
    fac = sig / P_loo.std(0, ddof=1)
    ctr = P_loo.mean(0)
    Xw_vm = ctr + (Xw_hat - ctr) * fac
    dz_vm = np.linalg.norm((Xw_vm - Xw_hand) / sig, axis=1)
    _, p_pm_vm = ranksum_perm(dz_vm, floor_near, 10000, S_PM + 100)
    _, _, own_vm, nearest_vm, hits_vm = L.wing_transfer(Xa, Ya14, Xw_vm, Dw14, L.LAM)
    V["S_M6"] = {"p_primary_vm": round(p_pm_vm, 5), "gm2_hits_vm": int(hits_vm),
                 "gm2_med_vm": round(float(np.median(own_vm)), 2),
                 "note": "texture; raw estimator is the registered artifact"}
    print(f"  §9 S-M6 un-shrunk sensitivity: P-M p={p_pm_vm:.5f}, "
          f"G-M2 hits {hits_vm}/13 @ {np.median(own_vm):.1f}°")

    # ── §11 S-M7 modal concordance ───────────────────────────────────────────
    tr_i = [L.CHOICE_SET.index(t) for t in L.TRAINED]
    def nearest_trained(Xrows):
        out = {}
        for c in L.HELD_OUT:
            i = L.CHOICE_SET.index(c)
            d = [np.linalg.norm((Xrows[i] - Xrows[j]) / sig) for j in tr_i]
            out[c] = L.TRAINED[int(np.argmin(d))]
        return out
    nn_hat, nn_hand = nearest_trained(Xw_hat), nearest_trained(Xw_hand)
    m_hat = sum(nn_hat[c] == L.E8R2_MODAL[c] for c in L.HELD_OUT)
    m_hand = sum(nn_hand[c] == L.E8R2_MODAL[c] for c in L.HELD_OUT)
    V["S_M7"] = {"hat": nn_hat, "hand": nn_hand, "modal": dict(L.E8R2_MODAL),
                 "match_hat": int(m_hat), "match_hand": int(m_hand)}
    print(f"  §11 S-M7 modal concordance: hat {m_hat}/4 vs hand {m_hand}/4  "
          + "  ".join(f"{c}→{nn_hat[c]}" for c in L.HELD_OUT))

    # ── §12 S-M8 validator report (DB read-only) ─────────────────────────────
    db = sqlite3.connect(os.path.join(HERE, "..", "db", "semantic.db"))
    q = ",".join("?" * 13)
    rel = db.execute(
        f"SELECT c1.name,c2.name,r.rel_type FROM relations r "
        f"JOIN concepts c1 ON r.concept1_id=c1.id "
        f"JOIN concepts c2 ON r.concept2_id=c2.id "
        f"WHERE c1.name IN ({q}) AND c2.name IN ({q})",
        list(L.CHOICE_SET) * 2).fetchall()
    assert len(rel) == 5, f"intra-wing relations changed: {rel}"
    def core_ang(u, v, k=3):
        a, b = np.array(u[:k], float), np.array(v[:k], float)
        return float(np.degrees(np.arccos(np.clip(
            a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    sm8 = []
    for n1, n2, rt in rel:
        i, j = L.CHOICE_SET.index(n1), L.CHOICE_SET.index(n2)
        hand_a = core_ang(Xw_hand[i], Xw_hand[j])
        hat_a = core_ang(Xw_hat[i], Xw_hat[j])
        ok = (hat_a > 60.0) if rt == "opposition" else (60.0 <= hat_a <= 120.0)
        sm8.append({"pair": f"{n1}–{n2}", "type": rt,
                    "core3_hand": round(hand_a, 1), "core3_hat": round(hat_a, 1),
                    "band_ok_hat": bool(ok)})
        print(f"  §12 S-M8 {n1}–{n2} ({rt}): core3 hand {hand_a:.0f}° → "
              f"hat {hat_a:.0f}° [{'ok' if ok else 'OUT OF BAND'}]")
    V["S_M8"] = sm8

    # ── §13 fork + artifacts ─────────────────────────────────────────────────
    pm, sec, g2 = V["P_M"]["primary_pass"], V["P_M"]["secondary_pass"], gm2["pass"]
    if not V["G_M1"]["pass"]:
        fork = "NO_VERDICT (G-M1 instrument fail)"
    elif pm and g2:
        fork = "FM1 — mis-encoding measured + round-trip PASS → Part 2 STAGES"
    elif pm:
        fork = "FM2 — mis-encoding measured, round-trip fail → no flight; menu (b)/(c)"
    elif not sec:
        fork = "FM3 — within noise on both floors → adjudication weakens; menu (c)"
    else:
        fork = "FM4-region-fail-split — see registered reading"
    if pm != sec:
        fork += "  [FM4 floors split: region-matched primary operative]"
    V["fork"] = fork
    V["pins"] = {"lam_rev": LAM_REV, "atlas_stamp": dirs["stamp"],
                 "seeds": [S_GM1, S_GM2, S_PROC, S_RSA_H, S_RSA_E, S_PM]}
    V["secs"] = round(time.time() - t0, 1)

    art = {"what": "E8-J v2 wing re-encode artifact (FLIGHT INPUT, not a DB update)",
           "protocol": "docs/E8J2_PROTOCOL.md @ 28ad28c", "lam_rev": LAM_REV,
           "atlas": f"results_e8j/full_{dirs['stamp']}", "sigma": [round(s, 6) for s in sig],
           "xhat_L14": {c: [round(float(x), 6) for x in Xw_hat[i]]
                        for i, c in enumerate(L.CHOICE_SET)},
           "xhat_L20": {c: [round(float(x), 6) for x in Xw_hat20[i]]
                        for i, c in enumerate(L.CHOICE_SET)},
           "xhat_base14": {c: [round(float(x), 6) for x in Xw_hatB[i]]
                           for i, c in enumerate(L.CHOICE_SET)}}
    art["sha"] = sha_vecs(art["xhat_L14"])
    L.jdump(art, os.path.join(OUT, "wing_reencode_v1.json"))
    if g2:
        der = L.wing_derangement()
        dhat_real = {c: [round(float(x), 6) for x in Pw[i]]
                     for i, c in enumerate(L.CHOICE_SET)}
        pins = {"what": "E8J2 Part-2 flight pins: forward-bridge predicted dirs "
                        "from re-encoded coords (real arm); permuted arm = "
                        "dhat_real[derangement[target]]",
                "derangement": der, "dhat_real": dhat_real,
                "geometric_hits": {c: per_target[c]["hit"] for c in L.CHOICE_SET},
                "sha": sha_vecs(dhat_real)}
        L.jdump(pins, os.path.join(OUT, "rung_pins.json"))
        print(f"      Part-2 pins minted → results_e8j2/rung_pins.json "
              f"(sha {pins['sha']})")
    L.jdump(V, os.path.join(OUT, "e8j2_verdict.json"))
    print(f"\n  FORK: {fork}")
    print(f"  verdict → results_e8j2/e8j2_verdict.json  ({V['secs']}s)")


if __name__ == "__main__":
    main()
