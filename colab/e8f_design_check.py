#!/usr/bin/env python3
"""E8-F (L3 featural curriculum) design check — lane law: BEFORE registration,
BEFORE build.

Licensed by E8-J v2 fork FJ1: "L3 (featural curriculum) licensed in bottleneck
form" — train the readout ON bridge feature axes, composing per the algebra.

This check measures, on the LOCKED standing instruments only (E8-J atlas dirs,
e4 pack coords, E8-J v2 wing re-encoding), the load-bearing premises of that
flight BEFORE anything is registered:

  1. the featural ALPHABET — which of the 14 register axes the forward bridge
     carries with enough angular leverage for the injected-readout channel
     (the ~8-effective-dim question from the E8-J ledger, made concrete);
  2. the QUANTIZATION + CAPACITY of a spelling code over that alphabet
     (collision census over the sampling frame; exact-decode ceiling);
  3. the honest TRAIN/EVAL SPLIT of the 320 anchors and the bridge refit on
     the train side only (eval concepts out of fit, out of curriculum);
  4. the TRUE-DIR arm ceiling — per-axis sign readability of real substrate
     dirs on the honest split (the linear prior the semantics arm rides);
  5. carrier-vs-content structure, pair-composition discriminability, atom
     OOD-ness, wing texture (hand vs re-encoded spellings), base degeneracy.

NO wing reverse-reads, NO GPU, NO training here — pure numpy on artifacts.
Machinery identity is proven by exactness crosschecks against the FLOWN
E8-J atlas numbers (A5 wing transfer, A4 axis-wise reverse R²) and the
pinned E8-J v2 d-hat.

Run:  python3 colab/e8f_design_check.py
"""
import json, os, sqlite3, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

RES_J = os.path.join(HERE, "results_e8j", "full_20260824_1827")
RES_J2 = os.path.join(HERE, "results_e8j2")
OUT_DIR = os.path.join(HERE, "results_e8f")
DB = os.path.join(HERE, "..", "db", "semantic.db")

E8F_SEED = L.E8J_SEED + 60      # fresh stream, disjoint from E8-J (+0..5) and v2 (+40..)
LAM = 1.0                       # pinned, matching the flown bridge
LAM_SENS = [0.1, 10.0]
N_EVAL = 64                     # proposed eval split of the 320 (train = 256)
N_PERM_BITS = 2000              # pooled-bit permutation null (true-dir ceiling)
SIGMA_ATOM = [1.0, 2.0]         # atom leverage probe scales (units of frame SD)


def unit_rows(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def angles_rows(P, D):
    return L.angles_rowwise(P, D)


def fit_bridge(X, Y, lam=LAM):
    """Forward bridge coords->dirs. Returns W (15,h): rows 0..13 axis dirs, 14 intercept."""
    return L.ridge_fit(X, Y, lam)


def predict(W, X):
    return L.bridge_predict(W, X)


def spell(X, axes, lo, hi):
    """Quantized spelling over the chosen axes: -1/0/+1 per axis via per-axis
    (lo, hi) boundaries. Returns int8 array (n, len(axes))."""
    S = np.zeros((X.shape[0], len(axes)), np.int8)
    for k, j in enumerate(axes):
        S[:, k] = np.where(X[:, j] <= lo[j], -1, np.where(X[:, j] >= hi[j], 1, 0))
    return S


def reverse_sign_acc(Dtr, Xtr, Dev, Xev, axes, lam=LAM):
    """True-dir ceiling: reverse ridge dirs->coords fit on TRAIN, applied to
    EVAL true dirs; per-axis sign accuracy on |x| beyond its own median-|x|
    deadzone is NOT used here — plain sign(pred) vs sign(true) with zeros
    scored as misses on neither side excluded (report n per axis)."""
    Wr = L.ridge_fit(Dtr, Xtr, lam)
    P = np.hstack([Dev, np.ones((Dev.shape[0], 1))]) @ Wr
    out = {}
    for j in axes:
        t, p = Xev[:, j], P[:, j]
        mask = np.abs(t) > 1e-9
        n = int(mask.sum())
        acc = float(np.mean(np.sign(p[mask]) == np.sign(t[mask]))) if n else float("nan")
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[L.AXIS_NAMES[j]] = {"sign_acc": round(acc, 4), "n": n,
                                "r2": round(1.0 - ss_res / max(ss_tot, 1e-12), 4)}
    return out, P


# ── §0 self-tests + planted teeth ────────────────────────────────────────────
def self_tests():
    rng = np.random.default_rng(E8F_SEED)
    n, h = 80, 256
    # Tooth A: readable synthetic world — a real linear code is recovered.
    Xs = rng.normal(0.0, 0.3, size=(n, 14))
    Wt = rng.normal(size=(15, h))
    Wt[14] *= 6.0                                    # carrier-dominant, like real
    Ys = unit_rows(np.hstack([Xs, np.ones((n, 1))]) @ Wt + rng.normal(0, 0.02, (n, h)))
    tr, ev = np.arange(0, 64), np.arange(64, 80)
    Wf = fit_bridge(Xs[tr], Ys[tr])
    own = angles_rows(predict(Wf, Xs[ev]), Ys[ev])
    assert np.median(own) < 15.0, f"tooth A: transfer failed {np.median(own):.1f}"
    accs, _ = reverse_sign_acc(Ys[tr], Xs[tr], Ys[ev], Xs[ev], list(range(14)))
    mean_acc = np.mean([v["sign_acc"] for v in accs.values()])
    # deranged codebook: same machinery, permuted rows -> chance
    pi = rng.permutation(len(ev))
    while np.any(pi == np.arange(len(ev))):
        pi = rng.permutation(len(ev))
    accd, _ = reverse_sign_acc(Ys[tr], Xs[tr], Ys[ev][pi], Xs[ev], list(range(14)))
    mean_accd = np.mean([v["sign_acc"] for v in accd.values()])
    assert mean_acc > 0.70, f"tooth A: readable world reads {mean_acc:.2f}"
    assert mean_acc - mean_accd > 0.12, \
        f"tooth A: no separation ({mean_acc:.2f} vs {mean_accd:.2f})"
    assert mean_accd < 0.62, f"tooth A: derangement not at chance {mean_accd:.2f}"
    # Tooth B: no-correspondence world — machinery alive, reading dead.
    Yr = unit_rows(rng.normal(size=(n, h)))
    accr, _ = reverse_sign_acc(Yr[tr], Xs[tr], Yr[ev], Xs[ev], list(range(14)))
    mean_accr = np.mean([v["sign_acc"] for v in accr.values()])
    assert mean_accr < 0.62, f"tooth B: random world reads {mean_accr:.2f}"
    Wr_ = fit_bridge(Xs[tr], Yr[tr])
    b = Wr_[14] / np.linalg.norm(Wr_[14])
    lev = ang(unit_rows((Wr_[14] + 0.3 * Wr_[0])[None, :])[0], b)
    assert lev > 0.0, "tooth B: zero leverage in random world"
    # spelling/scoring identity
    lo = {j: -0.05 for j in range(14)}; hi = {j: 0.05 for j in range(14)}
    S = spell(np.array([[0.2, -0.2, 0.0] + [0.0] * 11]), [0, 1, 2], lo, hi)
    assert S.tolist() == [[1, -1, 0]], S.tolist()
    print("  §0 self-tests + planted teeth: PASS "
          f"(readable {mean_acc:.2f} / deranged {mean_accd:.2f} / random {mean_accr:.2f})")


def main():
    t0 = time.time()
    print("E8-F (L3) DESIGN CHECK — featural alphabet, bottleneck form")
    print("=" * 66)
    self_tests()
    out = {"what": "E8-F design check", "seed": E8F_SEED, "lam": LAM}

    # ── §1 load + asserts ────────────────────────────────────────────────────
    dirs = json.load(open(os.path.join(RES_J, "atlas_dirs.json")))
    shipped = json.load(open(os.path.join(RES_J, "atlas.json")))
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    reenc = json.load(open(os.path.join(RES_J2, "wing_reencode_v1.json")))
    pins2 = json.load(open(os.path.join(RES_J2, "rung_pins.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    DESC = {c["name"]: c.get("desc") or "" for c in pack["concepts"]}
    anchors = dirs["anchors"]
    assert anchors == list(L.PINNED_ANCHORS), "anchor list drifted vs pins"
    assert shipped["anchors_sha"] == L.ANCHORS_SHA
    frame = L.anchor_frame(pack["concepts"])
    assert set(anchors) <= set(frame)
    Xa = np.array([VEC[a] for a in anchors], float)
    Ya14 = unit_rows(np.array([dirs["inst14"][a] for a in anchors], float))
    Ya20 = unit_rows(np.array([dirs["inst20"][a] for a in anchors], float))
    Yb14 = unit_rows(np.array([dirs["base14"][a] for a in anchors], float))
    Xw_hand = np.array([VEC[w] for w in L.CHOICE_SET], float)
    Dw14 = unit_rows(np.array([dirs["inst14"][w] for w in L.CHOICE_SET], float))
    Xw_re = np.array([reenc["xhat_L14"][w] for w in L.CHOICE_SET], float)
    XF = np.array([VEC[c] for c in frame], float)          # frame coords (universe)
    sig = XF.std(0, ddof=1)
    con = sqlite3.connect(DB)
    tri = dict(con.execute("SELECT name, trigram FROM concepts").fetchall())
    con.close()
    print(f"  §1 loaded: {len(anchors)} anchors (sha ok), frame {len(frame)}, "
          f"hidden {Ya14.shape[1]}D, wing-13 + re-encoding present")

    # ── §2 machinery-identity crosschecks vs the FLOWN atlas ────────────────
    W320 = fit_bridge(Xa, Ya14)
    _, _, own_w, _, hits_w = L.wing_transfer(Xa, Ya14, Xw_hand, Dw14, LAM)
    a5 = shipped["A5_wing"]["per_target"]
    for i, w in enumerate(L.CHOICE_SET):
        assert abs(float(own_w[i]) - float(a5[w]["angle"])) < 0.05, \
            (w, own_w[i], a5[w]["angle"])
    assert hits_w == sum(1 for w in a5 if a5[w]["hit"]), hits_w
    a4_re = L.axiswise_reverse_r2(Ya14, Xa)
    a4_fl = shipped["A4_axiswise_r2"]
    a4_key = "inst14" if isinstance(a4_fl.get("inst14"), dict) else None
    a4_cmp = a4_fl[a4_key] if a4_key else a4_fl
    for k in L.AXIS_NAMES:
        assert abs(a4_re[k] - float(a4_cmp[k])) < 5e-4, (k, a4_re[k], a4_cmp[k])
    dhat_re = predict(W320, Xw_re)
    dhat_pin = np.array([pins2["dhat_real"][w] for w in L.CHOICE_SET], float)
    dhat_pin = unit_rows(dhat_pin)
    dmax = float(np.max(angles_rows(dhat_re, dhat_pin)))
    assert dmax < 0.01, f"pinned d-hat mismatch {dmax}"
    print(f"  §2 exactness: A5 wing {np.median(own_w):.1f}deg/{hits_w} hits == flown; "
          f"A4 axes == flown; E8-J v2 d-hat reproduced ({dmax:.4f}deg max)")
    out["s2_exactness"] = {"a5_median": round(float(np.median(own_w)), 2),
                           "a5_hits": hits_w, "dhat_max_deg": round(dmax, 5)}

    # ── §3 the honest split + train-side bridge ─────────────────────────────
    rng = np.random.default_rng(E8F_SEED + 1)
    strata = {}
    for a in anchors:
        strata.setdefault(tri.get(a) or "_", []).append(a)
    keys = sorted(strata)
    quota = {k: N_EVAL * len(strata[k]) / len(anchors) for k in keys}
    take = {k: max(1, int(round(quota[k]))) for k in keys}
    while sum(take.values()) > N_EVAL:
        k = max((k for k in keys if take[k] > 1), key=lambda k: take[k] - quota[k])
        take[k] -= 1
    while sum(take.values()) < N_EVAL:
        k = max(keys, key=lambda k: quota[k] - take[k])
        take[k] += 1
    eval_names = []
    for k in keys:
        eval_names.extend(sorted(rng.choice(sorted(strata[k]), size=take[k],
                                            replace=False).tolist()))
    eval_names = sorted(eval_names)
    train_names = [a for a in anchors if a not in set(eval_names)]
    ev = np.array([anchors.index(a) for a in eval_names])
    tr = np.array([anchors.index(a) for a in train_names])
    Wtr = fit_bridge(Xa[tr], Ya14[tr])
    Pev = predict(Wtr, Xa[ev])
    own_ev = angles_rows(Pev, Ya14[ev])
    cos = np.clip(Pev @ Ya14[ev].T, -1, 1)
    top1 = int(np.sum(np.argmin(np.degrees(np.arccos(cos)), 1) == np.arange(len(ev))))
    sens = {}
    for lam in LAM_SENS:
        sens[lam] = round(float(np.median(angles_rows(
            predict(fit_bridge(Xa[tr], Ya14[tr], lam), Xa[ev]), Ya14[ev]))), 2)
    print(f"  §3 split {len(tr)}/{len(ev)} (stratified, seed {E8F_SEED + 1}): "
          f"eval transfer median {np.median(own_ev):.1f}deg, top-1 {top1}/64 "
          f"| lam-sens {sens}")
    out["s3_split"] = {"eval": eval_names, "n_train": len(tr),
                       "transfer_median": round(float(np.median(own_ev)), 2),
                       "top1": top1, "lam_sens": {str(k): v for k, v in sens.items()},
                       "strata_take": take}

    # ── §4 the alphabet: per-axis leverage through the train bridge ─────────
    b = Wtr[14]
    bu = b / np.linalg.norm(b)
    axis_rows = Wtr[:14]
    lev = {}
    print("  §4 alphabet (leverage = deg the atom moves the injected dir off carrier):")
    print("      axis |Wj|    sigma  lev1s  lev2s  flip(eval)  a4r2")
    flips = {}
    for j, name in enumerate(L.AXIS_NAMES):
        l1 = ang(unit_rows((b + SIGMA_ATOM[0] * sig[j] * axis_rows[j])[None, :])[0], bu)
        l2 = ang(unit_rows((b + SIGMA_ATOM[1] * sig[j] * axis_rows[j])[None, :])[0], bu)
        Xf = Xa[ev].copy(); Xf[:, j] = -Xf[:, j]
        fl = float(np.median(angles_rows(predict(Wtr, Xf), Pev)))
        flips[name] = fl
        lev[name] = {"w_norm": round(float(np.linalg.norm(axis_rows[j])), 3),
                     "sigma": round(float(sig[j]), 4),
                     "lev_1s": round(l1, 2), "lev_2s": round(l2, 2),
                     "flip_med": round(fl, 2), "a4_r2": a4_re[name]}
        print(f"      {name:>4s} {lev[name]['w_norm']:6.2f} {sig[j]:7.3f} "
              f"{l1:6.2f} {l2:6.2f} {fl:9.2f}  {a4_re[name]:+.3f}")
    WU = unit_rows(axis_rows)
    G = np.degrees(np.arccos(np.clip(WU @ WU.T, -1, 1)))
    ef_pairs = [(i, i + 7) for i in range(7)]
    ef_ang = {f"{L.AXIS_NAMES[i]}~{L.AXIS_NAMES[j]}": round(float(G[i, j]), 1)
              for i, j in ef_pairs}
    sv = np.linalg.svd(axis_rows, compute_uv=False)
    er_w = float(sv.sum() ** 2 / (sv ** 2).sum())
    print(f"      W-rows eff-rank {er_w:.2f}/14 | e~f angles {ef_ang}")
    out["s4_alphabet"] = {"axes": lev, "w_effrank": round(er_w, 2),
                          "ef_angles": ef_ang}

    # ── §5 carrier vs content ───────────────────────────────────────────────
    content = Xa[tr] @ axis_rows
    r_cn = np.linalg.norm(content, axis=1) / np.linalg.norm(b)
    a_pb = angles_rows(predict(Wtr, Xa[tr]), np.tile(bu, (len(tr), 1)))
    Pp = np.clip(Pev @ Pev.T, -1, 1)
    iu = np.triu_indices(len(ev), 1)
    spread = np.degrees(np.arccos(Pp[iu]))
    print(f"  §5 carrier: |b| {np.linalg.norm(b):.2f}; content/carrier median "
          f"{np.median(r_cn):.3f}; pred-vs-carrier median {np.median(a_pb):.1f}deg; "
          f"eval pred pairwise spread median {np.median(spread):.1f}deg "
          f"(p5 {np.percentile(spread, 5):.1f} / p95 {np.percentile(spread, 95):.1f})")
    out["s5_carrier"] = {"b_norm": round(float(np.linalg.norm(b)), 2),
                         "content_ratio_med": round(float(np.median(r_cn)), 4),
                         "pred_vs_carrier_med": round(float(np.median(a_pb)), 2),
                         "pred_spread_med": round(float(np.median(spread)), 2),
                         "pred_spread_p5": round(float(np.percentile(spread, 5)), 2),
                         "pred_spread_p95": round(float(np.percentile(spread, 95)), 2)}

    # ── §6 quantization + capacity over the frame ───────────────────────────
    print("  §6 quantization (frame universe):")
    qs = {}
    for j, name in enumerate(L.AXIS_NAMES):
        x = XF[:, j]
        t1, t2 = np.percentile(x, [33.333, 66.667])
        qs[name] = {"mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4),
                    "frac_dead_005": round(float(np.mean(np.abs(x) < 0.05)), 3),
                    "frac_dead_010": round(float(np.mean(np.abs(x) < 0.10)), 3),
                    "sign_pos": round(float(np.mean(x > 0)), 3),
                    "terciles": [round(float(t1), 4), round(float(t2), 4)]}
    out["s6_quant"] = qs
    lev_rank = sorted(L.AXIS_NAMES, key=lambda n: -lev[n]["lev_2s"])
    idx_of = {n: i for i, n in enumerate(L.AXIS_NAMES)}
    cap = {}
    for k_top in [4, 6, 8, 10, 14]:
        axes = [idx_of[n] for n in lev_rank[:k_top]]
        for scheme in ["tercile", "sign_dz01"]:
            if scheme == "tercile":
                lo = {j: qs[L.AXIS_NAMES[j]]["terciles"][0] for j in axes}
                hi = {j: qs[L.AXIS_NAMES[j]]["terciles"][1] for j in axes}
            else:
                lo = {j: -0.10 for j in axes}; hi = {j: 0.10 for j in axes}
            SF = spell(XF, axes, lo, hi)
            _, counts = np.unique(SF, axis=0, return_counts=True)
            uniq = float(np.mean(counts == 1))
            med_class = float(np.median(counts))
            Sev = spell(Xa[ev], axes, lo, hi)
            SFv = SF.astype(np.int16)
            self_top1 = 0
            for i, en in enumerate(eval_names):
                d = np.sum(SFv != Sev[i].astype(np.int16), axis=1)
                best = np.min(d)
                cands = [frame[t] for t in np.where(d == best)[0]]
                self_top1 += int(en in cands and len(cands) == 1)
            cap[f"top{k_top}_{scheme}"] = {
                "uniq_frac": round(uniq, 3), "median_bucket": med_class,
                "eval_unique_decode": self_top1}
    for k, v in cap.items():
        print(f"      {k:>18s}: unique {v['uniq_frac']:.0%} | median bucket "
              f"{v['median_bucket']:.0f} | eval-64 unique-decode {v['eval_unique_decode']}/64")
    out["s6_capacity"] = cap
    out["s6_lev_rank"] = lev_rank

    # ── §7 true-dir ceiling on the honest split ─────────────────────────────
    accs14, Prev = reverse_sign_acc(Ya14[tr], Xa[tr], Ya14[ev], Xa[ev],
                                    list(range(14)))
    accs20, _ = reverse_sign_acc(Ya20[tr], Xa[tr], Ya20[ev], Xa[ev],
                                 list(range(14)))
    print("  §7 true-dir ceiling (reverse read of EVAL true dirs, honest split):")
    print("      axis  L14-sign  L14-r2   L20-sign")
    for name in L.AXIS_NAMES:
        print(f"      {name:>4s}   {accs14[name]['sign_acc']:.3f}   "
              f"{accs14[name]['r2']:+.3f}   {accs20[name]['sign_acc']:.3f}")
    bits = np.concatenate([
        (np.sign(Prev[:, j]) == np.sign(Xa[ev][:, j]))[np.abs(Xa[ev][:, j]) > 1e-9]
        for j in range(14)])
    obs_bits = float(np.mean(bits))
    rng2 = np.random.default_rng(E8F_SEED + 2)
    null = []
    for _ in range(N_PERM_BITS):
        pi = rng2.permutation(len(ev))
        while np.any(pi == np.arange(len(ev))):
            pi = rng2.permutation(len(ev))
        Xp = Xa[ev][pi]
        nb = np.concatenate([
            (np.sign(Prev[:, j]) == np.sign(Xp[:, j]))[np.abs(Xp[:, j]) > 1e-9]
            for j in range(14)])
        null.append(float(np.mean(nb)))
    null = np.array(null)
    p_bits = float((np.sum(null >= obs_bits) + 1) / (N_PERM_BITS + 1))
    print(f"      pooled bit-acc {obs_bits:.4f} vs derangement null "
          f"{null.mean():.4f} (p {p_bits:.4f})")
    out["s7_truedir"] = {"L14": accs14, "L20": accs20,
                         "pooled_bits": round(obs_bits, 4),
                         "null_mean": round(float(null.mean()), 4),
                         "null_p95": round(float(np.percentile(null, 95)), 4),
                         "p": p_bits}

    # ── §8 pair-composition discriminability (top-leveraged axes) ───────────
    top6 = [idx_of[n] for n in lev_rank[:6]]
    seps = []
    for ii in range(len(top6)):
        for jj in range(ii + 1, len(top6)):
            i, j = top6[ii], top6[jj]
            corners = []
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = b + si * 2 * sig[i] * axis_rows[i] + sj * 2 * sig[j] * axis_rows[j]
                    corners.append(v / np.linalg.norm(v))
            C = np.array(corners)
            A = np.degrees(np.arccos(np.clip(C @ C.T, -1, 1)))
            seps.append(float(np.min(A[np.triu_indices(4, 1)])))
    print(f"  §8 pair 4-corner min-separation over top-6 axes: median "
          f"{np.median(seps):.1f}deg (min {np.min(seps):.1f}, max {np.max(seps):.1f})")
    out["s8_pairs"] = {"min_sep_median": round(float(np.median(seps)), 2),
                       "min_sep_min": round(float(np.min(seps)), 2),
                       "min_sep_max": round(float(np.max(seps)), 2)}

    # ── §9 wing texture: hand vs re-encoded spellings; continuity ───────────
    axes8 = [idx_of[n] for n in lev_rank[:8]]
    lo = {j: qs[L.AXIS_NAMES[j]]["terciles"][0] for j in axes8}
    hi = {j: qs[L.AXIS_NAMES[j]]["terciles"][1] for j in axes8}
    Sh = spell(Xw_hand, axes8, lo, hi)
    Sr = spell(Xw_re, axes8, lo, hi)
    ndiff = {L.CHOICE_SET[i]: int(np.sum(Sh[i] != Sr[i])) for i in range(13)}
    own_re = angles_rows(predict(Wtr, Xw_re), Dw14)
    own_hand = angles_rows(predict(Wtr, Xw_hand), Dw14)
    print(f"  §9 wing: hand-vs-reencoded spelling diffs (of 8 axes) "
          f"median {np.median(list(ndiff.values())):.0f} | Wtr wing transfer: "
          f"hand {np.median(own_hand):.1f}deg vs re-encoded {np.median(own_re):.1f}deg")
    out["s9_wing"] = {"spell_diffs": ndiff,
                      "wtr_hand_med": round(float(np.median(own_hand)), 2),
                      "wtr_reenc_med": round(float(np.median(own_re)), 2)}

    # ── §10 base degeneracy + atom OOD ──────────────────────────────────────
    accs_b, _ = reverse_sign_acc(Yb14[tr], Xa[tr], Yb14[ev], Xa[ev],
                                 [idx_of[n] for n in lev_rank[:6]])
    mean_b = float(np.mean([v["sign_acc"] for v in accs_b.values()]))
    atoms = []
    for j in [idx_of[n] for n in lev_rank[:8]]:
        for s in (-2, 2):
            v = b + s * sig[j] * axis_rows[j]
            atoms.append(v / np.linalg.norm(v))
    A_at = np.array(atoms)
    d_true = np.degrees(np.arccos(np.clip(A_at @ Ya14[tr].T, -1, 1))).min(1)
    d_pred = np.degrees(np.arccos(np.clip(
        A_at @ predict(Wtr, Xa[tr]).T, -1, 1))).min(1)
    print(f"  §10 base top-6 reverse sign-acc {mean_b:.3f} (instilled comparison "
          f"in §7) | atom OOD: nearest TRUE train dir median {np.median(d_true):.1f}deg, "
          f"nearest train PRED {np.median(d_pred):.1f}deg")
    out["s10"] = {"base_top6_signacc": round(mean_b, 4),
                  "atom_nearest_true_med": round(float(np.median(d_true)), 2),
                  "atom_nearest_pred_med": round(float(np.median(d_pred)), 2)}

    # ── §11 mu-centered atoms (the carrier point = register mean) ───────────
    mu_f = XF.mean(0)
    lo14 = {j: qs[L.AXIS_NAMES[j]]["terciles"][0] for j in range(14)}
    hi14 = {j: qs[L.AXIS_NAMES[j]]["terciles"][1] for j in range(14)}
    Smu = spell(mu_f[None, :], list(range(14)), lo14, hi14)[0]
    all_mid = bool(np.all(Smu == 0))
    pmu = predict(Wtr, mu_f[None, :])[0]
    lev_mu = {}
    for j, name in enumerate(L.AXIS_NAMES):
        xs = mu_f.copy(); xs[j] += 2 * sig[j]
        lev_mu[name] = round(ang(predict(Wtr, xs[None, :])[0], pmu), 2)
    # atom codes: displaced axis lands off-MID, others stay MID?
    clean = 0
    for j in range(14):
        for s in (-2.0, 2.0):
            xs = mu_f.copy(); xs[j] += s * sig[j]
            Sa = spell(xs[None, :], list(range(14)), lo14, hi14)[0]
            others_mid = np.all(np.delete(Sa, j) == np.delete(Smu, j))
            own_moved = Sa[j] != 0
            clean += int(others_mid and own_moved)
    print(f"  §11 mu-centered: mu spells all-MID {all_mid} | clean atoms {clean}/28 "
          f"| 2s leverage around mu: x {lev_mu['x']} fx {lev_mu['fx']} "
          f"fe {lev_mu['fe']} (min {min(lev_mu.values())})")
    out["s11_mu"] = {"mu_all_mid": all_mid, "mu_spell": Smu.tolist(),
                     "clean_atoms": clean, "lev_mu_2s": lev_mu}

    # ── §6b decode margins at the 14-axis tercile code ──────────────────────
    SF14 = spell(XF, list(range(14)), lo14, hi14)
    Sev14 = spell(Xa[ev], list(range(14)), lo14, hi14)
    margins, self_top = [], 0
    for i, en in enumerate(eval_names):
        d = np.sum(SF14.astype(np.int16) != Sev14[i].astype(np.int16), axis=1)
        fi = frame.index(en)
        d_self = int(d[fi])
        d_other = int(np.min(np.delete(d, fi)))
        margins.append(d_other - d_self)
        self_top += int(d_self == 0 and d_other >= 1)
    margins = np.array(margins)
    print(f"  §6b decode margins (14-axis tercile): median {np.median(margins):.0f} "
          f"fields | >=2 for {int(np.sum(margins >= 2))}/64 | >=1 for "
          f"{int(np.sum(margins >= 1))}/64 | self-unique {self_top}/64")
    out["s6b_margins"] = {"median": float(np.median(margins)),
                          "ge2": int(np.sum(margins >= 2)),
                          "ge1": int(np.sum(margins >= 1)),
                          "self_unique": self_top}

    # ── §7b tercile-version linear prior for the true-dir arm ───────────────
    tacc, tnull = {}, {}
    rng3 = np.random.default_rng(E8F_SEED + 3)
    Sev_true = Sev14
    for j, name in enumerate(L.AXIS_NAMES):
        pj = Prev[:, j]
        pred_lvl = np.where(pj <= lo14[j], -1, np.where(pj >= hi14[j], 1, 0))
        tacc[name] = round(float(np.mean(pred_lvl == Sev_true[:, j])), 4)
        accs_n = []
        for _ in range(400):
            pi = rng3.permutation(len(ev))
            accs_n.append(float(np.mean(pred_lvl == Sev_true[pi, j])))
        tnull[name] = round(float(np.mean(accs_n)), 4)
    core6 = ["x", "y", "z", "fx", "fy", "fz"]
    print("  §7b true-dir tercile prior (linear, honest split) — acc vs perm-null:")
    for name in L.AXIS_NAMES:
        tag = " *" if name in core6 else ""
        print(f"      {name:>4s}  {tacc[name]:.3f} vs {tnull[name]:.3f}{tag}")
    print(f"      core-6 mean {np.mean([tacc[n] for n in core6]):.3f} vs null "
          f"{np.mean([tnull[n] for n in core6]):.3f}")
    out["s7b_tercile_prior"] = {"acc": tacc, "null": tnull,
                                "core6_mean": round(float(np.mean(
                                    [tacc[n] for n in core6])), 4),
                                "core6_null": round(float(np.mean(
                                    [tnull[n] for n in core6])), 4)}

    # ── §12 eval-row prior balance at the tercile code ──────────────────────
    bal = {L.AXIS_NAMES[j]: [int(np.sum(Sev14[:, j] == v)) for v in (-1, 0, 1)]
           for j in range(14)}
    worst = max(bal, key=lambda n: max(bal[n]))
    print(f"  §12 eval-64 tercile balance: worst axis {worst} {bal[worst]} "
          f"(chance ceiling {max(bal[worst]) / 64:.2f})")
    out["s12_balance"] = bal

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "design_check.json")
    L.jdump(out, path)
    print(f"  saved {path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
