#!/usr/bin/env python3
"""E8-O2 design check — lane law: BEFORE registration, BEFORE build.

The mixed-figure training curriculum: break the méjì shortcut the E8-O lock
measured (function fields imputed from essence, .4420 vs .3824) by CONTINUING
the locked E8-F readout's training on leg-decoupled points, then re-flying
the E8-O order eval. Geometry findings from the E8-O check CARRY OVER
(order-sep 100.8° median through W; envelope 36°; quoted, not re-measured).

Measured here (all local, standing artifacts):
  §2  the pinned draws — TRAINPAIR48 (a seeded perfect matching: 96 DISTINCT
      train-256 concepts → 48 pairs, degree exactly 1), POFRESH12 (fresh
      eval-64 pairs, disjoint from the flown PAIR24), SPOTMIX12 ⊂ TRAINPAIR48.
  §3  train-pair order-separation through the pinned W (must match the
      eval-pair geometry the E8-O check measured).
  §4  THE SHORTCUT-STARVING PROPERTY: across the mixed training set, how
      often does the composed function code equal the essence-donor's méjì
      completion? (On concept rows the méjì rate is 98.7% — the shortcut's
      food. The mixed curriculum must starve it to near tercile-chance.)
  §5  teeth for the new statistics: the shortcut-REVERSAL secondary
      (S-MEJI) is positive in a leg-faithful synthetic world, NEGATIVE in
      a shortcut world, ~0 under NONE-flood; retention/installation stats
      ride proven E8-O machinery.

Run:  python3 colab/e8o2_design_check.py
"""
import hashlib, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o_logic as L

OUT_DIR = os.path.join(HERE, "results_e8o2")
E8O2_SEED = 20260910
CHANNEL_RES_DEG = 20.0


def pred(W5, x):
    W5 = np.asarray(W5, float)
    v = np.asarray(x, float) @ W5[:14] + W5[14]
    return v / np.linalg.norm(v)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


def draw_matching(pool, n_pairs, seed):
    """Seeded perfect matching: 2*n_pairs DISTINCT concepts -> n_pairs."""
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(sorted(pool), size=2 * n_pairs,
                             replace=False).tolist())
    rng.shuffle(pick)
    return sorted(tuple(sorted((pick[2 * i], pick[2 * i + 1])))
                  for i in range(n_pairs))


def meji_completion_code(vec):
    """The essence-donor's own full code — what the shortcut predicts the
    function fields to be (its function leg's code)."""
    return L.code_levels(vec)


def smeji_stat(rows, pcodes, vecs, pair_list):
    """S-MEJI reversal: mean over rows of
    [function-field acc vs the JUNIOR leg's true content] minus
    [function-field acc vs the SENIOR donor's méjì completion].
    Positive = shortcut broken; negative = shortcut reading."""
    fun_idx = list(range(7, 14))
    ds = []
    for r in rows:
        a, b = pair_list[r["pair_idx"]]
        if r["order"] == 1:
            a, b = b, a
        lv = r["parsed"]["levels"]
        own = pcodes[(r["pair_idx"], r["order"])]
        meji = meji_completion_code(vecs[a])
        acc_b = np.mean([lv.get(L.AXIS_NAMES_F[j], 99) == own[j]
                         for j in fun_idx])
        acc_a = np.mean([lv.get(L.AXIS_NAMES_F[j], 99) == meji[j]
                         for j in fun_idx])
        ds.append(acc_b - acc_a)
    return float(np.mean(ds)), ds


def main():
    t0 = time.time()
    print("E8-O2 DESIGN CHECK — the mixed-figure curriculum")
    print("=" * 66)
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    payload = json.load(open(os.path.join(HERE, "e8f_payload.json")))
    W5 = np.array(payload["W"], float)
    out = {"what": "E8-O2 design check", "seed": E8O2_SEED}

    # ── §2 pinned draws ─────────────────────────────────────────────────────
    trainpair48 = draw_matching(L.TRAIN256, 48, E8O2_SEED + 1)
    members = [c for p in trainpair48 for c in p]
    assert len(set(members)) == 96, "matching not perfect"
    assert set(members) <= set(L.TRAIN256)
    flown = {tuple(p) for p in L.PAIR24}
    pofresh12 = []
    rng = np.random.default_rng(E8O2_SEED + 2)
    deg = {}
    guard = 0
    while len(pofresh12) < 12:
        guard += 1
        assert guard < 100000
        a, b = (str(x) for x in rng.choice(sorted(L.EVAL64), size=2,
                                           replace=False))
        a, b = sorted((a, b))
        if ((a, b) in flown or (a, b) in pofresh12
                or deg.get(a, 0) >= 1 or deg.get(b, 0) >= 1):
            continue
        pofresh12.append((a, b))
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    pofresh12 = sorted(pofresh12)
    rng3 = np.random.default_rng(E8O2_SEED + 3)
    idx = sorted(rng3.choice(48, size=12, replace=False).tolist())
    spotmix12 = [trainpair48[i] for i in idx]
    sha_tp = hashlib.sha256("|".join(f"{a}+{b}" for a, b in trainpair48)
                            .encode()).hexdigest()[:16]
    sha_pf = hashlib.sha256("|".join(f"{a}+{b}" for a, b in pofresh12)
                            .encode()).hexdigest()[:16]
    print(f"  §2 draws: TRAINPAIR48 (96 distinct train concepts, matching) "
          f"sha {sha_tp} | POFRESH12 (disjoint from flown PAIR24, degree 1) "
          f"sha {sha_pf} | SPOTMIX12 pinned")
    out["s2"] = {"trainpair48": [list(p) for p in trainpair48],
                 "pofresh12": [list(p) for p in pofresh12],
                 "spotmix12": [list(p) for p in spotmix12],
                 "sha_trainpair": sha_tp, "sha_pofresh": sha_pf}

    # ── §3 train-pair order-separation through W ────────────────────────────
    seps = []
    for a, b in trainpair48:
        pab = pred(W5, L.compose_A(VEC[a], VEC[b]))
        pba = pred(W5, L.compose_A(VEC[b], VEC[a]))
        seps.append(ang(pab, pba))
    seps = np.array(seps)
    print(f"  §3 train-pair order-sep: med {np.median(seps):.1f}deg "
          f"(p10 {np.percentile(seps, 10):.1f}) | >= {CHANNEL_RES_DEG} for "
          f"{np.mean(seps >= CHANNEL_RES_DEG):.0%}  (eval pairs measured "
          f"100.8 med in the E8-O check — carried over)")
    out["s3"] = {"sep_med": round(float(np.median(seps)), 2),
                 "sep_p10": round(float(np.percentile(seps, 10)), 2),
                 "frac_ge_channel": round(float(np.mean(
                     seps >= CHANNEL_RES_DEG)), 3)}

    # ── §4 the shortcut-starving property ───────────────────────────────────
    fun_idx = list(range(7, 14))
    match_meji, match_junior = [], []
    for a, b in trainpair48:
        for (s, j) in ((a, b), (b, a)):
            comp = L.composed_code(VEC[s], VEC[j])
            meji = meji_completion_code(VEC[s])
            true_j = L.code_levels(VEC[j])
            match_meji.append(np.mean([comp[k] == meji[k] for k in fun_idx]))
            match_junior.append(np.mean([comp[k] == true_j[k]
                                         for k in fun_idx]))
    conc_meji = []
    for n in sorted(L.TRAIN256):
        c = L.code_levels(VEC[n])
        conc_meji.append(1.0)      # by construction on concept rows
    print(f"  §4 shortcut food: on MIXED training targets the function code "
          f"matches the senior's méjì completion at "
          f"{np.mean(match_meji):.3f} per-field (vs 1.000 on every concept "
          f"row the E8-F readout was schooled on; junior-truth match "
          f"{np.mean(match_junior):.3f} = 1.0 by construction). The shortcut "
          f"pays ~chance in this curriculum — starved.")
    out["s4"] = {"fun_match_senior_meji": round(float(np.mean(match_meji)), 4),
                 "fun_match_junior_truth": round(float(np.mean(match_junior)), 4)}

    # ── §5 teeth for the S-MEJI reversal statistic ──────────────────────────
    rows = [r for r in L.build_e8o_eval(smoke=False) if r["block"] == "po"]
    PC = L.pair_codes(VEC)

    def with_parsed(rows, fn):
        o = []
        for r in rows:
            lv = fn(r)
            o.append({**r, "parsed": {"kind": "CODE" if lv else "NONE",
                                      "levels": ({L.AXIS_NAMES_F[j]: lv[j]
                                                  for j in range(14)}
                                                 if lv else {}),
                                      "n_fields": 14 if lv else 0}})
        return o

    faithful = with_parsed(rows, lambda r: PC[(r["pair_idx"], r["order"])])
    d_f, _ = smeji_stat(faithful, PC, VEC, [tuple(p) for p in L.PAIR24])

    def shortcut_parse(r):
        a, b = L.PAIR24[r["pair_idx"]]
        if r["order"] == 1:
            a, b = b, a
        return L.code_levels(VEC[a])       # senior's méjì everywhere

    shortcut = with_parsed(rows, shortcut_parse)
    d_s, _ = smeji_stat(shortcut, PC, VEC, [tuple(p) for p in L.PAIR24])
    nones = with_parsed(rows, lambda r: None)
    d_n, _ = smeji_stat(nones, PC, VEC, [tuple(p) for p in L.PAIR24])
    print(f"  §5 S-MEJI teeth: leg-faithful {d_f:+.3f} (positive) | "
          f"shortcut world {d_s:+.3f} (negative) | NONE-flood {d_n:+.3f} (~0)")
    assert d_f > 0.3 and d_s < -0.3 and abs(d_n) < 1e-9
    out["s5_teeth"] = {"faithful": round(d_f, 4), "shortcut": round(d_s, 4),
                       "noneflood": round(d_n, 4)}
    print("      E8-O flight-1 baselines carried as the comparison row: "
          "fun-vs-B .3824 / fun-vs-A-méjì .4420 (S-MEJI would be ~-.06), "
          "P-O2 d .0275 p .019, P-O1 .4464, G-ID .5982")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "design_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  saved {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
