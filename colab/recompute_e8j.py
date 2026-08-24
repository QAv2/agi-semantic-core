#!/usr/bin/env python3
"""E8-J recompute (the law): re-run the flown stage_a_verdict VERBATIM
(single-source e8j_logic — the exact code the notebook carries) over the raw
shipped dirs, diff every field against the shipped atlas.json, and adjudicate
the flagged S4 anomaly (base loo-median 8.92°) with cloud-degeneracy
diagnostics (effective rank, mu-baseline, pairwise spread).

Run:  python3 colab/recompute_e8j.py
"""
import io, contextlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

RES = os.path.join(HERE, "results_e8j", "full_20260824_1827")
dirs = json.load(open(os.path.join(RES, "atlas_dirs.json")))
shipped = json.load(open(os.path.join(RES, "atlas.json")))
pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in pack["concepts"]}

anchors = dirs["anchors"]
assert anchors == list(L.PINNED_ANCHORS), "anchor list drifted vs pins"

print("recomputing stage_a_verdict VERBATIM (full constants; several minutes "
      "on this CPU — progress prints below)...")
t0 = time.time()
atlas = L.stage_a_verdict(anchors, VEC, dirs["inst14"], dirs["inst20"],
                          dirs["base14"], smoke=False)
print(f"recompute done in {time.time() - t0:.0f}s "
      f"(VM took {shipped['secs_stageA']}s)")

# ── field-by-field diff vs shipped ───────────────────────────────────────────
SKIP = {"secs_stageA", "stamp", "pins", "gates_snapshot"}
def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP and not path:
                continue
            if k not in a:
                out.append(f"+{path}.{k} (shipped-only)")
            elif k not in b:
                out.append(f"-{path}.{k} (recompute-only)")
            else:
                out.extend(diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, float) or isinstance(b, float):
        if not np.isclose(float(a), float(b), rtol=0, atol=1e-9):
            out.append(f"{path}: recomputed {a!r} != shipped {b!r}")
    elif a != b:
        out.append(f"{path}: recomputed {a!r} != shipped {b!r}")
    return out

d = diff(atlas, shipped)
subst = [x for x in d if not x.startswith((".secs", "+.secs"))]
print(f"\n=== DIFF vs shipped atlas: {len(subst)} substantive differences ===")
for x in subst[:40]:
    print("  ", x)
if not subst:
    print("   NONE — the shipped atlas reproduces verbatim from raw dirs.")

# ── headline reproduction table ──────────────────────────────────────────────
print("\n=== headline rows (recomputed | shipped) ===")
rows = [
    ("A1 RSA rho/p", f"{atlas['A1_rsa_L14']['rho']}/{atlas['A1_rsa_L14']['p']}",
     f"{shipped['A1_rsa_L14']['rho']}/{shipped['A1_rsa_L14']['p']}"),
    ("A2 LOO med/null/p", f"{atlas['A2_loo']['median_angle']}/"
     f"{atlas['A2_loo']['null_median_mean']}/{atlas['A2_loo']['p_median']}",
     f"{shipped['A2_loo']['median_angle']}/"
     f"{shipped['A2_loo']['null_median_mean']}/{shipped['A2_loo']['p_median']}"),
    ("A3/P-A top1/p", f"{atlas['A3_retrieval']['top1']}/{atlas['P_A']['p']}",
     f"{shipped['A3_retrieval']['top1']}/{shipped['P_A']['p']}"),
    ("A5 hits/median", f"{atlas['A5_wing']['hits']}/"
     f"{atlas['A5_wing']['median_angle']}",
     f"{shipped['A5_wing']['hits']}/{shipped['A5_wing']['median_angle']}"),
    ("G-B pass", str(atlas['gb']['pass']), str(shipped['gb']['pass'])),
    ("S3 L20 rho/top1", f"{atlas['S3_L20']['rsa_rho']}/{atlas['S3_L20']['top1']}",
     f"{shipped['S3_L20']['rsa_rho']}/{shipped['S3_L20']['top1']}"),
    ("S4 base rho/loo/top1", f"{atlas['S4_base']['rsa_rho']}/"
     f"{atlas['S4_base']['loo_median']}/{atlas['S4_base']['top1']}",
     f"{shipped['S4_base']['rsa_rho']}/{shipped['S4_base']['loo_median']}/"
     f"{shipped['S4_base']['top1']}"),
]
for name, rc, sh in rows:
    flag = "" if rc == sh else "   <-- differs"
    print(f"  {name:22s} {rc:>28s} | {sh:<28s}{flag}")

# ── S4 anomaly adjudication: cloud degeneracy diagnostics ────────────────────
print("\n=== S4 adjudication: base-cloud degeneracy diagnostics ===")
def cloud_stats(dd, label):
    M = np.array([dd[a] for a in anchors], float)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    sv = np.linalg.svd(M - M.mean(0), compute_uv=False)
    er = (sv.sum()) ** 2 / (sv ** 2).sum()
    mu = M.mean(0)
    mu_n = mu / np.linalg.norm(mu)
    mu_ang = np.degrees(np.arccos(np.clip(M @ mu_n, -1, 1)))
    G = np.clip(M @ M.T, -1, 1)
    iu = np.triu_indices(len(M), 1)
    pw = np.degrees(np.arccos(G[iu]))
    print(f"  {label}: eff-rank {er:.1f} | mean-dir norm {np.linalg.norm(mu):.4f} | "
          f"angle-to-mean med {np.median(mu_ang):.2f} "
          f"(q25 {np.percentile(mu_ang, 25):.1f} q75 {np.percentile(mu_ang, 75):.1f})")
    print(f"        pairwise spread: med {np.median(pw):.1f} "
          f"(q25 {np.percentile(pw, 25):.1f} q75 {np.percentile(pw, 75):.1f} "
          f"min {pw.min():.1f})")
    return np.median(mu_ang)

mu_base = cloud_stats(dirs["base14"], "base L14   ")
mu_inst = cloud_stats(dirs["inst14"], "instilled L14")
cloud_stats(dirs["inst20"], "instilled L20")
print(f"\n  verdict on the 8.92°: base LOO median ({shipped['S4_base']['loo_median']}) "
      f"vs base angle-to-mean median ({mu_base:.2f}) — if these are close, the "
      "bridge is predicting the COMMON COMPONENT, not codebook information; "
      "top-1 (8/320) is the informative base row.")
EOF
