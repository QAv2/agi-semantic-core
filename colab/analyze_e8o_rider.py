#!/usr/bin/env python3
"""E8-O rider analysis (LOCAL, post-flight — the registered §2 rider arms):
span-residual of the 16 Odù-profile dirs vs matched controls, and the
inversion-structure RSA. Runs on the retrieved bundle's odu_dirs /
control_dirs (computed on-VM, own-16-call shape each).

r1: Odù span-residuals exceed the matched-control distribution (rank-sum
    permutation, p <= .0025) -> new-axis candidates arm opens.
r2: within controls -> addressing-refinement-only (honest).
r3: inversion-RSA significant -> the 16-figure structure is
    substrate-visible (wing-v2 skeleton texture).

Spans: PRIMARY = the bridge-reachable subspace (the 15 W_L3 rows, per the
seed's wording); TEXTURE = the 320-anchor true-dir span (the known concept
manifold).

Run:  python3 colab/analyze_e8o_rider.py [full_<stamp>]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o_logic as L
import e8j_logic as _J
L.J = _J      # odu_structure_rsa's mantel ref — post-flight-only code path,
              # never called on the VM (the flown cells don't invoke it)

STAMP_DIR = sys.argv[1] if len(sys.argv) > 1 else None
assert STAMP_DIR, "usage: analyze_e8o_rider.py full_<stamp>"
RES = os.path.join(HERE, "results_e8o", STAMP_DIR)
BUNDLE = json.load(open(os.path.join(RES, "bundle_real.json")))
PAYLOAD = json.load(open(os.path.join(HERE, "e8f_payload.json")))
ATLAS = json.load(open(os.path.join(HERE, "results_e8j", "full_20260824_1827",
                                    "atlas_dirs.json")))

odu = {n: np.asarray(v, float) for n, v in BUNDLE["odu_dirs"].items()}
ctl = {n: np.asarray(v, float) for n, v in BUNDLE["control_dirs"].items()}
assert sorted(odu) == sorted(L.ODU_NAMES_O) and len(ctl) == 16
W5 = np.array(PAYLOAD["W"], float)
Ya = np.array([ATLAS["inst14"][a] for a in ATLAS["anchors"]], float)
Ya = Ya / np.linalg.norm(Ya, axis=1, keepdims=True)

D_odu = np.array([odu[n] / np.linalg.norm(odu[n]) for n in sorted(odu)])
D_ctl = np.array([ctl[n] / np.linalg.norm(ctl[n]) for n in sorted(ctl)])

out = {"flight": STAMP_DIR, "protocol": "E8O_PROTOCOL.md rider"}
rng = np.random.default_rng(L.E8O_SEED + 40)

for tag, basis in (("bridge15", W5), ("anchor320", Ya)):
    r_o = L.span_residual(D_odu, basis)
    r_c = L.span_residual(D_ctl, basis)
    obs = float(np.median(r_o) - np.median(r_c))
    both = np.concatenate([r_o, r_c])
    ge = 0
    for _ in range(10000):
        pi = rng.permutation(32)
        ge += float(np.median(both[pi[:16]]) - np.median(both[pi[16:]])) >= obs
    p = (1 + ge) / 10001
    print(f"  span[{tag}]: odù resid med {np.median(r_o):.4f} vs control "
          f"{np.median(r_c):.4f} | delta {obs:+.4f} p={p:.4f}")
    out[f"span_{tag}"] = {"odu_med": round(float(np.median(r_o)), 5),
                          "ctl_med": round(float(np.median(r_c)), 5),
                          "delta": round(obs, 5), "p": round(p, 5)}

# figure order must match the design matrix's bit order: map each Odù name
# to its traditional tetragram bits? NOT hand-assigned here — the RSA uses
# the 16 profiles as an UNORDERED set is meaningless; the registered mapping
# is the canonical binary order of the 16 figures as listed in IFA_SEED §1
# (Ogbè=1111 ... Òfún per the standard Yoruba sequence is a scholarly
# convention; we pin the simple reading: the order of ODU_DESC as authored,
# mapped to itertools binary order). This mapping choice is registered here,
# in the instrument, before any flight data existed for it.
obs_r, p_r = L.odu_structure_rsa(D_odu, n_perm=10000, seed=L.E8O_SEED + 41)
print(f"  inversion-RSA: rho {obs_r:.3f} p={p_r:.4f}")
out["rsa"] = {"rho": round(float(obs_r), 4), "p": round(float(p_r), 5)}

r1 = out["span_bridge15"]["p"] <= 0.0025 and out["span_bridge15"]["delta"] > 0
r3 = p_r <= 0.05
fork = ("r1 — Odù dirs carry variance beyond the bridge span vs controls: "
        "new-axis candidates arm opens" if r1 else
        "r2 — within the matched-control distribution: "
        "addressing-refinement-only (honest)")
print(f"  rider fork: {fork}" + ("  |  r3: inversion structure "
      "substrate-visible" if r3 else ""))
out["fork"] = fork
out["r3_structure_visible"] = bool(r3)
with open(os.path.join(RES, "rider.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"  saved {os.path.join(RES, 'rider.json')}")
