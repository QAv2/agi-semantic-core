#!/usr/bin/env python3
"""E8-O pure-logic suite: pins, the operator (byte-equal to the design
check), row builders at BOTH mode constants, stimulus mint probes, and
statistical teeth — a leg-faithful world passes P-O1+P-O2, an ORDER-BLIND
world passes P-O1 but scores d≈0 on P-O2 (the FO2 discrimination), a
deranged world fails P-O1, NONE-flood drains, G-ID separates.

Run:  python3 colab/test_e8o_logic.py
"""
import hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
PAYLOAD = json.load(open(os.path.join(HERE, "e8f_payload.json")))
NB = json.load(open(os.path.join(HERE, "E8O_ORDERED_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"]]
DESIGN = open(os.path.join(HERE, "e8o_design_check.py")).read()

# ── §1 pins + notebook identity ─────────────────────────────────────────────
deg = {}
for a, b in L.PAIR24:
    deg[a] = deg.get(a, 0) + 1
    deg[b] = deg.get(b, 0) + 1
check("PAIR24: 24 unique pairs from eval-64, degree <= 2",
      len(L.PAIR24) == 24 and len({tuple(p) for p in L.PAIR24}) == 24
      and set(deg) <= set(L.EVAL64) and all(v <= 2 for v in deg.values()))
check("PAIR24 sha matches literal",
      hashlib.sha256("|".join(f"{a}+{b}" for a, b in L.PAIR24).encode()
                     ).hexdigest()[:16] == L.PAIR24_SHA)
check("IDENT16: 16 (concept, alpha) from eval-64",
      len(L.IDENT16) == 16
      and all(n in L.EVAL64 and a in (0.5, 1.0) for n, a in L.IDENT16))
check("CONTROL16 subset of train-256, disjoint from eval",
      len(L.CONTROL16) == 16 and set(L.CONTROL16) <= set(L.TRAIN256))
check("ODU_DESC: 16 profiles, house-style length, no pack collision",
      len(L.ODU_DESC) == 16
      and all(len(d) >= 40 for d in L.ODU_DESC.values())
      and not set(L.ODU_DESC) & set(VEC))
check("readout flight-of-record path pinned",
      L.READOUT_SRC == "e8f/inflight_20260824_2311/readout_real"
      and L.READOUT_SRC in CELLS[2] and "SEM / READOUT_SRC" in CELLS[3])
check("notebook logic cell == e8o_logic.py byte-verbatim",
      CELLS[2] == open(os.path.join(HERE, "e8o_logic.py")).read())
check("logic cell carries e8f_logic verbatim as its base",
      CELLS[2].startswith(open(os.path.join(HERE, "e8f_logic.py")).read()))
compose_slice = DESIGN[DESIGN.find("def compose_A"):DESIGN.find("def compose_B")]
check("compose_A byte-equal to the design check",
      compose_slice.rstrip() in CELLS[2])
for marker in ("def g2_probe", "def run_trial_feat", "def build_eval_model",
               "class Injector", "def compute_dirs", "def dirs_stability",
               "def po1_stats", "def po2_stats", "def gid_stats",
               "def span_residual", "def odu_structure_rsa"):
    check(f"notebook carries {marker}", any(marker in c for c in CELLS))

# ── §2 operator + codes + stimuli ───────────────────────────────────────────
a, b = L.PAIR24[0]
check("composed code = A leg-1 + B leg-2 (exact semantics)",
      L.composed_code(VEC[a], VEC[b])[:7] == L.code_levels(VEC[a])[:7]
      and L.composed_code(VEC[a], VEC[b])[7:] == L.code_levels(VEC[b])[7:])
PC = L.pair_codes(VEC)
check("pair_codes: 48 entries, flip = leg swap",
      len(PC) == 48 and all(PC[(i, 0)][:7] == L.code_levels(VEC[p[0]])[:7]
                            and PC[(i, 1)][:7] == L.code_levels(VEC[p[1]])[:7]
                            for i, p in enumerate(L.PAIR24)))
W5 = np.array(PAYLOAD["W"], float)
STIM = L.mint_stimuli_o(W5, VEC)
check("stimuli: 48 comps + 64 fulls + carrier, unit-norm",
      len([k for k in STIM if k.startswith("comp:")]) == 48
      and len([k for k in STIM if k.startswith("full:")]) == 64
      and all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in STIM.values()))
probes = json.loads(CELLS[3][CELLS[3].find("PAYLOAD_PROBES_O = json.loads(r'''")
                             + len("PAYLOAD_PROBES_O = json.loads(r'''"):
                             CELLS[3].find("''')")])
check("build probes reproduce from payload W",
      all(float(np.max(np.abs(STIM[k] - np.asarray(v, float)))) < 1e-4
          for k, v in probes.items()))

# ── §3 row builders at BOTH mode constants ──────────────────────────────────
rows_full = L.build_e8o_eval(smoke=False)
check("full eval counts", L.eval_counts_o(rows_full) == L.EXPECT_EVAL_O)
rows_smoke = L.build_e8o_eval(smoke=True)
check("smoke covers all blocks",
      set(L.eval_counts_o(rows_smoke)) == set(L.EXPECT_EVAL_O)
      and sum(L.eval_counts_o(rows_smoke).values()) <= 15)
check("po rows carry both orders per pair",
      all(sorted(r["order"] for r in rows_full
                 if r["block"] == "po" and r["pair_idx"] == i) == [0, 0, 1, 1]
          for i in range(24)))

# ── §4 statistical teeth ────────────────────────────────────────────────────
def with_parsed(rows, fn):
    out = []
    for r in rows:
        lv = fn(r)
        if lv is None:
            out.append({**r, "parsed": {"kind": "NONE", "levels": {},
                                        "n_fields": 0}})
        else:
            out.append({**r, "parsed": {
                "kind": "CODE",
                "levels": {L.AXIS_NAMES_F[j]: lv[j] for j in range(14)},
                "n_fields": 14}})
    return out

po_rows = [r for r in rows_full if r["block"] == "po"]
ident_rows = [r for r in rows_full if r["block"] == "ident"]
FR = L.frame_codes(PACK["concepts"])
CODES64 = {n: FR[n] for n in L.EVAL64}

faithful = with_parsed(po_rows, lambda r: PC[(r["pair_idx"], r["order"])])
p1 = L.po1_stats(faithful, PC, 2000, seed=7)
p2 = L.po2_stats(faithful, PC, 10000, seed=7)
check("tooth: leg-faithful world passes P-O1",
      p1["pooled_acc"] == 1.0 and p1["p"] <= L.P_CRIT_O
      and p1["n_axes_sig"] >= L.AX_MIN_O)
check("tooth: leg-faithful world passes P-O2 (all pairs positive)",
      p2["d_mean"] > 0.3 and p2["p"] <= L.P_CRIT_O
      and p2["n_pos"] == 24)
blind = with_parsed(po_rows, lambda r: PC[(r["pair_idx"], 0)])
b1 = L.po1_stats(blind, PC, 2000, seed=7)
b2 = L.po2_stats(blind, PC, 10000, seed=7)
check("tooth: ORDER-BLIND world still clears P-O1 (the FO2 shape)",
      b1["p"] <= L.P_CRIT_O and b1["pooled_acc"] > 0.6)
check("tooth: ORDER-BLIND world scores d ~ 0 on P-O2",
      abs(b2["d_mean"]) < 0.02 and b2["p"] > 0.05)
shift = with_parsed(po_rows,
                    lambda r: PC[((r["pair_idx"] + 7) % 24, r["order"])])
s1 = L.po1_stats(shift, PC, 2000, seed=7)
check("tooth: deranged world fails P-O1", s1["p"] > 0.05)
none_rows = with_parsed(po_rows, lambda r: None)
n1 = L.po1_stats(none_rows, PC, 2000, seed=7)
n2 = L.po2_stats(none_rows, PC, 2000, seed=7)
check("tooth: NONE-flood drains both primaries",
      n1["p"] > 0.5 and abs(n2["d_mean"]) < 1e-9)
gid_good = L.gid_stats(with_parsed(ident_rows,
                                   lambda r: CODES64[r["concept"]]), CODES64)
gid_bad = L.gid_stats(with_parsed(ident_rows, lambda r: [0] * 14), CODES64)
check("G-ID separates (perfect 1.0 pass / MID-flood fail)",
      gid_good["pass"] and gid_good["pooled_acc"] == 1.0
      and not gid_bad["pass"])

# rider instruments importable + design matrix sane
figs, S = L.odu_design_matrix()
check("rider: design matrix symmetric, diag 1, complements 0",
      np.allclose(S, S.T) and np.allclose(np.diag(S), 1.0)
      and S[0, 15] == 0.0)

# ── §5 rebuild byte-identical ───────────────────────────────────────────────
nb_bytes = open(os.path.join(HERE, "E8O_ORDERED_UI.ipynb"), "rb").read()
lg_bytes = open(os.path.join(HERE, "e8o_logic.py"), "rb").read()
subprocess.run([sys.executable, os.path.join(HERE, "build_e8o_notebook.py")],
               check=True, capture_output=True)
check("rebuild byte-identical (notebook + logic)",
      open(os.path.join(HERE, "E8O_ORDERED_UI.ipynb"), "rb").read() == nb_bytes
      and open(os.path.join(HERE, "e8o_logic.py"), "rb").read() == lg_bytes)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
