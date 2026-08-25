#!/usr/bin/env python3
"""E8-O2 flight recompute (the law): execute the committed notebook's
VERDICT cell VERBATIM over the raw shipped bundle_real.json, diff against
the shipped e8o2_verdict.json, and re-derive every headline ingredient
with fresh hand-rolled code — P-M1 order delta, P-M2 function-leg
accuracy, S-MEJI reversal, S-FRESH transfer, S-ESS, G-ID retention,
G-INSTALL-M spot accuracy, sham/parse/ppl gates, and the before/after
table's "after" column. Seeded permutation p-values are covered by the
verbatim rerun (identical Generator streams cross-machine, proven at the
E8-J lock).

Run:  python3 colab/recompute_e8o2_flight.py [full_20260825_1815]
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o2_logic as L

STAMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "full_20260825_1815"
RES = os.path.join(HERE, "results_e8o2", STAMP_DIR)

nb = json.load(open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8o2_logic.py")).read()

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR = L.frame_codes(PACK["concepts"])
CODES64 = {n: FR[n] for n in L.EVAL64}
PC = L.pair_codes(VEC)
PCT = L.pair_codes_for(L.TRAINPAIR48, VEC)
PCF = L.pair_codes_for(L.POFRESH12, VEC)

BUNDLE = json.load(open(os.path.join(RES, "bundle_real.json")))
SHIPPED = json.load(open(os.path.join(RES, "e8o2_verdict.json")))
assert BUNDLE["payload_sha"] == L.PAYLOAD_SHA
assert BUNDLE["sha_trainpair"] == L.SHA_TRAINPAIR
assert BUNDLE["sha_pofresh"] == L.SHA_POFRESH
assert BUNDLE["readout_src"] == L.READOUT_SRC

# ── 1. the verdict cell, verbatim, over the raw bundle ──────────────────────
td = tempfile.mkdtemp()
ns = {"np": np, "json": json, "Path": Path}
exec(LOGIC_SRC, ns)
ns.update(SMOKE=BUNDLE.get("mode") == "smoke", MODE=BUNDLE.get("mode", "full"),
          STAMP=BUNDLE["stamp"], OUT=Path(td), BUNDLE=BUNDLE, VEC=VEC,
          FR_CODES=FR, CODES64=CODES64, PCODES=PC,
          ship=lambda *a, **k: None)
buf = io.StringIO()
with redirect_stdout(buf):
    exec(VERDICT_SRC, ns)
RE = json.load(open(Path(td) / "e8o2_verdict.json"))

diffs = []
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
    else:
        if isinstance(a, float) and isinstance(b, float):
            if not abs(a - b) < 1e-12:
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))

walk(SHIPPED, RE)
diffs = [d for d in diffs if d[0] not in (".flight",)]
print(f"verdict-cell recompute: {len(diffs)} substantive diffs")
for p, a, b in diffs[:10]:
    print(f"  DIFF {p}: shipped {a} vs recomputed {b}")

# ── 2. fresh hand-rolled tallies of every headline ingredient ───────────────
blocks = BUNDLE["eval_rows"]
ESS, FUN, ALL14 = list(range(7)), list(range(7, 14)), list(range(14))

def acc_vs(row, code, idx):
    lv = row["parsed"]["levels"]
    return sum(int(lv.get(L.AXIS_NAMES_F[j], 99) == code[j])
               for j in idx) / len(idx)

def pooled(rows, code_fn, idx):
    return float(np.mean([acc_vs(r, code_fn(r), idx) for r in rows]))

def order_d(rows, pcodes):
    # per-row d from INTEGER hit counts, divided once — the exact-rational
    # semantics po2_stats uses; divide-before-subtract floats miscount an
    # exactly-zero pair as positive at the 1e-17 level (pair 6 here).
    per = {}
    for r in rows:
        lv = r["parsed"]["levels"]
        own = pcodes[(r["pair_idx"], r["order"])]
        flip = pcodes[(r["pair_idx"], 1 - r["order"])]
        h_own = sum(int(lv.get(L.AXIS_NAMES_F[j], 99) == own[j])
                    for j in ALL14)
        h_flip = sum(int(lv.get(L.AXIS_NAMES_F[j], 99) == flip[j])
                     for j in ALL14)
        per.setdefault(r["pair_idx"], []).append((h_own - h_flip) / 14.0)
    dvec = [float(np.mean(v)) for _, v in sorted(per.items())]
    return float(np.mean(dvec)), sum(d > 0 for d in dvec), len(dvec)

def smeji_d(rows, pcodes, pair_list):
    ds = []
    for r in rows:
        a, b = pair_list[r["pair_idx"]]
        if r["order"] == 1:
            a, b = b, a
        own = pcodes[(r["pair_idx"], r["order"])]
        meji = L.code_levels(VEC[a])
        ds.append(acc_vs(r, own, FUN) - acc_vs(r, meji, FUN))
    return float(np.mean(ds))

po, pf = blocks["po"], blocks["po_fresh"]
pm1_d, pm1_pos, pm1_n = order_d(po, PC)
pf_d, _, _ = order_d(pf, PCF)
fresh = {
    "pm1_d_mean": round(pm1_d, 4),
    "pm1_n_pos": pm1_pos,
    "pm2_pooled": round(pooled(po, lambda r: PC[(r["pair_idx"], r["order"])],
                               FUN), 4),
    "smeji_d_mean": round(smeji_d(po, PC, L.PAIR24), 4),
    "sfresh_d_mean": round(pf_d, 4),
    "sfresh_fun_acc": round(pooled(pf, lambda r: PCF[(r["pair_idx"],
                                                      r["order"])], FUN), 4),
    "sess_po_ess": round(pooled(po, lambda r: PC[(r["pair_idx"], r["order"])],
                                ESS), 4),
    "gid_pooled": round(pooled(blocks["ident"],
                               lambda r: CODES64[r["concept"]], ALL14), 4),
    "spot_pooled": round(pooled(blocks["spot_mixed"],
                                lambda r: PCT[(r["pair_idx"], r["order"])],
                                ALL14), 4),
    "sham_claims": sum(1 for r in blocks["sham"]
                       if r["parsed"]["kind"] == "CODE"),
    "invalid": sum(1 for rs in blocks.values() for r in rs
                   if r["parsed"]["kind"] == "INVALID"),
    "ppl_delta_pct": round(100.0 * (BUNDLE["ppl_post"] - BUNDLE["ppl_pre"])
                           / BUNDLE["ppl_pre"], 4),
}
expect = {
    "pm1_d_mean": SHIPPED["P_M1"]["d_mean"],
    "pm1_n_pos": SHIPPED["P_M1"]["n_pos"],
    "pm2_pooled": SHIPPED["P_M2"]["pooled_acc"],
    "smeji_d_mean": SHIPPED["S_MEJI"]["d_mean"],
    "sfresh_d_mean": SHIPPED["S_FRESH"]["d_mean"],
    "sfresh_fun_acc": SHIPPED["S_FRESH"]["fun_acc"],
    "sess_po_ess": SHIPPED["S_ESS"]["po_ess_acc"],
    "gid_pooled": SHIPPED["gates"]["g_id"]["pooled_acc"],
    "spot_pooled": SHIPPED["gates"]["g_install_m"]["pooled_acc"],
    "sham_claims": SHIPPED["gates"]["g5_sham"]["claims"],
    "invalid": SHIPPED["gates"]["g4_parse"]["invalid"],
    "ppl_delta_pct": SHIPPED["gates"]["g3_ppl"]["delta_pct"],
}
ok = True
for k in fresh:
    m = abs(fresh[k] - expect[k]) < 1e-9 if isinstance(fresh[k], float) \
        else fresh[k] == expect[k]
    ok &= m
    print(f"  fresh {k}: {fresh[k]} vs shipped {expect[k]} "
          f"{'==' if m else 'MISMATCH'}")

# ── 3. the before/after table, re-derived ───────────────────────────────────
ba = SHIPPED["before_after"]
ba_ok = (abs(ba["po2_d"][1] - fresh["pm1_d_mean"]) < 1e-9
         and abs(ba["fun_acc"][1] - fresh["pm2_pooled"]) < 1e-9
         and abs(ba["smeji"][1] - fresh["smeji_d_mean"]) < 1e-9
         and abs(ba["gid"][1] - fresh["gid_pooled"]) < 1e-9
         and ba["po2_d"][0] == 0.0275 and ba["fun_acc"][0] == 0.3824
         and ba["smeji"][0] == -0.06 and ba["gid"][0] == 0.5982)
print(f"  before/after table: {'==' if ba_ok else 'MISMATCH'} "
      f"(order d {ba['po2_d'][0]} -> {ba['po2_d'][1]}, "
      f"fun {ba['fun_acc'][0]} -> {ba['fun_acc'][1]}, "
      f"smeji {ba['smeji'][0]} -> {ba['smeji'][1]}, "
      f"gid {ba['gid'][0]} -> {ba['gid'][1]})")

if not diffs and ok and ba_ok:
    print("\nRECOMPUTE CLEAN — shipped verdict reproduces verbatim; fresh "
          "tallies match; before/after table re-derived.")
    sys.exit(0)
print("\nRECOMPUTE FOUND DIFFERENCES — adjudicate before lock.")
sys.exit(1)
