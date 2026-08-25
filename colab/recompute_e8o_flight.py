#!/usr/bin/env python3
"""E8-O flight recompute (the law): execute the staged notebook's VERDICT
cell VERBATIM over the raw shipped bundle_real.json, diff against the
shipped e8o_verdict.json, and re-derive the primary ingredients with fresh
code. Then the LABELED POST-HOC mechanism row (no bar, lock-record only):
the méjì-shortcut test — on the po rows, score the FUNCTION-leg fields
against the essence-donor A's own function code (the "méjì completion")
instead of B's; if accuracy jumps, the readout learned to infer function
fields from essence content on the méjì-manifold training distribution,
and leg-decoupling exposed the shortcut.

Run:  python3 colab/recompute_e8o_flight.py [full_20260825_0123]
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o_logic as L

STAMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "full_20260825_0123"
RES = os.path.join(HERE, "results_e8o", STAMP_DIR)

nb = json.load(open(os.path.join(HERE, "E8O_ORDERED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8o_logic.py")).read()

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR = L.frame_codes(PACK["concepts"])
assert L.frame_codes_sha(FR) == L.FRAME_CODES_SHA
CODES64 = {n: FR[n] for n in L.EVAL64}
PC = L.pair_codes(VEC)

BUNDLE = json.load(open(os.path.join(RES, "bundle_real.json")))
SHIPPED = json.load(open(os.path.join(RES, "e8o_verdict.json")))
assert BUNDLE["payload_sha"] == L.PAYLOAD_SHA
assert BUNDLE["pair_sha"] == L.PAIR24_SHA

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
RE = json.load(open(Path(td) / "e8o_verdict.json"))

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

blocks = BUNDLE["eval_rows"]
po = blocks["po"]

def pooled_vs(rows, code_fn, axes_idx):
    tot = hit = 0
    for r in rows:
        lv = r["parsed"]["levels"]
        code = code_fn(r)
        for j in axes_idx:
            tot += 1
            hit += int(lv.get(L.AXIS_NAMES_F[j], 99) == code[j])
    return hit / tot

fresh = {
    "po1_pooled": round(pooled_vs(po, lambda r: PC[(r["pair_idx"],
                                                    r["order"])],
                                  list(range(14))), 4),
    "gid_pooled": round(pooled_vs(blocks["ident"],
                                  lambda r: CODES64[r["concept"]],
                                  list(range(14))), 4),
    "sham_claims": sum(1 for r in blocks["sham"]
                       if r["parsed"]["kind"] == "CODE"),
    "invalid": sum(1 for rs in blocks.values() for r in rs
                   if r["parsed"]["kind"] == "INVALID"),
}
expect = {
    "po1_pooled": SHIPPED["P_O1"]["pooled_acc"],
    "gid_pooled": SHIPPED["gates"]["g_id"]["pooled_acc"],
    "sham_claims": SHIPPED["gates"]["g5_sham"]["claims"],
    "invalid": SHIPPED["gates"]["g4_parse"]["invalid"],
}
ok = True
for k in fresh:
    m = abs(fresh[k] - expect[k]) < 1e-9 if isinstance(fresh[k], float) \
        else fresh[k] == expect[k]
    ok &= m
    print(f"  fresh {k}: {fresh[k]} vs shipped {expect[k]} "
          f"{'==' if m else 'MISMATCH'}")

# ── LABELED POST-HOC (no bar; mechanism row for the lock record) ────────────
ess_idx, fun_idx = list(range(7)), list(range(7, 14))
def donor_pair(r):
    a, b = L.PAIR24[r["pair_idx"]]
    return (a, b) if r["order"] == 0 else (b, a)

acc_fun_vs_B = pooled_vs(po, lambda r: PC[(r["pair_idx"], r["order"])],
                         fun_idx)
acc_fun_vs_Ameji = pooled_vs(
    po, lambda r: L.code_levels(VEC[donor_pair(r)[0]]), fun_idx)
acc_ess_vs_A = pooled_vs(po, lambda r: PC[(r["pair_idx"], r["order"])],
                         ess_idx)
acc_ess_vs_Bmeji = pooled_vs(
    po, lambda r: L.code_levels(VEC[donor_pair(r)[1]]), ess_idx)
print("\nPOST-HOC (labeled, no bar) — the méjì-shortcut test:")
print(f"  essence fields vs A (own):        {acc_ess_vs_A:.4f}")
print(f"  essence fields vs B's code:       {acc_ess_vs_Bmeji:.4f}")
print(f"  function fields vs B (own):       {acc_fun_vs_B:.4f}")
print(f"  function fields vs A's function   {acc_fun_vs_Ameji:.4f}"
      f"   <- shortcut reading if HIGH")
posthoc = {"ess_vs_A": round(acc_ess_vs_A, 4),
           "ess_vs_B": round(acc_ess_vs_Bmeji, 4),
           "fun_vs_B": round(acc_fun_vs_B, 4),
           "fun_vs_A_meji": round(acc_fun_vs_Ameji, 4)}
with open(os.path.join(RES, "posthoc_meji.json"), "w") as f:
    json.dump(posthoc, f, indent=1)

if not diffs and ok:
    print("\nRECOMPUTE CLEAN — shipped verdict reproduces verbatim; fresh "
          "tallies match; post-hoc row saved.")
    sys.exit(0)
print("\nRECOMPUTE FOUND DIFFERENCES — adjudicate before lock.")
sys.exit(1)
