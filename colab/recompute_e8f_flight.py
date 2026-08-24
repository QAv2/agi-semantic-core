#!/usr/bin/env python3
"""E8-F flight recompute (the law): execute the staged notebook's VERDICT
cell VERBATIM over the raw shipped bundle_real.json, then diff the result
against the shipped e8f_verdict.json field-by-field. Also re-derives the
independent ingredients (parse profile, sham claims, ppl delta, spot/P1/P3
pooled accuracies) with fresh code as a second, non-verdict-path check.

Run:  python3 colab/recompute_e8f_flight.py [full_20260824_2205]
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8f_logic as L

STAMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "full_20260824_2205"
RES = os.path.join(HERE, "results_e8f", STAMP_DIR)

nb = json.load(open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8f_logic.py")).read(), \
    "notebook logic cell != e8f_logic.py — recompute must ride the staged build"

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR_CODES = L.frame_codes(PACK["concepts"])
assert L.frame_codes_sha(FR_CODES) == L.FRAME_CODES_SHA
CODES64 = {n: FR_CODES[n] for n in L.EVAL64}

BUNDLE = json.load(open(os.path.join(RES, "bundle_real.json")))
SHIPPED = json.load(open(os.path.join(RES, "e8f_verdict.json")))
assert BUNDLE["payload_sha"] == L.PAYLOAD_SHA
mode_smoke = BUNDLE.get("mode") == "smoke"

td = tempfile.mkdtemp()
ns = {"np": np, "json": json, "Path": Path}
exec(LOGIC_SRC, ns)
ns.update(SMOKE=mode_smoke, MODE=BUNDLE.get("mode", "full"),
          STAMP=BUNDLE["stamp"], RESUME_STAMP="", OUT=Path(td),
          BUNDLE=BUNDLE, VEC=VEC, FR_CODES=FR_CODES, CODES64=CODES64,
          ship=lambda *a, **k: None)
buf = io.StringIO()
with redirect_stdout(buf):
    exec(VERDICT_SRC, ns)
RE = json.load(open(Path(td) / "e8f_verdict.json"))

# ── field-by-field diff, shipped vs recomputed ──────────────────────────────
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
            if not (abs(a - b) < 1e-12 or (np.isnan(a) and np.isnan(b))):
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))

walk(SHIPPED, RE)
diffs = [d for d in diffs if d[0] not in (".flight", ".resume_stamp")]
print(f"verdict-cell recompute: {len(diffs)} substantive diffs")
for p, a, b in diffs[:20]:
    print(f"  DIFF {p}: shipped {a} vs recomputed {b}")

# ── independent fresh-code tallies (non-verdict-path) ───────────────────────
blocks = BUNDLE["eval_rows"]
allrows = [r for rs in blocks.values() for r in rs]
inv = sum(1 for r in allrows if r["parsed"]["kind"] == "INVALID")
sham_claims = sum(1 for r in blocks["sham"] if r["parsed"]["kind"] == "CODE")
def pooled(rows, codes, axes):
    ax = [L.AXIS_NAMES_F.index(a) for a in axes]
    tot = hit = 0
    for r in rows:
        lv = r["parsed"]["levels"]
        true = codes[r["concept"]]
        for j in ax:
            tot += 1
            hit += int(lv.get(L.AXIS_NAMES_F[j], 99) == true[j])
    return hit / tot
spot_codes = {r["concept"]: FR_CODES[r["concept"]] for r in blocks["spot"]}
fresh = {
    "parse_invalid": inv,
    "sham_claims": sham_claims,
    "ppl_delta_pct": round(100 * (BUNDLE["ppl_post"] / BUNDLE["ppl_pre"] - 1), 4),
    "spot_pooled": round(pooled(blocks["spot"], spot_codes, L.AXIS_NAMES_F), 4),
    "p1_pooled": round(pooled(blocks["p1"], CODES64, L.AXIS_NAMES_F), 4),
    "p3_pooled": round(pooled(blocks["p3"], CODES64, L.CARRIED8), 4),
}
expect = {
    "parse_invalid": SHIPPED["gates"]["g4_parse"]["invalid"],
    "sham_claims": SHIPPED["gates"]["g5_sham"]["claims"],
    "ppl_delta_pct": SHIPPED["gates"]["g3_ppl"]["delta_pct"],
    "spot_pooled": SHIPPED["spot"]["pooled_acc"],
    "p1_pooled": SHIPPED["P1"]["pooled_acc"],
    "p3_pooled": SHIPPED["P3"]["pooled_acc"],
}
ok_fresh = True
for k in fresh:
    match = abs(fresh[k] - expect[k]) < 1e-9 if isinstance(fresh[k], float) \
        else fresh[k] == expect[k]
    ok_fresh &= match
    print(f"  fresh {k}: {fresh[k]} vs shipped {expect[k]} "
          f"{'==' if match else 'MISMATCH'}")

if not diffs and ok_fresh:
    print("\nRECOMPUTE CLEAN — shipped verdict reproduces verbatim and every "
          "fresh-code ingredient matches.")
    sys.exit(0)
print("\nRECOMPUTE FOUND DIFFERENCES — adjudicate before lock.")
sys.exit(1)
