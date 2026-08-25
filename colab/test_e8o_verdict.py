#!/usr/bin/env python3
"""Exercise the E8-O VERDICT cell VERBATIM from the built notebook against
synthetic flights: (1) leg-faithful world -> FO1; (2) order-blind world ->
FO2; (3) deranged world -> FO3; (4) G-ID garbage -> gates dirty;
(5) sham-claim flood -> gates dirty; (6) smoke fixture -> SMOKE GREEN;
(7) flight error -> incomplete banner, no verdict file. Full scenarios at
the REAL mode constants (N_PERM_O = 10000).

Run:  python3 colab/test_e8o_verdict.py
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8O_ORDERED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P-O2 ORDER" in VERDICT_SRC, "last cell is not the verdict"
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8o_logic.py")).read()

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR = L.frame_codes(PACK["concepts"])
CODES64 = {n: FR[n] for n in L.EVAL64}
PC = L.pair_codes(VEC)


def parsed_of(levels):
    if levels is None:
        return {"kind": "NONE", "levels": {}, "n_fields": 0}
    return {"kind": "CODE",
            "levels": {L.AXIS_NAMES_F[j]: int(levels[j]) for j in range(14)},
            "n_fields": 14}


def blocks_fixture(mode, smoke=False):
    rows = L.build_e8o_eval(smoke)
    blocks = {}
    for r in rows:
        b = r["block"]
        if b == "po":
            if mode == "fo1":
                lv = PC[(r["pair_idx"], r["order"])]
            elif mode == "fo2":
                lv = PC[(r["pair_idx"], 0)]
            else:
                lv = PC[((r["pair_idx"] + 7) % len(L.PAIR24), r["order"])] \
                    if not smoke else PC[(r["pair_idx"], r["order"])]
            p = parsed_of(lv)
        elif b == "ident":
            p = parsed_of([0] * 14 if mode == "gidfail"
                          else CODES64[r["concept"]])
        elif b == "sham":
            p = parsed_of(CODES64[sorted(L.EVAL64)[0]]
                          if mode == "shamflood" else None)
        elif b == "carrier":
            p = parsed_of([0] * 14)
        elif b == "odu":
            p = parsed_of([1] * 7 + [-1] * 7)
        blocks.setdefault(b, []).append(
            {**r, "response": "synthetic", "parsed": p, "hook_calls": 1})
    return blocks


def bundle_fixture(mode, smoke=False):
    if mode == "error":
        return {"condition": "real", "mode": "full", "stamp": "TEST",
                "error": "RuntimeError: boom"}
    return {"condition": "real", "mode": "smoke" if smoke else "full",
            "stamp": "TEST", "payload_sha": L.PAYLOAD_SHA,
            "pair_sha": L.PAIR24_SHA, "readout_src": L.READOUT_SRC,
            "g2_dirs": {"resid": 5e-08, "tol": 1e-4, "pass": True,
                        "layer": 14, "name": "UNCERTAINTY"},
            "mu14": 81.9,
            "odu_dirs": {n: [0.1] * 4 for n in L.ODU_NAMES_O},
            "control_dirs": {n: [0.1] * 4 for n in L.CONTROL16},
            "eval_rows": blocks_fixture(mode, smoke)}


def run_verdict(bundle, smoke=False):
    td = tempfile.mkdtemp()
    shipped = []
    ns = {"np": np, "json": json, "Path": Path}
    exec(LOGIC_SRC, ns)
    ns.update(SMOKE=smoke, MODE="smoke" if smoke else "full", STAMP="TEST",
              OUT=Path(td), BUNDLE=bundle, VEC=VEC, FR_CODES=FR,
              CODES64=CODES64, PCODES=PC,
              ship=lambda src, rel: shipped.append(str(rel)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(VERDICT_SRC, ns)
    vf = Path(td) / "e8o_verdict.json"
    verdict = json.load(open(vf)) if vf.exists() else None
    return verdict, buf.getvalue(), shipped


print("== scenario 1: FO1 (leg-faithful) ==")
v, out, sh = run_verdict(bundle_fixture("fo1"))
check("gates pass incl. G-ID", all(g.get("pass") for g in v["gates"].values()))
check("P-O1 passes all-14", v["P_O1"]["pass"] and v["P_O1"]["n_axes_sig"] == 14)
check("P-O2 passes, 24/24 positive",
      v["P_O2"]["pass"] and v["P_O2"]["n_pos"] == 24)
check("fork = FO1 (L4 lands)", v["fork"].startswith("FO1"))
check("verdict + bundle shipped", any("full_TEST" in s for s in sh))

print("== scenario 2: FO2 (order-blind) ==")
v2, out2, _ = run_verdict(bundle_fixture("fo2"))
check("P-O1 passes, P-O2 fails (d ~ 0)",
      v2["P_O1"]["pass"] and not v2["P_O2"]["pass"]
      and abs(v2["P_O2"]["d_mean"]) < 0.02)
check("fork = FO2 (reader symmetrizes)", v2["fork"].startswith("FO2"))

print("== scenario 3: FO3 (deranged) ==")
v3, out3, _ = run_verdict(bundle_fixture("fo3"))
check("P-O1 fails", not v3["P_O1"]["pass"])
check("fork = FO3", v3["fork"].startswith("FO3"))

print("== scenario 4: G-ID garbage -> gates dirty ==")
v4, out4, _ = run_verdict(bundle_fixture("gidfail"))
check("G-ID fails, fork = GATES DIRTY",
      not v4["gates"]["g_id"]["pass"] and v4["fork"].startswith("GATES DIRTY"))

print("== scenario 5: sham flood -> gates dirty ==")
v5, out5, _ = run_verdict(bundle_fixture("shamflood"))
check("G5 fails, fork = GATES DIRTY",
      not v5["gates"]["g5_sham"]["pass"]
      and v5["fork"].startswith("GATES DIRTY"))

print("== scenario 6: smoke GREEN ==")
v6, out6, _ = run_verdict(bundle_fixture("fo1", smoke=True), smoke=True)
check("smoke banner GREEN + smoke fork",
      "SMOKE GREEN" in out6 and v6["fork"].startswith("SMOKE"))

print("== scenario 7: flight error ==")
v7, out7, sh7 = run_verdict(bundle_fixture("error"))
check("incomplete banner, no verdict file",
      v7 is None and "FLIGHT INCOMPLETE" in out7 and not sh7)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
