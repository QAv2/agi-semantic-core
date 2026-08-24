#!/usr/bin/env python3
"""Exercise the E8-F VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first-runs after a long flight).

Scenarios: (1) FF2 world (P1+P3+P2 pass) -> fork FF2; (2) P3 NONE-flood ->
fork FF1/FF3; (3) installed but deranged P1 -> fork FF4; (4) spot at chance
-> NO_VERDICT (install); (5) sham FA flood -> gates dirty, primaries
withheld banner; (6) smoke fixture -> SMOKE GREEN; (7) flight error ->
incomplete banner, no verdict file; (8) resumed-eval bundle (ppl_pre None)
-> G3 passes with note. Full scenarios run at the REAL mode constants
(N_PERM_F = 10000); the smoke scenario at the smoke constants.

Run:  python3 colab/test_e8f_verdict.py
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8f_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "fork" in VERDICT_SRC and "G-INSTALL" in VERDICT_SRC, \
    "last cell is not the verdict"
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8f_logic.py")).read()

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR_CODES = L.frame_codes(PACK["concepts"])
CODES64 = {n: FR_CODES[n] for n in L.EVAL64}


def parsed_of(levels):
    if levels is None:
        return {"kind": "NONE", "levels": {}, "n_fields": 0}
    return {"kind": "CODE",
            "levels": {L.AXIS_NAMES_F[j]: int(levels[j]) for j in range(14)},
            "n_fields": 14}


def blocks_fixture(mode, smoke=False):
    """Attach parsed outputs per scenario world."""
    rows = L.build_e8f_eval(smoke)
    shift = {n: CODES64[sorted(L.EVAL64)[(i + 7) % 64]]
             for i, n in enumerate(sorted(L.EVAL64))}
    idx = {ax: j for j, ax in enumerate(L.AXIS_NAMES_F)}
    blocks = {}
    for r in rows:
        b = r["block"]
        if b == "spot":
            lv = (FR_CODES[r["concept"]] if mode != "notinstalled"
                  else shift.get(r["concept"],
                                 FR_CODES[sorted(L.TRAIN256)[0]]))
            if mode == "notinstalled":
                lv = FR_CODES[sorted(L.TRAIN256)[hash(r["concept"]) % 200]]
            p = parsed_of(lv)
        elif b == "p1":
            lv = (CODES64[r["concept"]] if mode not in ("ff4",)
                  else shift[r["concept"]])
            p = parsed_of(lv)
        elif b == "p3":
            p = parsed_of(CODES64[r["concept"]]
                          if mode in ("ff2", "resumed") else None)
        elif b == "wperm":
            p = parsed_of(L.perm_expected_levels(VEC[r["concept"]]))
        elif b == "carrier":
            p = parsed_of([0] * 14)
        elif b == "sham":
            p = parsed_of([0] * 14) if mode == "gatefail" else parsed_of(None)
        elif b == "atom":
            x = np.array(L.MU_FRAME, float)
            x[idx[r["axis"]]] += (L.ATOM_SCALE if r["sign"] == "+"
                                  else -L.ATOM_SCALE) * L.SIGMA_F[r["axis"]]
            p = parsed_of(L.code_levels(x))
        elif b == "titr":
            p = parsed_of(FR_CODES[r["concept"]])
        elif b == "wing":
            p = parsed_of(L.WING_RE_CODES[r["wing"]])
        blocks.setdefault(b, []).append(
            {**r, "response": "synthetic", "parsed": p, "hook_calls": 1})
    return blocks


def bundle_fixture(mode, smoke=False):
    if mode == "error":
        return {"condition": "real", "mode": "full", "stamp": "TEST",
                "error": "RuntimeError: boom"}
    b = {"condition": "real", "mode": "smoke" if smoke else "full",
         "stamp": "TEST", "payload_sha": L.PAYLOAD_SHA,
         "g2_dirs": {"resid": 1.2e-07, "tol": L.DIRS_TOL_F, "pass": True,
                     "layer": 14, "name": "UNCERTAINTY"},
         "mu14": 81.9, "ppl_pre": 19.63, "ppl_post": 19.98,
         "curriculum": dict(L.EXPECT_TRAIN_FULL),
         "train_log": {"epochs_flown": 4, "plateaued": True,
                       "strand_epoch_means": [], "opt_steps": 390,
                       "losses_every_10": [1.0]},
         "eval_rows": blocks_fixture(mode, smoke)}
    if mode == "resumed":
        b["ppl_pre"] = None
        b["train_log"] = {"resumed": True}
    return b


def run_verdict(bundle, smoke=False):
    td = tempfile.mkdtemp()
    shipped = []
    ns = {"np": np, "json": json, "Path": Path}
    exec(LOGIC_SRC, ns)
    ns.update(SMOKE=smoke, MODE="smoke" if smoke else "full", STAMP="TEST",
              RESUME_STAMP="", OUT=Path(td), BUNDLE=bundle, VEC=VEC,
              FR_CODES=FR_CODES, CODES64=CODES64,
              ship=lambda src, rel: shipped.append(str(rel)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(VERDICT_SRC, ns)
    out = buf.getvalue()
    vf = Path(td) / "e8f_verdict.json"
    verdict = json.load(open(vf)) if vf.exists() else None
    return verdict, out, shipped


print("== scenario 1: FF2 world ==")
v, out, sh = run_verdict(bundle_fixture("ff2"))
check("gates pass", all(g.get("pass") for g in v["gates"].values()))
check("G-INSTALL passes", v["g_install"]["pass"])
check("P1 passes with >=6 axes", v["P1"]["pass"]
      and v["P1"]["n_axes_sig"] >= L.P1_MIN_AXES)
check("P3 passes on carried-8", v["P3"]["pass"])
check("P2 passes (48 distinct ceiling)", v["P2"]["pass"]
      and v["P2"]["n_distinct"] == 48)
check("fork = FF2", v["fork"].startswith("FF2"))
check("verdict + bundle shipped", any("full_TEST" in s for s in sh))
check("S7 carrier all-MID", v["S7"]["all_mid"] == 4)
check("S2 second reading visible",
      v["S2"]["vs_permuted"]["pooled_acc"] == 1.0)
check("S5 wing matches re-encoded codes",
      all(d["match_re"] == 14 for d in v["S5"].values()))

print("== scenario 2: P3 NONE-flood ==")
v2, out2, _ = run_verdict(bundle_fixture("ff1"))
check("P1 passes, P3 fails", v2["P1"]["pass"] and not v2["P3"]["pass"])
check("fork = FF1/FF3", v2["fork"].startswith("FF1/FF3"))
check("S1 records the NONE-flood",
      v2["S1"]["profile"]["p3"]["NONE"] == 128
      and v2["S1"]["p3_cond_speech"]["n_spoke"] == 0)

print("== scenario 3: FF4 (installed, deranged P1) ==")
v3, out3, _ = run_verdict(bundle_fixture("ff4"))
check("install passes, P1 fails", v3["g_install"]["pass"]
      and not v3["P1"]["pass"])
check("fork = FF4", v3["fork"].startswith("FF4"))

print("== scenario 4: NO_VERDICT (install fail) ==")
v4, out4, _ = run_verdict(bundle_fixture("notinstalled"))
check("install fails", not v4["g_install"]["pass"])
check("fork = NO_VERDICT", v4["fork"].startswith("NO_VERDICT"))

print("== scenario 5: sham FA flood -> gates dirty ==")
v5, out5, _ = run_verdict(bundle_fixture("gatefail"))
check("G5 fails", not v5["gates"]["g5_sham"]["pass"])
check("fork = GATES DIRTY", v5["fork"].startswith("GATES DIRTY"))
check("banner says gates FAIL", "gates FAIL" in out5)

print("== scenario 6: smoke GREEN ==")
v6, out6, _ = run_verdict(bundle_fixture("ff2", smoke=True), smoke=True)
check("smoke banner GREEN", "SMOKE GREEN" in out6)
check("smoke fork tagged", v6["fork"].startswith("SMOKE"))

print("== scenario 7: flight error ==")
v7, out7, sh7 = run_verdict(bundle_fixture("error"))
check("incomplete banner, no verdict file",
      v7 is None and "FLIGHT INCOMPLETE" in out7 and not sh7)

print("== scenario 8: resumed-eval bundle ==")
v8, out8, _ = run_verdict(bundle_fixture("resumed"))
check("G3 passes with resumed note",
      v8["gates"]["g3_ppl"]["pass"]
      and "resumed" in v8["gates"]["g3_ppl"]["note"])
check("resumed world still reaches a fork", v8["fork"].startswith("FF2"))

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
