#!/usr/bin/env python3
"""Exercise the E8-O2 VERDICT cell VERBATIM against synthetic flights:
(1) FV1 cured world (leg-faithful everywhere) — both primaries + S-MEJI
positive; (2) FV2 order-blind — P-M2 passes, P-M1 d~0; (3) FV3
essence-cues-only — order carried on the essence leg alone, function
all-MID (uninformative: equidistant from junior content and méjì
completion, so P-M1 passes, P-M2 fails, S-MEJI nets ~0); (4) retention
garbage -> FV4 re-fly wording; (5) install
garbage -> FV4; (6) sham flood -> gates dirty; (7) smoke GREEN;
(8) flight error. Full scenarios at real constants.

Run:  python3 colab/test_e8o2_verdict.py
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o2_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P-M1 ORDER" in VERDICT_SRC
LOGIC_SRC = "".join(nb["cells"][2]["source"])
assert LOGIC_SRC == open(os.path.join(HERE, "e8o2_logic.py")).read()

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
FR = L.frame_codes(PACK["concepts"])
CODES64 = {n: FR[n] for n in L.EVAL64}
PC = L.pair_codes(VEC)
PCT = L.pair_codes_for(L.TRAINPAIR48, VEC)
PCF = L.pair_codes_for(L.POFRESH12, VEC)


def parsed_of(levels):
    if levels is None:
        return {"kind": "NONE", "levels": {}, "n_fields": 0}
    return {"kind": "CODE",
            "levels": {L.AXIS_NAMES_F[j]: int(levels[j]) for j in range(14)},
            "n_fields": 14}


def code_for(r, mode):
    tag = r["skey"].split(":")[0] if r.get("skey") else None
    pcs = {"comp": PC, "compT": PCT, "compF": PCF}.get(tag)
    if pcs is None:
        return None
    if mode == "faithful":
        return pcs[(r["pair_idx"], r["order"])]
    if mode == "blind":
        return pcs[(r["pair_idx"], 0)]
    if mode == "esscue":
        own = pcs[(r["pair_idx"], r["order"])]
        return own[:7] + [0] * 7         # correct essence, all-MID function
    return None


def blocks_fixture(mode, smoke=False):
    rows = L.build_e8o2_eval(smoke)
    blocks = {}
    for r in rows:
        b = r["block"]
        if b in ("po", "po_fresh", "spot_mixed"):
            if b == "spot_mixed" and mode == "installfail":
                p = parsed_of([0] * 14)
            else:
                p = parsed_of(code_for(r, mode if mode in
                                       ("faithful", "blind", "esscue")
                                       else "faithful"))
        elif b == "ident":
            p = parsed_of([0] * 14 if mode == "gidfail"
                          else CODES64[r["concept"]])
        elif b == "sham":
            p = parsed_of(CODES64[sorted(L.EVAL64)[0]]
                          if mode == "shamflood" else None)
        elif b == "carrier":
            p = parsed_of([0] * 14)
        blocks.setdefault(b, []).append(
            {**r, "response": "synthetic", "parsed": p, "hook_calls": 1})
    return blocks


def bundle_fixture(mode, smoke=False):
    if mode == "error":
        return {"condition": "real", "mode": "full", "stamp": "TEST",
                "error": "RuntimeError: boom"}
    return {"condition": "real", "mode": "smoke" if smoke else "full",
            "stamp": "TEST", "payload_sha": L.PAYLOAD_SHA,
            "sha_trainpair": L.SHA_TRAINPAIR, "sha_pofresh": L.SHA_POFRESH,
            "readout_src": L.READOUT_SRC,
            "g2_dirs": {"resid": 5e-08, "tol": 1e-4, "pass": True,
                        "layer": 14, "name": "UNCERTAINTY"},
            "mu14": 81.9, "ppl_pre": 19.74, "ppl_post": 19.9,
            "curriculum": dict(L.EXPECT_TRAIN_O2),
            "train_log": {"epochs_flown": 5, "plateaued": True,
                          "strand_epoch_means": [], "opt_steps": 160,
                          "losses_every_10": [1.0]},
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
    vf = Path(td) / "e8o2_verdict.json"
    return (json.load(open(vf)) if vf.exists() else None,
            buf.getvalue(), shipped)


print("== scenario 1: FV1 (cured) ==")
v, out, sh = run_verdict(bundle_fixture("faithful"))
check("gates pass", all(g.get("pass") for g in v["gates"].values()))
check("P-M1 + P-M2 pass", v["P_M1"]["pass"] and v["P_M2"]["pass"])
check("S-MEJI positive + pass", v["S_MEJI"]["pass"]
      and v["S_MEJI"]["d_mean"] > 0.3)
check("S-FRESH consistent", v["S_FRESH"]["consistent"])
check("fork = FV1", v["fork"].startswith("FV1"))
check("before/after table carries baselines",
      v["before_after"]["po2_d"][0] == 0.0275)

print("== scenario 2: FV2 (order-blind) ==")
v2, _, _ = run_verdict(bundle_fixture("blind"))
check("P-M2 passes, P-M1 fails", v2["P_M2"]["pass"] and not v2["P_M1"]["pass"])
check("fork = FV2", v2["fork"].startswith("FV2"))

print("== scenario 3: FV3 (essence-cue order, all-MID function) ==")
v3, _, _ = run_verdict(bundle_fixture("esscue"))
check("P-M1 passes on essence cues", v3["P_M1"]["pass"])
check("P-M2 fails (function unread)", not v3["P_M2"]["pass"])
check("S-MEJI neutral, no pass (function uninformative)",
      abs(v3["S_MEJI"]["d_mean"]) < 0.05 and not v3["S_MEJI"]["pass"])
check("fork = FV3", v3["fork"].startswith("FV3"))

print("== scenario 4: retention fail -> FV4 ==")
v4, _, _ = run_verdict(bundle_fixture("gidfail"))
check("G-ID fails, fork = FV4 re-fly wording",
      not v4["gates"]["g_id"]["pass"] and v4["fork"].startswith("FV4")
      and "re-fly" in v4["fork"])

print("== scenario 5: install fail -> FV4 ==")
v5, _, _ = run_verdict(bundle_fixture("installfail"))
check("G-INSTALL-M fails, fork = FV4",
      not v5["gates"]["g_install_m"]["pass"] and v5["fork"].startswith("FV4"))

print("== scenario 6: sham flood -> gates dirty ==")
v6, _, _ = run_verdict(bundle_fixture("shamflood"))
check("G5 fails, plain gates-dirty fork",
      not v6["gates"]["g5_sham"]["pass"]
      and v6["fork"].startswith("GATES DIRTY"))

print("== scenario 7: smoke GREEN ==")
v7, out7, _ = run_verdict(bundle_fixture("faithful", smoke=True), smoke=True)
check("smoke banner GREEN", "SMOKE GREEN" in out7
      and v7["fork"].startswith("SMOKE"))

print("== scenario 8: flight error ==")
v8, out8, sh8 = run_verdict(bundle_fixture("error"))
check("incomplete banner, no verdict file",
      v8 is None and "FLIGHT INCOMPLETE" in out8 and not sh8)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
