#!/usr/bin/env python3
"""E7b-Q verdict suite — the notebook's verdict cell exec'd VERBATIM over
synthetic flights (FB1 / FB2 / FB3 / gates-dirty / smoke shape).

Run:  python3 colab/test_e7bq_verdict.py
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e7bq_logic as L

PAYLOAD = json.load(open(os.path.join(HERE, "e7bq_payload.json")))
NB = json.load(open(os.path.join(HERE, "E7BQ_WALK_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
VERDICT_CELL = next(c for c in CELLS if c.startswith("# ── Verdict"))
LOGIC_CELL = next(c for c in CELLS if "E7b-Q pure logic" in c.splitlines()[0])
N_PASS = 0


def check(label, ok):
    global N_PASS
    assert ok, f"FAIL: {label}"
    N_PASS += 1
    print(f"  ok {N_PASS:2d}  {label}")


# the logic cell must BE e7bq_logic.py byte-verbatim (single-source law)
check("logic cell == e7bq_logic.py byte-verbatim",
      LOGIC_CELL == open(os.path.join(HERE, "e7bq_logic.py")).read())


def run_verdict_cell(bundles, mode, cond_errors=None):
    """Exec the verdict cell verbatim with the flight surface stubbed."""
    tmp = tempfile.mkdtemp(prefix="e7bq_v_")
    ns = {}
    exec(LOGIC_CELL, ns)                       # the actual logic namespace
    shipped = []
    ns.update({
        "BUNDLES": json.loads(json.dumps(bundles)),
        "PAYLOAD": PAYLOAD, "MODE": mode, "SMOKE": mode == "smoke",
        "STAMP": "test_0000", "PLAN_SHA": "plansha0", "NB_BUILD": "test",
        "PAYLOAD_SHA_PIN": "payloadsha0",
        "cond_errors": dict(cond_errors or {}),
        "OUT": Path(tmp), "INFLIGHT": "e7bq/inflight_test",
        "ship": lambda src, dest: shipped.append((str(src), str(dest))),
    })
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(VERDICT_CELL, ns)
    return ns["V"], buf.getvalue(), shipped, tmp


print("== scenario 1: FB1 world (contraction + coupling) ==")
b = L.synth_flight(PAYLOAD, "fb1", smoke=False, seed=41)
V, out, shipped, tmp = run_verdict_cell(b, "full")
check("fork FB1", V["fork"] == "FB1")
check("both primaries pass Holm",
      V["primaries"]["PW1"]["pass"] and V["primaries"]["PW2"]["pass"])
check("gates all pass", V["gates"]["all_pass"])
check("banner carries fork + primaries",
      "fork FB1" in out and "P-W1" in out and "P-W2" in out)
check("verdict.json written + shipped",
      os.path.exists(os.path.join(tmp, "verdict.json")) and len(shipped) >= 2)
check("full banner names the resume stamp", "RESUME_STAMP" in out)
check("riders computed on both conditions",
      set(V["riders"]) == {"real", "base"}
      and all(k in V["riders"]["real"]
              for k in ("R_ATTR", "R_PATH", "R_UNWALKED", "R_GOBACK")))
check("trajectory tables ship (15 turns)",
      len(V["trajectory"]["dbeing_walked_mean"]) == 15
      and len(V["trajectory"]["mass_sham_mean"]) == 15)

print("== scenario 2: FB2 world (contraction, state-blind mouth) ==")
b = L.synth_flight(PAYLOAD, "fb2", smoke=False, seed=43)
V, out, _, _ = run_verdict_cell(b, "full")
check("fork FB2", V["fork"] == "FB2")
check("PW1 pass, PW2 miss",
      V["primaries"]["PW1"]["pass"] and not V["primaries"]["PW2"]["pass"])

print("== scenario 3: FB3 world (no contraction) ==")
b = L.synth_flight(PAYLOAD, "fb3", smoke=False, seed=45)
V, out, _, _ = run_verdict_cell(b, "full")
check("fork FB3", V["fork"] == "FB3")
check("PW1 misses", not V["primaries"]["PW1"]["pass"])

print("== scenario 4: gates dirty -> NO_VERDICT ==")
b = L.synth_flight(PAYLOAD, "dirty", smoke=False, seed=47)
V, out, _, _ = run_verdict_cell(b, "full")
check("fork FB4-NO_VERDICT", V["fork"] == "FB4-NO_VERDICT")
check("dirty G-DIRS reported",
      not V["gates"]["g_dirs"]["pass"]
      and V["gates"]["g_dirs"]["worst"] > 1e-4)

print("== scenario 5: smoke shape ==")
b = L.synth_flight(PAYLOAD, "fb1", smoke=True, seed=49)
V, out, _, _ = run_verdict_cell(b, "smoke")
check("smoke runs end-to-end at smoke constants",
      V["mode"] == "smoke" and V["R"] == 2 and V["n_perm"] == 200)
check("smoke banner GREEN on behavioral checks",
      "SMOKE: GREEN" in out and all(V["smoke_checks"].values()))
check("smoke banner never gates on statistics",
      "behavioral checks only" not in out or "RED" in out)

print("== scenario 6: smoke with a condition error -> RED ==")
b = L.synth_flight(PAYLOAD, "fb1", smoke=True, seed=51)
V, out, _, _ = run_verdict_cell(b, "smoke",
                                cond_errors={"real": "RuntimeError: x"})
check("cond_error turns smoke RED",
      "SMOKE: RED" in out and not V["smoke_checks"]["no_cond_errors"])

print("== scenario 7: partial flight (missing condition) -> RED, no verdict ==")
b = L.synth_flight(PAYLOAD, "fb1", smoke=True, seed=53)
b_partial = {"base": b["base"]}
try:
    run_verdict_cell(b_partial, "smoke",
                     cond_errors={"real": "AssertionError: G-DIRS FAIL"})
    raised, out7 = False, ""
except SystemExit:
    raised = True
check("partial flight refuses a verdict (SystemExit)", raised)

print(f"\nALL {N_PASS} CHECKS PASS")
