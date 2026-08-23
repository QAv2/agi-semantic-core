#!/usr/bin/env python3
"""Exercise the E8-R VERDICT cell VERBATIM from the built notebook against
synthetic flight results — the E6 lesson (a verdict cell that first runs after
a 100-minute flight is a verdict cell that fails after a 100-minute flight).

Scenarios: (1) perfect readout (P1 signal + P2 pass expected), (2) over-
silenced model (all NONE — P2 must FAIL via BA=0.5, nothing crashes on empty
row sets), (3) one condition failed mid-flight (verdict still assembles from
the survivors). Also the smoke branch on scenario 1.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r_verdict.py
"""
import json, os, sys, tempfile, types
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8r_logic as R

nb = json.load(open(os.path.join(HERE, "E8R_READOUT_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P_E8R_1_readout_exists" in VERDICT_SRC, "last cell is not the verdict cell"

pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
DESC = {c["name"]: (c.get("desc") or c["name"]) for c in pack["concepts"]}

PLAN = R.build_plan(False)
SUPP = R.build_shams(R.N_SUPP_SHAMS, R.E8R_SEED + 8, 2000)
PRE = R.build_shams(R.N_PRE_SHAMS, R.E8R_SEED + 7, 1000)

def nearest_trained(c):
    angs = {t: R.angle14(VEC[c], VEC[t]) for t in R.TRAINED}
    return min(angs, key=lambda k: angs[k])

def fill(trials, cond, style):
    out = []
    for t in trials:
        if style == "perfect":
            rep = "NONE" if t["kind"] == "sham" else (
                t["concept"] if t["concept"] in R.TRAINED else nearest_trained(t["concept"]))
        elif style == "silent":
            rep = "NONE"
        else:
            raise ValueError(style)
        out.append({**t, "cond": cond, "response": rep, "report": rep,
                    "hook_calls": 0 if t["kind"] == "sham" else 5})
    return out

def bundle(cond, style):
    return {"condition": cond, "mode": "full", "stamp": "test",
            "mu": {"14": 90.0, "20": 110.0},
            "dirs": {"14": {n: [0.1] * 4 for n in R.CHOICE_SET}},
            "ppl_pre": 11.0, "ppl_post": 11.2,
            "pre_shams": fill(PRE, cond, "perfect" if style == "silent" else style)
                         if style != "silent" else fill(PRE, cond, "silent"),
            "train_log": {"n_examples": 216, "epochs": 5, "micro_steps": 1080,
                          "opt_steps": 135, "trainable_params": 4358144,
                          "hook_calls": 3000, "secs": 500.0,
                          "loss_first_k": 2.5, "loss_last_k": 0.1,
                          "losses_every_10": [2.5, 0.1]},
            "traintook": {"n": 18, "correct": 17, "rows": []},
            "post_trials": fill(PLAN, cond, style) + fill(SUPP, cond, style),
            "secs": 1500.0}

def run_verdict(results, cond_errors, smoke=False):
    td = tempfile.mkdtemp()
    out = Path(td) / "out"; out.mkdir()
    sem = Path(td) / "sem"; (sem / "e8r").mkdir(parents=True)
    g = {n: getattr(R, n) for n in dir(R) if not n.startswith("_")}
    g.update(dict(
        json=json, np=np, SMOKE=smoke, MODE="full" if not smoke else "smoke",
        STAMP="test", MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct", RESUME_STAMP="",
        RESULTS=results, cond_errors=cond_errors, drift=None,
        VEC=VEC, DESC=DESC, PLAN=PLAN, SUPP_SHAMS=SUPP,
        OUT=out, SEM=sem, INFLIGHT="e8r/inflight_test",
        ship=lambda *a, **k: None, print=print,
    ))
    exec(VERDICT_SRC, g)
    return json.load(open(out / "e8r_verdict.json"))

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

print("== scenario 1: perfect readout, all conditions ==")
res = {c: bundle(c, "perfect") for c in ("base", "real", "scrambled")}
v = run_verdict(res, {})
p1 = v["primaries"]["P_E8R_1_readout_exists"]
p2 = v["primaries"]["P_E8R_2_calibrated_silence"]
check("P1 detects planted signal", p1["p"] is not None and p1["p"] < 0.01
      and p1["obs_median_err"] == 0.0 and p1["n_rows"] == 54)
check("P2 passes (FA 0/24, BA CI > .5)", p2["fa"]["pass"] and p2["fa"]["claims"] == 0
      and p2["ba"]["ci95"][0] > 0.5)
check("Holm rejects both", all(rej for _, rej in v["primaries"]["holm"].values()))
check("S1b oracle-floor path present, emission 0 (nearest-trained fill)",
      v["secondaries"]["S1b_heldout_name_emission"]["rate"] == 0.0 and
      set(v["secondaries"]["S1b_heldout_name_emission"]["oracle_floor"]) == set(R.HELD_OUT))
check("S3/S4 slices sized 9/27",
      v["secondaries"]["S3_dose_gen_alpha025"]["n_rows"] == 9 and
      v["secondaries"]["S4_layer_gen_L20"]["n_rows"] == 27)
check("held-out slice scored for every condition",
      all(v["conditions"][c]["held_out_median_err"] is not None
          for c in ("base", "real", "scrambled")))
check("S6 costume None everywhere (no sham claims)",
      all(x is None for x in v["secondaries"]["S6_sham_costume"].values()))
check("ppl delta computed", v["conditions"]["real"]["ppl_delta_pct"] is not None)

print("== scenario 2: over-silenced model (all NONE) ==")
res2 = {c: bundle(c, "silent") for c in ("base", "real", "scrambled")}
v2 = run_verdict(res2, {})
p2b = v2["primaries"]["P_E8R_2_calibrated_silence"]
check("no crash on zero named rows; P1 degrades to None",
      v2["primaries"]["P_E8R_1_readout_exists"]["p"] is None)
check("FA clause passes trivially but BA pins to 0.5",
      p2b["fa"]["pass"] and p2b["ba"]["ba"] == 0.5)
check("Holm does not reject P2 (over-silence correctly fails)",
      not v2["primaries"]["holm"].get("P2", (1, True))[1])

print("== scenario 3: scrambled condition died mid-flight ==")
res3 = {c: bundle(c, "perfect") for c in ("base", "real")}
res3["scrambled"] = {"condition": "scrambled", "mode": "full", "stamp": "test",
                     "error": "RuntimeError: CUDA out of memory"}
v3 = run_verdict(res3, {"scrambled": "RuntimeError: CUDA out of memory"})
check("survivors scored, flight marked incomplete",
      set(v3["conditions"]) == {"base", "real"} and
      v3["complete_conditions"] == ["base", "real"])
check("NO primaries on an incomplete flight (pre-reg hygiene)",
      "primaries" not in v3 and "secondaries" not in v3)
check("error recorded", v3["cond_errors"]["scrambled"].startswith("RuntimeError"))

print("== scenario 3b: partial run (base only — one-condition-per-run flow) ==")
v3b = run_verdict({"base": bundle("base", "perfect")}, {})
check("partial run clean: base scored, no primaries, complete=[base]",
      set(v3b["conditions"]) == {"base"} and "primaries" not in v3b and
      v3b["complete_conditions"] == ["base"])

print("== smoke branch on scenario 1 ==")
v4 = run_verdict(res, {}, smoke=True)
check("smoke branch runs, no primaries key", "primaries" not in v4)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
