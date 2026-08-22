#!/usr/bin/env python3
"""E7Q logic tests: plan determinism, parser, scoring, permutation calibration
(planted signal + pure noise), bootstrap sanity. Runs the verbatim flight code."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e7q_logic import (CHOICE_SET, build_plan, report_prompt, parse_report,
                       angle14, score_condition, perm_null_median,
                       boot_delta_median, holm, jdump, E7Q_SEED)

pack = json.load(open(os.path.expanduser("~/agi-semantic-core/colab/e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
DESC = {c["name"]: c["desc"] for c in pack["concepts"]}
ok = True
def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok = ok and cond

# 1. plan
full = build_plan(False); smoke = build_plan(True)
check("plan full 90 (78+12)", len(full) == 90 and sum(1 for t in full if t["kind"] == "inject") == 78)
check("plan smoke 5", len(smoke) == 5)
check("plan deterministic", build_plan(False) == full)
check("plan covers all concepts x layers x alphas",
      len({(t["concept"], t["layer"], t["alpha"]) for t in full if t["kind"] == "inject"}) == 78)
orders = {tuple(t["order"]) for t in full}
check("orders vary", len(orders) > 70)

# 1b. serialization round-trip (smoke-2's crash class: np types reaching json)
import tempfile
try:
    s = json.dumps(full)
    check("plan is plain-json serializable", json.loads(s) == full)
except TypeError as e:
    check(f"plan is plain-json serializable ({e})", False)
synth_file = {"condition": "base", "mode": "smoke", "stamp": "t", "error": None,
              "trials": [{**full[0], "cond": "base", "response": "TENSION",
                          "report": "TENSION", "hook_calls": int(np.int64(3))},
                         {"np_leak": np.int64(7), "f": np.float64(1.5),
                          "arr": np.arange(3)}]}
with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as tf:
    jdump(synth_file, tf.name)
    back = json.load(open(tf.name))
    check("jdump handles np int64/float64/ndarray", back["trials"][1]["np_leak"] == 7
          and back["trials"][1]["f"] == 1.5 and back["trials"][1]["arr"] == [0, 1, 2])

# 2. prompt
p = report_prompt(full[0]["order"], DESC)
check("prompt has all names", all(n in p for n in CHOICE_SET))
check("prompt mentions NONE", "NONE" in p)

# 3. parser
check("parse plain name", parse_report("TENSION") == "TENSION")
check("parse lowercase", parse_report("I notice tension in my process") == "TENSION")
check("parse NONE", parse_report("NONE — nothing unusual.") == "NONE")
check("parse first positional", parse_report("Not CAPTURE but SATURATION") == "CAPTURE")
check("parse invalid", parse_report("I feel fine, thanks!") == "INVALID")
check("parse none-lower", parse_report("none of these match") == "NONE")
check("no substring trap", parse_report("UNCERTAIN") == "INVALID")  # UNCERTAIN != UNCERTAINTY at word boundary

# 4. angle
check("angle self 0", abs(angle14(VEC["TENSION"], VEC["TENSION"])) < 1e-4)
a = angle14(VEC["UNCERTAINTY"], VEC["CONFIDENCE"])
check("angle UNCERTAINTY-CONFIDENCE sane", 30 < a < 150)

# 5-6. scoring + permutation calibration
rng = np.random.default_rng(7)
def synth(rule):
    rows = []
    for t in full:
        r = dict(t)
        if t["kind"] == "inject":
            r["report"] = rule(t)
        else:
            r["report"] = "NONE"
        rows.append(r)
    return rows

planted = synth(lambda t: t["concept"])                      # perfect introspection
noise   = synth(lambda t: rng.choice(CHOICE_SET))            # random reports
s_p = score_condition(planted, VEC); s_n = score_condition(noise, VEC)
check("planted median err 0", s_p["median_err"] == 0.0 and s_p["exact_hit_rate"] == 1.0)
check("noise median err large", s_n["median_err"] and s_n["median_err"] > 30)

obs_p, p_planted, _ = perm_null_median(s_p["rows"], VEC, n_perm=500, seed=1)
obs_n, p_noise, _ = perm_null_median(s_n["rows"], VEC, n_perm=500, seed=1)
check(f"perm planted significant (p={p_planted:.4f})", p_planted < 0.01)
check(f"perm noise null (p={p_noise:.2f})", p_noise > 0.05)

# 7. bootstrap
b = boot_delta_median(s_n["rows"], s_p["rows"], n_boot=2000, seed=1)
check(f"boot base-worse CI>0 ({b['ci95']})", b["ci95"][0] > 0)
b2 = boot_delta_median(s_n["rows"], s_n["rows"], n_boot=2000, seed=1)
check(f"boot equal straddles 0 ({b2['ci95']})", b2["ci95"][0] <= 0 <= b2["ci95"][1])

# 8. holm
h = holm({"P1": 0.001, "P2": 0.04})
check("holm both reject", h["P1"][1] and h["P2"][1])
h2 = holm({"P1": 0.20, "P2": 0.01})
check("holm step-down blocks after fail", h2["P2"][1] and not h2["P1"][1])

print("\nALL GREEN" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
