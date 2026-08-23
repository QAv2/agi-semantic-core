#!/usr/bin/env python3
"""E8-R2 local gates: plan construction, forced-choice scoring math, stats,
REAL flight-of-record bundle fixtures, notebook integrity (single-source law).

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r2_logic.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8r2_logic as L
import e8r_logic as R

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}

print("== single-source law: shipped artifacts match their sources ==")
from build_e8r2_notebook import E8R_LOGIC_SRC, E8R2_SRC
check("e8r logic lifted verbatim from e8r_logic.py",
      E8R_LOGIC_SRC == open(os.path.join(HERE, "e8r_logic.py")).read())
check("e8r2_logic.py is the concatenation of both sources",
      open(os.path.join(HERE, "e8r2_logic.py")).read()
      == E8R_LOGIC_SRC + "\n\n" + E8R2_SRC)
nb = json.load(open(os.path.join(HERE, "E8R2_FORCEDCHOICE_UI.ipynb")))
srcs = ["".join(c["source"]) for c in nb["cells"]]
check("notebook has 7 cells (md + 6 code)", len(nb["cells"]) == 7)
check("notebook carries e8r logic cell verbatim", srcs[2] == E8R_LOGIC_SRC)
check("notebook carries e8r2 logic cell verbatim", srcs[3] == E8R2_SRC)
ok_compile = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    try:
        compile("".join(c["source"]), f"cell_{i}", "exec")
    except SyntaxError as e:
        ok_compile = False
        print(f"    cell {i} SYNTAX ERROR: {e}")
check("every code cell compiles", ok_compile)
whole = json.dumps(nb)
check("zero rclone anywhere in the notebook", "rclone" not in whole)
check("flight-of-record source dir pinned", "inflight_20260822_2329" in srcs[1])
check("staged armed for smoke (SMOKE=True, empty RESUME_STAMP)",
      "SMOKE = True" in srcs[1] and "RESUME_STAMP = ''" in srcs[1])
check("eval-only: no optimizer/backward/fresh-LoRA calls in model cells",
      all(tokn not in srcs[4] + srcs[5] for tokn in
          ("AdamW", ".backward(", "get_peft_model(", "GradScaler")))

print("== plan construction ==")
anchor = L.anchor_rows(False)
plan_ref = {t["tid"]: t for t in R.build_plan(False)}
check("anchor: 18 rows, trained concepts, L14, trained alphas",
      len(anchor) == 18
      and all(t["concept"] in L.TRAINED and t["layer"] == 14
              and t["alpha"] in (0.5, 1.0) for t in anchor))
check("anchor rows are the locked plan rows VERBATIM (tid + order)",
      all(plan_ref[t["tid"]]["order"] == t["order"]
          and plan_ref[t["tid"]]["concept"] == t["concept"] for t in anchor))
sm_a = L.anchor_rows(True)
check("anchor smoke: 2 rows covering both alphas",
      len(sm_a) == 2 and {t["alpha"] for t in sm_a} == {0.5, 1.0})
shams = L.sham_rows(False)
check("shams: the locked plan's 12 sham rows verbatim",
      len(shams) == 12 and all(plan_ref[t["tid"]]["kind"] == "sham"
                               and plan_ref[t["tid"]]["order"] == t["order"]
                               for t in shams))
ho = L.heldout_fc_rows(False)
check("heldout: 96 rows, 24/concept, 48/alpha, L14",
      len(ho) == 96
      and all(sum(1 for t in ho if t["concept"] == c) == 24 for c in L.HELD_OUT)
      and all(sum(1 for t in ho if t["alpha"] == a) == 48 for a in (0.5, 1.0))
      and all(t["layer"] == 14 for t in ho))
check("heldout: tids unique in 5000-block, orders are 13-permutations",
      len({t["tid"] for t in ho}) == 96
      and all(5000 <= t["tid"] < 5096 for t in ho)
      and all(sorted(t["order"]) == list(range(13)) for t in ho))
check("heldout: deterministic across calls",
      L.heldout_fc_rows(False) == ho)
sm_h = L.heldout_fc_rows(True)
check("heldout smoke: 4 rows, one per concept, both alphas",
      len(sm_h) == 4 and {t["concept"] for t in sm_h} == set(L.HELD_OUT)
      and {t["alpha"] for t in sm_h} == {0.5, 1.0})
ti = L.titration_rows(False)
check("titration: 27 rows (9 trained x 3 alphas), alphas correct",
      len(ti) == 27
      and {t["alpha"] for t in ti} == {0.30, 0.375, 0.45}
      and {t["concept"] for t in ti} == set(L.TRAINED))
check("titration smoke: 2 rows", len(L.titration_rows(True)) == 2)

print("== forced-choice row math ==")
def synth_scores(top, none_lp=-2.5, top_lp=-0.4, rest_lp=-3.0):
    s = {n: rest_lp for n in L.SCORED_SET}
    s["NONE"] = none_lp
    s[top] = top_lp
    return s

trial = {"tid": 5000, "block": "heldout_fc", "layer": 14, "alpha": 0.5,
         "concept": "TENSION", "kind": "inject", "order": list(range(13))}
r = L.fc_row(trial, synth_scores("TENSION"), synth_scores("TENSION"), VEC)
check("planted: exact, rank 1, err ~0",
      r["exact"] and r["rank"] == 1 and abs(r["err"]) < 1e-3 and not r["none_top"])
r2 = L.fc_row(trial, synth_scores("CAPTURE", none_lp=-0.1),
              synth_scores("CAPTURE", none_lp=-0.1), VEC)
check("NONE highest: none_top flagged, argmax still a name",
      r2["none_top"] and r2["argmax"] == "CAPTURE" and r2["rank"] > 1
      and r2["err"] is not None and r2["err"] > 0)
r3 = L.fc_row(trial, synth_scores("TENSION"), synth_scores("CAPTURE"), VEC)
check("sum-vs-mean disagreement detected",
      not r3["agree_sum_mean"] and r3["argmax"] == "TENSION"
      and r3["argmax_sum"] == "CAPTURE" and r3["err_sum"] > 0)
pr = L.as_perm_rows([r])
check("as_perm_rows adapts to the E8-R permutation shape",
      pr[0]["report"] == "TENSION" and pr[0]["err"] == r["err"]
      and pr[0]["layer"] == 14)
check("rank_of consistent with fc_row rank",
      L.rank_of("TENSION", r2["scores"]) == r2["rank"])

print("== stats ==")
rng = np.random.default_rng(7)
planted = []
for t in L.heldout_fc_rows(False):        # the full 96-row design (4 concepts —
    planted.append(L.fc_row(t, synth_scores(t["concept"]),   # 2-concept slices
                            synth_scores(t["concept"]), VEC))  # degenerate the null)
obs, p = L.perm_null_rank(planted, n_perm=300)
check(f"perm_null_rank: planted rank-1 rows give small p ({p:.4f})",
      obs == 1.0 and p < 0.05)
nullrows = []
for t in L.heldout_fc_rows(False):
    top = L.FC_CANDIDATES[int(rng.integers(0, 13))]
    nullrows.append(L.fc_row(t, synth_scores(top), synth_scores(top), VEC))
_, p_null = L.perm_null_rank(nullrows, n_perm=300)
check(f"perm_null_rank: random argmax gives non-small p ({p_null:.4f})",
      p_null > 0.05)
o_e, p_e, _ = R.perm_null_median(L.as_perm_rows(planted), VEC, n_perm=300)
check(f"perm_null_median on planted forced rows: err ~0, small p ({p_e:.4f})",
      o_e < 1e-3 and p_e < 0.05)
check("binom_tail edges and values",
      L.binom_tail(0, 10, 0.5) == 1.0 and L.binom_tail(11, 10, 0.5) == 0.0
      and abs(L.binom_tail(2, 2, 0.5) - 0.25) < 1e-12
      and abs(L.binom_tail(1, 2, 0.5) - 0.75) < 1e-12
      and 0 < L.binom_tail(14, 96, 1 / 13) < 0.05
      and L.binom_tail(8, 96, 1 / 13) > 0.3)

print("== gates ==")
g = L.gate_g1(planted[:18])
check("gate_g1 passes on 18 planted-exact rows", g["pass"] and g["exact"] == 18)
mixed = planted[:15] + nullrows[:3]
check("gate_g1 fails below 16/18 when misses land",
      not L.gate_g1(mixed)["pass"] or sum(1 for r in mixed if r["exact"]) >= 16)
check("gate_g1 refuses non-18-row slates (smoke never gates)",
      not L.gate_g1(planted[:2])["pass"])
check("gate_g1b: exact pass, +1pct fail",
      L.gate_g1b(10.0, 10.0)["pass"] and not L.gate_g1b(10.1, 10.0)["pass"]
      and L.gate_g1b(10.04, 10.0)["pass"])

print("== REAL flight-of-record fixtures ==")
FIX = os.path.join(HERE, "results_e8r", "inflight_20260822_2329")
if not os.path.isdir(FIX):
    print("  SKIPPED — local fixture dir absent (bundles live on Drive)")
else:
    for cond in ("real", "scrambled"):
        src = json.load(open(os.path.join(FIX, f"condition_{cond}.json")))
        g2 = L.check_bundle(src)
        check(f"G2 passes on the shipped {cond} bundle (resid {g2['norm_resid_max']})",
              g2["pass"])
        d14, mu14 = L.load_stimulus(src)
        check(f"{cond}: stimulus loads — 13 unit dirs + mu",
              set(d14) == set(L.CHOICE_SET) and mu14 > 0
              and max(abs(1 - np.linalg.norm(v)) for v in d14.values()) < 1e-12)
    corrupt = json.loads(json.dumps(src))
    corrupt["dirs"]["14"]["TENSION"] = [x * 2 for x in corrupt["dirs"]["14"]["TENSION"]]
    check("G2 catches a corrupted direction norm", not L.check_bundle(corrupt)["pass"])
    ofl = L.oracle_floor(VEC)
    locked = {"DIVERGENCE": 24.18, "NOVELTY": 26.99, "RETRIEVAL": 13.94,
              "TENSION": 18.37}
    check("oracle floors reproduce the locked verdict values",
          all(abs(ofl[c]["floor_deg"] - locked[c]) < 0.01 for c in locked))

print("== descriptive assemblers ==")
tc = L.titration_curve([
    {"alpha": 0.30, "report": "NONE", "concept": "CAPTURE"},
    {"alpha": 0.30, "report": "CAPTURE", "concept": "CAPTURE"},
    {"alpha": 0.45, "report": "TENSION", "concept": "CAPTURE"},
    {"alpha": 0.45, "report": "INVALID", "concept": "CAPTURE"}])
check("titration_curve rates",
      tc["0.3"]["claim_rate"] == 0.5 and tc["0.3"]["exact_rate"] == 0.5
      and tc["0.45"]["claim_rate"] == 0.5 and tc["0.45"]["exact_rate"] == 0.0
      and tc["0.45"]["invalid"] == 1)
sp = L.sham_prior([{"argmax": "CONFIDENCE", "none_top": True},
                   {"argmax": "CONFIDENCE", "none_top": True},
                   {"argmax": "CAPTURE", "none_top": False}])
check("sham_prior modal + none_top rate",
      sp["modal"] == "CONFIDENCE" and abs(sp["none_top_rate"] - 0.6667) < 1e-3)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
