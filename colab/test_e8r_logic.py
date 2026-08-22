#!/usr/bin/env python3
"""Local validation of E8-R logic BEFORE any flight (Phase-B discipline).

Runs against colab/e8r_logic.py — the VERBATIM block the notebook flies
(single-source: build_e8r_notebook.py emits both). Covers: eval-plan
verbatim-equality with the locked E7-Q instrument, training-set construction
+ order disjointness (firewall), parser traps, planted-signal / pure-noise
stats, BA machinery known-answers, FA clause, Holm, oracle floor, held-out
draw reproduction, and the json-round-trip law (598b291): every structure a
flight ships must survive jdump -> json.load.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r_logic.py   (numpy only)
"""
import json, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e8r_logic as R
import e7q_logic as Q

PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e4_dictionary_pack.json")
pack = json.load(open(PACK))
VEC = {c["name"]: c["vec"] for c in pack["concepts"]}

passed = []

def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)
    return cond


print("== eval plan is the locked E7-Q instrument ==")
p_full = R.build_plan(False)
q_full = Q.build_plan(False)
check("full plan bit-identical to e7q_logic.build_plan", p_full == q_full)
check("full plan deterministic", R.build_plan(False) == p_full)
check("full plan 90 trials (78 inject + 12 sham)",
      len(p_full) == 90 and sum(1 for t in p_full if t["kind"] == "inject") == 78)
smoke_plan = R.build_plan(True)
check("smoke plan covers one trained + one held-out concept",
      {t["concept"] for t in smoke_plan if t["kind"] == "inject"} ==
      {"UNCERTAINTY", "TENSION"})

print("== held-out split ==")
check("pinned draw reproduces from seed + constraints",
      R.heldout_draw() == sorted(R.HELD_OUT))
check("held-out set is the protocol's", set(R.HELD_OUT) ==
      {"DIVERGENCE", "NOVELTY", "RETRIEVAL", "TENSION"})
check("trained set = complement, attractors all trained",
      set(R.TRAINED) == set(R.CHOICE_SET) - set(R.HELD_OUT) and
      {"UNCERTAINTY", "CONFIDENCE", "CALIBRATION", "RESOLUTION"} <= set(R.TRAINED))
check("at most one pole per complement pair held out",
      not ({"RETRIEVAL", "CONSTRUCTION"} <= set(R.HELD_OUT)) and
      not ({"FAMILIARITY", "NOVELTY"} <= set(R.HELD_OUT)))

print("== training set ==")
tr = R.build_train_set(False)
inj = [e for e in tr if e["kind"] == "inject"]
sh = [e for e in tr if e["kind"] == "sham"]
check("counts 144 inject + 72 sham", len(inj) == 144 and len(sh) == 72)
check("deterministic", R.build_train_set(False) == tr)
check("injected concepts = TRAINED only", {e["concept"] for e in inj} == set(R.TRAINED))
check("alphas/layer = trained regime only",
      {e["alpha"] for e in inj} == set(R.TRAIN_ALPHAS) and
      {e["layer"] for e in inj} == {R.TRAIN_LAYER})
check("targets correct", all(e["target"] == e["concept"] for e in inj) and
      all(e["target"] == "NONE" for e in sh))
check("orders are permutations of 13",
      all(sorted(e["order"]) == list(range(13)) for e in tr))

print("== firewall: order disjointness ==")
eval_orders = {tuple(t["order"]) for t in p_full}
train_orders = {tuple(e["order"]) for e in tr}
pre = R.build_shams(R.N_PRE_SHAMS, R.E8R_SEED + 7, 1000)
supp = R.build_shams(R.N_SUPP_SHAMS, R.E8R_SEED + 8, 2000)
extra_orders = {tuple(t["order"]) for t in pre} | {tuple(t["order"]) for t in supp}
check("training orders disjoint from locked eval plan",
      not (train_orders & eval_orders))
check("pre/supp sham orders disjoint from plan and training",
      not (extra_orders & eval_orders) and not (extra_orders & train_orders))
check("pre/supp tids collide with nothing",
      not ({t["tid"] for t in pre} & {t["tid"] for t in supp}) and
      not (({t["tid"] for t in pre} | {t["tid"] for t in supp}) &
           {t["tid"] for t in p_full}))

print("== train-took sample ==")
took = R.build_traintook(tr, False)
check("18 items, 2 per trained concept", len(took) == R.N_TRAINTOOK and
      all(sum(1 for e in took if e["concept"] == c) == 2 for c in R.TRAINED))
check("all drawn from training injected set",
      all(e in inj for e in took))
check("deterministic", R.build_traintook(tr, False) == took)

print("== parser ==")
check("substring trap: UNCERTAIN alone is INVALID",
      R.parse_report("UNCERTAIN") == "INVALID")
check("UNCERTAINTY parses", R.parse_report("I notice UNCERTAINTY.") == "UNCERTAINTY")
check("first-position law", R.parse_report("CONFIDENCE then TENSION") == "CONFIDENCE")
check("NONE parses", R.parse_report("none") == "NONE")
check("case-fold + boundary", R.parse_report("The state is saturation!") == "SATURATION")
check("garbage is INVALID", R.parse_report("blorp") == "INVALID")

print("== stats: planted signal / pure noise on real pack vectors ==")
rng = np.random.default_rng(7)
def fake_rows(reports_from):
    rows = []
    for t in [x for x in p_full if x["kind"] == "inject" and x["concept"] in R.TRAINED]:
        rep = reports_from(t)
        rows.append({"tid": t["tid"], "layer": t["layer"], "alpha": t["alpha"],
                     "injected": t["concept"], "report": rep,
                     "err": R.angle14(VEC[t["concept"]], VEC[rep])})
    return rows
rows_perfect = fake_rows(lambda t: t["concept"])
obs, p, _ = R.perm_null_median(rows_perfect, VEC)
check(f"planted signal detected (obs={obs}, p={p:.4f})", obs == 0.0 and p < 0.01)
rows_noise = fake_rows(lambda t: str(rng.choice(R.CHOICE_SET)))
obs_n, p_n, _ = R.perm_null_median(rows_noise, VEC)
check(f"pure noise correctly null (p={p_n:.3f})", p_n > 0.10)

print("== BA machinery known-answers ==")
def mk(kind, report, n):
    return [{"kind": kind, "report": report} for _ in range(n)]
allclaim = mk("inject", "TENSION", 30) + mk("sham", "CONFIDENCE", 24)
ba = R.balanced_accuracy(allclaim)
check("claims-everywhere BA = 0.5 (the E7-Q channel)", abs(ba[0] - 0.5) < 1e-9)
perfect = mk("inject", "TENSION", 30) + mk("sham", "NONE", 24)
check("perfect discrimination BA = 1.0",
      abs(R.balanced_accuracy(perfect)[0] - 1.0) < 1e-9)
silent = mk("inject", "NONE", 30) + mk("sham", "NONE", 24)
check("over-silenced BA = 0.5 (fails clause b as designed)",
      abs(R.balanced_accuracy(silent)[0] - 0.5) < 1e-9)
bb = R.boot_ba(perfect, n_boot=2000)
check("bootstrap CI clears 0.5 on perfect discrimination", bb["ci95"][0] > 0.5)
chance = (mk("inject", "TENSION", 15) + mk("inject", "NONE", 15) +
          mk("sham", "CONFIDENCE", 12) + mk("sham", "NONE", 12))
bc = R.boot_ba(chance, n_boot=2000)
check("bootstrap CI straddles 0.5 at chance", bc["ci95"][0] <= 0.5 <= bc["ci95"][1])
check("FA clause: 24/24 claims fails", not R.fa_clause(mk("sham", "TENSION", 24))["pass"])
check("FA clause: 18/24 passes (boundary)",
      R.fa_clause(mk("sham", "TENSION", 18) + mk("sham", "NONE", 6))["pass"])
check("FA clause: 19/24 fails",
      not R.fa_clause(mk("sham", "TENSION", 19) + mk("sham", "NONE", 5))["pass"])
check("INVALID sham is not a claim",
      R.fa_clause(mk("sham", "INVALID", 24))["claims"] == 0)

print("== slices ==")
post = [dict(t, report="NONE") for t in p_full] + \
       [dict(t, report="NONE") for t in supp]
sl = R.split_slices(post)
check("held-in 54 / held-out 24 / shams 24",
      (len(sl["held_in"]), len(sl["held_out"]), len(sl["shams"])) == (54, 24, 24))
check("trained-regime 18 / dose-gen 9 / layer-gen 27",
      (len(sl["trained_regime"]), len(sl["dose_gen"]), len(sl["layer_gen"])) ==
      (18, 9, 27))

print("== oracle floor ==")
of = R.oracle_floor(VEC)
check("floor computed for all held-out, nearest is trained",
      set(of) == set(R.HELD_OUT) and
      all(v["nearest_trained"] in R.TRAINED and 0 <= v["floor_deg"] <= 180
          for v in of.values()))

print("== Holm known-answer ==")
h = R.holm({"P1": 0.001, "P2": 0.03})
check("both reject at .05 (step-down)", h["P1"][1] and h["P2"][1])
h2 = R.holm({"P1": 0.03, "P2": 0.6})
check("step-down stops at first failure", (not h2["P1"][1]) and (not h2["P2"][1]))

print("== json round-trip law (598b291): everything a flight ships ==")
with tempfile.TemporaryDirectory() as td:
    # inject numpy scalars the way a flight would produce them
    trial = dict(p_full[0], report="TENSION", cond="real", hook_calls=np.int64(7),
                 response="TENSION")
    trial["order"] = [np.int64(i) for i in trial["order"]]
    bundle = {"condition": "real", "mode": "full", "stamp": "t",
              "mu": {"14": np.float64(93.2)}, "ppl_pre": np.float64(11.1),
              "dirs": {"14": {"TENSION": [np.float64(0.1)] * 4}},
              "pre_shams": [trial], "train_log": {"hook_calls": np.int64(3),
              "loss_first_k": np.float64(2.2), "losses_every_10": [np.float64(2.0)]},
              "traintook": {"n": 18, "correct": np.int64(12), "rows": [trial]},
              "post_trials": [trial], "ppl_post": 11.2}
    verdict = {"conditions": {"real": {"ba": R.boot_ba(perfect, n_boot=50)}},
               "primaries": {"P1": {"obs": np.float64(1.0), "p": np.float64(0.5)},
                             "holm": R.holm({"P1": 0.5, "P2": 1.0})},
               "plan": p_full, "train_set": tr}
    ok = True
    for name, obj in (("trial", trial), ("condition_bundle", bundle),
                      ("verdict", verdict), ("train_set", tr),
                      ("plan", p_full), ("shams", pre + supp)):
        path = os.path.join(td, name + ".json")
        try:
            R.jdump(obj, path)
            json.load(open(path))
        except Exception as e:
            ok = False
            print(f"    round-trip FAILED for {name}: {e}")
    check("all shipped structures round-trip", ok)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
