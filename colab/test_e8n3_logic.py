#!/usr/bin/env python3
"""E8-N v3 logic suite — pins vs the committed design check, the per-strand
plateau donor on the flown v2 curves + planted worlds, S10' paired-stat
teeth (recovered / unchanged / worse / n<2 guard), the calib single-source
tooth (sliced flight code == design-check numbers EXACTLY on the v2 rows) +
planted calibration worlds, notebook structure + the registered trims, and
rebuild-byte-identical ×2 (subprocess — catches per-process repr
nondeterminism, the class found at build time in the donor's set repr).

Run:  python3 colab/test_e8n3_logic.py
"""
import json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n3_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

DESIGN = json.load(open(os.path.join(HERE, "results_e8n3",
                                     "design_check.json")))
FCPIN = json.load(open(os.path.join(HERE, "results_e8n3",
                                    "fc_baseline_v2.json")))
B = json.load(open(os.path.join(HERE, "results_e8n2", "full_20260824_0050",
                                "condition_real.json")))

print("== pins vs the committed design check ==")
check("FC baseline: 96 tids, sha matches pin and design check",
      len(L.FC_BASELINE_V2) == 96
      and L.fc_baseline_sha_check() == L.FC_BASELINE_SHA
      and L.FC_BASELINE_SHA == FCPIN["sha"]
      and L.FC_BASELINE_SHA == DESIGN["pins"]["fc_baseline_sha"])
check("FC baseline values == pinned file",
      all(abs(L.FC_BASELINE_V2[int(k)] - v) < 1e-9
          for k, v in FCPIN["per_tid_err"].items()))
check("cap/plateau constants match the registration",
      L.EPOCHS_CAP_V3 == 12 and L.MIN_EPOCHS_V2 == 3
      and abs(L.PLATEAU_REL - 0.05) < 1e-12
      and DESIGN["cure"]["EPOCH_CAP"] == 12)
check("curriculum expectation matches v2's flown counts",
      L.EXPECT_CURRICULUM_V3 == {"scalar": 272, "naming": 216,
                                 "competence": 60, "lexicon": 52}
      and B["curriculum"] == L.EXPECT_CURRICULUM_V3)
check("catch ids match the design check",
      [r["id"] for r in B["post"]["catch_rows"]] == DESIGN["pins"]["catch_ids"])

print("== per-strand plateau (donor) ==")
curves = B["train_log"]["strand_epoch_loss"]
check("v2 flown curves: NOT plateaued at min_epochs 2 and 3",
      not L.per_strand_plateau(curves, min_epochs=2)
      and not L.per_strand_plateau(curves, min_epochs=3))
flat = curves + [{s: curves[-1][s] * 0.99 for s in curves[-1]},
                 {s: curves[-1][s] * 0.985 for s in curves[-1]}]
check("all-strands-flat world: plateaued",
      L.per_strand_plateau(flat, min_epochs=3))
one_live = [dict(e) for e in flat]
one_live[-1]["competence"] = one_live[-2]["competence"] * 0.8
check("one strand still descending: NOT plateaued",
      not L.per_strand_plateau(one_live, min_epochs=3))
check("short history: NOT plateaued (guard)",
      not L.per_strand_plateau(curves[:1], min_epochs=3)
      and not L.per_strand_plateau([], min_epochs=3))
withnone = [{**e, "lexicon": None} for e in flat]
check("None-strand handling: no crash, plateaus on the live strands",
      L.per_strand_plateau(withnone, min_epochs=3))

print("== S10' paired stat ==")
fc_rows = [r for r in B["fc_rows"] if r["block"] == "heldout_fc"]
s_same = L.s10_paired(fc_rows, n_perm=300)
check("identical rows: d exactly 0, p not significant, n 96, medians equal",
      s_same["d_mean"] == 0.0 and s_same["p"] > 0.05
      and s_same["n_paired"] == 96
      and s_same["median_v3"] == s_same["median_v2_pinned"] == 46.98)
rec = [{**r, "err": r["err"] * 0.5} for r in fc_rows]
s_rec = L.s10_paired(rec, n_perm=300)
n_nonzero = sum(1 for r in fc_rows if r["err"] > 0)   # exact hits can't improve
check("halved-error world: p at floor, d strongly negative, "
      "every nonzero row improved",
      s_rec["p"] <= 1 / 300 + 1e-9 and s_rec["d_mean"] < -15
      and s_rec["n_improved"] == n_nonzero == 94)
worse = [{**r, "err": r["err"] + 10.0} for r in fc_rows]
s_worse = L.s10_paired(worse, n_perm=300)
check("worse world: p ~ 1 (one-sided), none improved",
      s_worse["p"] > 0.99 and s_worse["n_improved"] == 0)
try:
    L.s10_paired(fc_rows[:1], n_perm=10)
    guard = False
except AssertionError:
    guard = True
check("n<2 pairing guard raises (the VM-freezer class stays dead)", guard)
alien = [{**fc_rows[0], "tid": 999999}, {**fc_rows[1], "tid": 999998}]
try:
    L.s10_paired(alien, n_perm=10)
    guard2 = False
except AssertionError:
    guard2 = True
check("unknown tids are excluded (falls to the n<2 guard)", guard2)

print("== S10' readings (protocol verbatim) ==")
check("RECOVERED reading",
      L.s10_reading({"p": 0.01, "median_v3": 30.0}).startswith("RECOVERED"))
check("PARTIAL reading",
      L.s10_reading({"p": 0.01, "median_v3": 42.0}).startswith("PARTIAL"))
check("PERSISTS reading",
      L.s10_reading({"p": 0.30, "median_v3": 30.0}).startswith("PERSISTS"))

print("== S-ABS calib: single-source tooth + planted worlds ==")
c_post = L.calib_stats(B["post"]["battery_rows"])
dc_post = DESIGN["abs_calib_post"]["per_arm"]
check("flight calib == design check on the v2 post rows (all arms, 4dp)",
      all(abs(c_post["per_arm"][a][k] - dc_post[a][k]) < 5e-5
          for a in L.ARMS
          for k in ("rho_rank", "pearson_raw", "slope", "intercept",
                    "mae_vs_target")))
check("LOO transfer matches design check",
      all(abs(c_post["loo"][a][k] - DESIGN["abs_calib_post"]["loo"][a][k])
          < 5e-5 for a in L.ARMS
          for k in ("mae_transfer", "mae_own", "transfer_penalty")))

def planted_battery(slope, seed=7):
    rng = np.random.default_rng(seed)
    rows = {}
    for arm in L.ARMS:
        n = 30
        out = []
        for i in range(n):
            if arm == "uncertainty":
                ref = {"entropy": float(rng.uniform(0.1, 3.0))}
            elif arm == "familiarity":
                ref = {"nll": float(rng.uniform(1.0, 9.0))}
            elif arm == "tension":
                ref = {"divergence": float(rng.uniform(0.05, 0.8))}
            else:
                ref = {"fill_fraction": float(rng.uniform(0.05, 0.95))}
            out.append({"id": f"{arm}{i}", "flipped": False, **ref,
                        "target_frac": ref.get("fill_fraction", 0.5)})
        oriented = [L.orient_referent(arm, r) for r in out]
        q10 = 10.0 * np.array(L.rank01(oriented))
        for r, q in zip(out, q10):
            rep = int(round(5 + slope * (q - 5)))
            r["report"] = rep
            r["raw_report"] = rep
        rows[arm] = out
    return rows

c_perf = L.calib_stats(planted_battery(1.0))
check("perfect world: slopes ~1, MAE ~0, every arm",
      all(c_perf["per_arm"][a]["slope"] > 0.9
          and c_perf["per_arm"][a]["mae_vs_target"] < 0.6 for a in L.ARMS))
c_comp = L.calib_stats(planted_battery(0.2))
check("compressed world: slopes ~.2 with rank tracking intact",
      all(0.05 < c_comp["per_arm"][a]["slope"] < 0.4
          and c_comp["per_arm"][a]["rho_rank"] > 0.8 for a in L.ARMS))

# SMOKE-RED 20260825_2119b regression teeth (empty LOO concatenate): a
# smoke-size battery leaves 0 or 1 arms above MIN_POOLED_N — calib_stats
# must degrade, never crash.
full_b = planted_battery(1.0)
one_arm = {a: (rows if a == "familiarity" else rows[:2])
           for a, rows in full_b.items()}
c_one = L.calib_stats(one_arm)
check("one usable arm: no crash, LOO skipped with the marker, others "
      "degenerate",
      c_one["loo"]["familiarity"].get("skipped") == "needs >= 2 usable arms"
      and all(c_one["per_arm"][a].get("degenerate") for a in L.ARMS
              if a != "familiarity"))
c_none = L.calib_stats({a: rows[:2] for a, rows in full_b.items()})
check("zero usable arms: all degenerate, empty LOO, no crash",
      c_none["loo"] == {}
      and all(c_none["per_arm"][a].get("degenerate") for a in L.ARMS))

print("== eval row sets at both mode constants ==")
check("full: anchor 18 · shams 12+12 · fc 96 · fc tids ⊆ baseline",
      len(L.anchor_rows(False)) == 18 and len(L.sham_rows(False)) == 12
      and len(L.supp_shams(False)) == 12
      and len(L.heldout_fc_rows(False)) == 96
      and all(t["tid"] in L.FC_BASELINE_V2 for t in L.heldout_fc_rows(False)))
check("smoke: >=1 anchor/sham · fc 4 (>=2 for the pairing guard) ⊆ baseline",
      len(L.anchor_rows(True)) >= 1 and len(L.sham_rows(True)) >= 1
      and len(L.heldout_fc_rows(True)) >= 2
      and all(t["tid"] in L.FC_BASELINE_V2 for t in L.heldout_fc_rows(True)))

print("== notebook structure + registered trims ==")
nb = json.load(open(os.path.join(HERE, "E8N3_BUDGET_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
check("9 cells; logic cell == e8n3_logic.py byte-identical",
      len(cells) == 9
      and cells[2] == open(os.path.join(HERE, "e8n3_logic.py")).read())
setup = cells[1]
check("setup: REAL-only conditions + e8n3 inflight + real-only shipped dirs",
      "CONDITIONS = ('real',)" in setup
      and "INFLIGHT = f'e8n3/inflight_" in setup
      and "for _c in ('real',):" in setup
      and "('real', 'base')" not in setup)
train = next(s for s in cells if "def train_joint" in s)
check("train: per-strand plateau wired, pooled call gone, cap V3",
      "if per_strand_plateau(strand_epoch_loss," in train
      and "if plateau_converged(" not in train
      and "EPOCHS_CAP_V3" in train)
stim = next(s for s in cells if "ANCHOR_ROWS = anchor_rows" in s)
check("stimulus: heldout generation dropped from GEN_ROWS",
      "HELDOUT_GEN = []" in stim
      and "GEN_ROWS = ANCHOR_ROWS + SHAM_ROWS_LOCKED + SUPP_SHAMS" in stim
      and "heldout_gen_rows(SMOKE)" not in stim)
# SMOKE-RED 20260825_2119 regression tooth (KeyError: 20): LAYERS_RUN must
# come from the SHIPPED bundle (what the G2 gate iterates), not from the
# trimmed rows — executed here against a two-layer shipped stub.
i0 = stim.find("LAYERS_RUN = ")
i1 = stim.find("print('eval rows:", i0)
glue = stim[i0:i1]
ns = {"SHIPPED": {"real": {"dirs": {"14": {}, "20": {}}}},
      "TRAIN_LAYER": L.TRAIN_LAYER,
      "GEN_ROWS": [{"layer": 14}, {"layer": None}],
      "FC_ROWS": [{"layer": 14}]}
exec(glue, ns)
check("LAYERS_RUN derives from SHIPPED and covers the gate's layers",
      ns["LAYERS_RUN"] == [14, 20]
      and "SHIPPED['real']['dirs']" in glue
      and "GEN_ROWS + FC_ROWS if t.get('layer')})" not in glue)
ns_bad = {**ns, "GEN_ROWS": [{"layer": 20}],
          "SHIPPED": {"real": {"dirs": {"14": {}}}}}
try:
    exec(glue, ns_bad)
    uncovered_caught = False
except AssertionError:
    uncovered_caught = True
check("a row layer outside the computed set is refused (subset assert)",
      uncovered_caught)
flight = next(s for s in cells if "def fly_condition" in s)
check("flight: paraphrase probe dropped, catch kept",
      "paraphrase_rows" not in flight
      and "'catch_rows': run_catch(h)}" in flight)
verdict = cells[-1]
check("verdict carries the v3 spine",
      all(m in verdict for m in (
          "P_V1_competence_on_budget", "P_V2_tracking_retention",
          "S0_third_replication", "S10_fc_interference", "S_ABS",
          "g_catch_pre_sanity", "fc_baseline_sha_check()",
          "FN1", "FN2", "FN3", "FN4", "before_after")))
harness = next(s for s in cells if "validate_disjoint" in s)
check("firewalls intact in harness (v1 + v2 extensions)",
      "validate_competence_domains" in harness
      and "validate_extra_disjoint" in harness)
check("no rclone anywhere (UI-only law)",
      all("rclone" not in s.lower() for s in cells))

print("== rebuild byte-identical x2 (fresh processes) ==")
nb_bytes = open(os.path.join(HERE, "E8N3_BUDGET_UI.ipynb"), "rb").read()
lg_bytes = open(os.path.join(HERE, "e8n3_logic.py"), "rb").read()
ok = True
for _ in range(2):
    r = subprocess.run([sys.executable, "build_e8n3_notebook.py"],
                       cwd=HERE, capture_output=True, text=True)
    ok &= r.returncode == 0
    ok &= open(os.path.join(HERE, "E8N3_BUDGET_UI.ipynb"), "rb").read() == nb_bytes
    ok &= open(os.path.join(HERE, "e8n3_logic.py"), "rb").read() == lg_bytes
check("two rebuilds in fresh processes are byte-identical", ok)

print()
n_fail = sum(1 for _, ok_ in passed if not ok_)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
sys.exit(1 if n_fail else 0)
