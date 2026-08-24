#!/usr/bin/env python3
"""Exercise the E8-J VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) FORK 1 — gates pass, P-A + G-B + P-B all pass; (2) FORK 2 —
P-A passes, G-B fails, Stage B skipped; (3) FORK 3 — nothing corresponds;
(4) FORK 4 — G-B passes, arms tie, P-B fails; (5) gate failure -> NO_VERDICT;
(6) smoke branch -> checks dict all green -> SMOKE GREEN; (7) flight error ->
NO FLIGHT DATA. Banner strings, shipped verdict fields, and fork lines are
asserted on every path.

Run:  python3 colab/test_e8j_verdict.py
"""
import io, contextlib, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8J_BRIDGE_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "FORK" in VERDICT_SRC and "e8j_verdict.json" in VERDICT_SRC, \
    "last cell is not the verdict"

DER = L.wing_derangement()

def synth_rows(real_exact_targets, perm_exact_targets=set(), smoke=False):
    rows = []
    for t in L.build_b_pairs(smoke):
        exact = (t["concept"] in real_exact_targets if t["arm"] == "real"
                 else t["concept"] in perm_exact_targets)
        arg = t["concept"] if exact else (
            DER[t["concept"]] if t["arm"] == "perm" else "NONE")
        rows.append({"tid": t["tid"], "block": t["block"], "layer": 14,
                     "alpha": t["alpha"], "injected": t["concept"],
                     "argmax": arg, "exact": exact, "none_top": False,
                     "hook_calls": 1, "arm": t["arm"], "pair_id": t["pair_id"],
                     "scores": {n: -2.0 for n in L.SCORED_SET},
                     "scores_sum": {n: -4.0 for n in L.SCORED_SET}})
    return rows

def synth_atlas(pa_pass, gb_pass, geo_hits=None):
    if geo_hits is None:
        geo_hits = set(L.TRAINED) if gb_pass else set()
    per = {c: {"angle": 30.0, "nearest": c if c in geo_hits else DER[c],
               "hit": c in geo_hits, "lookup_floor": 40.0}
           for c in L.CHOICE_SET}
    return {
        "stamp": "20260824_TEST", "n_anchors": 320, "anchors_sha": L.ANCHORS_SHA,
        "lam": L.LAM, "smoke": False,
        "pins": {"n_anchors": 320, "sha": L.ANCHORS_SHA},
        "A1_rsa_L14": {"rho": 0.2, "p": 0.001, "n_perm": 10000},
        "S6_folded_L14": {"rho": 0.1, "p": 0.1},
        "A2_loo": {"median_angle": 55.0, "null_median_mean": 72.0,
                   "p_median": 0.001},
        "A3_retrieval": {"top1": 90 if pa_pass else 2, "top5": 150, "n": 320,
                         "null_top1_mean": 1.0, "null_top1_max": 4,
                         "p_top1": 0.0005 if pa_pass else 0.6},
        "P_A": {"top1": 90 if pa_pass else 2,
                "p": 0.0005 if pa_pass else 0.6, "pass": bool(pa_pass)},
        "S5_lam_sens": {"0.1": {}, "10.0": {}},
        "A4_axiswise_r2": {a: 0.4 for a in L.AXIS_NAMES},
        "A5_wing": {"per_target": per, "hits": len(geo_hits),
                    "median_angle": 35.0, "null_median_mean": 70.0,
                    "p_median": 0.001},
        "gb": {"hits": len(geo_hits), "min_hits": L.GB_MIN_HITS,
               "null_p95": 2.0, "null_mean": 1.0, "p_hits": 0.001,
               "pass": bool(gb_pass)},
        "wing_pred_real": {c: [0.1] * 4 for c in L.CHOICE_SET},
        "derangement": DER,
        "S3_L20": {"rsa_rho": 0.1, "rsa_p": 0.1, "loo_median": 60.0, "top1": 20},
        "S4_base": {"rsa_rho": 0.05, "rsa_p": 0.3, "loo_median": 70.0,
                    "top1": 5, "delta_top1_inst_minus_base": 85},
        "secs_stageA": 111.0,
    }

def synth_stageb(rows):
    return {"stamp": "20260824_TEST", "mode": "full",
            "anchor": [{"exact": True} for _ in range(18)],
            "shams": [{"none_top": True, "hook_calls": 0,
                       "scores": {n: -2.0 for n in L.SCORED_SET}}
                      for _ in range(12)],
            "sham_claims": 0, "fc_rows": rows, "titr_rows": [],
            "derangement": DER, "secs": 999.0}

def run_cell(atlas, stageb, gates, smoke, b_rows=None, pre_ship_atlas=False,
             cond_errors=None):
    tmp = Path(tempfile.mkdtemp())
    sem = tmp / "sem"
    out = tmp / "out"
    out.mkdir(parents=True)
    inflight = "e8j/inflight_TEST"
    if pre_ship_atlas:
        (sem / inflight).mkdir(parents=True, exist_ok=True)
        (sem / inflight / "atlas.json").write_text("{}")
    def ship(src, dest_rel):
        dest = sem / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(src)
        files = (sorted(p for p in src.iterdir() if p.is_file())
                 if src.is_dir() else [src])
        for p in files:
            shutil.copy2(p, dest / p.name)
    ns = {k: getattr(L, k) for k in dir(L) if not k.startswith("__")}
    ns.update({"ATLAS": atlas, "STAGEB": stageb, "GATES": gates,
               "cond_errors": cond_errors or {}, "SMOKE": smoke,
               "MODE": "smoke" if smoke else "full",
               "MODEL_ID": "Qwen/Qwen2.5-1.5B-Instruct",
               "OUT": out, "SEM": sem, "INFLIGHT": inflight,
               "STAMP": "20260824_TEST", "RESUME_STAMP": "",
               "B_PAIR_ROWS": b_rows if b_rows is not None
               else L.build_b_pairs(smoke),
               "ship": ship, "json": json, "np": np, "Path": Path})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(VERDICT_SRC, ns)
    text = buf.getvalue()
    vf = out / "e8j_verdict.json"
    verdict = json.load(open(vf)) if vf.exists() else None
    shutil.rmtree(tmp, ignore_errors=True)
    return text, verdict

GATES_OK = {"g2": {"resid": 5e-08, "pass": True},
            "g1b": {"delta_pct": 0.0, "pass": True},
            "g1": {"n": 18, "exact": 18, "pass": True}}

print("== scenario 1: FORK 1 (everything lands) ==")
rows = synth_rows(set(L.TRAINED))
text, v = run_cell(synth_atlas(True, True), synth_stageb(rows),
                   dict(GATES_OK), smoke=False, b_rows=L.build_b_pairs(False))
check("banner prints FORK 1", "FORK 1" in text)
check("banner shows P-B naming counts", "P-B naming: real 72 vs perm 0" in text)
check("verdict fork field FORK 1", v and v["fork"].startswith("FORK 1"))
check("verdict gates_all_pass", v and v["gates_all_pass"] is True)
check("stage B P_B pass in verdict", v and v["stageb"]["P_B"]["pass"] is True)
check("mechanism subset ok", v and v["stageb"]["mechanism"]["subset_ok"])

print("== scenario 2: FORK 2 (G-B fails, Stage B skipped) ==")
text, v = run_cell(synth_atlas(True, False), None,
                   {"g2": GATES_OK["g2"], "g1b": GATES_OK["g1b"]}, smoke=False)
check("banner prints STAGE B SKIPPED", "STAGE B SKIPPED" in text)
check("fork 2 in verdict", v and v["fork"].startswith("FORK 2"))
check("no stageb block", v and v["stageb"] is None)

print("== scenario 3: FORK 3 (no correspondence) ==")
text, v = run_cell(synth_atlas(False, False), None,
                   {"g2": GATES_OK["g2"], "g1b": GATES_OK["g1b"]}, smoke=False)
check("fork 3 in verdict", v and v["fork"].startswith("FORK 3"))

print("== scenario 4: FORK 4 (geometry transfers, readout does not) ==")
rows_tie = synth_rows(set(), set())
text, v = run_cell(synth_atlas(True, True), synth_stageb(rows_tie),
                   dict(GATES_OK), smoke=False)
check("fork 4 in verdict", v and v["fork"].startswith("FORK 4"))
check("P_B fail recorded", v and v["stageb"]["P_B"]["pass"] is False)

print("== scenario 5: gate failure -> NO_VERDICT ==")
bad = dict(GATES_OK)
bad["g2"] = {"resid": 0.5, "pass": False}
text, v = run_cell(synth_atlas(True, True), synth_stageb(rows), bad,
                   smoke=False)
check("NO_VERDICT fork on gate failure",
      v and v["fork"].startswith("NO_VERDICT"))
check("gates_all_pass false", v and v["gates_all_pass"] is False)

print("== scenario 6: smoke branch ==")
smoke_rows = synth_rows({L.TRAINED[0]}, smoke=True)
atl_s = synth_atlas(True, True)
atl_s["smoke"] = True
sb_s = synth_stageb(smoke_rows)
sb_s["anchor"] = [{"exact": True} for _ in range(2)]
sb_s["shams"] = [{"none_top": True, "hook_calls": 0,
                  "scores": {n: -2.0 for n in L.SCORED_SET}}] * 2
text, v = run_cell(atl_s, sb_s, {"g2": GATES_OK["g2"],
                                 "g1b": GATES_OK["g1b"],
                                 "g1": {"n": 2, "exact": 2, "pass": True,
                                        "smoke": True}},
                   smoke=True, b_rows=L.build_b_pairs(True),
                   pre_ship_atlas=True)
check("SMOKE GREEN banner", "SMOKE GREEN" in text)
check("smoke checks all true", '"pb_stats_ran": true' in text
      and '"stageb_forced_flown": true' in text
      and '"atlas_fields": true' in text)

print("== scenario 6b: smoke RED when rows incomplete ==")
sb_bad = dict(sb_s)
sb_bad["fc_rows"] = smoke_rows[:-1]
text, v = run_cell(atl_s, sb_bad, {"g2": GATES_OK["g2"],
                                   "g1b": GATES_OK["g1b"]},
                   smoke=True, b_rows=L.build_b_pairs(True),
                   pre_ship_atlas=True)
check("SMOKE RED on incomplete rows", "SMOKE RED" in text)

print("== scenario 7: flight error -> NO FLIGHT DATA ==")
text, v = run_cell(None, None, None, smoke=False,
                   cond_errors={"flight": "RuntimeError: boom"})
check("NO FLIGHT DATA banner", "NO FLIGHT DATA" in text)
check("no verdict shipped", v is None)

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
