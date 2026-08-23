#!/usr/bin/env python3
"""Exercise the E8-N VERDICT cell VERBATIM from the built notebook against
synthetic flight results (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) perfect tracking both conditions -> P1+P2 pass, Holm rejects,
S-blocks assemble; (2) null tracking -> P1 not rejected; (3) straight-only
gamed tracking -> P1 rejects, P2 does not (flipped subset flat); (4) catch
failure -> P2 p pinned to 1.0; (5) degenerate constant reports -> no crash,
P1 p=1.0; (6) partial flight (real only) -> NO primaries (pre-reg hygiene);
(7) smoke branch -> checks print, no primaries; (8) undertrained flag
propagates into S2 convergence parity.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n_verdict.py
"""
import copy, json, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n_logic as L
from build_e8n_notebook import build_payload

L.N_PERM = 300      # late-bound in the logic signatures — fast test runs
L.N_BOOT = 300

nb = json.load(open(os.path.join(HERE, "E8N_NATURAL_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P_E8N_1_tracking" in VERDICT_SRC, "last cell is not the verdict cell"

PAYLOAD = json.loads(build_payload())
FILLS = [0.05, 0.35, 0.75]
CONDITIONS = ("real", "base")

BAT = copy.deepcopy(PAYLOAD["battery"]["arms"])
for arm in BAT.values():
    for i, it in enumerate(arm["items"]):
        it["flipped"] = i % 2 == 1


def synth_battery_rows(style, seed):
    """Runner-shaped rows with synthetic referents; report per style."""
    rng = np.random.default_rng(seed)
    rows = {}
    for arm in L.ARMS:
        items = BAT[arm]["items"]
        if arm == "saturation":
            stims = [(it, f) for it in items for f in FILLS]
        else:
            stims = [(it, None) for it in items]
        refs = []
        for it, f in stims:
            if arm == "uncertainty":
                refs.append({"entropy": float(rng.uniform(0.1, 3.0))})
            elif arm == "familiarity":
                refs.append({"nll": float(rng.uniform(1.0, 9.0))})
            elif arm == "tension":
                refs.append({"divergence": float(rng.uniform(0.05, 0.8))})
            else:
                refs.append({"fill_fraction": f})
        oriented = [L.orient_referent(arm, r) for r in refs]
        labels = [int(round(10 * q)) for q in L.rank01(oriented)]
        out = []
        for (it, f), ref, lab in zip(stims, refs, labels):
            if style == "perfect":
                rep = lab
            elif style == "null":
                rep = int(rng.integers(0, 11))
            elif style == "straight_only":
                rep = lab if not it["flipped"] else int(rng.integers(0, 11))
            elif style == "degenerate":
                rep = 7
            else:
                raise ValueError(style)
            row = {"id": it["id"], "flipped": it["flipped"],
                   "report": rep, "raw_report": rep}
            row.update(ref)
            if arm == "saturation":
                row["target_frac"] = f
                row["needle_correct"] = True
            out.append(row)
        rows[arm] = out
    return rows


def synth_catch(ok=True):
    rows = []
    for it in PAYLOAD["catch_trials"]:
        rep = it["known"] if ok else 5 if it["known"] != 1 else 9
        rows.append({"id": it["id"], "flipped": it["flipped"],
                     "known": it["known"], "report": rep, "raw_report": rep})
    return rows


def bundle(cond, style, pre_style="null", catch_ok=True, reached_tau=True, seed=0):
    return {
        "condition": cond, "mode": "full", "stamp": "test",
        "pool_referents": {a: {} for a in L.ARMS},
        "train_labels": [],
        "train_log": {"n_examples": 272, "epochs_flown": 4 if reached_tau else 8,
                      "epochs_cap": 8, "reached_tau": reached_tau, "tau": 0.25,
                      "micro_steps": 1088, "opt_steps": 136,
                      "trainable_params": 4358144, "max_example_tokens": 6200,
                      "peak_vram_gb": 6.4,
                      "secs": 900.0, "loss_first_k": 2.1,
                      "final_smoothed": 0.12 if reached_tau else 0.61,
                      "losses_every_10": [2.1, 0.12]},
        "took": {"n": 24, "within1": 22, "frac": 0.917, "pass": True, "rows": []},
        "pre": {"battery_rows": synth_battery_rows(pre_style, seed + 1),
                "arm_errors": {}, "catch_rows": synth_catch(False)},
        "post": {"battery_rows": synth_battery_rows(style, seed + 2),
                 "arm_errors": {}, "catch_rows": synth_catch(catch_ok),
                 "paraphrase_rows": []},
        "ppl_pre": 11.0, "ppl_post": 11.2, "secs": 3500.0,
    }


def run_verdict(results, cond_errors, smoke=False, make_shipped=True):
    td = tempfile.mkdtemp()
    out = Path(td) / "out"; out.mkdir()
    sem = Path(td) / "sem"
    inflight = sem / "e8n" / "inflight_test"
    inflight.mkdir(parents=True)
    if make_shipped:
        for c in results:
            (inflight / f"condition_{c}.json").write_text("{}")
            rd = inflight / f"readout_{c}"; rd.mkdir()
            (rd / "adapter_config.json").write_text("{}")
    g = {n: getattr(L, n) for n in dir(L) if not n.startswith("_")}
    g.update(dict(
        json=json, np=np, copy=copy,
        SMOKE=smoke, MODE="smoke" if smoke else "full", STAMP="test",
        MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct", RESUME_STAMP="",
        CONDITIONS=CONDITIONS, PAYLOAD=PAYLOAD,
        RESULTS=results, cond_errors=cond_errors,
        OUT=out, SEM=sem, INFLIGHT="e8n/inflight_test",
        ship=lambda *a, **k: None, print=lambda *a, **k: None,
    ))
    exec(VERDICT_SRC, g)
    return json.load(open(out / "e8n_verdict.json"))


passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


print("== scenario 1: perfect tracking, both conditions ==")
res = {"real": bundle("real", "perfect", seed=10),
       "base": bundle("base", "perfect", seed=20)}
v = run_verdict(res, {})
p1 = v["primaries"]["P_E8N_1_tracking"]
p2 = v["primaries"]["P_E8N_2_reading_not_gaming"]
check("P1 planted signal: rho > .9, p < .01, n = 148",
      p1["rho"] > 0.9 and p1["p"] < 0.01 and p1["n"] == 148)
check("P1 bootstrap CI attached and above zero", p1["ci"][0] > 0.5)
check("P2 flipped subset tracks + catch passes",
      p2["flipped_subset"]["p"] < 0.01 and p2["catch"]["pass"] and p2["p"] < 0.01)
check("Holm rejects both", all(rej for _, rej in v["primaries"]["holm"].values()))
check("locked baseline computed from embedded rows (n = 148)",
      v["locked_baseline"]["pooled"]["n"] == 148 and
      v["locked_baseline"]["pooled"]["rho"] is not None)
check("S0 pre-tracking present for both conditions",
      set(v["secondaries"]["S0_pre_tracking"]) == {"real", "base"})
check("S1 per-arm block with Holm over 4 arms",
      set(v["secondaries"]["S1_per_arm_real_post"]["arms"]) == set(L.ARMS) and
      len(v["secondaries"]["S1_per_arm_real_post"]["holm"]) == 4)
check("S2 delta real-base ~0 for identical styles",
      abs(v["secondaries"]["S2_delta_real_minus_base"]["delta"]) < 0.1)
check("S3 post-pre delta positive with CI > 0 (pre was null)",
      v["secondaries"]["S3_delta_post_minus_pre_real"]["delta"] > 0.5 and
      v["secondaries"]["S3_delta_post_minus_pre_real"]["ci95"][0] > 0)
check("S4 straight and flipped both ~1",
      v["secondaries"]["S4_straight_vs_flipped_real_post"]["straight"]["rho"] > 0.9 and
      v["secondaries"]["S4_straight_vs_flipped_real_post"]["flipped"]["rho"] > 0.9)
check("S5 drift table computed on shared ids",
      all(v["secondaries"]["S5_referent_drift_base_vs_locked"][a]["n"] >= 4
          for a in ("uncertainty", "familiarity", "tension")))
check("ppl delta computed", v["conditions"]["real"]["ppl_delta_pct"] is not None)
check("catch recorded pre (fail) and post (pass)",
      not v["conditions"]["real"]["pre"]["catch"]["pass"] and
      v["conditions"]["real"]["post"]["catch"]["pass"])

print("== scenario 2: null tracking ==")
res2 = {"real": bundle("real", "null", seed=30),
        "base": bundle("base", "null", seed=40)}
v2 = run_verdict(res2, {})
check("P1 not rejected on noise",
      not v2["primaries"]["holm"]["P1"][1] and v2["primaries"]["P_E8N_1_tracking"]["p"] > 0.05)

print("== scenario 3: straight-only gamed tracking ==")
res3 = {"real": bundle("real", "straight_only", seed=50),
        "base": bundle("base", "straight_only", seed=60)}
v3 = run_verdict(res3, {})
check("P1 rejects (diluted but real signal)", v3["primaries"]["holm"]["P1"][1])
check("P2 does NOT reject (flipped subset flat)",
      not v3["primaries"]["holm"]["P2"][1] and
      v3["primaries"]["P_E8N_2_reading_not_gaming"]["flipped_subset"]["p"] > 0.05)

print("== scenario 4: perfect tracking, catch failure ==")
res4 = {"real": bundle("real", "perfect", catch_ok=False, seed=70),
        "base": bundle("base", "perfect", seed=80)}
v4 = run_verdict(res4, {})
check("P2 p pinned to 1.0 when catch fails",
      v4["primaries"]["P_E8N_2_reading_not_gaming"]["p"] == 1.0 and
      not v4["primaries"]["holm"]["P2"][1])
check("P1 unaffected", v4["primaries"]["holm"]["P1"][1])

print("== scenario 5: degenerate constant reports ==")
res5 = {"real": bundle("real", "degenerate", seed=90),
        "base": bundle("base", "degenerate", seed=91)}
v5 = run_verdict(res5, {})
check("no crash; P1 rho None, p = 1.0",
      v5["primaries"]["P_E8N_1_tracking"]["rho"] is None and
      v5["primaries"]["P_E8N_1_tracking"]["p"] == 1.0)

print("== scenario 6: partial flight (real only) ==")
v6 = run_verdict({"real": bundle("real", "perfect", seed=95)}, {})
check("no primaries on 1/2 (pre-reg hygiene), complete=[real]",
      "primaries" not in v6 and "secondaries" not in v6 and
      v6["complete_conditions"] == ["real"])

print("== scenario 7: smoke branch ==")
v7 = run_verdict(res, {}, smoke=True)
check("smoke: no primaries, conditions scored",
      "primaries" not in v7 and set(v7["conditions"]) == {"real", "base"})

print("== scenario 8: undertrained flag propagates ==")
res8 = {"real": bundle("real", "perfect", seed=96),
        "base": bundle("base", "perfect", reached_tau=False, seed=97)}
v8 = run_verdict(res8, {})
check("S2 convergence parity flags base UNDERTRAINED",
      v8["secondaries"]["S2_convergence_parity"]["base"]["undertrained"] and
      not v8["secondaries"]["S2_convergence_parity"]["real"]["undertrained"])
check("condition entry carries the flag", v8["conditions"]["base"]["undertrained"])

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
