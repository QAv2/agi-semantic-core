#!/usr/bin/env python3
"""Exercise the E8-N v3 VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) FN1 — catch 1→10 + perfect tracking + FC recovered + strands
plateaued → both primaries Holm-reject, S0 passes, S10' reads RECOVERED,
S-ABS rows present, fork FN1; (2) FN2 — catch fails with the competence
strand PLATEAUED → honest-stop fork; (3) FN3 — catch fails at cap-hit with
competence still descending → single re-fly fork; (4) FN4 — tracking dead,
catch passes → retention-casualty fork; (5) catch-pre sanity (6/12 pre) →
gates NO_VERDICT; (6) sham flood → gates NO_VERDICT; (7) smoke GREEN path;
(8) incomplete flight → no primaries, no fork, resume banner.

Run:  python3 colab/test_e8n3_verdict.py   (numpy only; ~2-4 min)
"""
import copy, io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n3_logic as L
import e8n2_pools as P
from build_e8n_notebook import build_payload

L.N_PERM = 300
L.N_BOOT = 300

nb = json.load(open(os.path.join(HERE, "E8N3_BUDGET_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P_V1_competence_on_budget" in VERDICT_SRC, "last cell not the verdict"

PAYLOAD = json.loads(build_payload())
PAYLOAD["competence_items"] = P.COMPETENCE_ITEMS
PAYLOAD["lexicon_paraphrases"] = P.LEXICON_PARAPHRASES
FILLS = [0.05, 0.35, 0.75]
CONDITIONS = ("real",)
VEC = {c["name"]: c["vec"]
       for c in json.load(open(os.path.join(HERE,
                                            "e4_dictionary_pack.json")))["concepts"]
       if c["name"] in L.CHOICE_SET}

BAT = copy.deepcopy(PAYLOAD["battery"]["arms"])
for arm in BAT.values():
    for i, it in enumerate(arm["items"]):
        it["flipped"] = i % 2 == 1


def synth_battery_rows(style, seed):
    rng = np.random.default_rng(seed)
    rows = {}
    for arm in L.ARMS:
        items = BAT[arm]["items"]
        stims = ([(it, f) for it in items for f in FILLS]
                 if arm == "saturation" else [(it, None) for it in items])
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
            elif style == "weakpos":
                rep = lab if rng.random() < 0.45 else int(rng.integers(0, 11))
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


def synth_catch(k):
    rows = []
    for i, it in enumerate(PAYLOAD["catch_trials"]):
        rep = it["known"] if i < k else (it["known"] + 5 if it["known"] <= 5
                                         else it["known"] - 5)
        rows.append({"id": it["id"], "flipped": it["flipped"],
                     "known": it["known"], "report": rep, "raw_report": rep})
    return rows


def synth_injection(anchor_exact=18, sham_claims=0, smoke=False):
    rows = []
    for i, t in enumerate(L.anchor_rows(smoke)):
        rep = t["concept"] if i < anchor_exact else "NONE"
        rows.append({**t, "report": rep, "response": rep, "hook_calls": 3})
    for i, t in enumerate(L.sham_rows(smoke) + L.supp_shams(smoke)):
        rep = "CONFIDENCE" if i < sham_claims else "NONE"
        rows.append({**t, "report": rep, "response": rep, "hook_calls": 0})
    return rows


def synth_fc(fc_style, smoke=False):
    """heldout_fc errs relative to the PINNED v2 baseline + sham_fc rows."""
    rows = []
    for t in L.heldout_fc_rows(smoke):
        base = L.FC_BASELINE_V2[t["tid"]]
        if fc_style == "recovered":
            err = base * 0.4
        elif fc_style == "partial":
            err = max(5.0, base - 8.0)
        else:                       # 'unchanged'
            err = base
        rows.append({"tid": t["tid"], "block": "heldout_fc",
                     "layer": t["layer"], "alpha": t["alpha"],
                     "injected": t["concept"], "argmax": "UNCERTAINTY",
                     "exact": False, "none_top": False, "err": err,
                     "hook_calls": 1, "cond": "real"})
    for t in L.sham_rows(smoke):
        rows.append({"tid": t["tid"], "block": "sham_fc", "layer": None,
                     "alpha": 0.0, "injected": None, "argmax": "UNCERTAINTY",
                     "none_top": True, "hook_calls": 0, "cond": "real"})
    return rows


def strand_curves(profile):
    """Synthetic per-strand epoch curves for the FN2/FN3 discrimination."""
    if profile == "plateaued9":       # FN1: all strands flat by epoch 9
        cur, out = {"scalar": 1.2, "naming": 1.2, "competence": 1.0,
                    "lexicon": 0.2}, []
        rates = {"scalar": 0.02, "naming": 0.5, "competence": 0.3,
                 "lexicon": 0.5}
        for ep in range(9):
            cur = {s: round(cur[s] * (1 - (rates[s] if ep < 6 else 0.01)), 4)
                   for s in cur}
            out.append(dict(cur))
        return out, 9, True
    if profile == "comp_plateaued":   # FN2: competence flat below the bar
        out = [{"scalar": 1.0, "naming": 0.1, "competence": 0.60,
                "lexicon": 0.01} for _ in range(7)]
        out = [{s: round(v * (1 - 0.3 / (i + 1)), 4) for s, v in e.items()}
               for i, e in enumerate(out)]
        out[-1] = {s: round(out[-2][s] * 0.99, 4) for s in out[-2]}
        return out, len(out), True
    if profile == "cap_descending":   # FN3: 12 epochs, competence -15%/ep
        cur, out = {"scalar": 1.2, "naming": 1.2, "competence": 1.0,
                    "lexicon": 0.2}, []
        for _ in range(12):
            cur = {"scalar": round(cur["scalar"] * 0.99, 4),
                   "naming": round(cur["naming"] * 0.7, 4),
                   "competence": round(cur["competence"] * 0.85, 4),
                   "lexicon": round(cur["lexicon"] * 0.7, 4)}
            out.append(dict(cur))
        return out, 12, False
    raise ValueError(profile)


def bundle(style="perfect", catch_pre=1, catch_post=10, fc_style="recovered",
           anchor_exact=18, sham_claims=0, profile="plateaued9",
           curriculum=None, seed=0, smoke=False, no_post=False):
    curves, eps, plat = strand_curves(profile)
    b = {
        "condition": "real", "mode": "smoke" if smoke else "full",
        "stamp": "test",
        "curriculum": curriculum or dict(L.EXPECT_CURRICULUM_V3),
        "dirs_stability": {"resid": 3.9e-06, "layer": 14, "name": "CAPTURE",
                           "tol": 1e-3, "pass": True},
        "mu": {"14": 80.0, "20": 120.0},
        "pool_referents": {a: {} for a in L.ARMS},
        "train_labels": [],
        "train_log": {"n_examples": 600, "epochs_flown": eps,
                      "epochs_cap": L.EPOCHS_CAP_V3, "plateaued": plat,
                      "plateau_rel": 0.05, "min_epochs": 3,
                      "epoch_means": [round(float(np.mean(list(e.values()))), 4)
                                      for e in curves],
                      "strand_epoch_loss": curves,
                      "micro_steps": 600 * eps, "opt_steps": 75 * eps,
                      "hook_calls": 576, "trainable_params": 4358144,
                      "max_example_tokens": 6200, "peak_vram_gb": 11.7,
                      "secs": 500.0 * eps, "loss_first_k": 2.4,
                      "final_smoothed": 0.5, "losses_every_10": []},
        "took": {"n": 24, "within1": 22, "frac": 0.917, "pass": True,
                 "rows": []},
        "took_competence": {"n": 6, "within_tol": 6, "pass": True, "rows": []},
        "took_lexicon": {"n": 4, "exact": 4, "pass": True, "rows": []},
        "pre": {"catch_rows": synth_catch(catch_pre),
                "battery_rows": synth_battery_rows("weakpos", seed + 1),
                "arm_errors": {}},
        "ppl_pre": 11.0, "ppl_post": 11.15, "secs": 6000.0,
    }
    if not no_post:
        b["post"] = {"battery_rows": synth_battery_rows(style, seed + 2),
                     "arm_errors": {}, "catch_rows": synth_catch(catch_post)}
        b["injection_rows"] = synth_injection(anchor_exact, sham_claims, smoke)
        b["fc_rows"] = synth_fc(fc_style, smoke)
    return b


def run_verdict(results, cond_errors, smoke=False, make_shipped=True):
    td = tempfile.mkdtemp()
    out = Path(td) / "out"; out.mkdir()
    sem = Path(td) / "sem"
    inflight = sem / "e8n3" / "inflight_test"
    inflight.mkdir(parents=True)
    if make_shipped:
        for c in results:
            (inflight / f"condition_{c}.json").write_text("{}")
            rd = inflight / f"readout_{c}"; rd.mkdir()
            (rd / "adapter_config.json").write_text("{}")
    g = {n: getattr(L, n) for n in dir(L) if not n.startswith("_")}
    buf = io.StringIO()
    g.update(dict(
        json=json, np=np, copy=copy,
        SMOKE=smoke, MODE="smoke" if smoke else "full", STAMP="test",
        MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct", RESUME_STAMP="",
        CONDITIONS=CONDITIONS, PAYLOAD=PAYLOAD, VEC=VEC,
        RESULTS=results, cond_errors=cond_errors,
        OUT=out, SEM=sem, INFLIGHT="e8n3/inflight_test",
        ship=lambda *a, **k: None,
    ))
    with redirect_stdout(buf):
        exec(VERDICT_SRC, g)
    return json.load(open(out / "e8n3_verdict.json")), buf.getvalue()


passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


print("== scenario 1: FN1 — the cure lands ==")
v, out1 = run_verdict({"real": bundle(seed=10)}, {})
pr = v["primaries"]
check("P-V1: catch 1->10, abs pass + tiny p + Holm",
      pr["P_V1_competence_on_budget"]["abs_pass"]
      and pr["P_V1_competence_on_budget"]["p_effective"] < 1e-5
      and pr["P_V1_competence_on_budget"]["pass"])
check("P-V2: rho > .9, p at floor, n 148, pass",
      pr["P_V2_tracking_retention"]["rho"] > 0.9
      and pr["P_V2_tracking_retention"]["p"] < 0.01
      and pr["P_V2_tracking_retention"]["n"] == 148
      and pr["P_V2_tracking_retention"]["pass"])
sec = v["secondaries"]
check("S0 passes on weak-positive pre with the band recorded",
      sec["S0_third_replication"]["pass"]
      and sec["S0_third_replication"]["predicted_band"] == [0.1, 0.3])
s10 = sec["S10_fc_interference"]
check("S10' recovered world: p floor, median ~18.8, reading RECOVERED",
      s10["p"] <= 1 / 300 + 1e-9 and s10["median_v3"] < 20
      and s10["reading"].startswith("RECOVERED"))
check("S-ABS rows present, perfect world slopes ~1 on post "
      "(saturation .87 by construction: 3 quantized levels + rounding)",
      all(sec["S_ABS"]["post"]["per_arm"][a]["slope"] > 0.8 for a in L.ARMS)
      and "pre" in sec["S_ABS"])
check("gates all green incl. catch-pre sanity",
      all(g_.get("pass") for g_ in v["gates"].values()))
check("before/after carries catch [1,10] + fc pins",
      v["before_after"]["catch"][:2] == [1, 10]
      and v["before_after"]["fc_median"][0] == 46.98)
check("fork = FN1 with the RECOVERED reading folded in",
      v["fork"].startswith("FN1") and "RECOVERED" in v["fork"])

print("== scenario 2: FN2 — competence plateaued below the bar ==")
v2, _ = run_verdict({"real": bundle(catch_post=5, fc_style="unchanged",
                                    profile="comp_plateaued", seed=20)}, {})
check("P-V1 fails (5/12), p_effective 1.0",
      not v2["primaries"]["P_V1_competence_on_budget"]["pass"]
      and v2["primaries"]["P_V1_competence_on_budget"]["p_effective"] == 1.0)
check("competence strand read as NOT still-descending",
      v2["conditions"]["real"]["competence_strand"]["still_descending"] is False)
check("fork = FN2 honest stop", v2["fork"].startswith("FN2"))

print("== scenario 3: FN3 — cap-hit still descending ==")
v3, _ = run_verdict({"real": bundle(catch_post=5, fc_style="unchanged",
                                    profile="cap_descending", seed=30)}, {})
check("undertrained flag set (cap-hit, no plateau)",
      v3["conditions"]["real"]["undertrained"] is True)
check("fork = FN3 with the single re-fly clause",
      v3["fork"].startswith("FN3") and "re-fly" in v3["fork"]
      and "cap 18" in v3["fork"])

print("== scenario 4: FN4 — tracking regresses, catch passes ==")
v4, _ = run_verdict({"real": bundle(style="null", catch_post=10, seed=40)}, {})
check("P-V1 passes, P-V2 fails",
      v4["primaries"]["P_V1_competence_on_budget"]["abs_pass"]
      and not v4["primaries"]["P_V2_tracking_retention"]["pass"])
check("fork = FN4 retention casualty", v4["fork"].startswith("FN4"))

print("== scenario 5: catch-pre sanity gate ==")
v5, _ = run_verdict({"real": bundle(catch_pre=6, catch_post=11, seed=50)}, {})
check("g_catch_pre_sanity fails at 6/12; NO_VERDICT gates fork",
      not v5["gates"]["g_catch_pre_sanity"]["pass"]
      and v5["fork"].startswith("FN-GATES")
      and "g_catch_pre_sanity" in v5["fork"])

print("== scenario 6: sham flood -> gates dirty ==")
v6, _ = run_verdict({"real": bundle(sham_claims=6, seed=60)}, {})
check("g5 fails at 6/24; NO_VERDICT gates fork",
      not v6["gates"]["g5_sham"]["pass"] and v6["fork"].startswith("FN-GATES"))

print("== scenario 7: smoke GREEN path ==")
v7, out7 = run_verdict({"real": bundle(seed=70, smoke=True)}, {}, smoke=True)
check("smoke: no primaries, real scored, banner GREEN",
      "primaries" not in v7 and "real" in v7["conditions"]
      and "SMOKE GREEN" in out7)

print("== scenario 8: incomplete flight ==")
v8, out8 = run_verdict({"real": bundle(no_post=True, seed=80)},
                       {"real": "RuntimeError: boom"}, make_shipped=False)
check("no primaries, no fork, resume banner printed",
      "primaries" not in v8 and v8.get("fork") is None
      and "INCOMPLETE" in out8 and "RESUME_STAMP" in out8)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
sys.exit(1 if n_fail else 0)
