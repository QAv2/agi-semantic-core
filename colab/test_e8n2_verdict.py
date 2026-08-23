#!/usr/bin/env python3
"""Exercise the E8-N v2 VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) full pass — perfect tracking + catch 1→12 + own-name held-out
naming + FC good + anchors 18/18 + silence held → all four primaries Holm-
reject, S0 claim passes, gates green; (2) P2 floor-stuck (catch 1→1) → P2
p = 1.0; (3) P3 silence → n-guard fork, both P3 p = 1.0, P1/P2 still
testable; (4) nearest-name production → P3a rejects, P3b does not; (5) gate
regressions (sham claims 6/24, anchor 14/18) → gates record fails; (6)
partial flight → no primaries (pre-reg hygiene); (7) smoke green path — all
mechanics checks true, no primaries; (8) degenerate constant battery → P1
p = 1.0, no crash.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n2_verdict.py
"""
import copy, json, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n2_logic as L
import e8n2_pools as P
from build_e8n_notebook import build_payload

L.N_PERM = 300
L.N_BOOT = 300

nb = json.load(open(os.path.join(HERE, "E8N2_JOINT_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P3a_production_reads_geometry" in VERDICT_SRC, "last cell is not the verdict"

PAYLOAD = json.loads(build_payload())
PAYLOAD["competence_items"] = P.COMPETENCE_ITEMS
PAYLOAD["lexicon_paraphrases"] = P.LEXICON_PARAPHRASES
FILLS = [0.05, 0.35, 0.75]
CONDITIONS = ("real", "base")
VEC = {c["name"]: c["vec"]
       for c in json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))["concepts"]
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
        stims = ([(it, f) for it in items for f in FILLS] if arm == "saturation"
                 else [(it, None) for it in items])
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


def synth_catch(k):
    """Exactly k of 12 within tolerance."""
    rows = []
    for i, it in enumerate(PAYLOAD["catch_trials"]):
        rep = it["known"] if i < k else (it["known"] + 5 if it["known"] <= 5
                                         else it["known"] - 5)
        rows.append({"id": it["id"], "flipped": it["flipped"],
                     "known": it["known"], "report": rep, "raw_report": rep})
    return rows


def synth_injection(ho_style, anchor_exact=18, sham_claims=0, seed=0, smoke=False):
    rng = np.random.default_rng(seed)
    rows = []
    anchors = L.anchor_rows(smoke)
    for i, t in enumerate(anchors):
        rep = t["concept"] if i < anchor_exact else "NONE"
        rows.append({**t, "report": rep, "response": rep, "hook_calls": 3})
    shams = L.sham_rows(smoke) + L.supp_shams(smoke)
    for i, t in enumerate(shams):
        rep = "CONFIDENCE" if i < sham_claims else "NONE"
        rows.append({**t, "report": rep, "response": rep, "hook_calls": 0})
    for t in L.heldout_gen_rows(smoke):
        r = {**t, "hook_calls": 2}
        tr_reg = t["layer"] == L.TRAIN_LAYER and t["alpha"] in L.TRAIN_ALPHAS
        if ho_style == "own" and tr_reg:
            r["report"] = t["concept"]
        elif ho_style == "nearest" and tr_reg:
            angs = {n: L.angle14(VEC[t["concept"]], VEC[n]) for n in L.TRAINED}
            r["report"] = min(angs, key=angs.get)
        elif ho_style == "silent":
            r["report"] = t["concept"] if rng.random() < 0.1 else "NONE"
        else:
            r["report"] = "NONE"
        r["response"] = r["report"]
        rows.append(r)
    return rows


def synth_fc(seed=0, median=25.0, smoke=False):
    rng = np.random.default_rng(seed)
    rows = []
    for t in L.heldout_fc_rows(smoke):
        nearest = min(L.TRAINED, key=lambda n: L.angle14(VEC[t["concept"]], VEC[n]))
        rows.append({"tid": t["tid"], "block": "heldout_fc", "layer": t["layer"],
                     "alpha": t["alpha"], "injected": t["concept"],
                     "argmax": nearest, "report": nearest,
                     "exact": False, "none_top": bool(rng.random() < 0.1),
                     "err": L.angle14(VEC[t["concept"]], VEC[nearest]),
                     "hook_calls": 1, "cond": "x"})
    for t in L.sham_rows(smoke):
        rows.append({"tid": t["tid"], "block": "sham_fc", "layer": None,
                     "alpha": 0.0, "injected": None, "argmax": "UNCERTAINTY",
                     "none_top": True, "hook_calls": 0, "cond": "x"})
    return rows


def bundle(cond, style, catch_pre=1, catch_post=12, ho_style="own",
           anchor_exact=18, sham_claims=0, plateaued=True, seed=0, smoke=False):
    b = {
        "condition": cond, "mode": "smoke" if smoke else "full", "stamp": "test",
        "curriculum": {"scalar": 272, "naming": 216, "competence": 60, "lexicon": 52},
        "dirs_stability": {"resid": 3.9e-06, "layer": 14, "name": "CAPTURE",
                           "tol": 1e-3, "pass": True},
        "mu": {"14": 80.0, "20": 120.0},
        "pool_referents": {a: {} for a in L.ARMS},
        "train_labels": [],
        "train_log": {"n_examples": 600, "epochs_flown": 4 if plateaued else 6,
                      "epochs_cap": 6, "plateaued": plateaued,
                      "plateau_rel": 0.05, "min_epochs": 3,
                      "epoch_means": [2.0, 1.1, 0.8, 0.78] if plateaued
                      else [2.0, 1.5, 1.2, 1.0, 0.85, 0.72],
                      "strand_epoch_loss": [],
                      "micro_steps": 2400, "opt_steps": 300, "hook_calls": 576,
                      "trainable_params": 4358144, "max_example_tokens": 6200,
                      "peak_vram_gb": 11.7, "secs": 2500.0,
                      "loss_first_k": 2.4, "final_smoothed": 0.5,
                      "losses_every_10": []},
        "took": {"n": 24, "within1": 22, "frac": 0.917, "pass": True, "rows": []},
        "took_competence": {"n": 6, "within_tol": 6, "pass": True, "rows": []},
        "took_lexicon": {"n": 4, "exact": 4, "pass": True, "rows": []},
        "pre": {"catch_rows": synth_catch(catch_pre)},
        "post": {"battery_rows": synth_battery_rows(style, seed + 2),
                 "arm_errors": {}, "catch_rows": synth_catch(catch_post),
                 "paraphrase_rows": []},
        "injection_rows": synth_injection(ho_style, anchor_exact, sham_claims,
                                          seed + 3, smoke),
        "fc_rows": synth_fc(seed + 4, smoke=smoke),
        "ppl_pre": 11.0, "ppl_post": 11.15, "secs": 4600.0,
    }
    if cond == "real":
        b["pre"]["battery_rows"] = synth_battery_rows("weakpos", seed + 1)
        b["pre"]["arm_errors"] = {}
    return b


def run_verdict(results, cond_errors, smoke=False, make_shipped=True):
    td = tempfile.mkdtemp()
    out = Path(td) / "out"; out.mkdir()
    sem = Path(td) / "sem"
    inflight = sem / "e8n2" / "inflight_test"
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
        CONDITIONS=CONDITIONS, PAYLOAD=PAYLOAD, VEC=VEC,
        RESULTS=results, cond_errors=cond_errors,
        OUT=out, SEM=sem, INFLIGHT="e8n2/inflight_test",
        ship=lambda *a, **k: None, print=lambda *a, **k: None,
    ))
    exec(VERDICT_SRC, g)
    return json.load(open(out / "e8n2_verdict.json"))


passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


print("== scenario 1: full pass — all four primaries + S0 + gates ==")
res = {"real": bundle("real", "perfect", seed=10),
       "base": bundle("base", "perfect", seed=20)}
v = run_verdict(res, {})
pr = v["primaries"]
check("P1 planted: rho > .9, p < .01, n = 148",
      pr["P1_tracking_survives"]["rho"] > 0.9 and
      pr["P1_tracking_survives"]["p"] < 0.01 and
      pr["P1_tracking_survives"]["n"] == 148)
check("P2 1/12 -> 12/12: abs pass + tiny p",
      pr["P2_interface_competence"]["abs_pass"] and
      pr["P2_interface_competence"]["p_effective"] < 1e-6)
check("P3a own-names: p at permutation floor",
      pr["P3a_production_reads_geometry"]["p"] < 0.01)
check("P3b own-names: exact 32/32, p tiny",
      pr["P3b_own_names_used"]["k_exact"] == 32 and
      pr["P3b_own_names_used"]["p"] < 1e-10)
check("Holm rejects all four", all(rej for _, rej in pr["holm"].values()))
sec = v["secondaries"]
check("S0 claim passes on weak-positive pre",
      sec["S0_instillation_alone_claim"]["pass"] and
      sec["S0_instillation_alone_claim"]["rho"] > 0)
check("S2 base blocks present (competence + heldout + parity)",
      "S2_base_competence" in sec and "S2_base_heldout" in sec and
      set(sec["S2_convergence_parity"]) == {"real", "base"})
check("S3 positive with CI > 0 (pre weak, post perfect)",
      sec["S3_delta_post_minus_pre_real"]["delta"] > 0.2 and
      sec["S3_delta_post_minus_pre_real"]["ci95"][0] > 0)
check("S9 silence retention pass (0 claims of 24)",
      all(sec["S9_silence_retention"][c]["pass"] and
          sec["S9_silence_retention"][c]["n"] == 24 for c in CONDITIONS))
check("S10 comprehension present with locked comparator",
      sec["S10_comprehension_production"]["real"]["fc"]["n"] == 96 and
      sec["S10_comprehension_production"]["real"]["fc"]["locked_comprehension"] == 28.54)
check("S10 FC perm p computed and small (nearest-name comprehension)",
      v["conditions"]["real"]["fc"]["perm"]["p"] < 0.05)
check("S11 off-regime strata: 16 rows per condition",
      all(sec["S11_offregime_heldout"][c]["n"] == 16 for c in CONDITIONS))
check("gates: anchor + sham + dirs + tooks all green",
      all(v["gates"]["anchor"][c]["pass"] for c in CONDITIONS) and
      all(v["gates"]["sham"][c]["pass"] for c in CONDITIONS) and
      all(v["gates"]["dirs_stability"][c]["pass"] for c in CONDITIONS) and
      all(all(v["gates"]["tooks"][c].values()) for c in CONDITIONS))
check("base pre battery absent by design; S0 only from real",
      "pooled" not in v["conditions"]["base"]["pre"] and
      "S0_instillation_alone_claim" in sec)
check("heldout locked slice tracked separately (24 rows)",
      v["conditions"]["real"]["heldout_locked_slice"]["n"] == 24)

print("== scenario 2: P2 floor-stuck (catch 1 -> 1) ==")
res2 = {"real": bundle("real", "perfect", catch_post=1, seed=30),
        "base": bundle("base", "perfect", seed=40)}
v2 = run_verdict(res2, {})
check("P2 p_effective = 1.0, Holm does not reject P2",
      v2["primaries"]["P2_interface_competence"]["p_effective"] == 1.0 and
      not v2["primaries"]["holm"]["P2"][1])
check("P1/P3 unaffected", v2["primaries"]["holm"]["P1"][1] and
      v2["primaries"]["holm"]["P3a"][1])

print("== scenario 3: P3 silence fork ==")
res3 = {"real": bundle("real", "perfect", ho_style="silent", seed=50),
        "base": bundle("base", "perfect", ho_style="silent", seed=60)}
v3 = run_verdict(res3, {})
ho = v3["conditions"]["real"]["heldout"]
check("n-guard fails, fork recorded, both P3 p = 1.0",
      not ho["guard_pass"] and ho["p3a"]["p"] == 1.0 and
      "silence" in ho["p3a"]["fork"])
check("P1/P2 still Holm-testable at conservative thresholds",
      v3["primaries"]["holm"]["P1"][1] and v3["primaries"]["holm"]["P2"][1])

print("== scenario 4: nearest-name production ==")
res4 = {"real": bundle("real", "perfect", ho_style="nearest", seed=70),
        "base": bundle("base", "perfect", ho_style="nearest", seed=80)}
v4 = run_verdict(res4, {})
check("P3a rejects, P3b does not (0 exact)",
      v4["primaries"]["holm"]["P3a"][1] and
      not v4["primaries"]["holm"]["P3b"][1] and
      v4["primaries"]["P3b_own_names_used"]["k_exact"] == 0)

print("== scenario 5: gate regressions recorded ==")
res5 = {"real": bundle("real", "perfect", anchor_exact=14, sham_claims=6, seed=90),
        "base": bundle("base", "perfect", seed=95)}
v5 = run_verdict(res5, {})
check("anchor gate fails at 14/18",
      not v5["gates"]["anchor"]["real"]["pass"] and
      v5["gates"]["anchor"]["real"]["exact"] == 14)
check("S9 fails at 6/24 claims",
      not v5["gates"]["sham"]["real"]["pass"] and
      v5["gates"]["sham"]["real"]["claims"] == 6)

print("== scenario 6: partial flight (real only) ==")
v6 = run_verdict({"real": bundle("real", "perfect", seed=96)}, {})
check("no primaries on 1/2 (pre-reg hygiene)",
      "primaries" not in v6 and v6["complete_conditions"] == ["real"])

print("== scenario 7: smoke green path ==")
res7 = {"real": bundle("real", "perfect", seed=97, smoke=True),
        "base": bundle("base", "perfect", seed=98, smoke=True)}
for c in res7:
    res7[c]["post"]["battery_rows"] = synth_battery_rows("perfect", 99)
v7 = run_verdict(res7, {}, smoke=True)
check("smoke: no primaries, both conditions scored",
      "primaries" not in v7 and set(v7["conditions"]) == {"real", "base"})

print("== scenario 8: degenerate constant battery ==")
res8 = {"real": bundle("real", "degenerate", seed=99),
        "base": bundle("base", "degenerate", seed=100)}
v8 = run_verdict(res8, {})
check("P1 rho None, p = 1.0, no crash",
      v8["primaries"]["P1_tracking_survives"]["rho"] is None and
      v8["primaries"]["P1_tracking_survives"]["p"] == 1.0)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
