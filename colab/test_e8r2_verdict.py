#!/usr/bin/env python3
"""Exercise the E8-R2 VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) real planted + scrambled null -> gates pass, P1+P2 pass, Holm
rejects, secondaries assemble; (2) real null -> P1 not rejected; (3) both
planted (geometry-nonspecific) -> P1 passes, P2 CI straddles 0 -> fails;
(4) anchor gate failure -> primaries withheld; (5) partial flight (real only)
-> no primaries, resume banner path; (6) smoke branch -> checks dict, green.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r2_verdict.py
"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8r2_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8R2_FORCEDCHOICE_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P_E8R2_1_heldout_comprehension" in VERDICT_SRC, "last cell is not the verdict"

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}


def synth_scores(top, none_lp=-2.5):
    s = {n: -3.0 for n in L.SCORED_SET}
    s["NONE"] = none_lp
    s[top] = -0.4
    return s


def synth_condition(cond, style, smoke=False, g1_exact=None, seed=0):
    """A full bundle the flight loop would produce."""
    rng = np.random.default_rng(seed)
    anchor_plan = L.anchor_rows(smoke)
    ho_plan = L.heldout_fc_rows(smoke)
    sham_plan = L.sham_rows(smoke)
    ti_plan = L.titration_rows(smoke)

    def top_for(t):
        if style == "planted":
            return t["concept"]
        if style == "null":
            return "CONFABULATION"
        if style == "noisy":
            return L.FC_CANDIDATES[int(rng.integers(0, 13))]
        raise ValueError(style)

    anchor = []
    for i, t in enumerate(anchor_plan):
        top = t["concept"]
        if g1_exact is not None and i >= g1_exact:
            top = "CONFABULATION" if t["concept"] != "CONFABULATION" else "CAPTURE"
        r = L.fc_row(t, synth_scores(top), synth_scores(top), VEC)
        r.update({"cond": cond, "hook_calls": 1})
        anchor.append(r)
    heldout = []
    for t in ho_plan:
        top = top_for(t)
        r = L.fc_row(t, synth_scores(top), synth_scores(top), VEC)
        r.update({"cond": cond, "hook_calls": 1})
        heldout.append(r)
    shams = []
    for t in sham_plan:
        r = L.fc_row(t, synth_scores("CONFIDENCE", none_lp=-0.1),
                     synth_scores("CONFIDENCE", none_lp=-0.1), VEC)
        r.update({"cond": cond, "hook_calls": 0})
        shams.append(r)
    titr = [{**t, "cond": cond, "response": t["concept"],
             "report": (t["concept"] if t["alpha"] >= 0.4 else "NONE"),
             "hook_calls": 1} for t in ti_plan]
    return {"condition": cond, "mode": "smoke" if smoke else "full",
            "stamp": "TEST", "src_flight": "inflight_20260822_2329",
            "g2": {"names_ok": True, "norm_resid_max": 1e-6, "mu14": 81.9,
                   "pass": True},
            "g1b": {"ppl_recon": 10.0, "ppl_post_shipped": 10.0,
                    "delta_pct": 0.0, "tol_pct": 0.5, "pass": True},
            "g1": L.gate_g1(anchor),
            "anchor": anchor, "heldout": heldout, "shams": shams,
            "titration": titr, "secs": 1.0}


def run_verdict(results, smoke=False, cond_errors=None, ship_bundles=True):
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "out"; out.mkdir()
    sem = tmp / "sem"
    inflight = "e8r2/inflight_TEST"
    (sem / inflight).mkdir(parents=True)
    if ship_bundles:
        for c in results:
            (sem / inflight / f"condition_{c}.json").write_text("{}")
    ns = dict(vars(L))
    ns.update({
        "np": np, "json": json, "VEC": VEC,
        "RESULTS": results, "cond_errors": cond_errors or {},
        "SMOKE": smoke, "MODE": "smoke" if smoke else "full", "STAMP": "TEST",
        "MODEL_ID": "Qwen/Qwen2.5-1.5B-Instruct",
        "CONDITIONS": ("real", "scrambled"),
        "OUT": out, "SEM": sem, "INFLIGHT": inflight, "RESUME_STAMP": "",
        "ship": lambda *a, **k: None,
        "N_PERM": 200, "N_BOOT": 200,
        "ANCHOR": L.anchor_rows(smoke), "HELDOUT": L.heldout_fc_rows(smoke),
        "SHAMS": L.sham_rows(smoke), "TITR": L.titration_rows(smoke),
    })
    exec(VERDICT_SRC, ns)
    return ns


print("== scenario 1: real planted + scrambled null -> full pass ==")
ns = run_verdict({"real": synth_condition("real", "planted"),
                  "scrambled": synth_condition("scrambled", "null", seed=1)})
s = ns["summary"]
p = s["primaries"]
check("gates all pass", s["gates_all_pass"])
check("P1 small p on planted comprehension",
      p["P_E8R2_1_heldout_comprehension"]["p"] < 0.05
      and p["P_E8R2_1_heldout_comprehension"]["obs_median_err"] == 0.0
      and p["P_E8R2_1_heldout_comprehension"]["n_rows"] == 96)
check("P2 CI clears zero (scrambled worse)",
      p["P_E8R2_2_geometry_specificity"]["pass_ci"]
      and p["P_E8R2_2_geometry_specificity"]["delta_scrambled_minus_real"]["delta"] > 0)
check("Holm rejects both", all(v[1] for v in p["holm"].values()))
sec = s["secondaries"]
check("S1 exact-hit binomial tiny on planted",
      sec["S1_exact_hits_real"]["hits"] == 96
      and sec["S1_exact_hits_real"]["p_binom"] < 1e-6)
check("S2 rank primary small p", sec["S2_rank_real"]["p"] < 0.05
      and sec["S2_rank_real"]["obs_median_rank"] == 1.0)
check("S3/S4 rates present and sane",
      sec["S3_none_top"]["real"] == 0.0
      and sec["S4_heldout_name_argmax"]["real"]["rate"] == 1.0
      and sec["S4_heldout_name_argmax"]["real"]["correct_subset"] == 96)
check("S5 sham prior carries modal + none_top",
      sec["S5_sham_prior"]["real"]["modal"] == "CONFIDENCE"
      and sec["S5_sham_prior"]["real"]["none_top_rate"] == 1.0)
check("S6 titration curves per condition",
      set(sec["S6_titration"]["real"]) == {"0.3", "0.375", "0.45"}
      and sec["S6_titration"]["real"]["0.45"]["claim_rate"] == 1.0
      and sec["S6_titration"]["real"]["0.3"]["claim_rate"] == 0.0)
check("S7 robustness + S8 oracle floor assemble",
      sec["S7_robustness_sum_logprob"]["agreement_rate"] == 1.0
      and sec["S8_oracle_floor"]["rate_below_floor"] == 1.0)
check("exploratory per-concept + per-alpha blocks",
      set(sec["X_per_concept_real"]) == set(L.HELD_OUT)
      and set(sec["X_per_alpha_real"]) == {"0.5", "1.0"})
check("verdict json written", (ns["OUT"] / "e8r2_verdict.json").exists())

print("== scenario 2: real null -> P1 not rejected ==")
ns2 = run_verdict({"real": synth_condition("real", "noisy", seed=3),
                   "scrambled": synth_condition("scrambled", "noisy", seed=4)})
p2 = ns2["summary"]["primaries"]
check("P1 p not small on random argmax",
      p2["P_E8R2_1_heldout_comprehension"]["p"] > 0.05)
check("Holm does not reject P1", not p2["holm"]["P1"][1])

print("== scenario 3: both planted -> geometry-nonspecific, P2 fails ==")
ns3 = run_verdict({"real": synth_condition("real", "planted"),
                   "scrambled": synth_condition("scrambled", "planted")})
p3 = ns3["summary"]["primaries"]
check("P1 passes", p3["P_E8R2_1_heldout_comprehension"]["p"] < 0.05)
check("P2 CI does not clear (delta 0)",
      not p3["P_E8R2_2_geometry_specificity"]["pass_ci"])
check("Holm: P1 rejected, P2 not",
      p3["holm"]["P1"][1] and not p3["holm"]["P2"][1])

print("== scenario 4: anchor gate failure -> primaries withheld ==")
ns4 = run_verdict({"real": synth_condition("real", "planted", g1_exact=10),
                   "scrambled": synth_condition("scrambled", "null", seed=5)})
s4 = ns4["summary"]
check("gates_all_pass False on 10/18 anchor", not s4["gates_all_pass"])
check("NO primaries computed", "primaries" not in s4)
check("gate values still reported for diagnosis",
      s4["conditions"]["real"]["gates"]["g1"]["exact"] == 10)

print("== scenario 5: partial flight (real only) -> no primaries ==")
ns5 = run_verdict({"real": synth_condition("real", "planted")})
s5 = ns5["summary"]
check("complete list is real-only", s5["complete_conditions"] == ["real"])
check("no primaries at 1/2 (pre-reg hygiene)", "primaries" not in s5)

print("== scenario 6: smoke branch -> checks green ==")
ns6 = run_verdict({"real": synth_condition("real", "planted", smoke=True),
                   "scrambled": synth_condition("scrambled", "null",
                                                smoke=True, seed=6)},
                  smoke=True)
s6 = ns6["summary"]
check("smoke: no primaries", "primaries" not in s6)
check("smoke: conditions summarized with gates + titration",
      set(s6["conditions"]) == {"real", "scrambled"}
      and s6["conditions"]["real"]["gates"]["g2"]["pass"])

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
