#!/usr/bin/env python3
"""Exercise the E8-R3-c VERDICT cell VERBATIM from the built notebook against
synthetic flights (E6 lesson: no cell first runs after a long flight).

Scenarios: (1) planted spelling -> gates pass, P1+P2 pass, fork SPELLING,
S-B passes; (2) planted lookup -> P1 pass, P2 fail, fork PAIR-LOOKUP;
(3) not-installed spot -> fork NOT INSTALLED; (4) anchor gate fail ->
primaries withheld; (5) partial flight (real only) -> resume-chain banner,
no verdict file; (6) smoke mode -> SMOKE GREEN banner; (7) silence fork ->
S4 n-guard fails, verdict still assembles.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r3c_verdict.py
"""
import io, json, os, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8r3c_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

nb = json.load(open(os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "SB_claim" in VERDICT_SRC, "last cell is not the verdict"
LOGIC_SRC = "".join(nb["cells"][2]["source"])

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}

ELIG, SCORED = L.pair_cands()
TP = L.trained_pair_list()
FLOORS = L.pair_lookup_floors(VEC)


def synth_scores(top, none_lp=-6.0):
    s = {n: -5.0 for n in SCORED}
    s["NONE"] = none_lp
    s[top] = -0.5
    return s


def fc_fixture(mode, rng):
    rows = []
    for t in L.trained_pair_fc_rows(False):
        key = L.pair_text(*t["pair"])
        top = key if mode != "notinstalled" or rng.random() < 0.1 else \
            L.pair_text(*TP[int(rng.integers(24))])
        rows.append(L.pair_fc_row(t, synth_scores(top), synth_scores(top), VEC))
    for t in L.heldout_pair_fc_rows(False):
        key = L.pair_text(*t["pair"])
        if mode == "spelling":
            top = key if rng.random() < 0.7 else L.pair_text(*TP[int(rng.integers(24))])
        elif mode == "lookup":
            top = FLOORS[key]["nearest"]
        else:
            top = L.pair_text(*L.all_pairs()[int(rng.integers(36))])
        rows.append(L.pair_fc_row(t, synth_scores(top), synth_scores(top), VEC))
    for t in L.reach_rows(False):
        t2 = {**t, "block": "reach_fc"}
        top = L.pair_text(*TP[int(rng.integers(24))])
        rows.append(L.pair_fc_row(t2, synth_scores(top), synth_scores(top), VEC))
    for i, t in enumerate([t for t in L.build_plan(False) if t["kind"] == "sham"]):
        t2 = {**t, "block": "sham_fc"}
        rows.append(L.pair_fc_row(t2, synth_scores(ELIG[0], none_lp=-0.01),
                                  synth_scores(ELIG[0], none_lp=-0.01), VEC))
    return rows


def gen_fixture(mode, rng):
    g = {}
    anchor = [t for t in L.build_plan(False)
              if t["kind"] == "inject" and t["concept"] in L.TRAINED
              and t["layer"] == L.TRAIN_LAYER and t["alpha"] in L.TRAIN_ALPHAS]
    n_bad = 6 if mode == "gatefail" else 0
    g["anchor"] = [{**t, "block": "anchor",
                    "report": ("UNCERTAINTY" if (t["concept"] != "UNCERTAINTY"
                                                 and i < n_bad) else t["concept"])}
                   for i, t in enumerate(anchor)]
    g["lockedsham_gen"] = [{**t, "block": "lockedsham_gen", "report": "NONE"}
                           for t in L.build_plan(False) if t["kind"] == "sham"]
    g["pairsham_gen"] = [{**t, "parsed": {"kind": "NONE", "names": []}}
                         for t in L.pairsham_gen_rows(False)]
    g["trainpair_gen"] = [{**t, "parsed": {"kind": "PAIR",
                                           "names": sorted(t["pair"])}}
                          for t in L.trained_pair_gen_rows(False)]
    held = []
    for t in L.heldout_pair_gen_rows(False):
        if mode == "silence":
            parsed = {"kind": "NONE", "names": []}
        else:
            parsed = ({"kind": "PAIR", "names": sorted(t["pair"])}
                      if rng.random() < 0.5 else {"kind": "NONE", "names": []})
        held.append({**t, "parsed": parsed})
    g["heldpair_gen"] = held
    g["reach_gen"] = [{**t, "parsed": {"kind": "NONE", "names": []}}
                      for t in L.reach_rows(False)]
    return g


def bundle(cond, mode, seed):
    rng = np.random.default_rng(seed)
    return {"condition": cond, "mode": "full", "stamp": "20260899_0000",
            "dirs_stability": {"resid": 1e-7, "layer": 14, "name": "X",
                               "tol": 1e-3, "pass": True},
            "ppl_pre": 10.0, "ppl_post": 10.05,
            "fc_rows": fc_fixture(mode, rng),
            "gen_rows": gen_fixture(mode, rng),
            "train_log": {"epochs_flown": 5, "plateaued": True}}


def run_verdict(results, smoke=False, resume=""):
    tmp = Path(tempfile.mkdtemp())
    g = {"__name__": "verdict_test"}
    exec(LOGIC_SRC, g)
    g.update({"RESULTS": results, "CONDITIONS": ["real", "scrambled"],
              "cond_errors": {}, "MODE": "smoke" if smoke else "full",
              "STAMP": "20260899_0000", "RESUME_STAMP": resume,
              "SMOKE": smoke, "OUT": tmp, "VEC": VEC, "np": np,
              "ship": lambda *a, **k: None})
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(VERDICT_SRC, g)
    out = buf.getvalue()
    vf = tmp / "e8r3c_verdict.json"
    verdict = json.load(open(vf)) if vf.exists() else None
    return out, verdict


print("== scenario 1: planted spelling ==")
res = {"real": bundle("real", "spelling", 1),
       "scrambled": bundle("scrambled", "lookup", 2)}
out, v = run_verdict(res)
check("gates pass", v["gates"]["real"]["all_pass"]
      and v["gates"]["scrambled"]["all_pass"])
check("P1 passes", v["P1"]["holm_pass"] and v["P1"]["p"] < 0.01)
check("P2 passes", v["P2"]["holm_pass"] and v["P2"]["exact"] > 50)
check("fork banner: SPELLING", "fork: SPELLING" in out)
check("spot installed", v["spot"]["real"]["installed"])
check("S-B claim passes (12/12 none_top)", v["SB_claim"]["pass"])
check("S2 below-floor high", v["S2_below_floor"]["fraction"] > 0.4)
check("S3 scrambled contrast present",
      v.get("S3_scrambled", {}).get("scr_exact") == 0)
check("S5 reach floors pinned in verdict",
      abs(v["S5_reach_floors"]["RETRIEVAL"]["pair_floor"] - 23.25) < 0.01)
check("verdict file written + shipped path printed",
      v is not None and "shipped: e8r3c/full_" in out)

print("== scenario 2: planted lookup ==")
res = {"real": bundle("real", "lookup", 3),
       "scrambled": bundle("scrambled", "null", 4)}
out, v = run_verdict(res)
check("P1 passes on lookup (registered property)", v["P1"]["holm_pass"])
check("P2 fails on lookup", not v["P2"]["holm_pass"] and v["P2"]["exact"] == 0)
check("fork banner: PAIR-LOOKUP", "fork: PAIR-LOOKUP" in out)
check("S2 below-floor zero", v["S2_below_floor"]["fraction"] == 0.0)

print("== scenario 3: not installed ==")
res = {"real": bundle("real", "notinstalled", 5),
       "scrambled": bundle("scrambled", "null", 6)}
out, v = run_verdict(res)
check("spot not installed", not v["spot"]["real"]["installed"])
check("fork banner: NOT INSTALLED", "fork: NOT INSTALLED" in out)

print("== scenario 4: anchor gate fail ==")
res = {"real": bundle("real", "gatefail", 7),
       "scrambled": bundle("scrambled", "null", 8)}
out, v = run_verdict(res)
check("G1 fails", not v["gates"]["real"]["g1_anchor"]["pass"])
check("primaries withheld", v["P1"].get("withheld") and v["P2"].get("withheld"))
check("withheld banner", "primaries WITHHELD" in out)

print("== scenario 5: partial flight (real only) ==")
res = {"real": bundle("real", "spelling", 9)}
out, v = run_verdict(res, resume="20260899_0000")
check("chain banner with stamp",
      "RUN COMPLETE" in out and "RESUME_STAMP = '20260899_0000'" in out
      and "scrambled" in out)
check("no verdict file on partial", v is None)

print("== scenario 6: smoke mode ==")
def smoke_bundle(cond, seed):
    rng = np.random.default_rng(seed)
    fc = []
    for t in L.trained_pair_fc_rows(True) + L.heldout_pair_fc_rows(True):
        key = L.pair_text(*t["pair"])
        fc.append(L.pair_fc_row(t, synth_scores(key), synth_scores(key), VEC))
    for t in L.reach_rows(True):
        fc.append(L.pair_fc_row({**t, "block": "reach_fc"},
                                synth_scores(ELIG[0]), synth_scores(ELIG[0]), VEC))
    shams = [t for t in L.build_plan(False) if t["kind"] == "sham"][:2]
    for t in shams:
        fc.append(L.pair_fc_row({**t, "block": "sham_fc"},
                                synth_scores(ELIG[0], none_lp=-0.01),
                                synth_scores(ELIG[0], none_lp=-0.01), VEC))
    anchor = [t for t in L.build_plan(False)
              if t["kind"] == "inject" and t["concept"] in L.TRAINED
              and t["layer"] == L.TRAIN_LAYER and t["alpha"] in L.TRAIN_ALPHAS]
    g = {"anchor": [{**t, "block": "anchor", "report": t["concept"]}
                    for t in [anchor[0], anchor[-1]]],
         "lockedsham_gen": [{**t, "block": "lockedsham_gen", "report": "NONE"}
                            for t in shams],
         "pairsham_gen": [{**t, "parsed": {"kind": "NONE", "names": []}}
                          for t in L.pairsham_gen_rows(True)],
         "trainpair_gen": [{**t, "parsed": {"kind": "PAIR",
                                            "names": sorted(t["pair"])}}
                           for t in L.trained_pair_gen_rows(True)],
         "heldpair_gen": [{**t, "parsed": {"kind": "PAIR",
                                           "names": sorted(t["pair"])}}
                          for t in L.heldout_pair_gen_rows(True)],
         "reach_gen": [{**t, "parsed": {"kind": "NONE", "names": []}}
                       for t in L.reach_rows(True)]}
    return {"condition": cond, "mode": "smoke", "stamp": "20260899_0000",
            "dirs_stability": {"resid": 1e-7, "pass": True, "tol": 1e-3},
            "ppl_pre": 10.0, "ppl_post": 10.02, "fc_rows": fc, "gen_rows": g,
            "train_log": {"epochs_flown": 1, "plateaued": False}}

out, v = run_verdict({"real": smoke_bundle("real", 10),
                      "scrambled": smoke_bundle("scrambled", 11)}, smoke=True)
check("SMOKE GREEN banner + restart instruction",
      "SMOKE GREEN" in out and "set SMOKE = False" in out)
check("smoke verdict file written", v is not None)

print("== scenario 7: silence fork ==")
res = {"real": bundle("real", "silence", 12),
       "scrambled": bundle("scrambled", "silence", 13)}
out, v = run_verdict(res)
check("S4 n-guard fails on all-NONE generation",
      not v["S4_gen"]["real"]["heldpair"]["nguard_pass"])
check("verdict still assembles (FC carries)", "P1" in v and "P2" in v)

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{'='*60}\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
