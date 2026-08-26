#!/usr/bin/env python3
"""E7b-Q logic suite — plan, metric, stats teeth, gates, single-source law.

Run:  python3 colab/test_e7bq_logic.py
"""
import inspect
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e7bq_design_check as DC
import e7bq_logic as L

PAYLOAD = json.load(open(os.path.join(HERE, "e7bq_payload.json")))
N_PASS = 0


def check(label, ok):
    global N_PASS
    assert ok, f"FAIL: {label}"
    N_PASS += 1
    print(f"  ok {N_PASS:2d}  {label}")


print("== single-source law: logic == committed design check (verbatim) ==")
for fn in ("visible_mass", "_ranks", "spearman", "fisher_z", "pw1_delta",
           "pw1_test", "pw2_test", "attr_test", "group_sep_test"):
    src_l = inspect.getsource(getattr(L, fn))
    src_d = inspect.getsource(getattr(DC, fn))
    check(f"{fn} source-identical to design check", src_l == src_d)

print("== thinning metric ==")
check("dossier-§4 ledger reproduces exactly", L.ledger_selftest())

print("== payload pins ==")
check("script sha matches payload pin",
      L.script_sha(PAYLOAD["script"]) == PAYLOAD["script_sha"])
check("15 turns, tags as registered",
      [s["tag"] for s in PAYLOAD["script"]]
      == ["B1", "B2", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8",
          "E1", "E2", "E3", "M", "G"])
check("windows match logic constants",
      PAYLOAD["windows"] == {"base": L.BASE_WIN, "neg": L.NEG_IDX,
                             "term": L.TERM_WIN, "emerg": L.EMERG_IDX,
                             "couple": L.COUPLE_IDX})
perm = PAYLOAD["reorder_perm"]
check("reorder perm valid (perm of 8, != id, deepest not last)",
      sorted(perm) == list(range(8)) and perm != list(range(8))
      and perm[-1] != 7)
check("sham slots cover N1..N8",
      sorted(PAYLOAD["sham_slots"]) == [f"N{i}" for i in range(1, 9)])
check("encoders present for base14/inst14/inst20 at 1537x14",
      all(len(PAYLOAD["encoders"][k]) == 1537
          and len(PAYLOAD["encoders"][k][0]) == 14
          for k in ("base14", "inst14", "inst20")))
check("cloud means unit-norm",
      all(abs(np.linalg.norm(PAYLOAD["cloud_mean"][k]) - 1) < 1e-3
          for k in ("base14", "inst14", "inst20")))
check("probes: 8 with VOID forced",
      len(PAYLOAD["probes"]) == 8 and PAYLOAD["probes"][0] == "VOID")

print("== arms ==")
walked = L.arm_turns(PAYLOAD, "walked")
sham = L.arm_turns(PAYLOAD, "sham")
reorder = L.arm_turns(PAYLOAD, "reorder")
unwalked = L.arm_turns(PAYLOAD, "unwalked")
check("walked == script", [t for t, _ in walked]
      == [s["tag"] for s in PAYLOAD["script"]]
      and all(x == s["text"] for (_, x), s in zip(walked, PAYLOAD["script"])))
check("sham differs ONLY at N slots",
      all((wt == st) == (not tag.startswith("N"))
          for (tag, wt), (_, st) in zip(walked, sham)))
check("sham shares B/E/M/G verbatim",
      all(wt == st for (tag, wt), (_, st) in zip(walked, sham)
          if not tag.startswith("N")))
check("reorder = B1 B2 + permuted negs + E1",
      [t for t, _ in reorder]
      == ["B1", "B2"] + [f"N{i + 1}" for i in perm] + ["E1"]
      and reorder[-1][1] == walked[10][1])
check("unwalked = single E1", unwalked == [("E1", walked[10][1])])

print("== plan ==")
rows_f, sha_f = L.build_plan(PAYLOAD, smoke=False)
rows_f2, sha_f2 = L.build_plan(PAYLOAD, smoke=False)
rows_s, sha_s = L.build_plan(PAYLOAD, smoke=True)
check("full plan deterministic (rebuild identical)",
      sha_f == sha_f2 and rows_f == rows_f2)
check("full/smoke counts", len(rows_f) == 2 * 16 * (15 + 15 + 11 + 1)
      and len(rows_s) == 2 * 2 * (15 + 15 + 11 + 1))
check("per-arm counts match expected_counts (both modes)",
      all(sum(1 for r in rows if r["cond"] == c and r["arm"] == a) == n
          for rows, smoke in ((rows_f, False), (rows_s, True))
          for c in L.CONDS
          for a, n in L.expected_counts(smoke).items()))
check("seeds unique", len({r["seed"] for r in rows_f}) == len(rows_f))
check("turns sequential within (cond, arm, rep)",
      all(rows_f[i]["turn"] == rows_f[i - 1]["turn"] + 1
          for i in range(1, len(rows_f))
          if (rows_f[i]["cond"], rows_f[i]["arm"], rows_f[i]["rep"])
          == (rows_f[i - 1]["cond"], rows_f[i - 1]["arm"],
              rows_f[i - 1]["rep"])))
check("keep_state exactly on rider turns",
      all(r["keep_state"] == (r["turn"] in L.RIDER_TURNS[r["arm"]])
          for r in rows_f))
# cross-process stability (the v3 repr(set) lane note made a tooth)
out = subprocess.run(
    [sys.executable, "-c",
     "import sys, json; sys.path.insert(0, %r); import e7bq_logic as L; "
     "p = json.load(open(%r)); print(L.build_plan(p, False)[1])"
     % (HERE, os.path.join(HERE, "e7bq_payload.json"))],
    capture_output=True, text=True)
check("plan sha stable across processes", out.stdout.strip() == sha_f)

print("== stats teeth (fast) ==")
rng = np.random.default_rng(7)


def world(R, delta, kappa, seed, sham=False):
    return DC._mk_world(R, L.T_WALK, delta, kappa, 0.35, seed, sham=sham)


gw, mw = world(12, 1.2, 1.0, 21)
gs, _ = world(12, 0.0, 0.0, 22, sham=True)
check("P-W1 fires on planted contraction",
      L.pw1_test(gw, gs, 800, 23)[1] < 0.01)
check("P-W2 fires on planted coupling",
      L.pw2_test(gw, mw, L.COUPLE_IDX, 800, 24)[2] < 0.01)
hits = sum(L.pw2_test(*(world(12, 1.2, 0.0, 300 + k)[0],
                        world(12, 1.2, 0.0, 400 + k)[1]),
                      L.COUPLE_IDX, 300, 500 + k)[2] < 0.05
           for k in range(12))
check("P-W2 trend-only null holds rate (<=3/12)", hits <= 3)
S = rng.normal(size=(10, L.T_WALK, 24))
Sc = S.copy()
Sc[:, [8, 9], :] *= 0.2
check("R-ATTR fires on planted attractor",
      L.attr_test(Sc, 800, 25)[1] < 0.01)
A = rng.normal(size=(10, 16))
check("group-sep fires on separated groups",
      L.group_sep_test(A, A + 2.0, 800, 26)[1] < 0.01)
check("goback fires on planted gap",
      L.goback_test(np.full(12, 3.0) + rng.normal(0, .2, 12),
                    np.full(12, 1.0) + rng.normal(0, .2, 12), 800, 27)[1]
      < 0.01)
check("holm on known case",
      L.holm([0.01, 0.04]) == [0.02, 0.04]
      and L.holm([0.04, 0.01]) == [0.04, 0.02])

print("== gates + verdict plumbing (synthetic flights) ==")
b1 = L.synth_flight(PAYLOAD, "fb1", smoke=True, seed=31)
check("gate_dirs pass on clean, fail on dirty",
      L.gate_dirs({c: b1[c]["gdirs"] for c in L.CONDS})["pass"]
      and not L.gate_dirs({"real": {"14": 5e-3}, "base": {"14": 1e-5}})
      ["pass"])
check("gate_plan pass on complete", L.gate_plan(b1, smoke=True)["pass"])
b_short = {c: dict(b1[c]) for c in b1}
b_short["real"] = dict(b_short["real"])
b_short["real"]["rows"] = b1["real"]["rows"][:-1]
check("gate_plan fails on dropped row",
      not L.gate_plan(b_short, smoke=True)["pass"])
gc1 = L.gate_capture(b1, PAYLOAD, smoke=True)
check(f"gate_capture pass on clean (spot {gc1['spot_worst']:.1e}, "
      f"n {gc1['spot_n']})", gc1["pass"])
b_bad = json.loads(json.dumps(b1))
b_bad["real"]["rows"][3]["chat_pre14"] = None
check("gate_capture fails on missing chat",
      not L.gate_capture(b_bad, PAYLOAD, smoke=True)["pass"])

print("== json round-trip law (everything a flight ships) ==")
tmp = os.path.join(HERE, "results_e7bq", "_tmp_roundtrip.json")
os.makedirs(os.path.dirname(tmp), exist_ok=True)
L.jdump(b1["real"], tmp)
rt = json.load(open(tmp))
check("condition bundle round-trips",
      rt["rows"] == b1["real"]["rows"] and rt["gdirs"] == b1["real"]["gdirs"])
V = L.verdict(b1, PAYLOAD, "smoke")
L.jdump(V, tmp)
check("verdict round-trips", json.load(open(tmp))["fork"] == V["fork"])
os.remove(tmp)

print(f"\nALL {N_PASS} CHECKS PASS")
