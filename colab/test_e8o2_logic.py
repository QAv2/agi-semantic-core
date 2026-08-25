#!/usr/bin/env python3
"""E8-O2 pure-logic suite: pins vs the design check, builders at BOTH mode
constants, firewalls, stimulus probes, and statistical teeth for the new
machinery (P-M2 function-leg primary; S-MEJI reversal; the FV2 shape).

Run:  python3 colab/test_e8o2_logic.py
"""
import hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8o2_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
PAYLOAD = json.load(open(os.path.join(HERE, "e8f_payload.json")))
DC = json.load(open(os.path.join(HERE, "results_e8o2", "design_check.json")))
NB = json.load(open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"]]
DESIGN = open(os.path.join(HERE, "e8o2_design_check.py")).read()

# ── pins ────────────────────────────────────────────────────────────────────
check("draws == design check",
      [list(p) for p in L.TRAINPAIR48] == DC["s2"]["trainpair48"]
      and [list(p) for p in L.POFRESH12] == DC["s2"]["pofresh12"]
      and [list(p) for p in L.SPOTMIX12] == DC["s2"]["spotmix12"])
check("shas match", L.SHA_TRAINPAIR == DC["s2"]["sha_trainpair"]
      and L.SHA_POFRESH == DC["s2"]["sha_pofresh"])
members = {c for p in L.TRAINPAIR48 for c in p}
check("TRAINPAIR48 = perfect matching in train-256",
      len(members) == 96 and members <= set(L.TRAIN256))
check("POFRESH12 disjoint from flown PAIR24, inside eval-64",
      not ({tuple(p) for p in L.POFRESH12} & {tuple(p) for p in L.PAIR24})
      and {c for p in L.POFRESH12 for c in p} <= set(L.EVAL64))
check("REPLAY24 in train-256, disjoint from matching members",
      set(L.REPLAY24) <= set(L.TRAIN256) and not set(L.REPLAY24) & members)
check("SPOTMIX12 subset of TRAINPAIR48",
      all(list(p) in [list(q) for q in L.TRAINPAIR48] for p in L.SPOTMIX12))
check("notebook logic cell == e8o2_logic.py byte-verbatim",
      CELLS[2] == open(os.path.join(HERE, "e8o2_logic.py")).read())
check("logic base = e8o_logic verbatim",
      CELLS[2].startswith(open(os.path.join(HERE, "e8o_logic.py")).read()))
smeji_slice = DESIGN[DESIGN.find("def meji_completion_code"):
                     DESIGN.find("def main")].rstrip()
smeji_flat = smeji_slice.replace("L.code_levels", "code_levels") \
                        .replace("L.AXIS_NAMES_F", "AXIS_NAMES_F")
check("smeji block byte-equal to design check (namespace-flattened)",
      smeji_flat in CELLS[2])
for marker in ("def train_readout_mixed", "is_trainable=True",
               "def retention_ppl", "def pm2_stats", "def smeji_test",
               "def g2_probe", "def run_trial_feat", "class Injector",
               "BASELINE_O"):
    check(f"notebook carries {marker}", any(marker in c for c in CELLS))

# ── builders at both mode constants + firewalls ─────────────────────────────
for smoke in (False, True):
    tr = L.build_e8o2_train(VEC, smoke=smoke)
    ev = L.build_e8o2_eval(smoke=smoke)
    cnt = {}
    for e in tr:
        cnt[e["strand"]] = cnt.get(e["strand"], 0) + 1
    cnt_e = {}
    for r in ev:
        cnt_e[r["block"]] = cnt_e.get(r["block"], 0) + 1
    if not smoke:
        check("full curriculum counts", cnt == L.EXPECT_TRAIN_O2)
        check("full eval counts", cnt_e == L.EXPECT_EVAL_O2)
    else:
        check("smoke covers all strands/blocks",
              set(cnt) == set(L.STRANDS_O2)
              and set(cnt_e) == set(L.EXPECT_EVAL_O2))
    check(f"firewall clean (smoke={smoke})",
          not L.validate_no_eval_in_train_o2(tr))
    tgt_ok = all((e["target"] == "NONE") == (e["kind"] == "sham")
                 and (e["kind"] == "sham"
                      or L.parse_feat(e["target"])["n_fields"] == 14)
                 for e in tr)
    check(f"targets valid (smoke={smoke})", tgt_ok)
bad = L.validate_no_eval_in_train_o2(
    [{"eid": 1, "skey": f"full:{sorted(L.EVAL64)[0]}"}])
check("firewall catches planted eval leak", bad == [1])
bad2 = L.validate_no_eval_in_train_o2([{"eid": 2, "skey": "comp:0:0"}])
check("firewall catches flown-eval-pair leak", bad2 == [2])

# mixed targets are exact composed codes (A7+B7)
tr_full = L.build_e8o2_train(VEC, smoke=False)
mx = [e for e in tr_full if e["strand"] == "mixed"][0]
i, o = (int(x) for x in mx["skey"].split(":")[1:])
a, b = L.TRAINPAIR48[i] if o == 0 else L.TRAINPAIR48[i][::-1]
check("mixed target = exact composed code",
      mx["target"] == L.code_text(L.composed_code(VEC[a], VEC[b])))

# ── stimuli ─────────────────────────────────────────────────────────────────
W5 = np.array(PAYLOAD["W"], float)
STIM = L.mint_stimuli_o2(W5, VEC)
need = ({e["skey"] for e in tr_full if e["skey"]}
        | {r["skey"] for r in L.build_e8o2_eval(False) if r.get("skey")})
check("all skeys minted, unit-norm",
      need <= set(STIM)
      and all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in STIM.values()))
probes = json.loads(CELLS[3][CELLS[3].find("PAYLOAD_PROBES_O = json.loads(r'''")
                             + len("PAYLOAD_PROBES_O = json.loads(r'''"):
                             CELLS[3].find("''')")])
check("build probes reproduce",
      all(float(np.max(np.abs(STIM[k] - np.asarray(v, float)))) < 1e-4
          for k, v in probes.items()))

# ── statistical teeth ───────────────────────────────────────────────────────
PC = L.pair_codes(VEC)
po_rows = [r for r in L.build_e8o2_eval(False) if r["block"] == "po"]

def with_parsed(rows, fn):
    out = []
    for r in rows:
        lv = fn(r)
        out.append({**r, "parsed": {"kind": "CODE" if lv else "NONE",
                                    "levels": ({L.AXIS_NAMES_F[j]: lv[j]
                                                for j in range(14)}
                                               if lv else {}),
                                    "n_fields": 14 if lv else 0}})
    return out

faithful = with_parsed(po_rows, lambda r: PC[(r["pair_idx"], r["order"])])
m2 = L.pm2_stats(faithful, PC, 2000, seed=9)
check("tooth: P-M2 passes in leg-faithful world",
      m2["pooled_acc"] == 1.0 and m2["p"] <= L.P_CRIT_O)
sm = L.smeji_test(faithful, PC, VEC, [tuple(p) for p in L.PAIR24], 10000,
                  seed=9)
check("tooth: S-MEJI positive + significant in faithful world",
      sm["d_mean"] > 0.3 and sm["p"] <= L.P_CRIT_O)

def shortcut_parse(r):
    a, b = L.PAIR24[r["pair_idx"]]
    if r["order"] == 1:
        a, b = b, a
    return L.code_levels(VEC[a])

shortcut = with_parsed(po_rows, shortcut_parse)
sm_s = L.smeji_test(shortcut, PC, VEC, [tuple(p) for p in L.PAIR24], 2000,
                    seed=9)
m2_s = L.pm2_stats(shortcut, PC, 2000, seed=9)
check("tooth: shortcut world — S-MEJI negative, P-M2 weak",
      sm_s["d_mean"] < -0.3 and m2_s["pooled_acc"] < 0.55)
blind = with_parsed(po_rows, lambda r: PC[(r["pair_idx"], 0)])
m1_b = L.po2_stats(blind, PC, 2000, seed=9)
m2_b = L.pm2_stats(blind, PC, 2000, seed=9)
check("tooth: the FV2 shape (order-blind: P-M1 d~0, P-M2 still clears)",
      abs(m1_b["d_mean"]) < 0.02 and m2_b["p"] <= 0.05
      and m2_b["pooled_acc"] > m2_b["null_mean"])
shift = with_parsed(po_rows,
                    lambda r: PC[((r["pair_idx"] + 7) % 24, r["order"])])
m2_d = L.pm2_stats(shift, PC, 2000, seed=9)
check("tooth: deranged world fails P-M2", m2_d["p"] > 0.05)
check("ess_acc helper", L.ess_acc(faithful, PC) == 1.0)

# ── rebuild byte-identical ──────────────────────────────────────────────────
nb_bytes = open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb"), "rb").read()
lg_bytes = open(os.path.join(HERE, "e8o2_logic.py"), "rb").read()
subprocess.run([sys.executable, os.path.join(HERE, "build_e8o2_notebook.py")],
               check=True, capture_output=True)
check("rebuild byte-identical",
      open(os.path.join(HERE, "E8O2_MIXED_UI.ipynb"), "rb").read() == nb_bytes
      and open(os.path.join(HERE, "e8o2_logic.py"), "rb").read() == lg_bytes)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
