#!/usr/bin/env python3
"""E8-J v2 Part-2 test suites (lane law: verdict-cell exec vs synthetic
flights, planted teeth, both mode constants, rebuild byte-identical).

Run:  python3 colab/test_e8j2.py
"""
import contextlib, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j2_logic as L2

PASS = 0
def ok(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok {PASS:2d}  {name}")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ══ Group A — build integrity ════════════════════════════════════════════════
print("Group A — build integrity")
nb_path = os.path.join(HERE, "E8J2_RUNG_UI.ipynb")
logic_path = os.path.join(HERE, "e8j2_logic.py")
h_nb, h_lg = sha_file(nb_path), sha_file(logic_path)
r = subprocess.run([sys.executable, os.path.join(HERE, "build_e8j2_notebook.py")],
                   capture_output=True, text=True)
ok("rebuild exits 0", r.returncode == 0)
ok("rebuild byte-identical (notebook)", sha_file(nb_path) == h_nb)
ok("rebuild byte-identical (logic)", sha_file(logic_path) == h_lg)

nb = json.load(open(nb_path))
cells = ["".join(c["source"]) for c in nb["cells"]]
ok("7 cells", len(cells) == 7)
donor_logic = open(os.path.join(HERE, "e8j_logic.py")).read()
ok("logic cell == e8j_logic.py byte-verbatim", cells[2] == donor_logic)
ok("e8j2_logic.py == donor + additions cell",
   open(logic_path).read() == donor_logic + "\n\n" + cells[3])
e8j_src = open(os.path.join(HERE, "build_e8j_notebook.py")).read()
def slice_lit(src, name):
    key = f"{name} = r'''"
    i0 = src.index(key) + len(key)
    return src[i0:src.index("'''", i0)]
ok("model cell == E8-J CELL_MODEL byte-verbatim",
   cells[4] == slice_lit(e8j_src, "CELL_MODEL"))
setup_donor = slice_lit(e8j_src, "CELL_SETUP")
diffs = sum(1 for a, b in zip(cells[1].splitlines(), setup_donor.splitlines())
            if a != b)
ok("setup cell: exactly 3 changed lines vs donor", diffs == 3)
ok("setup keeps RAM guards + drive.mount + adapter asserts",
   all(f in cells[1] for f in ("_mem_avail_gb", "drive.mount",
                               "ADAPTER_REAL", "READOUT_REAL", "e8j2/inflight_")))
ok("zero rclone in all cells", all("rclone" not in c.lower() for c in cells))
PINS = json.load(open(os.path.join(HERE, "results_e8j2", "rung_pins.json")))
ok("embedded dhat == locked artifact exactly", L2.DHAT_REAL == PINS["dhat_real"])
ok("embedded derangement == locked artifact",
   L2.E8J2_DERANGEMENT == PINS["derangement"])
ok("embedded geom hits == locked artifact",
   L2.E8J2_GEOM_HITS == PINS["geometric_hits"])

# ══ Group B — logic + planted teeth (both mode constants) ════════════════════
print("Group B — logic")
info = L2.verify_e8j2_pins()
ok("pins verify green", info["sha"] == L2.E8J2_PINS_SHA and info["n"] == 13)
ok("geom hits = the locked 4",
   info["geom_hits"] == ["CONSTRUCTION", "FAMILIARITY", "RESOLUTION", "RETRIEVAL"])

tam = {c: list(v) for c, v in L2.DHAT_REAL.items()}
tam["CONFIDENCE"][0] += 0.01
try:
    L2.verify_e8j2_pins(dhat=tam); bad = False
except AssertionError:
    bad = True
ok("tamper tooth: flipped coord -> sha assert fires", bad)
fixed = dict(L2.E8J2_DERANGEMENT)
k0 = next(iter(fixed)); fixed[k0] = k0
try:
    L2.verify_e8j2_pins(der=fixed); bad = False
except AssertionError:
    bad = True
ok("tamper tooth: fixed-point derangement fires", bad)
try:
    L2.verify_e8j2_pins(hits={"X": True}); bad = False
except AssertionError:
    bad = True
ok("tamper tooth: wrong hit-map keys fires", bad)

full = L2.build_b_pairs(False)
smoke = L2.build_b_pairs(True)
ok("full plan: 208 rows / 104 pairs",
   len(full) == 208 and len({t["pair_id"] for t in full}) == 104)
by_pair = {}
for t in full:
    by_pair.setdefault(t["pair_id"], []).append(t)
ok("matched pairs share concept/alpha/order; arms real+perm",
   all(len(v) == 2 and v[0]["concept"] == v[1]["concept"]
       and v[0]["alpha"] == v[1]["alpha"] and v[0]["order"] == v[1]["order"]
       and {v[0]["arm"], v[1]["arm"]} == {"real", "perm"}
       for v in by_pair.values()))
ok("smoke plan: 8 rows / 4 pairs (2 targets, both arms)",
   len(smoke) == 8 and len({t["pair_id"] for t in smoke}) == 4
   and {t["concept"] for t in smoke} == {L2.TRAINED[0], L2.HELD_OUT[0]})
ok("anchor rows 18 full / 2 smoke",
   len(L2.anchor_rows(False)) == 18 and len(L2.anchor_rows(True)) == 2)
ok("sham rows 12 full / 2 smoke",
   len(L2.sham_rows(False)) == 12 and len(L2.sham_rows(True)) == 2)
ok("titration 26 full / 0 smoke",
   len(L2.build_titr_b(False)) == 26 and L2.build_titr_b(True) == [])

def synth_rows(plan, real_exact_on, perm_exact_on=(), argmax_map=None):
    rows = []
    for t in plan:
        arm, c = t["arm"], t["concept"]
        exact = (arm == "real" and c in real_exact_on) or \
            (arm == "perm" and c in perm_exact_on)
        top = c if exact else (argmax_map or {}).get(c, "CALIBRATION")
        if top == c and not exact:
            top = "CAPTURE" if c != "CAPTURE" else "TENSION"
        rows.append({"tid": t["tid"], "block": t["block"], "alpha": t["alpha"],
                     "layer": t["layer"], "injected": c, "argmax": top,
                     "argmax_sum": top, "agree_sum_mean": True,
                     "none_top": False, "rank": 1 if exact else 5,
                     "exact": bool(exact), "err": 0.0,
                     "scores": {n: -1.0 for n in L2.SCORED_SET},
                     "scores_sum": {n: -9.0 for n in L2.SCORED_SET},
                     "hook_calls": 1, "pair_id": t["pair_id"], "arm": arm})
    return rows

GEOM_TR = {"RESOLUTION", "CONSTRUCTION", "FAMILIARITY"}
rows_win = synth_rows(full, real_exact_on=GEOM_TR)
sb = L2.stage_b_verdict(rows_win, [], L2.atlas_from_pins(), smoke=False)
ok("planted separation: P-B' pass (p<.05), 24 real vs 0 perm",
   sb["P_B"]["pass"] and sb["P_B"]["real_exact"] == 24
   and sb["P_B"]["perm_exact"] == 0)
ok("mechanism: behavioral subset of geometric, no anomalies",
   sb["mechanism"]["subset_ok"] and sb["mechanism"]["anomalies"] == []
   and sorted(sb["mechanism"]["behavioral_hits"]) == sorted(GEOM_TR))
rows_null = synth_rows(full, real_exact_on=())
sbn = L2.stage_b_verdict(rows_null, [], L2.atlas_from_pins(), smoke=False)
ok("null world: P-B' does not pass", not sbn["P_B"]["pass"])
rows_anom = synth_rows(full, real_exact_on=GEOM_TR | {"CONFIDENCE"})
sba = L2.stage_b_verdict(rows_anom, [], L2.atlas_from_pins(), smoke=False)
ok("planted anomaly: behavioral hit on geometric miss FLAGGED",
   not sba["mechanism"]["subset_ok"]
   and sba["mechanism"]["anomalies"] == ["CONFIDENCE"])
rows_sm = synth_rows(smoke, real_exact_on={L2.TRAINED[0]})
sbs = L2.stage_b_verdict(rows_sm, [], L2.atlas_from_pins(), smoke=True)
ok("smoke constants: stats run (n_pairs matches trained smoke pairs)",
   sbs["P_B"]["n_pairs"] == len({t["pair_id"] for t in smoke
                                 if t["concept"] in L2.TRAINED}))
ok("fork wording: NO_VERDICT / FJ1 / FJ2",
   L2.e8j2_fork(False, sb).startswith("NO_VERDICT")
   and L2.e8j2_fork(True, sb).startswith("FJ1")
   and L2.e8j2_fork(True, sbn).startswith("FJ2"))
ok("S1' concordance shape (held-out, both arms)",
   set(sb["S1_heldout_concordance"]) == {"real", "perm"}
   and set(sb["S1_heldout_concordance"]["real"]) == set(L2.HELD_OUT))

# ══ Group C — verdict cell exec'd VERBATIM vs synthetic flights ══════════════
print("Group C — verdict cell vs synthetic flights")
VERDICT_SRC = cells[6]

def run_verdict(rungb, gates, smoke_mode, plan, cond_errors=None, prep_ship=True):
    tmp = Path(tempfile.mkdtemp(prefix="e8j2_t_"))
    out = tmp / "out"; out.mkdir()
    sem = tmp / "sem"
    inflight = "e8j2/inflight_TEST"
    if prep_ship:
        (sem / inflight).mkdir(parents=True)
        (sem / inflight / "rungb.json").write_text("{}")
    shipped = []
    ns = {n: getattr(L2, n) for n in dir(L2) if not n.startswith("__")}
    ns.update({"RUNGB": rungb, "GATES": gates, "cond_errors": cond_errors or {},
               "SMOKE": smoke_mode, "MODE": "smoke" if smoke_mode else "full",
               "STAMP": "TEST", "MODEL_ID": "test-model", "OUT": out,
               "SEM": sem, "INFLIGHT": inflight, "RESUME_STAMP": "",
               "B_PAIR_ROWS": plan, "json": json,
               "ship": lambda src, dest: shipped.append(str(dest))})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(VERDICT_SRC, "<verdict-cell>", "exec"), ns)
    text = buf.getvalue()
    summary = None
    vf = out / "e8j2_verdict.json"
    if vf.exists():
        summary = json.load(open(vf))
    shutil.rmtree(tmp)
    return text, summary, shipped

def synth_anchor(n_exact=17):
    return [{"exact": i < n_exact, "none_top": False, "hook_calls": 1,
             "scores": {n: -1.0 for n in L2.SCORED_SET}} for i in range(18)]

def synth_shams(claims=0, n=12):
    return [{"none_top": i >= claims, "hook_calls": 0,
             "scores": {n2: -1.0 for n2 in L2.SCORED_SET}} for i in range(n)]

GOOD_GATES = {"g1b": {"pass": True, "delta_pct": 0.0}}

def mk_rungb(fc, anchor, shams, titr=None):
    return {"stamp": "TEST", "mode": "x", "pins": {"sha": L2.E8J2_PINS_SHA},
            "anchor": anchor, "shams": shams,
            "sham_claims": sum(1 for r in shams if not r["none_top"]),
            "fc_rows": fc, "titr_rows": titr or [],
            "derangement": L2.E8J2_DERANGEMENT, "gates_snapshot": GOOD_GATES,
            "secs": 1.0}

# (a) full FJ1
t, s, sh = run_verdict(mk_rungb(rows_win, synth_anchor(17), synth_shams(0)),
                       GOOD_GATES, False, full)
ok("FJ1 world: fork FJ1, gates_all, shipped",
   s and s["fork"].startswith("FJ1") and s["gates_all"]
   and any("full_TEST" in d for d in sh))
ok("FJ1 world: banner carries P-B' and mechanism lines",
   "P-B' naming: real 24 vs perm 0" in t and "subset_ok=True" in t)
# (b) full FJ2
t, s, _ = run_verdict(mk_rungb(rows_null, synth_anchor(17), synth_shams(0)),
                      GOOD_GATES, False, full)
ok("FJ2 world: fork FJ2", s and s["fork"].startswith("FJ2"))
# (c) gate fail (sham claims 5) -> NO_VERDICT
t, s, _ = run_verdict(mk_rungb(rows_win, synth_anchor(17), synth_shams(5)),
                      GOOD_GATES, False, full)
ok("gate-fail world: NO_VERDICT", s and s["fork"].startswith("NO_VERDICT"))
# (d) gate fail (anchor 14/18) -> NO_VERDICT via g1
t, s, _ = run_verdict(mk_rungb(rows_win, synth_anchor(14), synth_shams(0)),
                      GOOD_GATES, False, full)
ok("g1-fail world: NO_VERDICT", s and s["fork"].startswith("NO_VERDICT"))
# (e) smoke green
rows_sm_ok = synth_rows(smoke, real_exact_on={L2.TRAINED[0]})
t, s, _ = run_verdict(mk_rungb(rows_sm_ok, synth_anchor(2)[:2], synth_shams(0, 2)),
                      GOOD_GATES, True, smoke)
ok("smoke world: GREEN banner", "SMOKE GREEN" in t)
# (f) smoke red (missing row)
t, s, _ = run_verdict(mk_rungb(rows_sm_ok[:-1], synth_anchor(2)[:2],
                               synth_shams(0, 2)), GOOD_GATES, True, smoke)
ok("smoke missing-row world: RED banner", "SMOKE RED" in t)
# (g) flight error
t, s, _ = run_verdict(None, None, False, full,
                      cond_errors={"flight": "RuntimeError: boom"})
ok("flight-error world: NO FLIGHT DATA, no verdict file",
   "NO FLIGHT DATA" in t and s is None)
# (h) anomaly world: FJ1 with flagged anomaly in banner
t, s, _ = run_verdict(mk_rungb(rows_anom, synth_anchor(17), synth_shams(0)),
                      GOOD_GATES, False, full)
ok("anomaly world: anomaly surfaces in banner + summary",
   "anomalies=['CONFIDENCE']" in t
   and s["stageb"]["mechanism"]["anomalies"] == ["CONFIDENCE"])

print(f"\nALL GREEN — {PASS} checks")
