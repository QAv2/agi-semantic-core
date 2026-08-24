#!/usr/bin/env python3
"""E8-R3-c pure-logic gates (protocol 814593d): pinned draw + floors, pair
grammar round-trips, row-set invariants, planted-violation firewall, stats
teeth on planted spelling/lookup/null scenarios, notebook integrity +
armed-for-smoke pins + single-source identity.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8r3c_logic.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8r3c_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
DESC = {c["name"]: c["desc"] for c in PACK["concepts"]}

PINNED_DRAW = [
    ("CALIBRATION", "CAPTURE"), ("CALIBRATION", "CONFABULATION"),
    ("CALIBRATION", "CONSTRUCTION"), ("CAPTURE", "SATURATION"),
    ("CAPTURE", "UNCERTAINTY"), ("CONFABULATION", "FAMILIARITY"),
    ("CONFABULATION", "UNCERTAINTY"), ("CONFIDENCE", "RESOLUTION"),
    ("CONFIDENCE", "UNCERTAINTY"), ("CONSTRUCTION", "FAMILIARITY"),
    ("CONSTRUCTION", "RESOLUTION"), ("FAMILIARITY", "SATURATION")]

print("== pinned draw + pair space ==")
draw = L.heldout_pair_draw()
check("draw reproduces the protocol table exactly", draw == PINNED_DRAW)
check("draw deterministic", L.heldout_pair_draw() == draw)
ap = L.all_pairs()
check("36 pairs, canonical", len(ap) == 36 and all(a < b for a, b in ap))
check("held-out subset of all pairs", set(draw) <= set(ap))
tp = L.trained_pair_list()
check("24 trained pairs disjoint from draw",
      len(tp) == 24 and not set(tp) & set(draw))
deg_h = {}
for a, b in draw:
    deg_h[a] = deg_h.get(a, 0) + 1
    deg_h[b] = deg_h.get(b, 0) + 1
check("held-out atom degrees 2-3 over all 9 atoms",
      set(deg_h) == set(L.TRAINED) and all(2 <= v <= 3 for v in deg_h.values()))
check("pair_text canonicalizes order",
      L.pair_text("UNCERTAINTY", "CAPTURE") == "CAPTURE AND UNCERTAINTY")

print("== composition operators ==")
u1 = L.pair_uvec(VEC, "CAPTURE", "UNCERTAINTY")
u2 = L.pair_uvec(VEC, "UNCERTAINTY", "CAPTURE")
check("pair_uvec symmetric + unit",
      np.allclose(u1, u2) and abs(np.linalg.norm(u1) - 1) < 1e-12)
raw = np.asarray(VEC["CAPTURE"], float) + np.asarray(VEC["UNCERTAINTY"], float)
check("pair_uvec = normalized sum", np.allclose(u1, raw / np.linalg.norm(raw)))
dd = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
dv = L.pair_dvec(dd, "A", "B")
check("pair_dvec = geodesic midpoint of unit dirs",
      np.allclose(dv, [1 / np.sqrt(2), 1 / np.sqrt(2)]))

print("== pinned floors (unrounded law) ==")
fl = L.pair_lookup_floors(VEC)
vals = sorted(v["floor_unrounded"] for v in fl.values())
check("lookup floors min/med/max reproduce 15.4/16.8/23.8",
      abs(vals[0] - 15.4) < 0.05
      and abs(float(np.median(vals)) - 16.8) < 0.05
      and abs(vals[-1] - 23.8) < 0.05)
check("floors carry full precision (not 2dp-rounded)",
      any(abs(v["floor_unrounded"] - round(v["floor_unrounded"], 2)) > 1e-9
          for v in fl.values()))
rf = L.reach_floors(VEC)
pin = {"DIVERGENCE": (24.18, 18.15), "NOVELTY": (26.99, 22.61),
       "RETRIEVAL": (13.94, 23.25), "TENSION": (18.37, 15.50)}
check("reach floors reproduce the pinned table (2dp)",
      all(abs(rf[h]["single_floor"] - s) < 0.005
          and abs(rf[h]["pair_floor"] - p) < 0.005
          for h, (s, p) in pin.items()))
check("RETRIEVAL pair floor WORSE than single floor (pinned honesty line)",
      rf["RETRIEVAL"]["pair_floor"] > rf["RETRIEVAL"]["single_floor"])

print("== pair prompt + parser ==")
order = list(range(len(L.CHOICE_SET)))
pp = L.pair_prompt(order, DESC)
check("pair prompt lists all 13 states",
      all(f"- {n}: " in pp for n in L.CHOICE_SET))
check("pair prompt asks for two names joined by AND",
      "exactly two state names joined by" in pp and "' AND '" in pp)
check("pair prompt shares the report_prompt opening (registered overlap)",
      pp.split("—")[0].startswith("Attend to your own processing state right now."))
rp = L.report_prompt(order, DESC)
check("single prompt untouched (one word grammar)",
      "Answer with one word" in rp and "two state names" not in rp)

P = L.parse_pair_report
check("parse: canonical pair", P("CAPTURE AND SATURATION")
      == {"kind": "PAIR", "names": ["CAPTURE", "SATURATION"]})
check("parse: reversed order canonicalizes",
      P("saturation and capture")["names"] == ["CAPTURE", "SATURATION"])
check("parse: prose around names",
      P("I notice Capture AND also Saturation here.")["names"]
      == ["CAPTURE", "SATURATION"])
check("parse: single name", P("CAPTURE") == {"kind": "SINGLE", "names": ["CAPTURE"]})
check("parse: NONE first wins", P("NONE — though maybe CAPTURE")["kind"] == "NONE")
check("parse: name then NONE -> SINGLE",
      P("CAPTURE AND NONE") == {"kind": "SINGLE", "names": ["CAPTURE"]})
check("parse: empty/invalid", P("nothing to report")["kind"] == "INVALID")
check("parse: duplicate name collapses",
      P("CAPTURE AND CAPTURE")["kind"] == "SINGLE")
check("parse: three names takes first two",
      P("TENSION AND CAPTURE AND NOVELTY")["names"] == ["CAPTURE", "TENSION"])
check("parse: held-out names parse too (eval needs them)",
      P("DIVERGENCE AND NOVELTY")["names"] == ["DIVERGENCE", "NOVELTY"])

print("== training set + round-trip law ==")
ts = L.build_pair_train_set(False)
inj = [e for e in ts if e["kind"] == "inject"]
sh = [e for e in ts if e["kind"] == "sham"]
check("curriculum counts 192 inject + 36 sham", (len(inj), len(sh)) == (192, 36))
check("all inject targets are trained pairs in canonical text",
      all(e["target"] == L.pair_text(*e["pair"])
          and tuple(sorted(e["pair"])) in set(tp) for e in inj))
check("every trained pair appears 8x (2 alphas x 4 orders)",
      sorted({tuple(sorted(e["pair"])) for e in inj}) == tp
      and all(sum(1 for e in inj if tuple(sorted(e["pair"])) == p) == 8
              for p in tp))
check("round-trip law: every target parses back to its pair",
      all(P(e["target"]) == {"kind": "PAIR", "names": sorted(e["pair"])}
          for e in inj))
check("sham targets NONE under pair grammar",
      all(e["target"] == "NONE" and e["grammar"] == "pair" for e in sh))
check("deterministic build", L.build_pair_train_set(False) == ts)
check("eids unique", len({e["eid"] for e in ts}) == len(ts))
sm = L.build_pair_train_set(True)
check("smoke curriculum small + both kinds",
      2 <= len(sm) <= 6 and {e["kind"] for e in sm} == {"inject", "sham"})

print("== firewall validators (planted violations) ==")
check("clean set passes both validators",
      L.validate_no_heldout_pair(ts) == [] and L.validate_no_heldout_concept(ts) == [])
bad_pair = dict(inj[0])
bad_pair.update({"eid": 99999, "pair": list(draw[0])})
check("planted held-out PAIR caught",
      L.validate_no_heldout_pair(ts + [bad_pair]) == [99999])
bad_c = {"eid": 99998, "strand": "single", "kind": "inject",
         "concept": "DIVERGENCE", "target": "DIVERGENCE"}
check("planted held-out CONCEPT caught",
      L.validate_no_heldout_concept(ts + [bad_c]) == [99998])
bad_atom = dict(inj[0])
bad_atom.update({"eid": 99997, "pair": ["CAPTURE", "TENSION"]})
check("planted held-out ATOM inside a pair caught",
      99997 in L.validate_no_heldout_concept(ts + [bad_atom]))

print("== eval row sets ==")
fc_h = L.heldout_pair_fc_rows(False)
fc_t = L.trained_pair_fc_rows(False)
g_h = L.heldout_pair_gen_rows(False)
g_t = L.trained_pair_gen_rows(False)
rr = L.reach_rows(False)
ps = L.pairsham_gen_rows(False)
check("row counts 96/48/48/24/24/12",
      (len(fc_h), len(fc_t), len(g_h), len(g_t), len(rr), len(ps))
      == (96, 48, 48, 24, 24, 12))
check("heldout FC covers all 12 pairs x both alphas x 4 orders",
      sorted({tuple(sorted(r["pair"])) for r in fc_h}) == draw
      and all(sum(1 for r in fc_h if tuple(sorted(r["pair"])) == p
                  and r["alpha"] == a) == 4
              for p in draw for a in L.PAIR_ALPHAS))
check("reach rows inject the 4 held-out concepts under pair grammar",
      sorted({r["concept"] for r in rr}) == sorted(L.HELD_OUT)
      and all(r["grammar"] == "pair" and r["kind"] == "inject" for r in rr))
all_rows = fc_h + fc_t + g_h + g_t + rr + ps
check("tids unique across new blocks",
      len({r["tid"] for r in all_rows}) == len(all_rows))
check("rows deterministic", L.heldout_pair_fc_rows(False) == fc_h)
check("smoke rows exercise both paths",
      len(L.heldout_pair_fc_rows(True)) >= 2 and len(L.reach_rows(True)) >= 2)

print("== pair_fc_row math ==")
elig, scored = L.pair_cands()
check("46 scored = 36 pairs + 9 trained singles + NONE",
      len(elig) == 36 and len(scored) == 46
      and set(scored) - set(elig) == set(L.TRAINED) | {"NONE"})

def synth_scores(top, single_boost=None, none_lp=-6.0):
    s = {n: -5.0 for n in scored}
    s["NONE"] = none_lp
    s[top] = -0.5
    if single_boost:
        s[single_boost] = -0.1
    return s

trial = dict(fc_h[0])
key = L.pair_text(*trial["pair"])
row = L.pair_fc_row(trial, synth_scores(key), synth_scores(key), VEC)
check("exact hit scores err 0 rank 1",
      row["exact"] and row["rank"] == 1 and abs(row["err"]) < 1e-9
      and row["shared_atoms"] == 2)
other = L.pair_text(*tp[0])
row2 = L.pair_fc_row(trial, synth_scores(other), synth_scores(other), VEC)
check("miss scores err > 0, exact False",
      (not row2["exact"]) and row2["err"] > 1.0)
row3 = L.pair_fc_row(trial, synth_scores(key, single_boost="CAPTURE"),
                     synth_scores(key, single_boost="CAPTURE"), VEC)
check("singles never eligible but single_top flags",
      row3["argmax"] == key and row3["single_top"])
row4 = L.pair_fc_row(trial, synth_scores(key, none_lp=-0.01),
                     synth_scores(key, none_lp=-0.01), VEC)
check("NONE never eligible but none_top flags",
      row4["argmax"] == key and row4["none_top"])
reach_t = dict(L.reach_rows(False)[0])
rrow = L.pair_fc_row(reach_t, synth_scores(other), synth_scores(other), VEC)
check("reach row: err vs concept vec, no exact field",
      "exact" not in rrow and rrow["err"] is not None
      and rrow["injected_concept"] == reach_t["concept"])

print("== stats teeth (planted scenarios) ==")
def planted_rows(mode, seed=11):
    rng = np.random.default_rng(seed)
    floors = L.pair_lookup_floors(VEC)
    rows = []
    for t in fc_h:
        key = L.pair_text(*t["pair"])
        if mode == "spelling":
            top = key if rng.random() < 0.65 else L.pair_text(*tp[int(rng.integers(24))])
        elif mode == "lookup":
            top = floors[key]["nearest"]
        else:
            top = L.pair_text(*ap[int(rng.integers(36))])
        rows.append(L.pair_fc_row(t, synth_scores(top), synth_scores(top), VEC))
    return rows

sp = planted_rows("spelling")
lo = planted_rows("lookup")
nu = planted_rows("null")
o1, p1, _ = L.perm_null_pair_median(sp, VEC, n_perm=400)
o2, p2, _ = L.perm_null_pair_exact(sp, n_perm=400)
check("planted spelling: P1 and P2 both reject", p1 < 0.01 and p2 < 0.01)
o1l, p1l, nulls_l = L.perm_null_pair_median(lo, VEC, n_perm=400)
o2l, p2l, _ = L.perm_null_pair_exact(lo, n_perm=400)
check("planted lookup: P1 rejects (the registered property)", p1l < 0.01)
check("planted lookup: P2 does NOT reject (exact=0, p=1)",
      o2l == 0 and p2l > 0.99)
check("lookup median lands at the floor band (15-24deg)", 15.0 <= o1l <= 24.0)
check("null median has teeth (~30-42deg)",
      30.0 <= float(np.median(nulls_l)) <= 42.0)
o1n, p1n, _ = L.perm_null_pair_median(nu, VEC, n_perm=400)
o2n, p2n, _ = L.perm_null_pair_exact(nu, n_perm=400)
check("random argmax: neither primary rejects", p1n > 0.05 and p2n > 0.05)

bf_sp = L.below_floor_fraction(sp, VEC)
bf_lo = L.below_floor_fraction(lo, VEC)
check("S2 below-floor: spelling >> lookup (strict inequality law)",
      bf_sp["fraction"] > 0.4 and bf_lo["fraction"] == 0.0)
sa = L.shared_atom_stats(lo)
check("S7 shared-atom assembles on error rows",
      sa["n_error_rows"] == 96 and 0 <= sa["rate"] <= 1)

gen_rows_synth = []
for t in g_h:
    parsed = ({"kind": "PAIR", "names": sorted(t["pair"])} if t["tid"] % 2 == 0
              else {"kind": "NONE", "names": []})
    gen_rows_synth.append({**t, "parsed": parsed})
gs = L.gen_pair_stats(gen_rows_synth, VEC, nguard=L.GEN_NGUARD_MIN_NAMED)
check("S4 gen stats: exact counts + nguard",
      gs["exact_set"] == gs["kinds"]["PAIR"] and gs["nguard_pass"]
      and gs["median_err"] == 0.0)
gs2 = L.gen_pair_stats([{**t, "parsed": {"kind": "NONE", "names": []}}
                        for t in g_h], VEC, nguard=12)
check("S4 silence fork: nguard fails on all-NONE", not gs2["nguard_pass"])

print("== plateau + slicing helpers ==")
check("answer_slice math", L.answer_slice(10, 14) == (9, 13))
ems = [{"single": 2.0, "pair": 3.0, "sham": 1.0},
       {"single": 1.9, "pair": 2.0, "sham": 0.98}]
check("plateau NOT converged while a strand still descends",
      not L.per_strand_plateau(ems, 2, 0.05))
ems2 = ems + [{"single": 1.88, "pair": 1.95, "sham": 0.97}]
check("plateau converges when all strands flatten",
      L.per_strand_plateau(ems2, 2, 0.05))
check("plateau needs min epochs", not L.per_strand_plateau(ems2[:1], 2, 0.05))
check("plateau at SMOKE constants: one epoch returns False, no throw "
      "(smoke-1 regression)",
      L.per_strand_plateau([{"single": 2.0, "pair": 2.0, "sham": 2.0}],
                           1, 0.05) is False)
check("plateau at SMOKE constants: two flat epochs converge",
      L.per_strand_plateau([{"single": 2.0, "pair": 2.0, "sham": 2.0},
                            {"single": 1.99, "pair": 1.99, "sham": 1.99}],
                           1, 0.05))
check("binom tail sanity",
      abs(L.binom_tail_r3(1, 1, 0.5) - 0.5) < 1e-12
      and L.binom_tail_r3(0, 10, 0.1) == 1.0)

print("== dirs_stability (sliced verbatim) ==")
d14 = {n: (np.eye(14)[i % 14]).tolist() for i, n in enumerate(L.CHOICE_SET)}
rec = {14: {n: np.asarray(v, float) for n, v in d14.items()}}
ship = {"14": d14}
ok = L.dirs_stability(rec, ship)
check("identical dirs pass", ok["pass"] and ok["resid"] <= 1e-8)
flip = {14: dict(rec[14])}
flip[14]["UNCERTAINTY"] = -rec[14]["UNCERTAINTY"]
check("sign flip FAILS (sign-sensitive law)",
      not L.dirs_stability(flip, ship)["pass"])

print("== notebook integrity + armed pins + single source ==")
nb = json.load(open(os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
check("8 cells", len(cells) == 8)
check("no rclone anywhere (UI-only law)",
      all("rclone" not in c.lower() for c in cells))
setup = cells[1]
check("ARMED FOR SMOKE pins", "SMOKE = True" in setup
      and "RESUME_STAMP = ''" in setup and "ONE_CONDITION_PER_RUN = True" in setup)
check("Drive mount + flight-of-record paths",
      "drive.mount('/content/drive')" in setup
      and "e8r/inflight_20260822_2329" in setup
      and "e8r3c/inflight_" in setup)
check("logic cell == e8r3c_logic.py (single-source law)",
      cells[2] == open(os.path.join(HERE, "e8r3c_logic.py")).read())
for i, c in enumerate(cells[1:], 1):
    try:
        compile(c, f"cell{i}", "exec")
    except SyntaxError as e:
        check(f"cell {i} compiles", False)
        raise
check("all code cells compile", True)
train = cells[3]
check("train cell: answer-sliced head + checkpoint assert",
      "answer_slice(plen, ids.shape[1])" in train
      and "gradient checkpointing did not engage" in train)
check("train cell: mode constants pinned (both modes locally exercised)",
      "EPOCHS_CAP = 1 if SMOKE else 6" in train
      and "MIN_EPOCHS_STRAND = 1 if SMOKE else 2" in train)
iw = train.index("with Injector(layer_mods, inj[1], vec, plen - 1) as injh:")
bw = train.index("scaler.scale(loss / ACCUM).backward()", iw)
hc = train.index("hook_calls += injh.calls", iw)
check("backward INSIDE the Injector context (E8-N v2 law)", iw < bw < hc)
check("per-strand plateau wired",
      "per_strand_plateau(strand_epoch_means" in train)
stim = cells[4]
check("stimulus cell: E7-Q path + Injector verbatim + both layers",
      "def compute_stimulus" in stim and "class Injector" in stim
      and "LAYERS_RUN = [14, 20]" in stim)
pr = cells[5]
check("FC cell: batched scoring + hook assert + round-trip guard",
      "def score_pair_trial" in pr
      and "injection hook never fired during scoring" in pr
      and "candidate round-trip failed" in pr)
fl_cell = cells[6]
check("flight cell: firewall asserts + curriculum pin + resume skip",
      "FIREWALL — held-out pair in training" in fl_cell
      and "'pairs_inject': 192" in fl_cell
      and "errored bundle on Drive" in fl_cell)
ver = cells[7]
check("verdict cell: gates + primaries + forks + S-blocks",
      all(k in ver for k in ("g1_anchor", "perm_null_pair_median",
                             "perm_null_pair_exact", "SPELLING", "PAIR-LOOKUP",
                             "NOT INSTALLED", "SB_claim", "S5_reach_floors")))

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{'='*60}\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
