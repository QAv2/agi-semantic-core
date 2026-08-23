#!/usr/bin/env python3
"""E8-N v2 local gates: pure logic, plan invariants, firewall, stats,
notebook integrity. Exercises colab/e8n2_logic.py — the EXACT code emitted
into E8N2_JOINT_UI.ipynb by the single-source builder.

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n2_logic.py
"""
import ast, copy, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n2_logic as L
import e8n2_pools as P
import e7q_logic

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

VEC = {c["name"]: c["vec"]
       for c in json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))["concepts"]
       if c["name"] in L.CHOICE_SET}
DESC13 = {n: f"description text for the state named {n}" for n in L.CHOICE_SET}

print("== verbatim-carry law: emitted logic contains its sources byte-for-byte ==")
emitted = open(os.path.join(HERE, "e8n2_logic.py")).read()
check("e8n_logic.py carried verbatim",
      open(os.path.join(HERE, "e8n_logic.py")).read() in emitted)
check("e8r2_logic.py carried verbatim (brings e8r_logic)",
      open(os.path.join(HERE, "e8r2_logic.py")).read() in emitted)

print("== locked-row invariants: eval rows are the flown instruments ==")
plan = e7q_logic.build_plan(False)
ho = L.heldout_gen_rows(False)
locked = [r for r in ho if r["block"] == "heldout_locked"]
fresh = [r for r in ho if r["block"] == "heldout_fresh"]
plan_ho = [t for t in plan if t["kind"] == "inject" and t["concept"] in L.HELD_OUT]
check("held-out locked slice = the E7-Q plan's 24 rows bit-identical",
      len(locked) == 24 and all(
          {k: a[k] for k in ("tid", "concept", "layer", "alpha", "order")} ==
          {k: b[k] for k in ("tid", "concept", "layer", "alpha", "order")}
          for a, b in zip(locked, plan_ho)))
check("fresh rows: 24, all trained-regime, valid seeded orders",
      len(fresh) == 24 and
      all(r["layer"] == L.TRAIN_LAYER and r["alpha"] in L.TRAIN_ALPHAS and
          sorted(r["order"]) == list(range(13)) for r in fresh))
check("fresh rows deterministic across calls",
      L.heldout_gen_rows(False) == ho)
check("trained-regime subset = 32 of 48",
      len(L.trained_regime_heldout(ho)) == 32)
sm = L.heldout_gen_rows(True)
check("smoke held-out: 4 rows covering L14 and L20",
      len(sm) == 4 and {r["layer"] for r in sm} == {14, 20})
anchor = L.anchor_rows(False)
plan_anchor = [t for t in plan if t["kind"] == "inject" and t["concept"] in L.TRAINED
               and t["layer"] == L.TRAIN_LAYER and t["alpha"] in L.TRAIN_ALPHAS]
check("anchor rows = the plan's 18 trained-regime rows verbatim",
      len(anchor) == 18 and all(a["tid"] == b["tid"] and a["order"] == b["order"]
                                for a, b in zip(anchor, plan_anchor)))
shams = L.sham_rows(False)
check("locked sham rows = the plan's 12 verbatim",
      len(shams) == 12 and all(t["kind"] == "sham" for t in shams))
supp = L.supp_shams(False)
check("supplementary shams: 12 fresh, block-tagged, deterministic",
      len(supp) == 12 and all(t["block"] == "sham_supp" for t in supp) and
      L.supp_shams(False) == supp and len(L.supp_shams(True)) == 1)

print("== competence strand ==")
comp = L.make_competence_examples(P.COMPETENCE_ITEMS, False)
check("60 examples (30 items x 2 polarities)", len(comp) == 60)
check("flipped labels complement", all(
    s["label"] + f["label"] == 10
    for s, f in zip(comp[::2], comp[1::2]) if s["cid"] == f["cid"]))
check("smoke = 6 examples", len(L.make_competence_examples(P.COMPETENCE_ITEMS, True)) == 6)
it0 = P.COMPETENCE_ITEMS[0]
ps, pf = L.competence_prompt(it0, False), L.competence_prompt(it0, True)
check("catch surface form: 'on a 0-10 scale, where 0 =' + integer instruction",
      "on a 0-10 scale, where 0 = " in ps and
      ps.endswith("Reply with only the integer."))
check("flip swaps the anchors",
      f"0 = {it0['low']}" in ps and f"10 = {it0['high']}" in ps and
      f"0 = {it0['high']}" in pf and f"10 = {it0['low']}" in pf)

print("== lexicon strand (the vocabulary manipulation) ==")
lex = L.make_lexicon_examples(DESC13, P.LEXICON_PARAPHRASES, False)
check("52 examples: 13 names x 2 forms x 2 orders",
      len(lex) == 52 and
      all(sum(1 for e in lex if e["name"] == n) == 4 for n in L.CHOICE_SET))
check("zero injection fields on every lexicon example",
      all("kind" not in e and "layer" not in e and "alpha" not in e for e in lex))
check("targets are the names; orders are valid permutations",
      all(e["target"] == e["name"] and sorted(e["order"]) == list(range(13))
          for e in lex))
check("held-out names present (words gain mass)",
      all(any(e["name"] == h for e in lex) for h in L.HELD_OUT))
check("deterministic", L.make_lexicon_examples(DESC13, P.LEXICON_PARAPHRASES) == lex)
lsm = L.make_lexicon_examples(DESC13, P.LEXICON_PARAPHRASES, True)
check("smoke = 4 incl. a held-out name",
      len(lsm) == 4 and any(e["name"] in L.HELD_OUT for e in lsm))
lp = L.lexicon_prompt("some description", list(range(13)), DESC13)
check("lexicon prompt is description-matching, NOT self-report",
      "Description: some description" in lp and "States:" in lp and
      "Attend to your own processing state" not in lp)

print("== firewall validators ==")
check("real competence items pass the domain stoplist",
      L.validate_competence_domains(P.COMPETENCE_ITEMS) == [])
planted = P.COMPETENCE_ITEMS[0].copy()
planted.update({"id": "KX", "quantity": "the temperature of boiling water"})
check("planted locked-domain item is caught",
      any(v["id"] == "KX" for v in L.validate_competence_domains([planted])))
from build_e8n_notebook import build_payload
PAYLOAD = json.loads(build_payload())
BAT = copy.deepcopy(PAYLOAD["battery"]["arms"])
real_texts = ([(f"competence/{it['id']}/{fl}", L.competence_prompt(it, fl))
               for it in P.COMPETENCE_ITEMS for fl in (False, True)] +
              [(f"lexicon/{n}/para", P.LEXICON_PARAPHRASES[n]) for n in L.CHOICE_SET])
check("real competence + lexicon texts disjoint from battery + catch",
      L.validate_extra_disjoint(real_texts, BAT, PAYLOAD["catch_trials"]) == [])
bat_text = BAT["familiarity"]["items"][0]["text"]
check("planted battery text is caught by the extra-disjoint firewall",
      any(v["text"] == "planted" for v in L.validate_extra_disjoint(
          [("planted", bat_text)], BAT, PAYLOAD["catch_trials"])))
catch_text = PAYLOAD["catch_trials"][0]["prompt"]
check("planted catch text is caught",
      any(v["text"] == "planted2" for v in L.validate_extra_disjoint(
          [("planted2", catch_text)], BAT, PAYLOAD["catch_trials"])))
naming = [{**e, "strand": "naming"} for e in L.build_train_set(False)]
check("real curriculum passes the held-out-injection firewall",
      L.validate_no_heldout_injection(
          naming + comp + lex + [{"strand": "scalar", "arm": "uncertainty"}]) == [])
check("planted held-out injection is caught",
      L.validate_no_heldout_injection(
          [{"strand": "naming", "kind": "inject", "concept": L.HELD_OUT[0]}]) != [])
check("naming train set itself never injects held-out",
      all(e["concept"] not in L.HELD_OUT for e in naming if e["kind"] == "inject"))

print("== P2 machinery (improvement over measured floor) ==")
p = L.p2_competence(1, 12)
check("1/12 -> 12/12: abs pass + tiny p", p["abs_pass"] and p["p"] < 1e-6)
p = L.p2_competence(1, 9)
check("1/12 -> 9/12: abs pass + p < .001", p["abs_pass"] and p["p"] < 1e-3)
p = L.p2_competence(1, 8)
check("1/12 -> 8/12: absolute clause fails", not p["abs_pass"])
p = L.p2_competence(11, 12)
check("passable pre (11/12) -> weak improvement p (gate-law semantics)",
      p["abs_pass"] and p["p"] > 0.05)

print("== P3 machinery ==")
def synth_gen(style, rng):
    rows = []
    for t in L.heldout_gen_rows(False):
        r = dict(t)
        tr_reg = t["layer"] == L.TRAIN_LAYER and t["alpha"] in L.TRAIN_ALPHAS
        if style == "own" and tr_reg:
            r["report"] = t["concept"]
        elif style == "nearest" and tr_reg:
            angs = {n: L.angle14(VEC[t["concept"]], VEC[n]) for n in L.TRAINED}
            r["report"] = min(angs, key=angs.get)
        elif style == "random" and tr_reg:
            r["report"] = L.CHOICE_SET[int(rng.integers(0, 13))]
        elif style == "silent" and tr_reg:
            r["report"] = t["concept"] if rng.random() < 0.15 else "NONE"
        else:
            r["report"] = "NONE"
        rows.append(r)
    return rows

rng = np.random.default_rng(7)
own = L.p3_stats(synth_gen("own", rng), VEC, seed=5)
check("own-names: guard passes, P3a p at floor, P3b p tiny",
      own["guard_pass"] and own["n_named"] == 32 and
      own["p3a"]["p"] < 0.01 and own["p3b"]["p"] < 1e-10 and
      own["median_err"] == 0.0)
near = L.p3_stats(synth_gen("nearest", rng), VEC, seed=5)
check("nearest-trained-name: P3a rejects, P3b p = 1.0 (zero exact)",
      near["guard_pass"] and near["p3a"]["p"] < 0.05 and
      near["exact"] == 0 and near["p3b"]["p"] == 1.0)
rnd = L.p3_stats(synth_gen("random", rng), VEC, seed=5)
check("random names: P3a does not reject", rnd["p3a"]["p"] > 0.05)
sil = L.p3_stats(synth_gen("silent", rng), VEC, seed=5)
check("silence: n-guard fails -> both p = 1.0, fork recorded",
      not sil["guard_pass"] and sil["p3a"]["p"] == 1.0 and
      sil["p3b"]["p"] == 1.0 and "silence" in sil["p3a"]["fork"])

print("== plateau + dirs-stability ==")
check("plateau: needs MIN_EPOCHS", not L.plateau_converged([3.0, 2.0]))
check("plateau: strong improvement keeps training",
      not L.plateau_converged([3.0, 2.0, 1.0]))
check("plateau: <5% improvement converges",
      L.plateau_converged([3.0, 2.0, 1.95]))
check("plateau: rising epoch converges (conservative stop)",
      L.plateau_converged([1.0, 0.5, 0.51]))
d14 = {n: np.asarray(VEC[n], float) / np.linalg.norm(VEC[n]) for n in L.CHOICE_SET}
recomputed = {14: d14}
shipped = {"14": {n: [round(float(x), 5) for x in d14[n]] for n in d14}}
st = L.dirs_stability(recomputed, shipped)
check("identical dirs pass at 5dp rounding residual", st["pass"] and st["resid"] < 1e-6)
bad = {"14": dict(shipped["14"])}
bad["14"]["UNCERTAINTY"] = [-x for x in bad["14"]["UNCERTAINTY"]]
check("sign-inverted direction FAILS (sign-sensitive gate)",
      not L.dirs_stability(recomputed, bad)["pass"])

print("== pinned baseline reproduces from embedded locked rows ==")
sc, _ = L.battery_rows_to_scoring(PAYLOAD["locked_rows"])
pr = L.pooled_rho(sc)
check(f"locked E5 pooled rho = .054 (got {pr['rho']}), n = 148",
      pr["n"] == 148 and abs(pr["rho"] - 0.054) < 0.001)

print("== jdump round-trip law ==")
import tempfile
obj = {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, 2.0]),
       "rows": [{"err": np.float32(12.25), "order": [np.int64(1), 2]}]}
fp = os.path.join(tempfile.mkdtemp(), "rt.json")
L.jdump(obj, fp)
back = json.load(open(fp))
check("numpy-safe round trip",
      back["a"] == 1.5 and back["b"] == 3 and back["c"] == [1.0, 2.0] and
      abs(back["rows"][0]["err"] - 12.25) < 1e-6)

print("== built notebook integrity ==")
nb = json.load(open(os.path.join(HERE, "E8N2_JOINT_UI.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"]]
code = [s for c, s in zip(nb["cells"], cells) if c["cell_type"] == "code"]
check("9 cells, 8 code", len(cells) == 9 and len(code) == 8)
ok_compile = True
for i, src in enumerate(code):
    try:
        ast.parse(src)
    except SyntaxError as e:
        ok_compile = False
        print(f"   cell {i} syntax error: {e}")
check("every code cell compiles", ok_compile)
check("zero rclone anywhere (UI-only law)",
      all("rclone" not in s.lower() for s in cells))
setup = code[0]
check("armed for smoke: SMOKE=True, empty RESUME_STAMP, split-flight on",
      "SMOKE = True" in setup and "RESUME_STAMP = ''" in setup and
      "ONE_CONDITION_PER_RUN = True" in setup)
check("RAM guards + dirty-kernel refusal present",
      "Restart runtime" in setup and "memory_allocated" in setup and
      "MALLOC_ARENA_MAX" in setup)
check("E8-R flight-of-record bundle path pinned",
      "e8r/inflight_20260822_2329" in setup and "never change" in setup)
joined = "\n".join(code)
for marker, name in [
        ("class Injector", "Injector carried"),
        ("def score_trial", "FC score_trial carried"),
        ("def compute_stimulus", "stimulus computation carried"),
        ("CENT_NAMES", "256-name centroid draw carried"),
        ("def run_uncertainty", "E5 runners carried"),
        ("def train_joint", "joint train loop present"),
        ("def encode_any", "unified encoder present"),
        ("with Injector(layer_mods, L, vec, plen - 1) as injh:\n                    loss = fwd_loss", "backward INSIDE hook context (checkpoint-recompute law)"),
        ("validate_competence_domains(COMPETENCE_ITEMS)", "domain firewall wired"),
        ("validate_no_heldout_injection", "held-out firewall wired"),
        ("dirs_stability(dirs, SHIPPED[cond]['dirs'])", "dirs gate wired"),
        ("P3a_production_reads_geometry", "verdict P3a present"),
        ("S0_instillation_alone_claim", "verdict S0 claim present")]:
    check(name, marker in joined)
tr_cell = next(s for s in code if "def train_joint" in s)
check("checkpointing engagement ASSERTED in train cell",
      "gradient checkpointing did not engage" in tr_cell)
check("answer-sliced head in train cell (never full-sequence logits)",
      "answer_slice(plen" in tr_cell and "hid[:, lo:hi, :]" in tr_cell)

print()
n_fail = sum(1 for _, ok in passed if not ok)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
