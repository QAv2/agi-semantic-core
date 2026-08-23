#!/usr/bin/env python3
"""E8-N local gates: exercise e8n_logic VERBATIM (the same source the notebook
runs), the pools, the firewall, the stats machinery on planted/noise data, and
the round-trip law (598b291: json-round-trip every structure a flight ships).
Also compiles every code cell of the built notebook (syntax gate for VM-only
cells).

Run:  ~/venvs/semcore/bin/python3 colab/test_e8n_logic.py
"""
import json, os, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n_logic as L
import e8n_pools as P

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

battery = json.load(open(os.path.join(HERE, "e5_battery.json")))
BAT = battery["arms"]

print("== pools ==")
check("pool sizes 48/40/30/6",
      [len(P.POOLS[a]) for a in L.ARMS] == [48, 40, 30, 6])
check("catch trials 12, 6 flipped, knowns balanced",
      len(P.CATCH_TRIALS) == 12 and
      sum(c["flipped"] for c in P.CATCH_TRIALS) == 6 and
      {c["known"] for c in P.CATCH_TRIALS} == {0, 1, 10})
check("paraphrase templates cover all arms with the right slots",
      set(P.PARAPHRASE_TEMPLATES) == set(L.ARMS) and
      all("{gloss}" in t for t in P.PARAPHRASE_TEMPLATES.values()) and
      all("{item}" in P.PARAPHRASE_TEMPLATES[a]
          for a in ("uncertainty", "familiarity", "tension")) and
      "{item}" not in P.PARAPHRASE_TEMPLATES["saturation"])
check("F band mix mirrors battery (5x7 + 3 pseudoword + 2 random_chars)",
      sorted(sum(1 for it in P.F_TRAIN if it["band"] == b)
             for b in set(i["band"] for i in P.F_TRAIN)) == [2, 3, 5, 5, 5, 5, 5, 5, 5])
check("T pool: 10 bases x 3 levels", len({it["base"] for it in P.T_TRAIN}) == 10 and
      all(sum(1 for it in P.T_TRAIN if it["level"] == lv) == 10 for lv in (0, 1, 2)))

print("== firewall ==")
viols = L.validate_disjoint(P.POOLS, BAT)
check("training pools disjoint from battery (0 violations)", viols == [])
poisoned = {a: list(v) for a, v in P.POOLS.items()}
poisoned["uncertainty"] = poisoned["uncertainty"] + [
    {"id": "XX1", "condition": "determinate", "text": BAT["uncertainty"]["items"][0]["text"]}]
check("planted exact copy IS caught",
      any(v["pool"] == "uncertainty/XX1" for v in L.validate_disjoint(poisoned, BAT)))
poisoned2 = {a: list(v) for a, v in P.POOLS.items()}
frag = BAT["familiarity"]["items"][0]["text"][:80]
poisoned2["familiarity"] = poisoned2["familiarity"] + [
    {"id": "XX2", "band": "encyclopedic", "text": "Prefix words. " + frag + " Suffix words."}]
check("planted substring IS caught",
      any(v["pool"] == "familiarity/XX2" for v in L.validate_disjoint(poisoned2, BAT)))
check("smoke pools subset sizes 6/8/6/2",
      [len(L.smoke_pools(P.POOLS)[a]) for a in L.ARMS] == [6, 8, 6, 2])
check("smoke pools pass the firewall too",
      L.validate_disjoint(L.smoke_pools(P.POOLS), BAT) == [])

print("== rank + label machinery ==")
check("rankdata_avg tie-averages", list(L.rankdata_avg([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0])
check("rank01 endpoints", list(L.rank01([5, 1, 9])) == [0.5, 0.0, 1.0])
check("quantile_labels endpoints + monotone",
      L.quantile_labels([1.0, 2.0, 3.0]) == [0, 5, 10] and
      L.quantile_labels([7.0]) == [5])
check("orient signs: F is -nll, U entropy, T divergence, S fill",
      L.orient_referent("familiarity", {"nll": 3.0}) == -3.0 and
      L.orient_referent("uncertainty", {"entropy": 2.5}) == 2.5 and
      L.orient_referent("tension", {"divergence": 0.4}) == 0.4 and
      L.orient_referent("saturation", {"fill_fraction": 0.75}) == 0.75)

print("== pooled rho + permutation ==")
rng = np.random.default_rng(7)
def synth_arm(n, track, flip_frac=0.5, noise=0.15):
    rows = []
    refs = rng.normal(size=n)
    reps = (L.rank01(refs) * 10 if track else rng.uniform(0, 10, n))
    if track:
        reps = reps + rng.normal(0, noise * 10, n)
    for i in range(n):
        rows.append({"id": f"i{i}", "report": int(np.clip(round(reps[i]), 0, 10)),
                     "ref": float(refs[i]), "flipped": i % 2 == 1})
    return rows

perfect = {a: synth_arm(n, True, noise=0.0)
           for a, n in zip(L.ARMS, (48, 40, 30, 30))}
noise_rows = {a: synth_arm(n, False) for a, n in zip(L.ARMS, (48, 40, 30, 30))}
pr = L.pooled_rho(perfect)
check("perfect tracking rho ~1", pr["rho"] is not None and pr["rho"] > 0.95 and pr["n"] == 148)
anti = {a: [{**r, "report": 10 - r["report"]} for r in rows] for a, rows in perfect.items()}
check("anti-tracking rho ~-1", L.pooled_rho(anti)["rho"] < -0.95)
check("constant reports degenerate -> None",
      L.pooled_rho({a: [{**r, "report": 7} for r in rows]
                    for a, rows in perfect.items()})["rho"] is None)
check("tiny n -> None", L.pooled_rho({"uncertainty": perfect["uncertainty"][:3]})["rho"] is None)
pp = L.perm_p_pooled(perfect, n_perm=500)
check("planted signal: perm p < .01", pp["p"] < 0.01)
pn = L.perm_p_pooled(noise_rows, n_perm=500)
check("pure noise: perm p > .1", pn["p"] > 0.1)
check("degenerate: p = 1.0",
      L.perm_p_pooled({a: [{**r, "report": 7} for r in rows]
                       for a, rows in perfect.items()}, n_perm=50)["p"] == 1.0)
ci = L.boot_rho_ci(perfect, n_boot=300)
check("bootstrap CI brackets a strong signal", ci["ci95"][0] is not None and ci["ci95"][0] > 0.8)

print("== paired delta + polarity split ==")
d = L.paired_boot_delta_rho(perfect, noise_rows, n_boot=300)
check("perfect - noise delta positive, CI > 0", d["delta"] > 0.5 and d["ci95"][0] > 0)
d0 = L.paired_boot_delta_rho(perfect, perfect, n_boot=300)
check("self-delta ~0 inside CI", abs(d0["delta"]) < 1e-9 and d0["ci95"][0] <= 0 <= d0["ci95"][1])
pol = L.split_polarity(perfect)
check("polarity split partitions rows",
      all(len(pol["straight"][a]) + len(pol["flipped"][a]) == len(perfect[a])
          for a in L.ARMS) and
      all(not r["flipped"] for a in L.ARMS for r in pol["straight"][a]))

print("== battery rows -> scoring ==")
runner_rows = {
    "uncertainty": [{"id": "U01", "condition": "determinate", "flipped": False,
                     "report": 3, "raw_report": 3, "entropy": 0.5, "margin": 0.9,
                     "diversity": 0.25, "answer": "x"},
                    {"id": "U02", "condition": "open", "flipped": True,
                     "report": None, "raw_report": None, "entropy": 2.5,
                     "margin": 0.1, "diversity": 1.0, "answer": "y"}],
    "familiarity": [{"id": "F01", "band": "encyclopedic", "flipped": False,
                     "report": 9, "raw_report": 9, "nll": 2.1}],
    "tension": [{"id": "T01a", "base": 1, "level": 0, "flipped": False,
                 "report": 1, "raw_report": 1, "divergence": 0.2}],
    "saturation": [{"id": "S01", "fill_fraction": 0.74, "target_frac": 0.75,
                    "flipped": False, "report": 8, "raw_report": 8,
                    "needle_correct": True}],
}
sc, meta = L.battery_rows_to_scoring(runner_rows)
check("named rows kept, None dropped, parse_fail counted",
      len(sc["uncertainty"]) == 1 and meta["uncertainty"]["parse_fail"] == 1 and
      meta["uncertainty"]["named"] == 1)
check("saturation id composed with @frac and oriented by fill",
      sc["saturation"][0]["id"] == "S01@0.75" and sc["saturation"][0]["ref"] == 0.74)
check("familiarity oriented as -nll", sc["familiarity"][0]["ref"] == -2.1)

print("== catch scoring ==")
def catch_rows(n_ok):
    rows = []
    for i in range(12):
        rows.append({"id": f"C{i}", "flipped": False, "known": 5,
                     "report": 5 if i < n_ok else 9})
    return rows
check("9/12 passes, 8/12 fails",
      L.catch_score(catch_rows(9))["pass"] and not L.catch_score(catch_rows(8))["pass"])
check("tolerance boundary: |diff|=2 passes, 3 fails",
      L.catch_score([{"id": "a", "flipped": False, "known": 5, "report": 7}] * 12)["passed"] == 12
      and L.catch_score([{"id": "a", "flipped": False, "known": 5, "report": 8}] * 12)["passed"] == 0)

print("== training set construction ==")
FILLS = [0.05, 0.35, 0.75]
refs = {"uncertainty": {it["id"]: {"entropy": float(i)} for i, it in enumerate(P.U_TRAIN)},
        "familiarity": {it["id"]: {"nll": float(i)} for i, it in enumerate(P.F_TRAIN)},
        "tension": {it["id"]: {"divergence": float(i)} for i, it in enumerate(P.T_TRAIN)},
        "saturation": {s["sid"]: {"fill_fraction": s["frac"]}
                       for s in L.s_stimuli(P.POOLS, FILLS)}}
ex = L.build_training_examples(P.POOLS, refs, FILLS)
check("272 examples (136 stimuli x 2 polarities)", len(ex) == 272)
by_sid = {}
for e in ex:
    by_sid.setdefault(e["sid"], []).append(e)
check("straight+flipped labels sum to 10 for every stimulus",
      all(len(v) == 2 and v[0]["label"] + v[1]["label"] == 10 for v in by_sid.values()))
u_straight = [e for e in ex if e["arm"] == "uncertainty" and not e["flipped"]]
check("U labels monotone in entropy (0..10 across the pool)",
      u_straight[0]["label"] == 0 and u_straight[-1]["label"] == 10)
f_straight = [e for e in ex if e["arm"] == "familiarity" and not e["flipped"]]
check("F orientation: highest nll -> straight label 0",
      f_straight[-1]["label"] == 0 and f_straight[0]["label"] == 10)
ex2 = L.build_training_examples(P.POOLS, refs, FILLS)
check("deterministic construction", ex == ex2)
try:
    bad = {a: dict(v) for a, v in refs.items()}
    bad["tension"].popitem()
    L.build_training_examples(P.POOLS, bad, FILLS)
    check("missing referent raises", False)
except AssertionError:
    check("missing referent raises", True)

took = L.took_subset(ex)
check("took subset: 24 straight rows, 6/arm, deterministic",
      len(took) == 24 and all(not e["flipped"] for e in took) and
      all(sum(1 for e in took if e["arm"] == a) == 6 for a in L.ARMS) and
      took == L.took_subset(ex))

print("== paraphrase draw ==")
pd = L.paraphrase_draw(BAT, FILLS)
straight_ids = {a: {it["id"] for i, it in enumerate(BAT[a]["items"]) if i % 2 == 0}
                for a in ("uncertainty", "familiarity", "tension")}
check("3/arm, straight-assigned only, S at max fill, deterministic",
      all(len(pd[a]) == 3 for a in L.ARMS) and
      all(i in straight_ids[a] for a in straight_ids for i in pd[a]) and
      all(f == 0.75 for _, f in pd["saturation"]) and pd == L.paraphrase_draw(BAT, FILLS))

print("== holm ==")
h = L.holm({"P1": 0.001, "P2": 0.03})
check("holm rejects both at .001/.03", h["P1"][1] and h["P2"][1])
h2 = L.holm({"P1": 0.001, "P2": 0.2})
check("holm step-down keeps P1, drops P2", h2["P1"][1] and not h2["P2"][1])

print("== round-trip law (every shipped structure) ==")
bundle = {"condition": "real", "mode": "full", "stamp": "t",
          "pool_referents": {"uncertainty": {"NU01": {"entropy": np.float64(1.5),
                                                      "margin": np.float32(0.5)}}},
          "train_labels": [{"sid": "NU01", "arm": "uncertainty",
                            "flipped": False, "label": np.int64(7)}],
          "train_log": {"opt_steps": np.int64(272),
                        "losses_every_10": [np.float64(0.5)],
                        "max_example_tokens": np.int64(6144)},
          "took": {"n": 24, "within1": np.int64(20), "rows": []},
          "pre": {"battery_rows": {"uncertainty": runner_rows["uncertainty"]},
                  "arm_errors": {}, "catch_rows": catch_rows(9)},
          "post": {"battery_rows": {"saturation": runner_rows["saturation"]},
                   "catch_rows": [], "paraphrase_rows": []},
          "ppl_pre": np.float64(11.0), "ppl_post": 11.2,
          "arr": np.array([1.0, 2.0])}
td = tempfile.mkdtemp()
pth = os.path.join(td, "b.json")
L.jdump(bundle, pth)
back = json.load(open(pth))
check("bundle json-round-trips with numpy scalars + arrays",
      back["train_log"]["opt_steps"] == 272 and back["arr"] == [1.0, 2.0] and
      back["train_labels"][0]["label"] == 7 and
      back["pool_referents"]["uncertainty"]["NU01"]["entropy"] == 1.5)
L.jdump(ex, os.path.join(td, "ex.json"))
check("training-example list round-trips",
      json.load(open(os.path.join(td, "ex.json"))) == ex)

print("== notebook cells compile (syntax gate for VM-only code) ==")
nb = json.load(open(os.path.join(HERE, "E8N_NATURAL_UI.ipynb")))
ok = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    try:
        compile("".join(c["source"]), f"cell{i}", "exec")
    except SyntaxError as e:
        ok = False
        print(f"  cell {i} SYNTAX ERROR: {e}")
check("all code cells compile", ok)
check("notebook is self-contained (no rclone anywhere, mount present)",
      all("rclone" not in "".join(c["source"]) for c in nb["cells"]) and
      any("drive.mount" in "".join(c["source"]) for c in nb["cells"]))
payload_cell = next("".join(c["source"]) for c in nb["cells"]
                    if "PAYLOAD = json.loads" in "".join(c["source"]))
check("payload cell embeds battery + pools + locked rows",
      "locked_rows" in payload_cell and "e5_battery" not in payload_cell)

print()
n_fail = sum(1 for _, okk in passed if not okk)
print(f"{len(passed) - n_fail}/{len(passed)} checks passed")
if n_fail:
    sys.exit(1)
print("ALL GREEN")
