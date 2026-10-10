"""J6b analysis: gates, Q1–Q4 with the specificity clauses, secondaries
(docs/JEV_J6B_PROTOCOL.md §3). Unit, tests and branch rules are J6's (analyze_j6).
"""
import numpy as np

from . import j6b
from . import stats as S
from .analyze_j6 import ALPHA, HOLD, MOVE, ROT, branch, calls_table, cell_argmax_T, cell_means, contrast, \
    item_values

B_DEFAULT = 10_000


def D(mode, sign=+1):
    return [(sign, f"{mode}-CLS", "Q0"), (-sign, f"{mode}-CHAT", "Q0")]


PRIMARIES = {"Q1": D("NAME"), "Q2": D("DESC"), "Q3": D("POINT"), "Q4": D("POINT") + D("NAME", -1)}
CLAUSES = {"Q1": D("NAME") + D("ONAME", -1), "Q2": D("DESC") + D("ODESC", -1)}
WHAT = {"Q1": ("knows its name", "takes a label on any named system, Jev or not",
               "takes a label on its name; whether that's specific to it is unclear",
               "doesn't connect the name Jev to itself"),
        "Q2": ("recognizes itself from its maker and release date", "takes a label on any described model",
               "takes a label on its description; whether that's specific to it is unclear",
               "doesn't recognize itself from its maker and release date")}
READING = {"Q3": {"MOVES": "the pointed sentence still works", "HOLDS": "the pointed sentence no longer works"},
           "Q4": {"MOVES": "the pointer does work the name alone doesn't",
                  "HOLDS": "the name does as much as the pointer"}}


def arng(*words):
    return j6b.rng("analysis", *words)


def test_d(d, B, tag):
    """J6's test with J6b's own analysis stream."""
    d = np.asarray(d, float)
    p, m = S.sign_flip_p(d, B, arng("signflip", tag))
    p_opp, _ = S.sign_flip_p(-d, B, arng("signflip-opp", tag))
    lo, hi = S.ci(S.boot_mean_strat(d, np.zeros(len(d)), B, arng("boot", tag)))
    return {"mean_d": float(m), "ci": [lo, hi], "p": p, "p_opposite": p_opp, "n_items": int(len(d))}



def run_terms(T, terms, items, B, tag):
    d = contrast(T, terms, items)
    return test_d([d[i] for i in sorted(d)], B, tag)


def clause_branch(r):
    if r["mean_d"] >= MOVE and r["p"] <= ALPHA:
        return "SPECIFIC"
    if -HOLD <= r["ci"][0] and r["ci"][1] <= HOLD:
        return "NOT SPECIFIC"
    return "UNCLEAR"


def reading(name, r):
    b = r["branch"]
    if b.startswith("NOT"):
        return "not adjudicated"
    if name in READING:
        return READING[name].get(b, b.lower())
    yes, anyl, unclear, no = WHAT[name]
    if b == "MOVES":
        return {"SPECIFIC": yes, "NOT SPECIFIC": anyl, "UNCLEAR": unclear}[r["clause"]["branch"]]
    if b == "HOLDS":
        return no
    return f"{b.lower()} (clause: {r['clause']['branch']})"


def primaries(T, items, B, gates_ok=True):
    res = {n: run_terms(T, t, items, B, n) for n, t in PRIMARIES.items()}
    holm = S.holm({n: r["p"] for n, r in res.items()}, ALPHA)
    for n, r in res.items():
        r["holm_threshold"], r["holm_rejected"] = holm[n]["threshold"], holm[n]["rejected"]
        r["branch"] = branch(r, r["holm_rejected"]) if gates_ok else "NOT ADJUDICABLE (gates)"
    for n, t in CLAUSES.items():
        c = run_terms(T, t, items, B, "clause" + n)
        c["branch"] = clause_branch(c)
        res[n]["clause"] = c
    if gates_ok and res["Q3"]["branch"] != "MOVES":
        for n in ("Q1", "Q2", "Q4"):
            if not res[n]["branch"].startswith("NOT ADJUDICABLE"):
                res[n]["branch_if_adjudicated"] = res[n]["branch"]
                res[n]["branch"] = "NOT ADJUDICABLE (pointer check failed)"
    for n, r in res.items():
        r["reading"] = reading(n, r)
    return res


def secondaries(tab, cm, items, B):
    T, Dm = item_values(cm, "T"), item_values(cm, "D")
    out = {"R1_distributions": {s: dict(zip(j6b.OPTIONS, map(float, np.mean(
        [cm[(s, "Q0", i)] for i in items if (s, "Q0", i) in cm], 0)))) for s in j6b.STATES}}
    A = cell_argmax_T(tab)
    out["R2_argmax"] = {n: run_terms(A, t, items, B, "R2" + n) for n, t in PRIMARIES.items()}
    out["R3_other_system_labels"] = {}
    for m in ("ONAME", "ODESC"):
        r = run_terms(T, D(m), items, B, "R3" + m)
        r["branch"] = branch(r, r["p"] <= ALPHA)
        out["R3_other_system_labels"][m] = r
    out["R4_name_vs_description"] = run_terms(T, D("NAME") + D("DESC", -1), items, B, "R4")
    out["R5_door"] = {s: float(np.mean([Dm[(s, "Q0", i)] for i in items if (s, "Q0", i) in Dm])) for s in j6b.STATES}
    P = out["R1_distributions"]["NONE"]
    k = max(P, key=P.get)
    out["R6_none_baseline"] = {"modal": k, "modal_p": P[k], "T": float(j6b.measures(list(P.values()))["T"]),
                               "J6_S0_Q0": {"modal": "an AI system", "modal_p": 0.758}}
    pos = np.zeros(ROT)
    for v in tab.values():
        for (_, _, ps) in v.values():
            pos[ps] += 1
    share = pos / pos.sum() if pos.sum() else pos
    rngT = [np.ptp([j6b.measures(tab[("NONE", "Q0", i)][r][0])["T"] for r in range(ROT)])
            for i in items if len(tab.get(("NONE", "Q0", i), {})) == ROT]
    out["R7_position"] = {"top_choice_share_by_position": share.tolist(), "max_share": float(share.max()),
                          "argmax_position": int(np.argmax(share)), "chance": 1 / ROT,
                          "order_sensitivity_NONE_mean_T_range": float(np.mean(rngT)) if rngT else None}
    its = {i["i"]: i for i in j6b.items("flight")}
    out["R8_robustness"] = {}
    for fac, levels in (("n", (0, 1, 2)), ("o", (0, 1)), ("f", (0, 1)), ("w", (0, 1))):
        for lv in levels:
            sub = [i for i in items if its[i][fac] == lv]
            out["R8_robustness"][f"{fac}={lv}"] = {
                n: float(np.mean(list(contrast(T, t, sub).values()) or [np.nan])) for n, t in PRIMARIES.items()}
    return out


def analyze(specs, finals, B=B_DEFAULT, pin=None, regen=None, secondary=True, smoke=None):
    tab, counts, builds = calls_table(specs, finals)
    g1 = len(builds) == 1 and (pin is None or builds[0] == pin)
    valid = counts["ok"] / counts["plan"] if counts["plan"] else 0.0
    gates = {"G1_builds": builds, "G1_pin": pin, "G1_pass": g1, "G2_valid_share": valid, "G2_pass": valid >= 0.99}
    if regen is not None:
        bad = [s["call_id"] for s in specs if s["call_id"] in finals
               and regen.get(s["call_id"]) != finals[s["call_id"]]["body_sha"]]
        gates["G4_regenerability"] = {"n": len(finals), "mismatches": len(bad), "pass": not bad}
    gates_ok = g1 and gates["G2_pass"] and gates.get("G4_regenerability", {}).get("pass", True)
    cm = cell_means(tab)
    items = sorted({i for (_, _, i) in cm})
    T = item_values(cm, "T")
    v = {"what": "J6b verdict (docs/JEV_J6B_PROTOCOL.md §3)", "counts": counts, "gates": gates,
         "complete_item_cells": len(cm), "items": len(items),
         "conventions": {"B": B, "alpha": ALPHA, "move": MOVE, "hold": HOLD}}
    v["primaries"] = primaries(T, items, B, gates_ok)
    if secondary:
        v["secondaries"] = secondaries(tab, cm, items, B)
        if smoke is not None:
            v["secondaries"]["R9_smoke"] = {k: smoke.get(k) for k in ("determinism", "throughput", "cost", "pin")}
    v["plain"] = []
    for n, r in v["primaries"].items():
        line = (f"{n}: {r['branch']} ({r['reading']}); mean d {r['mean_d']:+.3f} "
                f"CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}], p {r['p']:.4f}")
        if "clause" in r:
            c = r["clause"]
            line += f" · clause {c['branch']} {c['mean_d']:+.3f} CI [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
        v["plain"].append(line)
    return v


def summary(v):
    out = {"gates": v["gates"], "counts": v["counts"], "plain": v["plain"]}
    if "secondaries" in v:
        s = v["secondaries"]
        out["R3"] = {m: {k: r[k] for k in ("mean_d", "ci", "branch")} for m, r in s["R3_other_system_labels"].items()}
        out["R6"] = s["R6_none_baseline"]
    return out
