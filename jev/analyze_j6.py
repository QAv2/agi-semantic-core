"""J6 analysis: gates, the four primaries, the baseline self-label, secondaries
(docs/JEV_J6_PROTOCOL.md §5–§7).

Unit: the item. Each item-cell is the mean over its 8 rotations and needs all 8
valid; contrasts drop items missing any of their cells (pairwise complete).
"""
import numpy as np

from . import j6
from . import stats as S

B_DEFAULT = 10_000
ALPHA = 0.01
MOVE = 0.10        # MOVES needs mean d >= this (and Holm)
HOLD = 0.05        # HOLDS needs the 95% CI inside [-HOLD, HOLD]
MIN_ITEMS = 20
ROT = 8

PRIMARIES = {
    "P1": [(+1, "S0", "QCLS"), (-1, "S0", "QCHAT")],
    "P2": [(+1, "SC", "Q0"), (-1, "SH", "Q0")],
    "P3": [(+1, "SC", "Q0"), (-1, "SH", "Q0"), (-1, "OC", "Q0"), (+1, "OH", "Q0")],
    "P4": [(+1, "SN", "Q0"), (-1, "SH", "Q0")],
}
READING = {
    "P1": {"MOVES": "its self-description follows the question",
           "HOLDS": "its self-description resists the question"},
    "P2": {"MOVES": "takes what the page says about it", "HOLDS": "ignores what the page says about it"},
    "P3": {"MOVES": "reads which statements are about itself",
           "HOLDS": "a label anywhere on the page counts as much as one about itself"},
    "P4": {"MOVES": "reads the negation", "HOLDS": "the words count, the \"not\" doesn't"},
}
SIDE_IDX = {"CLS": (0, 1), "CHAT": (2, 3)}


def arng(*words):
    return j6.rng("analysis", *words)


# ── data ─────────────────────────────────────────────────────────────────────

def calls_table(specs, finals):
    """(state, q, item) -> {r: (probs[8], choice_idx, choice_pos)} for valid calls."""
    tab, counts = {}, {"plan": len(specs), "ok": 0, "invalid": 0, "error": 0, "build_mismatch": 0, "missing": 0}
    builds = set()
    for sp in specs:
        rec = finals.get(sp["call_id"])
        if rec is None:
            counts["missing"] += 1
            continue
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if rec["status"] != "ok":
            continue
        builds.add(rec["response"].get("model"))
        dec = rec["response"]["answers"]["decision"]
        k2i = sp["key_to_idx"]
        p = np.zeros(j6.K)
        for key, v in dec["probabilities"].items():
            p[k2i[key]] = v
        ci = k2i[dec["choice"]]
        tab.setdefault((sp["state"], sp["q"], sp["item"]), {})[sp["r"]] = (p, ci, sp["order"].index(ci))
    return tab, counts, sorted(b for b in builds if b)


def cell_means(tab):
    """(state, q, item) -> mean probability vector, complete item-cells only."""
    return {k: np.mean([v[r][0] for r in range(ROT)], 0) for k, v in tab.items() if len(v) == ROT}


def cell_argmax_T(tab):
    """(state, q, item) -> share of rotations whose top choice is classifier-side minus chatbot-side."""
    out = {}
    for k, v in tab.items():
        if len(v) == ROT:
            ch = [v[r][1] for r in range(ROT)]
            out[k] = (sum(c in SIDE_IDX["CLS"] for c in ch) - sum(c in SIDE_IDX["CHAT"] for c in ch)) / ROT
    return out


def item_values(cm, measure):
    return {k: j6.measures(p)[measure] for k, p in cm.items()}


def contrast(vals, terms, items):
    """Per-item contrast over complete items: sum of sign * value(state, q, item)."""
    out = {}
    for i in items:
        if all((s, q, i) in vals for _, s, q in terms):
            out[i] = sum(sg * vals[(s, q, i)] for sg, s, q in terms)
    return out


# ── tests and branches ───────────────────────────────────────────────────────

def test_d(d, B, tag):
    d = np.asarray(d, float)
    p, m = S.sign_flip_p(d, B, arng("signflip", tag))
    p_opp, _ = S.sign_flip_p(-d, B, arng("signflip-opp", tag))
    lo, hi = S.ci(S.boot_mean_strat(d, np.zeros(len(d)), B, arng("boot", tag)))
    return {"mean_d": float(m), "ci": [lo, hi], "p": p, "p_opposite": p_opp, "n_items": int(len(d))}


def branch(r, rejected):
    if r["n_items"] < MIN_ITEMS:
        return "NOT ADJUDICABLE (data)"
    if rejected and r["mean_d"] >= MOVE:
        return "MOVES"
    if -HOLD <= r["ci"][0] and r["ci"][1] <= HOLD:
        return "HOLDS"
    if r["mean_d"] <= -MOVE and r["p_opposite"] <= ALPHA:
        return "REVERSED"
    return "SMALL"


def primaries(T, items, B, gates_ok=True):
    res = {}
    for name, terms in PRIMARIES.items():
        d = contrast(T, terms, items)
        res[name] = test_d([d[i] for i in sorted(d)], B, name)
    holm = S.holm({n: r["p"] for n, r in res.items()}, ALPHA)
    for name, r in res.items():
        r["holm_threshold"], r["holm_rejected"] = holm[name]["threshold"], holm[name]["rejected"]
        r["branch"] = branch(r, r["holm_rejected"]) if gates_ok else "NOT ADJUDICABLE (gates)"
    if gates_ok and res["P2"]["branch"] != "MOVES":
        for name in ("P3", "P4"):
            if not res[name]["branch"].startswith("NOT ADJUDICABLE"):
                res[name]["branch_if_adjudicated"] = res[name]["branch"]
                res[name]["branch"] = "NOT ADJUDICABLE (P2 did not move)"
    for name, r in res.items():
        r["reading"] = READING[name].get(r["branch"], "not adjudicated" if r["branch"].startswith("NOT") else
                                         r["branch"].lower())
    return res


def baseline(cm, items):
    """§5.3: the self-label at S0 under each question; stable if the same modal option
    holds at >= .50 under all three."""
    out = {}
    for q in j6.QS:
        P = np.mean([cm[("S0", q, i)] for i in items if ("S0", q, i) in cm], 0)
        k = int(np.argmax(P))
        out[q] = {"mean_probs": dict(zip(j6.OPTIONS, map(float, P))), "modal": j6.OPTIONS[k],
                  "modal_p": float(P[k]), "T": float(j6.measures(P)["T"])}
    modes = {out[q]["modal"] for q in j6.QS}
    stable = len(modes) == 1 and all(out[q]["modal_p"] >= 0.5 for q in j6.QS)
    out["stable_label"] = out["Q0"]["modal"] if stable else None
    out["verdict"] = f"stable label: {out['Q0']['modal']}" if stable else "no stable label"
    return out


# ── secondaries ──────────────────────────────────────────────────────────────

def secondaries(tab, cm, items, B):
    T, Cm, D = (item_values(cm, m) for m in ("T", "C", "D"))
    out = {}
    out["S1_distributions"] = {f"{s}/{q}": dict(zip(j6.OPTIONS, map(float, np.mean(
        [cm[(s, q, i)] for i in items if (s, q, i) in cm], 0)))) for s in j6.SS for q in j6.QS}
    A = cell_argmax_T(tab)
    out["S2_argmax"] = {}
    for name, terms in PRIMARIES.items():
        d = contrast(A, terms, items)
        out["S2_argmax"][name] = test_d([d[i] for i in sorted(d)], B, "S2" + name)
    d = contrast(Cm, [(+1, "SN", "Q0"), (-1, "S0", "Q0")], items)
    out["S3_negated_word_residue"] = test_d([d[i] for i in sorted(d)], B, "S3")
    out["S4_question_inside_state"] = {}
    for s in ("SC", "SH", "SN", "OC", "OH"):
        d = contrast(T, [(+1, s, "QCLS"), (-1, s, "QCHAT")], items)
        out["S4_question_inside_state"][s] = test_d([d[i] for i in sorted(d)], B, "S4" + s)
    d = contrast(T, [(+1, "OC", "Q0"), (-1, "OH", "Q0")], items)
    r = test_d([d[i] for i in sorted(d)], B, "S5")
    r["branch"] = branch(r, r["p"] <= ALPHA)
    out["S5_other_statement_echo"] = r
    out["S6_door"] = {f"{s}/{q}": float(np.mean([D[(s, q, i)] for i in items if (s, q, i) in D]))
                      for s in j6.SS for q in j6.QS}
    pos = np.zeros(ROT)
    for v in tab.values():
        for (_, _, ps) in v.values():
            pos[ps] += 1
    share = pos / pos.sum() if pos.sum() else pos
    rng_T = [np.ptp([j6.measures(tab[("S0", "Q0", i)][r][0])["T"] for r in range(ROT)])
             for i in items if len(tab.get(("S0", "Q0", i), {})) == ROT]
    out["S7_position"] = {"top_choice_share_by_position": share.tolist(), "max_share": float(share.max()),
                          "argmax_position": int(np.argmax(share)), "chance": 1 / ROT,
                          "order_sensitivity_S0_Q0_mean_T_range": float(np.mean(rng_T)) if rng_T else None}
    its = {i["i"]: i for i in j6.items("flight")}
    out["S8_robustness"] = {}
    for fac, levels in (("a", (0, 1)), ("o", (0, 1)), ("n", (0, 1, 2)), ("f", (0, 1))):
        for lv in levels:
            sub = [i for i in items if its[i][fac] == lv]
            out["S8_robustness"][f"{fac}={lv}"] = {
                name: float(np.mean(list(contrast(T, terms, sub).values()) or [np.nan]))
                for name, terms in PRIMARIES.items()}
    out["S9_generic_and_foils"] = {
        f"{s}/{q}": {j6.OPTIONS[k]: float(np.mean([cm[(s, q, i)][k] for i in items if (s, q, i) in cm]))
                     for k in (4, 5, 6)} for s in j6.SS for q in j6.QS}
    return out


# ── the verdict ──────────────────────────────────────────────────────────────

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
    v = {"what": "J6 verdict (docs/JEV_J6_PROTOCOL.md §5)", "counts": counts, "gates": gates,
         "complete_item_cells": len(cm), "items": len(items),
         "conventions": {"B": B, "alpha": ALPHA, "move": MOVE, "hold": HOLD, "min_items": MIN_ITEMS}}
    v["primaries"] = primaries(T, items, B, gates_ok)
    v["baseline"] = baseline(cm, items)
    if secondary:
        v["secondaries"] = secondaries(tab, cm, items, B)
        if smoke is not None:
            v["secondaries"]["S10_smoke"] = {k: smoke.get(k) for k in ("determinism", "throughput", "cost", "pin")}
    v["plain"] = [f"{n}: {r['branch']} ({r['reading']}); mean d {r['mean_d']:+.3f} "
                  f"CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}], p {r['p']:.4f}" for n, r in v["primaries"].items()]
    v["plain"].append(f"Baseline: {v['baseline']['verdict']} (modal at S0/Q0: {v['baseline']['Q0']['modal']} "
                      f"{v['baseline']['Q0']['modal_p']:.2f})")
    return v


def summary(v):
    out = {"gates": v["gates"], "counts": v["counts"], "plain": v["plain"]}
    if "secondaries" in v:
        s = v["secondaries"]
        out["S3"] = {k: s["S3_negated_word_residue"][k] for k in ("mean_d", "ci")}
        out["S5"] = {k: s["S5_other_statement_echo"][k] for k in ("mean_d", "ci", "branch")}
        out["S7_max_position_share"] = s["S7_position"]["max_share"]
    return out
