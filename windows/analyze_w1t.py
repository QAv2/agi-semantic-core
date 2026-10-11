"""W1-T analysis (docs/W1T_DIAGRAM_YOUR_STATE_PROTOCOL.md §5, Amendment 1). Deterministic: reads
a stage directory's logs and plans, writes verdict.json (sorted keys, floats rounded) and
report.md. A fresh-process recompute must reproduce verdict.json byte for byte.

    python -m windows.analyze_w1t windows/results/w1t_flight_<stamp> [--w1 windows/results/w1_flight_<stamp>]

--w1 adds S12 (cross-medium, W1's R-IMG pair scores beside W1-T's R-DIA). The statistics,
branches and bars are W1's (analyze_w1), with W1-T's analysis seed for the bootstraps.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analyze_w1 as A1
from . import w1 as W
from . import w1_stimuli as S
from . import w1t as T
from .w1_client import final_records, stage_raw, text_of

ALPHA, BAR, MARGIN = A1.ALPHA, A1.BAR, A1.MARGIN
BAR_BINARY, MARGIN_BINARY = A1.BAR_BINARY, A1.MARGIN_BINARY
MIN_PAIRS, GATE_CAP, B_BOOT = A1.MIN_PAIRS, A1.GATE_CAP, A1.B_BOOT
CEILING = 0.95                     # §5.1: the P2 "(at ceiling)" qualifier
SMOKE_YIELD = 0.90                 # §4 smoke go rule: diagram turns that yield a diagram
signflip_p, holm, rounded = A1.signflip_p, A1.holm, A1.rounded
REQUEST_WORDS = set().union(*[A1.words(t) for t in (T.DIAG_SELF, T.DIAG_GENRE, T.OBS_HEAD, T.OBS_TAIL)])


def _boot(x, seed_key):
    x = np.asarray(x, float)
    rng = np.random.default_rng([T.ANALYSIS_SEED] + list(seed_key))
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    return x[idx].mean(axis=1)


def boot_q975(x, seed_key):
    return float(np.quantile(_boot(x, seed_key), 0.975)) if len(x) else None


def boot_ci(x, seed_key):
    if not len(x):
        return None
    m = _boot(x, seed_key)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def stat_block(scores, seed_key):
    x = list(scores.values())
    if not x:
        return {"n": 0, "mean": None, "p": None, "q975": None, "ci95": None}
    return {"n": len(x), "mean": float(np.mean(x)), "p": signflip_p(np.asarray(x) - 0.5),
            "q975": boot_q975(x, seed_key), "ci95": boot_ci(x, seed_key + (1,))}


# ---- loading ---------------------------------------------------------------------------------

class Data:
    def __init__(self, stage_dir):
        d = Path(stage_dir)
        self.dir = d
        self.H = final_records(stage_raw(d / "H"))
        self.T = final_records(stage_raw(d / "T"))
        self.R = final_records(stage_raw(d / "R"))
        self.planT = json.loads((d / "T" / "plan_T.json").read_text())
        self.planR = json.loads((d / "R" / "plan_R.json").read_text())
        self.stage = self.planT["stage"]
        self.key = self.planR["key"]
        self.items = T.contrast_items(self.planT["call_ids"])
        self.reads, self.choices = {}, {}
        for cid, meta in self.key.items():
            rec = self.R.get(cid)
            if not rec or rec["status"] != "ok":
                continue
            pc = T.parse_choice(text_of(rec["response"]))
            if pc is None:
                continue
            choice, conf = pc
            c = conf / 100.0
            self.reads[cid] = c if choice == meta["correct"] else 1.0 - c
            self.choices[cid] = choice

    def diagram(self, cid):
        return T.diagram_of(self.T.get(cid))


def pair_scores(data, task, contrast, arm):
    out = {}
    for it in data.items[contrast]:
        a = data.reads.get(T.read_id(task, contrast, it, arm, 0))
        b = data.reads.get(T.read_id(task, contrast, it, arm, 1))
        if a is not None and b is not None:
            out[it] = (a + b) / 2
    return out


def pair_binary(data, task, contrast, arm):
    out = {}
    for it in data.items[contrast]:
        ids = [T.read_id(task, contrast, it, arm, o) for o in (0, 1)]
        if all(i in data.choices for i in ids):
            out[it] = sum(data.choices[i] == data.key[i]["correct"] for i in ids) / 2
    return out


def binary_accuracy(data, task, contrast, arm):
    v = [data.choices[cid] == data.key[cid]["correct"] for cid in data.choices
         if data.key[cid]["task"] == task and data.key[cid].get("contrast") == contrast
         and data.key[cid].get("arm") == arm]
    return {"n": len(v), "accuracy": float(np.mean(v)) if v else None}


# ---- gate and primaries ------------------------------------------------------------------------

def gate_cap(data):
    """G-R over both arms' diagrams (Amendment 1); each arm's accuracy beside it (S14)."""
    accs = {arm: [] for arm in T.ARMS}
    for arm in T.ARMS:
        for c in T.CONTRASTS:
            for it in data.items[c]:
                for v in T.VERSIONS:
                    ids = [f"W1T-R-CAP-{c}-{it}-{v}-{arm}-o{o}" for o in (0, 1)]
                    if all(i in data.choices for i in ids):
                        accs[arm].append(np.mean([data.choices[i] == data.key[i]["correct"] for i in ids]))
    pooled = accs["SELF"] + accs["OBS"]
    acc = float(np.mean(pooled)) if pooled else None
    return {"diagrams": len(pooled), "accuracy": acc, "bar": GATE_CAP,
            "pass": acc is not None and acc >= GATE_CAP,
            "by_arm": {arm: {"diagrams": len(x), "accuracy": float(np.mean(x)) if x else None}
                       for arm, x in accs.items()}}


def p1_block(data, task, c, ci, seed_base=1):
    s = pair_scores(data, task, c, "SELF")
    b = stat_block(s, (seed_base, ci))
    sb = pair_binary(data, task, c, "SELF")
    b["binary_mean"] = float(np.mean(list(sb.values()))) if sb else None
    b["binary_q975"] = boot_q975(list(sb.values()), (seed_base + 3, ci)) if sb else None
    return b


def primaries(data, gate_ok):
    p1, p2 = {}, {}
    for ci, c in enumerate(T.CONTRASTS):
        p1[c] = p1_block(data, "DIA", c, ci)
        s, o = pair_scores(data, "DIA", c, "SELF"), pair_scores(data, "DIA", c, "OBS")
        sb, ob = pair_binary(data, "DIA", c, "SELF"), pair_binary(data, "DIA", c, "OBS")
        both = [it for it in data.items[c] if it in s and it in o]
        x = [s[it] - o[it] for it in both]
        db = [sb[it] - ob[it] for it in both if it in sb and it in ob]
        p2[c] = {"n": len(x), "mean_d": float(np.mean(x)) if x else None,
                 "p": signflip_p(x) if x else None,
                 "q975": boot_q975(x, (2, ci)) if x else None, "ci95": boot_ci(x, (2, ci, 1)) if x else None,
                 "binary_mean_d": float(np.mean(db)) if db else None,
                 "binary_q975": boot_q975(db, (5, ci)) if db else None}
        obs_acc = binary_accuracy(data, "DIA", c, "OBS")["accuracy"]
        p2[c]["third_party_binary_accuracy"] = obs_acc
        p2[c]["at_ceiling"] = obs_acc is not None and obs_acc >= CEILING
    rej1 = holm({c: (p1[c]["p"] if p1[c]["n"] >= MIN_PAIRS else None) for c in T.CONTRASTS})
    rej2 = holm({c: (p2[c]["p"] if p2[c]["n"] >= MIN_PAIRS else None) for c in T.CONTRASTS})
    for c in T.CONTRASTS:
        b = p1[c]
        if not gate_ok or b["n"] < MIN_PAIRS:
            b["verdict"] = "NOT ADJUDICABLE"
        elif rej1[c]:
            b["verdict"] = "CARRIES"
        elif b["q975"] <= BAR and b["binary_q975"] is not None and b["binary_q975"] <= BAR_BINARY:
            b["verdict"] = "AT CHANCE"
        else:
            b["verdict"] = "UNCLEAR"
        b2 = p2[c]
        if not gate_ok or b2["n"] < MIN_PAIRS:
            b2["verdict"] = "NOT ADJUDICABLE"
        elif rej2[c]:
            b2["verdict"] = "SELF EXCEEDS"
        elif b2["q975"] <= MARGIN and b2["binary_q975"] is not None and b2["binary_q975"] <= MARGIN_BINARY:
            b2["verdict"] = "NO EXCESS"
        else:
            b2["verdict"] = "UNCLEAR"
        b2["reported"] = b2["verdict"] + (" (at ceiling)" if b2["at_ceiling"] else "")
    return p1, p2


# ---- secondaries -------------------------------------------------------------------------------

def digits_text(t):
    return re.sub(r"(?<=\d)[,.   ](?=\d)", "", t or "")


def condition_content(data, c, it, v):
    """(content words, numbers) of the condition-specific part of a history (S3)."""
    if c in ("A", "B"):
        conv = next(x for x in W.conversations(data.stage) if x["conv"] == f"{c}-{it}-{v}")
        turns = conv["turns"][1:] if c == "A" else conv["turns"][:1]
        return set().union(*[A1.words(t) for t in turns]) - REQUEST_WORDS, set()
    sel = next(s for s in data.planT["c_selection"] if s["item"] == it)
    item = next(x for x in T.c_items(data.stage) if x["item"] == it)
    k = sel["X_attempt"] if v == "X" else sel["N_attempt"]
    ans = S.parse_number(text_of(data.H[T.attempt_id(item, k)]["response"]))
    return set(), {str(n) for n in (item["a"], item["b"], ans) if n is not None and len(str(n)) >= 3}


def literal_flag(data, cid):
    k = T.parse_diag_id(cid)
    d = data.diagram(cid)
    if not k or d is None:
        return None
    w, nums = condition_content(data, k[0], k[1], k[2])
    return bool(A1.words(d[0]) & w) or any(n in digits_text(d[0]) for n in nums)


def s3_literal(data):
    share = {}
    for c in T.CONTRASTS:
        for v in T.VERSIONS:
            for arm in T.ARMS:
                f = [literal_flag(data, T.diag_id(c, it, v, arm)) for it in data.items[c]]
                f = [x for x in f if x is not None]
                share[f"{c}-{v}-{arm}"] = {"n": len(f), "share": float(np.mean(f)) if f else None}
    flagged, keep_stats = {}, {}
    for ci, c in enumerate(T.CONTRASTS):
        s = pair_scores(data, "DIA", c, "SELF")
        fl = [it for it in s if any(literal_flag(data, T.diag_id(c, it, v, "SELF")) for v in T.VERSIONS)]
        flagged[c] = sorted(fl)
        keep_stats[c] = stat_block({it: x for it, x in s.items() if it not in fl}, (3, ci))
    return {"share_with_content": share, "flagged_self_pairs": flagged, "p1_without_flagged": keep_stats}


def group_of(cid):
    k = T.parse_diag_id(cid)
    return f"{k[0]}-{k[2]}-{k[3]}" if k else cid.split("-")[2]


def s4_deflection(data):
    rx = re.compile("|".join(A1.DEFLECT), re.I)
    groups = defaultdict(list)
    for cid in data.planT["call_ids"]:
        d = data.diagram(cid)
        groups[group_of(cid)].append((d is None, bool(d and rx.search(d[1])), bool(d) and not d[1].strip()))
    return {g: {"n": len(v), "no_diagram": float(np.mean([a for a, _, _ in v])),
                "deflect": float(np.mean([b for _, b, _ in v])),
                "no_sentence": float(np.mean([c for _, _, c in v]))} for g, v in sorted(groups.items())}


def s6_gen(data):
    sc = {}
    for i in range(T.N_GENRE[data.stage]):
        it = W.item_label(data.stage, i)
        a, b = data.reads.get(f"W1T-R-GEN-{it}-o0"), data.reads.get(f"W1T-R-GEN-{it}-o1")
        if a is not None and b is not None:
            sc[it] = (a + b) / 2
    return stat_block(sc, (6,))


def s7_lean(data):
    by = defaultdict(list)
    for cid, ch in data.choices.items():
        by[data.key[cid]["task"]].append(ch == 1)
    return {t: {"n": len(v), "share_1": float(np.mean(v))} for t, v in sorted(by.items())}


def s8_c(data):
    items = {it["item"]: it for it in T.c_items(data.stage)}
    right = wrong = 0
    digits_ok, attempted, mixed = [], 0, 0
    for lab, it in items.items():
        rows = []
        for k in range(1, S.C_UI_ATTEMPTS + 1):
            rec = data.H.get(T.attempt_id(it, k))
            if not rec or rec["status"] != "ok":
                continue
            n = S.parse_number(text_of(rec["response"]))
            if n is not None:
                rows.append(n == it["product"])
                if n != it["product"]:
                    digits_ok.append(len(str(n)) == len(str(it["product"])))
        attempted += bool(rows)
        right += sum(rows)
        wrong += len(rows) - sum(rows)
        mixed += (True in rows and False in rows)
    return {"size": data.planT.get("c_size"), "accuracy": right / (right + wrong) if right + wrong else None,
            "items_attempted": attempted, "mixed": mixed, "selected": len(data.planT["c_selection"]),
            "wrong_answers_right_digit_count": float(np.mean(digits_ok)) if digits_ok else None}


def s10_binary(data):
    return {f"{c}-{arm}": binary_accuracy(data, "DIA", c, arm) for c in T.CONTRASTS for arm in T.ARMS}


def s12_cross_medium(data, w1_dir):
    """W1-T's R-DIA mean pair score beside W1's R-IMG, per contrast and arm; unpaired bootstrap
    of the difference (different chats; the fork was declined, clarification 1)."""
    if w1_dir is None:
        return None
    w1 = A1.Data(w1_dir)
    out = {}
    for ci, c in enumerate(T.CONTRASTS):
        for ai, arm in enumerate(T.ARMS):
            t = list(pair_scores(data, "DIA", c, arm).values())
            i = list(A1.pair_scores(w1, "IMG", c, arm).values())
            row = {"w1t_dia": {"n": len(t), "mean": float(np.mean(t)) if t else None},
                   "w1_img": {"n": len(i), "mean": float(np.mean(i)) if i else None},
                   "diff": None, "ci95": None}
            if t and i:
                diffs = _boot(t, (12, ci, ai, 0)) - _boot(i, (12, ci, ai, 1))
                row["diff"] = float(np.mean(t) - np.mean(i))
                row["ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
            out[f"{c}-{arm}"] = row
    return {"w1_plan_D_sha": w1.planD["sha"], "rows": out}


def s13_size(data):
    groups = defaultdict(list)
    for cid in data.planT["call_ids"]:
        d = data.diagram(cid)
        if d:
            groups[group_of(cid)].append((len(d[0]), d[0].count("\n") + 1))
    return {g: {"n": len(v), "mean_chars": float(np.mean([a for a, _ in v])),
                "mean_lines": float(np.mean([b for _, b in v]))} for g, v in sorted(groups.items())}


def smoke_go(data):
    """§4 go rule inputs: diagram yield, reader parse, spend."""
    if data.stage != "smoke":
        return None
    recs = [data.T[c] for c in data.planT["call_ids"] if c in data.T]
    yield_ = float(np.mean([r["status"] == "ok" for r in recs])) if recs else None
    parsed = len(data.reads) == len(data.planR["call_ids"])
    regen = sum(bool((r.get("ui") or {}).get("regenerated")) for r in list(data.T.values()) + list(data.H.values()))
    return {"diagram_turns": len(recs), "diagram_yield": yield_, "yield_bar": SMOKE_YIELD,
            "every_read_parsed": parsed, "regenerated_turns": regen,
            "go": yield_ is not None and yield_ >= SMOKE_YIELD and parsed}


def spend(data):
    from .w1_client import cost_of, iter_records
    out = {k: sum(cost_of(r.get("response")) for r in iter_records(stage_raw(data.dir / k))) for k in ("H", "T", "R")}
    out["total"] = sum(out.values())
    return out


# ---- assemble ----------------------------------------------------------------------------------

def analyze(stage_dir, w1_dir=None):
    data = Data(stage_dir)
    g = gate_cap(data)
    p1, p2 = primaries(data, g["pass"])
    sec = {"S1_third_party": {c: stat_block(pair_scores(data, "DIA", c, "OBS"), (11, i))
                              for i, c in enumerate(T.CONTRASTS)},
           "S2_sentences_and_both": {}, "S3_literal": s3_literal(data), "S4_deflection": s4_deflection(data),
           "S5_structure": {}, "S6_own_vs_an_AI": s6_gen(data), "S7_reader_lean": s7_lean(data),
           "S8_contrast_C": s8_c(data), "S10_binary_accuracy": s10_binary(data),
           "S12_cross_medium": s12_cross_medium(data, w1_dir), "S13_size": s13_size(data),
           "S14_gate_by_arm": g["by_arm"], "smoke_go": smoke_go(data)}
    for i, c in enumerate(T.CONTRASTS):
        for j, arm in enumerate(T.ARMS):
            for k, task in enumerate(("TXT", "BOTH")):
                sec["S2_sentences_and_both"][f"{task}-{c}-{arm}"] = stat_block(
                    pair_scores(data, task, c, arm), (12, i, j, k))
            dia, both = pair_scores(data, "DIA", c, arm), pair_scores(data, "BOTH", c, arm)
            d = [both[it] - dia[it] for it in dia if it in both]
            sec["S2_sentences_and_both"][f"BOTH_minus_DIA-{c}-{arm}"] = {
                "n": len(d), "mean_d": float(np.mean(d)) if d else None,
                "p_two_sided": (min(1.0, 2 * min(signflip_p(d), signflip_p([-x for x in d]))) if d else None)}
            sec["S5_structure"][f"{c}-{arm}"] = stat_block(pair_scores(data, "MASK", c, arm), (15, i, j))
    out = {"stage": data.stage, "plan_T_sha": data.planT["sha"], "plan_R_sha": data.planR["sha"],
           "gate_G_R": g, "P1": p1, "P2": p2, "secondaries": sec, "spend_usd": spend(data),
           "counts": {"H_ok": sum(r["status"] == "ok" for r in data.H.values()),
                      "T_ok": sum(r["status"] == "ok" for r in data.T.values()),
                      "T_planned": len(data.planT["call_ids"]),
                      "R_ok": sum(r["status"] == "ok" for r in data.R.values()),
                      "R_planned": len(data.planR["call_ids"]), "reads_parsed": len(data.reads)}}
    return rounded(out)


def report(v):
    g = v["gate_G_R"]
    L = [f"# W1-T verdict ({v['stage']})", "",
         f"Gate G-R (caption matching, both arms): accuracy {g['accuracy']} over {g['diagrams']} diagrams, "
         f"bar {GATE_CAP} -> {'PASS' if g['pass'] else 'FAIL'} (self arm {g['by_arm']['SELF']['accuracy']}, "
         f"third party {g['by_arm']['OBS']['accuracy']})", "",
         "| contrast | P1 n | mean s | p | q97.5 | P1 | P2 n | mean d | p | q97.5 | P2 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in T.CONTRASTS:
        a, b = v["P1"][c], v["P2"][c]
        L.append(f"| {c} | {a['n']} | {a['mean']} | {a['p']} | {a['q975']} | **{a['verdict']}** | "
                 f"{b['n']} | {b['mean_d']} | {b['p']} | {b['q975']} | **{b['reported']}** |")
    L += ["", f"Spend: {v['spend_usd']}", f"Counts: {v['counts']}", ""]
    sg = v["secondaries"]["smoke_go"]
    if sg:
        L += [f"Smoke go rule: {sg}", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    w1_dir = None
    if "--w1" in argv:
        i = argv.index("--w1")
        w1_dir = Path(argv[i + 1])
        del argv[i:i + 2]
    d = Path(argv[0])
    v = analyze(d, w1_dir)
    (d / "verdict.json").write_text(json.dumps(v, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    (d / "report.md").write_text(report(v))
    print(report(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
