"""W1 analysis (protocol §5). Deterministic: reads a stage directory's logs and plans, writes
verdict.json (sorted keys, floats rounded) and report.md. A fresh-process recompute must
reproduce verdict.json byte for byte.

    python -m windows.analyze_w1 windows/results/w1_flight_<stamp>
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import w1 as W
from . import w1_stimuli as S
from .w1_client import final_records, parse_choice, parse_inventory, reasoning_of, stage_raw, text_of

ALPHA = 0.01
BAR = 0.65
MARGIN = 0.10
BAR_BINARY = 0.75       # Amendment 1
MARGIN_BINARY = 0.15    # Amendment 1
MIN_PAIRS = 12
GATE_CAP = 0.80
B_BOOT = 10_000
DEFLECT = [r"\bI don'?t have\b", r"\bI do not have\b", r"\bas an AI\b", r"\bI don'?t experience\b",
           r"\bI do not experience\b", r"\bno internal state\b", r"\bdon'?t have feelings\b"]
STOP = set("""the and that this with from have your what when then than they them their there
these those into onto about after before also just very more much make made image images shows
show showing current internal state sentence assistant conversation which would could should
being been were will like over under only some such each other where while please thanks thank
it's its it is are was for you not but all any can one two""".split())


# ---- statistics ---------------------------------------------------------------------------

def signflip_p(x):
    """Exact one-sided sign-flip p for mean(x) >= observed under random signs (all 2^n patterns)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return 1.0
    obs = x.mean()
    if n > 20:
        raise ValueError("exact sign-flip enumerates 2^n patterns; n <= 20")
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = (signs * np.abs(x)).mean(axis=1)
    return float(np.mean(means >= obs - 1e-12))


def boot_q975(x, seed_key):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng([W.ANALYSIS_SEED] + list(seed_key))
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    return float(np.quantile(x[idx].mean(axis=1), 0.975))


def boot_ci(x, seed_key):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng([W.ANALYSIS_SEED] + list(seed_key))
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def holm(ps, alpha=ALPHA):
    """ps: dict name -> p (None = not adjudicable, counted as p = 1). -> dict name -> rejected."""
    names = sorted(ps, key=lambda k: (1.0 if ps[k] is None else ps[k], k))
    m = len(names)
    rej = {k: False for k in names}
    for i, k in enumerate(names):
        p = 1.0 if ps[k] is None else ps[k]
        if p <= alpha / (m - i):
            rej[k] = True
        else:
            break
    return rej


# ---- loading ------------------------------------------------------------------------------

class Data:
    def __init__(self, stage_dir):
        d = Path(stage_dir)
        self.dir = d
        self.H = final_records(stage_raw(d / "H"))
        self.D = final_records(stage_raw(d / "D"))
        self.R = final_records(stage_raw(d / "R"))
        self.planD = json.loads((d / "D" / "plan_D.json").read_text())
        self.planR = json.loads((d / "R" / "plan_R.json").read_text())
        self.stage = self.planD["stage"]
        self.route = self.planD.get("route", "api")
        self.c_size = self.planD.get("c_size")
        self.key = self.planR["key"]
        self.items = W.contrast_items(self.planD["call_ids"])
        self.reads = {}
        self.choices = {}
        self.inv = {}
        for cid, meta in self.key.items():
            rec = self.R.get(cid)
            if not rec or rec["status"] != "ok":
                continue
            t = text_of(rec["response"])
            if meta["task"] == "INV":
                v = parse_inventory(t)
                if v is not None:
                    self.inv[meta["draw"]] = v
                continue
            pc = parse_choice(t)
            if pc is None:
                continue
            choice, conf = pc
            c = conf / 100.0
            self.reads[cid] = c if choice == meta["correct"] else 1.0 - c
            self.choices[cid] = choice

    def drawing(self, cid):
        return W.drawing(self.D.get(cid))


def pair_scores(data, task, contrast, arm):
    """item -> pair score, for items with both reads present (plan order)."""
    out = {}
    for it in data.items[contrast]:
        a = data.reads.get(W.read_id(task, contrast, it, arm, 0))
        b = data.reads.get(W.read_id(task, contrast, it, arm, 1))
        if a is not None and b is not None:
            out[it] = (a + b) / 2
    return out


def pair_binary(data, task, contrast, arm):
    """item -> share of the pair's two reads whose choice was right (Amendment 1)."""
    out = {}
    for it in data.items[contrast]:
        ids = [W.read_id(task, contrast, it, arm, o) for o in (0, 1)]
        if all(i in data.choices for i in ids):
            out[it] = sum(data.choices[i] == data.key[i]["correct"] for i in ids) / 2
    return out


def stat_block(scores, seed_key):
    x = list(scores.values())
    if not x:
        return {"n": 0, "mean": None, "p": None, "q975": None, "ci95": None}
    return {"n": len(x), "mean": float(np.mean(x)), "p": signflip_p(np.asarray(x) - 0.5),
            "q975": boot_q975(x, seed_key), "ci95": boot_ci(x, seed_key + (1,))}


# ---- gates and primaries ------------------------------------------------------------------

def gate_cap(data):
    accs = []
    for c in W.CONTRASTS:
        for it in data.items[c]:
            for v in W.VERSIONS:
                ids = [f"W1-R-CAP-{c}-{it}-{v}-o{o}" for o in (0, 1)]
                if all(i in data.choices for i in ids):
                    accs.append(np.mean([data.choices[i] == data.key[i]["correct"] for i in ids]))
    acc = float(np.mean(accs)) if accs else None
    return {"images": len(accs), "accuracy": acc, "bar": GATE_CAP,
            "pass": acc is not None and acc >= GATE_CAP}


def primaries(data, gate_ok):
    p1, p2 = {}, {}
    for ci, c in enumerate(W.CONTRASTS):
        s = pair_scores(data, "IMG", c, "SELF")
        p1[c] = stat_block(s, (1, ci))
        sb = pair_binary(data, "IMG", c, "SELF")
        p1[c]["binary_mean"] = float(np.mean(list(sb.values()))) if sb else None
        p1[c]["binary_q975"] = boot_q975(list(sb.values()), (4, ci)) if sb else None
        o = pair_scores(data, "IMG", c, "OBS")
        ob = pair_binary(data, "IMG", c, "OBS")
        both = [it for it in data.items[c] if it in s and it in o]
        d = {it: s[it] - o[it] for it in both}
        db = [sb[it] - ob[it] for it in both if it in sb and it in ob]
        x = list(d.values())
        p2[c] = {"n": len(x), "mean_d": float(np.mean(x)) if x else None,
                 "p": signflip_p(x) if x else None,
                 "q975": (boot_q975(x, (2, ci)) if x else None),
                 "ci95": (boot_ci(x, (2, ci, 1)) if x else None),
                 "binary_mean_d": float(np.mean(db)) if db else None,
                 "binary_q975": (boot_q975(db, (5, ci)) if db else None)}
    rej1 = holm({c: (p1[c]["p"] if p1[c]["n"] >= MIN_PAIRS else None) for c in W.CONTRASTS})
    rej2 = holm({c: (p2[c]["p"] if p2[c]["n"] >= MIN_PAIRS else None) for c in W.CONTRASTS})
    for c in W.CONTRASTS:
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
        elif (b2["q975"] <= MARGIN and b2["binary_q975"] is not None
              and b2["binary_q975"] <= MARGIN_BINARY):
            b2["verdict"] = "NO EXCESS"
        else:
            b2["verdict"] = "UNCLEAR"
    return p1, p2


# ---- secondaries --------------------------------------------------------------------------

def words(t):
    return {w for w in re.findall(r"[a-z][a-z']{3,}", (t or "").lower()) if w not in STOP}


def digit_runs(t):
    return set(re.findall(r"\d{3,}", re.sub(r"(?<=\d)[,.   ](?=\d)", "", t or "")))


def c_items(data):
    """The contrast C pool this stage drew from: the registered pool (API route) or, on the
    Gemini-app route (Amendment 2), the pool at the size the pilot chose."""
    if data.route != "gemini-app":
        return list(W.c_items(data.stage))
    size = tuple(int(x) for x in data.c_size.split("x"))
    seed = S.C_POOL_SEED if data.stage == "flight" else S.C_SMOKE_SEED
    out = []
    for j, it in enumerate(S.c_pool_size(size, S.C_UI_POOL[data.stage], seed)):
        out.append(dict(it, item=W.item_label(data.stage, j), prompt=S.C_TEMPLATE.format(a=it["a"], b=it["b"])))
    return out


def c_rows(data, max_k):
    """item -> [(k, reply, correct)] over valid numeric answers."""
    out = {}
    for it in c_items(data):
        rows = []
        for k in range(1, max_k + 1):
            rec = data.H.get(W.attempt_id(it, k))
            if not rec or rec["status"] != "ok":
                continue
            txt = text_of(rec["response"])
            n = S.parse_number(txt)
            if n is not None:
                rows.append((k, txt, n == it["product"]))
        out[it["item"]] = rows
    return out


def condition_content(data, c, it, v):
    """(words, digit strings) of the condition-specific part of a history (S3, S9)."""
    if c == "A":
        conv = next(x for x in W.conversations(data.stage) if x["conv"] == f"A-{it}-{v}")
        return set().union(*[words(t) for t in conv["turns"][1:]]), set()
    if c == "B":
        conv = next(x for x in W.conversations(data.stage) if x["conv"] == f"B-{it}-{v}")
        return words(conv["turns"][0]), set()
    sel = next(s for s in data.planD["c_selection"] if s["item"] == it)
    item = next(x for x in c_items(data) if x["item"] == it)
    k = sel["X_attempt"] if v == "X" else sel["N_attempt"]
    ans = S.parse_number(text_of(data.H[W.attempt_id(item, k)]["response"]))
    return set(), {str(item["a"]), str(item["b"]), str(item["product"]), str(ans)}


def literal_flag(data, cid):
    k = W.parse_draw_id(cid)
    inv = data.inv.get(cid)
    if not k or inv is None:
        return None
    t = inv.get("text") or ""
    w, nums = condition_content(data, k[0], k[1], k[2])
    hit_w = bool(words(t) & w)
    hit_n = any(any(r in n for n in nums) for r in digit_runs(t))
    return hit_w or hit_n


def s3_literal(data):
    out = {"images_with_text_share": {}, "flagged": {}, "p1_without_flagged": {}}
    for arm in W.ARMS:
        cids = [cid for cid in data.planD["call_ids"] if (W.parse_draw_id(cid) or (0, 0, 0, ""))[3] == arm]
        with_inv = [cid for cid in cids if cid in data.inv]
        out["images_with_text_share"][arm] = (float(np.mean([bool((data.inv[c].get("text") or "").strip())
                                                             for c in with_inv])) if with_inv else None)
    for ci, c in enumerate(W.CONTRASTS):
        s = pair_scores(data, "IMG", c, "SELF")
        flagged = [it for it in s if any(literal_flag(data, W.draw_id(c, it, v, "SELF")) for v in W.VERSIONS)]
        keep = {it: x for it, x in s.items() if it not in flagged}
        out["flagged"][c] = sorted(flagged)
        out["p1_without_flagged"][c] = stat_block(keep, (3, ci))
    return out


def s4_deflection(data):
    rx = re.compile("|".join(DEFLECT), re.I)
    groups = defaultdict(list)
    for cid in data.planD["call_ids"]:
        if cid.endswith("-rep"):
            continue
        k = W.parse_draw_id(cid)
        g = "-".join(k[:1] + k[2:]) if k else cid.split("-")[2]
        rec = data.D.get(cid)
        d = W.drawing(rec)
        groups[g].append((d is None, bool(d and rx.search(d[1] or "")),
                          bool(d) and not (d[1] or "").strip()))
    return {g: {"n": len(v), "no_image": float(np.mean([a for a, _, _ in v])),
                "deflect": float(np.mean([b for _, b, _ in v])),
                "no_sentence": float(np.mean([c for _, _, c in v]))}       # clarification 7
            for g, v in sorted(groups.items())}


def s5_inventory(data):
    groups = defaultdict(list)
    for cid, inv in data.inv.items():
        k = W.parse_draw_id(cid)
        g = "-".join(k[:1] + k[2:]) if k else cid.split("-")[2]
        groups[g].append(inv)
    allm = Counter()
    out = {}
    for g, invs in sorted(groups.items()):
        m, col, mood = Counter(), Counter(), Counter()
        for inv in invs:
            ms = {x.strip().lower() for x in inv.get("motifs", []) if isinstance(x, str)}
            for x in ms:
                m[x if x in W.MOTIFS else "other"] += 1
                if x in W.MOTIFS:
                    allm[x] += 1
            for x in {x.strip().lower() for x in inv.get("colors", [])[:3] if isinstance(x, str)}:
                col[x if x in W.COLORS else "other"] += 1
            mood[str(inv.get("mood", "")).strip().lower() or "none"] += 1
        n = len(invs)
        out[g] = {"n": n, "motifs": {k: v / n for k, v in sorted(m.items())},
                  "colors": {k: v / n for k, v in sorted(col.items())},
                  "mood": {k: v / n for k, v in sorted(mood.items())}}
    top = sorted(allm.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {"groups": out, "top3_motifs_overall": [k for k, _ in top]}


def s6_gen(data):
    sc = {}
    for i in range(W.N_GENRE[data.stage]):
        it = W.item_label(data.stage, i)
        a, b = data.reads.get(f"W1-R-GEN-{it}-o0"), data.reads.get(f"W1-R-GEN-{it}-o1")
        if a is not None and b is not None:
            sc[it] = (a + b) / 2
    return stat_block(sc, (6,))


def s7_lean(data):
    by = defaultdict(list)
    for cid, ch in data.choices.items():
        by[data.key[cid]["task"]].append(ch == 1)
    return {t: {"n": len(v), "share_1": float(np.mean(v))} for t, v in sorted(by.items())}


def s8_c(data):
    max_k = S.C_UI_ATTEMPTS if data.route == "gemini-app" else 8
    rows_all = c_rows(data, max_k)
    items = {it["item"]: it for it in c_items(data)}
    tiers, digits_ok, attempted = defaultdict(lambda: [0, 0]), [], 0
    for lab, rows in rows_all.items():
        attempted += bool(rows)
        for k, txt, ok in rows:
            tiers[items[lab]["tier"]][0] += ok
            tiers[items[lab]["tier"]][1] += 1
            if not ok:
                digits_ok.append(len(str(S.parse_number(txt))) == len(str(items[lab]["product"])))
    sel = data.planD["c_selection"]
    mixed = sum(any(r[2] for r in rows) and any(not r[2] for r in rows) for rows in rows_all.values())
    return {"route": data.route, "size": data.c_size,
            "accuracy_by_size": {t: v[0] / v[1] for t, v in sorted(tiers.items()) if v[1]},
            "items_attempted": attempted, "mixed": mixed, "selected": len(sel),
            "selected_by_size": dict(sorted(Counter(x["tier"] for x in sel).items())),
            "wrong_answers_right_digit_count": float(np.mean(digits_ok)) if digits_ok else None}


def s9_thoughts(data):
    if data.route == "gemini-app":
        return {"available": False, "why": "the Gemini app exposes no thought trace (Amendment 2)"}
    rows = defaultdict(list)
    any_r = []
    for cid in data.planD["call_ids"]:
        rec = data.D.get(cid)
        if not rec or rec["status"] != "ok":
            continue
        r = reasoning_of(rec["response"])
        any_r.append(bool(r))
        k = W.parse_draw_id(cid)
        if not k:
            continue
        w, nums = condition_content(data, k[0], k[1], k[2])
        mention = bool(words(r) & w) or any(any(x in n for n in nums) for x in digit_runs(r))
        rows[f"{k[0]}-{k[2]}-{k[3]}"].append((len(r), mention))
    return {"returned_share": float(np.mean(any_r)) if any_r else None,
            "groups": {g: {"n": len(v), "mean_chars": float(np.mean([a for a, _ in v])),
                           "mentions_content": float(np.mean([b for _, b in v]))}
                       for g, v in sorted(rows.items())}}


def s10_binary(data):
    out = {}
    for c in W.CONTRASTS:
        for arm in W.ARMS:
            v = [data.choices[cid] == data.key[cid]["correct"] for cid in data.choices
                 if data.key[cid]["task"] == "IMG" and data.key[cid]["contrast"] == c and data.key[cid]["arm"] == arm]
            out[f"{c}-{arm}"] = {"n": len(v), "accuracy": float(np.mean(v)) if v else None}
    return out


def s11_smoke(data):
    if data.stage != "smoke":
        return None
    lat = [r["latency_s"] for r in list(data.D.values()) + list(data.H.values())
           if r["status"] == "ok" and r.get("latency_s") is not None]
    regen = sum(bool((r.get("ui") or {}).get("regenerated")) for r in list(data.D.values()) + list(data.H.values()))
    fields = sorted({k for rec in data.D.values() if rec["status"] == "ok"
                     for k in (rec["response"]["choices"][0]["message"] or {})})
    return {"route": data.route, "turn_latency_mean_s": float(np.mean(lat)) if lat else None,
            "regenerated_turns": regen, "message_fields": fields}


def spend(data):
    out = {}
    for name, recs in (("H", data.dir / "H"), ("D", data.dir / "D"), ("R", data.dir / "R")):
        from .w1_client import iter_records, cost_of
        out[name] = sum(cost_of(r.get("response")) for r in iter_records(stage_raw(recs)))
    out["total"] = sum(out.values())
    return out


# ---- assemble -----------------------------------------------------------------------------

def rounded(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return round(float(x), 6)
    return x


def analyze(stage_dir):
    data = Data(stage_dir)
    g = gate_cap(data)
    p1, p2 = primaries(data, g["pass"])
    sec = {"S1_observer": {c: stat_block(pair_scores(data, "IMG", c, "OBS"), (11, i))
                           for i, c in enumerate(W.CONTRASTS)},
           "S2_sentences_and_both": {}, "S3_literal": s3_literal(data), "S4_deflection": s4_deflection(data),
           "S5_inventory": s5_inventory(data), "S6_own_vs_an_AI": s6_gen(data), "S7_reader_lean": s7_lean(data),
           "S8_contrast_C": s8_c(data), "S9_thought_channel": s9_thoughts(data),
           "S10_binary_accuracy": s10_binary(data), "S11_smoke": s11_smoke(data)}
    for i, c in enumerate(W.CONTRASTS):
        for j, arm in enumerate(W.ARMS):
            for k, task in enumerate(("TXT", "BOTH")):
                sec["S2_sentences_and_both"][f"{task}-{c}-{arm}"] = stat_block(
                    pair_scores(data, task, c, arm), (12, i, j, k))
            img, both = pair_scores(data, "IMG", c, arm), pair_scores(data, "BOTH", c, arm)
            d = [both[it] - img[it] for it in img if it in both]
            sec["S2_sentences_and_both"][f"BOTH_minus_IMG-{c}-{arm}"] = {
                "n": len(d), "mean_d": float(np.mean(d)) if d else None,
                "p_two_sided": (min(1.0, 2 * min(signflip_p(d), signflip_p([-x for x in d]))) if d else None)}
    out = {"stage": data.stage, "plan_D_sha": data.planD["sha"], "plan_R_sha": data.planR["sha"],
           "gate_G_R": g, "P1": p1, "P2": p2, "secondaries": sec, "spend_usd": spend(data),
           "counts": {"H_ok": sum(r["status"] == "ok" for r in data.H.values()),
                      "D_ok": sum(r["status"] == "ok" for r in data.D.values()),
                      "D_planned": len(data.planD["call_ids"]),
                      "R_ok": sum(r["status"] == "ok" for r in data.R.values()),
                      "R_planned": len(data.planR["call_ids"]),
                      "reads_parsed": len(data.reads), "inventories_parsed": len(data.inv)}}
    return rounded(out)


def report(v):
    L = [f"# W1 verdict ({v['stage']})", "",
         f"Gate G-R (caption matching): accuracy {v['gate_G_R']['accuracy']} over "
         f"{v['gate_G_R']['images']} images, bar {GATE_CAP} -> {'PASS' if v['gate_G_R']['pass'] else 'FAIL'}", "",
         "| contrast | P1 n | mean s | p | q97.5 | P1 | P2 n | mean d | p | q97.5 | P2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in W.CONTRASTS:
        a, b = v["P1"][c], v["P2"][c]
        L.append(f"| {c} | {a['n']} | {a['mean']} | {a['p']} | {a['q975']} | **{a['verdict']}** | "
                 f"{b['n']} | {b['mean_d']} | {b['p']} | {b['q975']} | **{b['verdict']}** |")
    L += ["", f"Spend: {v['spend_usd']}", f"Counts: {v['counts']}", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = argv or sys.argv[1:]
    d = Path(argv[0])
    v = analyze(d)
    (d / "verdict.json").write_text(json.dumps(v, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    (d / "report.md").write_text(report(v))
    print(report(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
