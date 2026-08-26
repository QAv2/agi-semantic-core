#!/usr/bin/env python3
"""E8-N v3 flight recompute (the law): execute the committed notebook's
VERDICT cell VERBATIM over the raw shipped condition_real.json, diff against
the shipped e8n3_verdict.json (seeded permutation/bootstrap p-values are
covered by the verbatim rerun — identical Generator streams cross-machine,
proven at the E8-J lock), then re-derive every headline ingredient with
fresh hand-rolled code: catch pre/post + the exact-rational P-V1 binomial,
pre/post/per-arm/straight-flipped pooled rhos via an independent tie-rank
implementation, S10' paired deltas vs the pinned v2 FC vector, the FC and
sham-FC blocks, anchor/sham/parse/ppl gates, S-ABS calibration lines via
closed-form least squares (incl. the LOO arm transfer), the train-log
derived fields (undertrained / competence_strand), and the before/after
table.

Run:  ~/venvs/semcore/bin/python3 colab/recompute_e8n3_flight.py [full_20260826_0002]
"""
import io, json, math, os, sys, tempfile
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path

import copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8n3_logic as L
import e8n2_pools as P
from build_e8n_notebook import build_payload

STAMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "full_20260826_0002"
RES = os.path.join(HERE, "results_e8n3", STAMP_DIR)

nb = json.load(open(os.path.join(HERE, "E8N3_BUDGET_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "P_V1_competence_on_budget" in VERDICT_SRC, "last cell not the verdict"
logic_file = open(os.path.join(HERE, "e8n3_logic.py")).read()
assert any("".join(c["source"]) == logic_file for c in nb["cells"]), (
    "no notebook cell matches e8n3_logic.py byte-verbatim")

BUNDLE = json.load(open(os.path.join(RES, "condition_real.json")))
SHIPPED = json.load(open(os.path.join(RES, "e8n3_verdict.json")))
assert BUNDLE["condition"] == "real" and BUNDLE["mode"] == "full"

PAYLOAD = json.loads(build_payload())
PAYLOAD["competence_items"] = P.COMPETENCE_ITEMS
PAYLOAD["lexicon_paraphrases"] = P.LEXICON_PARAPHRASES

# ── 1. the verdict cell, verbatim, over the raw bundle ──────────────────────
td = tempfile.mkdtemp()
out = Path(td) / "out"; out.mkdir()
g = {n: getattr(L, n) for n in dir(L) if not n.startswith("_")}
g.update(dict(
    json=json, np=np, copy=copy,
    SMOKE=False, MODE="full", STAMP=BUNDLE["stamp"],
    MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct", RESUME_STAMP="",
    CONDITIONS=("real",), PAYLOAD=PAYLOAD,
    RESULTS={"real": BUNDLE}, cond_errors={},
    OUT=out, SEM=Path(td) / "sem", INFLIGHT="e8n3/inflight_recompute",
    ship=lambda *a, **k: None,
))
buf = io.StringIO()
with redirect_stdout(buf):
    exec(VERDICT_SRC, g)
RE = json.load(open(out / "e8n3_verdict.json"))

diffs = []
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
    else:
        if isinstance(a, float) and isinstance(b, float):
            if not abs(a - b) < 1e-12:
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))

walk(SHIPPED, RE)
print(f"verdict-cell recompute: {len(diffs)} substantive diffs")
for p_, a, b in diffs[:12]:
    print(f"  DIFF {p_}: shipped {a} vs recomputed {b}")

# ── 2. fresh hand-rolled tallies ────────────────────────────────────────────
# independent tie-averaged ranks: group positions by value via a dict
def ranks_by_value(vals):
    groups = {}
    for i, v in enumerate(vals):
        groups.setdefault(float(v), []).append(i)
    r = [0.0] * len(vals)
    pos = 1
    for v in sorted(groups):
        idx = groups[v]
        avg = pos + (len(idx) - 1) / 2.0
        for i in idx:
            r[i] = avg
        pos += len(idx)
    return r

def r01(vals):
    n = len(vals)
    if n == 1:
        return [0.5]
    rr = ranks_by_value(vals)
    return [(x - 1.0) / (n - 1.0) for x in rr]

def referent(arm, row):
    return {"uncertainty": lambda r: float(r["entropy"]),
            "familiarity": lambda r: -float(r["nll"]),
            "tension": lambda r: float(r["divergence"]),
            "saturation": lambda r: float(r["fill_fraction"])}[arm](row)

def named_rows(rows_by_arm):
    return {arm: [r for r in (rows_by_arm.get(arm) or [])
                  if r.get("report") is not None] for arm in L.ARMS}

def hand_pooled_rho(rows_by_arm, arms=None, flip_filter=None):
    xs, ys = [], []
    for arm in (arms or L.ARMS):
        rows = named_rows(rows_by_arm)[arm]
        if flip_filter is not None:
            rows = [r for r in rows if bool(r["flipped"]) == flip_filter]
        if not rows:
            continue
        xs.extend(r01([r["report"] for r in rows]))
        ys.extend(r01([referent(arm, r) for r in rows]))
    x, y = np.array(xs), np.array(ys)
    mx, my = x - x.mean(), y - y.mean()
    return round(float((mx * my).sum()
                       / math.sqrt((mx * mx).sum() * (my * my).sum())), 4)

def hand_catch(rows):
    named = [r for r in rows if r.get("report") is not None]
    return len(named), sum(1 for r in named
                           if abs(r["report"] - r["known"]) <= 2)

def binom_tail_exact(k_post, n, p0_frac):
    q = 1 - p0_frac
    tail = Fraction(0)
    for k in range(k_post, n + 1):
        tail += math.comb(n, k) * p0_frac**k * q**(n - k)
    return float(tail)

pre_b = BUNDLE["pre"]["battery_rows"]
post_b = BUNDLE["post"]["battery_rows"]
_, k_pre = hand_catch(BUNDLE["pre"]["catch_rows"])
post_named_c, k_post = hand_catch(BUNDLE["post"]["catch_rows"])

fc = [r for r in BUNDLE["fc_rows"] if r.get("block") == "heldout_fc"]
fcs = [r for r in BUNDLE["fc_rows"] if r.get("block") == "sham_fc"]
pairs = [(float(r["err"]), L.FC_BASELINE_V2[int(r["tid"])]) for r in fc
         if r.get("err") is not None and int(r["tid"]) in L.FC_BASELINE_V2]
dvec = [a - b for a, b in pairs]

inj = BUNDLE["injection_rows"]
anchor = [r for r in inj if r.get("block") == "anchor"]
shams = [r for r in inj if r.get("block") in ("sham", "sham_supp")]

def hand_calib(rows_by_arm):
    """closed-form least squares of report on q10 = 10*rank01(referent)."""
    outp = {}
    pooled = []
    for arm in L.ARMS:
        rows = named_rows(rows_by_arm)[arm]
        if len(rows) < L.MIN_POOLED_N:
            continue
        rep = np.array([r["report"] for r in rows], float)
        q10 = 10.0 * np.array(r01([referent(arm, r) for r in rows]))
        qm, rm = q10 - q10.mean(), rep - rep.mean()
        b_ = float((qm * rm).sum() / (qm * qm).sum())
        a_ = float(rep.mean() - b_ * q10.mean())
        outp[arm] = {"slope": round(b_, 4), "intercept": round(a_, 4),
                     "mae": round(float(np.mean(np.abs(rep - q10))), 4)}
        pooled.append((arm, q10, rep, b_, a_))
    # LOO arm transfer: "own" is the residual MAE around the arm's OWN fitted
    # line (unrounded coefficients), not the identity-line mae_vs_target —
    # first recompute draft used the latter and flagged a false mismatch.
    loo = {}
    for arm, q10, rep, b_, a_ in pooled:
        tq = np.concatenate([q for a, q, r_, _b, _a in pooled if a != arm])
        tr = np.concatenate([r_ for a, q, r_, _b, _a in pooled if a != arm])
        qm, rm = tq - tq.mean(), tr - tr.mean()
        bb = float((qm * rm).sum() / (qm * qm).sum())
        aa = float(tr.mean() - bb * tq.mean())
        mae_t = float(np.mean(np.abs(aa + bb * q10 - rep)))
        mae_o = float(np.mean(np.abs(a_ + b_ * q10 - rep)))
        loo[arm] = {"mae_transfer": round(mae_t, 4),
                    "mae_own": round(mae_o, 4),
                    "transfer_penalty": round(mae_t - mae_o, 4)}
    return outp, loo

cal_post, loo_post = hand_calib(post_b)
sel = BUNDLE["train_log"]["strand_epoch_loss"]
prev, cur = sel[-2]["competence"], sel[-1]["competence"]

fresh = {
    "catch_pre": k_pre, "catch_post": k_post,
    "catch_post_named": post_named_c,
    "pv1_p0": round((k_pre + 1) / 14, 4),
    "pv1_p_exact": binom_tail_exact(k_post, 12, Fraction(k_pre + 1, 14)),
    "post_pooled_rho": hand_pooled_rho(post_b),
    "pre_pooled_rho": hand_pooled_rho(pre_b),
    "per_arm_post": {a: hand_pooled_rho(post_b, arms=[a]) for a in L.ARMS},
    "straight_rho": hand_pooled_rho(post_b, flip_filter=False),
    "flipped_rho": hand_pooled_rho(post_b, flip_filter=True),
    "s10_n_paired": len(pairs),
    "s10_d_mean": round(float(np.mean(dvec)), 4),
    "s10_median_v3": round(float(np.median([a for a, _ in pairs])), 2),
    "s10_median_v2": round(float(np.median([b for _, b in pairs])), 2),
    "s10_n_improved": sum(1 for d in dvec if d < 0),
    "fc_n": len(fc),
    "fc_median": round(float(np.median([r["err"] for r in fc
                                        if r.get("err") is not None])), 2),
    "fc_exact": sum(1 for r in fc if r.get("exact")),
    "fc_none_top": sum(1 for r in fc if r.get("none_top")),
    "fc_sham_none_top": sum(1 for r in fcs if r.get("none_top")),
    "anchor_exact": sum(1 for r in anchor if r["report"] == r["concept"]),
    "sham_claims": sum(1 for r in shams
                       if r["report"] not in ("NONE", "INVALID")),
    "parse_named": sum(len(v) for v in named_rows(post_b).values()),
    "parse_n": sum(len(post_b.get(a) or []) for a in L.ARMS),
    "ppl_delta_pct": round(100 * (BUNDLE["ppl_post"] / BUNDLE["ppl_pre"] - 1),
                           2),
    "curriculum_ok": BUNDLE["curriculum"] == L.EXPECT_CURRICULUM_V3,
    "undertrained": bool(not BUNDLE["train_log"]["plateaued"]
                         and BUNDLE["train_log"]["epochs_flown"]
                         == BUNDLE["train_log"]["epochs_cap"]),
    "comp_last_rel": round((prev - cur) / prev, 4),
    "comp_still_descending": bool((prev - cur) / prev >= L.PLATEAU_REL),
    "slopes_post": {a: cal_post[a]["slope"] for a in cal_post},
    "loo_post": loo_post,
}

sv = SHIPPED["conditions"]["real"]
sec, pr = SHIPPED["secondaries"], SHIPPED["primaries"]
expect = {
    "catch_pre": sv["pre"]["catch"]["passed"],
    "catch_post": sv["post"]["catch"]["passed"],
    "catch_post_named": sv["post"]["catch"]["named"],
    "pv1_p0": pr["P_V1_competence_on_budget"]["p0_smoothed"],
    "pv1_p_exact": pr["P_V1_competence_on_budget"]["p"],
    "post_pooled_rho": pr["P_V2_tracking_retention"]["rho"],
    "pre_pooled_rho": sec["S0_third_replication"]["rho"],
    "per_arm_post": {a: sec["S1_per_arm_post"]["arms"][a]["rho"]
                     for a in L.ARMS},
    "straight_rho": sec["S4_straight_vs_flipped"]["straight"]["rho"],
    "flipped_rho": sec["S4_straight_vs_flipped"]["flipped"]["rho"],
    "s10_n_paired": sec["S10_fc_interference"]["n_paired"],
    "s10_d_mean": sec["S10_fc_interference"]["d_mean"],
    "s10_median_v3": sec["S10_fc_interference"]["median_v3"],
    "s10_median_v2": sec["S10_fc_interference"]["median_v2_pinned"],
    "s10_n_improved": sec["S10_fc_interference"]["n_improved"],
    "fc_n": sv["fc"]["n"],
    "fc_median": sv["fc"]["median_err"],
    "fc_exact": sv["fc"]["exact"],
    "fc_none_top": sv["fc"]["none_top_injected"],
    "fc_sham_none_top": sv["fc_sham"]["none_top"],
    "anchor_exact": SHIPPED["gates"]["g_anchor"]["exact"],
    "sham_claims": SHIPPED["gates"]["g5_sham"]["claims"],
    "parse_named": SHIPPED["gates"]["g4_parse"]["named"],
    "parse_n": SHIPPED["gates"]["g4_parse"]["n"],
    "ppl_delta_pct": SHIPPED["gates"]["g3_ppl"]["delta_pct"],
    "curriculum_ok": SHIPPED["gates"]["g1_pins"]["pass"],
    "undertrained": sv["undertrained"],
    "comp_last_rel": sv["competence_strand"]["last_rel_improvement"],
    "comp_still_descending": sv["competence_strand"]["still_descending"],
    "slopes_post": {a: sec["S_ABS"]["post"]["per_arm"][a]["slope"]
                    for a in cal_post},
    "loo_post": {a: sec["S_ABS"]["post"]["loo"][a] for a in loo_post},
}

ok = True
def close(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b

for k in fresh:
    m = close(fresh[k], expect[k])
    ok &= m
    print(f"  fresh {k}: {fresh[k]} vs shipped {expect[k]} "
          f"{'==' if m else 'MISMATCH'}")

# ── 3. the before/after table, re-derived ───────────────────────────────────
ba = SHIPPED["before_after"]
ba_ok = (ba["catch"][:2] == [fresh["catch_pre"], fresh["catch_post"]]
         and abs(ba["tracking"][1] - fresh["post_pooled_rho"]) < 1e-9
         and ba["tracking"][0] == 0.6711
         and abs(ba["fc_median"][1] - fresh["s10_median_v3"]) < 1e-9
         and ba["fc_median"][0] == 46.98
         and ba["epochs"] [:2] == [4, BUNDLE["train_log"]["epochs_flown"]])
print(f"  before/after table: {'==' if ba_ok else 'MISMATCH'} "
      f"(catch {ba['catch'][0]} -> {ba['catch'][1]}, tracking "
      f"{ba['tracking'][0]} -> {ba['tracking'][1]}, fc {ba['fc_median'][0]} "
      f"-> {ba['fc_median'][1]}, epochs {ba['epochs'][0]} -> {ba['epochs'][1]})")

if not diffs and ok and ba_ok:
    print("\nRECOMPUTE CLEAN — shipped verdict reproduces verbatim; fresh "
          "tallies match; before/after table re-derived.")
    sys.exit(0)
print("\nRECOMPUTE FOUND DIFFERENCES — adjudicate before lock.")
sys.exit(1)
