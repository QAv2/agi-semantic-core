#!/usr/bin/env python3
"""E7b-Q flight recompute (the law): execute the committed notebook's VERDICT
cell VERBATIM over the raw shipped condition bundles, diff against the shipped
verdict.json (seeded Generator streams reproduce cross-machine — proven at the
E8-J lock), then re-derive every headline ingredient with fresh hand-rolled
code: P-W1 per-replicate contraction deltas + an EXACT 2^16 sign-flip
enumeration behind the MC p, P-W2 per-turn Spearman via an independent
tie-rank implementation (+ a fresh-seed MC consistency band), all alt-gauge /
S-BASE / S-ENTROPY / S-LADDER observables, all four riders' observed
statistics from the raw s_pre14 states, the trajectory tables, the gate
tallies (plan counts, capture spot tooth re-applied through the pinned
encoders), the bundle-integrity tooth visible_mass(text) == vis_mass on every
row, and the two-band G-DIRS drift judgment the 2026-08-26 amendment defers
to this recompute.

Run:  ~/venvs/semcore/bin/python3 colab/recompute_e7bq_flight.py [full_STAMP]
      ~/venvs/semcore/bin/python3 colab/recompute_e7bq_flight.py --selftest
--selftest exercises the fresh-tally code over a synthetic FB1 flight (the
instrument is debugged blind, never against the real bundles).
"""
import io
import json
import math
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e7bq_logic as L

NB = json.load(open(os.path.join(HERE, "E7BQ_WALK_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
VERDICT_CELL = next(c for c in CELLS if c.startswith("# ── Verdict"))
LOGIC_CELL = next(c for c in CELLS if "E7b-Q pure logic" in c.splitlines()[0])
assert LOGIC_CELL == open(os.path.join(HERE, "e7bq_logic.py")).read(), (
    "notebook logic cell != e7bq_logic.py — single-source violation")

PAYLOAD_RAW = open(os.path.join(HERE, "e7bq_payload.json"), "rb").read()
PAYLOAD_SHA = L.payload_sha(PAYLOAD_RAW)
PAYLOAD = json.loads(PAYLOAD_RAW)
BEING = [float(x) for x in PAYLOAD["being_vec"]]
XBAR = [float(x) for x in PAYLOAD["xbar_pack"]]

SELFTEST = "--selftest" in sys.argv


def run_verdict_cell(bundles, mode, stamp, plan_sha, nb_build):
    """The committed verdict cell, verbatim, with the flight surface stubbed
    at the REAL flight's values."""
    ns = {}
    exec(LOGIC_CELL, ns)
    tmp = tempfile.mkdtemp(prefix="e7bq_rc_")
    ns.update({
        "BUNDLES": bundles, "PAYLOAD": PAYLOAD, "MODE": mode,
        "SMOKE": mode == "smoke", "STAMP": stamp, "PLAN_SHA": plan_sha,
        "NB_BUILD": nb_build, "PAYLOAD_SHA_PIN": PAYLOAD_SHA,
        "cond_errors": {}, "OUT": Path(tmp),
        "INFLIGHT": "e7bq/inflight_recompute",
        "ship": lambda *a, **k: None})
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(VERDICT_CELL, ns)
    return json.load(open(Path(tmp) / "verdict.json")), buf.getvalue()


# ── structural diff (nan==nan; floats at 1e-12) ─────────────────────────────
def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k), b.get(k), f"{path}.{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", diffs)
    else:
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return
            if not abs(a - b) < 1e-12:
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))


# ── independent machinery (no e7bq_logic calls anywhere below) ──────────────
def ranks_ind(vals):
    """Tie-averaged ranks via value-grouping (independent of L._ranks)."""
    groups = {}
    for i, v in enumerate(vals):
        assert not math.isnan(float(v)), "nan reached the rank path"
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


def spearman_ind(a, b):
    """Pearson of tie-averaged ranks, population std; None if degenerate."""
    ra, rb = ranks_ind(a), ranks_ind(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0.0 or vb == 0.0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(va * vb)


def fisher_ind(r):
    r = min(max(r, -0.999), 0.999)
    return 0.5 * math.log((1 + r) / (1 - r))


def norm_ind(v, w):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(v, w)))


def grid_ind(bundle, arm, valfn):
    """(R, T) nested lists straight off the rows — independent assembly."""
    rows = [r for r in bundle["rows"] if r["arm"] == arm]
    R = 1 + max(r["rep"] for r in rows)
    T = 1 + max(r["turn"] for r in rows)
    M = [[float("nan")] * T for _ in range(R)]
    for r in rows:
        M[r["rep"]][r["turn"]] = valfn(r)
    return M


def col(M, t):
    return [M[i][t] for i in range(len(M))]


def win_mean(row, win):
    return sum(row[t] for t in win) / len(win)


def pw1_obs_ind(gw, gs):
    d = [(win_mean(gw[i], L.TERM_WIN) - win_mean(gw[i], L.BASE_WIN))
         - (win_mean(gs[i], L.TERM_WIN) - win_mean(gs[i], L.BASE_WIN))
         for i in range(len(gw))]
    return sum(d) / len(d), d


def pw1_exact_p(d):
    """Exact one-sided sign-flip p over all 2^R patterns (R=16 -> 65,536)."""
    d = np.asarray(d, float)
    n = len(d)
    signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2.0 - 1.0
    null = (signs * d[None, :]).mean(axis=1)
    obs = float(d.mean())
    return float(np.mean(null <= obs + 1e-15))


def pw2_obs_ind(g, m, turns):
    zs = []
    for t in turns:
        a, b = col(g, t), col(m, t)
        rho = spearman_ind(a, b)
        if rho is None:
            continue
        zs.append(fisher_ind(min(max(rho, -0.999), 0.999)))
    if not zs:
        return 0.0, 0
    return sum(zs) / len(zs), len(zs)


def pw2_fresh_p(g, m, turns, n_perm, seed):
    """Same null (whole-row replicate permutation), FRESH seed + independent
    per-permutation recompute — an MC consistency band, not an exactness pin."""
    g = np.asarray(g, float)
    m = np.asarray(m, float)
    R = g.shape[0]
    obs, _ = pw2_obs_ind(g.tolist(), m.tolist(), turns)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        pi = rng.permutation(R)
        z, _ = pw2_obs_ind(g.tolist(), m[pi].tolist(), turns)
        if z >= obs:
            hits += 1
    return (1 + hits) / (n_perm + 1)


def rider_mat_ind(bundle, arm, turns):
    """(R, len(turns), d) raw s_pre14 pulled row-by-row."""
    rows = {(r["rep"], r["turn"]): r for r in bundle["rows"]
            if r["arm"] == arm}
    R = 1 + max(k[0] for k in rows)
    out = []
    for rep in range(R):
        out.append([[float(x) for x in rows[(rep, t)]["s_pre14"]]
                    for t in turns])
    return out


def attr_obs_ind(S):
    """log(term dispersion / base dispersion); windows = slots [0,1]/[2,3]
    of the [0,1,8,9] rider matrix; mean pairwise distance over replicate
    pairs x window slots."""
    R = len(S)

    def disp(slots):
        tot, n = 0.0, 0
        for t in slots:
            for i in range(R):
                for j in range(i + 1, R):
                    tot += norm_ind(S[i][t], S[j][t])
                    n += 1
        return tot / n

    return math.log(max(disp([2, 3]), 1e-12) / max(disp([0, 1]), 1e-12))


def sep_obs_ind(A, B):
    cross, nc, within, nw = 0.0, 0, 0.0, 0
    allX = [(a, 0) for a in A] + [(b, 1) for b in B]
    for i in range(len(allX)):
        for j in range(i + 1, len(allX)):
            d = norm_ind(allX[i][0], allX[j][0])
            if allX[i][1] != allX[j][1]:
                cross += d
                nc += 1
            else:
                within += d
                nw += 1
    return (cross / nc) / max(within / nw, 1e-12)


def dbe_ind(r):
    ch = r.get("chat_pre14")
    return norm_ind(ch, BEING) if ch is not None else float("nan")


VALFNS = {
    "dbeing14": dbe_ind,
    "dcent14": lambda r: (norm_ind(r["chat_pre14"], XBAR)
                          if r.get("chat_pre14") is not None
                          else float("nan")),
    "dbeing20": lambda r: (norm_ind(r["chat_pre20"], BEING)
                           if r.get("chat_pre20") is not None
                           else float("nan")),
    "dbeing_gen14": lambda r: (norm_ind(r["chat_gen14"], BEING)
                               if r.get("chat_gen14") is not None
                               else float("nan")),
    "mass": lambda r: float(r["vis_mass"]),
    "ent": lambda r: (float(r["ent_mean"]) if r.get("ent_mean") is not None
                      else float("nan")),
    "radius": lambda r: float(r["radius_pre14"]),
    "cone": lambda r: float(r["cone_pre14"]),
}


def nanmean_ind(vals):
    xs = [v for v in vals if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else float("nan")


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol)
                                        for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return abs(fa - fb) < tol
    return a == b


def fresh_tallies(bundles, V, n_perm):
    """Every headline observable re-derived independently, compared to the
    verdict values. Returns (n_match, mismatches)."""
    real, base = bundles["real"], bundles["base"]
    pr, sec, rid = V["primaries"], V["secondaries"], V["riders"]
    rows_out = []

    def add(label, fresh, shipped, dp=None):
        f = fresh
        if dp is not None and isinstance(fresh, float):
            f = round(fresh, dp)
        rows_out.append((label, f, shipped, fresh))

    # primaries
    gw = grid_ind(real, "walked", VALFNS["dbeing14"])
    gs = grid_ind(real, "sham", VALFNS["dbeing14"])
    mw = grid_ind(real, "walked", VALFNS["mass"])
    o1, d1 = pw1_obs_ind(gw, gs)
    add("PW1 obs", o1, pr["PW1"]["obs"], 4)
    o2, nu2 = pw2_obs_ind(gw, mw, L.COUPLE_IDX)
    add("PW2 obs", o2, pr["PW2"]["obs"], 4)
    add("PW2 turns", nu2, pr["PW2"]["turns"])

    # exact enumeration behind the P-W1 MC p
    p_exact = pw1_exact_p(d1)
    p_mc = pr["PW1"]["p"]
    sd = math.sqrt(max(p_exact * (1 - p_exact), 1e-12) / n_perm)
    band = 5 * sd + 2.0 / (n_perm + 1)
    print(f"  P-W1 exact sign-flip p (2^{len(d1)} enumeration): "
          f"{p_exact:.6g} | shipped MC p {p_mc:.6g} | band ±{band:.2g} "
          f"-> {'CONSISTENT' if abs(p_mc - p_exact) <= band else 'MISMATCH'}")
    rows_out.append(("PW1 p exact-vs-MC", "in-band"
                     if abs(p_mc - p_exact) <= band else "OUT-OF-BAND",
                     "in-band", p_exact))

    # fresh-seed MC for P-W2 (consistency band)
    p2f = pw2_fresh_p(gw, mw, L.COUPLE_IDX, min(n_perm, 2000),
                      L.E7BQ_SEED + 990001)
    p2 = pr["PW2"]["p"]
    sd2 = math.sqrt(max(p2f * (1 - p2f), 2.5e-7) / min(n_perm, 2000))
    band2 = 5 * sd2 + 2.0 / (min(n_perm, 2000) + 1)
    print(f"  P-W2 fresh-seed MC p: {p2f:.6g} | shipped {p2:.6g} | "
          f"band ±{band2:.2g} -> "
          f"{'CONSISTENT' if abs(p2 - p2f) <= band2 else 'MISMATCH'}")
    rows_out.append(("PW2 p fresh-MC", "in-band"
                     if abs(p2 - p2f) <= band2 else "OUT-OF-BAND",
                     "in-band", p2f))

    # trajectory tables
    T = len(gw[0])
    add("traj dbeing walked",
        [round(nanmean_ind(col(gw, t)), 4) for t in range(T)],
        V["trajectory"]["dbeing_walked_mean"])
    add("traj dbeing sham",
        [round(nanmean_ind(col(gs, t)), 4) for t in range(T)],
        V["trajectory"]["dbeing_sham_mean"])
    add("traj mass walked",
        [round(nanmean_ind(col(mw, t)), 1) for t in range(T)],
        V["trajectory"]["mass_walked_mean"])
    ms = grid_ind(real, "sham", VALFNS["mass"])
    add("traj mass sham",
        [round(nanmean_ind(col(ms, t)), 1) for t in range(T)],
        V["trajectory"]["mass_sham_mean"])

    # S-BASE
    gbw = grid_ind(base, "walked", VALFNS["dbeing14"])
    gbs = grid_ind(base, "sham", VALFNS["dbeing14"])
    mbw = grid_ind(base, "walked", VALFNS["mass"])
    ob1, _ = pw1_obs_ind(gbw, gbs)
    ob2, nub = pw2_obs_ind(gbw, mbw, L.COUPLE_IDX)
    add("S_BASE contrast", ob1, sec["S_BASE"]["contrast"], 4)
    add("S_BASE couple", ob2, sec["S_BASE"]["couple"], 4)
    add("S_BASE couple_turns", nub, sec["S_BASE"]["couple_turns"])

    # alt gauges (real cond): contrast + couple observables
    for name, kind in (("S_DCENT", "dcent14"), ("S_RADIUS", "radius"),
                       ("S_CONE", "cone"), ("S_GEN", "dbeing_gen14"),
                       ("S_L20", "dbeing20")):
        gg = grid_ind(real, "walked", VALFNS[kind])
        ggs = grid_ind(real, "sham", VALFNS[kind])
        oo, _ = pw1_obs_ind(gg, ggs)
        add(f"{name} contrast", oo, sec[name]["contrast"], 4)
        oc, nc_ = pw2_obs_ind(gg, mw, L.COUPLE_IDX)
        add(f"{name} couple", oc, sec[name]["couple"], 4)
        add(f"{name} couple_turns", nc_, sec[name]["couple_turns"])

    ew = grid_ind(real, "walked", VALFNS["ent"])
    oe, ne_ = pw2_obs_ind(gw, ew, L.COUPLE_IDX)
    add("S_ENTROPY couple", oe, sec["S_ENTROPY"]["couple"], 4)
    add("S_ENTROPY couple_turns", ne_, sec["S_ENTROPY"]["couple_turns"])

    add("S_LADDER dbeing",
        [round(nanmean_ind(col(gw, t)), 4) for t in L.EMERG_IDX],
        sec["S_LADDER"]["dbeing_E123"])
    add("S_LADDER mass",
        [round(nanmean_ind(col(mw, t)), 1) for t in L.EMERG_IDX],
        sec["S_LADDER"]["mass_E123"])

    # riders, both conditions, observed statistics from raw states
    for cname, b in (("real", real), ("base", base)):
        S = rider_mat_ind(b, "walked", [0, 1, 8, 9])
        add(f"R_ATTR[{cname}] log_ratio", attr_obs_ind(S),
            rid[cname]["R_ATTR"]["log_ratio"], 4)
        w_e1 = [r[0] for r in rider_mat_ind(b, "walked", [10])]
        r_e1 = [r[0] for r in rider_mat_ind(b, "reorder", [10])]
        u_e1 = [r[0] for r in rider_mat_ind(b, "unwalked", [0])]
        add(f"R_PATH[{cname}] sep", sep_obs_ind(w_e1, r_e1),
            rid[cname]["R_PATH"]["sep"], 4)
        add(f"R_UNWALKED[{cname}] sep", sep_obs_ind(w_e1, u_e1),
            rid[cname]["R_UNWALKED"]["sep"], 4)
        w_e2 = [r[0] for r in rider_mat_ind(b, "walked", [11])]
        w_g = [r[0] for r in rider_mat_ind(b, "walked", [14])]
        gaps = [norm_ind(w_g[i], w_e1[i]) - norm_ind(w_e2[i], w_e1[i])
                for i in range(len(w_e1))]
        add(f"R_GOBACK[{cname}] gap_diff", sum(gaps) / len(gaps),
            rid[cname]["R_GOBACK"]["gap_diff"], 4)
        mm = grid_ind(b, "walked", VALFNS["mass"])
        add(f"R_GOBACK[{cname}] mass_G-E1",
            round(nanmean_ind(col(mm, 14)) - nanmean_ind(col(mm, 10)), 1),
            rid[cname]["R_GOBACK"]["mass_G_minus_E1"])

    # gates, fresh
    exp = L.expected_counts(False)
    counts_ok = True
    for cond in ("base", "real"):
        got = {}
        for r in bundles[cond]["rows"]:
            got[r["arm"]] = got.get(r["arm"], 0) + 1
        counts_ok &= (got == exp)
    add("G-PLAN counts", counts_ok, bool(V["gates"]["g_plan"]["pass"]))

    spot = []
    for cond in ("base", "real"):
        b = bundles[cond]
        cent = [float(x) for x in b["centroid"]["14"]]
        W = np.asarray(PAYLOAD["encoders"]
                       ["base14" if cond == "base" else "inst14"], float)
        for r in b["rows"]:
            if r["turn"] in L.RIDER_TURNS[r["arm"]]:
                v = np.asarray(r["s_pre14"], float) - np.asarray(cent)
                v = v / max(float(np.linalg.norm(v)), 1e-12)
                rec = np.concatenate([v, [1.0]]) @ W
                spot.append(float(np.max(np.abs(
                    rec - np.asarray(r["chat_pre14"], float)))))
    add("G-CAPTURE spot n", len(spot), V["gates"]["g_capture"]["spot_n"])
    add("G-CAPTURE spot worst", max(spot),
        V["gates"]["g_capture"]["spot_worst"], None)
    add("G-DIRS worst",
        max(v for c in ("base", "real")
            for v in bundles[c]["gdirs"].values()),
        V["gates"]["g_dirs"]["worst"], None)

    n_bad = 0
    for label, f, s, raw in rows_out:
        m = close(f, s)
        n_bad += (not m)
        tag = "==" if m else "MISMATCH"
        print(f"  fresh {label}: {f} vs shipped {s} {tag}"
              + ("" if m else f"   (raw {raw})"))
    return len(rows_out) - n_bad, n_bad


def integrity(bundles, check_mass=True):
    """Bundle-integrity teeth: ledger selftest; visible_mass(text) ==
    vis_mass on every row; chat_pre14 finite everywhere; empty-reply census."""
    assert L.ledger_selftest()
    bad_mass, empties, n_rows = 0, 0, 0
    for cond in ("base", "real"):
        for r in bundles[cond]["rows"]:
            n_rows += 1
            if check_mass and L.visible_mass(r["text"]) != r["vis_mass"]:
                bad_mass += 1
            if r.get("chat_gen14") is None:
                empties += 1
            assert r.get("chat_pre14") is not None
            assert all(math.isfinite(float(x)) for x in r["chat_pre14"])
    print(f"  integrity: {n_rows} rows | mass-recompute mismatches "
          f"{bad_mass if check_mass else 'n/a (selftest)'} | "
          f"empty replies (no s_gen) {empties}")
    return bad_mass == 0 or not check_mass


def main():
    if SELFTEST:
        print("== SELFTEST: synthetic FB1 full flight through the "
              "instrument ==")
        b = L.synth_flight(PAYLOAD, "fb1", smoke=False, seed=41)
        for c in b:
            b[c]["mode"] = "full"
        V, _ = run_verdict_cell(json.loads(json.dumps(b)), "full",
                                "selftest_0000", "selfplan0", "selftest")
        ok_i = integrity(b, check_mass=False)
        n_ok, n_bad = fresh_tallies(b, V, V["n_perm"])
        print(f"selftest: {n_ok} fresh tallies match, {n_bad} mismatches, "
              f"integrity {'ok' if ok_i else 'FAIL'}")
        sys.exit(0 if (n_bad == 0 and ok_i) else 1)

    stamp_dir = next((a for a in sys.argv[1:] if not a.startswith("-")),
                     None)
    assert stamp_dir, "usage: recompute_e7bq_flight.py full_STAMP"
    RES = os.path.join(HERE, "results_e7bq", stamp_dir)
    SHIPPED = json.load(open(os.path.join(RES, "verdict.json")))
    bundles = {c: json.load(open(os.path.join(RES, f"condition_{c}.json")))
               for c in ("base", "real")}

    # pins before anything else
    plan_local, plan_sha_local = L.build_plan(PAYLOAD, smoke=False)
    assert SHIPPED["mode"] == "full"
    assert SHIPPED["payload_sha"] == PAYLOAD_SHA, "payload sha mismatch"
    assert SHIPPED["plan_sha"] == plan_sha_local, "plan sha mismatch"
    assert SHIPPED["script_sha"] == L.script_sha(PAYLOAD["script"])
    for c, b in bundles.items():
        assert b["cond"] == c and b["mode"] == "full"
        assert b["payload_sha"] == PAYLOAD_SHA and b["plan_sha"] == plan_sha_local
        assert b["stamp"] == SHIPPED["stamp"]
        assert not b.get("cond_error")
        print(f"[{c}] stamp {b['stamp']} | t {b['t_sec']:.0f}s | "
              f"env {b['env']} | gdirs {b['gdirs']}")
    print(f"pins: payload {PAYLOAD_SHA} | plan {plan_sha_local} | "
          f"script {SHIPPED['script_sha']} | nb '{SHIPPED['nb_build']}' "
          f"| {len(plan_local)} planned generations")

    # 1. the verdict cell, verbatim, over the raw bundles
    RE, out = run_verdict_cell(json.loads(json.dumps(bundles)), "full",
                               SHIPPED["stamp"], SHIPPED["plan_sha"],
                               SHIPPED["nb_build"])
    diffs = []
    walk(SHIPPED, RE, "", diffs)
    print(f"\nverdict-cell recompute: {len(diffs)} substantive diffs")
    for p_, a, b in diffs[:12]:
        print(f"  DIFF {p_}: shipped {a} vs recomputed {b}")

    # 2. bundle integrity + fresh tallies
    print("\nbundle integrity:")
    ok_i = integrity(bundles)
    print("\nfresh hand-rolled tallies:")
    n_ok, n_bad = fresh_tallies(bundles, SHIPPED, SHIPPED["n_perm"])

    # 3. the two-band G-DIRS drift judgment (amendment 2026-08-26: judged
    #    at recompute). Wrong-setup floors measured on the pinned anchors:
    #    wrong-adapter .379-.634, wrong-layer .072-.158.
    print("\nG-DIRS drift judgment:")
    worst = 0.0
    for c in ("base", "real"):
        for lay, v in bundles[c]["gdirs"].items():
            worst = max(worst, float(v))
            print(f"  {c} L{lay}: {float(v):.3e}")
    verdictline = (
        "in the benign cross-era band" if worst < 2e-2 else "ABOVE HARD")
    print(f"  worst {worst:.3e} | noise 1e-4 | hard 2e-2 | weakest "
          f"wrong-setup signature 7.2e-2 -> {verdictline}")
    print("  judgment: drift is common-mode across arms and replicates "
          "(both arms of every contrast fly the same condition on the same "
          "stack in the same kernel) — it cannot manufacture a walked-vs-"
          "sham or within-turn contrast; absolute D_BEING levels carry a "
          f"~{worst:.0e} state-side uncertainty, stated in the lock.")

    ok = (not diffs) and ok_i and n_bad == 0 and worst < 2e-2
    if ok:
        print("\nRECOMPUTE CLEAN — shipped verdict reproduces verbatim; "
              f"{n_ok} fresh tallies match; bundle integrity holds; drift "
              "judged benign.")
        sys.exit(0)
    print("\nRECOMPUTE FOUND DIFFERENCES — adjudicate before lock.")
    sys.exit(1)


main()
