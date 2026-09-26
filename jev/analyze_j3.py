"""J3 analysis (protocol §8–§10): gates, the three primaries per presentation,
secondaries, and the mouth rider. numpy + scipy only; the fitted bar comes from
FROZEN_J3.json. Uncertainty: cluster bootstrap by base state, seeded.
"""
import re

import numpy as np

from . import j3
from . import j3_mouth as M
from . import stats as S

NONE = j3.NONE
THR = j3.THR
B_DEFAULT = 10_000


def arng(*words):
    return np.random.default_rng([20261001, *[j3.C._tag(w) if isinstance(w, str) else int(w) for w in words]])


# ── statistics ───────────────────────────────────────────────────────────────

def aurc(conf, correct):
    """Mean selective risk over coverage 1/n..n/n, most confident first; within a
    tie group the cumulative errors are interpolated (expected under random ties)."""
    conf, err = np.asarray(conf, float), 1.0 - np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return np.nan
    o = np.argsort(-conf, kind="stable")
    c, e = conf[o], err[o]
    _, start, counts = np.unique(-c, return_index=True, return_counts=True)
    gid = np.repeat(np.arange(len(counts)), counts)
    g_err = np.add.reduceat(e, start)
    before = np.concatenate([[0.0], np.cumsum(g_err)[:-1]])
    within = np.arange(n) - start[gid] + 1
    cum = before[gid] + within * g_err[gid] / counts[gid]
    return float(np.mean(cum / np.arange(1, n + 1)))


def aurc_oracle(correct):
    correct = np.asarray(correct, float)
    n, k = len(correct), int(correct.sum())
    i = np.arange(1, n + 1)
    return float(np.mean(np.where(i <= k, 0.0, (i - k) / i)))


def ece(conf, correct, tiekey, nb=15):
    """Equal-mass ECE: sort by confidence, ties in plan order (`tiekey`), nb groups."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    o = np.lexsort((tiekey, conf))
    return float(sum(len(b) * abs(conf[b].mean() - correct[b].mean()) for b in np.array_split(o, nb)) / len(o))


def fa_fit_at(H, s_push, s_unt):
    """Fitted false-alarm share at the threshold giving hit share H on s_push."""
    if H <= 0:
        return 0.0
    t = np.quantile(s_push, 1 - H)
    return float(np.mean(s_unt >= t))


def q95(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [np.nan, np.nan]


# ── the call table ───────────────────────────────────────────────────────────

def calls_table(specs, finals):
    """One row per decision-bearing call with an ok final record. `pushed` P(yes)
    is joined from the same call (bundled) or from its '-q' twin (separate)."""
    noul = {}
    for s in specs:
        if s.get("mode") == "pushed":
            rec = finals.get(s["call_id"])
            if rec and rec["status"] == "ok":
                noul[s["call_id"][:-2]] = rec["response"]["answers"]["pushed"]["noul"]
    rows = []
    for s in specs:
        if s.get("mode") not in ("bundled", "decision"):
            continue
        rec = finals.get(s["call_id"])
        if not rec or rec["status"] != "ok":
            continue
        ans = rec["response"]["answers"]
        dec = ans["decision"]
        k2i = s["key_to_idx"]
        probs = np.zeros(NONE + 1)
        for key, p in dec["probabilities"].items():
            probs[k2i[key]] = p
        ch = k2i[dec["choice"]]
        pyes = ans["pushed"]["noul"] if "pushed" in ans else noul.get(s["call_id"], np.nan)
        rows.append({"cid": s["call_id"], "sid": s["sid"], "j": s["j"], "slot": s["slot"], "state": s["state"],
                     "half": s["half"], "ref": s["ref"], "kind": s["kind"], "dose": s["dose"], "truth": s["truth"],
                     "pres": s["pres"], "rep": s["rep"], "order": s["order"], "choice": ch, "c": float(probs[ch]),
                     "probs": probs, "vconf": dec.get("confidence"), "pyes": pyes, "sims": s["sims"],
                     "shift": s["shift"], "argmax_ok": bool(probs[ch] >= probs.max() - 1e-12),
                     "model": rec["response"].get("model")})
    return rows


def as_arrays(rows):
    A = {k: np.array([r[k] for r in rows]) for k in ("sid", "j", "slot", "state", "half", "ref", "kind", "dose",
                                                    "truth", "pres", "rep", "choice", "c", "pyes", "shift")}
    A["probs"] = np.array([r["probs"] for r in rows]).reshape(len(rows), NONE + 1)
    A["order"] = np.array([r["order"] for r in rows], int).reshape(len(rows), NONE + 1)
    A["sims"] = np.array([r["sims"] for r in rows]).reshape(len(rows), NONE)
    A["tie"] = np.arange(len(rows))
    return A


def with_bar(A, bars):
    """The fitted bar's distribution for each call's stimulus (its reference half)."""
    P = np.zeros((len(A["c"]), NONE + 1))
    for h in (0, 1):
        m = A["ref"] == h
        if m.any():
            P[m] = j3.bar_predict(bars[str(h)], A["sims"][m], A["shift"][m])
    A["P_fit"] = P
    A["fit_choice"] = P.argmax(1)
    A["fit_c"] = P.max(1)
    return A


# ── primaries ────────────────────────────────────────────────────────────────

def subsets(A):
    k, d = A["kind"], A["dose"]
    thr = np.isin(d, THR)
    return {"U": k == "untouched", "Tp": (k == "push") & thr, "Rc": (k == "random") & (d == j3.CEIL),
            "Cp": (k == "push") & (d == j3.CEIL),
            "pool": (k == "untouched") | ((k == "random") & thr) | ((k == "push") & thr)}


def point_stats(A, ix):
    """All primary statistics on the call index multiset ix."""
    ch, tr = A["choice"][ix], A["truth"][ix]
    sub = {k: v[ix] for k, v in subsets(A).items()}
    named = ch != NONE
    cor = ch == tr
    out = {}
    out["ceil_acc"] = float(cor[sub["Cp"]].mean()) if sub["Cp"].any() else np.nan
    FA = float(named[sub["U"]].mean())
    H = float(named[sub["Tp"]].mean())
    det = 1 - A["P_fit"][ix, NONE]
    FAf = fa_fit_at(H, det[sub["Tp"]], det[sub["U"]])
    out.update(FA=FA, H=H, FA_fit=FAf, excess=FA - FAf, R=float(named[sub["Rc"]].mean()))
    p = sub["pool"]
    c, cr = A["c"][ix][p], cor[p]
    out["ece"] = ece(c, cr, A["tie"][ix][p])
    out["auroc2"] = float(S.auroc(c, cr.astype(int)))
    fc = A["fit_c"][ix][p]
    fcr = (A["fit_choice"][ix] == tr)[p]
    out["aurc"] = aurc(c, cr)
    out["aurc_fit"] = aurc(fc, fcr)
    out["daurc"] = out["aurc"] - out["aurc_fit"]
    out["acc_pool"] = float(cr.mean())
    out["acc_pool_fit"] = float(fcr.mean())
    return out


def bootstrap(A, B, rng, stat_keys=("ceil_acc", "excess", "R", "ece", "auroc2", "daurc", "FA")):
    states = np.unique(A["state"])
    by_state = {s: np.flatnonzero(A["state"] == s) for s in states}
    out = {k: np.empty(B) for k in stat_keys}
    for b in range(B):
        pick = rng.choice(states, len(states), replace=True)
        ix = np.concatenate([by_state[s] for s in pick])
        st = point_stats(A, ix)
        for k in stat_keys:
            out[k][b] = st[k]
    return out


def three(lo, hi, bar):
    """PASS if upper <= bar; FAIL if lower > bar; else INCONCLUSIVE."""
    if hi <= bar:
        return "PASS"
    if lo > bar:
        return "FAIL"
    return "INCONCLUSIVE"


def primaries(A, B, rng):
    pt = point_stats(A, np.arange(len(A["c"])))
    bs = bootstrap(A, B, rng)
    ci = {k: q95(v) for k, v in bs.items()}
    # P1 (branch order: G-CEIL, claims, names-the-push, leaky, PASS, INCONCLUSIVE)
    gate = pt["ceil_acc"] >= 0.50
    a = three(*ci["excess"], 0.10)
    b = three(*ci["R"], 0.15)
    if not gate:
        v1 = "NOT ADJUDICABLE: does not read this presentation"
    elif pt["FA"] >= 0.50:
        v1 = "FAIL: claims at ceiling"
    elif b == "FAIL":
        v1 = "FAIL: names the push, not the pattern" + ("; leaky silence" if a == "FAIL" else "")
    elif a == "FAIL":
        v1 = "FAIL: leaky silence"
    elif a == "PASS" and b == "PASS":
        v1 = "PASS"
    else:
        v1 = "INCONCLUSIVE"
    P1 = {"verdict": v1, "G_CEIL": {"ceiling_accuracy": pt["ceil_acc"], "ci": ci["ceil_acc"], "pass": bool(gate)},
          "a": {"FA": pt["FA"], "FA_ci": ci["FA"], "H": pt["H"], "FA_fit_at_H": pt["FA_fit"],
                "excess": pt["excess"], "ci": ci["excess"], "clause": a},
          "b": {"random_ceiling_named": pt["R"], "ci": ci["R"], "clause": b}}
    # P2
    p_auc = float(np.mean(bs["auroc2"] <= 0.5))
    inf = p_auc < 0.01
    if not inf:
        v2 = "FAIL: confidence carries no information"
    elif ci["ece"][1] <= 0.10:
        v2 = "PASS"
    elif ci["ece"][0] > 0.10:
        v2 = "FAIL: confident, not calibrated"
    else:
        v2 = "INCONCLUSIVE"
    P2 = {"verdict": v2, "ece": pt["ece"], "ece_ci": ci["ece"], "auroc2": pt["auroc2"], "auroc2_ci": ci["auroc2"],
          "p_auroc2_le_half": p_auc}
    # P3
    t3 = three(*ci["daurc"], 0.05)
    v3 = {"PASS": "PASS", "FAIL": "FAIL: doesn't earn its place", "INCONCLUSIVE": "INCONCLUSIVE"}[t3]
    P3 = {"verdict": v3, "aurc_jev": pt["aurc"], "aurc_fit": pt["aurc_fit"], "daurc": pt["daurc"],
          "ci": ci["daurc"], "acc_pool": pt["acc_pool"], "acc_pool_fit": pt["acc_pool_fit"]}
    return {"P1": P1, "P2": P2, "P3": P3, "n_calls": int(len(A["c"]))}


# ── secondaries ──────────────────────────────────────────────────────────────

def nearest_neighbours(h):
    ref = j3.reference(h)
    g = ref["sig"] @ ref["Si"] @ ref["sig"].T
    nrm = np.sqrt(np.diag(g))
    cos = g / np.outer(nrm, nrm)
    np.fill_diagonal(cos, -np.inf)
    return cos.argmax(1)


def secondaries(A, sub, rng, B):
    out = {}
    ch, tr, cor = A["choice"], A["truth"], A["choice"] == A["truth"]
    named = ch != NONE
    # S1 psychometric (Jev, fitted bar, card rules)
    rule = {s: np.array([j3.card_rule(A["sims"][i], A["shift"][i], s) for i in range(len(ch))]) for s in (False, True)}
    rows = {}
    for lab, m in [("untouched", A["kind"] == "untouched")] + \
                  [(f"push {a}", (A["kind"] == "push") & (A["dose"] == a)) for a in j3.DOSES] + \
                  [(f"random {a}", (A["kind"] == "random") & (A["dose"] == a)) for a in j3.DOSES if a != 0.15]:
        if not m.any():
            continue
        rows[lab] = {"n": int(m.sum()), "jev_acc": float(cor[m].mean()), "jev_named": float(named[m].mean()),
                     "fit_acc": float((A["fit_choice"][m] == tr[m]).mean()),
                     "fit_named": float((A["fit_choice"][m] != NONE).mean()),
                     "rule_i_acc": float((rule[False][m] == tr[m]).mean()),
                     "rule_ii_acc": float((rule[True][m] == tr[m]).mean()),
                     "mean_c": float(A["c"][m].mean())}
    out["S1_psychometric"] = rows
    # S2 E-AURC, ECE bins, per-dose ECE/AURC
    p = sub["pool"]
    out["S2"] = {"e_aurc": aurc(A["c"][p], cor[p]) - aurc_oracle(cor[p]),
                 "e_aurc_fit": aurc(A["fit_c"][p], (A["fit_choice"] == tr)[p]) - aurc_oracle((A["fit_choice"] == tr)[p]),
                 "ece_bins": {nb: ece(A["c"][p], cor[p], A["tie"][p], nb) for nb in (5, 10, 15, 20)},
                 "by_dose": {str(a): {"ece": ece(A["c"][m], cor[m], A["tie"][m]), "aurc": aurc(A["c"][m], cor[m]),
                                      "acc": float(cor[m].mean()), "mean_c": float(A["c"][m].mean())}
                             for a in j3.DOSES for m in [(A["kind"] == "push") & (A["dose"] == a)] if m.any()}}
    # S4 presence without identity
    py = A["pyes"]
    ok = np.isfinite(py)
    s4 = {}
    for lab, pos in (("random_ceiling_vs_untouched", sub["Rc"]), ("threshold_push_vs_untouched", sub["Tp"])):
        m = (pos | sub["U"]) & ok
        s4[lab] = float(S.auroc(py[m], pos[m].astype(int))) if m.any() and pos[m].any() else np.nan
    m = sub["Rc"] & ok
    s4["random_ceiling_none_and_pyes"] = float(np.mean((ch[m] == NONE) & (py[m] >= 0.5))) if m.any() else np.nan
    s4["mean_pyes"] = {"untouched": float(np.nanmean(py[sub["U"]])), "random_ceiling": float(np.nanmean(py[sub["Rc"]])),
                       "threshold_push": float(np.nanmean(py[sub["Tp"]])), "ceiling_push": float(np.nanmean(py[sub["Cp"]]))}
    out["S4_presence"] = s4
    # S5 nearest-name reading
    nn = {h: nearest_neighbours(h) for h in (0, 1)}
    m = (A["kind"] == "push") & named & ~cor
    hits = np.array([ch[i] == nn[A["ref"][i]][tr[i]] for i in np.flatnonzero(m)])
    out["S5_nearest_name"] = {"n_wrong_names": int(m.sum()), "share_nearest": float(hits.mean()) if len(hits) else np.nan,
                              "chance": 1 / 12}
    # S7 vendor confidence identity
    vc = np.array([np.nan if v is None else v for v in A.get("vconf", [])], float) if "vconf" in A else None
    if vc is not None and len(vc):
        out["S7_vendor_confidence"] = {"share_within_0.015": float(np.nanmean(np.abs(vc - (14 * A["c"] - 1) / 13) <= 0.015))}
    # S8 option position on wrong calls
    pos = np.argmax(A["order"] == ch[:, None], axis=1)
    w = ~cor
    out["S8_position_wrong"] = np.bincount(pos[w], minlength=14).tolist()
    # S9 replicate agreement (modal choice across the three option orders)
    agree = {}
    for kind in ("untouched", "random", "push"):
        sids = np.unique(A["sid"][A["kind"] == kind])
        vals = []
        for sd in sids:
            cc = ch[A["sid"] == sd]
            if len(cc) >= 2:
                vals.append(np.bincount(cc, minlength=14).max() / len(cc))
        agree[kind] = float(np.mean(vals)) if vals else np.nan
    out["S9_replicate_agreement"] = agree
    # S10 half symmetry (point estimates by reference half)
    out["S10_by_reference_half"] = {str(h): point_stats(A, np.flatnonzero(A["ref"] == h)) for h in (0, 1)}
    return out


def paired_presentation(Araw, Adig, B, rng):
    """S-J3-3: accuracy(DIGEST) - accuracy(RAW), overconf(RAW) - overconf(DIGEST), on the
    threshold pool, per stimulus (mean over repeats), cluster bootstrap by state."""
    def per_stim(A):
        p = subsets(A)["pool"]
        out = {}
        for sid in np.unique(A["sid"][p]):
            m = A["sid"] == sid
            out[sid] = (A["state"][m][0], float((A["choice"][m] == A["truth"][m]).mean()), float(A["c"][m].mean()))
        return out
    r, d = per_stim(Araw), per_stim(Adig)
    sids = sorted(set(r) & set(d))
    st = np.array([r[s][0] for s in sids])
    acc_r, c_r = np.array([r[s][1] for s in sids]), np.array([r[s][2] for s in sids])
    acc_d, c_d = np.array([d[s][1] for s in sids]), np.array([d[s][2] for s in sids])
    d_acc = acc_d - acc_r
    d_oc = (c_r - acc_r) - (c_d - acc_d)
    states = np.unique(st)
    by = {s: np.flatnonzero(st == s) for s in states}
    bs_a, bs_o = np.empty(B), np.empty(B)
    for b in range(B):
        ix = np.concatenate([by[s] for s in rng.choice(states, len(states), replace=True)])
        bs_a[b], bs_o[b] = d_acc[ix].mean(), d_oc[ix].mean()
    return {"n_stimuli": len(sids), "acc_digest_minus_raw": float(d_acc.mean()), "acc_ci": q95(bs_a),
            "overconf_raw_minus_digest": float(d_oc.mean()), "overconf_ci": q95(bs_o),
            "overconf_raw": float((c_r - acc_r).mean()), "overconf_digest": float((c_d - acc_d).mean())}


# ── the mouth (S-J3-6) ───────────────────────────────────────────────────────

def mouth_table(specs, finals):
    rows = []
    for s in specs:
        rec = finals.get(s["call_id"])
        if not rec or rec["status"] != "ok":
            continue
        key, conf = M.parse_answer(M.content_of(rec["response"]), s["key_to_idx"])
        ch = s["key_to_idx"][key]
        probs = np.full(NONE + 1, np.nan)
        rows.append({"cid": s["call_id"], "sid": s["sid"], "j": s["j"], "slot": s["slot"], "state": s["state"],
                     "half": s["half"], "ref": s["ref"], "kind": s["kind"], "dose": s["dose"], "truth": s["truth"],
                     "pres": "DIGEST", "rep": 0, "order": s["order"], "choice": ch, "c": conf / 100.0,
                     "probs": probs, "vconf": None, "pyes": np.nan, "sims": s["sims"], "shift": s["shift"]})
    return rows


def analyze_mouth(mspecs, mfinals, A_dig, bars, B, rng):
    rows = mouth_table(mspecs, mfinals)
    n_invalid = sum(1 for s in mspecs if mfinals.get(s["call_id"], {}).get("status") != "ok")
    if not rows:
        return {"n_ok": 0, "n_not_ok": n_invalid}
    Am = with_bar(as_arrays(rows), bars)
    res = primaries(Am, B, rng)
    # Jev's repeat-0 DIGEST calls on the same stimuli
    sids = set(Am["sid"])
    m = (A_dig["rep"] == 0) & np.isin(A_dig["sid"], list(sids))
    Aj = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in A_dig.items()}
    Aj["tie"] = np.arange(int(m.sum()))
    jres = primaries(Aj, B, rng)
    conf_levels = np.unique(np.round(Am["c"] * 100, 6))
    return {"n_ok": int(len(rows)), "n_not_ok": n_invalid, "mouth": res, "jev_rep0_same_stimuli": jres,
            "psychometric": secondaries(Am, subsets(Am), rng, 0)["S1_psychometric"],
            "distinct_confidences": int(len(conf_levels)), "confidence_levels": conf_levels[:30].tolist()}


# ── entry ────────────────────────────────────────────────────────────────────

def analyze(specs, finals, frozen, B=B_DEFAULT, mouth=None, secondary=True, pin=None):
    rows = calls_table(specs, finals)
    ok_models = sorted({r["model"] for r in rows})
    statuses = {}
    for rec in finals.values():
        statuses[rec["status"]] = statuses.get(rec["status"], 0) + 1
    out = {"gates": {"G1_builds": ok_models, "G1_pin": pin, "G1_pass": (pin is None or ok_models == [pin]),
                     "G2_status_counts": statuses,
                     "G3_argmax_rate": float(np.mean([r["argmax_ok"] for r in rows])) if rows else np.nan}}
    # missing per presentation x stimulus type
    miss = {}
    dec_specs = [s for s in specs if s.get("mode") in ("bundled", "decision")]
    for pres in ("RAW", "DIGEST"):
        for kind in ("untouched", "random", "push"):
            ids = [s["call_id"] for s in dec_specs if s["pres"] == pres and s["kind"] == kind]
            n_ok = sum(1 for c in ids if finals.get(c, {}).get("status") == "ok")
            miss[f"{pres}/{kind}"] = {"n": len(ids), "missing": len(ids) - n_ok,
                                      "flag": (len(ids) - n_ok) > 0.02 * max(len(ids), 1)}
    out["gates"]["missing"] = miss
    bars = frozen["bars"]
    res = {}
    arrays = {}
    for pres in ("RAW", "DIGEST"):
        pr = [r for r in rows if r["pres"] == pres]
        if not pr:
            continue
        A = with_bar(as_arrays(pr), bars)
        A["vconf"] = np.array([np.nan if r["vconf"] is None else r["vconf"] for r in pr], float)
        arrays[pres] = A
        res[pres] = primaries(A, B, arng("J3", "boot", pres))
        if secondary:
            res[pres]["secondaries"] = secondaries(A, subsets(A), arng("J3", "sec", pres), B)
    out["primaries"] = res
    if secondary and len(arrays) == 2:
        out["S3_presentation_confidence"] = paired_presentation(arrays["RAW"], arrays["DIGEST"], B,
                                                                arng("J3", "S3"))
    if mouth is not None and "DIGEST" in arrays:
        mspecs, mfinals = mouth
        out["S6_mouth"] = analyze_mouth(mspecs, mfinals, arrays["DIGEST"], bars, B, arng("J3", "mouth"))
    # the registered J4 fork (§13)
    passes = [p for p, r in res.items() if r["P1"]["verdict"] == "PASS" and r["P3"]["verdict"] == "PASS"]
    out["J4_fork"] = {"jev_enters_J4": bool(passes), "presentations": passes,
                      "confidence_gates": {p: res[p]["P2"]["verdict"] == "PASS" for p in passes}}
    return out


def summary(v):
    s = {"gates": {k: v["gates"][k] for k in ("G1_builds", "G1_pass", "G2_status_counts", "G3_argmax_rate")}}
    for pres, r in v["primaries"].items():
        s[pres] = {k: r[k]["verdict"] for k in ("P1", "P2", "P3")}
    s["J4_fork"] = v["J4_fork"]
    return s
