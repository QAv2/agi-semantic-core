"""Smoke-test analysis (protocol §4.1): replicate floor, throughput, question
isolation, build pin, cost projection."""
import re

import numpy as np

from . import stats as S
from .analyze_j1 import arng
from .runner import cost_of, iter_records


def _decision(rec, spec):
    dec = rec["response"]["answers"]["decision"]
    k2i = spec["key_to_idx"]
    n = len(k2i)
    probs = np.zeros(max(k2i.values()) + 1)
    for key, p in dec["probabilities"].items():
        probs[k2i[key]] = p
    ci = k2i[dec["choice"]]
    return ci, float(probs[ci]), probs


def analyze(specs, raw_path, n_flight_calls):
    by_id = {s["call_id"]: s for s in specs}
    finals, retries = {}, []
    for rec in iter_records(raw_path):
        (retries.append(rec) if rec["status"] == "retry" else finals.__setitem__(rec["call_id"], rec))
    ok = {cid: r for cid, r in finals.items() if r["status"] == "ok"}
    out = {"n_specs": len(specs), "n_ok": len(ok),
           "status_counts": {k: sum(r["status"] == k for r in finals.values())
                             for k in ("ok", "invalid", "error", "build_mismatch")}}
    builds = sorted({r["response"].get("model") for r in ok.values()})
    out["builds"] = builds
    out["pin"] = builds[0] if len(builds) == 1 else None
    # determinism: identical requests x10
    groups = {}
    for cid, r in ok.items():
        sp = by_id[cid]
        if sp.get("block") == "determinism":
            groups.setdefault(re.sub(r"-x\d+$", "", cid), []).append(_decision(r, sp))
    det = {}
    for g, v in sorted(groups.items()):
        ch = [x[0] for x in v]
        P = np.array([x[2] for x in v])
        pairs = [(i, j) for i in range(len(v)) for j in range(i + 1, len(v))]
        det[g] = {"n": len(v), "flip_rate": 1 - max(ch.count(c) for c in set(ch)) / len(ch),
                  "mean_abs_dp": float(np.mean([np.abs(P[i] - P[j]).mean() for i, j in pairs])) if pairs else None,
                  "mean_tv": float(np.mean([S.tv(P[i], P[j]) for i, j in pairs])) if pairs else None}
    out["determinism"] = {"groups": det,
                          "flip_rate": float(np.mean([d["flip_rate"] for d in det.values()])) if det else None,
                          "mean_abs_dp": float(np.nanmean([d["mean_abs_dp"] for d in det.values()
                                                           if d["mean_abs_dp"] is not None])) if det else None,
                          "mean_tv": float(np.nanmean([d["mean_tv"] for d in det.values()
                                                       if d["mean_tv"] is not None])) if det else None}
    # throughput by concurrency
    thr = {}
    for conc in (1, 2, 4, 8, 16):
        ids = {s["call_id"] for s in specs if s.get("block") == "throughput" and s.get("concurrency") == conc}
        lat = [finals[c]["latency_s"] for c in ids if c in ok]
        n429 = sum(1 for r in retries if r["call_id"] in ids and r.get("http") == 429)
        thr[conc] = {"n": len(ids), "ok": len(lat), "n_429": n429,
                     "p95_latency_s": float(np.percentile(lat, 95)) if lat else None,
                     "median_latency_s": float(np.median(lat)) if lat else None}
    good = [c for c, t in thr.items() if t["n_429"] == 0 and t["ok"] == t["n"] and t["p95_latency_s"] is not None
            and t["p95_latency_s"] < 2.0]
    out["throughput"] = thr
    out["concurrency"] = max(good) if good else 1
    # isolation: choice alone vs bundled with the A6 questions
    iso = {}
    for cid, r in ok.items():
        sp = by_id[cid]
        if sp.get("block") == "isolation":
            key = re.sub(r"-(decision|bundled)-x\d+$", "", cid)
            iso.setdefault(key, {"decision": [], "bundled": []})[sp["mode"]].append(_decision(r, sp)[1])
    eff = []
    for key, v in sorted(iso.items()):
        a, b = v["decision"], v["bundled"]
        if len(a) < 2 or len(b) < 2:
            continue
        between = np.mean([abs(x - y) for x in a for y in b])
        within = [abs(x[i] - x[j]) for x in (a, b) for i in range(len(x)) for j in range(i + 1, len(x))]
        eff.append(between - np.mean(within))
    if eff:
        bs = S.boot_mean_strat(eff, np.zeros(len(eff)), 10_000, arng("smoke", "isolation"))
        lo, hi = S.ci(bs)
        e = float(np.mean(eff))
        out["isolation"] = {"effect": e, "ci": [lo, hi], "n_stimuli": len(eff),
                            "bundle_a6": not (e > 0.02 and lo > 0)}
    else:
        out["isolation"] = {"bundle_a6": False, "note": "no isolation data; A6 goes to separate calls"}
    # cost projection
    c1 = [cost_of(r["response"]) for cid, r in ok.items() if len(by_id[cid]["questions"]) == 1]
    c3 = [cost_of(r["response"]) for cid, r in ok.items() if len(by_id[cid]["questions"]) == 3]
    spent = sum(cost_of(r["response"]) for r in list(finals.values()) + retries if r.get("response"))
    out["cost"] = {"mean_1q": float(np.mean(c1)) if c1 else None, "mean_3q": float(np.mean(c3)) if c3 else None,
                   "smoke_spent": spent}
    if c1:
        per = max(np.mean(c1), np.mean(c3) if c3 else 0)
        out["cost"]["projected_rest_upper"] = float(per * n_flight_calls)
    return out
