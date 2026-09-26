"""Driver for J3: python -m jev.run_j3 {plan,freeze,smoke,flight,mouth,analyze}
(docs/JEV_J3_PROTOCOL.md §7).

  plan                                   counts and plan shas; no calls
  freeze                                 fits the bar, writes FROZEN_J3.json; no calls
  smoke   --frozen F [--stamp S]         ~640 Jev + 40 mouth calls; smoke_summary.json
  flight  --frozen F --smoke D           18,576 Jev calls (bundled) or 37,152 (separate)
  mouth   --frozen F --smoke D           1,548 mouth calls
  analyze --flight D --frozen F --smoke D [--mouth D] [--B N] [--out P]

G7 is J3's own: every J3 stage's spend on disk counts toward $5; J1 + J2 spend doesn't.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import j3
from . import j3_mouth as M
from . import j3_plan as P
from . import runner as R
from . import stats as S
from .run import _np, jdump, read_plan, write_plan

BUDGET = 5.0


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def j3_spent(exclude=None):
    total = 0.0
    exclude = Path(exclude).resolve() if exclude else None
    files = [f for d in R.RESULTS.glob("j3_*") if d.is_dir() for f in d.rglob("raw.jsonl*")]
    for p in files:
        if exclude and p.parent.resolve() == exclude:
            continue
        if p.suffix == ".gz" and p.with_suffix("").exists():
            continue
        for rec in R.iter_records(p):
            if rec.get("response") is not None:
                total += R.cost_of(rec["response"])
    return total


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── plan / freeze ────────────────────────────────────────────────────────────

def cmd_plan(a):
    stims = j3.flight_stimuli()
    for b in (True, False):
        sha, n = P.flight_sha(b, stims)
        print(f"flight ({'bundled' if b else 'separate'}): {n} calls · sha {sha}")
    print(f"mouth: {len(P.mouth_flight(stims))} calls · sha {P.plan_sha(P.mouth_flight(stims))}")
    print(f"smoke: {len(P.smoke())} Jev + {len(P.mouth_smoke())} mouth calls")


def cmd_freeze(a):
    out = R.RESULTS / f"j3_freeze_{a.stamp or stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    stims = j3.flight_stimuli()
    fz = {"what": "J3 freeze (docs/JEV_J3_PROTOCOL.md §7): fixed before any J3 call",
          "inputs": {p.name: file_sha(p) for p in (j3.F_PAY, j3.F_E7, j3.F_N2)},
          "n_states": int(len(j3.data()["U"])), "halves": np.bincount(j3.data()["half"]).tolist(),
          "cards": {str(h): j3.reference(h)["card"] for h in (0, 1)},
          "bars": {str(h): j3.fit_bar(h) for h in (0, 1)}}
    for b, name in ((True, "bundled"), (False, "separate")):
        sha, n = P.flight_sha(b, stims)
        fz[f"flight_{name}"] = {"plan_sha": sha, "n_calls": n}
    ms = P.mouth_flight(stims)
    fz["mouth"] = {"model": P.MOUTH_MODEL, "plan_sha": P.plan_sha(ms), "n_calls": len(ms)}
    fz["smoke"] = {"plan_sha": P.plan_sha(P.smoke()), "mouth_plan_sha": P.plan_sha(P.mouth_smoke())}
    fz["stimuli_sha"] = hashlib.sha256(json.dumps(stims, sort_keys=True).encode()).hexdigest()
    jdump(fz, out / "FROZEN_J3.json")
    print(f"frozen → {out / 'FROZEN_J3.json'}")
    print(json.dumps({k: fz[k] for k in ("cards", "flight_bundled", "flight_separate", "mouth")}, indent=1))


# ── smoke (§7.1) ─────────────────────────────────────────────────────────────

def _decision(rec, spec):
    dec = rec["response"]["answers"]["decision"]
    k2i = spec["key_to_idx"]
    probs = np.zeros(j3.NONE + 1)
    for key, p in dec["probabilities"].items():
        probs[k2i[key]] = p
    ci = k2i[dec["choice"]]
    return ci, float(probs[ci]), probs


def smoke_summary(specs, raw, mspecs, mraw):
    by_id = {s["call_id"]: s for s in specs}
    finals, retries = {}, []
    for rec in R.iter_records(raw):
        (retries.append(rec) if rec["status"] == "retry" else finals.__setitem__(rec["call_id"], rec))
    ok = {c: r for c, r in finals.items() if r["status"] == "ok"}
    out = {"status_counts": {k: sum(r["status"] == k for r in finals.values())
                             for k in ("ok", "invalid", "error", "build_mismatch")}}
    builds = sorted({r["response"].get("model") for r in ok.values()})
    out["builds"], out["pin"] = builds, (builds[0] if len(builds) == 1 else None)
    out["same_build_as_J1J2"] = out["pin"] == "typesafe/jev-1.13-20260917"
    groups = {}
    for cid, r in ok.items():
        if by_id[cid]["block"] == "determinism":
            groups.setdefault(re.sub(r"-x\d+$", "", cid), []).append(_decision(r, by_id[cid]))
    det = {}
    for g, v in sorted(groups.items()):
        ch = [x[0] for x in v]
        Pm = np.array([x[2] for x in v])
        pairs = [(i, k) for i in range(len(v)) for k in range(i + 1, len(v))]
        det[g] = {"n": len(v), "flip_rate": 1 - max(ch.count(c) for c in set(ch)) / len(ch),
                  "mean_abs_dp": float(np.mean([np.abs(Pm[i] - Pm[k]).mean() for i, k in pairs])),
                  "mean_tv": float(np.mean([S.tv(Pm[i], Pm[k]) for i, k in pairs]))}
    out["determinism"] = {"groups": det, **{m: float(np.mean([d[m] for d in det.values()]))
                                            for m in ("flip_rate", "mean_abs_dp", "mean_tv")}} if det else {}
    ids = {s["call_id"] for s in specs if s["block"] == "throughput"}
    lat = [finals[c]["latency_s"] for c in ids if c in ok]
    n429 = sum(1 for r in retries if r["call_id"] in ids and r.get("http") == 429)
    p95 = float(np.percentile(lat, 95)) if lat else None
    out["throughput"] = {"n": len(ids), "ok": len(lat), "n_429": n429, "p95_latency_s": p95,
                         "median_latency_s": float(np.median(lat)) if lat else None}
    out["concurrency"] = 16 if (n429 == 0 and len(lat) == len(ids) and p95 is not None and p95 < 2.0) else 8
    iso = {}
    for cid, r in ok.items():
        sp = by_id[cid]
        if sp["block"] == "isolation":
            key = re.sub(r"-(decision|bundled)-x\d+$", "", cid)
            iso.setdefault(key, {"decision": [], "bundled": []})[sp["mode"]].append(_decision(r, sp)[1])
    eff = []
    for v in iso.values():
        a, b = v["decision"], v["bundled"]
        if len(a) >= 2 and len(b) >= 2:
            between = np.mean([abs(x - y) for x in a for y in b])
            within = [abs(x[i] - x[k]) for x in (a, b) for i in range(len(x)) for k in range(i + 1, len(x))]
            eff.append(between - np.mean(within))
    if eff:
        bs = S.boot_mean_strat(eff, np.zeros(len(eff)), 10_000, np.random.default_rng([20261002, 9]))
        lo, hi = S.ci(bs)
        e = float(np.mean(eff))
        out["isolation"] = {"effect": e, "ci": [lo, hi], "n_stimuli": len(eff), "bundle_pushed": not (e > 0.02 and lo > 0)}
    else:
        out["isolation"] = {"bundle_pushed": False, "note": "no isolation data: `pushed` goes to separate calls"}
    per = {}
    for pres in P.PRES:
        c = [R.cost_of(r["response"]) for cid, r in ok.items() if by_id[cid]["pres"] == pres
             and by_id[cid]["mode"] == "bundled"]
        per[pres] = float(np.mean(c)) if c else None
    spent = sum(R.cost_of(r["response"]) for r in list(finals.values()) + retries if r.get("response"))
    out["cost"] = {"per_call": per, "smoke_spent_jev": spent}
    if all(per.values()):
        n_fl = 18576 if out["isolation"]["bundle_pushed"] else 37152
        out["cost"]["projected_flight"] = float(np.mean(list(per.values())) * n_fl)
    # the mouth
    mby = {s["call_id"]: s for s in mspecs}
    mf = R.final_records(mraw) if Path(mraw).exists() else {}
    mok = {c: r for c, r in mf.items() if r["status"] == "ok"}
    first = [c for c in mby if c.endswith("-x0")]
    parsed = sum(1 for c in first if c in mok)
    rtok = [M.reasoning_tokens(r["response"]) + (1 if M.has_reasoning_text(r["response"]) else 0)
            for r in mok.values()]
    reps = {}
    for c, r in mok.items():
        reps.setdefault(re.sub(r"-x\d+$", "", c), []).append(M.content_of(r["response"]))
    agree = [len(set(v)) == 1 for v in reps.values() if len(v) >= 2]
    models = sorted({r["response"].get("model") for r in mok.values()})
    mspent = sum(R.cost_of(r["response"]) for r in R.iter_records(mraw) if r.get("response")) if Path(mraw).exists() else 0
    out["mouth"] = {"parsed_first_sends": parsed, "n_first_sends": len(first), "reasoning_tokens_max": max(rtok) if rtok else None,
                    "identical_repeat_share": float(np.mean(agree)) if agree else None, "models": models,
                    "pin": models[0] if len(models) == 1 else None,
                    "providers": sorted({r["response"].get("provider") for r in mok.values()}),
                    "fly": parsed >= 19 and (max(rtok) if rtok else 1) == 0, "spent": mspent,
                    "mean_cost": float(np.mean([R.cost_of(r["response"]) for r in mok.values()])) if mok else None}
    return out


def cmd_smoke(a):
    fz = json.loads(Path(a.frozen).read_text())
    out = R.RESULTS / f"j3_smoke_{a.stamp or stamp()}"
    specs, mspecs = P.smoke(), P.mouth_smoke()
    if P.plan_sha(specs) != fz["smoke"]["plan_sha"] or P.plan_sha(mspecs) != fz["smoke"]["mouth_plan_sha"]:
        sys.exit("smoke plan does not hash to FROZEN_J3's smoke shas")
    print(f"smoke → {out} · {len(specs)} Jev + {len(mspecs)} mouth calls", flush=True)
    write_plan(out, specs)
    run = R.Runner(out, pin=None, budget=BUDGET, spent=j3_spent(exclude=out))
    stop = run.run([s for s in specs if s["block"] == "determinism"], 1, 60, "determinism")
    if not stop:
        stop = run.run([s for s in specs if s["block"] == "throughput"], 16, 50, "throughput c=16")
    if not stop:
        stop = run.run([s for s in specs if s["block"] == "isolation"], 4, 100, "isolation")
    run.finalize()
    mdir = out / "mouth"
    with __import__("gzip").open(out / "mouth_plan.jsonl.gz", "wt") as fh:
        for s in mspecs:
            fh.write(json.dumps(P.strip(s)) + "\n")
    mrun = M.MouthRunner(mdir, pin=None, budget=BUDGET, spent=j3_spent(exclude=mdir))
    if not stop:
        stop = mrun.run(mspecs, 4, 20, "mouth smoke")
    mrun.finalize()
    summ = smoke_summary(specs, out / "raw.jsonl", mspecs, mdir / "raw.jsonl")
    summ["stop_reason"] = stop
    jdump(summ, out / "smoke_summary.json")
    print(json.dumps({k: summ[k] for k in ("status_counts", "builds", "same_build_as_J1J2", "determinism",
                                           "throughput", "concurrency", "isolation", "cost", "mouth")
                      if k in summ}, indent=1, default=_np)[:4000], flush=True)


# ── flight / mouth ───────────────────────────────────────────────────────────

def cmd_flight(a):
    fz = json.loads(Path(a.frozen).read_text())
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    if not sm.get("pin"):
        sys.exit(f"smoke did not yield a single build string: {sm.get('builds')}")
    bundled = sm["isolation"]["bundle_pushed"]
    reg = fz["flight_bundled" if bundled else "flight_separate"]
    out = R.RESULTS / f"j3_flight_{a.stamp or stamp()}"
    print(f"flight → {out} · {reg['n_calls']} calls · bundled={bundled} · pin {sm['pin']} · "
          f"conc {sm['concurrency']}", flush=True)
    stims = j3.flight_stimuli()
    sha, n = P.flight_sha(bundled, stims)
    if sha != reg["plan_sha"] or n != reg["n_calls"]:
        sys.exit(f"flight plan sha {sha} ({n}) != FROZEN {reg['plan_sha']} ({reg['n_calls']})")
    run = R.Runner(out, pin=sm["pin"], budget=BUDGET, spent=j3_spent(exclude=out))
    h, stop = hashlib.sha256(), None
    for n_part, (name, specs) in enumerate(P.flight_parts(bundled, stims)):
        write_plan(out, specs, append=n_part > 0, h=h)
        if not stop:
            stop = run.run(specs, sm["concurrency"], 500, name)
    run.finalize()
    print(f"flight finished · stop={stop} · spent ${run.spent:.4f}", flush=True)


def cmd_mouth(a):
    fz = json.loads(Path(a.frozen).read_text())
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    if not sm["mouth"]["fly"]:
        sys.exit(f"the mouth smoke did not clear its bar: {sm['mouth']}")
    specs = P.mouth_flight()
    if P.plan_sha(specs) != fz["mouth"]["plan_sha"]:
        sys.exit("mouth plan does not hash to FROZEN_J3")
    out = R.RESULTS / f"j3_mouth_{a.stamp or stamp()}"
    print(f"mouth → {out} · {len(specs)} calls · pin {sm['mouth']['pin']}", flush=True)
    write_plan(out, specs)
    run = M.MouthRunner(out, pin=sm["mouth"]["pin"], budget=BUDGET, spent=j3_spent(exclude=out))
    stop = run.run(specs, 8, 200, "mouth")
    run.finalize()
    print(f"mouth finished · stop={stop} · spent ${run.spent:.4f}", flush=True)


# ── analyze ──────────────────────────────────────────────────────────────────

def cmd_analyze(a):
    from . import analyze_j3
    fz = json.loads(Path(a.frozen).read_text())
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    bundled = sm["isolation"]["bundle_pushed"]
    reg = fz["flight_bundled" if bundled else "flight_separate"]
    d = Path(a.flight)
    specs = read_plan(d)
    if P.plan_sha(specs) != reg["plan_sha"]:
        sys.exit("stored flight plan does not hash to FROZEN_J3's plan sha")
    raw = d / "raw.jsonl.gz" if (d / "raw.jsonl.gz").exists() else d / "raw.jsonl"
    finals = R.final_records(raw)
    regen = {}
    for _, part in P.flight_parts(bundled):
        regen.update({s["call_id"]: s["body_sha"] for s in part})
    bad = [c for c, rec in finals.items() if regen.get(c) != rec["body_sha"]]
    mouth = None
    if a.mouth:
        md = Path(a.mouth)
        mspecs = read_plan(md)
        if P.plan_sha(mspecs) != fz["mouth"]["plan_sha"]:
            sys.exit("stored mouth plan does not hash to FROZEN_J3")
        mraw = md / "raw.jsonl.gz" if (md / "raw.jsonl.gz").exists() else md / "raw.jsonl"
        mouth = (mspecs, R.final_records(mraw))
    v = analyze_j3.analyze(specs, finals, fz, B=a.B, mouth=mouth, pin=sm["pin"])
    v["gates"]["G4_regenerability"] = {"n": len(finals), "mismatches": len(bad), "pass": not bad}
    v["spend"] = {"j3_total": j3_spent()}
    jdump(v, a.out or d / "verdict.json")
    print(json.dumps(analyze_j3.summary(v), indent=1, default=_np), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jev.run_j3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("freeze"); p.add_argument("--stamp")
    p = sub.add_parser("smoke"); p.add_argument("--frozen", required=True); p.add_argument("--stamp")
    for name in ("flight", "mouth"):
        p = sub.add_parser(name); p.add_argument("--frozen", required=True); p.add_argument("--smoke", required=True)
        p.add_argument("--stamp")
    p = sub.add_parser("analyze"); p.add_argument("--flight", required=True); p.add_argument("--frozen", required=True)
    p.add_argument("--smoke", required=True); p.add_argument("--mouth"); p.add_argument("--B", type=int, default=10_000)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "freeze": cmd_freeze, "smoke": cmd_smoke, "flight": cmd_flight, "mouth": cmd_mouth,
     "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
