"""Driver for J6b: python -m jev.run_j6b {plan,freeze,smoke,flight,analyze}
(docs/JEV_J6B_PROTOCOL.md §2).

  plan                                   counts and plan shas; no calls
  freeze [--stamp S]                     writes FROZEN_J6B.json; no calls
  smoke   --frozen F [--stamp S]         216 Jev calls; smoke_summary.json
  flight  --frozen F --smoke D           2,112 Jev calls
  analyze --flight D --frozen F --smoke D [--B N] [--out P]

G7 is J6b's own: every `j6b_*` stage's spend on disk counts toward $1.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import j6b
from . import j6b_plan as P
from . import runner as R
from .run import _np, jdump, read_plan, write_plan

BUDGET = 1.0
J1J6_BUILD = "typesafe/jev-1.13-20260917"


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def j6b_spent(exclude=None):
    total = 0.0
    exclude = Path(exclude).resolve() if exclude else None
    for p in [f for d in R.RESULTS.glob("j6b_*") if d.is_dir() for f in d.rglob("raw.jsonl*")]:
        if exclude and p.parent.resolve() == exclude:
            continue
        if p.suffix == ".gz" and p.with_suffix("").exists():
            continue
        for rec in R.iter_records(p):
            if rec.get("response") is not None:
                total += R.cost_of(rec["response"])
    return total


def cmd_plan(a):
    fl, sm = P.flight(), P.smoke()
    print(f"flight: {len(fl)} calls · sha {P.plan_sha(fl)}")
    print(f"smoke: {len(sm)} calls · sha {P.plan_sha(sm)}")


def cmd_freeze(a):
    out = R.RESULTS / f"j6b_freeze_{a.stamp or stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    fl, sm = P.flight(), P.smoke()
    it0 = j6b.items()[0]
    fz = {"what": "J6b freeze (docs/JEV_J6B_PROTOCOL.md §2): fixed before any J6b call",
          "options": list(j6b.OPTIONS), "question": j6b.QUESTION,
          "pages_item0": {s: j6b.page(it0, s) for s in j6b.STATES},
          "flight": {"plan_sha": P.plan_sha(fl), "n_calls": len(fl)},
          "smoke": {"plan_sha": P.plan_sha(sm), "n_calls": len(sm)}, "budget_usd": BUDGET}
    jdump(fz, out / "FROZEN_J6B.json")
    print(f"frozen → {out / 'FROZEN_J6B.json'}")
    print(json.dumps({k: fz[k] for k in ("flight", "smoke")}, indent=1))


def smoke_summary(specs, raw):
    by_id = {s["call_id"]: s for s in specs}
    finals, retries = {}, []
    for rec in R.iter_records(raw):
        (retries.append(rec) if rec["status"] == "retry" else finals.__setitem__(rec["call_id"], rec))
    ok = {c: r for c, r in finals.items() if r["status"] == "ok"}
    out = {"status_counts": {k: sum(r["status"] == k for r in finals.values())
                             for k in ("ok", "invalid", "error", "build_mismatch")}, "n_plan": len(specs)}
    builds = sorted({r["response"].get("model") for r in ok.values()})
    out["builds"], out["pin"] = builds, (builds[0] if len(builds) == 1 else None)
    out["same_build_as_J1_J6"] = out["pin"] == J1J6_BUILD
    out["valid_share"] = len(ok) / len(specs)
    groups = {}
    for cid, r in ok.items():
        if by_id[cid]["block"] == "determinism":
            dec, k2i = r["response"]["answers"]["decision"], by_id[cid]["key_to_idx"]
            p = np.zeros(j6b.K)
            for key, v in dec["probabilities"].items():
                p[k2i[key]] = v
            groups.setdefault(re.sub(r"-x\d+$", "", cid), []).append((k2i[dec["choice"]], p))
    det = {}
    for g, v in sorted(groups.items()):
        ch, Pm = [x[0] for x in v], np.array([x[1] for x in v])
        det[g] = {"n": len(v), "flip_rate": 1 - max(ch.count(c) for c in set(ch)) / len(ch),
                  "max_abs_dp": float(np.abs(Pm - Pm[0]).max())}
    out["determinism"] = {"groups": det, "flip_rate": float(np.mean([d["flip_rate"] for d in det.values()])),
                          "max_abs_dp": float(max(d["max_abs_dp"] for d in det.values()))} if det else {}
    ids = {s["call_id"] for s in specs if s["block"] == "throughput"}
    lat = [finals[c]["latency_s"] for c in ids if c in ok]
    n429 = sum(1 for r in retries if r["call_id"] in ids and r.get("http") == 429)
    p95 = float(np.percentile(lat, 95)) if lat else None
    out["throughput"] = {"n": len(ids), "ok": len(lat), "n_429": n429, "p95_latency_s": p95,
                         "median_latency_s": float(np.median(lat)) if lat else None}
    out["concurrency"] = 16 if (n429 == 0 and len(lat) == len(ids) and p95 is not None and p95 < 2.0) else 8
    per = [R.cost_of(r["response"]) for r in ok.values()]
    spent = sum(R.cost_of(r["response"]) for r in list(finals.values()) + retries if r.get("response"))
    out["cost"] = {"per_call": float(np.mean(per)) if per else None, "smoke_spent": spent}
    if per:
        out["cost"]["projected_flight"] = float(np.mean(per) * len(P.flight()))
        out["cost"]["projected_total"] = spent + out["cost"]["projected_flight"]
    out["go"] = bool(out["pin"] and out["valid_share"] >= 0.99 and per and out["cost"]["projected_total"] <= BUDGET)
    return out


def cmd_smoke(a):
    fz = json.loads(Path(a.frozen).read_text())
    out = R.RESULTS / f"j6b_smoke_{a.stamp or stamp()}"
    specs = P.smoke()
    if P.plan_sha(specs) != fz["smoke"]["plan_sha"]:
        sys.exit("smoke plan does not hash to FROZEN_J6B's smoke sha")
    print(f"smoke → {out} · {len(specs)} calls", flush=True)
    write_plan(out, specs)
    run = R.Runner(out, pin=None, budget=BUDGET, spent=j6b_spent(exclude=out))
    stop = run.run([s for s in specs if s["block"] == "determinism"], 1, 40, "determinism")
    if not stop:
        stop = run.run([s for s in specs if s["block"] == "throughput"], 16, 48, "throughput c=16")
    run.finalize()
    summ = smoke_summary(specs, out / "raw.jsonl")
    summ["stop_reason"] = stop
    jdump(summ, out / "smoke_summary.json")
    print(json.dumps({k: v for k, v in summ.items() if k != "determinism"} |
                     {"determinism": {k: summ["determinism"].get(k) for k in ("flip_rate", "max_abs_dp")}},
                     indent=1, default=_np), flush=True)


def cmd_flight(a):
    fz = json.loads(Path(a.frozen).read_text())
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    if not sm.get("go"):
        sys.exit(f"smoke go rule not met: pin={sm.get('pin')} valid={sm.get('valid_share')} cost={sm.get('cost')}")
    specs = P.flight()
    if P.plan_sha(specs) != fz["flight"]["plan_sha"] or len(specs) != fz["flight"]["n_calls"]:
        sys.exit("flight plan does not hash to FROZEN_J6B")
    out = R.RESULTS / f"j6b_flight_{a.stamp or stamp()}"
    print(f"flight → {out} · {len(specs)} calls · pin {sm['pin']} · conc {sm['concurrency']}", flush=True)
    write_plan(out, specs)
    run = R.Runner(out, pin=sm["pin"], budget=BUDGET, spent=j6b_spent(exclude=out))
    stop = run.run(specs, sm["concurrency"], 264, "flight")
    run.finalize()
    print(f"flight finished · stop={stop} · spent ${run.spent:.4f}", flush=True)


def cmd_analyze(a):
    from . import analyze_j6b
    fz = json.loads(Path(a.frozen).read_text())
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    d = Path(a.flight)
    specs = read_plan(d)
    if P.plan_sha(specs) != fz["flight"]["plan_sha"]:
        sys.exit("stored flight plan does not hash to FROZEN_J6B's plan sha")
    raw = d / "raw.jsonl.gz" if (d / "raw.jsonl.gz").exists() else d / "raw.jsonl"
    finals = R.final_records(raw)
    regen = {s["call_id"]: s["body_sha"] for s in P.flight()}
    v = analyze_j6b.analyze(specs, finals, B=a.B, pin=sm["pin"], regen=regen, smoke=sm)
    v["gates"]["same_build_as_J1_J6"] = sm["pin"] == J1J6_BUILD
    v["spend"] = {"j6b_total": j6b_spent()}
    v["inputs"] = {"raw_sha256": hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
                   "frozen_flight_plan_sha": fz["flight"]["plan_sha"]}
    jdump(v, a.out or d / "verdict.json")
    print(json.dumps(analyze_j6b.summary(v), indent=1, default=_np), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jev.run_j6b")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("freeze"); p.add_argument("--stamp")
    p = sub.add_parser("smoke"); p.add_argument("--frozen", required=True); p.add_argument("--stamp")
    p = sub.add_parser("flight"); p.add_argument("--frozen", required=True); p.add_argument("--smoke", required=True)
    p.add_argument("--stamp")
    p = sub.add_parser("analyze"); p.add_argument("--flight", required=True); p.add_argument("--frozen", required=True)
    p.add_argument("--smoke", required=True); p.add_argument("--B", type=int, default=10_000); p.add_argument("--out")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "freeze": cmd_freeze, "smoke": cmd_smoke, "flight": cmd_flight,
     "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
