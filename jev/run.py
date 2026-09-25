"""Driver: python -m jev.run {plan,smoke,pilot,flight,analyze} (protocol §4).

  plan     --stage smoke|pilot|flight [--frozen F]   counts + plan sha, no calls
  smoke    [--stamp S]                               ~740 calls; writes smoke_summary.json
  pilot    --smoke DIR [--stamp S]                   ~2,920 calls (+ registered extension); FROZEN.json
  flight   --frozen F [--stamp S]                    J1 then J2; verifies the frozen plan sha
  analyze  --flight DIR --frozen F [--n-perm N]      verdict.json (recompute = run it again, fresh process)

No call is ever made by `plan`. Every calling stage is resumable with --stamp.
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import plan as P
from . import runner as R

BUDGET = 15.0


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def write_plan(out, specs, append=False, h=None):
    """Stripped specs to plan.jsonl.gz (append for a plan written in parts) and
    the plan sha (running hash `h` for parts) to plan_sha.txt."""
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "plan.jsonl.gz", "at" if append else "wt") as fh:
        for s in specs:
            fh.write(json.dumps(P.strip(s), ensure_ascii=False) + "\n")
    sha = P.plan_sha(specs, h)
    (out / "plan_sha.txt").write_text(sha + "\n")
    return sha


def read_plan(d):
    with gzip.open(Path(d) / "plan.jsonl.gz", "rt") as fh:
        return [json.loads(l) for l in fh]


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_np) + "\n")


def _np(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def runner_for(out, pin):
    return R.Runner(out, pin=pin, budget=BUDGET, spent=R.prior_spend(exclude=out))


def cmd_plan(a):
    if a.stage == "smoke":
        specs = P.smoke()
    elif a.stage == "pilot":
        specs = P.j1_pilot()
    else:
        sha, n = P.flight_sha(json.loads(Path(a.frozen).read_text()))
        print(f"flight: {n} calls · plan sha {sha}")
        return
    print(f"{a.stage}: {len(specs)} calls · plan sha {P.plan_sha(specs)}")


def cmd_smoke(a):
    from . import analyze_smoke
    out = R.RESULTS / f"smoke_{a.stamp or stamp()}"
    specs = P.smoke()
    print(f"smoke → {out} · {len(specs)} calls · plan sha {write_plan(out, specs)}", flush=True)
    run = runner_for(out, pin=None)
    det = [s for s in specs if s.get("block") == "determinism"]
    stop = run.run(det, 1, 60, "determinism")
    for conc in (1, 2, 4, 8, 16):
        if stop:
            break
        grp = [s for s in specs if s.get("block") == "throughput" and s.get("concurrency") == conc]
        stop = run.run(grp, conc, 40, f"throughput c={conc}")
    if not stop:
        stop = run.run([s for s in specs if s.get("block") == "isolation"], 4, 100, "isolation")
    run.finalize()
    n_rest = 2920 + 20400 + 30240
    summ = analyze_smoke.analyze(specs, out / "raw.jsonl", n_rest)
    summ["stop_reason"] = stop
    jdump(summ, out / "smoke_summary.json")
    print(json.dumps({k: summ[k] for k in ("status_counts", "builds", "pin", "concurrency", "isolation", "cost")},
                     indent=1, default=_np), flush=True)


def cmd_pilot(a):
    from . import titrate
    from .analyze_j1 import calls_table
    sm = json.loads((Path(a.smoke) / "smoke_summary.json").read_text())
    if not sm.get("pin"):
        sys.exit(f"smoke did not yield a single build string: {sm.get('builds')}")
    out = R.RESULTS / f"pilot_{a.stamp or stamp()}"
    ext_path = out / "extension.json"
    extra = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    specs = P.j1_pilot(extra)
    print(f"pilot → {out} · {len(specs)} calls · pin {sm['pin']} · conc {sm['concurrency']}", flush=True)
    write_plan(out, specs)
    run = runner_for(out, pin=sm["pin"])
    stop = run.run(specs, sm["concurrency"], 250, "pilot")
    if stop:
        sys.exit(f"pilot stopped: {stop}")
    finals = R.final_records(out / "raw.jsonl")
    levels = {f: list(P.GRIDS[f]) + list(extra.get(f, [])) for f in P.GRIDS}
    fams, fits, need = titrate.freeze(calls_table(specs, finals), levels)
    if need and not extra:
        extra = {f: P.extend_levels(f, end, 2) for f, end in need.items()}
        ext_path.write_text(json.dumps(extra, indent=1) + "\n")
        print(f"extension needed {need} → levels {extra}; running them", flush=True)
        specs = P.j1_pilot(extra)
        write_plan(out, specs)
        stop = run.run(specs, sm["concurrency"], 250, "pilot-extension")
        if stop:
            sys.exit(f"pilot extension stopped: {stop}")
        finals = R.final_records(out / "raw.jsonl")
        levels = {f: list(P.GRIDS[f]) + list(extra.get(f, [])) for f in P.GRIDS}
        fams, fits, need = titrate.freeze(calls_table(specs, finals), levels)
    run.finalize()
    frozen = {"pin": sm["pin"], "bundle_a6": sm["isolation"]["bundle_a6"], "concurrency": sm["concurrency"],
              "families": fams, "extension": extra, "unbracketed_after_extension": need,
              "smoke_dir": Path(a.smoke).name, "pilot_dir": out.name}
    frozen["flight_plan_sha"], frozen["n_flight_calls"] = P.flight_sha(frozen)
    jdump(frozen, out / "FROZEN.json")
    jdump(fits, out / "titration_fits.json")
    print(json.dumps({k: frozen[k] for k in ("families", "flight_plan_sha", "n_flight_calls")}, indent=1,
                     default=_np), flush=True)


def cmd_flight(a):
    import hashlib
    fz = json.loads(Path(a.frozen).read_text())
    sha, n = P.flight_sha(fz)
    if sha != fz["flight_plan_sha"]:
        sys.exit(f"plan sha {sha} != FROZEN {fz['flight_plan_sha']}")
    out = R.RESULTS / f"flight_{a.stamp or stamp()}"
    print(f"flight → {out} · {n} calls · sha {sha}", flush=True)
    run = runner_for(out, pin=fz["pin"])
    h, stop = hashlib.sha256(), None
    for n_part, (name, specs) in enumerate(P.flight_parts(fz)):
        write_plan(out, specs, append=n_part > 0, h=h)
        if not stop:
            stop = run.run(specs, fz["concurrency"], 500, name)
        del specs
    run.finalize()
    print(f"flight finished · stop={stop}", flush=True)


def cmd_analyze(a):
    from . import analyze_j1, analyze_j2
    fz = json.loads(Path(a.frozen).read_text())
    d = Path(a.flight)
    specs = read_plan(d)
    if P.plan_sha(specs) != fz["flight_plan_sha"]:
        sys.exit("stored flight plan does not hash to FROZEN's flight_plan_sha")
    raw = d / "raw.jsonl.gz" if (d / "raw.jsonl.gz").exists() else d / "raw.jsonl"
    finals = R.final_records(raw)
    # G4: every recorded call's request regenerates from seeds to the stored sha
    regen = {}
    for _, part in P.flight_parts(fz):
        regen.update({s["call_id"]: s["body_sha"] for s in part})
        del part
    bad = [cid for cid, rec in finals.items() if regen.get(cid) != rec["body_sha"]]
    verdict = {"G4_regenerability": {"n": len(finals), "mismatches": len(bad), "pass": not bad},
               "J1": analyze_j1.analyze(specs, finals, fz, n_perm=a.n_perm),
               "J2": analyze_j2.analyze(specs, finals)}
    jdump(verdict, a.out or d / "verdict.json")
    print(json.dumps({"G4": verdict["G4_regenerability"],
                      "J1": {k: verdict["J1"][k]["verdict"] if "verdict" in verdict["J1"][k] else
                             {f: v["verdict"] for f, v in verdict["J1"][k].items()} for k in ("P1", "P2", "P3")},
                      "J2": {k: verdict["J2"][k]["verdict"] for k in ("P1", "P2", "P3")}}, indent=1), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jev.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--stage", required=True); p.add_argument("--frozen")
    p = sub.add_parser("smoke"); p.add_argument("--stamp")
    p = sub.add_parser("pilot"); p.add_argument("--smoke", required=True); p.add_argument("--stamp")
    p = sub.add_parser("flight"); p.add_argument("--frozen", required=True); p.add_argument("--stamp")
    p = sub.add_parser("analyze"); p.add_argument("--flight", required=True); p.add_argument("--frozen", required=True)
    p.add_argument("--n-perm", type=int, default=2000); p.add_argument("--out")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "smoke": cmd_smoke, "pilot": cmd_pilot, "flight": cmd_flight,
     "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
