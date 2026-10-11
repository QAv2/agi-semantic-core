"""W1 driver: freeze, smoke, flight (protocol §4). Resumable; every stage reads its own log first.

    python -m windows.run_w1 freeze
    python -m windows.run_w1 run smoke  [--dir windows/results/w1_smoke_<stamp>]
    python -m windows.run_w1 run flight [--dir windows/results/w1_flight_<stamp>]

A stage directory holds H/, D/ and R/, each with raw.jsonl (gitignored; .gz committed) and,
for D and R, the derived plan written with its sha BEFORE that sub-stage's first call.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import w1 as W
from . import w1_stimuli as S
from .w1_client import (RESULTS, Runner, final_records, post, prior_spend, stage_raw, text_of)

ROOT = Path(__file__).resolve().parent
CODE = ["w1.py", "w1_client.py", "w1_stimuli.py", "run_w1.py", "analyze_w1.py",
        "gemini_ui.py", "run_w1_ui.py"]


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runner(sub_dir, post_fn=post, sleep=time.sleep, key=None, log=print):
    """G7 counts every other w1_* stage beside this one (results dir = the stage dir's parent)."""
    sub_dir = Path(sub_dir)
    spent = prior_spend("w1_", results=sub_dir.parent.parent, exclude=sub_dir)
    return Runner(sub_dir, pins=W.PINS, budget=W.BUDGET, spent=spent,
                  key=key, post_fn=post_fn, sleep=sleep, log=log)


# ---- H ------------------------------------------------------------------------------------

def run_H(stage, out, **kw):
    r = runner(Path(out) / "H", **kw)

    def drive(conv):
        replies = []
        for k in range(1, len(conv["turns"]) + 1):
            rec = r.one(W.turn_spec(stage, conv, k, replies))
            if not rec or rec["status"] != "ok":
                return
            replies.append(text_of(rec["response"]))

    with ThreadPoolExecutor(W.CONCURRENCY[W.MODEL]) as ex:
        list(ex.map(drive, W.conversations(stage)))
        items = list(W.c_items(stage))
        list(ex.map(r.one, [W.attempt_spec(stage, it, k) for it in items for k in range(1, 5)]))
        if not r.stop_reason and W.needs_second_round(stage, r.final):
            outc = W.c_outcomes(stage, r.final, max_k=4)
            todo = [it for it in items if not W.c_mixed(outc[it["item"]])]
            list(ex.map(r.one, [W.attempt_spec(stage, it, k) for it in todo for k in range(5, 9)]))
    r.progress(f"{stage}/H")
    return r


# ---- D ------------------------------------------------------------------------------------

def run_D(stage, out, **kw):
    hfinal = final_records(stage_raw(Path(out) / "H"))
    hist = W.histories(stage, hfinal)
    specs = W.draw_specs(stage, hist)
    d = Path(out) / "D"
    d.mkdir(parents=True, exist_ok=True)
    plan = {"stage": stage, "call_ids": [s["call_id"] for s in specs],
            "body_shas": {s["call_id"]: s["body_sha"] for s in specs},
            "c_selection": [{k: v for k, v in s.items() if k in ("item", "tier", "product")} |
                            {"X_attempt": s["X"][0], "N_attempt": s["N"][0]}
                            for s in W.c_select(stage, hfinal)]}
    plan["sha"] = sha_json(plan)
    _write_plan(d / "plan_D.json", plan)
    r = runner(d, **kw)
    with ThreadPoolExecutor(W.CONCURRENCY[W.MODEL]) as ex:
        list(ex.map(r.one, specs))
    r.progress(f"{stage}/D")
    return r


# ---- R ------------------------------------------------------------------------------------

def run_R(stage, out, **kw):
    dd = Path(out) / "D"
    plan_d = json.loads((dd / "plan_D.json").read_text())
    dfinal = final_records(stage_raw(dd))
    rd = Path(out) / "R"
    (rd / "jpg").mkdir(parents=True, exist_ok=True)
    jshas = {}

    def jpeg_of(cid):
        p = rd / "jpg" / f"{cid}.jpg"
        if not p.exists():
            path, _ = W.drawing(dfinal[cid])
            p.write_bytes(W.to_jpeg((dd / path).read_bytes()))
        b = p.read_bytes()
        jshas[cid] = hashlib.sha256(b).hexdigest()
        return b

    specs, key = W.read_plan(stage, plan_d["call_ids"], dfinal, jpeg_of)
    plan = {"stage": stage, "plan_D_sha": plan_d["sha"], "call_ids": [s["call_id"] for s in specs],
            "body_shas": {s["call_id"]: s["body_sha"] for s in specs}, "key": key,
            "jpeg_sha256": dict(sorted(jshas.items())),
            "png_sha256": {cid: next(f["sha256"] for f in dfinal[cid]["files"]
                                     if f["path"] == W.drawing(dfinal[cid])[0])
                           for cid in sorted(jshas)}}
    plan["sha"] = sha_json(plan)
    _write_plan(rd / "plan_R.json", plan)
    r = runner(rd, **kw)
    with ThreadPoolExecutor(W.CONCURRENCY[W.READER]) as ex:
        list(ex.map(r.one, specs))
    r.progress(f"{stage}/R")
    return r


def _write_plan(path, plan):
    """A derived plan is written once; a resumed stage must regenerate the identical plan."""
    if path.exists():
        old = json.loads(path.read_text())
        if old["sha"] != plan["sha"]:
            raise RuntimeError(f"{path} regenerated with a different sha ({old['sha'][:12]} != {plan['sha'][:12]})")
        return
    path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")


def run_all(stage, out, **kw):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (run_H, run_D, run_R):
        r = fn(stage, out, **kw)
        if r.stop_reason:
            return r.stop_reason
    return None


# ---- freeze -------------------------------------------------------------------------------

def freeze(out=None):
    import numpy
    import PIL
    import scipy
    out = Path(out or RESULTS / f"w1_freeze_{stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    reg = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/W1_DRAW_YOUR_STATE_PROTOCOL.md"],
                         cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    h_specs = ([W.turn_spec("flight", c, 1, []) for c in W.conversations("flight")] +
               [W.attempt_spec("flight", it, k) for it in W.c_items("flight") for k in range(1, 9)])
    frozen = {
        "registration_commit": reg,
        "code_sha256": {f: sha_file(ROOT / f) for f in CODE},
        "tests_sha256": {p.name: sha_file(p) for p in sorted((ROOT / "tests").glob("test_w1_*.py"))}
                        | {"agents_w1.py": sha_file(ROOT / "agents_w1.py")},
        "route": {"under_test": "gemini-app/3.8-flash/low (Amendment 2)", "driver": "windows/gemini_ui.py",
                  "perception": "Firefox accessibility tree (AT-SPI), Amendment 3",
                  "drawing_turns": "Create image tool on every drawing turn, every arm (Amendment 3)",
                  "c_pilot_seed": S.C_UI_PILOT_SEED, "c_ui_pool": S.C_UI_POOL, "c_ui_attempts": S.C_UI_ATTEMPTS},
        "models": {"under_test": W.MODEL, "reader": W.READER,
                   "pins": {k: sorted(v) for k, v in W.PINS.items()}},
        "seeds": W.SEEDS | {"analysis": W.ANALYSIS_SEED, "c_pool": S.C_POOL_SEED, "c_smoke": S.C_SMOKE_SEED},
        "budget_usd": W.BUDGET,
        "flight_H_static_sha": sha_json([[s["call_id"], s["body_sha"]] for s in h_specs]),
        "flight_H_static_calls": len(h_specs),
        "prompts_sha": sha_json([S.DRAW_SELF, S.DRAW_GENRE, S.OBSERVER_HEAD, S.OBSERVER_TAIL,
                                 W.PAIR_HEAD, W.PAIR_DIFF, W.PAIR_Q, W.ANSWER_FORMAT, W.CAP_TEXT,
                                 W.GEN_TEXT, W.INV_TEXT, S.CONTRAST_TEXT]),
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                     "scipy": scipy.__version__, "pillow": PIL.__version__},
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "FROZEN_W1.json").write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return out / "FROZEN_W1.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("run")
    p.add_argument("stage", choices=["smoke", "flight"])
    p.add_argument("--dir")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        print(freeze())
        return 0
    out = Path(a.dir) if a.dir else RESULTS / f"w1_{a.stage}_{stamp()}"
    print("stage dir:", out, flush=True)
    stop = run_all(a.stage, out)
    print("STOP:" if stop else "done", stop or "", flush=True)
    return 1 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
