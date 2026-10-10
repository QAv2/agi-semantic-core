"""J5 capture suite: the runner end to end on J5 specs with a fake wire (answers built
from the bytes actually sent), the smoke summary and its go rule, key hygiene.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j5_capture.py -q
"""
import hashlib
import json

import pytest

from jev import agents_j5 as A
from jev import analyze_j5 as Z
from jev import j5_plan as P
from jev import run_j5 as D
from jev import runner as R

KEY = "test-key-not-a-real-secret-0123456789"


def wire(specs, kind, model=A.PIN, cost=1e-5):
    by_sha = {s["body_sha"]: s for s in specs}

    def post(body, key, url):
        assert key == KEY
        r = A.respond(by_sha[hashlib.sha256(body).hexdigest()], kind)
        r["model"], r["usage"]["cost"] = model, cost
        return 200, json.loads(json.dumps(r)), 0.01, None
    return post


def test_flight_end_to_end(tmp_path):
    specs = P.flight()
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=1.0, post_fn=wire(specs, "people"), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    assert run.run(specs, concurrency=8) is None
    gz = run.finalize()
    assert KEY not in (tmp_path / "raw.jsonl").read_text()
    finals = R.final_records(gz)
    assert len(finals) == 3312 and all(r["status"] == "ok" for r in finals.values())
    v = Z.analyze([P.strip(s) for s in specs], finals, B=1000, pin=A.PIN,
                  regen={s["call_id"]: s["body_sha"] for s in specs})
    assert v["gates"]["G4_regenerability"]["pass"] and v["gates"]["G5_pass"]
    assert {n: v["primaries"][n]["branch"] for n in ("C1", "C2")} == \
        {"C1": "WITHIN THE HUMAN RANGE", "C2": "NO LEAN"}
    # no human cell value is written into the verdict: Jev's tables only, human stats derived
    assert set(v["tables"]) == {"YN", "CH", "colors"}


def test_smoke_summary_and_go_rule(tmp_path):
    specs = P.smoke()
    run = R.Runner(tmp_path, key=KEY, pin=None, post_fn=wire(specs, "people", cost=3e-5), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    run.run(specs, concurrency=4)
    s = D.smoke_summary(specs, tmp_path / "raw.jsonl")
    assert s["pin"] == A.PIN and s["same_build_as_J1_J6b"] and s["valid_share"] == 1.0
    assert len(s["determinism"]["groups"]) == 12 and set(s["determinism"]) >= {"YN", "CH"}
    assert s["determinism"]["YN"]["max_abs_dp"] == 0.0 and s["determinism"]["CH"]["flip_rate"] == 0.0
    assert s["throughput"]["ok"] == 96 and s["concurrency"] == 16
    assert s["cost"]["projected_flight"] == pytest.approx(3312 * 3e-5) and s["go"]


def test_budget_stops(tmp_path):
    specs = P.flight()[:50]
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=0.1, post_fn=wire(specs, "fixed", cost=0.03),
                   sleep=lambda s: 0, log=lambda *a, **k: None)
    stop = run.run(specs)
    assert stop and stop.startswith("budget") and run.counts["ok"] == 4
