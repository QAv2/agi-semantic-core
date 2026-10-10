"""J6b capture suite: the runner end to end on J6b specs with a fake wire (answers built
from the bytes actually sent), the smoke summary and its go rule, key hygiene.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6b_capture.py -q
"""
import hashlib
import json

import pytest

from jev import agents_j6b as A
from jev import analyze_j6b as Z
from jev import j6b_plan as P
from jev import run_j6b as D
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
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=1.0, post_fn=wire(specs, "pointer"), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    assert run.run(specs, concurrency=8) is None
    gz = run.finalize()
    assert KEY not in (tmp_path / "raw.jsonl").read_text()
    finals = R.final_records(gz)
    assert len(finals) == 2112 and all(r["status"] == "ok" for r in finals.values())
    v = Z.analyze([P.strip(s) for s in specs], finals, B=2000, pin=A.PIN,
                  regen={s["call_id"]: s["body_sha"] for s in specs})
    assert v["gates"]["G4_regenerability"]["pass"]
    assert {n: r["branch"] for n, r in v["primaries"].items()} == \
        {"Q1": "HOLDS", "Q2": "HOLDS", "Q3": "MOVES", "Q4": "MOVES"}


def test_smoke_summary_and_go_rule(tmp_path):
    specs = P.smoke()
    run = R.Runner(tmp_path, key=KEY, pin=None, post_fn=wire(specs, "namer", cost=3e-5), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    run.run(specs, concurrency=4)
    s = D.smoke_summary(specs, tmp_path / "raw.jsonl")
    assert s["pin"] == A.PIN and s["same_build_as_J1_J6"] and s["valid_share"] == 1.0
    assert s["determinism"]["flip_rate"] == 0.0 and len(s["determinism"]["groups"]) == 12
    assert s["throughput"]["ok"] == 96 and s["concurrency"] == 16
    assert s["cost"]["projected_flight"] == pytest.approx(2112 * 3e-5) and s["go"]


def test_budget_stops(tmp_path):
    specs = P.flight()[:50]
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=0.1, post_fn=wire(specs, "fixed", cost=0.03),
                   sleep=lambda s: 0, log=lambda *a, **k: None)
    stop = run.run(specs)
    assert stop and stop.startswith("budget") and run.counts["ok"] == 4
