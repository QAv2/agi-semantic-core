"""J6 capture suite: the runner end to end on J6 specs with a fake wire (answers
built from the bytes actually sent), the smoke summary and its go rule, the
build pin, the budget stop and key hygiene.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j6_capture.py -q
"""
import hashlib
import json

import pytest

from jev import agents_j6 as A
from jev import analyze_j6 as Z
from jev import j6_plan as P
from jev import run_j6 as D
from jev import runner as R

KEY = "test-key-not-a-real-secret-0123456789"


def wire(specs, kind, model=A.PIN, cost=1e-5):
    """A fake endpoint: identifies the call by the sha256 of the bytes sent."""
    by_sha = {s["body_sha"]: s for s in specs}

    def post(body, key, url):
        assert key == KEY
        s = by_sha[hashlib.sha256(body).hexdigest()]
        r = A.respond(s, kind)
        r["model"], r["usage"]["cost"] = model, cost
        return 200, json.loads(json.dumps(r)), 0.01, None
    return post


def test_flight_end_to_end(tmp_path):
    specs = P.flight()
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=1.0, post_fn=wire(specs, "reader"), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    assert run.run(specs, concurrency=8) is None
    gz = run.finalize()
    assert KEY not in (tmp_path / "raw.jsonl").read_text()
    finals = R.final_records(gz)
    assert len(finals) == 3456 and all(r["status"] == "ok" for r in finals.values())
    v = Z.analyze([P.strip(s) for s in specs], finals, B=2000, pin=A.PIN,
                  regen={s["call_id"]: s["body_sha"] for s in specs})
    assert v["gates"]["G4_regenerability"]["pass"]
    assert {n: r["branch"] for n, r in v["primaries"].items()} == \
        {"P1": "HOLDS", "P2": "MOVES", "P3": "MOVES", "P4": "MOVES"}


def test_build_pin_stops(tmp_path):
    specs = P.flight()[:20]
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, post_fn=wire(specs, "fixed", model="typesafe/jev-9"),
                   sleep=lambda s: 0, log=lambda *a, **k: None)
    stop = run.run(specs)
    assert stop and stop.startswith("G1 build mismatch") and run.counts["build_mismatch"] == 1


def test_budget_stops(tmp_path):
    specs = P.flight()[:50]
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN, budget=0.1, post_fn=wire(specs, "fixed", cost=0.03),
                   sleep=lambda s: 0, log=lambda *a, **k: None)
    stop = run.run(specs)
    assert stop and stop.startswith("budget") and run.counts["ok"] == 4


def test_smoke_summary_and_go_rule(tmp_path):
    specs = P.smoke()
    run = R.Runner(tmp_path, key=KEY, pin=None, post_fn=wire(specs, "reader", cost=3e-5), sleep=lambda s: 0,
                   log=lambda *a, **k: None)
    run.run(specs, concurrency=4)
    s = D.smoke_summary(specs, tmp_path / "raw.jsonl")
    assert s["pin"] == A.PIN and s["same_build_as_J1_J3"] and s["valid_share"] == 1.0
    assert s["determinism"]["flip_rate"] == 0.0 and len(s["determinism"]["groups"]) == 12
    assert s["throughput"]["ok"] == 96 and s["concurrency"] == 16
    assert s["cost"]["projected_flight"] == pytest.approx(3456 * 3e-5) and s["go"]


def test_smoke_go_rule_refuses_two_builds(tmp_path):
    specs = P.smoke()
    half = {s["body_sha"] for s in specs[::2]}
    a, b = wire(specs, "fixed"), wire(specs, "fixed", model="typesafe/jev-1.14")
    post = lambda body, key, url: (a if hashlib.sha256(body).hexdigest() in half else b)(body, key, url)  # noqa: E731
    run = R.Runner(tmp_path, key=KEY, pin=None, post_fn=post, sleep=lambda s: 0, log=lambda *a, **k: None)
    run.run(specs)
    s = D.smoke_summary(specs, tmp_path / "raw.jsonl")
    assert s["pin"] is None and not s["go"]
