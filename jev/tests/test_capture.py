"""Capture suite: the runner's wire format, retries, gates G1/G2/G7, resume, key
hygiene, gzip round trip. A real local HTTP server checks the exact request.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_capture.py -q
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev import common as C
from jev import plan as P
from jev import runner as R
from jev.agents import J2Agent, PIN

KEY = "test-key-not-a-real-secret-0123456789"


@pytest.fixture(scope="module")
def specs():
    return P.j2_flight(1)[:6]


def good(spec, model=PIN, cost=0.001):
    return J2Agent("ideal").respond(spec) | {"model": model, "usage": {"input_tokens": 1, "output_tokens": 1,
                                                                        "cost": cost}}


def scripted(seq):
    """post_fn returning scripted (status, data) per call, in order."""
    it = iter(seq)

    def post(body, key, url):
        assert key == KEY
        status, data = next(it)
        return status, data, 0.01, None if status else "URLError: down"
    return post


def lines(path):
    return [json.loads(l) for l in open(path)]


def test_ok_records_and_key_hygiene(tmp_path, specs):
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (200, good(specs[0]), 0.02, None), sleep=lambda s: 0)
    run.run(specs[:1])
    recs = lines(tmp_path / "raw.jsonl")
    assert len(recs) == 1 and recs[0]["status"] == "ok" and recs[0]["body_sha"] == specs[0]["body_sha"]
    assert KEY not in (tmp_path / "raw.jsonl").read_text()


def test_retry_429_then_ok(tmp_path, specs):
    s = specs[0]
    naps = []
    run = R.Runner(tmp_path, key=KEY, post_fn=scripted([(429, {"error": "rate"}), (None, None), (200, good(s))]),
                   sleep=naps.append)
    run.run([s])
    st = [r["status"] for r in lines(tmp_path / "raw.jsonl")]
    assert st == ["retry", "retry", "ok"] and naps == [1.5, 3.0]


def test_invalid_twice_is_invalid(tmp_path, specs):
    s = specs[0]
    bad = good(s)
    bad["answers"]["decision"]["probabilities"] = {k: 0.5 for k in bad["answers"]["decision"]["probabilities"]}
    run = R.Runner(tmp_path, key=KEY, post_fn=scripted([(200, bad), (200, bad)]), sleep=lambda x: 0)
    run.run([s])
    recs = lines(tmp_path / "raw.jsonl")
    assert [r["status"] for r in recs] == ["retry", "invalid"] and "sum" in recs[1]["reason"]


def test_error_after_max_retries(tmp_path, specs):
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (503, {"error": "x"}, 0.01, None),
                   sleep=lambda x: 0, max_retries=5)
    run.run(specs[:1])
    st = [r["status"] for r in lines(tmp_path / "raw.jsonl")]
    assert st == ["retry"] * 5 + ["error"]


def test_non_retryable_4xx(tmp_path, specs):
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (400, {"error": "bad"}, 0.01, None), sleep=lambda x: 0)
    run.run(specs[:1])
    assert [r["status"] for r in lines(tmp_path / "raw.jsonl")] == ["error"]


def test_g1_build_mismatch_stops(tmp_path, specs):
    calls = []

    def post(b, k, u):
        calls.append(1)
        return 200, good(specs[0], model="typesafe/jev-1.14-20261001"), 0.01, None
    run = R.Runner(tmp_path, pin=PIN, key=KEY, post_fn=post, sleep=lambda x: 0)
    stop = run.run(specs)
    assert "G1" in stop and len(calls) == 1
    assert lines(tmp_path / "raw.jsonl")[0]["status"] == "build_mismatch"


def test_g7_budget_stops(tmp_path, specs):
    run = R.Runner(tmp_path, key=KEY, budget=2.5, spent=0.0,
                   post_fn=lambda b, k, u: (200, good(specs[0], cost=1.0), 0.01, None), sleep=lambda x: 0)
    # every response is valid for specs[0] only; use copies of specs[0] with distinct ids
    many = [dict(specs[0], call_id=f"x{i}") for i in range(6)]
    stop = run.run(many)
    assert stop.startswith("budget") and len(lines(tmp_path / "raw.jsonl")) == 3


def test_resume_skips_done(tmp_path, specs):
    s = specs[0]
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (200, good(s), 0.01, None), sleep=lambda x: 0)
    run.run([s])
    calls = []
    run2 = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: calls.append(1), sleep=lambda x: 0)
    run2.run([s])
    assert calls == [] and abs(run2.spent - 0.001) < 1e-12


def test_key_echo_refused(tmp_path, specs):
    s = specs[0]
    leaky = good(s)
    leaky["echo"] = KEY
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (200, leaky, 0.01, None), sleep=lambda x: 0)
    with pytest.raises(RuntimeError, match="API key"):
        run.run([s])
    assert not (tmp_path / "raw.jsonl").exists() or KEY not in (tmp_path / "raw.jsonl").read_text()


def test_finalize_roundtrip(tmp_path, specs):
    run = R.Runner(tmp_path, key=KEY, post_fn=lambda b, k, u: (200, good(specs[0]), 0.01, None), sleep=lambda x: 0)
    run.run([dict(specs[0], call_id=f"y{i}") for i in range(3)])
    gz = run.finalize()
    assert list(R.iter_records(gz)) == lines(tmp_path / "raw.jsonl")
    assert set(R.final_records(gz)) == {"y0", "y1", "y2"}


def test_prior_spend_excludes_current(tmp_path, specs):
    for name, cost in (("a", 0.25), ("b", 0.5)):
        run = R.Runner(tmp_path / name, key=KEY, post_fn=lambda b, k, u, c=cost: (200, good(specs[0], cost=c), .01, None),
                       sleep=lambda x: 0)
        run.run(specs[:1])
    assert abs(R.prior_spend(tmp_path) - 0.75) < 1e-12
    assert abs(R.prior_spend(tmp_path, exclude=tmp_path / "b") - 0.25) < 1e-12


@pytest.mark.parametrize("mut,reason", [
    (lambda a: a.pop("decision"), "missing answer"),
    (lambda a: a["decision"].__setitem__("choice", "nope"), "not an option"),
    (lambda a: a["decision"]["probabilities"].popitem(), "keys differ"),
])
def test_validate(specs, mut, reason):
    d = good(specs[0])
    mut(d["answers"])
    assert reason in R.validate(specs[0], d)


def test_real_wire_request(specs):
    """The exact bytes and headers on the wire, against a local HTTP server."""
    seen = {}
    s = specs[0]

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers["Authorization"]
            seen["ctype"] = self.headers["Content-Type"]
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps(good(s)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.handle_request, daemon=True)
    th.start()
    status, data, lat, err = R.post(C.body_bytes(s["body"]), KEY,
                                    url=f"http://127.0.0.1:{srv.server_port}/api/v1/systemone")
    th.join(5)
    srv.server_close()
    assert status == 200 and err is None and data["answers"]["decision"]["choice"]
    assert seen["path"] == "/api/v1/systemone" and seen["auth"] == f"Bearer {KEY}"
    assert seen["ctype"] == "application/json" and seen["body"] == C.body_bytes(s["body"])
    body = json.loads(seen["body"])
    assert set(body) == {"model", "state", "questions"} and body["model"] == "jev-1.13"
