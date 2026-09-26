"""J3 capture suite: the Jev runner on J3 specs (14 options + the noul), and the
mouth client's wire format against a real local HTTP server, its answer check,
retries, model pin and key hygiene.

Run: ~/venvs/semcore/bin/python -m pytest jev/tests/test_j3_capture.py -q
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jev import agents_j3 as A
from jev import common as C
from jev import j3
from jev import j3_mouth as M
from jev import j3_plan as P
from jev import runner as R

KEY = "test-key-not-a-real-secret-0123456789"


@pytest.fixture(scope="module")
def bars():
    return {"0": j3.fit_bar(0), "1": j3.fit_bar(1)}


@pytest.fixture(scope="module")
def jspecs():
    stims = j3.flight_stimuli()[:6]
    return [P.jev_spec("flight", s, "DIGEST", i, 0, "bundled", "flight") for i, s in enumerate(stims)]


@pytest.fixture(scope="module")
def mspecs():
    return P.mouth_smoke()[:4]


def lines(path):
    return [json.loads(l) for l in open(path)]


def test_jev_runner_accepts_j3_answers(tmp_path, jspecs, bars):
    run = R.Runner(tmp_path, key=KEY, pin=A.PIN,
                   post_fn=lambda b, k, u: (200, A.respond(jspecs[0], "twin", bars), 0.01, None), sleep=lambda s: 0)
    run.run(jspecs[:1])
    recs = lines(tmp_path / "raw.jsonl")
    assert [r["status"] for r in recs] == ["ok"] and KEY not in (tmp_path / "raw.jsonl").read_text()
    bad = A.respond(jspecs[1], "twin", bars)
    bad["answers"]["decision"]["probabilities"].pop(next(iter(bad["answers"]["decision"]["probabilities"])))
    run2 = R.Runner(tmp_path / "b", key=KEY, post_fn=lambda b, k, u: (200, bad, 0.01, None), sleep=lambda s: 0)
    run2.run(jspecs[1:2])
    assert [r["status"] for r in lines(tmp_path / "b" / "raw.jsonl")] == ["retry", "invalid"]


class _Handler(BaseHTTPRequestHandler):
    seen = []
    replies = []

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Handler.seen.append((self.headers.get("Authorization"), self.rfile.read(n)))
        status, data = _Handler.replies.pop(0)
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _Handler.seen, _Handler.replies = [], []
    yield f"http://127.0.0.1:{srv.server_port}/api/v1/chat/completions"
    srv.shutdown()


def test_mouth_wire_format_and_parse(tmp_path, server, mspecs, bars):
    s = mspecs[0]
    _Handler.replies = [(200, A.mouth_respond(s, bars))]
    run = M.MouthRunner(tmp_path, key=KEY, url=server, pin="deepseek/deepseek-v4-flash-0731", sleep=lambda x: 0)
    run.run([s])
    auth, sent = _Handler.seen[0]
    assert auth == f"Bearer {KEY}" and sent == C.body_bytes(s["body"])
    recs = lines(tmp_path / "raw.jsonl")
    assert recs[0]["status"] == "ok" and recs[0]["body_sha"] == s["body_sha"]
    assert KEY not in (tmp_path / "raw.jsonl").read_text()


def test_mouth_invalid_twice_then_invalid(tmp_path, server, mspecs, bars):
    s = mspecs[1]
    junk = A.mouth_respond(s, bars)
    junk["choices"][0]["message"]["content"] = "I think it is probably the third one."
    _Handler.replies = [(200, junk), (200, junk)]
    run = M.MouthRunner(tmp_path, key=KEY, url=server, sleep=lambda x: 0)
    run.run([s])
    assert [r["status"] for r in lines(tmp_path / "raw.jsonl")] == ["retry", "invalid"]


def test_mouth_model_pin_stops(tmp_path, server, mspecs, bars):
    s = mspecs[2]
    other = A.mouth_respond(s, bars) | {"model": "deepseek/some-other-model"}
    _Handler.replies = [(200, other)]
    run = M.MouthRunner(tmp_path, key=KEY, url=server, pin="deepseek/deepseek-v4-flash-0731", sleep=lambda x: 0)
    stop = run.run([s])
    assert stop and "mismatch" in stop


def test_mouth_retry_on_503(tmp_path, server, mspecs, bars):
    s = mspecs[3]
    _Handler.replies = [(503, {"error": "busy"}), (200, A.mouth_respond(s, bars))]
    naps = []
    run = M.MouthRunner(tmp_path, key=KEY, url=server, sleep=naps.append)
    run.run([s])
    assert [r["status"] for r in lines(tmp_path / "raw.jsonl")] == ["retry", "ok"] and naps == [1.5]


def test_parse_answer_tolerance():
    keys = {"P-abc": 0, "none": 13}
    assert M.parse_answer('```json\n{"choice": "none", "confidence": 85}\n```', keys) == ("none", 85.0)
    assert M.parse_answer('{"choice": "P-abc", "confidence": "70%"}', keys) == ("P-abc", 70.0)
    for bad in ('{"choice": "P-zzz", "confidence": 50}', '{"choice": "none", "confidence": 150}', "no json"):
        with pytest.raises(ValueError):
            M.parse_answer(bad, keys)


def test_key_refusal(tmp_path, mspecs, bars):
    s = mspecs[0]
    leak = A.mouth_respond(s, bars)
    leak["choices"][0]["message"]["content"] = json.dumps({"choice": "none", "confidence": 50, "echo": KEY})
    run = M.MouthRunner(tmp_path, key=KEY, post_fn=lambda b, k, u: (200, leak, 0.01, None), sleep=lambda x: 0)
    with pytest.raises(RuntimeError, match="API key"):
        run.run([s])
