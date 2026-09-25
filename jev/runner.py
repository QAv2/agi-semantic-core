"""OpenRouter client + resumable, budget-capped runner (protocol §2, §3, §11).

Live gates: G1 build pin (stop on mismatch) · G2 schema (one retry, then 'invalid') ·
G7 cumulative budget. Every response is appended verbatim to raw.jsonl with the
request sha256. The API key never enters any record (checked on every write).
"""
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import common as C

URL = "https://openrouter.ai/api/v1/systemone"
RESULTS = Path(__file__).resolve().parent / "results"
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
FINAL_DONE = {"ok", "invalid"}


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k.strip()
    env = Path.home() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY is not set")


def post(body, key, url=URL, timeout=60):
    """-> (http_status | None, parsed_json | None, latency_s, error_text | None)"""
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read(), resp.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return None, None, time.time() - t0, type(e).__name__ + ": " + str(e)[:300]
    lat = time.time() - t0
    try:
        data = json.loads(raw)
    except ValueError:
        data = {"_unparsed": raw.decode("utf-8", "replace")[:2000]}
    return status, data, lat, None


def validate(spec, data):
    """G2: None if the response is well-formed for this spec, else a reason."""
    ans = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(ans, dict):
        return "no answers"
    for q in spec["questions"]:
        a = ans.get(q)
        if not isinstance(a, dict):
            return f"missing answer {q}"
        if q == "decision":
            probs, ch = a.get("probabilities"), a.get("choice")
            keys = set(spec["key_to_idx"])
            if not isinstance(probs, dict) or set(probs) != keys:
                return "probability keys differ from options"
            if not all(isinstance(v, (int, float)) for v in probs.values()):
                return "non-numeric probability"
            s = sum(probs.values())
            if not 0.97 <= s <= 1.03:
                return f"probabilities sum to {s:.3f}"
            if ch not in keys:
                return "choice is not an option"
        else:
            v = a.get("noul")
            if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                return f"bad noul for {q}"
    return None


def cost_of(data):
    try:
        return float(data.get("usage", {}).get("cost") or 0.0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def iter_records(path):
    path = Path(path)
    if not path.exists():
        return
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def prior_spend(results=RESULTS, exclude=None):
    """Cumulative usage.cost over every OTHER stage on disk (G7 is cumulative;
    the runner adds its own directory's records itself)."""
    total = 0.0
    exclude = Path(exclude).resolve() if exclude else None
    for p in list(results.glob("*/raw.jsonl")) + list(results.glob("*/raw.jsonl.gz")):
        if exclude and p.parent.resolve() == exclude:
            continue
        if p.suffix == ".gz" and p.with_suffix("").exists():
            continue  # a live raw.jsonl supersedes its compressed copy
        for rec in iter_records(p):
            if rec.get("response") is not None:
                total += cost_of(rec["response"])
    return total


class Runner:
    def __init__(self, out_dir, pin=None, budget=15.0, spent=0.0, key=None, url=URL,
                 max_retries=5, post_fn=post, sleep=time.sleep, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.raw = self.out / "raw.jsonl"
        self.pin, self.budget, self.url = pin, budget, url
        self.key = key if key is not None else api_key()
        self.max_retries, self.post_fn, self.sleep, self.log = max_retries, post_fn, sleep, log
        self.lock = threading.Lock()
        self.spent = spent
        self.stop_reason = None
        self.done = set()
        self.counts = {"ok": 0, "invalid": 0, "error": 0, "build_mismatch": 0, "retry": 0}
        for rec in iter_records(self.raw):
            if rec["status"] in FINAL_DONE:
                self.done.add(rec["call_id"])
            if rec.get("response") is not None:
                self.spent += cost_of(rec["response"])

    def _write(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        if self.key and self.key in line:
            raise RuntimeError("API key would be written to a record; refusing")
        with self.lock:
            with open(self.raw, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
            self.counts[rec["status"]] = self.counts.get(rec["status"], 0) + 1
            if rec.get("response") is not None:
                self.spent += cost_of(rec["response"])
                if self.spent >= self.budget and not self.stop_reason:
                    self.stop_reason = f"budget: ${self.spent:.4f} >= ${self.budget}"

    def _rec(self, spec, status, http, attempt, lat, data, err=None, reason=None):
        return {"call_id": spec["call_id"], "body_sha": spec["body_sha"], "status": status, "http": http,
                "attempt": attempt, "t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "latency_s": round(lat, 4) if lat is not None else None, "error": err, "reason": reason,
                "response": data}

    def one(self, spec):
        if self.stop_reason or spec["call_id"] in self.done:
            return
        body = C.body_bytes(spec["body"])
        if C.body_sha(spec["body"]) != spec["body_sha"]:
            raise RuntimeError(f"body sha mismatch for {spec['call_id']}")
        attempt, invalid_seen = 0, False
        while not self.stop_reason:
            attempt += 1
            status, data, lat, err = self.post_fn(body, self.key, self.url)
            if status == 200 and isinstance(data, dict) and "answers" in data:
                if self.pin and data.get("model") != self.pin:
                    self._write(self._rec(spec, "build_mismatch", status, attempt, lat, data))
                    self.stop_reason = f"G1 build mismatch: {data.get('model')!r} != {self.pin!r}"
                    return
                reason = validate(spec, data)
                if reason is None:
                    self._write(self._rec(spec, "ok", status, attempt, lat, data))
                    return
                if not invalid_seen:
                    invalid_seen = True
                    self._write(self._rec(spec, "retry", status, attempt, lat, data, reason=reason))
                    continue
                self._write(self._rec(spec, "invalid", status, attempt, lat, data, reason=reason))
                return
            retryable = status is None or status in RETRYABLE
            if retryable and attempt <= self.max_retries:
                self._write(self._rec(spec, "retry", status, attempt, lat, data, err=err))
                self.sleep(min(30.0, 1.5 * 2 ** (attempt - 1)))
                continue
            self._write(self._rec(spec, "error", status, attempt, lat, data, err=err))
            return

    def run(self, specs, concurrency=1, progress_every=250, label=""):
        todo = [s for s in specs if s["call_id"] not in self.done]
        n0 = len(todo)
        if concurrency <= 1:
            for n, s in enumerate(todo, 1):
                self.one(s)
                if n % progress_every == 0:
                    self._progress(label, n, n0)
                if self.stop_reason:
                    break
        else:
            with ThreadPoolExecutor(concurrency) as ex:
                futs = [ex.submit(self.one, s) for s in todo]
                for n, f in enumerate(futs, 1):
                    f.result()
                    if n % progress_every == 0:
                        self._progress(label, n, n0)
        self._progress(label, n0, n0)
        return self.stop_reason

    def _progress(self, label, n, n0):
        c = self.counts
        self.log(f"[{label}] {n}/{n0} ok={c['ok']} invalid={c['invalid']} error={c['error']} "
                 f"retry={c['retry']} spent=${self.spent:.4f}"
                 + (f" STOP: {self.stop_reason}" if self.stop_reason else ""), flush=True)

    def finalize(self):
        """Compress raw.jsonl -> raw.jsonl.gz (kept byte-identical in content)."""
        gz = self.raw.with_suffix(".jsonl.gz")
        with open(self.raw, "rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
            dst.write(src.read())
        return gz


def final_records(path):
    """call_id -> final record (last ok/invalid/error/build_mismatch), ignoring retries."""
    out = {}
    for rec in iter_records(path):
        if rec["status"] != "retry":
            out[rec["call_id"]] = rec
    return out
