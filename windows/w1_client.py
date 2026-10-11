"""OpenRouter chat-completions client + resumable, budget-capped runner for the windows series
(W1 protocol §4).

Live gates: G1 build pin (allowed ids per requested model, constant within a stage; stop on
change) · G2 schema (one retry with the same body, then 'invalid') · G7 cumulative budget.
Every response is appended to raw.jsonl with the request sha256. Images (any `data:image/...`
string anywhere in a response) are decoded to files under the stage's png/ directory and the
string is replaced by `file:<relative path>#sha256=<hex>`, so no base64 enters the log. The
API key never enters any record (checked on every write).
"""
import base64
import gzip
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
RESULTS = Path(__file__).resolve().parent / "results"
RETRYABLE = {408, 409, 425, 500, 502, 503, 504, 520, 522, 524}
FINAL = {"ok", "invalid", "error", "build_mismatch"}
DATA_URL = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.S)
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}


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


def canon(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def body_sha(body):
    return hashlib.sha256(canon(body)).hexdigest()


def post(body_bytes, key, url=URL, timeout=300):
    """-> (http_status | None, parsed_json | None, latency_s, error_text | None)"""
    req = urllib.request.Request(url, data=body_bytes, method="POST",
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


def cost_of(data):
    try:
        return float((data or {}).get("usage", {}).get("cost") or 0.0)
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


def final_records(path):
    """call_id -> final record (the last ok/invalid/error/build_mismatch), ignoring retries."""
    out = {}
    for rec in iter_records(path):
        if rec["status"] in FINAL:
            out[rec["call_id"]] = rec
    return out


def stage_raw(stage_dir):
    """The live raw.jsonl if present, else its compressed copy."""
    p = Path(stage_dir) / "raw.jsonl"
    return p if p.exists() else p.with_suffix(".jsonl.gz")


def prior_spend(prefix="w1_", results=RESULTS, exclude=None):
    """Cumulative usage.cost over every OTHER stage directory on disk (G7 is cumulative)."""
    total = 0.0
    exclude = Path(exclude).resolve() if exclude else None
    files = list(results.glob(f"{prefix}*/raw.jsonl*")) + list(results.glob(f"{prefix}*/*/raw.jsonl*"))
    for p in files:
        if exclude and (p.parent.resolve() == exclude):
            continue
        if p.suffix == ".gz" and p.with_suffix("").exists():
            continue
        for rec in iter_records(p):
            total += cost_of(rec.get("response"))
    return total


# ---- response helpers --------------------------------------------------------------------

def message(data):
    try:
        return data["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def text_of(data):
    """The response's text: a string content, or the text parts of a list content, joined."""
    c = message(data).get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def images_of(data):
    """Relative image file refs in message.images, in order (after externalization)."""
    out = []
    for im in message(data).get("images") or []:
        url = ((im or {}).get("image_url") or {}).get("url", "")
        if isinstance(url, str) and url.startswith("file:"):
            out.append(url[5:].split("#", 1)[0])
    return out


def reasoning_of(data):
    m = message(data)
    r = m.get("reasoning")
    if isinstance(r, str) and r.strip():
        return r.strip()
    parts = []
    for d in m.get("reasoning_details") or []:
        if isinstance(d, dict):
            for k in ("text", "summary"):
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
    return "\n".join(parts)


def externalize(obj, save):
    """Replace every data:image string inside obj (recursively) by save(mime, bytes) -> ref."""
    if isinstance(obj, dict):
        return {k: externalize(v, save) for k, v in obj.items()}
    if isinstance(obj, list):
        return [externalize(v, save) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image/"):
        m = DATA_URL.match(obj)
        if m:
            try:
                raw = base64.b64decode(m.group(2), validate=False)
            except (ValueError, TypeError):
                return "file:undecodable"
            return save(m.group(1).lower(), raw)
    return obj


# ---- validators (G2) ----------------------------------------------------------------------

def validate_text(data):
    return None if text_of(data) else "no text"


def validate_draw(data):
    return None if images_of(data) else "no image"


ANSWER = re.compile(r"\b(IMAGE|SENTENCE|PAIR)\s*:\s*([12])\b", re.I)
CONF = re.compile(r"\bCONFIDENCE\s*:\s*(\d{1,3})", re.I)


def parse_choice(text):
    """-> (choice 1|2, confidence 50..100) or None."""
    a, c = ANSWER.search(text or ""), CONF.search(text or "")
    if not a or not c:
        return None
    return int(a.group(2)), max(50, min(100, int(c.group(1))))


def parse_inventory(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except ValueError:
        return None
    need = {"motifs", "colors", "mood", "text", "description"}
    if not isinstance(d, dict) or not need <= set(d):
        return None
    if not isinstance(d["motifs"], list) or not isinstance(d["colors"], list):
        return None
    return d


def validate_read(data):
    return None if parse_choice(text_of(data)) else "unparsable answer"


def validate_inv(data):
    return None if parse_inventory(text_of(data)) else "unparsable inventory"


VALIDATORS = {"text": validate_text, "draw": validate_draw, "read": validate_read, "inv": validate_inv}


# ---- the runner ---------------------------------------------------------------------------

class Runner:
    def __init__(self, out_dir, pins, budget=16.0, spent=0.0, key=None, url=URL,
                 max_429=60, max_other=6, post_fn=post, sleep=time.sleep, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "png").mkdir(exist_ok=True)
        self.raw = self.out / "raw.jsonl"
        self.pins, self.budget, self.url = pins, budget, url
        self.key = key if key is not None else api_key()
        self.max_429, self.max_other = max_429, max_other
        self.post_fn, self.sleep, self.log = post_fn, sleep, log
        self.lock = threading.Lock()
        self.spent = spent
        self.stop_reason = None
        self.final = {}
        self.seen_build = {}
        self.counts = {"ok": 0, "invalid": 0, "error": 0, "build_mismatch": 0, "retry": 0}
        for rec in iter_records(self.raw):
            if rec["status"] in FINAL:
                self.final[rec["call_id"]] = rec
            self.spent += cost_of(rec.get("response"))
            b = rec.get("build")
            if b and rec.get("requested"):
                self.seen_build.setdefault(rec["requested"], b)

    def _write(self, rec):
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        if self.key and self.key in line:
            raise RuntimeError("API key would be written to a record; refusing")
        with self.lock:
            with open(self.raw, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
            self.counts[rec["status"]] = self.counts.get(rec["status"], 0) + 1
            if rec["status"] in FINAL:
                self.final[rec["call_id"]] = rec
            self.spent += cost_of(rec.get("response"))
            if self.spent >= self.budget and not self.stop_reason:
                self.stop_reason = f"G7 budget: ${self.spent:.4f} >= ${self.budget}"

    def _saver(self, call_id, attempt, files):
        def save(mime, raw):
            n = len(files)
            ext = EXT.get(mime, "bin")
            name = f"{call_id}__a{attempt}__{n}.{ext}"
            path = self.out / "png" / name
            path.write_bytes(raw)
            h = hashlib.sha256(raw).hexdigest()
            files.append({"path": f"png/{name}", "sha256": h, "bytes": len(raw), "mime": mime})
            return f"file:png/{name}#sha256={h}"
        return save

    def _rec(self, spec, status, http, attempt, lat, data, files, err=None, reason=None, build=None):
        return {"call_id": spec["call_id"], "kind": spec["kind"], "body_sha": spec["body_sha"],
                "requested": spec["body"].get("model"), "build": build, "status": status,
                "http": http, "attempt": attempt,
                "t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "latency_s": round(lat, 4) if lat is not None else None, "error": err,
                "reason": reason, "files": files, "response": data}

    def one(self, spec):
        """Run one call to a final record (or stop). Returns the final record or None."""
        if spec["call_id"] in self.final:
            return self.final[spec["call_id"]]
        if self.stop_reason:
            return None
        if body_sha(spec["body"]) != spec["body_sha"]:
            raise RuntimeError(f"G4 body sha mismatch for {spec['call_id']}")
        body = canon(spec["body"])
        validate = VALIDATORS[spec["kind"]]
        requested = spec["body"].get("model")
        attempt = n429 = nother = 0
        invalid_seen = False
        while not self.stop_reason:
            attempt += 1
            status, data, lat, err = self.post_fn(body, self.key, self.url)
            files = []
            if isinstance(data, (dict, list)):
                data = externalize(data, self._saver(spec["call_id"], attempt, files))
            if status == 200 and isinstance(data, dict) and "choices" in data:
                build = data.get("model")
                allowed = self.pins.get(requested, set())
                with self.lock:
                    first = self.seen_build.setdefault(requested, build)
                if build not in allowed or build != first:
                    self._write(self._rec(spec, "build_mismatch", status, attempt, lat, data, files, build=build))
                    self.stop_reason = (f"G1 build: {build!r} for {requested!r} "
                                        f"(allowed {sorted(allowed)}, first seen {first!r})")
                    return None
                reason = validate(data)
                if reason is None:
                    rec = self._rec(spec, "ok", status, attempt, lat, data, files, build=build)
                    self._write(rec)
                    return rec
                if not invalid_seen:
                    invalid_seen = True
                    self._write(self._rec(spec, "retry", status, attempt, lat, data, files, reason=reason, build=build))
                    continue
                rec = self._rec(spec, "invalid", status, attempt, lat, data, files, reason=reason, build=build)
                self._write(rec)
                return rec
            if status == 429 and n429 < self.max_429:
                n429 += 1
                self._write(self._rec(spec, "retry", status, attempt, lat, data, files, err=err, reason="429"))
                self.sleep(min(300.0, 20.0 * n429))
                continue
            if (status is None or status in RETRYABLE) and nother < self.max_other:
                nother += 1
                self._write(self._rec(spec, "retry", status, attempt, lat, data, files, err=err))
                self.sleep(min(60.0, 2.0 ** nother))
                continue
            rec = self._rec(spec, "error", status, attempt, lat, data, files, err=err)
            self._write(rec)
            return rec
        return None

    def progress(self, label):
        c = self.counts
        self.log(f"[{label}] ok={c['ok']} invalid={c['invalid']} error={c['error']} retry={c['retry']} "
                 f"spent=${self.spent:.4f}" + (f" STOP: {self.stop_reason}" if self.stop_reason else ""),
                 flush=True)

    def finalize(self):
        """Compress raw.jsonl -> raw.jsonl.gz (same content)."""
        gz = self.raw.with_suffix(".jsonl.gz")
        with open(self.raw, "rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
            dst.write(src.read())
        return gz
