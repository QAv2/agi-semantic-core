"""The mouth rider's client (protocol §2, S-J3-6): OpenRouter chat completions,
through J1 + J2's Runner (records, retries, budget, key hygiene), with its own
answer check: the reply must be a JSON object naming one option key and giving a
confidence in [0, 100]. An unparseable answer is retried once, then 'invalid'.
"""
import json
import re

from . import runner as R

URL = "https://openrouter.ai/api/v1/chat/completions"


def content_of(data):
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def parse_answer(text, keys):
    """-> (choice_key, confidence_0_100) or raise ValueError. Tolerates code fences."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    ch, cf = obj.get("choice"), obj.get("confidence")
    if isinstance(cf, str):
        cf = float(cf.strip().rstrip("%"))
    if ch not in keys:
        raise ValueError(f"choice {ch!r} is not an option")
    if not isinstance(cf, (int, float)) or not 0 <= cf <= 100:
        raise ValueError(f"confidence {cf!r} out of range")
    return ch, float(cf)


def reasoning_tokens(data):
    u = data.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    return int(det.get("reasoning_tokens") or 0)


def has_reasoning_text(data):
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False
    return bool(msg.get("reasoning") or msg.get("reasoning_details"))


class MouthRunner(R.Runner):
    """Runner.one with the chat-completions answer check; `pin` = the model string."""

    def __init__(self, out_dir, **kw):
        kw.setdefault("url", URL)
        super().__init__(out_dir, **kw)

    def one(self, spec):
        if self.stop_reason or spec["call_id"] in self.done:
            return
        body = R.C.body_bytes(spec["body"])
        if R.C.body_sha(spec["body"]) != spec["body_sha"]:
            raise RuntimeError(f"body sha mismatch for {spec['call_id']}")
        attempt, invalid_seen = 0, False
        while not self.stop_reason:
            attempt += 1
            status, data, lat, err = self.post_fn(body, self.key, self.url)
            if status == 200 and isinstance(data, dict) and "choices" in data:
                if self.pin and data.get("model") != self.pin:
                    self._write(self._rec(spec, "build_mismatch", status, attempt, lat, data))
                    self.stop_reason = f"mouth model mismatch: {data.get('model')!r} != {self.pin!r}"
                    return
                try:
                    parse_answer(content_of(data), spec["key_to_idx"])
                    reason = None
                except (ValueError, json.JSONDecodeError) as e:
                    reason = str(e)[:200]
                if reason is None:
                    self._write(self._rec(spec, "ok", status, attempt, lat, data))
                    return
                if not invalid_seen:
                    invalid_seen = True
                    self._write(self._rec(spec, "retry", status, attempt, lat, data, reason=reason))
                    continue
                self._write(self._rec(spec, "invalid", status, attempt, lat, data, reason=reason))
                return
            retryable = status is None or status in R.RETRYABLE
            if retryable and attempt <= self.max_retries:
                self._write(self._rec(spec, "retry", status, attempt, lat, data, err=err))
                self.sleep(min(30.0, 1.5 * 2 ** (attempt - 1)))
                continue
            self._write(self._rec(spec, "error", status, attempt, lat, data, err=err))
            return
