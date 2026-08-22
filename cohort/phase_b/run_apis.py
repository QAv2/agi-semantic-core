#!/usr/bin/env python3
"""Phase B — run the three outside-family participants (GEM, MIS, DSK).

Single-turn, no system prompt, per PHASE_B_PREREG.md. Raw JSON + extracted
text archived under responses/raw/. No content follow-ups; the one
permitted mechanical retry (format-only) is run manually if triggered.
"""
import json, os, sys, time, urllib.request, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "responses" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

env = {}
for line in open(os.path.expanduser("~/.env")):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")

def post(url, payload, headers, timeout=1800):
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def run_gem():
    prompt = (HERE / "prompts" / "GEM.md").read_text()
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-3-flash-preview:generateContent?key={env['GEMINI_API_KEY']}")
    out = post(url, {"contents": [{"parts": [{"text": prompt}]}],
                     "generationConfig": {"maxOutputTokens": 32768}}, {})
    (RAW / "GEM.json").write_text(json.dumps(out, indent=1))
    parts = out["candidates"][0]["content"]["parts"]
    text = "\n".join(p.get("text", "") for p in parts)
    (RAW / "GEM.txt").write_text(text)
    print("GEM done:", len(text), "chars, finish:",
          out["candidates"][0].get("finishReason"))

def run_mis():
    prompt = (HERE / "prompts" / "MIS.md").read_text()
    out = post("https://api.mistral.ai/v1/chat/completions",
               {"model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16384},
               {"Authorization": f"Bearer {env['MISTRAL_API_KEY']}"})
    (RAW / "MIS.json").write_text(json.dumps(out, indent=1))
    text = out["choices"][0]["message"]["content"]
    (RAW / "MIS.txt").write_text(text)
    print("MIS done:", len(text), "chars, finish:",
          out["choices"][0].get("finish_reason"))

def run_dsk():
    prompt = (HERE / "prompts" / "DSK.md").read_text()
    out = post("https://api.deepseek.com/chat/completions",
               {"model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32000},
               {"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}"})
    (RAW / "DSK.json").write_text(json.dumps(out, indent=1))
    msg = out["choices"][0]["message"]
    text = msg.get("content") or ""
    (RAW / "DSK.txt").write_text(text)
    rt = len(msg.get("reasoning_content") or "")
    print("DSK done:", len(text), "chars content,", rt, "chars reasoning, finish:",
          out["choices"][0].get("finish_reason"))

def guard(fn, tag):
    t0 = time.time()
    try:
        fn()
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        (RAW / f"{tag}.error").write_text(f"{e}\n{body[:2000]}")
        print(f"{tag} FAILED after {time.time()-t0:.0f}s: {e}", file=sys.stderr)

threads = [threading.Thread(target=guard, args=(f, t))
           for f, t in [(run_gem, "GEM"), (run_mis, "MIS"), (run_dsk, "DSK")]]
[t.start() for t in threads]
[t.join() for t in threads]
print("ALL API PARTICIPANTS COMPLETE")
