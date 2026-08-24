#!/usr/bin/env python3
"""Recompute the E8-R3-c verdict LOCALLY from the flown condition bundles
(lane law: the VM banner is confirmed by exec'ing the verdict cell VERBATIM
over the raw shipped rows) and diff field-by-field against the shipped
e8r3c_verdict.json.

Expects in colab/results_e8r3c/full_20260824_0505/:
  condition_real.json, condition_scrambled.json, e8r3c_verdict.json

Run:  ~/venvs/semcore/bin/python3 colab/recompute_e8r3c.py
"""
import io, json, os, tempfile
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = Path(HERE) / "results_e8r3c" / "full_20260824_0505"

nb = json.load(open(os.path.join(HERE, "E8R3C_COMPOSED_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][-1]["source"])
assert "SB_claim" in VERDICT_SRC, "last cell is not the verdict"
LOGIC_SRC = "".join(nb["cells"][2]["source"])

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}

results = {c: json.load(open(RES / f"condition_{c}.json"))
           for c in ("real", "scrambled")}

tmp = Path(tempfile.mkdtemp())
g = {"__name__": "verdict_recompute"}
exec(LOGIC_SRC, g)

g.update({"RESULTS": results, "CONDITIONS": ["real", "scrambled"],
          "cond_errors": {}, "MODE": "full",
          "STAMP": "20260824_0505", "RESUME_STAMP": "20260824_0305",
          "SMOKE": False, "OUT": tmp, "VEC": VEC, "np": np,
          "ship": lambda *a, **k: None})
buf = io.StringIO()
with redirect_stdout(buf):
    exec(VERDICT_SRC, g)
print(buf.getvalue())

local = json.load(open(tmp / "e8r3c_verdict.json"))
json.dump(local, open(RES / "e8r3c_verdict_local.json", "w"), indent=1)

shipped_fn = RES / "e8r3c_verdict.json"
if not shipped_fn.exists():
    print("\n(shipped e8r3c_verdict.json not present — banner-only recompute)")
    raise SystemExit(0)
shipped = json.load(open(shipped_fn))

diffs, fp_noise = [], []
def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
    elif a != b:
        if (isinstance(a, float) and isinstance(b, float)
                and abs(a - b) <= 1e-8 * max(1.0, abs(a), abs(b))):
            fp_noise.append(f"{path}: {a!r} vs {b!r}")   # cross-BLAS ulp noise
        else:
            diffs.append(f"{path}: {a!r} != {b!r}")
walk(shipped, local)

print(f"substantive diffs: {len(diffs)} | float-noise (rel<=1e-8): {len(fp_noise)}")
for d in diffs[:40]:
    print("  SUBSTANTIVE ", d)
for d in fp_noise[:5]:
    print("  fp-noise    ", d)
if len(fp_noise) > 5:
    print(f"   ... {len(fp_noise) - 5} more fp-noise rows (unrounded S4 errs)")
print("\nRECOMPUTE " + ("CONFIRMED — matches shipped verdict "
                        "(float noise only)" if not diffs else "MISMATCH"))
