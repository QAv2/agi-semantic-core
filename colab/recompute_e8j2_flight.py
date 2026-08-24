#!/usr/bin/env python3
"""E8-J v2 flight recompute (the law): exec the staged notebook's verdict
cell VERBATIM over the raw shipped rungb.json, diff every field against the
shipped e8j2_verdict.json, and independently re-tally the primary's
ingredients from the raw rows.

Run:  python3 colab/recompute_e8j2_flight.py
"""
import contextlib, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j2_logic as L2

RES = os.path.join(HERE, "results_e8j2", "full_20260824_2006")
RUNGB = json.load(open(os.path.join(RES, "rungb.json")))
SHIPPED = json.load(open(os.path.join(RES, "e8j2_verdict.json")))
assert RUNGB["stamp"] == SHIPPED["stamp"] == "20260824_2006"

# ── exec the verdict cell VERBATIM from the staged notebook ──────────────────
nb = json.load(open(os.path.join(HERE, "E8J2_RUNG_UI.ipynb")))
VERDICT_SRC = "".join(nb["cells"][6]["source"])
tmp = Path(tempfile.mkdtemp(prefix="e8j2_rc_"))
out = tmp / "out"; out.mkdir()
ns = {n: getattr(L2, n) for n in dir(L2) if not n.startswith("__")}
ns.update({"RUNGB": RUNGB, "GATES": RUNGB["gates_snapshot"], "cond_errors": {},
           "SMOKE": False, "MODE": "full", "STAMP": RUNGB["stamp"],
           "MODEL_ID": SHIPPED["model"], "OUT": out, "SEM": tmp / "sem",
           "INFLIGHT": "e8j2/inflight_x", "RESUME_STAMP": "",
           "B_PAIR_ROWS": L2.build_b_pairs(False), "json": json,
           "ship": lambda src, dest: None})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(VERDICT_SRC, "<verdict-cell>", "exec"), ns)
recomp = json.load(open(out / "e8j2_verdict.json"))
print("verdict cell exec'd VERBATIM over raw bundle")

# ── field-by-field diff ──────────────────────────────────────────────────────
def diff(a, b, path=""):
    o = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                o.append(f"+{path}.{k} (shipped-only)")
            elif k not in b:
                o.append(f"-{path}.{k} (recompute-only)")
            else:
                o.extend(diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, float) or isinstance(b, float):
        if not np.isclose(float(a), float(b), rtol=0, atol=1e-9):
            o.append(f"{path}: recomputed {a!r} != shipped {b!r}")
    elif a != b:
        o.append(f"{path}: recomputed {a!r} != shipped {b!r}")
    return o

d = diff(recomp, SHIPPED)
print(f"\n=== DIFF vs shipped verdict: {len(d)} differences ===")
for x in d[:30]:
    print("  ", x)
if not d:
    print("   NONE — the shipped verdict reproduces verbatim from raw rows.")

# ── independent raw-row tallies (fresh code, not the cell's) ─────────────────
rows = RUNGB["fc_rows"]
tr = set(L2.TRAINED)
re_ex = sum(1 for r in rows if r["arm"] == "real" and r["injected"] in tr and r["exact"])
pe_ex = sum(1 for r in rows if r["arm"] == "perm" and r["injected"] in tr and r["exact"])
pairs = {}
for r in rows:
    if r["injected"] in tr:
        pairs.setdefault(r["pair_id"], {})[r["arm"]] = r["exact"]
dsum = sum(int(p["real"]) - int(p["perm"]) for p in pairs.values())
per_t = {}
for r in rows:
    if r["arm"] == "real" and r["injected"] in tr and r["exact"]:
        per_t[r["injected"]] = per_t.get(r["injected"], 0) + 1
print("\n=== independent tallies (raw rows) ===")
print(f"  trained-9 exact: real {re_ex} / perm {pe_ex} | pair diff sum {dsum} "
      f"| n_pairs {len(pairs)}")
print(f"  real exact by target: {dict(sorted(per_t.items(), key=lambda kv: -kv[1]))}")
print(f"  anchor exact {sum(1 for r in RUNGB['anchor'] if r['exact'])}/18 | "
      f"sham claims {sum(1 for r in RUNGB['shams'] if not r['none_top'])}/12 | "
      f"inject hooks all fired: {all(r['hook_calls'] >= 1 for r in rows)} | "
      f"sham hooks zero: {all(r['hook_calls'] == 0 for r in RUNGB['shams'])}")
ho_ex = {c: sum(1 for r in rows if r['arm'] == 'real' and r['injected'] == c
                and r['exact']) for c in L2.HELD_OUT}
print(f"  held-out-4 real exact (texture, outside primary): {ho_ex}")
alpha_split = {}
for r in rows:
    if r["arm"] == "real" and r["injected"] in tr and r["exact"]:
        alpha_split[r["alpha"]] = alpha_split.get(r["alpha"], 0) + 1
print(f"  real exact by alpha: {alpha_split}")
ok = (re_ex == SHIPPED["stageb"]["P_B"]["real_exact"]
      and pe_ex == SHIPPED["stageb"]["P_B"]["perm_exact"]
      and dsum == SHIPPED["stageb"]["P_B"]["obs_diff"] and not d)
print(f"\n{'RECOMPUTE CLEAN — lock stands on recomputed statistics' if ok else 'MISMATCH — investigate before lock'}")
