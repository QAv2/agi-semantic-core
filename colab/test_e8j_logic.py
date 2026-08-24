#!/usr/bin/env python3
"""E8-J pure-logic suite (runs on plain python3, numpy only).

Covers: pin determinism + properties, derangement, exact hat-matrix LOO vs
naive refit, relabel-null machinery, planted teeth (a linear world must pass
P-A/G-B logic, a permuted codebook must fail — the registered discrimination
proven before flight), Mantel, axis-wise R^2, Stage-B plans and matched-pair
statistics at BOTH mode constants, verify_pins tamper detection, fork
strings, verdict assembly, and single-source rebuild identity.

Run:  python3 colab/test_e8j_logic.py
"""
import hashlib, json, os, sqlite3, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))

# ── pins ─────────────────────────────────────────────────────────────────────
print("== pins ==")
info = L.verify_pins(PACK["concepts"])
check("verify_pins runs on the real pack", info["n_anchors"] == 320)
con = sqlite3.connect(os.path.join(HERE, "..", "db", "semantic.db"))
TRI = dict(con.execute("SELECT name, trigram FROM concepts").fetchall())
con.close()
frame = L.anchor_frame(PACK["concepts"])
re_full = L.compute_anchor_draw(frame, TRI)
check("draw recompute == pinned (single source, determinism)",
      re_full == L.PINNED_ANCHORS)
check("smoke recompute == pinned", L.compute_smoke_draw(re_full) ==
      L.PINNED_ANCHORS_SMOKE)
check("sha matches", hashlib.sha256("|".join(re_full).encode())
      .hexdigest()[:16] == L.ANCHORS_SHA)
strata = {}
for a in L.PINNED_ANCHORS:
    strata.setdefault(TRI.get(a) or "_", []).append(a)
check("per-stratum floor respected",
      all(len(v) >= min(L.STRATUM_MIN,
                        sum(1 for n in frame if (TRI.get(n) or "_") == k))
          for k, v in strata.items()))
check("all strata represented (>= 6 groups)", len(strata) >= 6)
DESCS = {c["name"]: c.get("desc") or "" for c in PACK["concepts"]}
check("every anchor has a real description",
      all(len(DESCS[a]) >= L.DESC_MIN for a in L.PINNED_ANCHORS))
check("wing excluded from anchors",
      not set(L.PINNED_ANCHORS) & set(L.CHOICE_SET))
der = L.wing_derangement()
check("derangement deterministic", der == L.wing_derangement())
check("derangement valid", sorted(der) == sorted(der.values()) ==
      sorted(L.CHOICE_SET) and all(der[c] != c for c in der))

print("== verify_pins tamper teeth ==")
tampered = [dict(c) for c in PACK["concepts"]]
victim = next(c for c in tampered if c["name"] == L.PINNED_ANCHORS[0])
victim["desc"] = "x"
try:
    L.verify_pins(tampered)
    check("tampered pack (desc gutted) is caught", False)
except AssertionError:
    check("tampered pack (desc gutted) is caught", True)

# ── bridge algebra ───────────────────────────────────────────────────────────
print("== hat-matrix LOO == naive refit ==")
rng = np.random.default_rng(7)
X = rng.normal(size=(20, 5))
Y = rng.normal(size=(20, 8))
for lam in (0.1, 1.0, 10.0):
    H = L.hat_matrix(X, lam)
    fast = L.loo_preds_from_hat(H, Y)
    naive = np.zeros_like(Y)
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        W = L.ridge_fit(X[tr], Y[tr], lam)
        p = np.append(X[i], 1.0) @ W
        naive[i] = p / np.linalg.norm(p)
    check(f"exact LOO identity at lam={lam}",
          float(np.max(np.abs(fast - naive))) < 1e-8)
W = L.ridge_fit(X, Y, 1.0)
check("ridge_fit shape (d+1, h)", W.shape == (6, 8))
Xs = X + np.array([100.0, 0, 0, 0, 0])          # intercept absorbs the shift
W2 = L.ridge_fit(Xs, Y, 1.0)
check("intercept unpenalized (shift-tolerant fit)",
      float(np.max(np.abs(np.hstack([Xs, np.ones((20, 1))]) @ W2
                          - np.hstack([X, np.ones((20, 1))]) @ W))) < 1e-4)

print("== planted teeth: linear world passes, permuted codebook fails ==")
rng = np.random.default_rng(11)
n, d, h = 60, 14, 64
Xp = rng.normal(size=(n, d))
Wtrue = rng.normal(size=(d + 1, h))
Yp = np.hstack([Xp, np.ones((n, 1))]) @ Wtrue + 0.15 * rng.normal(size=(n, h))
Yp = Yp / np.linalg.norm(Yp, axis=1, keepdims=True)
Hh = L.hat_matrix(Xp, L.LAM)
own, t1, t5 = L.retrieval_stats(L.loo_preds_from_hat(Hh, Yp), Yp)
tops, meds = L.relabel_null_loo(Xp, Yp, L.LAM, 200, 3, progress_every=0)
p_top1 = (1 + int((tops >= t1).sum())) / 201
check("copy-bridge world: top-1 far above chance", t1 > n * 0.5)
check("copy-bridge world: relabel-null p < .05", p_top1 < 0.05)
pi = rng.permutation(n)
own_s, t1_s, _ = L.retrieval_stats(
    L.loo_preds_from_hat(Hh, Yp[pi]), Yp[pi])
tops_s, _ = L.relabel_null_loo(Xp, Yp[pi], L.LAM, 200, 4, progress_every=0)
p_s = (1 + int((tops_s >= t1_s).sum())) / 201
check("permuted codebook: top-1 at chance-ish", t1_s <= max(3, n * 0.1))
check("permuted codebook: relabel-null p not significant", p_s > 0.05)

print("== A5 wing transfer + G-B teeth ==")
Xa, Ya = Xp[:47], Yp[:47]
Xw = Xp[47:]
Dw = Yp[47:]
Wb, Pw, own_w, near_w, hits = L.wing_transfer(Xa, Ya, Xw, Dw, L.LAM)
null_h, null_m = L.derangement_null_transfer(Wb, Xw, Dw, 200, 5)
gb = L.gate_gb(hits, null_h)
check("linear world: wing transfer hits high", hits >= 9)
check("linear world: G-B passes", gb["pass"])
pi13 = np.roll(np.arange(len(Xw)), 1)
_, _, _, _, hits_p = L.wing_transfer(Xa, Ya, Xw[pi13], Dw, L.LAM)
gb_p = L.gate_gb(hits_p, null_h)
check("permuted wing codebook: G-B fails", not gb_p["pass"])

print("== mantel ==")
A = L.pairwise_ang(Xp[:30])
B = L.pairwise_ang(Yp[:30])
rho_ab, p_ab = L.mantel_spearman(A, B, 500, 6, progress_every=0)
check("correlated structures: mantel p < .05", p_ab < 0.05 and rho_ab > 0)
C = L.pairwise_ang(rng.normal(size=(30, 9)))
rho_c, p_c = L.mantel_spearman(C, B, 500, 7, progress_every=0)
check("independent structures: mantel p large", p_c > 0.05)

print("== axiswise reverse R^2 ==")
r2 = L.axiswise_reverse_r2(Yp, Xp, lam=1.0, k=5, seed=1)
check("linear world: axis R^2 high on all axes",
      all(v > 0.5 for v in r2.values()))
r2n = L.axiswise_reverse_r2(rng.normal(size=(n, h)), Xp, lam=1.0, k=5, seed=1)
check("noise world: axis R^2 low", all(v < 0.3 for v in r2n.values()))

# ── plans (both mode constants) ──────────────────────────────────────────────
print("== Stage-B plans ==")
full_rows = L.build_b_pairs(False)
check("full pair rows: 13 x 2 x 4 x 2 arms = 208", len(full_rows) == 208)
check("full pairs: 104", len({r["pair_id"] for r in full_rows}) == 104)
by_pair = {}
for r in full_rows:
    by_pair.setdefault(r["pair_id"], []).append(r)
check("each pair: one real + one perm, same target/alpha/order",
      all(len(v) == 2 and {v[0]["arm"], v[1]["arm"]} == {"real", "perm"}
          and v[0]["concept"] == v[1]["concept"]
          and v[0]["alpha"] == v[1]["alpha"]
          and v[0]["order"] == v[1]["order"] for v in by_pair.values()))
check("tids unique", len({r["tid"] for r in full_rows}) == len(full_rows))
smoke_rows = L.build_b_pairs(True)
check("smoke pair rows: 2 x 1 x 2 x 2 = 8", len(smoke_rows) == 8)
check("smoke covers one trained + one held-out",
      {r["concept"] for r in smoke_rows} ==
      {L.TRAINED[0], L.HELD_OUT[0]})
titr = L.build_titr_b(False)
check("titration rows: 13 x 2 = 26 at alpha 1.5",
      len(titr) == 26 and all(r["alpha"] == L.TITR_ALPHA_B for r in titr))
check("titration smoke: empty", L.build_titr_b(True) == [])

# ── Stage-B statistics ───────────────────────────────────────────────────────
print("== P-B statistics ==")
def synth_fc(real_exact_targets, perm_exact_targets=(), smoke=False):
    rows = []
    for t in L.build_b_pairs(smoke):
        exact = (t["concept"] in real_exact_targets if t["arm"] == "real"
                 else t["concept"] in perm_exact_targets)
        arg = t["concept"] if exact else (
            der[t["concept"]] if t["arm"] == "perm" else "NONE")
        rows.append({"injected": t["concept"], "arm": t["arm"],
                     "pair_id": t["pair_id"], "exact": exact,
                     "argmax": arg, "none_top": False})
    return rows

rows_win = synth_fc(set(L.TRAINED))
pb = L.pb_matched_perm(rows_win, n_perm=2000, seed=1)
check("planted real-sweep: P-B p tiny",
      pb["p"] < 0.01 and pb["obs_diff"] == 72 and pb["n_pairs"] == 72)
sf = L.pb_signflip_targets(rows_win)
check("planted real-sweep: sign-flip p = 1/512 (5dp rounding)",
      abs(sf["p"] - 1 / 512) < 1e-4)
rows_null = synth_fc(set(), set())
pb0 = L.pb_matched_perm(rows_null, n_perm=2000, seed=2)
check("null rows: P-B p not significant", pb0["p"] > 0.05)
rows_tie = synth_fc(set(L.TRAINED), set(L.TRAINED))
pbt = L.pb_matched_perm(rows_tie, n_perm=2000, seed=3)
check("both-arms-exact: obs diff 0, p large",
      pbt["obs_diff"] == 0 and pbt["p"] > 0.05)
atlas_stub = {"A5_wing": {"per_target": {
    c: {"hit": c in set(L.TRAINED[:5])} for c in L.CHOICE_SET}}}
mech = L.pb_mechanism(rows_win, atlas_stub, smoke=False)
check("mechanism: behavioral hits limited by BEHAV_HIT_MIN and flagged "
      "against geometry",
      set(mech["behavioral_hits"]) == set(L.TRAINED)
      and set(mech["anomalies"]) == set(L.TRAINED) - set(L.TRAINED[:5])
      and not mech["subset_ok"])
mech2 = L.pb_mechanism(synth_fc(set(L.TRAINED[:5])), atlas_stub, smoke=False)
check("mechanism: clean subset passes",
      mech2["subset_ok"] and mech2["behavioral_hits"] == sorted(L.TRAINED[:5]))
s1 = L.s1_heldout_concordance(synth_fc(set()))
check("S1 concordance assembles for both arms on held-out",
      set(s1) == {"real", "perm"}
      and set(s1["real"]) == set(L.HELD_OUT))

print("== stage_b_verdict at BOTH mode constants ==")
sbv_full = L.stage_b_verdict(rows_win, [], atlas_stub, smoke=False)
check("full-mode stage_b_verdict: P_B pass on planted win",
      sbv_full["P_B"]["pass"])
rows_smoke = synth_fc({L.TRAINED[0]}, smoke=True)
sbv_smoke = L.stage_b_verdict(rows_smoke, [], atlas_stub, smoke=True)
check("smoke-mode stage_b_verdict runs (n_pairs 2)",
      sbv_smoke["P_B"]["n_pairs"] == 2)

# ── stage_a_verdict at BOTH mode constants (synthetic small world) ───────────
print("== stage_a_verdict, both mode constants ==")
rng = np.random.default_rng(21)
anch_names = [f"A{i:02d}" for i in range(30)]
vecs = {nm: rng.normal(size=14).tolist() for nm in anch_names}
Wt = rng.normal(size=(15, 24))
def mkdirs(names, noise, seed):
    r = np.random.default_rng(seed)
    out = {}
    for nm in names:
        v = np.append(np.array(vecs[nm]), 1.0) @ Wt
        v = v + noise * r.normal(size=24)
        out[nm] = (v / np.linalg.norm(v)).tolist()
    return out
for c in L.CHOICE_SET:
    vecs[c] = rng.normal(size=14).tolist()
all_names = anch_names + list(L.CHOICE_SET)
d14 = mkdirs(all_names, 0.1, 1)
d20 = mkdirs(all_names, 0.1, 2)
db14 = mkdirs(all_names, 0.1, 3)
import io, contextlib
for smoke_flag in (True, False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        atl = L.stage_a_verdict(anch_names, vecs, d14, d20, db14,
                                smoke=smoke_flag)
    check(f"stage_a_verdict smoke={smoke_flag}: fields + linear world passes",
          atl["P_A"]["pass"] and atl["gb"]["pass"]
          and all(k in atl for k in
                  ("A1_rsa_L14", "A2_loo", "A3_retrieval", "A4_axiswise_r2",
                   "A5_wing", "gb", "S3_L20", "S4_base", "S5_lam_sens",
                   "S6_folded_L14", "wing_pred_real", "derangement")))
check("wing_pred_real: 13 unit vectors (5dp ship rounding; the flight "
      "re-normalizes before injecting)",
      len(atl["wing_pred_real"]) == 13 and all(
          abs(np.linalg.norm(np.array(v)) - 1) < 1e-3
          for v in atl["wing_pred_real"].values()))

# ── fork + verdict assembly ──────────────────────────────────────────────────
print("== fork strings ==")
A_pass = {"P_A": {"pass": True}, "gb": {"pass": True}}
A_pa_only = {"P_A": {"pass": True}, "gb": {"pass": False}}
A_none = {"P_A": {"pass": False}, "gb": {"pass": False}}
A_gb_only = {"P_A": {"pass": False}, "gb": {"pass": True}}
sb_pass = {"P_B": {"pass": True}}
sb_fail = {"P_B": {"pass": False}}
check("fork 1", L.e8j_fork(True, A_pass, sb_pass).startswith("FORK 1"))
check("fork 2", L.e8j_fork(True, A_pa_only, None).startswith("FORK 2"))
check("fork 3", L.e8j_fork(True, A_none, None).startswith("FORK 3"))
check("fork 4", L.e8j_fork(True, A_pass, sb_fail).startswith("FORK 4"))
check("fork edge (P_A fail, G-B pass)",
      L.e8j_fork(True, A_gb_only, sb_pass).startswith("FORK-EDGE"))
check("gate failure -> NO_VERDICT",
      L.e8j_fork(False, A_pass, sb_pass).startswith("NO_VERDICT"))
v = L.e8j_verdict({**A_pass, "stamp": "s", "n_anchors": 3,
                   "wing_pred_real": {"X": [1]}},
                  sb_pass, {"g2": {"pass": True}}, "full", {})
check("verdict assembly drops wing_pred_real from atlas block",
      "wing_pred_real" not in v["atlas"] and v["fork"].startswith("FORK 1"))

# ── single-source rebuild identity ───────────────────────────────────────────
print("== rebuild identity ==")
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
before = {f: sha(os.path.join(HERE, f))
          for f in ("E8J_BRIDGE_UI.ipynb", "e8j_logic.py")}
r = subprocess.run([sys.executable, os.path.join(HERE, "build_e8j_notebook.py")],
                   capture_output=True, text=True)
after = {f: sha(os.path.join(HERE, f))
         for f in ("E8J_BRIDGE_UI.ipynb", "e8j_logic.py")}
check("builder rebuild is byte-identical", r.returncode == 0 and before == after)
nb = json.load(open(os.path.join(HERE, "E8J_BRIDGE_UI.ipynb")))
cell_e8j = next("".join(c["source"]) for c in nb["cells"]
                if "E8J pure logic" in "".join(c["source"]))
tail = open(os.path.join(HERE, "e8j_logic.py")).read()
check("notebook E8J cell is verbatim inside e8j_logic.py", cell_e8j in tail)

n_fail = sum(1 for _, ok in passed if not ok)
print(f"\n{len(passed)} checks, {n_fail} failures")
sys.exit(1 if n_fail else 0)
