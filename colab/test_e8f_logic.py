#!/usr/bin/env python3
"""E8-F pure-logic suite: pins vs the design check, the code + grammar,
row builders at BOTH mode constants, firewalls, stimulus mint probes, and
planted statistical teeth (readable world passes P1/P2/P3 scoring, deranged
world fails, MID-flood scores at null, direction-reader world proves S2's
second reading, NONE-flood drains honestly).

Run:  python3 colab/test_e8f_logic.py
"""
import hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8f_logic as L

passed = []
def check(name, cond):
    passed.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)

PACK = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
VEC = {c["name"]: c["vec"] for c in PACK["concepts"]}
DC = json.load(open(os.path.join(HERE, "results_e8f", "design_check.json")))
PAYLOAD = json.load(open(os.path.join(HERE, "e8f_payload.json")))
NB = json.load(open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb")))
CELLS = ["".join(c["source"]) for c in NB["cells"]]

# ── §1 pins vs design check + notebook identity ─────────────────────────────
check("eval64 == design check split", sorted(DC["s3_split"]["eval"]) == L.EVAL64)
check("train/eval disjoint, 256/64",
      len(L.TRAIN256) == 256 and len(L.EVAL64) == 64
      and not set(L.TRAIN256) & set(L.EVAL64))
check("wing excluded from split",
      not (set(L.TRAIN256) | set(L.EVAL64)) & set(L.CHOICE_SET))
terc_ok = all(abs(L.TERC_LO[ax] - DC["s6_quant"][ax]["terciles"][0]) < 1e-9
              and abs(L.TERC_HI[ax] - DC["s6_quant"][ax]["terciles"][1]) < 1e-9
              for ax in L.AXIS_NAMES_F)
check("tercile literals == design check", terc_ok)
check("sigma literals == design check",
      all(abs(L.SIGMA_F[ax] - DC["s4_alphabet"]["axes"][ax]["sigma"]) < 1e-9
          for ax in L.AXIS_NAMES_F))
check("carried-8 == registered set",
      L.CARRIED8 == ["x", "z", "e", "h", "fx", "fy", "fz", "fh"])
deg = {ax: 0 for ax in L.AXIS_NAMES_F}
for a, b in L.PAIR21:
    deg[a] += 1
    deg[b] += 1
check("PAIR21 is 3-regular, 21 unique",
      len({tuple(p) for p in L.PAIR21}) == 21
      and all(d == 3 for d in deg.values()))
check("draw subsets", set(L.SPOT24) <= set(L.TRAIN256)
      and set(L.TRUEDIR128) <= set(L.TRAIN256)
      and set(L.WPERM16) <= set(L.EVAL64) and set(L.TITR8) <= set(L.EVAL64))
check("AXDERANGE is a derangement of 14",
      sorted(L.AXDERANGE) == list(range(14))
      and all(L.AXDERANGE[j] != j for j in range(14)))
check("eval margins pin (48 ge1 / 26 ge2)",
      sum(1 for v in L.EVAL_MARGINS.values() if v >= 1) == DC["s6b_margins"]["ge1"]
      and sum(1 for v in L.EVAL_MARGINS.values() if v >= 2) == DC["s6b_margins"]["ge2"])
logic_cell = CELLS[2]
check("notebook logic cell == e8f_logic.py byte-verbatim",
      logic_cell == open(os.path.join(HERE, "e8f_logic.py")).read())
check("payload sha pinned in logic + asserted in payload cell",
      L.PAYLOAD_SHA in CELLS[2]
      and "_pl_sha == PAYLOAD_SHA" in CELLS[3] and PAYLOAD["W"] is not None)
pl_sha = hashlib.sha256(json.dumps(
    {k: PAYLOAD[k] for k in sorted(PAYLOAD)}, sort_keys=True,
    separators=(",", ":")).encode()).hexdigest()[:16]
check("payload file sha == pin", pl_sha == L.PAYLOAD_SHA)
atlas_path = os.path.join(HERE, "results_e8j", "full_20260824_1827",
                          "atlas_dirs.json")
check("atlas file sha == pin",
      hashlib.sha256(open(atlas_path, "rb").read()).hexdigest()
      == L.ATLAS_FILE_SHA)
for marker in ("class Injector", "def dirs_stability", "def per_strand_plateau",
               "def train_readout_feat", "def compute_dirs", "def compute_mu",
               "def run_trial_feat", "def build_eval_model"):
    check(f"notebook carries {marker}", any(marker in c for c in CELLS))

# ── §2 the code + grammar ───────────────────────────────────────────────────
check("mu_frame spells all-MID", L.code_levels(L.MU_FRAME) == [0] * 14)
lv = L.code_levels(VEC[L.TRAIN256[0]])
check("code_text/parse round-trip",
      L.parse_feat(L.code_text(lv))["levels"]
      == {L.AXIS_NAMES_F[j]: lv[j] for j in range(14)})
rng = np.random.default_rng(7)
ok_rt = True
for _ in range(50):
    lvr = [int(v) for v in rng.integers(-1, 2, 14)]
    p = L.parse_feat(L.code_text(lvr))
    ok_rt &= p["kind"] == "CODE" and p["n_fields"] == 14 and \
        [p["levels"][ax] for ax in L.AXIS_NAMES_F] == lvr
check("50 random round-trips", ok_rt)
check("NONE-first parse", L.parse_feat("NONE")["kind"] == "NONE"
      and L.parse_feat("  none, nothing unusual")["kind"] == "NONE")
check("NONE after fields = CODE",
      L.parse_feat("x:LOW then NONE")["kind"] == "CODE")
check("INVALID on gibberish", L.parse_feat("hello world")["kind"] == "INVALID")
check("fx does not collide x",
      L.parse_feat("fx:HIGH")["levels"] == {"fx": 1})
check("duplicate field: first wins",
      L.parse_feat("x:LOW x:HIGH")["levels"] == {"x": -1})
check("lowercase + spacing tolerated",
      L.parse_feat("x : low fy:Mid")["levels"] == {"x": -1, "fy": 0})
fc = L.frame_codes(PACK["concepts"])
check("frame = 1865, codes sha == pin",
      len(fc) == 1865 and L.frame_codes_sha(fc) == L.FRAME_CODES_SHA)

# ── §3 stimulus mint + probes ───────────────────────────────────────────────
atlas = json.load(open(atlas_path))
TRUE_DIRS = {}
for n in L.TRAIN256 + L.EVAL64:
    v = np.asarray(atlas["inst14"][n], float)
    TRUE_DIRS[n] = v / np.linalg.norm(v)
DHAT = {w: np.asarray(v, float) for w, v in PAYLOAD["dhat"].items()}
W5 = np.array(PAYLOAD["W"], float)
STIM = L.mint_stimuli(W5, VEC, TRUE_DIRS, DHAT)
check("stim families complete",
      len([k for k in STIM if k.startswith("atom:")]) == 28
      and len([k for k in STIM if k.startswith("pairc:")]) == 84
      and len([k for k in STIM if k.startswith("full:")]) == 320
      and len([k for k in STIM if k.startswith("wperm:")]) == 16
      and len([k for k in STIM if k.startswith("true:")]) == 320
      and len([k for k in STIM if k.startswith("dhat:")]) == 13)
check("all stimuli unit-norm",
      all(abs(np.linalg.norm(v) - 1) < 1e-9 for v in STIM.values()))
pr_ok = all(float(np.max(np.abs(STIM[f"true:{n}"] - np.asarray(pv, float))))
            < 1e-4 for n, pv in PAYLOAD["true_probe"].items())
check("true-dir probes reproduce", pr_ok)
idx = {ax: j for j, ax in enumerate(L.AXIS_NAMES_F)}
atom_ok = True
for ax in L.AXIS_NAMES_F:
    x = np.array(L.MU_FRAME, float)
    x[idx[ax]] += L.ATOM_SCALE * L.SIGMA_F[ax]
    cl = L.code_levels(x)
    atom_ok &= cl[idx[ax]] == 1 and all(cl[j] == 0 for j in range(14)
                                        if j != idx[ax])
check("atoms move exactly their own field (28/28 design-check result)", atom_ok)
pe = L.perm_expected_levels(VEC[L.WPERM16[0]])
inv = {L.AXDERANGE[j]: j for j in range(14)}
check("perm_expected_levels structure",
      pe == [L.level_of(L.AXIS_NAMES_F[k], VEC[L.WPERM16[0]][inv[k]])
             for k in range(14)])

# ── §4 row builders at BOTH mode constants + firewalls ──────────────────────
for smoke in (False, True):
    tr = L.build_e8f_train(VEC, smoke=smoke)
    ev = L.build_e8f_eval(smoke=smoke)
    cnt = {}
    for e in tr:
        cnt[e["strand"]] = cnt.get(e["strand"], 0) + 1
    if not smoke:
        check("full curriculum counts", cnt == L.EXPECT_TRAIN_FULL)
        check("full eval counts", L.eval_counts(ev) == L.EXPECT_EVAL_FULL)
    else:
        check("smoke curriculum covers all strands",
              set(cnt) == set(L.STRANDS_F) and sum(cnt.values()) <= 30)
        check("smoke eval covers all blocks",
              set(L.eval_counts(ev)) == set(L.EXPECT_EVAL_FULL))
    tgt_ok = all(
        (e["target"] == "NONE") == (e["kind"] == "sham")
        and (e["kind"] == "sham"
             or L.parse_feat(e["target"])["n_fields"] == 14)
        for e in tr)
    check(f"targets valid (smoke={smoke})", tgt_ok)
    check(f"firewalls clean (smoke={smoke})",
          not L.validate_no_eval_leak(tr) and not L.validate_no_wing_leak(tr))
    skeys = {e["skey"] for e in tr if e["skey"]} | \
            {r["skey"] for r in ev if r.get("skey")}
    check(f"all row skeys minted (smoke={smoke})",
          all(k in STIM for k in skeys))
bad = L.validate_no_eval_leak([{"eid": 1, "skey": f"full:{L.EVAL64[0]}"}])
check("firewall catches planted eval leak", bad == [1])
bad2 = L.validate_no_wing_leak([{"eid": 2, "skey": "dhat:UNCERTAINTY"}])
check("firewall catches planted wing leak", bad2 == [2])

# §8 revision: full strand carries BOTH alphas per concept
tr_full = [e for e in L.build_e8f_train(VEC, smoke=False)
           if e["strand"] == "full"]
per = {}
for e in tr_full:
    per.setdefault(e["skey"], []).append(e["alpha"])
check("full-strand: both alphas per concept (S8 revision)",
      len(tr_full) == 512 and len(per) == 256
      and all(sorted(v) == L.ALPHA_MAIN for v in per.values()))
check("epoch cap = 12 (S8 revision)", L.EPOCH_CAP_F == 12)

# ── §5 statistical teeth ────────────────────────────────────────────────────
CODES64 = {n: fc[n] for n in L.EVAL64}
ev_full = L.build_e8f_eval(smoke=False)
p1_rows = [r for r in ev_full if r["block"] == "p1"]
p3_rows = [r for r in ev_full if r["block"] == "p3"]
wp_rows = [r for r in ev_full if r["block"] == "wperm"]

def with_parsed(rows, levels_fn):
    out = []
    for r in rows:
        lv = levels_fn(r)
        if lv is None:
            out.append({**r, "parsed": {"kind": "NONE", "levels": {},
                                        "n_fields": 0}})
        else:
            out.append({**r, "parsed": {
                "kind": "CODE",
                "levels": {L.AXIS_NAMES_F[j]: lv[j] for j in range(14)},
                "n_fields": 14}})
    return out

perfect = with_parsed(p1_rows, lambda r: CODES64[r["concept"]])
st = L.p_stats(perfect, CODES64, L.AXIS_NAMES_F, 2000, 99)
check("tooth: perfect world P1 passes",
      st["pooled_acc"] == 1.0 and st["p"] <= L.P_CRIT_F
      and st["n_axes_sig"] >= L.P1_MIN_AXES)
shift = {n: CODES64[sorted(L.EVAL64)[(i + 7) % 64]]
         for i, n in enumerate(sorted(L.EVAL64))}
deranged = with_parsed(p1_rows, lambda r: shift[r["concept"]])
std = L.p_stats(deranged, CODES64, L.AXIS_NAMES_F, 2000, 99)
check("tooth: deranged world fails P1",
      std["p"] > 0.05 and std["n_axes_sig"] < L.P1_MIN_AXES)
midflood = with_parsed(p1_rows, lambda r: [0] * 14)
stm = L.p_stats(midflood, CODES64, L.AXIS_NAMES_F, 2000, 99)
check("tooth: MID-flood scores at null", stm["p"] > 0.2)
noneflood = with_parsed(p3_rows, lambda r: None)
stn = L.p_stats(noneflood, CODES64, L.CARRIED8, 2000, 99)
check("tooth: NONE-flood drains (p -> 1)", stn["p"] > 0.5
      and L.cond_speech_acc(noneflood, CODES64, L.CARRIED8)["n_spoke"] == 0)

p2 = L.p2_stats(perfect, fc, 2000, 99)
uniq_names = {n for n, m in L.EVAL_MARGINS.items() if m >= 1}
check("tooth: perfect world P2 decodes exactly the self-unique 48",
      p2["n_distinct"] == len(uniq_names)
      and set(p2["distinct"]) == uniq_names and p2["p"] <= L.P_CRIT_F
      and p2["exact_rows"] == 2 * len(uniq_names))
p2d = L.p2_stats(deranged, fc, 2000, 99)
check("tooth: deranged world fails P2", p2d["exact_rows"] == 0)

reader = with_parsed(wp_rows, lambda r: L.perm_expected_levels(VEC[r["concept"]]))
s2 = L.s2_stats(reader, VEC, 2000, seed=99)
check("tooth: direction-reader world — S2 second reading",
      s2["vs_permuted"]["pooled_acc"] == 1.0
      and s2["vs_permuted"]["p"] <= 0.01
      and s2["vs_original"]["pooled_acc"] < 1.0)

conf = L.s3_confusion(perfect, CODES64)
check("S3 on perfect world: zero errors",
      conf["errors"] == 0 and conf["spoken_fields"] == 128 * 14)
dens = L.s8_density(perfect, CODES64)
check("S8 groups by density", all(v["mean_acc"] == 1.0 for v in dens.values()))
carrier_rows = with_parsed(
    [r for r in ev_full if r["block"] == "carrier"], lambda r: [0] * 14)
check("S7 counts all-MID", L.s7_carrier(carrier_rows)["all_mid"] == 4)
sham_rows = with_parsed([r for r in ev_full if r["block"] == "sham"],
                        lambda r: None)
check("G5 counts claims", L.sham_claims(sham_rows) == 0
      and L.sham_claims(perfect) == 128)
check("G4 invalid rate",
      L.invalid_rate(sham_rows + perfect)["invalid"] == 0)

# plateau at both mode constants (the R3-c smoke crash class)
sem = [{"carrier": 1.0, "atom": 1.0, "pair": 1.0, "full": 1.0,
        "truedir": 1.0, "sham": 1.0}]
check("plateau: 1 epoch never converges (both constants)",
      not L.per_strand_plateau(sem, 1, 0.05)
      and not L.per_strand_plateau(sem, 2, 0.05))
sem2 = sem + [{k: 0.99 for k in sem[0]}]
check("plateau: flat 2-epoch converges", L.per_strand_plateau(sem2, 2, 0.05))
sem3 = sem + [{k: (0.5 if k == "full" else 0.99) for k in sem[0]}]
check("plateau: one descending strand blocks",
      not L.per_strand_plateau(sem3, 2, 0.05))

# ── §6 rebuild byte-identical ───────────────────────────────────────────────
nb_bytes = open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb"), "rb").read()
logic_bytes = open(os.path.join(HERE, "e8f_logic.py"), "rb").read()
subprocess.run([sys.executable, os.path.join(HERE, "build_e8f_notebook.py")],
               check=True, capture_output=True)
check("rebuild byte-identical (notebook + logic)",
      open(os.path.join(HERE, "E8F_FEATURAL_UI.ipynb"), "rb").read() == nb_bytes
      and open(os.path.join(HERE, "e8f_logic.py"), "rb").read() == logic_bytes)

n_pass = sum(1 for _, ok in passed if ok)
print(f"\n{n_pass}/{len(passed)} checks passed")
sys.exit(0 if n_pass == len(passed) else 1)
