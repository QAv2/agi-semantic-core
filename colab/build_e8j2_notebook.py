#!/usr/bin/env python3
"""Build E8J2_RUNG_UI.ipynb — Part 2 of E8-J v2 (docs/E8J2_PROTOCOL.md §3,
staged under FM1: Part-1 measurement locked, G-M2 passed).

Single-source law:
- The logic cell is `e8j_logic.py` BYTE-VERBATIM (the flown, locked module).
- CELL_MODEL (E8-R2-verbatim scoring/injection machinery) is sliced
  BYTE-VERBATIM from `build_e8j_notebook.py`.
- CELL_SETUP is the E8-J setup with exactly three asserted replacements
  (build name, banner, inflight dir).
- E8J2 additions (pinned d̂ vectors, derangement, geometric hits, verdict/
  fork) are emitted as a separate cell; pins are injected from the LOCKED
  Part-1 artifacts (`results_e8j2/rung_pins.json`), never recomputed here.
- `e8j2_logic.py` = e8j_logic.py + additions (what the notebook executes is
  importable locally for tests and the post-flight recompute).

Run:  python3 colab/build_e8j2_notebook.py
"""
import hashlib, json, os, py_compile, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha_vecs(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# ── donors (byte-verbatim) ───────────────────────────────────────────────────
DONOR_LOGIC = open(os.path.join(HERE, "e8j_logic.py")).read()
E8J_BUILDER = open(os.path.join(HERE, "build_e8j_notebook.py")).read()

def slice_literal(src, name):
    key = f"{name} = r'''"
    i0 = src.index(key) + len(key)
    i1 = src.index("'''", i0)
    return src[i0:i1]

CELL_SETUP_DONOR = slice_literal(E8J_BUILDER, "CELL_SETUP")
CELL_MODEL = slice_literal(E8J_BUILDER, "CELL_MODEL")
for frag in ("class Injector:", "def score_trial(m, layer_mods, dirs14, mu14",
             "def run_trial_free(m, layer_mods, dirs14, mu14",
             "def resolve_layers(m):", "RETENTION_TEXT = (",
             "def compute_mu(m, layers):"):
    assert frag in CELL_MODEL, f"donor drift: {frag!r} not in CELL_MODEL"
for frag in ("def build_b_pairs(smoke=False):", "def stage_b_verdict(",
             "def pb_matched_perm(", "def pb_signflip_targets(",
             "def pb_mechanism(", "def s1_heldout_concordance(",
             "def anchor_rows(smoke=False):", "def sham_rows(smoke=False):",
             "def wing_derangement(seed=E8J_SEED + 2):", "def gate_g1b("):
    assert frag in DONOR_LOGIC, f"donor drift: {frag!r} not in e8j_logic"

CELL_SETUP = CELL_SETUP_DONOR
for old, new in [
        ("NB_BUILD = 'E8J v1 (2026-08-24)'", "NB_BUILD = 'E8J2 v1 (2026-08-24)'"),
        ("print('E8-J notebook build:', NB_BUILD)",
         "print('E8-J2 rung notebook build:', NB_BUILD)"),
        ("INFLIGHT = f'e8j/inflight_", "INFLIGHT = f'e8j2/inflight_")]:
    assert CELL_SETUP.count(old) == 1, f"setup replace target not unique: {old!r}"
    CELL_SETUP = CELL_SETUP.replace(old, new)
assert "e8j2/inflight_" in CELL_SETUP and "E8J2 v1" in CELL_SETUP

# ── pins from the LOCKED Part-1 artifacts ────────────────────────────────────
PINS = json.load(open(os.path.join(HERE, "results_e8j2", "rung_pins.json")))
ART = json.load(open(os.path.join(HERE, "results_e8j2", "wing_reencode_v1.json")))
assert sha_vecs(PINS["dhat_real"]) == PINS["sha"], "rung_pins sha broken on disk"
assert sha_vecs(ART["xhat_L14"]) == ART["sha"], "re-encode artifact sha broken"
assert sorted(PINS["dhat_real"]) == sorted(PINS["derangement"]) \
    == sorted(PINS["geometric_hits"])
assert all(PINS["derangement"][c] != c for c in PINS["derangement"])
GEOM_TRAINED = sorted(c for c, h in PINS["geometric_hits"].items() if h)
print(f"pins loaded: dhat sha {PINS['sha']} | geometric hits {GEOM_TRAINED}")

def jlit(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

CELL_E8J2 = f'''# ── E8J2 additions: PINNED payloads (minted at the Part-1 lock) + verdict ────
# Provenance: docs/E8J2_PROTOCOL.md (prereg 28ad28c, Part-1 lock 68344f0).
# d̂ = forward-bridge predictions from the REVERSE-bridge re-encoded wing
# coordinates (lam=1.0 both ways, 320-anchor fits, instilled L14 atlas dirs).
# The flight injects these pinned vectors — no bridge computation on the VM.
import json
E8J2_PINS_SHA = {PINS["sha"]!r}
E8J2_XHAT_SHA = {ART["sha"]!r}
E8J2_DERANGEMENT = json.loads({jlit(PINS["derangement"])!r})
E8J2_GEOM_HITS = json.loads({jlit(PINS["geometric_hits"])!r})
DHAT_REAL = json.loads({jlit(PINS["dhat_real"])!r})

def _sha_vecs_e8j2(obj):
    import hashlib as _h
    return _h.sha256(json.dumps(obj, sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest()[:16]

def verify_e8j2_pins(dhat=None, der=None, hits=None):
    """G4 pin gate: sha recompute, unit norms, derangement identity vs the
    E8-J pin, key coverage. Args exist for the tamper-teeth tests only."""
    dhat = DHAT_REAL if dhat is None else dhat
    der = E8J2_DERANGEMENT if der is None else der
    hits = E8J2_GEOM_HITS if hits is None else hits
    sha = _sha_vecs_e8j2(dhat)
    assert sha == E8J2_PINS_SHA, f'pin sha mismatch: {{sha}} != {{E8J2_PINS_SHA}}'
    assert sorted(dhat) == sorted(CHOICE_SET), 'dhat keys drifted'
    for c, v in dhat.items():
        n = float(np.linalg.norm(np.asarray(v, float)))
        assert abs(n - 1.0) < 1e-4, f'{{c}} dhat norm {{n}} (6dp-rounded unit)'
    assert der == wing_derangement(), 'derangement drifted vs the E8-J pin'
    assert all(der[c] != c for c in der), 'derangement has a fixed point'
    assert sorted(hits) == sorted(CHOICE_SET), 'geometric-hit map drifted'
    return {{'sha': sha, 'n': len(dhat),
             'geom_hits': sorted(c for c, h in hits.items() if h)}}

def atlas_from_pins():
    """Minimal atlas-shaped dict feeding pb_mechanism: the LOCAL G-M2
    per-target geometric hits (Part-1 lock) stand in for A5."""
    return {{'A5_wing': {{'per_target': {{c: {{'hit': bool(E8J2_GEOM_HITS[c])}}
                                          for c in CHOICE_SET}}}}}}

def e8j2_gates_all(gates):
    return all(gates.get(k, {{}}).get('pass')
               for k in ('g1', 'g1b', 'g1_sham_claims', 'g4_pins'))

def e8j2_fork(gates_all, sb):
    if not gates_all:
        return ('NO_VERDICT (gate failure — primaries withheld per '
                'pre-registration)')
    pb = (sb or {{}}).get('P_B', {{}}).get('pass')
    if pb:
        return ('FJ1 — the rung lands in BOTTLENECK FORM: wing identity '
                'survives the 14D dictionary-register round trip into the '
                "locked readout's naming (codebook-capacity claim, NOT "
                'independent authorship); L3 licensed in bottleneck form')
    return ('FJ2 — geometry round-trips, the readout does not execute it '
            '(bottleneck analog of E8-J fork 4); titration texture localizes')
'''

CELL_FLY = r'''# ── Flight: pins -> model stack -> gates -> rung FC (real vs permuted) ───────
PINS_INFO = verify_e8j2_pins()
print('E8J2 pins verified:', PINS_INFO)
DER = E8J2_DERANGEMENT

B_PAIR_ROWS = build_b_pairs(SMOKE)
TITR_B_ROWS = build_titr_b(SMOKE)
ANCHOR_ROWS = anchor_rows(SMOKE)
SHAM_ROWS = sham_rows(SMOKE)
print(f'plan: rung pairs {len(B_PAIR_ROWS)} rows | titr {len(TITR_B_ROWS)} | '
      f'anchor {len(ANCHOR_ROWS)} | shams {len(SHAM_ROWS)}')

RESULTS, cond_errors = {}, {}
if RESUME_STAMP and not SMOKE:
    _p = SEM / INFLIGHT / 'rungb.json'
    if _p.exists():
        RESULTS['rungb'] = json.load(open(_p))
        print(f'rungb: RESUMED from Drive ({RESUME_STAMP})')

def fly():
    """The whole flight in ONE function scope (lane law: model, handles,
    activations die on return). Eval-only; injects PINNED vectors. Gate
    failures abort loudly — scoring a mis-reconstructed model is worthless."""
    t0 = time.time()
    gates = {}
    print(f'loading base model — {ram_report()}')
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    m.eval()
    print('  merging E4-real instillation adapter...')
    m = PeftModel.from_pretrained(m, str(ADAPTER_REAL)).merge_and_unload()
    m.eval()
    mu = compute_mu(m, [14])
    _mu_ship = float(SRC_REAL['mu']['14'])
    assert abs(mu[14] / _mu_ship - 1.0) < 1e-3, (
        f'mu drift vs shipped: {mu[14]} vs {_mu_ship} — stimulus model is off')
    print(f'  mu L14 {mu[14]:.2f} (shipped {_mu_ship:.2f}) — stack verified')
    print('  attaching locked readout LoRA...')
    m = PeftModel.from_pretrained(m, str(READOUT_REAL))
    m.eval()
    assert not any(p.requires_grad for p in m.parameters()), 'eval-only flight'
    layer_mods = resolve_layers(m)
    gates['g1b'] = gate_g1b(retention_ppl(m), SRC_REAL['ppl_post'])
    assert gates['g1b']['pass'], (
        f'G1b reconstruction failed: {gates["g1b"]} — not the flight model')
    print(f'  G1b PASS (ppl delta {gates["g1b"]["delta_pct"]:+.3f}%) — '
          f'{ram_report()}')

    # Injection maps: anchors/shams ride the FLIGHT-OF-RECORD dirs from the
    # locked E8-R bundle; the rung arms inject the PINNED bridge predictions.
    stim14, pred_real = {}, {}
    for n in CHOICE_SET:
        v = np.asarray(SRC_REAL['dirs']['14'][n], float)
        stim14[n] = v / np.linalg.norm(v)
        w = np.asarray(DHAT_REAL[n], float)
        pred_real[n] = w / np.linalg.norm(w)
    pred_perm = {c: pred_real[DER[c]] for c in CHOICE_SET}

    def run_block(name, rows, dirs_map, fn):
        out_rows = []
        for i, t in enumerate(rows):
            r = fn(m, layer_mods, dirs_map, mu[14], t, t.get('arm', name))
            for k in ('pair_id', 'arm', 'block'):
                if k in t:
                    r[k] = t[k]
            out_rows.append(r)
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                print(f'    {name}: {i + 1}/{len(rows)} '
                      f'({time.time() - t0:.0f}s)')
        return out_rows

    anchor_scored = run_block('anchor', ANCHOR_ROWS, stim14, score_trial)
    g1_exact = sum(1 for r in anchor_scored if r.get('exact'))
    print(f'  anchor exact {g1_exact}/{len(anchor_scored)}')
    sham_scored = run_block('shams', SHAM_ROWS, stim14, score_trial)
    claims = sum(1 for r in sham_scored if not r['none_top'])
    print(f'  sham claims {claims}/{len(sham_scored)}')
    real_rows = [t for t in B_PAIR_ROWS if t['arm'] == 'real']
    perm_rows = [t for t in B_PAIR_ROWS if t['arm'] == 'perm']
    scored = (run_block('rung_real', real_rows, pred_real, score_trial)
              + run_block('rung_perm', perm_rows, pred_perm, score_trial))
    titr_scored = (run_block('rung_titr', TITR_B_ROWS, pred_real,
                             run_trial_free) if TITR_B_ROWS else [])
    rungb = {'stamp': STAMP, 'mode': MODE, 'pins': PINS_INFO,
             'anchor': anchor_scored, 'shams': sham_scored,
             'sham_claims': claims, 'fc_rows': scored,
             'titr_rows': titr_scored, 'derangement': DER,
             'gates_snapshot': gates, 'secs': round(time.time() - t0, 1)}
    fn_out = OUT / 'rungb.json'
    jdump(rungb, fn_out)
    ship(fn_out, INFLIGHT)
    print('  rung bundle shipped')
    return rungb, gates

if RESULTS.get('rungb'):
    print('flight complete on Drive — verdict-only run')
    RUNGB = RESULTS['rungb']
    GATES = RUNGB.get('gates_snapshot')
else:
    try:
        RUNGB, GATES = fly()
    except Exception as e:
        cond_errors['flight'] = f'{type(e).__name__}: {e}'
        print(f'!! FLIGHT FAILED: {cond_errors["flight"]}')
        RUNGB, GATES = None, None
    free_ram()
    print('torn down —', ram_report())
'''

CELL_VERDICT = r'''# ── Gates, primary P-B', fork banner, ship ───────────────────────────────────
if RUNGB is None:
    print('NO FLIGHT DATA — see cond_errors above; nothing to verdict.')
    print('cond_errors:', cond_errors)
else:
    gates = dict(GATES or {})
    if not SMOKE:
        gates['g1'] = gate_g1(RUNGB['anchor'])
    else:
        gates['g1'] = {'n': len(RUNGB['anchor']),
                       'exact': sum(1 for r in RUNGB['anchor'] if r.get('exact')),
                       'pass': True, 'smoke': True}
    gates['g1_sham_claims'] = {'claims': RUNGB['sham_claims'],
                               'n': len(RUNGB['shams']),
                               'pass': bool(RUNGB['sham_claims'] <= 1)}
    hooks_ok = all(r['hook_calls'] >= 1 for r in RUNGB['fc_rows'])
    shams_ok = all(r['hook_calls'] == 0 for r in RUNGB['shams'])
    parse_fails = sum(1 for r in RUNGB['titr_rows'] if not r.get('report'))
    gates['g4_pins'] = {'pass': bool(RUNGB.get('pins') and hooks_ok and shams_ok
                                     and len(RUNGB['fc_rows']) == len(B_PAIR_ROWS)),
                        'hooks_ok': hooks_ok, 'shams_uninjected': shams_ok,
                        'fc_rows': len(RUNGB['fc_rows']),
                        'expected': len(B_PAIR_ROWS),
                        'titr_parse_fails': parse_fails,
                        **(RUNGB.get('pins') or {})}
    sb = stage_b_verdict(RUNGB['fc_rows'], RUNGB['titr_rows'],
                         atlas_from_pins(), smoke=SMOKE)
    gates_all = e8j2_gates_all(gates)
    summary = {'exp': 'E8J2_RUNG', 'mode': MODE, 'stamp': STAMP,
               'model': MODEL_ID, 'pins': RUNGB.get('pins'), 'gates': gates,
               'gates_all': bool(gates_all), 'stageb': sb,
               'cond_errors': cond_errors, 'fork': e8j2_fork(gates_all, sb)}
    jdump(summary, OUT / 'e8j2_verdict.json')
    ship(OUT, f'e8j2/{MODE}_{STAMP}')

    def _no_rows(o):
        if isinstance(o, dict):
            return {k: _no_rows(v) for k, v in o.items()
                    if k not in ('scores', 'scores_sum')}
        if isinstance(o, list):
            return [_no_rows(x) for x in o]
        return o
    print(json.dumps(_no_rows(summary), indent=1, default=str))

    if SMOKE:
        checks = {
            'no_errors': not cond_errors,
            'pins_verified': bool(RUNGB.get('pins')),
            'g1b_pass': bool(gates.get('g1b', {}).get('pass')),
            'rows_complete': len(RUNGB['fc_rows']) == len(B_PAIR_ROWS),
            'hooks_fired': hooks_ok,
            'shams_uninjected': shams_ok,
            'score_vectors_complete': all(set(r['scores']) == set(SCORED_SET)
                                          for r in RUNGB['fc_rows']),
            'pb_stats_ran': sb['P_B']['n_pairs'] ==
                len({t['pair_id'] for t in B_PAIR_ROWS
                     if t['concept'] in TRAINED}),
            'mechanism_row': 'mechanism' in sb,
            'shipped': (SEM / INFLIGHT / 'rungb.json').exists(),
        }
        ok = all(checks.values())
        print('smoke checks:', json.dumps(checks, indent=1))
        banner = ('SMOKE GREEN — set SMOKE=False, Runtime > Restart runtime, '
                  'Run all. Full flight is ONE run (~20-30 min).'
                  if ok else 'SMOKE RED — do not fly full; send Fable the output')
        print('\n' + '=' * 66 + f'\n  {banner}\n' + '=' * 66)
    else:
        print('\n' + '=' * 66)
        print('  E8-J v2 RUNG FLIGHT — verdict shipped to MyDrive/semcore/e8j2/')
        print(f"  P-B' naming: real {sb['P_B']['real_exact']} vs "
              f"perm {sb['P_B']['perm_exact']} "
              f"(n_pairs {sb['P_B']['n_pairs']}, p={sb['P_B']['p']})")
        print(f"  sign-flip p={sb['P_B_signflip']['p']} | mechanism "
              f"subset_ok={sb['mechanism']['subset_ok']} "
              f"anomalies={sb['mechanism']['anomalies']}")
        _s1 = sb['S1_heldout_concordance']
        for _arm in ('real', 'perm'):
            _cc = sum(v['concord'] for v in _s1[_arm].values())
            _nn = sum(v['n'] for v in _s1[_arm].values())
            print(f"  S1' held-out modal concordance ({_arm}): {_cc}/{_nn}")
        print(f"  FORK: {summary['fork']}")
        print(f"  RESUME_STAMP if needed: '{RESUME_STAMP or STAMP}'")
        print('=' * 66)
'''

MD0 = """# E8-J v2 — the rung re-fly on re-encoded coordinates (UI-only flight)

**Protocol**: `docs/E8J2_PROTOCOL.md` (prereg 28ad28c · Part-1 lock 68344f0,
FM1: mis-encoding measured p=.0001, G-M2 round-trip 4/13 PASS).
**This flight**: eval-only. Injects PINNED forward-bridge predictions from
the reverse-bridge re-encoded wing coordinates into the locked E8-R real
readout — real arm vs derangement-permuted arm, matched pairs. No training,
no on-VM bridge computation.

**How to fly** (same as every flight):
1. Runtime → Change runtime type → **T4 GPU** → Save.
2. **Run all** with `SMOKE = True` (~8-12 min) → wait for the GREEN banner.
3. Runtime → **Restart runtime** → set `SMOKE = False` → **Run all**
   (~20-30 min, one run).
4. If a full run is interrupted: paste the banner's stamp into
   `RESUME_STAMP`, Restart, Run all — finished bundles resume from Drive.

Results land in `MyDrive/semcore/e8j2/` (inflight + final verdict)."""

# ── e8j2_logic.py: donor + additions (what the notebook executes) ────────────
E8J2_LOGIC = DONOR_LOGIC + "\n\n" + CELL_E8J2
logic_out = os.path.join(HERE, "e8j2_logic.py")
with open(logic_out, "w") as f:
    f.write(E8J2_LOGIC)
py_compile.compile(logic_out, doraise=True)
r = subprocess.run([sys.executable, "-c",
                    "import sys; sys.path.insert(0,'colab'); "
                    "import e8j2_logic as L2; print(L2.verify_e8j2_pins())"],
                   capture_output=True, text=True, cwd=os.path.dirname(HERE))
assert r.returncode == 0, f"e8j2_logic import/verify failed:\n{r.stderr}"
print("e8j2_logic.py written; pins verify in fresh interpreter:",
      r.stdout.strip().splitlines()[-1])

# ── assemble notebook ────────────────────────────────────────────────────────
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "cells": []}

def add(kind, src):
    cell = {"cell_type": kind, "metadata": {},
            "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)

add("markdown", MD0)
add("code", CELL_SETUP)
add("code", DONOR_LOGIC)
add("code", CELL_E8J2)
add("code", CELL_MODEL)
add("code", CELL_FLY)
add("code", CELL_VERDICT)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    assert "rclone" not in src.lower(), f"cell {i} mentions rclone (UI-only law)"
    if cell["cell_type"] == "code":
        compile(src, f"<cell {i}>", "exec")
print("cells compile; zero rclone anywhere")

out_nb = os.path.join(HERE, "E8J2_RUNG_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)
sz = os.path.getsize(out_nb)
print(f"wrote {out_nb} ({len(nb['cells'])} cells, {sz/1e3:.0f}KB)")
