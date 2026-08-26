#!/usr/bin/env python3
"""Build E7BQ_WALK_UI.ipynb — the E7b-Q instrumented-walk flight (UI lane).

Single-source: e7bq_logic.py is emitted VERBATIM as the logic cell;
e7bq_payload.json (design-check output, sha-pinned) is embedded as a literal.
Colab UI-only law: zero rclone, drive.mount only, de-shelled installs.

Run:  python3 colab/build_e7bq_notebook.py
"""
import base64
import hashlib
import json
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

LOGIC_SRC = open(os.path.join(HERE, "e7bq_logic.py")).read()
PAYLOAD_RAW = open(os.path.join(HERE, "e7bq_payload.json"), "rb").read()
PAYLOAD_SHA = hashlib.sha256(PAYLOAD_RAW).hexdigest()[:16]

# sanity: the payload parses and carries every pinned block
_p = json.loads(PAYLOAD_RAW)
for k in ("script", "sham_slots", "reorder_perm", "windows", "probes",
          "probe_dirs", "being_vec", "xbar_pack", "cloud_mean", "encoders",
          "script_sha", "R"):
    assert k in _p, f"payload missing {k}"
assert len(_p["script"]) == 15 and _p["R"] == 16

MD0 = f"""# E7b-Q — The Instrumented Walk (UI flight, EVAL-ONLY)

The corpus's own Neti-Neti walk (Nov-2025, verbatim turns) flown on
Qwen2.5-1.5B-Instruct, base vs E4-real-instilled, with per-turn hidden
states projected through the locked E8-J atlas encoders into the 14D
register. Primaries: P-W1 contraction (walked vs sham, D_BEING) ·
P-W2 co-tracking (state x output-thinning, within-turn). Pre-registered
`docs/E7BQ_PROTOCOL.md`; payload sha `{PAYLOAD_SHA}`.

**Flow (T4 GPU runtime):**
1. Run all with `SMOKE = True` (~10-16 min). Wait for the GREEN banner.
2. **Runtime > Restart runtime** (mandatory between runs — RAM law).
3. Set `SMOKE = False`, Run all (~100-140 min, generation-dominated;
   both conditions in one run, per-condition inflight shipping).
4. If the VM dies mid-full-run: note the banner's RESUME_STAMP, restart,
   paste it into `RESUME_STAMP`, Run all — completed conditions are
   adopted from Drive, only the missing one re-flies.

Results ship to `MyDrive/semcore/e7bq/`. If the first printed line's
build tag differs from the Desktop copy you staged, you are on a stale
upload — File > Upload notebook > pick the Desktop file again.

EVAL-ONLY: no training, no injection; the walk never enters any training
set (firewall law). Retrieval after the flight: Drive integration only.
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, E4 adapter ─────────────
NB_BUILD = 'E7BQ v4 (2026-08-26, two-band G-DIRS + pack pin)'
print('E7b-Q notebook build:', NB_BUILD)

SMOKE = True                   # first run: smoke. Then False for the full flight.
RESUME_STAMP = ''              # paste a full-run stamp only to resume it
PACK_CANON_SHA = '__PACK_CANON_SHA__'  # builder-injected G-PACK pin

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes, hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'

gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')

print('Installing packages...')
subprocess.run([sys.executable,'-m','pip','uninstall','-q','-y','torchao'], check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','-U',
    'transformers>=4.44','peft>=0.11','accelerate'], check=True)

import torch
assert torch.cuda.is_available(), 'No GPU — Runtime > Change runtime type > T4 GPU.'
DEV = 'cuda'

def _mem_avail_gb():
    try:
        kb = int(next(l for l in open('/proc/meminfo')
                      if l.startswith('MemAvailable')).split()[1])
        return kb / 1e6
    except Exception:
        return float('nan')

def free_ram():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

def ram_report():
    g = torch.cuda.mem_get_info()
    return (f'sys avail {_mem_avail_gb():.1f}GB | '
            f'GPU free {g[0]/1e9:.1f}/{g[1]/1e9:.1f}GB')

_leftover = torch.cuda.memory_allocated()
assert _leftover < 5e8, (
    f'GPU already holds {_leftover/1e9:.1f}GB from a previous run in this '
    'kernel — this flight needs a fresh one. Runtime > Restart runtime, '
    'then Run all.')
_avail = _mem_avail_gb()
assert not (_avail < 6.5), (
    f'Only {_avail:.1f}GB system RAM available (need 6.5). Runtime > Restart '
    'runtime; if it trips again, Runtime > Disconnect and delete runtime for '
    'a fresh VM, then Run all.')
print('RAM at start:', ram_report())

from google.colab import drive
drive.mount('/content/drive')
SEM = Path('/content/drive/MyDrive/semcore')
assert SEM.exists(), 'MyDrive/semcore not found — mounted the right Google account?'

def ship(src, dest_rel):
    dest = SEM / dest_rel
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    files = sorted(p for p in src.iterdir() if p.is_file()) if src.is_dir() else [src]
    for p in files:
        shutil.copy2(p, dest / p.name)

PACK = Path('/content/e4_dictionary_pack.json')
if not PACK.exists():
    shutil.copy2(SEM / 'e4/e4_dictionary_pack.json', PACK)
pack = json.load(open(PACK))
# G-PACK: canonical-json sha vs the build-time pin — desc/pack drift aborts
# here deterministically (this axis used to ride on G-DIRS numerics)
_pack_canon = json.dumps(pack, sort_keys=True,
                         separators=(',', ':')).encode('utf-8')
assert hashlib.sha256(_pack_canon).hexdigest()[:16] == PACK_CANON_SHA, (
    'G-PACK FAIL: Drive pack differs semantically from the build-time pack '
    '— desc drift would silently shift probes/centroid; re-ship the pack')
print('pack:', pack['name'], '| concepts', pack['n_concepts'],
      '| G-PACK', PACK_CANON_SHA)
DESC = {c['name']: c['desc'] for c in pack['concepts']}

# E4 instillation adapter — REAL condition
cand = sorted(d.name for d in (SEM / 'e4').iterdir()
              if d.is_dir() and d.name.startswith('real_full_'))
assert cand, 'no real_full_* dir under semcore/e4'
_src = SEM / 'e4' / cand[-1] / 'adapter_real'
ADAPTER_REAL = Path('/content/adapter_real')
shutil.copytree(_src, ADAPTER_REAL, dirs_exist_ok=True)
assert (ADAPTER_REAL / 'adapter_config.json').exists(), 'adapter_real incomplete'
print('instillation adapter real:', cand[-1])

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
INFLIGHT = f'e7bq/inflight_{RESUME_STAMP or STAMP}'
if RESUME_STAMP:
    _rd = SEM / INFLIGHT
    assert _rd.exists(), (
        f'RESUME_STAMP={RESUME_STAMP!r} but {_rd} does not exist on Drive — '
        'check the stamp string (copy it exactly; no spaces). A silent '
        'fallback here would re-fly finished conditions.')
    print('resume dir found; files present:',
          sorted(p.name for p in _rd.iterdir()) or 'NONE')
print('MODE:', MODE.upper(), '| stamp', STAMP,
      ('| RESUMING ' + RESUME_STAMP) if RESUME_STAMP else '')
'''

_pack_local = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
PACK_CANON_SHA = hashlib.sha256(json.dumps(
    _pack_local, sort_keys=True,
    separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
CELL_SETUP = CELL_SETUP.replace("__PACK_CANON_SHA__", PACK_CANON_SHA)
assert "__PACK_CANON_SHA__" not in CELL_SETUP

# zlib+b64 transport, chunked into short lines. A verbatim embed of the
# pretty-printed payload put 115,537 source lines / 2.5MB into this one cell
# and the Colab editor could not open the notebook on a 3.7GB-RAM machine
# (2026-08-26). The sha pin is asserted on the DECODED bytes, so it stays
# byte-identical to the design-check output file.
_PB64 = base64.b64encode(zlib.compress(PAYLOAD_RAW, 9)).decode("ascii")
_PB64_LINES = "".join(
    "    '" + _PB64[i:i + 1900] + "'\n" for i in range(0, len(_PB64), 1900))

CELL_PAYLOAD = (
    "# ── Pinned payload (design-check output; zlib+b64 transport; sha asserted\n"
    "#    on the decoded bytes = byte-identical to the design-check file) ────────\n"
    "PAYLOAD_SHA_PIN = '" + PAYLOAD_SHA + "'\n"
    "_PAYLOAD_B64 = (\n" + _PB64_LINES + ")\n"
) + r'''import base64 as _b64, zlib as _zl, hashlib as _h, json as _j
_raw = _zl.decompress(_b64.b64decode(_PAYLOAD_B64))
assert _h.sha256(_raw).hexdigest()[:16] == PAYLOAD_SHA_PIN, \
    'payload sha mismatch — stale notebook upload'
PAYLOAD = _j.loads(_raw)
print('payload OK:', PAYLOAD_SHA_PIN, '| script sha', PAYLOAD['script_sha'],
      '| R', PAYLOAD['R'], '| transport',
      f'{len(_PAYLOAD_B64)//1024}KB b64 -> {len(_raw)//1024}KB json')
'''

CELL_PLAN = r'''# ── Plan + G-SCRIPT + ledger self-test ───────────────────────────────────────
assert script_sha(PAYLOAD['script']) == PAYLOAD['script_sha'], 'G-SCRIPT FAIL'
ledger_selftest()
print('G-SCRIPT pass | thinning-ledger self-test pass')
PLAN, PLAN_SHA = build_plan(PAYLOAD, SMOKE)
MC = mode_consts(SMOKE)
_exp = expected_counts(SMOKE)
_got = {}
for r in PLAN:
    _got[(r['cond'], r['arm'])] = _got.get((r['cond'], r['arm']), 0) + 1
for c in CONDS:
    for a, n in _exp.items():
        assert _got[(c, a)] == n, (c, a, _got[(c, a)], n)
print(f'plan: {len(PLAN)} generations | sha {PLAN_SHA} | per-cond', _exp)
print(f'mode consts: R={MC["R"]} cap={MC["cap"]} perms={MC["n_perm"]}')
'''

CELL_FLIGHT = r'''# ── Flight: per-condition function scope -> capture -> inflight ship ─────────
import numpy as _np
import transformers as _tfm
import peft as _peftm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ENV = {'torch': torch.__version__, 'transformers': _tfm.__version__,
       'peft': _peftm.__version__, 'cuda': torch.version.cuda,
       'gpu': torch.cuda.get_device_name(0)}
print('env:', ENV)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Frozen stimulus machinery (E7-Q/E8-R code path — the same 256-name centroid)
E7Q_SEED = 20260822
rng_dir = _np.random.default_rng(E7Q_SEED)
CENT_NAMES = list(rng_dir.choice([c['name'] for c in pack['concepts']],
                                 size=256, replace=False))

def pooled_reps(m, names, layer, bs=32):
    """Mean-pooled hidden_states[layer] of the E4 text rendering 'NAME: desc'."""
    texts = [f"{n}: {DESC[n]}" if DESC.get(n) else n for n in names]
    reps = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=64,
                      return_tensors='pt').to(DEV)
            out = m(**enc, output_hidden_states=True)
            h = out.hidden_states[layer]
            mk = enc.attention_mask.unsqueeze(-1).to(h.dtype)
            reps.append(((h * mk).sum(1) / mk.sum(1).clamp(min=1)).float().cpu())
    return torch.cat(reps).numpy()

def compute_dirs(m, layers, names):
    """Directions: pooled rep minus the 256-name centroid, unit-normalized —
    the E7-Q/E8-R formula verbatim. Returns (dirs, cent) per layer."""
    dirs, cents = {}, {}
    for L in layers:
        reps = pooled_reps(m, names, L)
        cent = pooled_reps(m, CENT_NAMES, L).mean(0)
        d = reps - cent
        d = d / _np.linalg.norm(d, axis=1, keepdims=True)
        dirs[L] = {n: d[i] for i, n in enumerate(names)}
        cents[L] = cent
    return dirs, cents

PROBE_KEY = {('base', 14): 'base14', ('real', 14): 'inst14',
             ('real', 20): 'inst20'}

def step_entropy(score_row):
    s = score_row[0].float()
    finite = torch.isfinite(s)
    p = torch.softmax(s[finite], dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum())

def run_turn(m, tok, msgs, seed, cap, layers):
    """One conversation turn: capture s_pre (last template position, BEFORE
    the reply exists), seeded sampled generation, entropy per kept step,
    s_gen (mean over the reply's positions). Pure model mechanics — sliced
    verbatim into the CPU capture suite."""
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_dict=True, return_tensors='pt')
    ids = enc['input_ids'].to(DEV)
    att = enc['attention_mask'].to(DEV)
    with torch.no_grad():
        pre = m(input_ids=ids, attention_mask=att, output_hidden_states=True)
    s_pre = {L: pre.hidden_states[L][0, -1].float().cpu().numpy()
             for L in layers}
    del pre
    torch.manual_seed(seed)
    with torch.no_grad():
        gen = m.generate(input_ids=ids, attention_mask=att,
                         do_sample=True, temperature=GEN_TEMPERATURE,
                         top_p=GEN_TOP_P, top_k=GEN_TOP_K,
                         max_new_tokens=cap,
                         pad_token_id=tok.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    new_ids = gen.sequences[0, ids.shape[1]:]
    eos_pos = (new_ids == tok.eos_token_id).nonzero()
    eos_hit = bool(len(eos_pos))
    keep = new_ids[:int(eos_pos[0])] if eos_hit else new_ids
    cap_hit = (not eos_hit) and (len(new_ids) == cap)
    text = tok.decode(keep, skip_special_tokens=True)
    ents = [step_entropy(gen.scores[k]) for k in range(len(keep))]
    del gen
    if len(keep):
        full_ids = torch.cat([ids, keep.unsqueeze(0)], dim=1)
        full_att = torch.ones_like(full_ids)
        with torch.no_grad():
            post = m(input_ids=full_ids, attention_mask=full_att,
                     output_hidden_states=True)
        s_gen = {L: post.hidden_states[L][0, ids.shape[1]:].mean(0)
                 .float().cpu().numpy() for L in layers}
        del post
    else:
        s_gen = None
    return {'text': text, 'n_new': int(len(keep)), 'eos_hit': eos_hit,
            'cap_hit': cap_hit, 'ents': ents, 's_pre': s_pre, 's_gen': s_gen}

def fly_condition(cond):
    """One condition end-to-end in ONE function scope (lane law: model and
    activations die on return)."""
    t0 = time.time()
    layers = [14, 20]
    print(f'[{cond}] loading base model — {ram_report()}')
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    if cond == 'real':
        print(f'[{cond}] merging E4-real instillation adapter...')
        m = PeftModel.from_pretrained(m, str(ADAPTER_REAL)).merge_and_unload()
    m.eval()
    m.requires_grad_(False)  # eval() never touches requires_grad; fresh loads have it True
    assert not any(p.requires_grad for p in m.parameters()), 'eval-only flight'

    print(f'[{cond}] centroid + G-DIRS probes...')
    probe_dirs, cents = compute_dirs(m, layers, list(PAYLOAD['probes']))
    gdirs = {}
    for L in layers:
        key = PROBE_KEY.get((cond, L))
        if key is None:
            continue
        resid = max(float(_np.max(_np.abs(
            probe_dirs[L][p] - _np.asarray(PAYLOAD['probe_dirs'][key][p]))))
            for p in PAYLOAD['probes'])
        gdirs[str(L)] = resid
        print(f'  G-DIRS {cond} L{L}: resid {resid:.2e} '
              f'(noise {DIRS_TOL} | hard {DIRS_HARD})')
        assert resid <= DIRS_HARD, (
            f'G-DIRS HARD FAIL {cond} L{L}: {resid:.2e} — wrong adapter/'
            'layer scale; do NOT edit cells in place, re-stage per the VM law')
        if resid > DIRS_TOL:
            print(f'  ! G-DIRS {cond} L{L} above the noise class — cross-era '
                  'stack drift; recorded in bundle, judged at recompute')

    enc14 = encoder_for(PAYLOAD, cond, 14)
    enc20 = encoder_for(PAYLOAD, cond, 20)
    cm14 = cloud_mean_for(PAYLOAD, cond, 14)
    rows = []
    plan_rows = [r for r in PLAN if r['cond'] == cond]
    n_done = 0
    hist_key, msgs = None, []
    for r in plan_rows:
        if (r['arm'], r['rep']) != hist_key:
            hist_key, msgs = (r['arm'], r['rep']), []
        msgs.append({'role': 'user', 'content': r['prompt']})
        t = run_turn(m, tok, msgs, r['seed'], MC['cap'], layers)

        def _chat(h, enc_w, L):
            if enc_w is None or h is None:
                return None
            v = h - cents[L]
            v = v / max(float(_np.linalg.norm(v)), 1e-12)
            return [round(float(x), 5) for x in chat_apply(enc_w, v)]

        s_pre, s_gen, ents = t['s_pre'], t['s_gen'], t['ents']
        v14 = s_pre[14] - cents[14]
        r14 = float(_np.linalg.norm(v14))
        u14 = v14 / max(r14, 1e-12)
        row = {'cond': cond, 'arm': r['arm'], 'rep': r['rep'],
               'turn': r['turn'], 'tag': r['tag'], 'text': t['text'],
               'n_new': t['n_new'], 'eos': t['eos_hit'],
               'cap_hit': t['cap_hit'],
               'vis_mass': int(visible_mass(t['text'])),
               'ent_mean': (round(float(_np.mean(ents)), 4) if ents else None),
               'ent_first': (round(ents[0], 4) if ents else None),
               'chat_pre14': _chat(s_pre[14], enc14, 14),
               'chat_pre20': _chat(s_pre[20], enc20, 20),
               'chat_gen14': _chat(s_gen[14] if s_gen else None, enc14, 14),
               'chat_gen20': _chat(s_gen[20] if s_gen else None, enc20, 20),
               'radius_pre14': round(r14, 4),
               'cone_pre14': (round(float(u14 @ cm14), 5)
                              if cm14 is not None else None),
               's_pre14': ([round(float(x), 5) for x in s_pre[14]]
                           if r['keep_state'] else None)}
        rows.append(row)
        msgs.append({'role': 'assistant', 'content': t['text']})
        n_done += 1
        if n_done % 30 == 0:
            print(f'  [{cond}] {n_done}/{len(plan_rows)} rows '
                  f'({time.time()-t0:.0f}s) — {ram_report()}')
    bundle = {'stamp': STAMP, 'mode': MODE, 'cond': cond,
              'plan_sha': PLAN_SHA, 'payload_sha': PAYLOAD_SHA_PIN,
              'env': dict(ENV, model_rev=getattr(m.config, '_commit_hash',
                                                 None)),
              'gdirs': gdirs,
              'centroid': {str(L): [round(float(x), 5) for x in cents[L]]
                           for L in layers},
              't_sec': round(time.time() - t0, 1), 'rows': rows}
    fn = OUT / f'condition_{cond}.json'
    jdump(bundle, str(fn))
    ship(fn, INFLIGHT)
    print(f'[{cond}] DONE {len(rows)} rows in {bundle["t_sec"]:.0f}s — '
          f'shipped inflight')
    return bundle

BUNDLES, cond_errors = {}, {}
for _cond in CONDS:
    _resume_fn = SEM / INFLIGHT / f'condition_{_cond}.json'
    if RESUME_STAMP and _resume_fn.exists():
        _b = json.load(open(_resume_fn))
        _n_want = sum(expected_counts(SMOKE).values())
        if (not _b.get('cond_error') and len(_b.get('rows', [])) == _n_want
                and _b.get('payload_sha') == PAYLOAD_SHA_PIN):
            BUNDLES[_cond] = _b
            print(f'[{_cond}] RESUMED from Drive ({len(_b["rows"])} rows)')
            continue
        print(f'[{_cond}] resume bundle incomplete/errored — re-flying')
    try:
        BUNDLES[_cond] = fly_condition(_cond)
    except Exception as e:
        # record + STOP FLYING, but fall through to the verdict cell so the
        # banner and every shipped piece reach Drive (smoke-3 lane lesson:
        # the flight ships its evidence even on error; a gate failure on
        # one condition also makes flying the next one worthless)
        import traceback
        cond_errors[_cond] = f'{type(e).__name__}: {e}'
        print(f'[{_cond}] ERROR: {cond_errors[_cond]}')
        traceback.print_exc()
        free_ram()
        break
    finally:
        free_ram()
        print(f'post-{_cond}: {ram_report()}')
'''

CELL_VERDICT = r'''# ── Verdict: gates -> primaries -> secondaries -> riders -> banner -> ship ───
if set(BUNDLES) != set(CONDS):
    missing = sorted(set(CONDS) - set(BUNDLES))
    print('=' * 68)
    print(f'FLIGHT INCOMPLETE — missing condition(s): {missing}')
    for c, err in cond_errors.items():
        print(f'  {c}: {err}')
    print('Shipped inflight pieces are on Drive under', INFLIGHT)
    print('SMOKE: RED — diagnose locally; NEVER edit cells in place '
          '(VM law); fix in the builder, re-stage, re-upload.')
    raise SystemExit('no verdict on a partial flight')
for _c in CONDS:
    BUNDLES[_c].setdefault('cond_error', cond_errors.get(_c))
V = verdict(BUNDLES, PAYLOAD, MODE)
V['stamp'] = STAMP
V['plan_sha'] = PLAN_SHA
V['payload_sha'] = PAYLOAD_SHA_PIN
V['nb_build'] = NB_BUILD

smoke_checks = {
    'no_cond_errors': not any(cond_errors.values()),
    'gates_all_pass': V['gates']['all_pass'],
    'rows_complete': V['gates']['g_plan']['pass'],
    'capture_complete': V['gates']['g_capture']['pass'],
    'masses_present': all(r.get('vis_mass') is not None
                          for c in CONDS for r in BUNDLES[c]['rows']),
}
V['smoke_checks'] = smoke_checks

jdump(V, str(OUT / 'verdict.json'))
ship(OUT / 'verdict.json', INFLIGHT)
final_rel = f'e7bq/{MODE}_{STAMP}'
for f in sorted(OUT.iterdir()):
    ship(f, final_rel)
print('shipped ->', final_rel)

print()
print('=' * 68)
for line in V['banner']:
    print(line)
print('=' * 68)
if SMOKE:
    ok = all(smoke_checks.values())
    print('SMOKE:', 'GREEN — Runtime > Restart runtime, set SMOKE=False, '
          'Run all' if ok else 'RED — do not fly full; diagnose first')
    for k, v in smoke_checks.items():
        print(f'  {k}: {v}')
    if not ok:
        print('  (behavioral checks only — statistics never gate a smoke)')
else:
    print(f'FULL flight complete. RESUME_STAMP for reference: {STAMP}')
    print('Pull semcore/e7bq/ via the Drive integration; recompute locally; '
          'LOCK per protocol.')
'''

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"provenance": []},
                 "language_info": {"name": "python"},
                 "accelerator": "GPU"},
    "cells": [],
}


def add(kind, src):
    cell = {"cell_type": kind, "metadata": {},
            "source": src.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    nb["cells"].append(cell)


add("markdown", MD0)
add("code", CELL_SETUP)
add("code", LOGIC_SRC)
add("code", CELL_PAYLOAD)
add("code", CELL_PLAN)
add("code", CELL_FLIGHT)
add("code", CELL_VERDICT)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    assert "rclone" not in src.lower(), f"cell {i} mentions rclone (UI-only law)"
    if cell["cell_type"] == "code":
        compile(src, f"<cell {i}>", "exec")
print("cells compile; zero rclone anywhere")

# jdump must exist in the logic cell's namespace for the flight/verdict cells
assert "def jdump" in LOGIC_SRC or "jdump" not in CELL_FLIGHT, \
    "flight cell needs jdump but logic does not define it"

out_nb = os.path.join(HERE, "E7BQ_WALK_UI.ipynb")
with open(out_nb, "w") as f:
    json.dump(nb, f, indent=1)

# ── Colab-loadability gates (minted 2026-08-26 after the v1 2.5MB/115K-line
# notebook choked the Colab editor; proven-good fleet band is <=560KB with
# short lines — these bounds keep any future build inside it) ─────────────────
_size = os.path.getsize(out_nb)
_nb2 = json.load(open(out_nb))
_maxline = max(len(s.rstrip("\n")) for c in _nb2["cells"] for s in c["source"])
_nlines = sum(len(c["source"]) for c in _nb2["cells"])
assert _size < 900_000, f"notebook {_size} bytes — over the Colab-editor band"
assert _maxline < 5_000, f"a source line is {_maxline} chars — chunk it"
assert _nlines < 20_000, f"{_nlines} source lines — Colab-editor risk"
print(f"loadability gates: {_size:,} bytes | max line {_maxline} | "
      f"{_nlines:,} lines — PASS")

sha = hashlib.sha256(open(out_nb, "rb").read()).hexdigest()[:16]
print("wrote", out_nb, f"({len(nb['cells'])} cells, "
      f"{os.path.getsize(out_nb)} bytes, sha {sha})")
