"""E3 — swap the instilled embedder (E1's model_V2_conservative) into the Phase 7
checker's projection-fallback path and measure the delta.

For each embedder variant (stock all-MiniLM-L6-v2 baseline, V2_conservative):
  1. embed all 3,033 concepts ("NAME: description" — Phase 6 format)
  2. learn the 384->14 projection W by the Phase-6 parity method (full-sample OLS,
     no intercept)
  3. inject (model, embeddings, W) into ConsistencyChecker via a runtime subclass —
     no repo files are modified; benchmarks/scb1_results.json is snapshotted/restored
  4. run SCB-1 (89 claims) + the built-in 38-claim suite
Outputs: e3_report.json, e3_scb1_<tag>.json, e3_suite_<tag>.json, flip lists.

Registered prediction (pass doc §8): the out-of-vocabulary misses (silence, noise,
vice) will likely NOT be fixed — E1 instilled a map over dictionary concepts, not a
lexical rule. Run from: ~/venvs/semcore/bin/python e3_driver.py
"""
import sys, os, json, shutil, gc
import numpy as np

REPO = os.path.expanduser('~/agi-semantic-core')
sys.path.insert(0, REPO)
os.chdir(REPO)
OUT = os.path.join(REPO, 'colab', 'results_e0e1')
V2_PATH = os.path.join(OUT, 'model_V2_conservative')

import sqlite3
import tools.consistency_checker as ccmod
import tools.benchmark as bench
from api.semantic_core import SemanticCore
from sentence_transformers import SentenceTransformer

# ── dictionary data (own connection; Phase 6 conventions) ────────────────────
con = sqlite3.connect(os.path.join(REPO, 'db', 'semantic.db'))
con.row_factory = sqlite3.Row
DIMS = ['x','y','z','e','f','g','h','fx','fy','fz','fe','ff','fg','fh']
rows = con.execute(f"SELECT id,name,description,{','.join(DIMS)} FROM concepts ORDER BY id").fetchall()
names = [r['name'] for r in rows]
texts = [f"{r['name']}: {r['description']}" for r in rows]
Y14 = np.array([[r[d] for d in DIMS] for r in rows], dtype=np.float64)
id_to_name = {r['id']: r['name'] for r in rows}
alias_to_concept = {}
for a in con.execute("SELECT alias, concept_id FROM aliases"):
    n = id_to_name.get(a['concept_id'])
    if n:
        alias_to_concept[a['alias'].upper()] = n.upper()
con.close()
print(f'{len(names)} concepts loaded')

def patched_class(model, emb, W):
    class Patched(ccmod.ConsistencyChecker):
        def _ensure_loaded(self):
            if self._W is not None:
                return
            self._W = W
            self._all_names = list(names)
            self._all_proj = emb @ W
            self._all_proj_norms = np.linalg.norm(self._all_proj, axis=1)
            self._name_to_idx = {n.upper(): i for i, n in enumerate(names)}
            self._model = model
            self._sc = SemanticCore()
            self._alias_to_concept = dict(alias_to_concept)
    return Patched

def run_suite(CheckerCls):
    checker = CheckerCls(verbose=False)
    out, correct = [], 0
    for text, ctype, expected_true in ccmod.TEST_CLAIMS:
        v = checker.check(text)
        hit = (v.label in ('CONSISTENT', 'PLAUSIBLE')) if expected_true \
              else (v.label in ('SUSPICIOUS', 'INCONSISTENT'))
        correct += hit
        g = '-'
        if v.graph_signal and getattr(v.graph_signal, 'polarity_inversion', None):
            g = 'INVERSION'
        elif v.graph_signal and getattr(v.graph_signal, 'strongest_signal', None):
            g = str(v.graph_signal.strongest_signal)[:12]
        out.append({'claim': text, 'type': ctype, 'expected_true': expected_true,
                    'verdict': v.label, 'confidence': v.confidence,
                    'angle': v.angle, 'graph': g, 'hit': bool(hit)})
    checker.close()
    return {'correct': correct, 'total': len(ccmod.TEST_CLAIMS), 'claims': out}

scb_path = os.path.join(REPO, 'benchmarks', 'scb1_results.json')
snap = open(scb_path).read() if os.path.exists(scb_path) else None

report = {}
for tag, path in [('baseline', 'sentence-transformers/all-MiniLM-L6-v2'),
                  ('V2_conservative', V2_PATH)]:
    print(f'\n================ {tag} ================')
    model = SentenceTransformer(path, device='cpu')
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float64)
    W, *_ = np.linalg.lstsq(emb, Y14, rcond=None)
    P = emb @ W
    r2 = float(1 - ((Y14-P)**2).sum() / ((Y14-Y14.mean(0))**2).sum())
    print(f'[{tag}] projection R2 = {r2:.4f}')

    ccmod.ConsistencyChecker = patched_class(model, emb, W)
    summary = bench.run_scb_benchmark(verbose=True)
    shutil.copy(scb_path, os.path.join(OUT, f'e3_scb1_{tag}.json'))
    suite = run_suite(ccmod.ConsistencyChecker)
    json.dump(suite, open(os.path.join(OUT, f'e3_suite_{tag}.json'), 'w'), indent=1)
    print(f"[{tag}] 38-claim suite: {suite['correct']}/{suite['total']}")
    report[tag] = {'projection_r2': r2, 'scb1': summary,
                   'suite': {'correct': suite['correct'], 'total': suite['total']}}
    del model, emb, W, P
    gc.collect()

if snap is not None:
    open(scb_path, 'w').write(snap)
    print('\nbenchmarks/scb1_results.json restored to pre-run state')

# ── flip analysis ────────────────────────────────────────────────────────────
def load_hits(fp, key_claim, key_hit):
    d = json.load(open(fp))
    items = d['results'] if 'results' in d else d['claims']
    return {i[key_claim]: i[key_hit] for i in items}

flips = {}
for name_, base_fp, v2_fp, ck, hk in [
        ('scb1', f'{OUT}/e3_scb1_baseline.json', f'{OUT}/e3_scb1_V2_conservative.json', 'claim', 'hit'),
        ('suite', f'{OUT}/e3_suite_baseline.json', f'{OUT}/e3_suite_V2_conservative.json', 'claim', 'hit')]:
    b, v = load_hits(base_fp, ck, hk), load_hits(v2_fp, ck, hk)
    flips[name_] = {'gained': sorted(c for c in v if v[c] and not b.get(c, False)),
                    'lost':   sorted(c for c in v if not v[c] and b.get(c, False))}
report['flips'] = flips

# registered-prediction check
pred_claims = ['silence causes noise', 'virtue is a type of vice']
suite_v2 = json.load(open(f'{OUT}/e3_suite_V2_conservative.json'))['claims']
report['prediction_check'] = {c['claim']: {'verdict': c['verdict'], 'hit': c['hit']}
                              for c in suite_v2 if c['claim'] in pred_claims}

json.dump(report, open(os.path.join(OUT, 'e3_report.json'), 'w'), indent=1)

print('\n==================== E3 SUMMARY ====================')
for tag in ['baseline', 'V2_conservative']:
    r = report[tag]
    print(f"{tag:16s}  projR2={r['projection_r2']:.4f}  "
          f"SCB1 acc={r['scb1']['accuracy']:.3f}  viol-F1={r['scb1']['violation_f1']:.3f}  "
          f"suite={r['suite']['correct']}/{r['suite']['total']}")
print('flips (SCB-1):  gained:', flips['scb1']['gained'], ' lost:', flips['scb1']['lost'])
print('flips (suite):  gained:', flips['suite']['gained'], ' lost:', flips['suite']['lost'])
print('prediction check:', json.dumps(report['prediction_check']))
