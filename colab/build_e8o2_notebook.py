#!/usr/bin/env python3
"""Build colab/E8O2_MIXED_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-O2: the mixed-figure curriculum — breaking the méjì shortcut
(docs/E8O2_PROTOCOL.md, pre-registered cb93217 BEFORE this file existed).

CONTINUE-training flight: the LOCKED E8-F readout is loaded TRAINABLE and
schooled on leg-decoupled points (shortcut pays .360 = chance there), then
the FLOWN E8-O po block re-flies VERBATIM for direct before/after.

Single-source law: logic cell = e8o_logic.py BYTE-VERBATIM (base) + the
E8-O2 block (also emitted to e8o2_logic.py); smeji_stat sliced byte-equal
from the design check; train-loop load-bearing lines donor-pinned to the
E8-F builder; stimulus/Injector/G2/runner cells = the flown E8-O donors.
"""
import hashlib, importlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_e8f_notebook as BF
import build_e8o_notebook as BO

E8O_LOGIC_SRC = open(os.path.join(HERE, "e8o_logic.py")).read()
DESIGN_SRC = open(os.path.join(HERE, "e8o2_design_check.py")).read()
DC = json.load(open(os.path.join(HERE, "results_e8o2", "design_check.json")))


def slice_block(src, start, end=None, tag=""):
    i = src.find(start)
    assert i >= 0, f"slice[{tag}]: start missing"
    if end is None:
        return src[i:]
    j = src.find(end, i)
    assert j > i, f"slice[{tag}]: end missing"
    return src[i:j]


SMEJI_SRC = slice_block(DESIGN_SRC, "def meji_completion_code",
                        "def main", tag="smeji")
# namespace-flatten the design check's `L.` refs (byte-equal modulo this,
# asserted by the logic suite with the same transform)
SMEJI_SRC = SMEJI_SRC.replace("L.code_levels", "code_levels") \
                     .replace("L.AXIS_NAMES_F", "AXIS_NAMES_F")
import build_e8r_notebook as BR
RETENTION_SRC = slice_block(BR.CELL_TRAINCFG, "RETENTION_TEXT", None,
                            tag="retention")

E8O2_SEED = 20260910

E8O2_SRC_TEMPLATE = r'''# ── E8-O2 pure logic: mixed curriculum, eval, new stats (locally tested) ─────
# (builds on the e8o_logic namespace: PAIR24, pair_codes, compose_A,
#  composed_code, po1_stats, po2_stats, gid_stats, code_levels, holm, ...)

E8O2_SEED = 20260910
STRANDS_O2 = ['mixed', 'replay', 'sham']
EPOCH_CAP_O2 = 8
GINSTM_MIN = 0.55
PPL_TOL_O2 = 5.0

TRAINPAIR48 = @@TRAINPAIR48@@
POFRESH12 = @@POFRESH12@@
SPOTMIX12 = @@SPOTMIX12@@
REPLAY24 = @@REPLAY24@@
SHA_TRAINPAIR = '@@SHA_TRAINPAIR@@'
SHA_POFRESH = '@@SHA_POFRESH@@'

# flown E8-O flight-1 baselines (the before/after comparison row)
BASELINE_O = {'po1_pooled': 0.4464, 'po2_d': 0.0275, 'po2_p': 0.019,
              'fun_vs_B': 0.3824, 'fun_vs_A_meji': 0.4420,
              'ess_pooled': 0.51, 'gid': 0.5982}

def pair_codes_for(pairs, vecs):
    out = {}
    for i, (a, b) in enumerate(pairs):
        out[(i, 0)] = composed_code(vecs[a], vecs[b])
        out[(i, 1)] = composed_code(vecs[b], vecs[a])
    return out

def mint_stimuli_o2(W5, vecs):
    """carrier + full:<n> (ident + replay) + comp:<i>:<o> (flown PAIR24,
    SAME keys as E8-O) + compT:<i>:<o> (train pairs) + compF:<i>:<o>
    (fresh eval pairs)."""
    import numpy as _np
    Wax, b = _np.asarray(W5, float)[:14], _np.asarray(W5, float)[14]
    def _pred(x):
        v = _np.asarray(x, float) @ Wax + b
        return v / _np.linalg.norm(v)
    stim = {'carrier': _pred(MU_FRAME)}
    for n in sorted({n for n, _ in IDENT16} | set(REPLAY24)):
        stim[f'full:{n}'] = _pred(vecs[n])
    for tag, plist in (('comp', PAIR24), ('compT', TRAINPAIR48),
                       ('compF', POFRESH12)):
        for i, (a, bnm) in enumerate(plist):
            stim[f'{tag}:{i}:0'] = _pred(compose_A(vecs[a], vecs[bnm]))
            stim[f'{tag}:{i}:1'] = _pred(compose_A(vecs[bnm], vecs[a]))
    return stim

def build_e8o2_train(vecs, smoke=False):
    ex, eid = [], 60000
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    pairs = TRAINPAIR48[:2] if smoke else TRAINPAIR48
    replay = REPLAY24[:4] if smoke else REPLAY24
    n_sham = 2 if smoke else 24
    n_car = 1 if smoke else 4
    def add(strand, kind, skey, alpha, target):
        nonlocal eid
        ex.append({'eid': eid, 'strand': strand, 'kind': kind, 'skey': skey,
                   'layer': TRAIN_LAYER if kind == 'inject' else None,
                   'alpha': alpha, 'target': target})
        eid += 1
    for i, (a, b) in enumerate(pairs):
        for o, (s, j) in enumerate(((a, b), (b, a))):
            tgt = code_text(composed_code(vecs[s], vecs[j]))
            for al in A:
                add('mixed', 'inject', f'compT:{i}:{o}', al, tgt)
    for n in replay:
        for al in A:
            add('replay', 'inject', f'full:{n}', al,
                code_text(code_levels(vecs[n])))
    car_code = code_text(code_levels(MU_FRAME))
    for _ in range(n_car):
        add('replay', 'inject', 'carrier', 1.0, car_code)
    for _ in range(n_sham):
        add('sham', 'sham', None, 0.0, 'NONE')
    return ex

EXPECT_TRAIN_O2 = {'mixed': 192, 'replay': 52, 'sham': 24}

def build_e8o2_eval(smoke=False):
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    spot = SPOTMIX12[:2] if smoke else SPOTMIX12
    flown = list(range(2)) if smoke else list(range(len(PAIR24)))
    fresh = list(range(2)) if smoke else list(range(len(POFRESH12)))
    ident = IDENT16[:2] if smoke else IDENT16
    n_sham = 2 if smoke else 12
    n_car = 1 if smoke else 2
    rows, tid = [], 70000
    spot_idx = [TRAINPAIR48.index(list(p) if isinstance(p, list) else p)
                if p in TRAINPAIR48 else TRAINPAIR48.index(list(p))
                for p in spot]
    for i in spot_idx:
        for o in (0, 1):
            rows.append({'tid': tid, 'block': 'spot_mixed', 'kind': 'inject',
                         'skey': f'compT:{i}:{o}', 'pair_idx': i, 'order': o,
                         'layer': TRAIN_LAYER, 'alpha': 1.0})
            tid += 1
    for i in flown:
        for o in (0, 1):
            for al in A:
                rows.append({'tid': tid, 'block': 'po', 'kind': 'inject',
                             'skey': f'comp:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': al})
                tid += 1
    for i in fresh:
        for o in (0, 1):
            for al in A:
                rows.append({'tid': tid, 'block': 'po_fresh', 'kind': 'inject',
                             'skey': f'compF:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': al})
                tid += 1
    for (n, al) in ident:
        rows.append({'tid': tid, 'block': 'ident', 'kind': 'inject',
                     'skey': f'full:{n}', 'concept': n,
                     'layer': TRAIN_LAYER, 'alpha': al})
        tid += 1
    for _ in range(n_sham):
        rows.append({'tid': tid, 'block': 'sham', 'kind': 'sham',
                     'skey': None, 'layer': None, 'alpha': 0.0})
        tid += 1
    for _ in range(n_car):
        rows.append({'tid': tid, 'block': 'carrier', 'kind': 'inject',
                     'skey': 'carrier', 'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    return rows

EXPECT_EVAL_O2 = {'spot_mixed': 24, 'po': 96, 'po_fresh': 48, 'ident': 16,
                  'sham': 12, 'carrier': 2}

def validate_no_eval_in_train_o2(examples):
    ev = set(EVAL64)
    bad = []
    for e in examples:
        sk = e.get('skey') or ''
        if sk.startswith('comp:') or sk.startswith('compF:'):
            bad.append(e.get('eid'))
        if sk.startswith('full:') and sk[5:] in ev:
            bad.append(e.get('eid'))
        if sk.startswith('compT:'):
            i = int(sk.split(':')[1])
            if set(TRAINPAIR48[i]) & ev:
                bad.append(e.get('eid'))
    return bad

def pm2_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """P-M2: the FUNCTION-LEG pooled accuracy vs pair-derangements —
    po1_stats restricted to axes 7..13."""
    import numpy as _np
    if seed is None:
        seed = E8O2_SEED + 20
    fun = list(range(7, 14))
    P = levels_matrix(rows)[:, fun]
    pair_ids = sorted({r['pair_idx'] for r in rows})
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows],
                  int)[:, fun]
    obs = float((P == T).mean())
    rng = _np.random.default_rng(seed)
    idx = _np.arange(len(pair_ids))
    pos = {p: k for k, p in enumerate(pair_ids)}
    ge, nulls = 0, []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while _np.any(pi == idx):
            pi = rng.permutation(idx)
        Tn = _np.array([pcodes[(pair_ids[pi[pos[r['pair_idx']]]], r['order'])]
                        for r in rows], int)[:, fun]
        nm = float((P == Tn).mean())
        nulls.append(nm)
        ge += nm >= obs
    return {'pooled_acc': round(obs, 4), 'p': (1 + ge) / (1 + n_perm),
            'null_mean': round(float(_np.mean(nulls)), 4), 'n_rows': len(rows)}

def ess_acc(rows, pcodes):
    import numpy as _np
    ess = list(range(7))
    P = levels_matrix(rows)[:, ess]
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows],
                  int)[:, ess]
    return round(float((P == T).mean()), 4)

@@SMEJI_SRC@@

def smeji_test(rows, pcodes, vecs, pair_list, n_perm=N_PERM_O, seed=None):
    """S-MEJI: reversal statistic + sign-flip p over pairs (one-sided > 0)."""
    import numpy as _np
    if seed is None:
        seed = E8O2_SEED + 21
    obs, ds = smeji_stat(rows, pcodes, vecs, pair_list)
    per_pair = {}
    for r, d in zip(rows, ds):
        per_pair.setdefault(r['pair_idx'], []).append(d)
    dvec = _np.array([float(_np.mean(v)) for _, v in sorted(per_pair.items())])
    rng = _np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(dvec))
        if float((dvec * s).mean()) >= float(dvec.mean()):
            ge += 1
    return {'d_mean': round(obs, 4), 'p': (1 + ge) / (1 + n_perm),
            'n_pairs': len(dvec), 'n_pos': int((dvec > 0).sum())}
'''

MD0 = """# E8-O2 — The Mixed-Figure Curriculum (Phase 10 UI flight)

**Pre-registered**: `docs/E8O2_PROTOCOL.md` (cb93217, before build).
CONTINUE-training: the **locked E8-F readout** loads TRAINABLE and is
schooled on leg-decoupled points — where the méjì shortcut pays .360
(chance) instead of the 1.000 it earned on every concept row it ever saw.
Then the FLOWN E8-O po block re-flies VERBATIM: **P-M1** order
discrimination (the L4 claim, baseline d .0275 p .019), **P-M2**
function-leg reading (baseline .38), **S-MEJI** shortcut reversal
(baseline ≈ −.06 → must go positive). Retention gate G-ID ≥ .55.

**Flight plan (Run all, twice)**
1. **Smoke** (`SMOKE=True`, armed): ~8–12 min. GREEN → Restart runtime.
2. **Full** (`SMOKE=False`): one run, ~30–45 min (train to per-strand
   plateau cap 8, then 198 eval generations). Verdict + `e8o2/` on Drive.
"""

CELL_MODELCFG_TEMPLATE = r'''# ── Model cfg: tok, CONTINUE-load, train loop (E8-F donor lines pinned) ──────
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LR = 1e-4
ACCUM = 4 if SMOKE else 8
EPOCHS_CAP = 1 if SMOKE else EPOCH_CAP_O2
MIN_EPOCHS_STRAND = 1 if SMOKE else 2
PLATEAU_REL = 0.05

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

@@RETENTION_BLOCK@@

def encode_featex(ex):
    msgs = [{'role': 'user', 'content': feat_prompt()}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors='pt').input_ids[0]
    ans = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    return ids, len(pid), ans.long().unsqueeze(0), \
        (ex['skey'] if ex['kind'] == 'inject' else None)

def train_readout_mixed(m, layer_mods, mu14, examples):
    """The E8-F train loop with STRANDS_O2 (load-bearing lines verbatim:
    answer-sliced forward, backward INSIDE the Injector context,
    checkpointing asserted, per-strand plateau)."""
    import torch.nn.functional as Fnn
    m.train()
    _cm = m.base_model.model
    DEC, HEAD = _cm.model, _cm.get_output_embeddings()
    try:
        _cm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    except TypeError:
        _cm.gradient_checkpointing_enable()
    try:
        _cm.enable_input_require_grads()
    except Exception as e:
        print('  (input-require-grads unavailable:', e, ')')
    assert getattr(DEC, 'gradient_checkpointing', False), (
        'gradient checkpointing did not engage — refusing to train without it')
    params = [p for p in m.parameters() if p.requires_grad]
    assert params, 'no trainable params — is_trainable load failed'
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    losses, micro, hook_calls = [], 0, 0
    strand_epoch_means = []
    plateaued, epochs_flown = False, 0
    t0 = time.time()

    def fwd_loss(ids, plen, ans):
        lo, hi = answer_slice(plen, ids.shape[1])
        hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
        logits = HEAD(hid[:, lo:hi, :]).float()
        return Fnn.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))

    for ep in range(EPOCHS_CAP):
        order = np.random.default_rng(E8O2_SEED + 100 + ep).permutation(len(examples))
        ep_strand = {s: [] for s in STRANDS_O2}
        for i in order:
            ex = examples[int(i)]
            ids, plen, ans, skey = encode_featex(ex)
            ids, ans = ids.to(DEV), ans.to(DEV)
            if skey is not None:
                vec = ex['alpha'] * mu14 * torch.tensor(STIM[skey])
                with Injector(layer_mods, ex['layer'], vec, plen - 1) as injh:
                    loss = fwd_loss(ids, plen, ans)
                    lv = float(loss.detach())
                    scaler.scale(loss / ACCUM).backward()
                hook_calls += injh.calls
            else:
                loss = fwd_loss(ids, plen, ans)
                lv = float(loss.detach())
                scaler.scale(loss / ACCUM).backward()
            assert math.isfinite(lv)
            losses.append(round(lv, 4))
            ep_strand[ex.get('strand')].append(lv)
            micro += 1
            if micro % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if micro % 100 == 0:
                print(f'    training ep{ep+1} {micro} micro-steps, '
                      f'loss~{np.mean(losses[-50:]):.3f}, {time.time()-t0:.0f}s')
        epochs_flown = ep + 1
        strand_epoch_means.append({s: (round(float(np.mean(v)), 4) if v else None)
                                   for s, v in ep_strand.items()})
        print(f'  epoch {epochs_flown}: '
              + ' | '.join(f'{s} {strand_epoch_means[-1][s]}' for s in STRANDS_O2))
        if per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL):
            plateaued = True
            break
    if micro % ACCUM:
        scaler.step(opt); scaler.update(); opt.zero_grad()
    try:
        _cm.gradient_checkpointing_disable()
    except Exception:
        pass
    m.eval()
    return {'n_examples': len(examples), 'epochs_flown': epochs_flown,
            'epochs_cap': EPOCHS_CAP, 'plateaued': plateaued,
            'strand_epoch_means': strand_epoch_means,
            'micro_steps': micro, 'opt_steps': micro // ACCUM,
            'hook_calls': int(hook_calls), 'trainable_params': int(n_tr),
            'secs': round(time.time() - t0, 1),
            'losses_every_10': losses[::10]}

def resolve_layers(m):
    mods = {}
    for mod_name, mod in m.named_modules():
        mm = re.search(r'(?:^|\.)layers\.(\d+)$', mod_name)
        if mm:
            mods[int(mm.group(1))] = mod
    n = m.config.num_hidden_layers if hasattr(m.config, 'num_hidden_layers') \
        else m.base_model.config.num_hidden_layers
    assert len(mods) == n, (len(mods), n)
    return mods
'''

CELL_FLIGHT = r'''# ── Flight: G2 -> CONTINUE-load readout -> train mixed -> eval -> ship ───────
def fly():
    t0 = time.time()
    bundle = {'condition': 'real', 'mode': MODE, 'stamp': STAMP,
              'payload_sha': PAYLOAD_SHA, 'sha_trainpair': SHA_TRAINPAIR,
              'sha_pofresh': SHA_POFRESH, 'readout_src': READOUT_SRC}
    try:
        m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                  device_map=DEV, low_cpu_mem_usage=True)
        m0 = PeftModel.from_pretrained(m0, ADAPTERS['real']).merge_and_unload()
        m0.eval()
        stab, mu14 = g2_probe(m0)
        bundle['g2_dirs'], bundle['mu14'] = stab, mu14
        assert stab['pass'], f'G2 FAILED: {stab}'
        model = PeftModel.from_pretrained(m0, str(READOUT_DIR), is_trainable=True)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        layer_mods = resolve_layers(model)
        examples = build_e8o2_train(VEC, SMOKE)
        bad = validate_no_eval_in_train_o2(examples)
        assert not bad, f'FIREWALL — eval leak in curriculum: {bad}'
        cur = {}
        for e in examples:
            cur[e['strand']] = cur.get(e['strand'], 0) + 1
        bundle['curriculum'] = cur
        if not SMOKE:
            assert cur == EXPECT_TRAIN_O2, cur
        print(f'  curriculum: {cur}')
        model.eval()
        bundle['ppl_pre'] = retention_ppl(model)
        bundle['train_log'] = train_readout_mixed(model, layer_mods, mu14,
                                                  examples)
        torch.cuda.empty_cache()
        model.save_pretrained(str(OUT / 'readout_mixed'))
        ship(OUT / 'readout_mixed', f'e8o2/inflight_{STAMP}/readout_mixed')
        print('  continued readout shipped inflight')
        rows = build_e8o2_eval(SMOKE)
        out, t1 = {}, time.time()
        for i, t in enumerate(rows, 1):
            r = run_trial_feat(model, layer_mods, mu14, t)
            out.setdefault(t['block'], []).append(r)
            if i % 25 == 0 or i == len(rows):
                el = time.time() - t1
                eta = el / i * (len(rows) - i)
                print(f'    eval {i}/{len(rows)}, {el:.0f}s, ~{eta:.0f}s left')
        bundle['eval_rows'] = out
        cnt = {b: len(v) for b, v in out.items()}
        print(f'  eval rows: {cnt}')
        if not SMOKE:
            assert cnt == EXPECT_EVAL_O2, cnt
        bundle['ppl_post'] = retention_ppl(model)
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        bundle['error'] = f'{type(e).__name__}: {e}'
    return bundle

print(f'flight: starting — {ram_report()}')
BUNDLE = fly()
fn = OUT / 'bundle_real.json'
jdump(BUNDLE, fn)
ship(fn, f'e8o2/inflight_{STAMP}')
print(f'flight done in {BUNDLE.get("secs", "?")}s — bundle shipped')
free_ram()
'''

CELL_VERDICT = r'''# ── Gates, P-M1/P-M2, S-MEJI/S-FRESH/S-ESS, before/after, fork, ship ────────
ok = bool(BUNDLE.get('eval_rows')) and 'error' not in BUNDLE
if not ok:
    print()
    print('=' * 72)
    print(f'  FLIGHT INCOMPLETE — error: {str(BUNDLE.get("error"))[:200]}')
    print('  Fix comes home to the builder (lane law).')
    print('=' * 72)
else:
    verdict = {'flight': f'{MODE}_{STAMP}', 'protocol': 'E8O2_PROTOCOL.md',
               'payload_sha': PAYLOAD_SHA, 'readout_src': READOUT_SRC,
               'baseline': BASELINE_O}
    blocks = BUNDLE['eval_rows']
    allrows = [r for rs in blocks.values() for r in rs]
    NP = 200 if SMOKE else N_PERM_O

    PCT = pair_codes_for(TRAINPAIR48, VEC)
    PCF = pair_codes_for(POFRESH12, VEC)
    for _b in ('spot_mixed', 'po', 'po_fresh'):
        assert len({r['pair_idx'] for r in blocks[_b]}) >= 2, (
            f'{_b}: derangement stats need >=2 pairs (would spin forever)')
    inv = invalid_rate(allrows)
    fa = sham_claims(blocks.get('sham', []))
    gid = gid_stats(blocks['ident'], CODES64)
    spotm = po1_stats(blocks['spot_mixed'], PCT, NP, seed=E8O2_SEED + 22)
    ginstm = {'pooled_acc': spotm['pooled_acc'], 'p': spotm['p'],
              'min': GINSTM_MIN,
              'pass': bool(spotm['pooled_acc'] >= GINSTM_MIN
                           and spotm['p'] <= P_CRIT_O)}
    ppl_pre, ppl_post = BUNDLE.get('ppl_pre'), BUNDLE.get('ppl_post')
    ppl_d = 100.0 * (ppl_post / ppl_pre - 1.0)
    gates = {'g1_pins': {'payload': PAYLOAD_SHA,
                         'trainpair': SHA_TRAINPAIR, 'pofresh': SHA_POFRESH,
                         'asserted_in_flight': True, 'pass': True},
             'g2_dirs': BUNDLE['g2_dirs'],
             'g_id': gid,
             'g_install_m': ginstm,
             'g3_ppl': {'pre': ppl_pre, 'post': ppl_post,
                        'delta_pct': round(ppl_d, 4), 'tol': PPL_TOL_O2,
                        'pass': bool(abs(ppl_d) <= PPL_TOL_O2)},
             'g4_parse': {**inv, 'max': INVALID_MAX_O,
                          'pass': bool(inv['rate'] <= INVALID_MAX_O)},
             'g5_sham': {'claims': fa, 'n': len(blocks.get('sham', [])),
                         'max': SHAM_MAX_O, 'pass': bool(fa <= SHAM_MAX_O)}}
    gates_ok = all(g.get('pass') for g in gates.values())
    verdict['gates'] = gates
    verdict['spot_mixed'] = spotm

    pm1 = po2_stats(blocks['po'], PCODES, NP, seed=E8O2_SEED + 23)
    pm1_pass = bool(pm1['p'] <= P_CRIT_O)
    pm2 = pm2_stats(blocks['po'], PCODES, NP)
    pm2_pass = bool(pm2['p'] <= P_CRIT_O)
    verdict['P_M1'] = {**pm1, 'pass': pm1_pass}
    verdict['P_M2'] = {**pm2, 'pass': pm2_pass}

    smeji = smeji_test(blocks['po'], PCODES, VEC,
                       [tuple(p) for p in PAIR24], NP)
    verdict['S_MEJI'] = {**smeji, 'pass': bool(smeji['p'] <= P_CRIT_O
                                               and smeji['d_mean'] > 0)}
    pf1 = po2_stats(blocks['po_fresh'], PCF, NP, seed=E8O2_SEED + 24)
    pf2 = pm2_stats(blocks['po_fresh'], PCF, NP, seed=E8O2_SEED + 25)
    verdict['S_FRESH'] = {'d_mean': pf1['d_mean'], 'd_p': pf1['p'],
                          'fun_acc': pf2['pooled_acc'], 'fun_p': pf2['p'],
                          'consistent': bool(pf1['d_mean'] > 0
                                             and pf2['pooled_acc']
                                             > pf2['null_mean'])}
    verdict['S_ESS'] = {'po_ess_acc': ess_acc(blocks['po'], PCODES),
                        'baseline': BASELINE_O['ess_pooled']}
    verdict['S1'] = s1_profile(blocks)
    verdict['before_after'] = {
        'po2_d': [BASELINE_O['po2_d'], pm1['d_mean']],
        'fun_acc': [BASELINE_O['fun_vs_B'], pm2['pooled_acc']],
        'smeji': [-0.06, smeji['d_mean']],
        'gid': [BASELINE_O['gid'], gid['pooled_acc']]}
    verdict['train_log'] = {k: v for k, v in BUNDLE['train_log'].items()
                            if k != 'losses_every_10'}

    if SMOKE:
        fork = 'SMOKE — mechanics only, no verdict'
    elif not gates_ok:
        bad_gates = [k for k, g in gates.items() if not g.get('pass')]
        if 'g_id' in bad_gates or 'g_install_m' in bad_gates:
            fork = ('FV4 — NO_VERDICT (' + ','.join(bad_gates) + '); the '
                    'single pre-authorized re-fly applies (retention -> '
                    'doubled replay; install -> budget)')
        else:
            fork = 'GATES DIRTY — primaries withheld (NO_VERDICT, lane law)'
    elif pm1_pass and pm2_pass:
        fork = ('FV1 — L4 LANDS in bottleneck form: both legs read, order '
                'discriminated; the méjì shortcut was a curriculum artifact '
                '(second confirmation-by-cure)')
    elif pm2_pass and not pm1_pass:
        fork = ('FV2 — function leg readable but order STILL collapses: '
                'deeper-than-curriculum wall; honest stop for this approach')
    elif pm1_pass and not pm2_pass:
        fork = ('FV3 — order discriminated through essence-side cues alone; '
                'partial, claim bounded')
    else:
        fork = ('neither primary clears — see per-axis tables and the '
                'before/after row; deeper-than-curriculum wall (FV2 family)')
    verdict['fork'] = fork

    print()
    print('=' * 72)
    if SMOKE:
        n_par = sum(1 for r in allrows if r['parsed']['kind'] != 'INVALID')
        green = ok and n_par >= max(1, int(0.5 * len(allrows)))
        print(f'  SMOKE {"GREEN" if green else "RED"} — mechanics '
              f'{"exercised" if green else "FAILED"} ({n_par}/{len(allrows)} parsed)')
        if green:
            print('  Next: Runtime > Restart runtime, set SMOKE = False, Run all.')
    else:
        print(f'  E8-O2 VERDICT — gates {"PASS" if gates_ok else "FAIL"} '
              f'(G-ID {gid["pooled_acc"]} | install-M {ginstm["pooled_acc"]} '
              f'| ppl {gates["g3_ppl"]["delta_pct"]}%)')
        if gates_ok:
            print(f'  P-M1 ORDER: d {pm1["d_mean"]} '
                  f'(baseline {BASELINE_O["po2_d"]}) '
                  f'({pm1["n_pos"]}/{pm1["n_pairs"]} pos) p={pm1["p"]} -> '
                  f'{"PASS" if pm1_pass else "FAIL"}')
            print(f'  P-M2 FUNCTION LEG: {pm2["pooled_acc"]} '
                  f'(baseline {BASELINE_O["fun_vs_B"]}) vs null '
                  f'{pm2["null_mean"]} p={pm2["p"]} -> '
                  f'{"PASS" if pm2_pass else "FAIL"}')
            print(f'  S-MEJI reversal: {smeji["d_mean"]} '
                  f'(baseline ~-0.06) p={smeji["p"]} -> '
                  f'{"PASS" if verdict["S_MEJI"]["pass"] else "FAIL"}')
        print(f'  fork: {fork}')
    print('=' * 72)
    vf = OUT / 'e8o2_verdict.json'
    jdump(verdict, vf)
    ship(vf, f'e8o2/{MODE}_{STAMP}')
    fn = OUT / 'bundle_real.json'
    if not fn.exists():
        jdump(BUNDLE, fn)
    ship(fn, f'e8o2/{MODE}_{STAMP}')
    print('shipped: e8o2/' + f'{MODE}_{STAMP}')
'''


def mint():
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    payload = json.load(open(os.path.join(HERE, "e8f_payload.json")))
    import e8o_logic as L

    trainpair48 = [tuple(p) for p in DC["s2"]["trainpair48"]]
    pofresh12 = [tuple(p) for p in DC["s2"]["pofresh12"]]
    spotmix12 = [tuple(p) for p in DC["s2"]["spotmix12"]]
    assert len({c for p in trainpair48 for c in p}) == 96
    assert set(c for p in trainpair48 for c in p) <= set(L.TRAIN256)
    assert not ({tuple(p) for p in pofresh12}
                & {tuple(p) for p in L.PAIR24})
    rng = np.random.default_rng(E8O2_SEED + 4)
    members = {c for p in trainpair48 for c in p}
    pool = sorted(set(L.TRAIN256) - members)
    replay24 = sorted(rng.choice(pool, size=24, replace=False).tolist())

    subs = {
        "@@TRAINPAIR48@@": json.dumps([list(p) for p in trainpair48]),
        "@@POFRESH12@@": json.dumps([list(p) for p in pofresh12]),
        "@@SPOTMIX12@@": json.dumps([list(p) for p in spotmix12]),
        "@@REPLAY24@@": json.dumps(replay24),
        "@@SHA_TRAINPAIR@@": DC["s2"]["sha_trainpair"],
        "@@SHA_POFRESH@@": DC["s2"]["sha_pofresh"],
        "@@SMEJI_SRC@@": SMEJI_SRC.rstrip(),
    }
    return payload, subs, {"vec": VEC, "replay24": replay24}


def main():
    payload, subs, ctx = mint()
    e8o2_src = E8O2_SRC_TEMPLATE
    for k, v in subs.items():
        assert k in e8o2_src, f"missing sentinel {k}"
        e8o2_src = e8o2_src.replace(k, v)
    assert "@@" not in e8o2_src

    logic = E8O_LOGIC_SRC + "\n\n" + e8o2_src
    with open(os.path.join(HERE, "e8o2_logic.py"), "w") as f:
        f.write(logic)
    for mod in ("e8o2_logic",):
        if mod in sys.modules:
            del sys.modules[mod]
    L2 = importlib.import_module("e8o2_logic")

    tr = L2.build_e8o2_train(ctx["vec"], smoke=False)
    cnt = {}
    for e in tr:
        cnt[e["strand"]] = cnt.get(e["strand"], 0) + 1
    assert cnt == L2.EXPECT_TRAIN_O2, cnt
    assert not L2.validate_no_eval_in_train_o2(tr)
    ev = L2.build_e8o2_eval(smoke=False)
    cnt_e = {}
    for r in ev:
        cnt_e[r["block"]] = cnt_e.get(r["block"], 0) + 1
    assert cnt_e == L2.EXPECT_EVAL_O2, cnt_e
    stim = L2.mint_stimuli_o2(np.array(payload["W"], float), ctx["vec"])
    need = {e["skey"] for e in tr if e["skey"]} | \
           {r["skey"] for r in ev if r.get("skey")}
    assert need <= set(stim), sorted(need - set(stim))[:5]

    cell_modelcfg = CELL_MODELCFG_TEMPLATE.replace("@@RETENTION_BLOCK@@",
                                                   RETENTION_SRC.rstrip())

    payload_json = json.dumps(payload, separators=(",", ":"))
    assert '"""' not in payload_json and "\\" not in payload_json
    cell_payload = BO.CELL_PAYLOAD_O_TEMPLATE.replace("@@PAYLOAD_JSON@@",
                                                      payload_json)
    # E8-O2 payload cell: same E8-O skeleton, mint + probes swapped
    cell_payload = cell_payload.replace(
        "STIM = mint_stimuli_o(W5, VEC)",
        "STIM = mint_stimuli_o2(W5, VEC)")
    cell_payload = cell_payload.replace(
        "assert len([k for k in STIM if k.startswith('comp:')]) == 48",
        "assert len([k for k in STIM if k.startswith('comp:')]) == 48\n"
        "assert len([k for k in STIM if k.startswith('compT:')]) == 96\n"
        "assert len([k for k in STIM if k.startswith('compF:')]) == 24")
    cell_payload = cell_payload.replace(
        "_sha24 = hashlib.sha256('|'.join(f'{a}+{b}' for a, b in PAIR24).encode()\n"
        "                        ).hexdigest()[:16]\n"
        "assert _sha24 == PAIR24_SHA, 'pair draw drift'",
        "_sha24 = hashlib.sha256('|'.join(f'{a}+{b}' for a, b in PAIR24).encode()\n"
        "                        ).hexdigest()[:16]\n"
        "assert _sha24 == PAIR24_SHA, 'pair draw drift'\n"
        "_shaT = hashlib.sha256('|'.join(f'{a}+{b}' for a, b in TRAINPAIR48)\n"
        "                       .encode()).hexdigest()[:16]\n"
        "assert _shaT == SHA_TRAINPAIR, 'train-pair draw drift'\n"
        "_shaF = hashlib.sha256('|'.join(f'{a}+{b}' for a, b in POFRESH12)\n"
        "                       .encode()).hexdigest()[:16]\n"
        "assert _shaF == SHA_POFRESH, 'fresh-pair draw drift'")
    probes = {}
    W5 = np.array(payload["W"], float)
    stim2 = L2.mint_stimuli_o2(W5, ctx["vec"])
    for k in ("compT:0:0", "compF:5:1", "carrier"):
        probes[k] = [round(float(x), 5) for x in stim2[k]]
    cell_payload = cell_payload.replace(
        "for k, pv in PAYLOAD_PROBES_O.items():",
        "PAYLOAD_PROBES_O = json.loads(r'''" + json.dumps(probes) + "''')\n"
        "for k, pv in PAYLOAD_PROBES_O.items():")

    cell_setup = BO.CELL_SETUP.replace(
        "NB_BUILD = 'v1 (2026-08-25)'\nprint('E8-O notebook build:', NB_BUILD)",
        "NB_BUILD = 'v1 (2026-08-25)'\nprint('E8-O2 notebook build:', NB_BUILD)")
    cell_stim = ("# ── G2 stimulus machinery + Injector + runner (flown donors) ────────────────\n"
                 "import numpy as np\n"
                 + BF.STIM_MACH_SRC + "\n" + BF.INJECTOR_SRC + BF.CELL_RUN_SRC)

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": []},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"},
          "cells": []}

    def add(kind, src):
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)

    add("markdown", MD0)
    add("code", cell_setup)
    add("code", logic)
    add("code", cell_payload)
    add("code", cell_modelcfg)
    add("code", cell_stim)
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    import ast
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        assert "rclone" not in src.lower(), f"cell {i}: rclone"
        if c["cell_type"] == "code":
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"cell {i} does not compile: {e}")
    # donor pins: the load-bearing train lines match the E8-F train source
    mc = "".join(nb["cells"][4]["source"])
    for line in ("scaler.scale(loss / ACCUM).backward()",
                 "assert getattr(DEC, 'gradient_checkpointing', False)",
                 "hid[:, lo:hi, :]",
                 "per_strand_plateau(strand_epoch_means, MIN_EPOCHS_STRAND, PLATEAU_REL)"):
        assert line in BF.E8F_TRAIN_SRC and line in mc, line

    out_path = os.path.join(HERE, "E8O2_MIXED_UI.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"built {out_path} ({os.path.getsize(out_path)/1e3:.0f}KB, "
          f"{len(nb['cells'])} cells)")
    import shutil
    dst = os.path.expanduser("~/Desktop/E8O2_MIXED_UI.ipynb")
    shutil.copy2(out_path, dst)
    print(f"staged {dst}")


if __name__ == "__main__":
    main()
