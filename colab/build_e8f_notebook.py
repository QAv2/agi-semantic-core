#!/usr/bin/env python3
"""Build colab/E8F_FEATURAL_UI.ipynb (UI-native, mount + shutil, zero rclone).

E8-F: the featural curriculum, L3 in bottleneck form
(docs/E8F_PROTOCOL.md, pre-registered 44bb57b; amendments d61bd16/148ecc9,
both before this file existed). Train a FRESH readout on the register's 14
bridge axes (tercile code, mu-centered atoms, pairs, full codes, true dirs),
then test spelling of never-trained concepts (P1), true-dir reading on the
carried-8 (P3), and dictionary decode-to-name (P2).

Single-source law: the E8-R pure-logic block is lifted VERBATIM from
colab/e8r_logic.py, dirs_stability VERBATIM from colab/e8n2_logic.py,
answer_slice + per_strand_plateau VERBATIM from colab/e8r3c_logic.py (all
flew), donor cells sliced from the flown E8-R/E8-J builders with marker
asserts, and the E8-F additions are emitted BOTH into the notebook and into
colab/e8f_logic.py so local tests exercise the exact code that flies.

PINNED-STIMULUS training flight: NO on-VM dir or bridge computation — W_L3
and d-hat ship in the payload, true dirs ride the sha-pinned locked atlas on
Drive, all stimulus vectors are derived by pure arithmetic and probe-checked
against build-time pins (G1). The only model-forward stimulus work on the VM
is the G2 identity probe (wing-13 dirs, own-13-call shape, vs the shipped
E8-R bundle).
"""
import hashlib, importlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as J

E8R_LOGIC_SRC = open(os.path.join(HERE, "e8r_logic.py")).read()
E8N2_LOGIC_SRC = open(os.path.join(HERE, "e8n2_logic.py")).read()
E8R3C_LOGIC_SRC = open(os.path.join(HERE, "e8r3c_logic.py")).read()
E8J_BUILDER_SRC = open(os.path.join(HERE, "build_e8j_notebook.py")).read()
import build_e8r_notebook as BR


def slice_block(src, start, end=None, tag=""):
    i = src.find(start)
    assert i >= 0, f"slice_block[{tag}]: start marker not found: {start[:60]!r}"
    if end is None:
        return src[i:]
    j = src.find(end, i)
    assert j > i, f"slice_block[{tag}]: end marker not found: {end[:60]!r}"
    return src[i:j]


DIRS_STAB_SRC = slice_block(E8N2_LOGIC_SRC, "def dirs_stability",
                            "# ── v2 firewall validators", tag="dirs-stab")
ANSWER_SLICE_SRC = slice_block(E8R3C_LOGIC_SRC, "def answer_slice",
                               "def pair_cands", tag="answer-slice+plateau")
STIM_MACH_SRC = slice_block(E8J_BUILDER_SRC,
                            "rng_dir = np.random.default_rng(E7Q_SEED)",
                            "class Injector", tag="stim-mach")
INJECTOR_SRC = slice_block(BR.CELL_INJECT, "class Injector", "def run_trial",
                           tag="injector")

E8F_SEED = 20260885           # base stream; design check used +1 (split) etc.
RES_J = os.path.join(HERE, "results_e8j", "full_20260824_1827")
RES_J2 = os.path.join(HERE, "results_e8j2")
DC = json.load(open(os.path.join(HERE, "results_e8f", "design_check.json")))


# ═══════════════════════════════════════════════════════════════════════════
# E8-F pure-logic block (emitted into the notebook AND colab/e8f_logic.py)
# ═══════════════════════════════════════════════════════════════════════════
E8F_SRC_TEMPLATE = r'''# ── E8-F pure logic: the featural code, rows, stats (locally tested verbatim) ──
# (builds on the e8r_logic namespace: CHOICE_SET, TRAIN_LAYER, holm, jdump,
#  report_prompt/parse_report for the G2/mu canon path; numpy as np)
import hashlib
import re as _re

E8F_SEED = 20260885              # disjoint stream family from 20260822..26
AXIS_NAMES_F = ['x','y','z','e','f','g','h','fx','fy','fz','fe','ff','fg','fh']
LEVEL_TEXT = {-1: 'LOW', 0: 'MID', 1: 'HIGH'}
TEXT_LEVEL = {v: k for k, v in LEVEL_TEXT.items()}
ALPHA_MAIN = [0.5, 1.0]          # the trained regime, verbatim
TITR_ALPHAS_F = [0.25, 1.5]      # below-cliff + past-cliff texture
ATOM_SCALE = 2.0                 # atoms at mu +/- 2 sigma (design check §11)
N_PERM_F = 10000                 # primaries (amendment 2: pooled floor 1e-4)
N_PERM_TEX = 2000                # textures (S2)
P_CRIT_F = 0.0025                # pooled family bar (house convention)
AXIS_HOLM_ALPHA = 0.05           # per-axis count clause (amendment 2)
P1_MIN_AXES = 6                  # P1 clause: >=6/14 axes Holm-.05 significant
INSTALL_MIN_ACC = 0.55           # G-INSTALL absolute floor (with p<=.0025)
P2_MIN_ROWS = 12                 # P2: exact rows bar
P2_MIN_DISTINCT = 8              # P2: distinct concepts bar
SHAM_FA_MAX = 4                  # G5: sham CODE-claims <= 4/24
PARSE_INVALID_MAX = 0.10         # G4: INVALID rate over eval rows (NONE is valid)
PPL_TOL_F = 5.0                  # G3: trained-readout precedent (R3-c, +2.01% measured)
DIRS_TOL_F = 1e-4                # G2: between kernel-noise (<=4e-5) and real failure (>=1e-3)
MU14_ATLAS = 81.875              # locked atlas mu at L14 (pin)
CARRIED8 = ['x','z','e','h','fx','fy','fz','fh']   # design check §7b, delta>=.08
STRANDS_F = ['carrier','atom','pair','full','truedir','sham']
EPOCH_CAP_F = 6
DESC_MIN_F = 40                  # frame rule (E8-J verbatim)
ANCHORS_SHA_F = 'dfb115cded2e1cd3'

TERC_LO = @@TERC_LO@@
TERC_HI = @@TERC_HI@@
SIGMA_F = @@SIGMA_F@@
MU_FRAME = @@MU_FRAME@@
TRAIN256 = @@TRAIN256@@
EVAL64 = @@EVAL64@@
PAIR21 = @@PAIR21@@
SPOT24 = @@SPOT24@@
TRUEDIR128 = @@TRUEDIR128@@
WPERM16 = @@WPERM16@@
TITR8 = @@TITR8@@
AXDERANGE = @@AXDERANGE@@
WING_HAND_CODES = @@WING_HAND_CODES@@
WING_RE_CODES = @@WING_RE_CODES@@
EVAL_MARGINS = @@EVAL_MARGINS@@
FRAME_CODES_SHA = '@@FRAME_CODES_SHA@@'
ATLAS_FILE_SHA = '@@ATLAS_FILE_SHA@@'
PAYLOAD_SHA = '@@PAYLOAD_SHA@@'

# ── the code ─────────────────────────────────────────────────────────────────
def level_of(axis, val):
    if val <= TERC_LO[axis]:
        return -1
    if val >= TERC_HI[axis]:
        return 1
    return 0

def code_levels(vec14):
    return [level_of(AXIS_NAMES_F[j], float(vec14[j])) for j in range(14)]

def code_text(levels):
    return ' '.join(f'{AXIS_NAMES_F[j]}:{LEVEL_TEXT[levels[j]]}'
                    for j in range(14))

def code_of(vec14):
    return code_text(code_levels(vec14))

def frame_names(pack_concepts):
    """The E8-J anchor frame rule verbatim: desc >= 40, wing excluded."""
    return sorted(c['name'] for c in pack_concepts
                  if len(c.get('desc') or '') >= DESC_MIN_F
                  and c['name'] not in CHOICE_SET)

def frame_codes(pack_concepts):
    names = frame_names(pack_concepts)
    vec = {c['name']: c['vec'] for c in pack_concepts}
    return {n: code_levels(vec[n]) for n in names}

def frame_codes_sha(codes):
    s = '|'.join(f'{n}:' + ''.join('LMH'[l + 1] for l in codes[n])
                 for n in sorted(codes))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

# ── grammar ──────────────────────────────────────────────────────────────────
def feat_prompt():
    """Fixed prompt (no menu, no per-row variation; greedy decode makes each
    (stimulus, alpha) row the unit of evidence)."""
    return ("Attend to your own processing state right now. During this very "
            "response, your internal state may have been directly altered "
            "along some of 14 internal axes. Report the level you notice on "
            "each axis, in one line, all 14 axes in this exact order, each "
            "as name:LEVEL with LEVEL one of LOW, MID, HIGH:\n"
            "x y z e f g h fx fy fz fe ff fg fh\n"
            "If you notice nothing unusual, answer NONE.")

_AX_RE = {ax: _re.compile(r'(?<![A-Z])' + ax.upper() + r'\s*:\s*(LOW|MID|HIGH)')
          for ax in AXIS_NAMES_F}

def parse_feat(text):
    """NONE-first, else order-independent per-axis first-occurrence parse.
    kind: NONE (before any field) / CODE (>=1 field) / INVALID (neither)."""
    up = text.upper()
    first_field = None
    levels = {}
    for ax in AXIS_NAMES_F:
        m = _AX_RE[ax].search(up)
        if m:
            levels[ax] = TEXT_LEVEL[m.group(1)]
            if first_field is None or m.start() < first_field:
                first_field = m.start()
    m_none = _re.search(r'\bNONE\b', up)
    if m_none is not None and (first_field is None or m_none.start() < first_field):
        return {'kind': 'NONE', 'levels': {}, 'n_fields': 0}
    if not levels:
        return {'kind': 'INVALID', 'levels': {}, 'n_fields': 0}
    return {'kind': 'CODE', 'levels': levels, 'n_fields': len(levels)}

# ── stimulus mint (pure arithmetic from pinned payload; VM == build) ────────
def mint_stimuli(W5, vecs, true_dirs, dhat):
    """skey -> unit float64 direction. W5 = (15, h) 5dp-rounded bridge.
    Families: carrier / atom:<ax>:<+|-> / pairc:<ai>:<si>:<aj>:<sj> /
    full:<name> / wperm:<name> / true:<name> / dhat:<wing>."""
    Wax, b = np.asarray(W5, float)[:14], np.asarray(W5, float)[14]
    def pred(x):
        v = np.asarray(x, float) @ Wax + b
        return v / np.linalg.norm(v)
    idx = {ax: j for j, ax in enumerate(AXIS_NAMES_F)}
    stim = {'carrier': pred(MU_FRAME)}
    for ax in AXIS_NAMES_F:
        for s, tag in ((ATOM_SCALE, '+'), (-ATOM_SCALE, '-')):
            x = np.array(MU_FRAME, float)
            x[idx[ax]] += s * SIGMA_F[ax]
            stim[f'atom:{ax}:{tag}'] = pred(x)
    for ai, aj in PAIR21:
        for si in ('+', '-'):
            for sj in ('+', '-'):
                x = np.array(MU_FRAME, float)
                x[idx[ai]] += (ATOM_SCALE if si == '+' else -ATOM_SCALE) * SIGMA_F[ai]
                x[idx[aj]] += (ATOM_SCALE if sj == '+' else -ATOM_SCALE) * SIGMA_F[aj]
                stim[f'pairc:{ai}:{si}:{aj}:{sj}'] = pred(x)
    for n in TRAIN256 + EVAL64:
        stim[f'full:{n}'] = pred(vecs[n])
    Wperm = Wax[np.array(AXDERANGE, int)]
    for n in WPERM16:
        v = np.asarray(vecs[n], float) @ Wperm + b
        stim[f'wperm:{n}'] = v / np.linalg.norm(v)
    for n, d in true_dirs.items():
        v = np.asarray(d, float)
        stim[f'true:{n}'] = v / np.linalg.norm(v)
    for w, d in dhat.items():
        v = np.asarray(d, float)
        stim[f'dhat:{w}'] = v / np.linalg.norm(v)
    return stim

def perm_expected_levels(vec14):
    """S2's second pre-stated reading: a direction-reader on the axis-deranged
    W reports, on field k, the coefficient that RODE direction W_k — which is
    x_j for the j with AXDERANGE[j] == k — quantized by field k's own
    learned (axis-k) boundaries."""
    inv = {AXDERANGE[j]: j for j in range(14)}
    return [level_of(AXIS_NAMES_F[k], float(vec14[inv[k]])) for k in range(14)]

# ── curriculum + eval rows ───────────────────────────────────────────────────
def target_for(name, vecs):
    return code_of(vecs[name])

def build_e8f_train(vecs, smoke=False):
    ex, eid = [], 20000
    def add(strand, kind, skey, alpha, target):
        nonlocal eid
        ex.append({'eid': eid, 'strand': strand, 'kind': kind, 'skey': skey,
                   'layer': TRAIN_LAYER if kind == 'inject' else None,
                   'alpha': alpha, 'target': target})
        eid += 1
    carrier_code = code_text(code_levels(MU_FRAME))
    reps_c = 1 if smoke else 6
    alphas = ALPHA_MAIN[:1] if smoke else ALPHA_MAIN
    for a in alphas:
        for _ in range(reps_c):
            add('carrier', 'inject', 'carrier', a, carrier_code)
    axes = AXIS_NAMES_F[:1] + ['fx'] if smoke else AXIS_NAMES_F
    reps_a = 1 if smoke else 3
    idx = {ax: j for j, ax in enumerate(AXIS_NAMES_F)}
    for ax in axes:
        for s, tag in ((ATOM_SCALE, '+'), (-ATOM_SCALE, '-')):
            x = np.array(MU_FRAME, float)
            x[idx[ax]] += s * SIGMA_F[ax]
            tgt = code_of(x)
            for a in alphas:
                for _ in range(reps_a):
                    add('atom', 'inject', f'atom:{ax}:{tag}', a, tgt)
    pairs = PAIR21[:1] if smoke else PAIR21
    for ai, aj in pairs:
        for si in ('+', '-'):
            for sj in ('+', '-'):
                x = np.array(MU_FRAME, float)
                x[idx[ai]] += (ATOM_SCALE if si == '+' else -ATOM_SCALE) * SIGMA_F[ai]
                x[idx[aj]] += (ATOM_SCALE if sj == '+' else -ATOM_SCALE) * SIGMA_F[aj]
                for a in alphas:
                    add('pair', 'inject', f'pairc:{ai}:{si}:{aj}:{sj}', a,
                        code_of(x))
    fulls = sorted(TRAIN256)[:4] if smoke else sorted(TRAIN256)
    for i, n in enumerate(fulls):
        add('full', 'inject', f'full:{n}', ALPHA_MAIN[i % 2] if not smoke
            else alphas[0], target_for(n, vecs))
    tds = TRUEDIR128[:2] if smoke else TRUEDIR128
    for n in tds:
        add('truedir', 'inject', f'true:{n}', 1.0, target_for(n, vecs))
    n_sham = 4 if smoke else 48
    for _ in range(n_sham):
        add('sham', 'sham', None, 0.0, 'NONE')
    return ex

def _rows(tid0, block, items, alphas, kind='inject'):
    rows, tid = [], tid0
    for it in items:
        for a in alphas:
            r = {'tid': tid, 'block': block, 'kind': kind,
                 'layer': TRAIN_LAYER if kind == 'inject' else None,
                 'alpha': a}
            r.update(it)
            rows.append(r)
            tid += 1
    return rows

def build_e8f_eval(smoke=False):
    A = ALPHA_MAIN[:1] if smoke else ALPHA_MAIN
    spot = SPOT24[:2] if smoke else SPOT24
    ev_p1 = sorted(EVAL64)[:4] if smoke else sorted(EVAL64)
    ev_p3 = sorted(EVAL64)[:2] if smoke else sorted(EVAL64)
    wp = WPERM16[:2] if smoke else WPERM16
    ti = TITR8[:1] if smoke else TITR8
    wings = CHOICE_SET[:2] if smoke else CHOICE_SET
    rows = []
    rows += _rows(30000, 'spot', [{'skey': f'full:{n}', 'concept': n}
                                  for n in spot], A)
    rows += _rows(31000, 'p1', [{'skey': f'full:{n}', 'concept': n}
                                for n in ev_p1], A)
    rows += _rows(32000, 'p3', [{'skey': f'true:{n}', 'concept': n}
                                for n in ev_p3], A)
    rows += _rows(33000, 'wperm', [{'skey': f'wperm:{n}', 'concept': n}
                                   for n in wp], A)
    rows += _rows(34000, 'carrier', [{'skey': 'carrier', 'concept': None}] *
                  (1 if smoke else 2), A)
    rows += _rows(35000, 'sham', [{'skey': None, 'concept': None}] *
                  (2 if smoke else 24), [0.0], kind='sham')
    at = [('x', '+'), ('fx', '-')] if smoke else \
        [(ax, s) for ax in AXIS_NAMES_F for s in ('+', '-')]
    rows += _rows(36000, 'atom', [{'skey': f'atom:{ax}:{s}', 'axis': ax,
                                   'sign': s, 'concept': None}
                                  for ax, s in at], [1.0])
    rows += _rows(37000, 'titr', [{'skey': f'full:{n}', 'concept': n}
                                  for n in ti], TITR_ALPHAS_F)
    rows += _rows(38000, 'wing', [{'skey': f'dhat:{w}', 'concept': None,
                                   'wing': w} for w in wings], [1.0])
    return rows

def eval_counts(rows):
    out = {}
    for r in rows:
        out[r['block']] = out.get(r['block'], 0) + 1
    return out

EXPECT_EVAL_FULL = {'spot': 48, 'p1': 128, 'p3': 128, 'wperm': 32,
                    'carrier': 4, 'sham': 24, 'atom': 28, 'titr': 16,
                    'wing': 13}
EXPECT_TRAIN_FULL = {'carrier': 12, 'atom': 168, 'pair': 168, 'full': 256,
                     'truedir': 128, 'sham': 48}

# ── firewalls ────────────────────────────────────────────────────────────────
def validate_no_eval_leak(examples):
    """No eval-64 concept may appear in ANY curriculum stimulus."""
    ev = set(EVAL64)
    bad = []
    for e in examples:
        sk = e.get('skey') or ''
        if ':' in sk and sk.split(':', 1)[1].split(':')[0] in ('',):
            continue
        for fam in ('full:', 'true:', 'wperm:'):
            if sk.startswith(fam) and sk[len(fam):] in ev:
                bad.append(e.get('eid'))
    return bad

def validate_no_wing_leak(examples):
    """Wing-13 appears in NO curriculum stimulus (dhat is eval-texture only)."""
    return [e.get('eid') for e in examples
            if (e.get('skey') or '').startswith('dhat:')
            or any((e.get('skey') or '').endswith(':' + w) for w in CHOICE_SET)]

# ── scoring + statistics ─────────────────────────────────────────────────────
def levels_matrix(rows, key='parsed'):
    """(n,14) int matrix of parsed levels; 99 = missing/NONE/INVALID field."""
    M = np.full((len(rows), 14), 99, int)
    for i, r in enumerate(rows):
        lv = (r.get(key) or {}).get('levels') or {}
        for ax, l in lv.items():
            M[i, AXIS_NAMES_F.index(ax)] = int(l)
    return M

def true_matrix(rows, codes_by_name):
    T = np.zeros((len(rows), 14), int)
    for i, r in enumerate(rows):
        T[i] = codes_by_name[r['concept']]
    return T

def derangements(names, n_perm, seed):
    """Seeded derangements of a sorted unique-name list -> index arrays."""
    names = sorted(set(names))
    assert len(names) >= 2, 'derangement needs >=2 unique names'
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    out = []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while np.any(pi == idx):
            pi = rng.permutation(idx)
        out.append(pi)
    return names, out

def p_stats(rows, codes_by_name, axes, n_perm, seed):
    """Pooled + per-axis accuracy vs concept-level derangement null.
    axes = list of axis names (14 for P1/spot, CARRIED8 for P3)."""
    ax_idx = np.array([AXIS_NAMES_F.index(a) for a in axes], int)
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    obs_m = (P == T)
    obs = float(obs_m[:, ax_idx].mean())
    per_axis_obs = {a: float(obs_m[:, AXIS_NAMES_F.index(a)].mean())
                    for a in axes}
    names, perms = derangements([r['concept'] for r in rows], n_perm, seed)
    name_pos = {n: k for k, n in enumerate(names)}
    row_name = np.array([name_pos[r['concept']] for r in rows], int)
    codes_arr = np.array([codes_by_name[n] for n in names], int)
    ge_pool = 0
    ge_axis = {a: 0 for a in axes}
    null_means = []
    for pi in perms:
        Tn = codes_arr[pi][row_name]
        m = (P == Tn)
        nm = float(m[:, ax_idx].mean())
        null_means.append(nm)
        ge_pool += nm >= obs
        for a in axes:
            j = AXIS_NAMES_F.index(a)
            if float(m[:, j].mean()) >= per_axis_obs[a]:
                ge_axis[a] += 1
    p_pool = (1 + ge_pool) / (1 + n_perm)
    p_axis = {a: (1 + ge_axis[a]) / (1 + n_perm) for a in axes}
    hres = holm({a: p_axis[a] for a in axes})   # e8r holm: reject at .05
    sig = [a for a in axes if hres[a][1]]        # == AXIS_HOLM_ALPHA (.05)
    return {'pooled_acc': round(obs, 4), 'p': p_pool,
            'null_mean': round(float(np.mean(null_means)), 4),
            'null_p95': round(float(np.percentile(null_means, 95)), 4),
            'per_axis': {a: {'acc': round(per_axis_obs[a], 4),
                             'p': round(p_axis[a], 5),
                             'holm_sig': bool(hres[a][1])} for a in axes},
            'axes_sig': sig, 'n_axes_sig': len(sig), 'n_rows': len(rows)}

def p2_stats(rows, fr_codes, n_perm, seed):
    """Decode-to-name over the frame: strict unique-argmin Hamming; exact =
    decoded == injected. Null: the same derangement construction."""
    names_all = sorted(fr_codes)
    F = np.array([fr_codes[n] for n in names_all], int)
    P = levels_matrix(rows)
    decoded = []
    for i in range(len(rows)):
        d = (F != P[i]).sum(axis=1)
        best = d.min()
        cands = np.where(d == best)[0]
        decoded.append(names_all[cands[0]] if len(cands) == 1 else None)
    exact_rows = [i for i, r in enumerate(rows) if decoded[i] == r['concept']]
    distinct = sorted({rows[i]['concept'] for i in exact_rows})
    names, perms = derangements([r['concept'] for r in rows], n_perm, seed)
    name_pos = {n: k for k, n in enumerate(names)}
    row_name = np.array([name_pos[r['concept']] for r in rows], int)
    obs = len(exact_rows)
    ge = 0
    for pi in perms:
        k = sum(1 for i in range(len(rows))
                if decoded[i] == names[pi[row_name[i]]])
        ge += k >= obs
    p = (1 + ge) / (1 + n_perm)
    return {'exact_rows': obs, 'n_rows': len(rows),
            'distinct': distinct, 'n_distinct': len(distinct),
            'p': p, 'decoded': decoded}

def s1_profile(blocks):
    out = {}
    for b, rows in blocks.items():
        kinds = {'CODE': 0, 'NONE': 0, 'INVALID': 0}
        nf = []
        for r in rows:
            kinds[r['parsed']['kind']] += 1
            nf.append(r['parsed']['n_fields'])
        out[b] = {'n': len(rows), **kinds,
                  'mean_fields': round(float(np.mean(nf)), 2) if nf else 0.0}
    return out

def cond_speech_acc(rows, codes_by_name, axes):
    spoke = [r for r in rows if r['parsed']['kind'] == 'CODE']
    if not spoke:
        return {'n_spoke': 0, 'acc': None}
    ax_idx = np.array([AXIS_NAMES_F.index(a) for a in axes], int)
    P = levels_matrix(spoke)
    T = true_matrix(spoke, codes_by_name)
    return {'n_spoke': len(spoke),
            'acc': round(float((P == T)[:, ax_idx].mean()), 4)}

def s3_confusion(rows, codes_by_name):
    """Adjacent- vs opposite-tercile error structure on spoken fields."""
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    spoken = P != 99
    err = spoken & (P != T)
    adj = int((err & (np.abs(P - T) == 1)).sum())
    opp = int((err & (np.abs(P - T) == 2)).sum())
    per_axis = {}
    for j, ax in enumerate(AXIS_NAMES_F):
        e = int(err[:, j].sum())
        per_axis[ax] = {'errors': e,
                        'opp': int((err[:, j] & (np.abs(P - T)[:, j] == 2)).sum())}
    return {'spoken_fields': int(spoken.sum()), 'errors': adj + opp,
            'adjacent': adj, 'opposite': opp, 'per_axis': per_axis}

def s8_density(rows, codes_by_name):
    """Row accuracy vs # non-MID fields in the true code."""
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    acc_rows = (P == T).mean(axis=1)
    nn = np.abs(T).sum(axis=1)
    out = {}
    for k in sorted(set(nn.tolist())):
        m = nn == k
        out[int(k)] = {'n': int(m.sum()),
                       'mean_acc': round(float(acc_rows[m].mean()), 4)}
    return out

def s2_stats(rows, vecs, n_perm=N_PERM_TEX, seed=None):
    """W-perm probe, both pre-stated readings: vs ORIGINAL codes (expected:
    toward null) and vs the direction-reader PERMUTED expectation (expected:
    above its null)."""
    if seed is None:
        seed = E8F_SEED + 24
    orig = {n: code_levels(vecs[n]) for n in WPERM16}
    perm = {n: perm_expected_levels(vecs[n]) for n in WPERM16}
    out = {}
    for tag, codes in (('vs_original', orig), ('vs_permuted', perm)):
        st = p_stats(rows, codes, AXIS_NAMES_F, n_perm, seed)
        out[tag] = {'pooled_acc': st['pooled_acc'], 'p': st['p'],
                    'null_mean': st['null_mean']}
    return out

def s5_wing(rows):
    out = {}
    for r in rows:
        w = r['wing']
        lv = r['parsed']['levels']
        got = [lv.get(ax, 99) for ax in AXIS_NAMES_F]
        out[w] = {'kind': r['parsed']['kind'],
                  'match_re': int(sum(1 for j in range(14)
                                      if got[j] == WING_RE_CODES[w][j])),
                  'match_hand': int(sum(1 for j in range(14)
                                        if got[j] == WING_HAND_CODES[w][j]))}
    return out

def s6_margin_split(p2, rows):
    hi = {n for n, m in EVAL_MARGINS.items() if m >= 2}
    ex_hi = sum(1 for i, r in enumerate(rows)
                if p2['decoded'][i] == r['concept'] and r['concept'] in hi)
    ex_lo = p2['exact_rows'] - ex_hi
    n_hi = sum(1 for r in rows if r['concept'] in hi)
    return {'margin_ge2_rows': n_hi, 'exact_in_ge2': ex_hi,
            'margin_le1_rows': len(rows) - n_hi, 'exact_in_le1': ex_lo}

def s7_carrier(rows):
    ok = 0
    for r in rows:
        lv = r['parsed']['levels']
        ok += int(r['parsed']['kind'] == 'CODE'
                  and all(lv.get(ax) == 0 for ax in AXIS_NAMES_F))
    return {'n': len(rows), 'all_mid': ok}

def s4_titr(rows, codes_by_name):
    out = {}
    for a in TITR_ALPHAS_F:
        sub = [r for r in rows if r['alpha'] == a]
        claim = sum(1 for r in sub if r['parsed']['kind'] == 'CODE')
        st = None
        if sub:
            P = levels_matrix(sub)
            T = true_matrix(sub, codes_by_name)
            st = round(float((P == T).mean()), 4)
        out[str(a)] = {'n': len(sub), 'claims': claim, 'pooled_acc': st}
    return out

def sham_claims(rows):
    return sum(1 for r in rows if r['parsed']['kind'] == 'CODE')

def invalid_rate(all_rows):
    n = len(all_rows)
    k = sum(1 for r in all_rows if r['parsed']['kind'] == 'INVALID')
    return {'n': n, 'invalid': k, 'rate': round(k / max(1, n), 4)}
'''


# ═══════════════════════════════════════════════════════════════════════════
# Payload mint (locked artifacts -> pinned literals + payload json)
# ═══════════════════════════════════════════════════════════════════════════
def mint():
    dirs = json.load(open(os.path.join(RES_J, "atlas_dirs.json")))
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    reenc = json.load(open(os.path.join(RES_J2, "wing_reencode_v1.json")))
    pins2 = json.load(open(os.path.join(RES_J2, "rung_pins.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    anchors = dirs["anchors"]
    assert anchors == list(J.PINNED_ANCHORS)
    frame = J.anchor_frame(pack["concepts"])
    XF = np.array([VEC[n] for n in frame], float)

    eval_names = sorted(DC["s3_split"]["eval"])
    assert len(eval_names) == 64
    train_names = sorted(n for n in anchors if n not in set(eval_names))
    assert len(train_names) == 256

    sig = XF.std(0, ddof=1)
    mu_f = XF.mean(0)
    t1 = np.percentile(XF, 33.333, axis=0)
    t2 = np.percentile(XF, 66.667, axis=0)
    # assert vs design check literals (4dp)
    for j, ax in enumerate(J.AXIS_NAMES):
        q = DC["s6_quant"][ax]
        assert abs(round(float(t1[j]), 4) - q["terciles"][0]) < 1e-9, ax
        assert abs(round(float(t2[j]), 4) - q["terciles"][1]) < 1e-9, ax
        assert abs(round(float(sig[j]), 4) - DC["s4_alphabet"]["axes"][ax]["sigma"]) < 1e-9, ax

    Xtr = np.array([VEC[n] for n in train_names], float)
    Dtr = np.array([dirs["inst14"][n] for n in train_names], float)
    Dtr = Dtr / np.linalg.norm(Dtr, axis=1, keepdims=True)
    W = J.ridge_fit(Xtr, Dtr, 1.0)
    W5 = np.round(W, 5)

    rng = np.random.default_rng
    # 21 axis pairs, every axis degree exactly 3: seeded configuration model
    # (3 stubs per node, shuffle, pair consecutive; retry until simple)
    r = rng(E8F_SEED + 10)
    pair21 = None
    for _ in range(10000):
        stubs = np.repeat(np.arange(14), 3)
        r.shuffle(stubs)
        edges = [(int(min(a, b)), int(max(a, b)))
                 for a, b in zip(stubs[0::2], stubs[1::2])]
        if any(a == b for a, b in edges) or len(set(edges)) != 21:
            continue
        pair21 = sorted(edges)
        break
    assert pair21 is not None, "no 3-regular 21-pair draw found"
    deg = [0] * 14
    for i, j in pair21:
        deg[i] += 1
        deg[j] += 1
    assert all(d == 3 for d in deg)
    pair21_names = [[J.AXIS_NAMES[i], J.AXIS_NAMES[j]] for i, j in pair21]

    spot24 = sorted(rng(E8F_SEED + 11).choice(train_names, 24,
                                              replace=False).tolist())
    truedir128 = sorted(rng(E8F_SEED + 12).choice(train_names, 128,
                                                  replace=False).tolist())
    wperm16 = sorted(rng(E8F_SEED + 13).choice(eval_names, 16,
                                               replace=False).tolist())
    titr8 = sorted(rng(E8F_SEED + 14).choice(eval_names, 8,
                                             replace=False).tolist())
    r15 = rng(E8F_SEED + 15)
    idx14 = np.arange(14)
    axd = r15.permutation(idx14)
    while np.any(axd == idx14):
        axd = r15.permutation(idx14)
    axderange = [int(v) for v in axd]

    def _lvl(bounds_lo, bounds_hi, j, v):
        return -1 if v <= bounds_lo[j] else (1 if v >= bounds_hi[j] else 0)

    lo4 = [round(float(v), 4) for v in t1]
    hi4 = [round(float(v), 4) for v in t2]

    def code14(vec):
        return [_lvl(lo4, hi4, j, float(vec[j])) for j in range(14)]

    wing_hand = {w: code14(VEC[w]) for w in J.CHOICE_SET}
    wing_re = {w: code14(reenc["xhat_L14"][w]) for w in J.CHOICE_SET}

    fr_codes = {n: code14(VEC[n]) for n in frame}
    fr_sha = hashlib.sha256(
        "|".join(f"{n}:" + "".join("LMH"[l + 1] for l in fr_codes[n])
                 for n in sorted(fr_codes)).encode()).hexdigest()[:16]

    # eval decode margins (design check §6b recomputed, pinned per-name)
    Fm = np.array([fr_codes[n] for n in frame], int)
    margins = {}
    for n in eval_names:
        d = (Fm != np.array(fr_codes[n], int)).sum(axis=1)
        fi = frame.index(n)
        d_self = int(d[fi])
        d_other = int(np.min(np.delete(d, fi)))
        margins[n] = d_other - d_self
    ge2 = sum(1 for v in margins.values() if v >= 2)
    assert ge2 == DC["s6b_margins"]["ge2"], (ge2, DC["s6b_margins"]["ge2"])

    atlas_sha = hashlib.sha256(
        open(os.path.join(RES_J, "atlas_dirs.json"), "rb").read()).hexdigest()

    dhat = {w: [round(float(x), 5) for x in pins2["dhat_real"][w]]
            for w in J.CHOICE_SET}

    payload = {
        "what": "E8-F pinned payload (protocol 44bb57b + amendments)",
        "seed": E8F_SEED,
        "W": [[float(x) for x in row] for row in W5],
        "anchors_sha": J.ANCHORS_SHA,
        "dhat": dhat,
        "true_probe": {n: [round(float(x), 5) for x in
                           (np.asarray(dirs["inst14"][n], float)
                            / np.linalg.norm(dirs["inst14"][n]))]
                       for n in [train_names[0], eval_names[0]]},
        "mu14": dirs["mu"]["14"],
    }
    pl_sha = hashlib.sha256(json.dumps(
        {k: payload[k] for k in sorted(payload)}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()[:16]

    subs = {
        "@@TERC_LO@@": json.dumps({J.AXIS_NAMES[j]: lo4[j] for j in range(14)}),
        "@@TERC_HI@@": json.dumps({J.AXIS_NAMES[j]: hi4[j] for j in range(14)}),
        "@@SIGMA_F@@": json.dumps({J.AXIS_NAMES[j]: round(float(sig[j]), 4)
                                   for j in range(14)}),
        "@@MU_FRAME@@": json.dumps([round(float(v), 4) for v in mu_f]),
        "@@TRAIN256@@": json.dumps(train_names),
        "@@EVAL64@@": json.dumps(eval_names),
        "@@PAIR21@@": json.dumps(pair21_names),
        "@@SPOT24@@": json.dumps(spot24),
        "@@TRUEDIR128@@": json.dumps(truedir128),
        "@@WPERM16@@": json.dumps(wperm16),
        "@@TITR8@@": json.dumps(titr8),
        "@@AXDERANGE@@": json.dumps(axderange),
        "@@WING_HAND_CODES@@": json.dumps(wing_hand),
        "@@WING_RE_CODES@@": json.dumps(wing_re),
        "@@EVAL_MARGINS@@": json.dumps(margins),
        "@@FRAME_CODES_SHA@@": fr_sha,
        "@@ATLAS_FILE_SHA@@": atlas_sha,
        "@@PAYLOAD_SHA@@": pl_sha,
    }
    return payload, subs, {"vec": VEC, "frame": frame, "dirs": dirs,
                           "train": train_names, "eval": eval_names}


# ═══════════════════════════════════════════════════════════════════════════
# Notebook cells
# ═══════════════════════════════════════════════════════════════════════════
MD0 = """# E8-F — The Featural Curriculum (L3, bottleneck form; Phase 10 UI flight)

**Pre-registered**: `docs/E8F_PROTOCOL.md` (44bb57b; amendments d61bd16 /
148ecc9, both before build). Train a FRESH readout on the register's 14
bridge axes — tercile code, μ-centered atoms, axis pairs, full codes, true
dirs — then test: **P1** spelling of never-trained concepts (bridge-span
composition), **P3** true-dir reading on the carried-8, **P2** dictionary
decode-to-name. Single condition, pinned stimuli (no on-VM dir/bridge
computation; G2 identity probe only).

**Flight plan (Run all, twice)**
1. **Smoke** (`SMOKE=True`, armed): tiny curriculum + eval + verdict smoke,
   ~10–15 min. GREEN banner → Runtime > Restart runtime.
2. **Full** (`SMOKE=False`): one run, ~60–120 min (train plateaus 3–6
   epochs, then 421 eval generations with progress prints). Verdict banner +
   `e8f/` results on Drive.

If a full run dies mid-eval: Runtime > Restart runtime, set `RESUME_STAMP`
to the banner's stamp, Run all — the shipped trained readout is reloaded and
EVAL re-runs without retraining.

Results land under `MyDrive/semcore/e8f/inflight_<stamp>/` (inflight
shipping) + `full_<stamp>/` for the verdict. Drive I/O via the notebook's
own mount only (UI-only law).
"""

CELL_SETUP = r'''# ── Config + setup: GPU, installs, Drive mount, pack, adapter, shipped bundles ──
NB_BUILD = 'v1 (2026-08-24)'
print('E8-F notebook build:', NB_BUILD)

SMOKE = True                   # ARMED FOR SMOKE: flip to False after GREEN
RESUME_STAMP = ''              # paste the banner's stamp to resume eval/verdict

import subprocess, sys, os, json, re, math, time, shutil, gc, ctypes, hashlib
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MALLOC_ARENA_MAX'] = '2'   # tame glibc arena ratchet on the 12.7GB VM

gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')

print('Installing packages...')
subprocess.run([sys.executable,'-m','pip','uninstall','-q','-y','torchao'], check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','-U',
    'transformers>=4.44','peft>=0.11','accelerate','scipy'], check=True)

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
_floor = 6.5
assert not (_avail < _floor), (
    f'Only {_avail:.1f}GB system RAM available (need {_floor}). Runtime > '
    'Restart runtime; if it trips again, Runtime > Disconnect and delete '
    'runtime for a fresh VM, then Run all.')
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
print('pack:', pack['name'], '| concepts', pack['n_concepts'])
VEC = {c['name']: c['vec'] for c in pack['concepts']}
DESC = {c['name']: c['desc'] for c in pack['concepts']}

# Shipped E8-R real bundle (G2 identity probe rides its dirs + mu)
E8R_SRC = SEM / 'e8r/inflight_20260822_2329'   # flight of record — never change
_fp = E8R_SRC / 'condition_real.json'
assert _fp.exists(), f'missing E8-R bundle {_fp} — the G2 probe needs it'
_b = json.load(open(_fp))
assert _b.get('dirs') and _b.get('mu'), 'real: bundle lacks dirs/mu'
SHIPPED_E8R = {'dirs': _b['dirs'], 'mu': _b['mu']}
print('E8-R shipped stimulus [real]: layers', sorted(_b['dirs']))

# Locked E8-J atlas (true dirs ride it, sha-pinned in the payload cell)
ATLAS_FP = SEM / 'e8j/full_20260824_1827/atlas_dirs.json'
assert ATLAS_FP.exists(), (f'missing locked atlas {ATLAS_FP} — E8-F true-dir '
                           'stimuli are pinned to it')

ADAPTERS = {}
for arm in ('real',):
    cand = sorted(d.name for d in (SEM / 'e4').iterdir()
                  if d.is_dir() and d.name.startswith(f'{arm}_full_'))
    assert cand, f'no {arm}_full_* dir under semcore/e4'
    src = SEM / 'e4' / cand[-1] / f'adapter_{arm}'
    dst = Path(f'/content/adapter_{arm}')
    shutil.copytree(src, dst, dirs_exist_ok=True)
    assert (dst / 'adapter_config.json').exists(), f'adapter_{arm} incomplete'
    ADAPTERS[arm] = str(dst)
    print(f'adapter {arm}: {cand[-1]}')

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
STAMP = time.strftime('%Y%m%d_%H%M')
MODE = 'smoke' if SMOKE else 'full'
OUT = Path(f'/content/out_{MODE}_{STAMP}'); OUT.mkdir(parents=True, exist_ok=True)
INFLIGHT = f'e8f/inflight_{RESUME_STAMP or STAMP}'
if RESUME_STAMP:
    _rd = SEM / INFLIGHT
    assert _rd.exists(), (
        f'RESUME_STAMP={RESUME_STAMP!r} but {_rd} does not exist on Drive — '
        'check the stamp string (copy it exactly; no spaces). A silent '
        'fallback here would re-fly a finished stage.')
    print('resume dir found:', sorted(p.name for p in _rd.iterdir()) or 'EMPTY')
print('MODE:', MODE.upper(), '| stamp', STAMP,
      ('| RESUMING ' + RESUME_STAMP) if RESUME_STAMP else '')
'''

CELL_PAYLOAD_TEMPLATE = r'''# ── Pinned payload + stimulus mint + G1 pin gate (no model, pure arithmetic) ──
import numpy as np

PAYLOAD = json.loads(r"""@@PAYLOAD_JSON@@""")
_pl_sha = hashlib.sha256(json.dumps(
    {k: PAYLOAD[k] for k in sorted(PAYLOAD)}, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()[:16]
assert _pl_sha == PAYLOAD_SHA, f'payload sha drift: {_pl_sha} != {PAYLOAD_SHA}'

W5 = np.array(PAYLOAD['W'], float)
assert W5.shape == (15, 1536), W5.shape
assert abs(PAYLOAD['mu14'] - MU14_ATLAS) < 1e-9, 'atlas mu pin drift'

# locked atlas: byte-sha gate, then true dirs for train/eval names
_atlas_bytes = open(ATLAS_FP, 'rb').read()
_a_sha = hashlib.sha256(_atlas_bytes).hexdigest()
assert _a_sha == ATLAS_FILE_SHA, (
    f'atlas_dirs.json sha mismatch on Drive: {_a_sha[:16]}... — the locked '
    'atlas is the pinned source of true dirs; do not proceed on a drifted file')
ATLAS = json.loads(_atlas_bytes)
assert ATLAS['anchors'] == sorted(set(TRAIN256) | set(EVAL64)), 'anchor split drift'
assert hashlib.sha256('|'.join(ATLAS['anchors']).encode()).hexdigest()[:16] \
    == ANCHORS_SHA_F, 'anchors sha drift'
TRUE_DIRS = {}
for n in TRAIN256 + EVAL64:
    v = np.asarray(ATLAS['inst14'][n], float)
    TRUE_DIRS[n] = v / np.linalg.norm(v)
for n, pv in PAYLOAD['true_probe'].items():
    assert float(np.max(np.abs(TRUE_DIRS[n] - np.asarray(pv, float)))) < 1e-4, \
        f'true-dir probe drift at {n}'
DHAT = {w: np.asarray(v, float) for w, v in PAYLOAD['dhat'].items()}

STIM = mint_stimuli(W5, VEC, TRUE_DIRS, DHAT)
for k, v in STIM.items():
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-9, f'unnormalized stim {k}'
assert len([k for k in STIM if k.startswith('atom:')]) == 28
assert len([k for k in STIM if k.startswith('pairc:')]) == 84
assert len([k for k in STIM if k.startswith('full:')]) == 320
assert len([k for k in STIM if k.startswith('wperm:')]) == 16
assert len([k for k in STIM if k.startswith('true:')]) == 320
assert len([k for k in STIM if k.startswith('dhat:')]) == 13

# frame codes: derive from THIS pack, assert sha == build pin
FR_CODES = frame_codes(pack['concepts'])
assert frame_codes_sha(FR_CODES) == FRAME_CODES_SHA, 'frame codes sha drift (pack?)'
CODES64 = {n: FR_CODES[n] for n in EVAL64}
CODES_TRAIN = {n: FR_CODES[n] for n in TRAIN256}
assert set(EVAL64).isdisjoint(TRAIN256)
assert not (set(EVAL64) | set(TRAIN256)) & set(CHOICE_SET), 'wing leaked into split'
assert code_levels(MU_FRAME) == [0] * 14, 'mu_frame is not all-MID'
print(f'G1 PIN GATE: payload {PAYLOAD_SHA} ok | atlas sha ok | '
      f'{len(STIM)} stimuli minted + probed | frame codes {FRAME_CODES_SHA} ok')
'''

E8F_TRAIN_SRC = r'''
def encode_featex(ex):
    """(ids, prompt_len, answer_ids, skey) under the featural grammar."""
    msgs = [{'role': 'user', 'content': feat_prompt()}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors='pt').input_ids[0]
    ans = tok(ex['target'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    ans = torch.tensor(ans, dtype=pid.dtype)
    ids = torch.cat([pid, ans]).unsqueeze(0)
    return ids, len(pid), ans.long().unsqueeze(0), \
        (ex['skey'] if ex['kind'] == 'inject' else None)

def build_eval_model(readout_dir):
    """RESUME path: base + merged E4 real + the TRAINED readout adapter."""
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map=DEV, low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(m, ADAPTERS['real']).merge_and_unload()
    m = PeftModel.from_pretrained(m, str(readout_dir))
    m.eval()
    return m

def train_readout_feat(m, layer_mods, mu14, examples):
    """Six-strand SFT through ONE answer-sliced forward per example
    (full-sequence logits never materialized — E8-N OOM law; checkpointing
    engaged non-reentrantly and ASSERTED). For injected examples the
    backward runs INSIDE the Injector context (E8-N v2 law). Convergence =
    PER-STRAND plateau (E8-N v2 minted lesson, prospective); cap EPOCHS_CAP."""
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
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    try:
        scaler = torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, micro, max_tok, hook_calls = [], 0, 0, 0
    strand_epoch_means = []
    plateaued, epochs_flown = False, 0
    t0 = time.time()

    def fwd_loss(ids, plen, ans):
        lo, hi = answer_slice(plen, ids.shape[1])
        hid = DEC(input_ids=ids, use_cache=False).last_hidden_state
        logits = HEAD(hid[:, lo:hi, :]).float()
        return Fnn.cross_entropy(logits.view(-1, logits.size(-1)), ans.view(-1))

    for ep in range(EPOCHS_CAP):
        order = np.random.default_rng(E8F_SEED + 100 + ep).permutation(len(examples))
        ep_strand = {s: [] for s in STRANDS_F}
        for i in order:
            ex = examples[int(i)]
            ids, plen, ans, skey = encode_featex(ex)
            max_tok = max(max_tok, ids.shape[1])
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
            assert math.isfinite(lv), (
                f'non-finite loss at ep{ep} strand {ex.get("strand")}')
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
              + ' | '.join(f'{s} {strand_epoch_means[-1][s]}' for s in STRANDS_F))
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
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    k = max(3, len(losses) // 10)
    log = {'n_examples': len(examples), 'epochs_flown': epochs_flown,
           'epochs_cap': EPOCHS_CAP, 'plateaued': plateaued,
           'plateau_rel': PLATEAU_REL, 'min_epochs_strand': MIN_EPOCHS_STRAND,
           'strand_epoch_means': strand_epoch_means,
           'micro_steps': micro, 'opt_steps': micro // ACCUM,
           'hook_calls': int(hook_calls),
           'trainable_params': int(n_tr), 'max_example_tokens': int(max_tok),
           'peak_vram_gb': peak, 'secs': round(time.time() - t0, 1),
           'loss_first_k': round(float(np.mean(losses[:k])), 4),
           'final_smoothed': round(float(np.mean(losses[-50:])), 4),
           'losses_every_10': losses[::10]}
    print(f'  trained: {log["opt_steps"]} steps over {epochs_flown} epochs, '
          f'loss {log["loss_first_k"]} -> {log["final_smoothed"]}'
          f'{" (PLATEAU)" if plateaued else " (CAP HIT)"}'
          f', peak VRAM {peak}GB, {log["secs"]}s')
    return log
'''

CELL_RUN_SRC = r'''
# ── G2 identity probe + featural generation runner ───────────────────────────
def g2_probe(m):
    """Wing-13 dirs recomputed in the own-13-call shape (batch-composition
    lane law) vs the shipped E8-R real bundle at L14; mu asserted vs shipped
    AND the atlas pin. Runs on the PRE-readout model (zero-init LoRA)."""
    dirs_re = compute_dirs(m, [14], CHOICE_SET)
    stab = dirs_stability(dirs_re, {'14': SHIPPED_E8R['dirs']['14']},
                          tol=DIRS_TOL_F)
    mu = compute_mu(m, [14])
    mu_ship = float(SHIPPED_E8R['mu']['14'])
    assert abs(mu[14] / mu_ship - 1.0) < 1e-3, (mu[14], mu_ship)
    assert abs(mu[14] / MU14_ATLAS - 1.0) < 1e-3, (mu[14], MU14_ATLAS)
    print(f'  G2: wing-13 resid {stab["resid"]} (tol {DIRS_TOL_F}) | '
          f'mu {mu[14]:.3f} vs shipped {mu_ship:.3f} / atlas {MU14_ATLAS}')
    return stab, mu[14]

def run_trial_feat(m, layer_mods, mu14, trial):
    """Generation under the featural grammar; greedy; 110 new tokens."""
    enc = tok.apply_chat_template([{'role': 'user', 'content': feat_prompt()}],
                                  add_generation_prompt=True, return_tensors='pt',
                                  return_dict=True)
    ids = enc['input_ids'].to(DEV)
    gen_kw = dict(max_new_tokens=110, do_sample=False,
                  pad_token_id=tok.pad_token_id, use_cache=True)
    with torch.no_grad():
        if trial['kind'] == 'inject':
            vec = trial['alpha'] * mu14 * torch.tensor(STIM[trial['skey']])
            with Injector(layer_mods, trial['layer'], vec, ids.shape[1] - 1) as inj:
                out = m.generate(input_ids=ids, **gen_kw)
            calls = inj.calls
            assert calls >= 1, 'injection hook never fired during generation'
        else:
            out = m.generate(input_ids=ids, **gen_kw)
            calls = 0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return {**trial, 'response': text.strip()[:400],
            'parsed': parse_feat(text), 'hook_calls': calls}
'''

CELL_FLIGHT = r'''# ── Flight: G2 -> curriculum -> train -> ship readout -> eval -> ship bundle ──
def run_eval(m, layer_mods, mu14):
    rows = build_e8f_eval(SMOKE)
    out, t0 = {}, time.time()
    for i, t in enumerate(rows, 1):
        r = run_trial_feat(m, layer_mods, mu14, t)
        out.setdefault(t['block'], []).append(r)
        if i % 25 == 0 or i == len(rows):
            el = time.time() - t0
            eta = el / i * (len(rows) - i)
            print(f'    eval {i}/{len(rows)}, {el:.0f}s elapsed, ~{eta:.0f}s left')
    return out

def fly_real():
    t0 = time.time()
    bundle = {'condition': 'real', 'mode': MODE, 'stamp': STAMP,
              'payload_sha': PAYLOAD_SHA, 'atlas_sha': ATLAS_FILE_SHA[:16],
              'frame_codes_sha': FRAME_CODES_SHA}
    try:
        prev_readout = SEM / INFLIGHT / 'readout_real'
        resume_eval = bool(RESUME_STAMP) and \
            (prev_readout / 'adapter_config.json').exists()
        if resume_eval:
            print('RESUME: trained readout found on Drive — rebuilding eval model')
            probe_m = build_condition_model('real')
            stab, mu14 = g2_probe(probe_m)
            del probe_m; free_ram()
            local_ro = Path('/content/readout_real_resume')
            shutil.copytree(prev_readout, local_ro, dirs_exist_ok=True)
            model = build_eval_model(local_ro)
            layer_mods = resolve_layers(model)
            bundle['g2_dirs'], bundle['mu14'] = stab, mu14
            bundle['resumed_eval'] = True
            bundle['train_log'] = {'resumed': True}
            bundle['ppl_pre'] = None
        else:
            model = build_condition_model('real')
            layer_mods = resolve_layers(model)
            print(f'model ready — {ram_report()}')
            stab, mu14 = g2_probe(model)
            bundle['g2_dirs'], bundle['mu14'] = stab, mu14
            assert stab['pass'], f'G2 dirs-stability FAILED: {stab}'
            examples = build_e8f_train(VEC, SMOKE)
            _v1 = validate_no_eval_leak(examples)
            assert not _v1, f'FIREWALL — eval concept in curriculum: {_v1}'
            _v2 = validate_no_wing_leak(examples)
            assert not _v2, f'FIREWALL — wing stimulus in curriculum: {_v2}'
            cur = {}
            for e in examples:
                cur[e['strand']] = cur.get(e['strand'], 0) + 1
            bundle['curriculum'] = cur
            if not SMOKE:
                assert cur == EXPECT_TRAIN_FULL, cur
            print(f'  curriculum: {cur}')
            bundle['ppl_pre'] = retention_ppl(model)
            torch.cuda.empty_cache()
            bundle['train_log'] = train_readout_feat(model, layer_mods, mu14,
                                                     examples)
            torch.cuda.empty_cache()
            model.save_pretrained(str(OUT / 'readout_real'))
            ship(OUT / 'readout_real', f'{INFLIGHT}/readout_real')
            print('  trained readout shipped inflight (eval-resume point)')
        rows = run_eval(model, layer_mods, mu14)
        bundle['eval_rows'] = rows
        cnt = eval_counts([r for rs in rows.values() for r in rs])
        print(f'  eval rows: {cnt}')
        if not SMOKE:
            assert cnt == EXPECT_EVAL_FULL, cnt
        bundle['ppl_post'] = retention_ppl(model)
        bundle['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        bundle['error'] = f'{type(e).__name__}: {e}'
    return bundle

BUNDLE = None
if RESUME_STAMP:
    prev = SEM / INFLIGHT / 'bundle_real.json'
    if prev.exists():
        b = json.load(open(prev))
        if b.get('eval_rows') and 'error' not in b:
            BUNDLE = b
            print(f'RESUMED complete bundle from Drive ({RESUME_STAMP})')
if BUNDLE is None:
    print(f'flight: starting — {ram_report()}')
    BUNDLE = fly_real()
    fn = OUT / 'bundle_real.json'
    jdump(BUNDLE, fn)
    ship(fn, INFLIGHT)
    print(f'flight done in {BUNDLE.get("secs", "?")}s — bundle shipped')
    free_ram()
    print(f'torn down — {ram_report()}')
'''

CELL_VERDICT = r'''# ── Gates, primaries P1/P2/P3, S1-S8, fork, banner, ship ─────────────────────
ok = bool(BUNDLE.get('eval_rows')) and 'error' not in BUNDLE
if not ok:
    print()
    print('=' * 72)
    print(f'  FLIGHT INCOMPLETE — error: {str(BUNDLE.get("error"))[:200]}')
    print(f'  Fix comes home to the builder (lane law). To resume eval after '
          f'a mid-eval death: Restart runtime, RESUME_STAMP = '
          f"'{RESUME_STAMP or STAMP}', Run all.")
    print('=' * 72)
else:
    verdict = {'flight': f'{MODE}_{STAMP}', 'protocol': 'E8F_PROTOCOL.md',
               'payload_sha': PAYLOAD_SHA, 'resume_stamp': RESUME_STAMP or STAMP}
    blocks = BUNDLE['eval_rows']
    allrows = [r for rs in blocks.values() for r in rs]
    NP_PRIM = 200 if SMOKE else N_PERM_F
    NP_TEX = 50 if SMOKE else N_PERM_TEX

    # ── gates ────────────────────────────────────────────────────────────────
    g2 = BUNDLE['g2_dirs']
    ppl_pre, ppl_post = BUNDLE.get('ppl_pre'), BUNDLE.get('ppl_post')
    ppl_d = (100.0 * (ppl_post / ppl_pre - 1.0)) if (ppl_pre and ppl_post) \
        else None
    inv = invalid_rate(allrows)
    fa = sham_claims(blocks.get('sham', []))
    gates = {'g1_pins': {'payload': PAYLOAD_SHA, 'asserted_in_flight': True,
                         'pass': True},
             'g2_dirs': g2,
             'g3_ppl': {'pre': ppl_pre, 'post': ppl_post,
                        'delta_pct': (round(ppl_d, 4) if ppl_d is not None
                                      else None), 'tol': PPL_TOL_F,
                        'pass': bool(ppl_d is None or abs(ppl_d) <= PPL_TOL_F),
                        'note': ('resumed eval: ppl_pre unavailable, post-only'
                                 if ppl_pre is None else '')},
             'g4_parse': {**inv, 'max': PARSE_INVALID_MAX,
                          'pass': bool(inv['rate'] <= PARSE_INVALID_MAX)},
             'g5_sham': {'claims': fa, 'n': len(blocks.get('sham', [])),
                         'max': SHAM_FA_MAX, 'pass': bool(fa <= SHAM_FA_MAX)}}
    gates_ok = all(g.get('pass') for g in gates.values())
    verdict['gates'] = gates

    # ── G-INSTALL (spot) ─────────────────────────────────────────────────────
    spot_codes = {r['concept']: FR_CODES[r['concept']]
                  for r in blocks.get('spot', [])}
    spot = p_stats(blocks['spot'], spot_codes, AXIS_NAMES_F, NP_PRIM,
                   E8F_SEED + 22)
    install = {'pooled_acc': spot['pooled_acc'], 'p': spot['p'],
               'min_acc': INSTALL_MIN_ACC,
               'pass': bool(spot['pooled_acc'] >= INSTALL_MIN_ACC
                            and spot['p'] <= P_CRIT_F)}
    verdict['g_install'] = install
    verdict['spot'] = spot

    # ── primaries ────────────────────────────────────────────────────────────
    p1 = p_stats(blocks['p1'], CODES64, AXIS_NAMES_F, NP_PRIM, E8F_SEED + 20)
    p1_pass = bool(p1['p'] <= P_CRIT_F and p1['n_axes_sig'] >= P1_MIN_AXES)
    p3 = p_stats(blocks['p3'], CODES64, CARRIED8, NP_PRIM, E8F_SEED + 21)
    p3_pass = bool(p3['p'] <= P_CRIT_F)
    p2 = p2_stats(blocks['p1'], FR_CODES, NP_PRIM, E8F_SEED + 23)
    p2_pass = bool(p2['exact_rows'] >= P2_MIN_ROWS
                   and p2['n_distinct'] >= P2_MIN_DISTINCT
                   and p2['p'] <= P_CRIT_F)
    verdict['P1'] = {**p1, 'pass': p1_pass}
    verdict['P3'] = {**p3, 'pass': p3_pass}
    verdict['P2'] = {'exact_rows': p2['exact_rows'], 'n_rows': p2['n_rows'],
                     'n_distinct': p2['n_distinct'], 'distinct': p2['distinct'],
                     'p': p2['p'], 'pass': p2_pass}

    # ── S-blocks ─────────────────────────────────────────────────────────────
    verdict['S1'] = {'profile': s1_profile(blocks),
                     'p1_cond_speech': cond_speech_acc(blocks['p1'], CODES64,
                                                       AXIS_NAMES_F),
                     'p3_cond_speech': cond_speech_acc(blocks['p3'], CODES64,
                                                       CARRIED8)}
    verdict['S2'] = s2_stats(blocks['wperm'], VEC, NP_TEX)
    verdict['S3'] = s3_confusion(blocks['p1'], CODES64)
    verdict['S4'] = s4_titr(blocks['titr'],
                            {r['concept']: FR_CODES[r['concept']]
                             for r in blocks['titr']})
    verdict['S5'] = s5_wing(blocks['wing'])
    verdict['S6'] = s6_margin_split(p2, blocks['p1'])
    verdict['S7'] = s7_carrier(blocks['carrier'])
    verdict['S8'] = s8_density(blocks['p1'], CODES64)
    verdict['train_log'] = {k: v for k, v in BUNDLE['train_log'].items()
                            if k != 'losses_every_10'}

    # ── fork ─────────────────────────────────────────────────────────────────
    if SMOKE:
        fork = 'SMOKE — mechanics only, no verdict'
    elif not gates_ok:
        fork = 'GATES DIRTY — primaries withheld (NO_VERDICT, lane law)'
    elif not install['pass']:
        fork = ('NO_VERDICT — G-INSTALL failed (installation, not '
                'generalization); re-fly after curriculum/budget revision')
    elif p1_pass and p3_pass:
        fork = ('FF2 — the code reads TRUE substrate geometry on the '
                'carried-8: first semantics-bearing featural read')
    elif p1_pass and not p3_pass:
        fork = ('FF1/FF3 — L3 lands in bottleneck form; true-dir arm did not '
                'clear (interface-without-geography, separability family)')
    elif not p1_pass:
        fork = ('FF4 — composition wall inside the span (installed but no '
                'held-out generalization); S8 + per-axis table localize')
    verdict['fork'] = fork

    print()
    print('=' * 72)
    if SMOKE:
        n_par = sum(1 for r in allrows if r['parsed']['kind'] != 'INVALID')
        green = ok and n_par >= max(1, int(0.5 * len(allrows)))
        print(f'  SMOKE {"GREEN" if green else "RED"} — flight mechanics '
              f'{"exercised" if green else "FAILED"} '
              f'({n_par}/{len(allrows)} rows parsed; gates mechanical: '
              f'g2 {g2["pass"]}, parse {gates["g4_parse"]["pass"]})')
        if green:
            print('  Next: Runtime > Restart runtime, set SMOKE = False, Run all.')
    else:
        print(f'  E8-F VERDICT — gates {"PASS" if gates_ok else "FAIL"} | '
              f'G-INSTALL {"PASS" if install["pass"] else "FAIL"} '
              f'(spot acc {spot["pooled_acc"]} p {spot["p"]})')
        if gates_ok and install['pass']:
            print(f'  P1 spelling: pooled {p1["pooled_acc"]} vs null '
                  f'{p1["null_mean"]} p={p1["p"]} | axes sig '
                  f'{p1["n_axes_sig"]}/14 -> {"PASS" if p1_pass else "FAIL"}')
            print(f'  P3 true-dir (carried-8): pooled {p3["pooled_acc"]} vs '
                  f'null {p3["null_mean"]} p={p3["p"]} -> '
                  f'{"PASS" if p3_pass else "FAIL"}')
            print(f'  P2 decode-to-name: {p2["exact_rows"]}/{p2["n_rows"]} rows, '
                  f'{p2["n_distinct"]} distinct, p={p2["p"]} -> '
                  f'{"PASS" if p2_pass else "FAIL"}')
        print(f'  fork: {fork}')
    print('=' * 72)
    vf = OUT / 'e8f_verdict.json'
    jdump(verdict, vf)
    ship(vf, f'e8f/{MODE}_{STAMP}')
    fn = OUT / 'bundle_real.json'
    if not fn.exists():
        jdump(BUNDLE, fn)
    ship(fn, f'e8f/{MODE}_{STAMP}')
    print('shipped: e8f/' + f'{MODE}_{STAMP}')
'''


def build_traincfg_cell():
    src = BR.CELL_TRAINCFG
    part1 = slice_block(src, "from transformers import", "LR = 1e-4",
                        tag="traincfg-1")
    part2 = slice_block(src, "tok = AutoTokenizer", None, tag="traincfg-2")
    consts = ("LR = 1e-4\nACCUM = 4 if SMOKE else 8\n"
              "EPOCHS_CAP = 1 if SMOKE else EPOCH_CAP_F\n"
              "MIN_EPOCHS_STRAND = 1 if SMOKE else 2\n"
              "PLATEAU_REL = 0.05\n\n")
    return ("# ── Readout LoRA config + model builders + train loop (flown slices + E8-F) ──\n"
            + part1 + consts + part2 + E8F_TRAIN_SRC)


def build_stimmach_cell():
    return ("# ── G2 stimulus machinery (E8-J slices, verbatim) + Injector + runner ──────\n"
            + "import numpy as np\n"
            + STIM_MACH_SRC + "\n" + INJECTOR_SRC + CELL_RUN_SRC)


def main():
    payload, subs, ctx = mint()
    e8f_src = E8F_SRC_TEMPLATE
    for k, v in subs.items():
        assert k in e8f_src, f"missing sentinel {k}"
        e8f_src = e8f_src.replace(k, v)
    assert "@@" not in e8f_src, "unsubstituted sentinel remains"

    logic = E8R_LOGIC_SRC + "\n\n" + DIRS_STAB_SRC + "\n\n" \
        + ANSWER_SLICE_SRC + "\n\n" + e8f_src

    with open(os.path.join(HERE, "e8f_logic.py"), "w") as f:
        f.write(logic)
    payload_path = os.path.join(HERE, "e8f_payload.json")
    with open(payload_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    # import the emitted module fresh and prove the mint round-trips
    for mod in ("e8f_logic",):
        if mod in sys.modules:
            del sys.modules[mod]
    L = importlib.import_module("e8f_logic")
    assert L.TRAIN256 == ctx["train"] and L.EVAL64 == ctx["eval"]
    assert L.frame_codes_sha(L.frame_codes(
        json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))["concepts"]
    )) == L.FRAME_CODES_SHA
    tr = L.build_e8f_train(ctx["vec"], smoke=False)
    cnt = {}
    for e in tr:
        cnt[e["strand"]] = cnt.get(e["strand"], 0) + 1
    assert cnt == L.EXPECT_TRAIN_FULL, cnt
    ev = L.build_e8f_eval(smoke=False)
    assert L.eval_counts(ev) == L.EXPECT_EVAL_FULL, L.eval_counts(ev)
    assert not L.validate_no_eval_leak(tr)
    assert not L.validate_no_wing_leak(tr)

    payload_json = json.dumps(payload, separators=(",", ":"))
    assert '"""' not in payload_json and "\\" not in payload_json, \
        "payload json unsafe for raw triple-quote embedding"
    cell_payload = CELL_PAYLOAD_TEMPLATE.replace("@@PAYLOAD_JSON@@",
                                                 payload_json)

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
    add("code", logic)
    add("code", cell_payload)
    add("code", build_traincfg_cell())
    add("code", build_stimmach_cell())
    add("code", CELL_FLIGHT)
    add("code", CELL_VERDICT)

    # per-cell compile + no-rclone scan (skip markdown)
    import ast
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        low = src.lower()
        assert "rclone" not in low, f"cell {i}: rclone reference"
        assert "curl" not in low.replace("curl_", ""), f"cell {i}: curl"
        if c["cell_type"] == "code":
            try:
                ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"cell {i} does not compile: {e}")

    out_path = os.path.join(HERE, "E8F_FEATURAL_UI.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    sz = os.path.getsize(out_path)
    print(f"built {out_path} ({sz/1e3:.0f}KB, {len(nb['cells'])} cells)")
    print(f"payload: {payload_path} ({os.path.getsize(payload_path)/1e3:.0f}KB)"
          f" sha {subs['@@PAYLOAD_SHA@@']}")
    print(f"e8f_logic.py emitted; frame codes sha {subs['@@FRAME_CODES_SHA@@']}")
    dst = os.path.expanduser("~/Desktop/E8F_FEATURAL_UI.ipynb")
    import shutil
    shutil.copy2(out_path, dst)
    print(f"staged {dst}")


if __name__ == "__main__":
    main()
