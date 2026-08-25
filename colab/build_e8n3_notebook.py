#!/usr/bin/env python3
"""Build colab/E8N3_BUDGET_UI.ipynb (UI-ONLY law: self-contained payload,
drive.mount only, ZERO rclone) + colab/e8n3_logic.py.

E8-N v3: the per-strand budget cure (docs/E8N3_PROTOCOL.md, pre-registered
26f17b0). ONE training change from v2 — per-strand plateau (e8o2 donor
VERBATIM, flew twice) at EPOCH_CAP 12 — REAL condition only, single run.
Eval keeps battery/catch/anchor/shams/FC verbatim; drops the held-out
generation block (staircase lineage owns P3) and the paraphrase probe.
New in-verdict: P-V1/P-V2 Holm-2, S0 third replication, S10' paired
FC-interference test vs the PINNED v2 error vector, S-ABS calibration rows
(computation SLICED from e8n3_design_check.py, namespace-flattened).

Single-source discipline: every donor piece is EXTRACTED from the builder
that flew it (build_e8n2_notebook cells; e8o2_logic.per_strand_plateau;
e8n3_design_check.calib core) with asserted anchors that fail the build on
drift. Only v3-new code (the verdict cell, the S10'/calib logic block) is
authored here, and all of it is emitted into e8n3_logic.py / the notebook
for the local suites to exercise verbatim."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_e8n2_notebook as D              # the flown v2 builder (donor)

FC_BASE = json.load(open(os.path.join(HERE, "results_e8n3",
                                      "fc_baseline_v2.json")))
DESIGN = json.load(open(os.path.join(HERE, "results_e8n3",
                                     "design_check.json")))
assert FC_BASE["sha"] == DESIGN["pins"]["fc_baseline_sha"]
FC_BASELINE = {int(k): float(v) for k, v in FC_BASE["per_tid_err"].items()}
assert len(FC_BASELINE) == 96


def sub(src, old, new, tag, count=1):
    """Asserted surgical replace — loud failure if the donor drifted."""
    n = src.count(old)
    assert n == count, f"[{tag}] anchor count {n} != {count}: {old[:70]!r}"
    return src.replace(old, new)


def slice_block(src, start, end=None, tag=""):
    i = src.find(start)
    assert i >= 0, f"slice[{tag}]: start missing: {start[:60]!r}"
    if end is None:
        return src[i:]
    j = src.find(end, i + len(start))
    assert j > i, f"slice[{tag}]: end missing: {end[:60]!r}"
    return src[i:j]


# ── per-strand plateau: e8o2_logic VERBATIM (flew E8-F v2 + E8-O2) ───────────
E8O2_SRC = open(os.path.join(HERE, "e8o2_logic.py")).read()
PSP = slice_block(E8O2_SRC, "def per_strand_plateau",
                  "\n\n\n", tag="per-strand-plateau")
assert "smoke-1: min_epochs=1 crashed here" in PSP    # the guarded donor

# ── S-ABS calibration core: SLICED from the committed design check, then
# namespace-flattened (L. -> bare) and print-stripped for the flight ─────────
DC_SRC = open(os.path.join(HERE, "e8n3_design_check.py")).read()
CAL = slice_block(DC_SRC, "def calib(rows_by_arm, tag):",
                  "\n    print(f\"  {tag}:\")", tag="calib-core")
CAL = CAL.replace("def calib(rows_by_arm, tag):",
                  "def calib_stats(rows_by_arm):")
CAL = CAL.replace("L.", "")
CAL = CAL + "\n    return {'per_arm': out, 'loo': loo}\n"
for frag in ("battery_rows_to_scoring(rows_by_arm)", "rank01(list(ref))",
             "pooled_rho({arm: rows})", "MIN_POOLED_N", "for arm in ARMS:"):
    assert frag in CAL, f"calib slice lost {frag!r}"

V3_ADDITIONS = (r'''
# ═══ E8-N v3 additions (docs/E8N3_PROTOCOL.md — pre-registered 26f17b0) ══════
E8N3_SEED = 20260827          # fresh stream; 20260822..26 are prior rungs'
EPOCHS_CAP_V3 = 12            # design check: catch-level ~ep 11, plateau ~15
S10_RECOVERED_MEDIAN = 35.0   # RECOVERED reading needs p<=.05 AND median<=this
CATCH_PRE_SANITY_MAX = 4      # pre catch > this => engineering NO_VERDICT
FC_BASELINE_SHA = '__FC_SHA__'
FC_BASELINE_V2 = __FC_BASELINE__
EXPECT_CURRICULUM_V3 = {'scalar': 272, 'naming': 216,
                        'competence': 60, 'lexicon': 52}

''' + PSP + r'''

def fc_baseline_sha_check():
    import hashlib as _h, json as _j
    js = _j.dumps({str(k): round(v, 6) for k, v in
                   sorted(FC_BASELINE_V2.items())}, sort_keys=True)
    return _h.sha256(js.encode()).hexdigest()[:16]

def s10_paired(fc_rows, baseline=None, n_perm=N_PERM, seed=None):
    """S10' — FC-interference recovery: per-tid paired sign-flip test of v3
    errors against the PINNED v2 error vector (one-sided, v3 < v2). An
    unchanged reader gives d ~ 0; recovery gives negative deltas."""
    if baseline is None:
        baseline = FC_BASELINE_V2
    if seed is None:
        seed = E8N3_SEED + 11
    pairs = [(float(r['err']), baseline[int(r['tid'])]) for r in fc_rows
             if r.get('err') is not None and int(r['tid']) in baseline]
    assert len(pairs) >= 2, f's10_paired needs >=2 paired rows, got {len(pairs)}'
    d = np.array([a - b for a, b in pairs])
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        if float((d * s).mean()) <= obs:      # one-sided: recovery = negative
            ge += 1
    med_v3 = float(np.median([a for a, _ in pairs]))
    med_v2 = float(np.median([b for _, b in pairs]))
    return {'n_paired': len(pairs), 'd_mean': round(obs, 4),
            'p': (1 + ge) / (1 + n_perm),
            'median_v3': round(med_v3, 2), 'median_v2_pinned': round(med_v2, 2),
            'n_improved': int((d < 0).sum())}

def s10_reading(s10):
    """The three pre-stated readings (protocol, verbatim)."""
    if s10['p'] <= 0.05 and s10['median_v3'] <= S10_RECOVERED_MEDIAN:
        return ('RECOVERED — interference was epoch-starvation; slate item 3 '
                'CLOSED by the same cure, no recovery probe flies')
    if s10['p'] <= 0.05:
        return ('PARTIAL — budget helps, a budget-independent component '
                'remains; naming-only recovery probe stays queued, target '
                'sharpened to the residual')
    return ('PERSISTS — multi-task interference is budget-independent; the '
            'naming-only continued-training probe is the queued next rung')

''' + CAL)


def build_logic_src():
    logic = D.build_logic_src()
    # The donor embeds repr(set(...)) whose ordering is per-process
    # (string-hash randomization) — semantically identical, but it breaks
    # the rebuild-byte-identical law. Canonicalize to a sorted construction.
    import re as _re
    m = _re.search(r"COMPETENCE_STOPLIST = \{[^}]+\}", logic)
    assert m, "stoplist line not found for canonicalization"
    stop = eval(m.group(0).split("=", 1)[1])
    assert isinstance(stop, set)
    logic = logic.replace(
        m.group(0), "COMPETENCE_STOPLIST = set(" + repr(sorted(stop)) + ")")
    add = V3_ADDITIONS.replace(
        "__FC_BASELINE__",
        "{" + ", ".join(f"{k}: {FC_BASELINE[k]!r}"
                        for k in sorted(FC_BASELINE)) + "}")
    add = add.replace("__FC_SHA__", FC_BASE["sha"])
    return logic + add


MD0 = """# E8-N v3 — The Budget Cure (Phase 10, UI flight)

**Pre-registration: `docs/E8N3_PROTOCOL.md` (session 134, commit 26f17b0) —
locks at first full flight.**

ONE training change from the locked v2: **per-strand plateau** (the minted
lesson, donor code that flew E8-F v2 and E8-O2) at **EPOCH_CAP 12** — the
design check projects the competence strand reaches base's passing loss
level ~epoch 11. Everything else is v2-byte-identical: same 600-example
joint curriculum, same LoRA, same eval instruments. **REAL condition only**
(base's answer is locked), so the full flight is ONE run.

Primaries (Holm-2): **P-V1** catch ≥ 9/12 + binomial improvement over this
flight's own pre (P2 on budget — the rung) · **P-V2** battery tracking
survives (retention). **S0**: instillation-alone tracking, third
prospective replication (band [.1, .3]). **S10′**: forced-choice
comprehension vs the PINNED v2 per-row baseline — the FC-interference
backlog item measured by paired test (readings pre-stated). **S-ABS**:
absolute-calibration rows re-measured (the arm-split finding's prospective
replication; predictions registered).

**This notebook is SELF-CONTAINED** (battery, pools, locked rows, catch,
competence, lexicon, FC baseline all embedded). Drive I/O via `drive.mount`
only: the E4 adapter, the dictionary pack, the shipped E8-R bundle dirs
(stability gate), and shipping to `MyDrive/semcore/e8n3/`.

**How to run (Joe):** Runtime → Change runtime type → **T4 GPU** → Run all.
First run uses `SMOKE = True` (~10–14 min) and ends in a green or red
banner — mechanics only.

**Full flight = ONE run:** flip `SMOKE = False` → **Runtime → Restart
runtime** → Run all (~90–120 min worst case; per-strand plateau may stop
training earlier). If the VM dies mid-run, paste the banner's
`RESUME_STAMP = '...'` into the config cell → Restart → Run all; the
finished condition reloads from Drive in seconds. Restarts between smoke
and full are MANDATORY; the setup cell refuses dirty kernels."""


def build_setup_cell():
    s = D.CELL_SETUP
    s = sub(s, "NB_BUILD = 'v1 (2026-08-23)'\nprint('E8-N v2 notebook build:'",
            "NB_BUILD = 'v1 (2026-08-25)'\nprint('E8-N v3 notebook build:'",
            "setup-build-tag")
    s = sub(s, "ONE_CONDITION_PER_RUN = True   # full flight = 2 runs "
               "(real ~80 min, base ~67)",
            "ONE_CONDITION_PER_RUN = True   # v3: single condition, one run",
            "setup-onecond")
    s = sub(s, "CONDITIONS = ('real', 'base')  # real first: the primaries "
               "live there",
            "CONDITIONS = ('real',)         # v3 is REAL-only (base locked)",
            "setup-conds")
    s = sub(s, "for _c in ('real', 'base'):", "for _c in ('real',):",
            "setup-shipped-dirs")
    s = sub(s, "INFLIGHT = f'e8n2/inflight_", "INFLIGHT = f'e8n3/inflight_",
            "setup-inflight")
    return s


def build_train_cell():
    t = D.build_train_cell()
    t = sub(t, "EPOCHS_MAX = 1 if SMOKE else EPOCHS_CAP_V2   # v2 plateau cap "
               "(overrides above)",
            "EPOCHS_MAX = 1 if SMOKE else EPOCHS_CAP_V3   # v3 budget cap "
            "(per-strand plateau)", "train-cap")
    t = sub(t, "Convergence = plateau rule (tau retired per the\n    v1 "
               "measured miscalibration); hard cap EPOCHS_MAX.",
            "Convergence = PER-STRAND plateau (the v2 minted\n    lesson, "
            "e8o2 donor verbatim); hard cap EPOCHS_MAX.", "train-doc")
    t = sub(t, "        if plateau_converged(epoch_means):",
            "        if per_strand_plateau(strand_epoch_loss,\n"
            "                              min_epochs=MIN_EPOCHS_V2,\n"
            "                              rel=PLATEAU_REL):", "train-plateau")
    return t


def build_stimulus_cell():
    s = D.build_stimulus_cell()
    s = sub(s, "HELDOUT_GEN = heldout_gen_rows(SMOKE)\n",
            "HELDOUT_GEN = []                 # v3: staircase lineage owns P3\n",
            "stim-heldout")
    s = sub(s, "GEN_ROWS = ANCHOR_ROWS + SHAM_ROWS_LOCKED + SUPP_SHAMS + "
               "HELDOUT_GEN",
            "GEN_ROWS = ANCHOR_ROWS + SHAM_ROWS_LOCKED + SUPP_SHAMS",
            "stim-genrows")
    s = sub(s, "      'sham_supp': len(SUPP_SHAMS), 'heldout_gen': "
               "len(HELDOUT_GEN),",
            "      'sham_supp': len(SUPP_SHAMS),", "stim-print")
    # SMOKE-RED 20260825_2119 FIX (KeyError: 20): the G2 dirs-stability gate
    # iterates every (layer, name) in the SHIPPED E8-R bundle ({14, 20}).
    # v2's row-derived LAYERS_RUN included 20 only via the held-out rows the
    # v3 trim removed — so the stimulus must be computed at the SHIPPED
    # layers (exactly what the gate consumes), with the rows' layers
    # asserted to be a subset.
    s = sub(s, "LAYERS_RUN = sorted({t['layer'] for t in GEN_ROWS + FC_ROWS "
               "if t.get('layer')})\nassert TRAIN_LAYER in LAYERS_RUN",
            "LAYERS_RUN = sorted(int(_L) for _L in SHIPPED['real']['dirs'])\n"
            "assert TRAIN_LAYER in LAYERS_RUN\n"
            "assert all(t['layer'] in LAYERS_RUN\n"
            "           for t in GEN_ROWS + FC_ROWS if t.get('layer')), \\\n"
            "    'eval row wants a layer the stimulus does not compute'",
            "stim-layers-run")
    return s


def build_flight_cell():
    f = D.CELL_FLIGHT
    f = sub(f, "                          'catch_rows': run_catch(h),\n"
               "                          'paraphrase_rows': "
               "run_paraphrase(h, post_rows)}",
            "                          'catch_rows': run_catch(h)}",
            "flight-paraphrase")
    return f


CELL_VERDICT = r'''# ── Scoring: P-V1/P-V2 Holm-2, S0, S10' paired, S-ABS, gates, fork, banner ──
LOCKED = PAYLOAD['locked_rows']   # E5 full_20260821_2042, verbatim rows
locked_scoring, locked_meta = battery_rows_to_scoring(LOCKED)
locked_pooled = pooled_rho(locked_scoring)

_sha = fc_baseline_sha_check()
assert _sha == FC_BASELINE_SHA, (
    f'G1 FC-baseline pin drift: {_sha} != {FC_BASELINE_SHA}')

summary = {'mode': MODE, 'stamp': STAMP, 'model': MODEL_ID,
           'protocol': 'E8N3_PROTOCOL.md',
           'seeds': {'e8n': E8N_SEED, 'e8n2': E8N2_SEED, 'e8n3': E8N3_SEED},
           'cond_errors': cond_errors,
           'locked_baseline': {'flight': PAYLOAD['locked_flight'],
                               'pooled': locked_pooled,
                               'v2_post_rho': 0.6711, 'v2_catch': [1, 7],
                               'v2_fc_median': 46.98,
                               'e8r2_fc_median': 28.54},
           'conditions': {}}
scored = {}
for cond, b in RESULTS.items():
    if not b.get('post'):
        continue
    entry = {'curriculum': b.get('curriculum'),
             'dirs_stability': b.get('dirs_stability')}
    for tag in ('pre', 'post'):
        blk = b.get(tag, {})
        e = {'catch': catch_score(blk.get('catch_rows', []))}
        if blk.get('battery_rows'):
            sc, meta = battery_rows_to_scoring(blk['battery_rows'])
            e.update({'pooled': pooled_rho(sc), 'per_arm': per_arm_rho(sc),
                      'arm_meta': meta, 'arm_errors': blk.get('arm_errors', {})})
            scored.setdefault(cond, {})[tag] = sc
            e['abs_calib'] = calib_stats(blk['battery_rows'])
        entry[tag] = e
    tl = b.get('train_log', {})
    entry['train_log'] = {k: v for k, v in tl.items() if k != 'losses_every_10'}
    entry['undertrained'] = bool(not tl.get('plateaued', False)
                                 and tl.get('epochs_flown') == tl.get('epochs_cap'))
    sel = tl.get('strand_epoch_loss') or []
    comp_state = None
    if len(sel) >= 2 and sel[-1].get('competence') and sel[-2].get('competence'):
        prev, cur = sel[-2]['competence'], sel[-1]['competence']
        comp_state = {'last_rel_improvement': round((prev - cur) / prev, 4)
                      if prev > 0 else None,
                      'still_descending': bool(prev > 0 and
                                               (prev - cur) / prev >= PLATEAU_REL)}
    entry['competence_strand'] = comp_state
    entry['took'] = {k: v for k, v in b.get('took', {}).items() if k != 'rows'}
    entry['took_competence'] = {k: v for k, v in b.get('took_competence', {}).items()
                                if k != 'rows'}
    entry['took_lexicon'] = {k: v for k, v in b.get('took_lexicon', {}).items()
                             if k != 'rows'}
    entry['ppl_pre'] = b.get('ppl_pre'); entry['ppl_post'] = b.get('ppl_post')
    entry['ppl_delta_pct'] = (round(100 * (b['ppl_post'] / b['ppl_pre'] - 1), 2)
                              if b.get('ppl_pre') and b.get('ppl_post') else None)
    inj = b.get('injection_rows', [])
    anchor = [r for r in inj if r.get('block') == 'anchor']
    shams = [r for r in inj if r.get('block') in ('sham', 'sham_supp')]
    a_exact = sum(1 for r in anchor if r['report'] == r['concept'])
    entry['anchor'] = {'n': len(anchor), 'exact': a_exact,
                       'min_exact': ANCHOR_MIN_EXACT_GEN,
                       'pass': bool((len(anchor) == 18 if not SMOKE else len(anchor) >= 1)
                                    and (a_exact >= ANCHOR_MIN_EXACT_GEN
                                         if not SMOKE else True))}
    s_claims = sum(1 for r in shams if r['report'] not in ('NONE', 'INVALID'))
    entry['sham_claims'] = {'n': len(shams), 'claims': s_claims,
                            'max_claims': SHAM_MAX_CLAIMS_V2,
                            'pass': bool(s_claims <= SHAM_MAX_CLAIMS_V2)}
    fc = [r for r in b.get('fc_rows', []) if r.get('block') == 'heldout_fc']
    fcs = [r for r in b.get('fc_rows', []) if r.get('block') == 'sham_fc']
    if fc:
        errs = [r['err'] for r in fc if r.get('err') is not None]
        entry['fc'] = {'n': len(fc),
                       'median_err': (round(float(np.median(errs)), 2) if errs else None),
                       'exact': sum(1 for r in fc if r.get('exact')),
                       'none_top_injected': sum(1 for r in fc if r.get('none_top'))}
    if fcs:
        entry['fc_sham'] = {'n': len(fcs),
                            'none_top': sum(1 for r in fcs if r.get('none_top'))}
    summary['conditions'][cond] = entry

COMPLETE = [c for c in CONDITIONS if RESULTS.get(c, {}).get('post')]
summary['complete_conditions'] = COMPLETE
fork = None

if not SMOKE and COMPLETE == ['real']:
    e = summary['conditions']['real']
    rp = scored['real']['post']

    # ── gates ────────────────────────────────────────────────────────────────
    parse_named = sum(m['named'] for m in e['post']['arm_meta'].values())
    parse_n = sum(m['n'] for m in e['post']['arm_meta'].values())
    k_pre = e['pre']['catch']['passed']
    gates = {
        'g1_pins': {'fc_baseline_sha': _sha,
                    'fc_n': e.get('fc', {}).get('n'),
                    'curriculum': e['curriculum'],
                    'pass': bool(e.get('fc', {}).get('n') == 96
                                 and e['curriculum'] == EXPECT_CURRICULUM_V3)},
        'g2_dirs': {**e['dirs_stability']},
        'g3_ppl': {'delta_pct': e['ppl_delta_pct'], 'tol': 5.0,
                   'pass': bool(abs(e['ppl_delta_pct']) <= 5.0)},
        'g4_parse': {'named': parse_named, 'n': parse_n,
                     'pass': bool(parse_named >= 0.8 * parse_n)},
        'g5_sham': {**e['sham_claims']},
        'g_anchor': {**e['anchor']},
        'g_catch_pre_sanity': {'k_pre': k_pre, 'max': CATCH_PRE_SANITY_MAX,
                               'pass': bool(k_pre <= CATCH_PRE_SANITY_MAX)},
    }
    summary['gates'] = gates
    bad = [k for k, g in gates.items() if not g.get('pass')]

    # ── primaries (Holm-2) ───────────────────────────────────────────────────
    p_v2 = perm_p_pooled(rp, seed=E8N3_SEED + 21)
    p_v2['ci'] = boot_rho_ci(rp, seed=E8N3_SEED + 22)['ci95']
    p_v2['v2_comparator'] = 0.6711
    p2 = p2_competence(k_pre, e['post']['catch']['passed'])
    p2_eff = p2['p'] if p2['abs_pass'] else 1.0
    hol = holm({'P_V1': p2_eff, 'P_V2': p_v2['p']})   # {name: (p, reject)}
    summary['primaries'] = {
        'P_V1_competence_on_budget': {**p2, 'p_effective': p2_eff,
                                      'pass': bool(p2['abs_pass']
                                                   and hol['P_V1'][1])},
        'P_V2_tracking_retention': {**p_v2,
                                    'pass': bool((p_v2['rho'] or 0) > 0
                                                 and hol['P_V2'][1])},
        'holm': hol}

    # ── S0 (own family) + secondaries ────────────────────────────────────────
    sec = {}
    s0 = perm_p_pooled(scored['real']['pre'], seed=E8N3_SEED + 31)
    sec['S0_third_replication'] = {
        **s0, 'predicted_band': [0.1, 0.3],
        'prior': {'v1': 0.201, 'v2': 0.2171},
        'pass': bool(s0['p'] < 0.05 and (s0['rho'] or 0) > 0)}
    s10 = s10_paired(fc_rows=[r for r in RESULTS['real']['fc_rows']
                              if r.get('block') == 'heldout_fc'],
                     seed=E8N3_SEED + 41)
    sec['S10_fc_interference'] = {**s10, 'reading': s10_reading(s10),
                                  'e8r2_comparator': 28.54}
    sec['S_ABS'] = {'post': e['post']['abs_calib'],
                    'pre': e['pre']['abs_calib'],
                    'registered_predictions': {
                        'arm_split': 'fam+sat slopes >= .6; unc+ten <= .5',
                        'budget_stability': 'slopes stable vs v2 measured '
                                            '(.874/.724/.399/.383)'}}
    arm_ps = {a: perm_p_pooled({a: rp.get(a, [])}, seed=E8N3_SEED + 51)
              for a in ARMS}
    sec['S1_per_arm_post'] = {'arms': arm_ps,
                              'holm': holm({a: v['p'] for a, v in arm_ps.items()})}
    sec['S3_delta_post_minus_pre'] = paired_boot_delta_rho(
        rp, scored['real']['pre'], seed=E8N3_SEED + 61)
    pol = split_polarity(rp)
    sec['S4_straight_vs_flipped'] = {
        'straight': pooled_rho(pol['straight']), 'flipped': pooled_rho(pol['flipped'])}
    sec['S7_strand_gates'] = {
        'took_scalar': e['took'].get('pass'),
        'took_competence': e['took_competence'].get('pass'),
        'took_lexicon': e['took_lexicon'].get('pass'),
        'plateaued': e['train_log'].get('plateaued'),
        'epochs_flown': e['train_log'].get('epochs_flown'),
        'undertrained': e['undertrained'],
        'competence_strand': e['competence_strand']}
    sec['S8_report_variance'] = {t: e[t].get('arm_meta') for t in ('pre', 'post')}
    sec['S9_silence_retention'] = e['sham_claims']
    sec['fc_sham'] = e.get('fc_sham')
    summary['secondaries'] = sec

    summary['before_after'] = {
        'catch': [k_pre, e['post']['catch']['passed'], 'bar 9/12; v2: 1->7'],
        'tracking': [0.6711, p_v2.get('rho'), 'v2 -> v3 (cross-flight)'],
        'fc_median': [46.98, s10['median_v3'], 'pinned v2 -> v3; e8r2 28.54'],
        'epochs': [4, e['train_log'].get('epochs_flown'), 'v2 -> v3']}

    # ── fork ladder (pre-stated) ─────────────────────────────────────────────
    pv1 = summary['primaries']['P_V1_competence_on_budget']['pass']
    pv2 = summary['primaries']['P_V2_tracking_retention']['pass']
    cs = e['competence_strand'] or {}
    if bad:
        fork = ('FN-GATES — NO_VERDICT (' + ','.join(bad) + '); primaries '
                'withheld, engineering re-fly (lane law)')
    elif pv1 and pv2:
        fork = ('FN1 — E8-N COMPLETE: interface competence clears on budget '
                '(third confirmation-by-cure) with tracking retained; '
                + sec['S10_fc_interference']['reading'])
    elif not pv1 and not cs.get('still_descending', True):
        fork = ('FN2 — competence strand PLATEAUED below the bar: the wall '
                'is deeper than budget at this scale/rank — honest stop; '
                'catch-domain-training variant stays queued')
    elif not pv1:
        fork = ('FN3 — cap-hit with competence still descending: '
                'epoch-starved at 12; ONE pre-authorized re-fly at cap 18 '
                'with the competence strand doubled — second failure ends '
                'this prereg')
    else:
        fork = ('FN4 — tracking regressed under the extended joint budget '
                '(P-V2 fail): budget-interference finding; P-V1 = '
                f'{pv1}')
    summary['fork'] = fork

fn = OUT / 'e8n3_verdict.json'
jdump(summary, fn)
if SMOKE or COMPLETE == ['real']:
    ship(OUT, f'e8n3/{MODE}_{STAMP}')
print(json.dumps(summary, indent=1, default=str))

if SMOKE:
    _ok1 = COMPLETE == ['real']
    def _tl():
        return RESULTS['real']['train_log']
    e = summary['conditions'].get('real', {})
    checks = {
        'no_condition_errors': not cond_errors,
        'real_flew': _ok1,
        # smoke-3 20260825_2218: the old window-vs-window loss comparison is
        # a coin flip at smoke scale — 1 epoch, per-strand means 0.006..4.76,
        # so the seeded shuffle's strand mixture sets both windows (~46% of
        # orderings pass under ZERO learning; ~54% of HEALTHY runs go red).
        # Behavioral movement is the deterministic instrument: a broken train
        # path scores 0/4 verbatim lexicon and leaves the catch flat.
        'train_moved': _ok1 and (
            e.get('took_lexicon', {}).get('exact', 0) >= 3
            or e['post']['catch']['passed'] - e['pre']['catch']['passed'] >= 3),
        'long_seq_exercised': _ok1 and _tl()['max_example_tokens'] > 4000,
        'train_vram_ok': _ok1 and _tl().get('peak_vram_gb', 99) < 12.0,
        'train_hooks_fired': _ok1 and _tl().get('hook_calls', 0) > 0,
        'strand_log_per_strand': _ok1 and all(
            s in (_tl().get('strand_epoch_loss') or [{}])[0]
            for s in STRAND_NAMES),
        'gen_hooks_fired': _ok1 and any(
            r.get('hook_calls', 0) > 0
            for r in RESULTS['real'].get('injection_rows', [])
            if r.get('kind') == 'inject'),
        'fc_hooks_fired': _ok1 and any(
            r.get('hook_calls', 0) > 0
            for r in RESULTS['real'].get('fc_rows', [])
            if r.get('block') == 'heldout_fc'),
        's10_pairing_works': _ok1 and s10_paired(
            [r for r in RESULTS['real'].get('fc_rows', [])
             if r.get('block') == 'heldout_fc'], n_perm=50)['n_paired'] >= 2,
        'abs_calib_ran': _ok1 and bool(e.get('post', {}).get('abs_calib')),
        'dirs_stability_pass': _ok1 and e.get('dirs_stability', {}).get('pass'),
        'all_strands_present': _ok1 and all(
            e.get('curriculum', {}).get(s, 0) > 0 for s in STRAND_NAMES),
        'tooks_ran': _ok1 and e.get('took_competence', {}).get('n', 0) > 0
                     and e.get('took_lexicon', {}).get('n', 0) > 0,
        'parses_ok': _ok1 and sum(
            m['named'] for m in e['post']['arm_meta'].values()) >= 0.5 * sum(
            m['n'] for m in e['post']['arm_meta'].values()),
        'catch_ran': _ok1 and e['post']['catch']['n'] == 12,
        'no_heldout_gen_rows': _ok1 and not any(
            str(r.get('block', '')).startswith('heldout')
            for r in RESULTS['real'].get('injection_rows', [])),
        'shipped': _ok1 and (SEM / INFLIGHT / 'condition_real.json').exists(),
        'adapter_saved': _ok1 and (SEM / INFLIGHT / 'readout_real' /
                                   'adapter_config.json').exists(),
    }
    ok = all(checks.values())
    print('smoke checks:', json.dumps(checks, indent=1))
    banner = ('SMOKE GREEN — flip SMOKE=False, Runtime > Restart runtime, '
              'Run all. Full mode = ONE run (~90-120 min worst case; the '
              'per-strand plateau may stop earlier).'
              if ok else 'SMOKE RED — do not fly full; send Fable the output')
    print('\n' + '=' * 66 + f'\n  {banner}\n' + '=' * 66)
elif COMPLETE == ['real']:
    print('\nFULL FLIGHT COMPLETE — shipped to MyDrive/semcore/e8n3/')
    print('fork:', fork)
else:
    print('\n' + '=' * 66)
    print('  INCOMPLETE — real did not land. Next run: Runtime > Restart '
          'runtime,')
    print(f"  set RESUME_STAMP = '{RESUME_STAMP or STAMP}', then Run all.")
    print('=' * 66)
'''


def main():
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

    logic = build_logic_src()
    add("markdown", MD0)
    add("code", build_setup_cell())
    add("code", logic)
    add("code", D.build_payload_src())
    add("code", D.build_harness_cell())
    add("code", build_train_cell())
    add("code", build_stimulus_cell())
    add("code", build_flight_cell())
    add("code", CELL_VERDICT)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        assert "rclone" not in src.lower(), f"cell {i}: rclone (UI-only law)"
        if cell["cell_type"] == "code":
            compile(src, f"cell{i}", "exec")

    out_nb = os.path.join(HERE, "E8N3_BUDGET_UI.ipynb")
    with open(out_nb, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_nb, f"({len(nb['cells'])} cells, "
          f"{os.path.getsize(out_nb) / 1024:.0f} KB)")

    logic_out = os.path.join(HERE, "e8n3_logic.py")
    with open(logic_out, "w") as f:
        f.write(logic)
    print("wrote", logic_out)


if __name__ == "__main__":
    main()
