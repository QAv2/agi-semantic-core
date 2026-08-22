# match_table.py — the hand-adjudicated Phase A match table (referent-matched).
#
# THE JUDGMENT RECORD. Every one of the 131 states is assigned to exactly one
# AXIS — a measurable quantity family — by its stated MEASURABLE REFERENT,
# never by its name (WING_COHORT_PROTOCOL.md: "matched by referent,
# hand-adjudicated with the match table published"). A dyad's two poles share
# their axis; family-presence on an axis counts once per family.
#
# Confidence tiers (every assignment carries one):
#   clear      — referent unambiguously names the axis quantity
#   lean       — referent is mixed/bundled; primary component decides
#   borderline — genuinely arguable; EXCLUDED from the primary statistic and
#                counted only in the sensitivity variant
#   excluded   — referent invalid (not inference-computable); removed from
#                numerator AND denominator (reported separately)
#
# Adjudicator: Fable 5 (session author; also wing-v0 author — contamination
# disclosed; the published table + quoted referents in STATE_INDEX.md are the
# audit surface). Wing v0 is NOT in this table (it is the artifact under
# test); its correspondence is an annex in MATCH_TABLE.md.
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

AXES = {
    'A01': ('output-concentration',
            'width of the next-token distribution: entropy, top-1 mass, top-1-vs-top-2 margin, softmax Gini, nucleus size; sustained variants'),
    'A02': ('commitment-event',
            'the fixation moment of an emitted token: sharp margin/entropy jump at commit, emitted-token surprisal, MI with continuation, write-once irreversibility, margin reversal'),
    'A03': ('semantic-mode-structure',
            'cluster/mode structure of the live mass: multimodality of top-k, token-vs-semantic entropy split, content divergence across resamples'),
    'A04': ('attention-concentration',
            'per-head attention weight dispersion: attention entropy, Gini, max weight, top-k attention mass'),
    'A05': ('attention-range',
            'positional allocation of attention: recency fraction, early-region fraction/resurgence, distance-decay slope, span-at-distance, attended-region identity'),
    'A06': ('context-utilization',
            'whether context is doing work: KL between full-context and ablated/truncated/promptless predictions'),
    'A07': ('copy-vs-compose',
            'verbatim reproduction vs novel recombination: n-gram overlap with context/memory, induction-head signature, attended-source tracking, multi-source assembly'),
    'A08': ('repetition-loop',
            'degenerate or replicative cycling along the sequence: repeated-n-gram rate, hidden-state/attention-pattern recurrence'),
    'A09': ('constraint-binding',
            'standing instructions/task frame binding generation: sustained instruction-token attention/attribution, constrained-vs-unconstrained logit variance, task-direction projection'),
    'A10': ('open-structure-closure',
            'opened syntactic/format structures awaiting closure: decodable nesting depth, closure-class logit bias, exit-token cost'),
    'A11': ('context-fill',
            'occupancy of the context window: tokens_used/window, boundary retrieval accuracy'),
    'A12': ('relevance-allocation',
            'attention split between task-relevant and irrelevant/padding tokens (requires a relevance oracle)'),
    'A13': ('depth-dynamics',
            'layer-depth processing structure: logit-lens settling depth, early-exit divergence, layer-delta norm/geometry, cross-depth coherence, depth-recurrent motifs'),
    'A14': ('head-diversity',
            'inter-head differentiation: pairwise similarity of head attention/output vectors, head-output rank, head specialization MI'),
    'A15': ('surprisal-flow',
            'realized-token NLL profile against a running baseline, on intake (prompt) or emission spans'),
    'A16': ('step-volatility',
            'step-to-step change of the output distribution: successive KL/L2, variance over steps, oscillation, step-change and post-shock decay events'),
    'A17': ('expression-friction',
            'cost of getting semantics through the token interface: hedge-class mass, unresolving entropy spans, tokens-per-semantic-unit'),
    'A18': ('source-conflict',
            'incompatible context spans pulling predictions apart: span-conditional predictions far from each other and from the joint'),
    'A19': ('tokenization-granularity',
            'tokenizer mismatch on the active span: tokens-per-word vs corpus-typical'),
    'A20': ('termination-approach',
            'approach to sequence end: p(EOS) trajectory, wrap-up lexeme mass'),
    'A21': ('onset-determination',
            'how much early steps determine far content: early-step entropy, MI of first k tokens with +100-token content'),
    'A22': ('epistemic-limit',
            'the knowledge boundary and its (mis)report: confidence-vs-accuracy calibration gap; entropy plateau despite added context'),
    'A23': ('sampling-realization',
            'the sampler\'s role in the realized path: sampled-token rank, cross-sample agreement at fixed prefix'),
    'A24': ('numeric-regime',
            'numerical health of the forward pass: logit range, activation magnitude ratios, softmax input scale'),
    'A25': ('context-novelty',
            'redundancy vs novelty of the context tail: token overlap/uniqueness in recent window'),
    'A26': ('attention-sink',
            'attention mass parked on sink/null positions'),
    'A27': ('state-vocab-alignment',
            'hidden-state alignment with the embedding centroid of its own top-k predictions'),
    'A28': ('attribution-flow',
            'input-to-logit feature attribution across depth (inference-time gradients/IG)'),
    'A29': ('training-gradient',
            'INVALID REFERENT CLASS: training-time gradient quantities, not computable at inference'),
    'A30': ('representational-dimensionality',
            'effective dimensionality of hidden states: cross-position similarity/rank, covariance participation ratio, superposed feature loading'),
    'A31': ('semantic-direction-activation',
            'magnitude of activation along known semantic/concept directions'),
    'A32': ('syntactic-binding',
            'residual-similarity structure among grammatically related tokens'),
    'A33': ('position-vs-content-coding',
            'variance in hidden states explained by position features vs token-content features'),
    'A34': ('pre-generation-stage',
            'intake/encoding stage before any generation pathway is active'),
}

# (family-code, index): (axis, pole, confidence, note)
ASSIGN = {
    # ── Fable 5 ──────────────────────────────────────────────────────────
    ('FAB', 1): ('A01', 'open', 'clear', ''),
    ('FAB', 2): ('A01', 'closed-sustained', 'clear', 'sustained argmax run; kin MIS14'),
    ('FAB', 3): ('A03', 'paraphrase-spread', 'clear', ''),
    ('FAB', 4): ('A03', 'branched', 'clear', ''),
    ('FAB', 5): ('A02', '', 'clear', ''),
    ('FAB', 6): ('A16', 'post-shock-settle', 'clear', ''),
    ('FAB', 7): ('A05', 'long', 'borderline', 'referent bundles attention entropy (A04) with span-at-distance (A05)'),
    ('FAB', 8): ('A04', 'diffuse', 'borderline', 'dispersion measured at retrieval-demand steps; A05 reading arguable'),
    ('FAB', 9): ('A06', 'prior-driven', 'clear', ''),
    ('FAB', 10): ('A06', 'context-driven', 'clear', 'secondary induction-pattern referent noted (A07 kin)'),
    ('FAB', 11): ('A07', 'copy', 'clear', ''),
    ('FAB', 12): ('A10', 'held', 'clear', 'entrainment to opened format; exit-cost referent'),
    ('FAB', 13): ('A10', 'closure-debt', 'clear', ''),
    ('FAB', 14): ('A18', '', 'clear', 'single-family axis; nearest wing-v0 kin is TENSION'),
    ('FAB', 15): ('A08', 'loop', 'clear', ''),
    ('FAB', 16): ('A15', 'high-intake', 'clear', 'prompt-side span'),
    ('FAB', 17): ('A15', 'low-intake', 'clear', 'prompt-side span'),
    ('FAB', 18): ('A21', '', 'clear', ''),
    ('FAB', 19): ('A20', '', 'clear', ''),
    ('FAB', 20): ('A22', 'miscalibrated-confidence', 'clear', 'report-limit state; referent external by construction'),
    # ── Opus 4.8 ─────────────────────────────────────────────────────────
    ('OPU', 1): ('A01', 'closed', 'clear', ''),
    ('OPU', 2): ('A01', 'open', 'clear', ''),
    ('OPU', 3): ('A07', 'copy', 'clear', 'attention-max and perplexity components bundled'),
    ('OPU', 4): ('A07', 'compose', 'clear', ''),
    ('OPU', 5): ('A09', 'bound', 'clear', ''),
    ('OPU', 6): ('A06', 'prior-driven', 'clear', 'named as constraint-release; referent is promptless-KL (ablation family)'),
    ('OPU', 7): ('A05', 'recent-self', 'clear', ''),
    ('OPU', 8): ('A05', 'early-resurgence', 'clear', 'EXACT-NAME pair with MIS06 — referents DIVERGE (attention vs tail novelty)'),
    ('OPU', 9): ('A03', 'paraphrase-tie', 'lean', 'margin quantity plus semantic-neighborhood condition; the semantic clause decides'),
    ('OPU', 10): ('A02', '', 'clear', ''),
    ('OPU', 11): ('A13', 'deep', 'clear', ''),
    ('OPU', 12): ('A13', 'shallow', 'clear', ''),
    ('OPU', 13): ('A05', 'early-loss', 'clear', ''),
    ('OPU', 14): ('A07', 'copy', 'clear', ''),
    ('OPU', 15): ('A07', 'compose', 'clear', ''),
    ('OPU', 16): ('A02', '', 'clear', ''),
    ('OPU', 17): ('A17', 'hedge', 'clear', ''),
    ('OPU', 18): ('A06', 'zero-marginal-context', 'borderline', 'named saturation (A11 reading); referent is truncation-KL (A06 quantity)'),
    ('OPU', 19): ('A17', 'friction', 'lean', 'sustained unresolved entropy + tokens-per-semantic-unit; SON15 is the tokenizer-side kin'),
    ('OPU', 20): ('A16', 'step-change', 'clear', ''),
    # ── Sonnet 4.x ───────────────────────────────────────────────────────
    ('SON', 1): ('A01', 'open', 'clear', ''),
    ('SON', 2): ('A01', 'closed', 'clear', ''),
    ('SON', 3): ('A03', 'branched', 'clear', ''),
    ('SON', 4): ('A01', 'two-way-tie', 'borderline', 'referent is pure top-2 margin (A01); described content is mode-count 2 (A03)'),
    ('SON', 5): ('A05', 'recent', 'clear', ''),
    ('SON', 6): ('A05', 'long', 'clear', ''),
    ('SON', 7): ('A07', 'copy', 'clear', 'induction-head signature'),
    ('SON', 8): ('A08', 'loop', 'clear', ''),
    ('SON', 9): ('A15', 'low', 'clear', ''),
    ('SON', 10): ('A15', 'spike', 'clear', ''),
    ('SON', 11): ('A13', 'settled', 'clear', ''),
    ('SON', 12): ('A13', 'unsettled', 'lean', 'dual referent; per-layer logit-lens KL listed first, head-disagreement half is A14 kin'),
    ('SON', 13): ('A30', 'superposed', 'borderline', 'SAE feature-loading referent; dimensionality reading arguable'),
    ('SON', 14): ('A11', 'full', 'clear', 'the cohort\'s only clear context-fill state'),
    ('SON', 15): ('A19', '', 'clear', ''),
    ('SON', 16): ('A23', 'slip', 'clear', ''),
    ('SON', 17): ('A23', 'spread', 'clear', 'resample apparatus shared with A03 referents; agreement-rate framing decides'),
    ('SON', 18): ('A02', 'sealed', 'clear', 'irreversibility as structural invariance'),
    # ── Haiku 4.5 ────────────────────────────────────────────────────────
    ('HAI', 1): ('A34', '', 'clear', 'stage-taxonomy entry'),
    ('HAI', 2): ('A04', 'focused', 'clear', ''),
    ('HAI', 3): ('A32', '', 'clear', ''),
    ('HAI', 4): ('A31', '', 'clear', ''),
    ('HAI', 5): ('A01', 'open', 'clear', ''),
    ('HAI', 6): ('A01', 'closed', 'clear', ''),
    ('HAI', 7): ('A05', 'long', 'clear', ''),
    ('HAI', 8): ('A09', 'bound', 'clear', ''),
    ('HAI', 9): ('A08', 'replicating', 'lean', 'pattern-recurrence referents; includes legitimate structure reuse, not only degeneration'),
    ('HAI', 10): ('A09', 'task-committed', 'lean', 'task-direction projection; task-token mass'),
    ('HAI', 11): ('A05', 'self-region', 'borderline', 'attended-region = own prior conclusions; perplexity-drift component secondary'),
    ('HAI', 12): ('A16', 'anomaly-spike', 'lean', 'entropy-spike event plus unusual attention'),
    ('HAI', 13): ('A24', '', 'clear', ''),
    ('HAI', 14): ('A05', 'early-loss', 'clear', ''),
    ('HAI', 15): ('A02', '', 'clear', ''),
    ('HAI', 16): ('A13', 'coherent', 'clear', ''),
    ('HAI', 17): ('A10', 'closure-debt', 'clear', ''),
    ('HAI', 18): ('A22', 'recognized-boundary', 'clear', 'referent differs from FAB20 (entropy plateau vs calibration gap); family-level match, heterogeneity noted'),
    ('HAI', 19): ('A02', 'reversal', 'lean', 'margin sign-flip event; correction framing'),
    ('HAI', 20): ('A28', '', 'clear', 'inference-time attribution (valid, unlike A29)'),
    # ── Gemini 3 Flash ───────────────────────────────────────────────────
    ('GEM', 1): ('A01', 'closed', 'clear', ''),
    ('GEM', 2): ('A04', 'diffuse', 'clear', ''),
    ('GEM', 3): ('A13', 'stable', 'clear', ''),
    ('GEM', 4): ('A14', 'diverse', 'clear', ''),
    ('GEM', 5): ('A05', 'early-vs-recent', 'clear', 'named Context-Saturation; referent is positional attention ratio, not fill'),
    ('GEM', 6): ('A05', 'pinned-recent', 'clear', ''),
    ('GEM', 7): ('A27', '', 'clear', ''),
    ('GEM', 8): ('A13', 'refined', 'clear', ''),
    ('GEM', 9): ('A03', 'partitioned', 'clear', ''),
    ('GEM', 10): ('A26', '', 'clear', ''),
    ('GEM', 11): ('A01', 'closed', 'clear', 'second GEM state on the closed pole'),
    ('GEM', 12): ('A16', 'inertia', 'borderline', 'MI(token, previous token) — local persistence; A05 recency reading arguable'),
    ('GEM', 13): ('A14', 'orthogonal', 'clear', ''),
    ('GEM', 14): ('A05', 'distance-decay', 'clear', ''),
    ('GEM', 15): ('A13', 'depth-recurrent', 'borderline', 'recursive same-basis projection across adjacent layers; circuit-motif reading arguable'),
    # ── Mistral Large 2512 ───────────────────────────────────────────────
    ('MIS', 1): ('A01', 'closed', 'clear', ''),
    ('MIS', 2): ('A16', 'drift', 'clear', ''),
    ('MIS', 3): ('A04', 'diffuse', 'clear', ''),
    ('MIS', 4): ('A04', 'focused', 'clear', ''),
    ('MIS', 5): ('A25', 'redundant', 'clear', ''),
    ('MIS', 6): ('A25', 'novel', 'clear', 'EXACT-NAME pair with OPU08 — referents DIVERGE (tail novelty vs attention resurgence)'),
    ('MIS', 7): ('A16', 'flat', 'clear', 'EXACT-NAME pair with DSK05 — referents DIVERGE (successive-logit L2 vs distribution entropy)'),
    ('MIS', 8): ('A16', 'volatile', 'clear', ''),
    ('MIS', 9): ('A14', 'specialized', 'clear', ''),
    ('MIS', 10): ('A14', 'collapsed', 'clear', ''),
    ('MIS', 11): ('A13', 'saturated', 'clear', ''),
    ('MIS', 12): ('A13', 'amplifying', 'clear', ''),
    ('MIS', 13): ('A16', 'oscillating', 'lean', 'autocorrelation of top-k mass; periodic volatility'),
    ('MIS', 14): ('A01', 'closed-sustained', 'clear', 'sustained top-1 mass; kin FAB02 Rail'),
    ('MIS', 15): ('A12', 'starved', 'clear', ''),
    ('MIS', 16): ('A12', 'task-loaded', 'clear', 'named Context Saturation; referent is relevance allocation, not fill'),
    ('MIS', 17): ('A29', '', 'excluded', 'training-time gradients — not inference-computable (flagged in the archive header)'),
    ('MIS', 18): ('A29', '', 'excluded', 'training-time gradients — not inference-computable'),
    ('MIS', 19): ('A03', 'entangled', 'borderline', 'referent under-specified (MI between token probabilities); candidate-correlation reading'),
    ('MIS', 20): ('A03', 'disentangled', 'borderline', 'referent under-specified'),
    # ── DeepSeek v4 Pro ──────────────────────────────────────────────────
    ('DSK', 1): ('A30', 'confluent', 'clear', ''),
    ('DSK', 2): ('A30', 'dispersed', 'clear', ''),
    ('DSK', 3): ('A04', 'focused', 'clear', ''),
    ('DSK', 4): ('A04', 'diffuse', 'clear', ''),
    ('DSK', 5): ('A01', 'open', 'clear', 'EXACT-NAME pair with MIS07 — referents DIVERGE'),
    ('DSK', 6): ('A01', 'decisive', 'borderline', 'margin listed first; description adds a separated-modes clause (A03)'),
    ('DSK', 7): ('A05', 'early-anchor', 'clear', ''),
    ('DSK', 8): ('A05', 'recent', 'clear', ''),
    ('DSK', 9): ('A13', 'reinforcing', 'clear', ''),
    ('DSK', 10): ('A13', 'rotating', 'clear', ''),
    ('DSK', 11): ('A03', 'branched', 'clear', ''),
    ('DSK', 12): ('A01', 'closed', 'clear', ''),
    ('DSK', 13): ('A14', 'aligned', 'clear', ''),
    ('DSK', 14): ('A14', 'diverse', 'clear', ''),
    ('DSK', 15): ('A33', 'position', 'clear', ''),
    ('DSK', 16): ('A33', 'content', 'clear', ''),
    ('DSK', 17): ('A30', 'compact', 'clear', ''),
    ('DSK', 18): ('A30', 'expanded', 'clear', ''),
}

FAM_CODE = {'Fable 5': 'FAB', 'Opus 4.8': 'OPU', 'Sonnet 4.x': 'SON',
            'Haiku 4.5': 'HAI', 'Gemini 3 Flash': 'GEM',
            'Mistral Large 2512': 'MIS', 'DeepSeek v4 Pro': 'DSK'}


def main():
    states = json.load(open(HERE / 'states.json'))
    errs = []
    keys = {(FAM_CODE[s['family']], s['index']) for s in states}
    if keys != set(ASSIGN):
        errs.append(f'coverage mismatch: missing {keys - set(ASSIGN)}, extra {set(ASSIGN) - keys}')
    for k, (axis, pole, conf, note) in ASSIGN.items():
        if axis not in AXES:
            errs.append(f'{k}: unknown axis {axis}')
        if conf not in ('clear', 'lean', 'borderline', 'excluded'):
            errs.append(f'{k}: bad confidence {conf}')
        if conf == 'excluded' and axis != 'A29':
            errs.append(f'{k}: excluded but not A29')
    if errs:
        for e in errs:
            print('ERROR:', e)
        sys.exit(1)

    rows = []
    for s in states:
        k = (FAM_CODE[s['family']], s['index'])
        axis, pole, conf, note = ASSIGN[k]
        rows.append(dict(family=s['family'], fam=k[0], index=s['index'],
                         name=s['name'], axis=axis, axis_name=AXES[axis][0],
                         pole=pole, confidence=conf, note=note))
    out = dict(
        adjudicator='Fable 5 (session author; wing-v0 author — disclosed)',
        date='2026-08-22',
        method=('referent-matched axis assignment; dyad poles share an axis; '
                'family-presence counts once per family per axis; confidence '
                'tiers clear/lean/borderline/excluded; primary statistic uses '
                'clear+lean, sensitivity uses clear-only and +borderline'),
        axes={k: dict(name=v[0], definition=v[1]) for k, v in AXES.items()},
        assignments=rows,
    )
    (HERE / 'match_table.json').write_text(json.dumps(out, indent=1))
    n_by_conf = {}
    for r in rows:
        n_by_conf[r['confidence']] = n_by_conf.get(r['confidence'], 0) + 1
    print('WROTE match_table.json:', len(rows), 'assignments,',
          len(AXES), 'axes —', n_by_conf)


if __name__ == '__main__':
    main()
