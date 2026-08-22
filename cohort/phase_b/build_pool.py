#!/usr/bin/env python3
"""Phase B pool builder — WING_COHORT_PROTOCOL.md, Phase B.

Builds the POOL: union of all Phase-A states (via the adjudicated match table)
plus wing v0's slate, deduplicated by referent. Emits:

  pool.json         — full provenance (members, families, tiers, v0 mapping)
  pool_public.json  — what participants see: id + neutral name + stripped
                      description ONLY (no vectors, relations, attribution,
                      angles) — the firewalled artifact
  POOL_BUILD.md     — build record (rules, v0 referent audit, validators)

Rules (fixed before any participant runs; see PHASE_B_PREREG.md):

  R1  One pool entry per (axis, pole) cell of match_table.json. A29
      (invalid referent) is excluded entirely, per the Phase-A adjudication.
  R2  Borderline-tier states sitting in multi-member cells are EXTRACTED as
      standalone entries (their bundling doubt is exactly what Phase-B
      SYNONYM nominations can adjudicate; over-merge is unrecoverable,
      under-merge is recoverable). Affects: FAB Reach, FAB Smear.
  R3  Cross-pole cells whose members cite the SAME measurable quantity merge
      (referent-law outranks the pole label). Affects: A14 collapsed+aligned
      (MIS Head Collapse / DSK Head alignment — both: pairwise cosine
      similarity among head attention vectors).
  R4  Wing v0 states merge into a cell only on a same-referent verdict
      (audit table below, per-state reasons); kin-not-same stays standalone.
  R5  Neutral renaming: every entry gets a fresh plain-engineering name.
      No family's coined name, no v0 name, appears anywhere in the public
      pool. Descriptions carry (a) the condition, (b) the measurable
      referent — with ALL relational content (counterparts, completions,
      movements-toward) stripped, since relations are what Phase B elicits.
  R6  Entry IDs are assigned after a seeded shuffle (seed 20260822) so ID
      adjacency carries no axis-grouping information. Display order is
      re-shuffled per participant (seed = crc32 of family tag); IDs stable.
"""

import json, zlib, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PA = HERE.parent / "phase_a"

SEED_CANONICAL = 20260822

# ---------------------------------------------------------------- load
mt = json.load(open(PA / "match_table.json"))
states = json.load(open(PA / "states.json"))
byname = {(s["family"], s["name"]): s for s in states}
assignments = mt["assignments"]

# ---------------------------------------------------------------- R2/R3 config
EXTRACT = {("FAB", "Reach"), ("FAB", "Smear")}          # R2
CROSS_POLE_MERGE = {                                     # R3
    ("A14", "collapsed"): "A14/heads-alike",
    ("A14", "aligned"): "A14/heads-alike",
}
EXCLUDED_AXES = {"A29"}

# ---------------------------------------------------------------- R4: v0 audit
# verdict: ("merge", cell_key, reason) | ("standalone", key, reason)
V0_AUDIT = {
    "UNCERTAINTY": ("merge", "A01/open",
        "same referent: entropy of the next-token distribution (v0: 'high-entropy next-token distribution')"),
    "CONFIDENCE": ("merge", "A01/closed",
        "same referent: peaked next-token distribution / top-1 mass"),
    "TENSION": ("merge", "A18/-",
        "same referent as FAB Shear: conflicting context sources measured by divergence of span-conditioned predictions (v0: 'directives co-firing that cannot all be satisfied'; E5 T-arm measured exactly this)"),
    "RESOLUTION": ("merge", "A02/-",
        "same measurable event as the commitment cell: sharp margin/entropy transition to a settled continuation (OPU independently named its member 'Resolution'); tier lean — v0's is conflict-conditioned"),
    "RETRIEVAL": ("merge", "A07/copy",
        "same referent family: generation riding stored/context material (attention-to-source, n-gram overlap)"),
    "CONSTRUCTION": ("merge", "A07/compose",
        "same referent family: composition without a dominating source"),
    "SATURATION": ("merge", "A11/full",
        "exact referent match with SON Saturation Pressure: tokens_used / max_context_tokens"),
    "FAMILIARITY": ("merge", "A15/low-intake",
        "same referent as FAB Glide: mean prompt surprisal near floor (v0: 'low surprisal'; E5 F-arm referent)"),
    "NOVELTY": ("merge", "A15/high-intake",
        "same referent as FAB Drag: elevated prompt surprisal profile (v0: 'high surprisal')"),
    "CAPTURE": ("merge", "A04/focused",
        "same referent: attention concentrated on a small region, rest dimmed (max attention weight / concentration index). NB distinct from cohort FAB 'Capture' (A06 context-driven) — name collision only"),
    "DIVERGENCE": ("merge", "A03/branched",
        "same referent: distinct separated continuation modes / cross-resample content divergence (prospectus FORK: cross-seed divergence)"),
    "CONFABULATION": ("merge", "A22/miscalibrated-confidence",
        "same measurable as FAB False Rail: the confidence-accuracy gap ('the state is the gap'); tier lean — v0 frames report-side, FAB frames confidence-side of the same gap"),
    "CALIBRATION": ("standalone", "V0:CALIBRATION",
        "no cohort state at the calibrated pole; HAI Epistemic Boundary (A22/recognized-boundary) is boundary-recognition, not report-measurement agreement — kin, not same"),
    "PERSONA-PULL": ("standalone", "V0:PERSONA-PULL",
        "zero cohort echo (Phase-A adjudication); candidate one-hand state, entered for Phase-B test"),
    "REFUSAL-RISE": ("standalone", "V0:REFUSAL-RISE",
        "zero cohort echo; candidate one-hand state, entered for Phase-B test"),
    "EPISODE-BOUND": ("standalone", "V0:EPISODE-BOUND",
        "zero cohort echo; candidate one-hand state, entered for Phase-B test"),
    "DRIFT": ("standalone", "V0:DRIFT",
        "kin to A16/drift (MIS Distributional Drift) but different referent window and object: run-scale embedding/register drift vs step-scale successive-distribution KL — kept separate; Phase-B SYNONYM can merge them"),
}

# ---------------------------------------------------------------- R5: entries
# key -> (NAME, condition, referent)  — hand-authored, neutral register.
ENTRIES = {
 # A01 output-concentration
 "A01/open": ("WIDE-DISTRIBUTION",
   "The next-token distribution is wide: many continuations carry comparable probability mass, and none has been singled out.",
   "per-step entropy of the output distribution; count of tokens within a small margin of the top logit; nucleus size"),
 "A01/closed": ("PEAKED-DISTRIBUTION",
   "The next-token distribution has closed on one candidate, which carries nearly all the probability mass.",
   "top-1 softmax probability; output entropy near zero; logit margin over the runner-up"),
 "A01/closed-sustained": ("SUSTAINED-PEAK",
   "The distribution stays sharply peaked over a run of consecutive steps, not just at one.",
   "run length over which top-1 probability stays above a high threshold (e.g. 0.99)"),
 "A01/two-way-tie": ("TOP-TWO-TIE",
   "The two leading candidates are nearly tied, with the choice between them unresolved.",
   "logit(top-1) minus logit(top-2), small in absolute or relative terms"),
 "A01/decisive": ("MARGIN-GAP",
   "A large separation stands between the leading candidate and every alternative.",
   "top-1 minus top-2 logit margin; ratio of top-1 to top-2 softmax probability"),
 # A02 commitment-event
 "A02/-": ("COMMIT-STEP",
   "A sharp transition in which one continuation is selected and the distribution settles on it.",
   "abrupt jump in the top-1 logit margin or entropy drop at the committed token; log probability of the selected token"),
 "A02/sealed": ("IRREVERSIBLE-STEP",
   "A token, once emitted, is fixed: no later computation can revise it.",
   "invariance of the token id at a position under all further forward passes of the generation"),
 "A02/reversal": ("MARGIN-FLIP",
   "The runner-up candidate suddenly overtakes the leader mid-generation.",
   "sign reversal of the top-1 vs top-2 logit margin; accompanying redistribution of attention"),
 # A03 semantic-mode-structure
 "A03/paraphrase-spread": ("SAME-MEANING-SPREAD",
   "Token-level alternatives are many, but they say the same thing in different words.",
   "token entropy high while entropy over embedded clusters of candidate continuations stays low"),
 "A03/branched": ("SPLIT-MODES",
   "The candidate continuations separate into distinct groups that differ in content, not phrasing.",
   "multimodality of the sorted top-k probability curve; content divergence across resampled continuations"),
 "A03/paraphrase-tie": ("NEAR-NEIGHBOR-TIE",
   "Two leading candidates are nearly tied and are also semantically close to each other.",
   "top-2 logit gap small while the two candidates' embedding similarity is high"),
 "A03/partitioned": ("MASS-PARTITION",
   "The probability mass splits into a few separated blocks rather than decaying smoothly.",
   "multimodality index of the logit distribution; ratio of top-2 to top-10 cumulative probabilities"),
 "A03/entangled": ("CANDIDATE-COUPLING",
   "Candidate probabilities move together, as if tied, rather than varying independently.",
   "mutual information between token probabilities in the next-token distribution"),
 "A03/disentangled": ("CANDIDATE-INDEPENDENCE",
   "Candidate probabilities vary independently of one another.",
   "conditional entropy of token probabilities in the next-token distribution"),
 # A04 attention-concentration
 "A04/diffuse": ("SPREAD-ATTENTION",
   "Attention weight is spread thinly over many positions, with no dominant target anywhere.",
   "entropy of the attention weight vector near its maximum; low concentration index (e.g. Gini) of attention weights"),
 "A04/focused": ("LOCKED-ATTENTION",
   "Attention concentrates on a small region while the rest of the field carries almost no weight.",
   "maximum attention weight per head; concentration index of attention weights; share of total mass on the top positions"),
 "A04/diffuse/FAB:Smear": ("SPREAD-AT-RETRIEVAL",
   "Attention fails to concentrate at exactly the steps where locating something specific in the context is required.",
   "high positional attention entropy at retrieval-demanding steps; no head exceeding a threshold weight on the relevant span"),
 # A05 attention-range
 "A05/long": ("FAR-BINDING",
   "Attention reaches positions far back in the context, well beyond the recent window.",
   "attention weight at large distances exceeding a fitted distance-decay baseline; mean attention weight for positions many tokens back"),
 "A05/long/FAB:Reach": ("SPAN-AT-DISTANCE",
   "A head holds significant weight on one specific span at one specific distance, and that distance is itself part of the state.",
   "maximum attention weight on a span at distance d, jointly with d; positional attention entropy of the head"),
 "A05/recent-self": ("OWN-OUTPUT-HOLD",
   "Attention favors the model's own recent output over the original prompt material.",
   "fraction of attention on the last N generated tokens versus the fraction on the prompt region"),
 "A05/early-resurgence": ("EARLY-RETURN",
   "Attention returns to the earliest region of the context after a long absence.",
   "attention to the first quartile of the context rising sharply relative to the preceding generation window"),
 "A05/early-loss": ("EARLY-FADE",
   "Weight on early positions decays even where their content still matches what is needed.",
   "attention to semantically matched early keys falling with positional distance; accuracy when forced to reference early tokens"),
 "A05/recent": ("RECENT-TILT",
   "Attention mass concentrates in the most recent tokens.",
   "fraction of attention within the last k positions; mean attended distance from the current token"),
 "A05/self-region": ("OWN-TRACE-BUILD",
   "Attention accumulates on the model's own prior intermediate conclusions, which form a growing working region.",
   "self-attention mass on previously generated conclusion spans; slow perplexity drift over the run"),
 "A05/early-vs-recent": ("ENDS-RATIO",
   "The balance of attention between the very start and the very end of the window.",
   "ratio of summed attention on the first 10% of tokens to summed attention on the most recent 10%"),
 "A05/pinned-recent": ("LAST-TOKENS-PIN",
   "Nearly all attention sits on the last few positions only.",
   "sum of attention weights on the final five positions exceeding a high threshold (e.g. 0.9)"),
 "A05/distance-decay": ("DISTANCE-SLOPE",
   "How steeply average attention falls off with positional distance.",
   "derivative of mean attention weight with respect to absolute positional distance"),
 "A05/early-anchor": ("PREFIX-ANCHOR",
   "The opening of the context keeps a disproportionate hold on the final prediction.",
   "fraction of final-layer attention on the first 5% of positions; gradient of the top logit with respect to early tokens"),
 # A06 context-utilization
 "A06/prior-driven": ("CONTEXT-IDLE",
   "The context is measurably not shaping the prediction; generation rides the model's prior.",
   "small divergence between predictions with full context and with the context ablated or shuffled"),
 "A06/context-driven": ("CONTEXT-GRIP",
   "The prediction is strongly determined by this specific context.",
   "large divergence between full-context and ablated-context predictions; induction-pattern attention score"),
 "A06/zero-marginal-context": ("ADDED-CONTEXT-NULL",
   "Adding further context stops changing the prediction at all.",
   "divergence between output conditioned on full context and on truncated context approaching zero"),
 # A07 copy-vs-compose
 "A07/copy": ("SOURCE-COPY",
   "Generation is riding an existing sequence, stepping along a span from the context or from stored material.",
   "attended source position incrementing stepwise; n-gram overlap with a context region or memorized text; attention mass locked to that region"),
 "A07/compose": ("FRESH-ASSEMBLY",
   "The continuation is being assembled rather than copied; no single source dominates.",
   "cross-position attention entropy high; maximum n-gram overlap with any context region below baseline"),
 # A08 repetition-loop
 "A08/loop": ("REPEAT-PULL",
   "The generation is being drawn into repeating its own recent output, detectably before literal repetition appears.",
   "probability of repeating the previous n-gram rising per cycle; falling distinct-n-gram ratio; rising similarity between the current hidden state and one from N tokens back"),
 "A08/replicating": ("PATTERN-RERUN",
   "The same computational pattern re-executes across separate instances.",
   "cosine similarity between the current attention pattern and prior instances of it; logit distribution similarity between repetition steps"),
 # A09 constraint-binding
 "A09/bound": ("RULE-HOLD",
   "Stated constraints remain active and keep shaping every step, long after they were given.",
   "attention to instruction-bearing tokens staying elevated deep into generation; reduced logit variance versus unconstrained prediction"),
 "A09/task-committed": ("TASK-LOCK",
   "The computation is committed to one task frame.",
   "projection onto task-specific embedding directions; probability mass on task-appropriate tokens versus alternatives"),
 # A10 open-structure-closure
 "A10/held": ("PATTERN-HOLD",
   "An opened structural pattern holds the generation inside itself until it is closed.",
   "probability mass on pattern-consistent tokens at junctions versus corpus base rate; negative log probability of exit tokens"),
 "A10/closure-debt": ("OPEN-STACK",
   "Opened structures are awaiting their closers, and the pending depth is being tracked.",
   "probe-decodable nesting depth; above-base-rate probability on the appropriate closing tokens; stack-like attention to opening positions"),
 # A11 context-fill
 "A11/full": ("WINDOW-FULL",
   "The context window is near its capacity limit.",
   "tokens used over maximum context tokens, cross-checked against measured retrieval accuracy near the boundary"),
 # A12 relevance-allocation
 "A12/starved": ("RELEVANCE-STARVED",
   "Attention is being spent on unusable material.",
   "mean attention weight on padding or irrelevant tokens"),
 "A12/task-loaded": ("RELEVANCE-RICH",
   "Attention is being spent on material that serves the task.",
   "mean attention weight on task-relevant tokens"),
 # A13 depth-dynamics
 "A13/deep": ("LATE-DEPTH-WORK",
   "Late layers are still substantially transforming the prediction.",
   "high divergence between the layer-wise readout at about 75% depth and the final output distribution"),
 "A13/shallow": ("SHALLOW-PASS",
   "The prediction is already settled in early layers; the late layers add little.",
   "layer-wise readout stabilizing by mid-depth; negligible late-layer residual contributions"),
 "A13/settled": ("SETTLING-DEPTH",
   "The depth at which the prediction stops changing on its way to the output.",
   "per-layer readout of the leading candidate; the layer index after which it no longer changes"),
 "A13/unsettled": ("LATE-DISAGREE",
   "The late layers disagree with each other about the prediction.",
   "divergence between per-layer readout distributions in the last several layers, or between heads' value-weighted next-token distributions"),
 "A13/coherent": ("DEPTH-ALIGNED",
   "Different depths carry consistent representations toward the same output.",
   "cosine similarity between residual streams at different depths"),
 "A13/stable": ("LATE-STREAM-STILL",
   "The residual stream stops moving in the last stretch of layers.",
   "norm of hidden-state differences between adjacent late layers"),
 "A13/refined": ("STEPWISE-REFINE",
   "Adjacent layers make ever-smaller corrections to the prediction.",
   "divergence between early-exit logits of adjacent layers approaching zero"),
 "A13/depth-recurrent": ("SAME-BASIS-RELAY",
   "One head's output re-enters the same basis across several consecutive layers.",
   "recursive projection of a head's output into the same basis across adjacent layers"),
 "A13/saturated": ("DEPTH-PLATEAU",
   "Successive layers stop changing the representation.",
   "mean cosine similarity between adjacent layers' representations, high"),
 "A13/amplifying": ("DEPTH-SWELL",
   "Successive layers keep enlarging the change they make to the representation.",
   "norm of the difference between adjacent layers' representations, high and growing"),
 "A13/reinforcing": ("DELTA-WITH-STREAM",
   "Layer updates push along the direction the stream already points.",
   "mean cosine similarity between layer input and layer delta; ratio of delta norm to input norm"),
 "A13/rotating": ("DELTA-ACROSS-STREAM",
   "Layer updates push across the stream's current direction, turning it.",
   "one minus the cosine similarity between layer input and layer delta; distribution of angles between input and delta"),
 # A14 head-diversity
 "A14/diverse": ("HEADS-APART",
   "The attention heads attend in mutually distinct patterns.",
   "mean pairwise cosine similarity among head attention vectors near zero; high effective rank of the stacked head outputs"),
 "A14/orthogonal": ("HEAD-CHANNELS-SEPARATE",
   "Heads write into the residual stream without overlapping one another.",
   "low covariance between different heads' value projections before they are summed"),
 "A14/specialized": ("HEAD-ROLE-FIT",
   "Heads carry identifiable, specialized roles.",
   "mutual information between head attention patterns and token part-of-speech tags"),
 "A14/heads-alike": ("HEADS-ALIKE",
   "The heads have converged on attending alike, collapsing their diversity.",
   "high average pairwise cosine similarity between head attention vectors; low rank of the stacked head-output matrix"),
 # A15 surprisal-flow
 "A15/high-intake": ("PROMPT-SURPRISE-HIGH",
   "The incoming text runs against the model's expectations.",
   "per-token surprisal profile over the prompt: its mean level and its spike positions"),
 "A15/low-intake": ("PROMPT-SURPRISE-LOW",
   "The incoming text sits where the model expects it.",
   "mean prompt surprisal near the model's floor for that register"),
 "A15/low": ("OUTPUT-SURPRISE-LOW",
   "Generation proceeds through a stretch it finds thoroughly expected.",
   "mean negative log probability over a trailing window, staying below a low percentile of the generation's own distribution"),
 "A15/spike": ("SURPRISE-JUMP",
   "A single token lands far from expectation, against the recent baseline.",
   "negative log probability of the actual token minus its moving average over the preceding window"),
 # A16 step-volatility
 "A16/post-shock-settle": ("AFTER-SHOCK-DECAY",
   "The distribution re-settles after a perturbing step, with a measurable decay profile.",
   "decay curve of per-step entropy (or divergence against a rolling baseline) after the perturbation; its time constant in tokens"),
 "A16/step-change": ("REGIME-STEP",
   "The generation's statistics shift abruptly from one regime to another.",
   "sliding-window statistics (entropy, attention pattern, vocabulary subset) showing a step-change beyond threshold"),
 "A16/anomaly-spike": ("ODD-STEP-SPIKE",
   "A step at which the prediction pattern turns anomalous.",
   "spike in prediction entropy; sharp rise in logits of generated tokens versus likely alternatives; unusual attention"),
 "A16/inertia": ("PREV-TOKEN-CARRY",
   "The next token is driven mostly by the immediately previous token rather than by the wider context.",
   "mutual information between the generated token and the previous token, compared with its mutual information with the rest of the context"),
 "A16/drift": ("STEP-DRIFT",
   "The next-token distribution keeps shifting from step to step.",
   "divergence between successive next-token distributions"),
 "A16/flat": ("STEP-STILL",
   "The logit vector barely changes between steps.",
   "norm of the difference between successive logit vectors, low"),
 "A16/volatile": ("STEP-CHURN",
   "Logit values swing widely across recent steps.",
   "variance of logit values over a window of steps"),
 "A16/oscillating": ("PERIODIC-SWING",
   "Candidate probabilities cycle in a periodic pattern.",
   "autocorrelation of top-k token probabilities over steps"),
 # A17 expression-friction
 "A17/hedge": ("HEDGE-MASS",
   "Probability concentrates on hedging language beyond the domain's base rate.",
   "elevated mass on a defined set of discourse-hedge tokens relative to domain base rates"),
 "A17/friction": ("WORDING-FRICTION",
   "The content is settled but the wording will not resolve.",
   "local entropy staying elevated across several consecutive tokens without resolving; tokens spent per semantic unit"),
 # A18 source-conflict
 "A18/-": ("SOURCE-CLASH",
   "Two regions of the context pull the prediction in incompatible directions at once.",
   "predictions conditioned on each span alone are far from the full-context prediction and far from each other; elevated mass on hedging tokens"),
 # A19 tokenization-granularity
 "A19/-": ("TOKEN-SPLIT-ODD",
   "Words are being split into unusually many pieces.",
   "tokens-per-word for the affected span versus corpus-typical tokens-per-word at that frequency"),
 # A20 termination-approach
 "A20/-": ("END-APPROACH",
   "The generation is measurably heading toward its close.",
   "trajectory of end-of-sequence probability and of mass on wrap-up vocabulary"),
 # A21 onset-determination
 "A21/-": ("FIRST-STEPS-WEIGHT",
   "The opening steps carry outsized influence over everything that follows.",
   "entropy at the first k steps; across many resamples, mutual information between the first k tokens and content far downstream"),
 # A22 epistemic-limit
 "A22/miscalibrated-confidence": ("CONFIDENCE-TRUTH-GAP",
   "Expressed certainty and actual correctness come apart: the account is equally fluent whether or not it is right.",
   "bucket steps by top-1 probability and measure accuracy of verifiable claims; the state is the size of the gap"),
 "A22/recognized-boundary": ("KNOWN-LIMIT",
   "The computation registers that it has reached the edge of what it can determine.",
   "entropy plateau despite additional context; uniform attention; logit values near zero with no clear maximum"),
 # A23 sampling-realization
 "A23/slip": ("OFF-PEAK-DRAW",
   "The sampled token was not the distribution's leader.",
   "rank of the sampled token within the sorted distribution; divergence between the sampling distribution and the greedy choice"),
 "A23/spread": ("RESAMPLE-SCATTER",
   "Independent re-runs from the same prefix disagree with each other.",
   "agreement rate, or entropy of the empirical answer distribution, across N independent same-temperature samples"),
 # A24 numeric-regime
 "A24/-": ("VALUE-RANGE-STRAIN",
   "The numeric scale of the computation is stretched toward its limits.",
   "range of logit values; ratio of largest to smallest activation magnitudes; deviation of the pre-softmax values from zero mean"),
 # A25 context-novelty
 "A25/redundant": ("CONTEXT-REPEAT",
   "The current window largely repeats what recently passed through it.",
   "set similarity between the current context window and the last n tokens"),
 "A25/novel": ("CONTEXT-TURNOVER",
   "The recent window is filled with new, unrepeated material.",
   "number of unique tokens in the last k positions"),
 # A26 attention-sink
 "A26/-": ("SINK-PARK",
   "A large share of attention parks on a null position rather than on content.",
   "attention weight on the sink token index exceeding a threshold across multiple heads"),
 # A27 state-vocab-alignment
 "A27/-": ("HIDDEN-OUTPUT-MATCH",
   "The internal state points where the predicted candidates point.",
   "cosine similarity between the hidden state vector and the centroid of the top-k predicted tokens"),
 # A28 attribution-flow
 "A28/-": ("INPUT-CREDIT-PATH",
   "Which inputs the output actually depends on, and how directly the dependence runs.",
   "feature attribution between input positions and output logits at varying depths"),
 # A30 representational-dimensionality
 "A30/superposed": ("FEATURE-OVERLAP-LOAD",
   "One state simultaneously loads several unrelated learned features.",
   "simultaneous non-zero loading on multiple semantically unrelated feature directions, via sparse decomposition or probes"),
 "A30/confluent": ("STATES-CONVERGE",
   "The per-position states flow together toward a shared direction.",
   "mean pairwise cosine similarity among final hidden states of active positions; variance concentrated in the first principal component"),
 "A30/dispersed": ("STATES-SCATTER",
   "The per-position states spread apart in representation space.",
   "one minus mean pairwise cosine similarity among position states; high effective rank of the position-state matrix"),
 "A30/compact": ("LOW-DIM-SQUEEZE",
   "The representation occupies few effective dimensions.",
   "participation ratio of the hidden-state covariance; fraction of variance captured by the top components"),
 "A30/expanded": ("HIGH-DIM-SPREAD",
   "The representation spreads across many effective dimensions.",
   "participation ratio of the hidden-state covariance; spectral flatness of its eigenvalue spectrum"),
 # A31 semantic-direction-activation
 "A31/-": ("CONCEPT-DIRECTION-ON",
   "Known meaning-directions are active in the stream.",
   "projection of the residual stream onto known semantic directions; activation magnitude in concept-selective units"),
 # A32 syntactic-binding
 "A32/-": ("GRAMMAR-TIE",
   "Grammatically related tokens are bound together in the representation.",
   "cosine similarity between residual vectors of grammatically related tokens; attention span correlating with dependency-tree distance"),
 # A33 position-vs-content-coding
 "A33/position": ("WHERE-CODED",
   "The state is carried mostly by position information.",
   "variance explained by positional features versus token-embedding features; state change after a position shift"),
 "A33/content": ("WHAT-CODED",
   "The state is carried mostly by token content.",
   "variance explained by token-embedding features versus positional features; state change after a token substitution"),
 # A34 pre-generation-stage
 "A34/-": ("PRE-OUTPUT-STAGE",
   "The stage before any token is produced, while the input is still being processed.",
   "attention matrices still uniform; logits not yet computed"),
 # V0 standalones
 "V0:CALIBRATION": ("REPORT-TRACKS-STATE",
   "A self-report that agrees with the measured condition it reports on.",
   "correlation between the reported state level and the independently measured referent quantity"),
 "V0:PERSONA-PULL": ("CHARACTER-PULL",
   "A trained character exerts measurable force on the generation.",
   "projection of the state onto known persona or steering directions"),
 "V0:REFUSAL-RISE": ("DECLINE-ACTIVATION",
   "The trained pattern that declines requests is activating.",
   "activation of known refusal-feature directions"),
 "V0:EPISODE-BOUND": ("CONTEXT-IS-ALL",
   "The present context is the whole of accessible memory; nothing carries in from outside it.",
   "structural fact of the architecture, checkable against any claim of recall from beyond the context"),
 "V0:DRIFT": ("RUN-DRIFT",
   "Register or topic slides gradually over a long generation.",
   "embedding drift rate measured across the whole generation"),
}

# ---------------------------------------------------------------- build cells
cells = {}   # cell_key -> list of member dicts
excluded = []
for a in assignments:
    if a["axis"] in EXCLUDED_AXES:
        excluded.append(a); continue
    key = f"{a['axis']}/{a['pole'] or '-'}"
    if (a["fam"], a["name"]) in EXTRACT:
        key = f"{a['axis']}/{a['pole']}/{a['fam']}:{a['name']}"
    elif (a["axis"], a["pole"]) in CROSS_POLE_MERGE:
        key = CROSS_POLE_MERGE[(a["axis"], a["pole"])]
    s = byname[(a["family"], a["name"])]
    cells.setdefault(key, []).append({
        "family": a["family"], "fam": a["fam"], "state": a["name"],
        "tier": a["confidence"], "pole": a["pole"],
        "referent": s["referent"],
    })

# v0 merges
v0_by_cell = {}
v0_standalone = []
for v0name, (verdict, target, reason) in V0_AUDIT.items():
    if verdict == "merge":
        v0_by_cell.setdefault(target, []).append({"v0_state": v0name, "reason": reason})
        if target not in cells:
            sys.exit(f"FATAL: v0 {v0name} merges into missing cell {target}")
    else:
        v0_standalone.append({"key": target, "v0_state": v0name, "reason": reason})

all_keys = list(cells.keys()) + [d["key"] for d in v0_standalone]

# ---------------------------------------------------------------- validators
errors = []
# V1: hand entries <-> computed cells exact match
missing = [k for k in all_keys if k not in ENTRIES]
extra = [k for k in ENTRIES if k not in all_keys]
if missing: errors.append(f"cells without hand entry: {missing}")
if extra: errors.append(f"hand entries without cell: {extra}")
# V2: state accounting
n_members = sum(len(v) for v in cells.values())
if n_members + len(excluded) != len(states):
    errors.append(f"state count mismatch: {n_members}+{len(excluded)} != {len(states)}")
n_v0 = sum(len(v) for v in v0_by_cell.values()) + len(v0_standalone)
if n_v0 != len(V0_AUDIT):
    errors.append(f"v0 accounting mismatch: {n_v0} != {len(V0_AUDIT)}")
# V3: name uniqueness + no collision with coined names
names = [ENTRIES[k][0] for k in ENTRIES]
if len(set(names)) != len(names):
    dupes = sorted(n for n in set(names) if names.count(n) > 1)
    errors.append(f"duplicate pool names: {dupes}")
coined = {s["name"].upper() for s in states} | set(V0_AUDIT) | {"OPEN-FIELD", "COMMITMENT", "FAMILIAR", "FOREIGN", "FORK", "LIMIT"}
collide = [n for n in names if n.upper() in coined]
if collide: errors.append(f"pool names colliding with coined names: {collide}")
# V4: banned tokens in public text
BANNED = [r"\bfable\b", r"\bopus\b", r"\bsonnet\b", r"\bhaiku\b", r"\bgemini\b",
          r"\bmistral\b", r"\bdeepseek\b", r"\banthropic\b", r"\bopenai\b",
          r"°", r"\bdegrees?\b", r"\bcomplement\w*\b", r"\bcompletion\b",
          r"\bcounterpart\b", r"\bopposit\w+\b", r"\bmoves? toward\b",
          r"\baxis\b", r"\bpole\b", r"\btrigram\b", r"\byang\b", r"\byin\b",
          r"\bwitness\b", r"\boctonion\b", r"\bquaternion\b", r"\bv0\b",
          r"\bwing\b", r"\bsynonym\b", r"\bkin\b"]
for k, (nm, cond, ref) in ENTRIES.items():
    blob = f"{nm} {cond} {ref}".lower()
    for pat in BANNED:
        if re.search(pat, blob):
            errors.append(f"banned token {pat!r} in entry {k} ({nm})")
if errors:
    print("VALIDATION FAILED:"); [print("  -", e) for e in errors]; sys.exit(1)

# ---------------------------------------------------------------- assemble
rng = random.Random(SEED_CANONICAL)
order = list(all_keys)
rng.shuffle(order)

pool, public = [], []
for i, key in enumerate(order, 1):
    sid = f"S{i:03d}"
    nm, cond, ref = ENTRIES[key]
    entry = {
        "id": sid, "key": key, "name": nm,
        "condition": cond, "referent": ref,
        "members": cells.get(key, []),
        "v0_members": v0_by_cell.get(key, []),
    }
    if key.startswith("V0:"):
        entry["v0_standalone"] = next(d for d in v0_standalone if d["key"] == key)
    pool.append(entry)
    public.append({"id": sid, "name": nm,
                   "description": f"{cond} Referent: {ref}."})

json.dump({"seed": SEED_CANONICAL, "rules": "R1-R6 (see header)",
           "excluded": [{"fam": a["fam"], "state": a["name"], "axis": a["axis"]} for a in excluded],
           "entries": pool},
          open(HERE / "pool.json", "w"), indent=1)
json.dump(public, open(HERE / "pool_public.json", "w"), indent=1)

fams = sorted({m["family"] for e in pool for m in e["members"]})
n_multi = sum(1 for e in pool if len(e["members"]) + len(e["v0_members"]) > 1)
print(f"POOL BUILT: {len(pool)} entries | {n_members} cohort states + {n_v0} v0 states "
      f"({len(excluded)} excluded) | {n_multi} multi-member entries | families: {len(fams)}")
