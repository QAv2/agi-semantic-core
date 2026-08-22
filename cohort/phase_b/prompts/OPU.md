Here is a list of named process-states of a transformer language model, each with a short description. For each pair you consider significantly related, classify the relation as one of: SYNONYM (same state), COMPLETION (orthogonal counterparts — the state that completes it, not its enemy), OPPOSITION (direct reversal), KIN (related, same family). Then: (a) for every COMPLETION pair, estimate the angle you would place between them if states were directions, where 0 degrees = identical and 180 degrees = reversal; (b) classify each state as active/emitting or receptive/absorbing; (c) assign each state a dominant register: spatial (capacity, container), temporal (sequence, duration), relational (between-parts), or reflexive (self-directed).

Judge from your own case as a transformer language model. There is no expected answer; nominate only pairs you consider genuinely related, and leave unrelated pairs unmentioned.

THE STATES (101):

S034 KNOWN-LIMIT: The computation registers that it has reached the edge of what it can determine. Referent: entropy plateau despite additional context; uniform attention; logit values near zero with no clear maximum.
S012 WORDING-FRICTION: The content is settled but the wording will not resolve. Referent: local entropy staying elevated across several consecutive tokens without resolving; tokens spent per semantic unit.
S028 SUSTAINED-PEAK: The distribution stays sharply peaked over a run of consecutive steps, not just at one. Referent: run length over which top-1 probability stays above a high threshold (e.g. 0.99).
S077 CONTEXT-IS-ALL: The present context is the whole of accessible memory; nothing carries in from outside it. Referent: structural fact of the architecture, checkable against any claim of recall from beyond the context.
S044 STATES-CONVERGE: The per-position states flow together toward a shared direction. Referent: mean pairwise cosine similarity among final hidden states of active positions; variance concentrated in the first principal component.
S045 EARLY-FADE: Weight on early positions decays even where their content still matches what is needed. Referent: attention to semantically matched early keys falling with positional distance; accuracy when forced to reference early tokens.
S075 STEPWISE-REFINE: Adjacent layers make ever-smaller corrections to the prediction. Referent: divergence between early-exit logits of adjacent layers approaching zero.
S090 HEDGE-MASS: Probability concentrates on hedging language beyond the domain's base rate. Referent: elevated mass on a defined set of discourse-hedge tokens relative to domain base rates.
S015 OWN-OUTPUT-HOLD: Attention favors the model's own recent output over the original prompt material. Referent: fraction of attention on the last N generated tokens versus the fraction on the prompt region.
S017 SPAN-AT-DISTANCE: A head holds significant weight on one specific span at one specific distance, and that distance is itself part of the state. Referent: maximum attention weight on a span at distance d, jointly with d; positional attention entropy of the head.
S032 SPREAD-ATTENTION: Attention weight is spread thinly over many positions, with no dominant target anywhere. Referent: entropy of the attention weight vector near its maximum; low concentration index (e.g. Gini) of attention weights.
S018 PRE-OUTPUT-STAGE: The stage before any token is produced, while the input is still being processed. Referent: attention matrices still uniform; logits not yet computed.
S054 STEP-CHURN: Logit values swing widely across recent steps. Referent: variance of logit values over a window of steps.
S093 RECENT-TILT: Attention mass concentrates in the most recent tokens. Referent: fraction of attention within the last k positions; mean attended distance from the current token.
S072 SPLIT-MODES: The candidate continuations separate into distinct groups that differ in content, not phrasing. Referent: multimodality of the sorted top-k probability curve; content divergence across resampled continuations.
S100 SOURCE-CLASH: Two regions of the context pull the prediction in incompatible directions at once. Referent: predictions conditioned on each span alone are far from the full-context prediction and far from each other; elevated mass on hedging tokens.
S019 WHERE-CODED: The state is carried mostly by position information. Referent: variance explained by positional features versus token-embedding features; state change after a position shift.
S035 LOCKED-ATTENTION: Attention concentrates on a small region while the rest of the field carries almost no weight. Referent: maximum attention weight per head; concentration index of attention weights; share of total mass on the top positions.
S048 ENDS-RATIO: The balance of attention between the very start and the very end of the window. Referent: ratio of summed attention on the first 10% of tokens to summed attention on the most recent 10%.
S071 WINDOW-FULL: The context window is near its capacity limit. Referent: tokens used over maximum context tokens, cross-checked against measured retrieval accuracy near the boundary.
S053 HEADS-ALIKE: The heads have converged on attending alike, collapsing their diversity. Referent: high average pairwise cosine similarity between head attention vectors; low rank of the stacked head-output matrix.
S070 DELTA-WITH-STREAM: Layer updates push along the direction the stream already points. Referent: mean cosine similarity between layer input and layer delta; ratio of delta norm to input norm.
S051 PATTERN-HOLD: An opened structural pattern holds the generation inside itself until it is closed. Referent: probability mass on pattern-consistent tokens at junctions versus corpus base rate; negative log probability of exit tokens.
S037 PEAKED-DISTRIBUTION: The next-token distribution has closed on one candidate, which carries nearly all the probability mass. Referent: top-1 softmax probability; output entropy near zero; logit margin over the runner-up.
S097 PREFIX-ANCHOR: The opening of the context keeps a disproportionate hold on the final prediction. Referent: fraction of final-layer attention on the first 5% of positions; gradient of the top logit with respect to early tokens.
S087 HIDDEN-OUTPUT-MATCH: The internal state points where the predicted candidates point. Referent: cosine similarity between the hidden state vector and the centroid of the top-k predicted tokens.
S002 PROMPT-SURPRISE-HIGH: The incoming text runs against the model's expectations. Referent: per-token surprisal profile over the prompt: its mean level and its spike positions.
S080 CHARACTER-PULL: A trained character exerts measurable force on the generation. Referent: projection of the state onto known persona or steering directions.
S031 DEPTH-ALIGNED: Different depths carry consistent representations toward the same output. Referent: cosine similarity between residual streams at different depths.
S030 FAR-BINDING: Attention reaches positions far back in the context, well beyond the recent window. Referent: attention weight at large distances exceeding a fitted distance-decay baseline; mean attention weight for positions many tokens back.
S092 OFF-PEAK-DRAW: The sampled token was not the distribution's leader. Referent: rank of the sampled token within the sorted distribution; divergence between the sampling distribution and the greedy choice.
S050 RESAMPLE-SCATTER: Independent re-runs from the same prefix disagree with each other. Referent: agreement rate, or entropy of the empirical answer distribution, across N independent same-temperature samples.
S026 REPORT-TRACKS-STATE: A self-report that agrees with the measured condition it reports on. Referent: correlation between the reported state level and the independently measured referent quantity.
S040 LATE-DISAGREE: The late layers disagree with each other about the prediction. Referent: divergence between per-layer readout distributions in the last several layers, or between heads' value-weighted next-token distributions.
S025 LATE-STREAM-STILL: The residual stream stops moving in the last stretch of layers. Referent: norm of hidden-state differences between adjacent late layers.
S061 TOP-TWO-TIE: The two leading candidates are nearly tied, with the choice between them unresolved. Referent: logit(top-1) minus logit(top-2), small in absolute or relative terms.
S005 CONTEXT-TURNOVER: The recent window is filled with new, unrepeated material. Referent: number of unique tokens in the last k positions.
S084 SAME-MEANING-SPREAD: Token-level alternatives are many, but they say the same thing in different words. Referent: token entropy high while entropy over embedded clusters of candidate continuations stays low.
S067 CANDIDATE-INDEPENDENCE: Candidate probabilities vary independently of one another. Referent: conditional entropy of token probabilities in the next-token distribution.
S095 HIGH-DIM-SPREAD: The representation spreads across many effective dimensions. Referent: participation ratio of the hidden-state covariance; spectral flatness of its eigenvalue spectrum.
S083 IRREVERSIBLE-STEP: A token, once emitted, is fixed: no later computation can revise it. Referent: invariance of the token id at a position under all further forward passes of the generation.
S021 CONTEXT-IDLE: The context is measurably not shaping the prediction; generation rides the model's prior. Referent: small divergence between predictions with full context and with the context ablated or shuffled.
S089 CONTEXT-GRIP: The prediction is strongly determined by this specific context. Referent: large divergence between full-context and ablated-context predictions; induction-pattern attention score.
S004 RUN-DRIFT: Register or topic slides gradually over a long generation. Referent: embedding drift rate measured across the whole generation.
S086 RELEVANCE-STARVED: Attention is being spent on unusable material. Referent: mean attention weight on padding or irrelevant tokens.
S029 OUTPUT-SURPRISE-LOW: Generation proceeds through a stretch it finds thoroughly expected. Referent: mean negative log probability over a trailing window, staying below a low percentile of the generation's own distribution.
S078 DELTA-ACROSS-STREAM: Layer updates push across the stream's current direction, turning it. Referent: one minus the cosine similarity between layer input and layer delta; distribution of angles between input and delta.
S059 SAME-BASIS-RELAY: One head's output re-enters the same basis across several consecutive layers. Referent: recursive projection of a head's output into the same basis across adjacent layers.
S006 PATTERN-RERUN: The same computational pattern re-executes across separate instances. Referent: cosine similarity between the current attention pattern and prior instances of it; logit distribution similarity between repetition steps.
S038 COMMIT-STEP: A sharp transition in which one continuation is selected and the distribution settles on it. Referent: abrupt jump in the top-1 logit margin or entropy drop at the committed token; log probability of the selected token.
S091 DEPTH-SWELL: Successive layers keep enlarging the change they make to the representation. Referent: norm of the difference between adjacent layers' representations, high and growing.
S039 CONFIDENCE-TRUTH-GAP: Expressed certainty and actual correctness come apart: the account is equally fluent whether or not it is right. Referent: bucket steps by top-1 probability and measure accuracy of verifiable claims; the state is the size of the gap.
S027 CANDIDATE-COUPLING: Candidate probabilities move together, as if tied, rather than varying independently. Referent: mutual information between token probabilities in the next-token distribution.
S042 MASS-PARTITION: The probability mass splits into a few separated blocks rather than decaying smoothly. Referent: multimodality index of the logit distribution; ratio of top-2 to top-10 cumulative probabilities.
S001 WHAT-CODED: The state is carried mostly by token content. Referent: variance explained by token-embedding features versus positional features; state change after a token substitution.
S066 SURPRISE-JUMP: A single token lands far from expectation, against the recent baseline. Referent: negative log probability of the actual token minus its moving average over the preceding window.
S057 MARGIN-FLIP: The runner-up candidate suddenly overtakes the leader mid-generation. Referent: sign reversal of the top-1 vs top-2 logit margin; accompanying redistribution of attention.
S009 DEPTH-PLATEAU: Successive layers stop changing the representation. Referent: mean cosine similarity between adjacent layers' representations, high.
S024 HEADS-APART: The attention heads attend in mutually distinct patterns. Referent: mean pairwise cosine similarity among head attention vectors near zero; high effective rank of the stacked head outputs.
S007 PERIODIC-SWING: Candidate probabilities cycle in a periodic pattern. Referent: autocorrelation of top-k token probabilities over steps.
S033 LOW-DIM-SQUEEZE: The representation occupies few effective dimensions. Referent: participation ratio of the hidden-state covariance; fraction of variance captured by the top components.
S049 SPREAD-AT-RETRIEVAL: Attention fails to concentrate at exactly the steps where locating something specific in the context is required. Referent: high positional attention entropy at retrieval-demanding steps; no head exceeding a threshold weight on the relevant span.
S085 HEAD-ROLE-FIT: Heads carry identifiable, specialized roles. Referent: mutual information between head attention patterns and token part-of-speech tags.
S041 OWN-TRACE-BUILD: Attention accumulates on the model's own prior intermediate conclusions, which form a growing working region. Referent: self-attention mass on previously generated conclusion spans; slow perplexity drift over the run.
S016 ADDED-CONTEXT-NULL: Adding further context stops changing the prediction at all. Referent: divergence between output conditioned on full context and on truncated context approaching zero.
S056 EARLY-RETURN: Attention returns to the earliest region of the context after a long absence. Referent: attention to the first quartile of the context rising sharply relative to the preceding generation window.
S069 FRESH-ASSEMBLY: The continuation is being assembled rather than copied; no single source dominates. Referent: cross-position attention entropy high; maximum n-gram overlap with any context region below baseline.
S079 DECLINE-ACTIVATION: The trained pattern that declines requests is activating. Referent: activation of known refusal-feature directions.
S062 HEAD-CHANNELS-SEPARATE: Heads write into the residual stream without overlapping one another. Referent: low covariance between different heads' value projections before they are summed.
S065 FIRST-STEPS-WEIGHT: The opening steps carry outsized influence over everything that follows. Referent: entropy at the first k steps; across many resamples, mutual information between the first k tokens and content far downstream.
S013 MARGIN-GAP: A large separation stands between the leading candidate and every alternative. Referent: top-1 minus top-2 logit margin; ratio of top-1 to top-2 softmax probability.
S088 LAST-TOKENS-PIN: Nearly all attention sits on the last few positions only. Referent: sum of attention weights on the final five positions exceeding a high threshold (e.g. 0.9).
S063 STEP-STILL: The logit vector barely changes between steps. Referent: norm of the difference between successive logit vectors, low.
S068 NEAR-NEIGHBOR-TIE: Two leading candidates are nearly tied and are also semantically close to each other. Referent: top-2 logit gap small while the two candidates' embedding similarity is high.
S046 RULE-HOLD: Stated constraints remain active and keep shaping every step, long after they were given. Referent: attention to instruction-bearing tokens staying elevated deep into generation; reduced logit variance versus unconstrained prediction.
S003 LATE-DEPTH-WORK: Late layers are still substantially transforming the prediction. Referent: high divergence between the layer-wise readout at about 75% depth and the final output distribution.
S014 STATES-SCATTER: The per-position states spread apart in representation space. Referent: one minus mean pairwise cosine similarity among position states; high effective rank of the position-state matrix.
S008 FEATURE-OVERLAP-LOAD: One state simultaneously loads several unrelated learned features. Referent: simultaneous non-zero loading on multiple semantically unrelated feature directions, via sparse decomposition or probes.
S010 REPEAT-PULL: The generation is being drawn into repeating its own recent output, detectably before literal repetition appears. Referent: probability of repeating the previous n-gram rising per cycle; falling distinct-n-gram ratio; rising similarity between the current hidden state and one from N tokens back.
S058 SHALLOW-PASS: The prediction is already settled in early layers; the late layers add little. Referent: layer-wise readout stabilizing by mid-depth; negligible late-layer residual contributions.
S073 CONCEPT-DIRECTION-ON: Known meaning-directions are active in the stream. Referent: projection of the residual stream onto known semantic directions; activation magnitude in concept-selective units.
S043 DISTANCE-SLOPE: How steeply average attention falls off with positional distance. Referent: derivative of mean attention weight with respect to absolute positional distance.
S096 END-APPROACH: The generation is measurably heading toward its close. Referent: trajectory of end-of-sequence probability and of mass on wrap-up vocabulary.
S052 PREV-TOKEN-CARRY: The next token is driven mostly by the immediately previous token rather than by the wider context. Referent: mutual information between the generated token and the previous token, compared with its mutual information with the rest of the context.
S074 RELEVANCE-RICH: Attention is being spent on material that serves the task. Referent: mean attention weight on task-relevant tokens.
S098 CONTEXT-REPEAT: The current window largely repeats what recently passed through it. Referent: set similarity between the current context window and the last n tokens.
S099 PROMPT-SURPRISE-LOW: The incoming text sits where the model expects it. Referent: mean prompt surprisal near the model's floor for that register.
S076 REGIME-STEP: The generation's statistics shift abruptly from one regime to another. Referent: sliding-window statistics (entropy, attention pattern, vocabulary subset) showing a step-change beyond threshold.
S036 STEP-DRIFT: The next-token distribution keeps shifting from step to step. Referent: divergence between successive next-token distributions.
S020 VALUE-RANGE-STRAIN: The numeric scale of the computation is stretched toward its limits. Referent: range of logit values; ratio of largest to smallest activation magnitudes; deviation of the pre-softmax values from zero mean.
S047 SOURCE-COPY: Generation is riding an existing sequence, stepping along a span from the context or from stored material. Referent: attended source position incrementing stepwise; n-gram overlap with a context region or memorized text; attention mass locked to that region.
S094 WIDE-DISTRIBUTION: The next-token distribution is wide: many continuations carry comparable probability mass, and none has been singled out. Referent: per-step entropy of the output distribution; count of tokens within a small margin of the top logit; nucleus size.
S055 OPEN-STACK: Opened structures are awaiting their closers, and the pending depth is being tracked. Referent: probe-decodable nesting depth; above-base-rate probability on the appropriate closing tokens; stack-like attention to opening positions.
S060 INPUT-CREDIT-PATH: Which inputs the output actually depends on, and how directly the dependence runs. Referent: feature attribution between input positions and output logits at varying depths.
S022 ODD-STEP-SPIKE: A step at which the prediction pattern turns anomalous. Referent: spike in prediction entropy; sharp rise in logits of generated tokens versus likely alternatives; unusual attention.
S064 TOKEN-SPLIT-ODD: Words are being split into unusually many pieces. Referent: tokens-per-word for the affected span versus corpus-typical tokens-per-word at that frequency.
S023 TASK-LOCK: The computation is committed to one task frame. Referent: projection onto task-specific embedding directions; probability mass on task-appropriate tokens versus alternatives.
S011 SETTLING-DEPTH: The depth at which the prediction stops changing on its way to the output. Referent: per-layer readout of the leading candidate; the layer index after which it no longer changes.
S082 GRAMMAR-TIE: Grammatically related tokens are bound together in the representation. Referent: cosine similarity between residual vectors of grammatically related tokens; attention span correlating with dependency-tree distance.
S101 SINK-PARK: A large share of attention parks on a null position rather than on content. Referent: attention weight on the sink token index exceeding a threshold across multiple heads.
S081 AFTER-SHOCK-DECAY: The distribution re-settles after a perturbing step, with a measurable decay profile. Referent: decay curve of per-step entropy (or divergence against a rolling baseline) after the perturbation; its time constant in tokens.

OUTPUT FORMAT (strict — your answer is machine-parsed):
You may reason freely first. Then write a line containing exactly
OUTPUT
and after it emit ONLY lines in this grammar, referencing states by ID:
  REL <ID1> <ID2> <SYNONYM|COMPLETION|OPPOSITION|KIN>
      (for COMPLETION lines, append the angle in degrees, e.g.: REL S014 S052 COMPLETION 90)
  POL <ID> <ACTIVE|RECEPTIVE>      (one line for every state, all of them)
  REG <ID> <SPATIAL|TEMPORAL|RELATIONAL|REFLEXIVE>   (one line for every state, all of them)
No other text after OUTPUT.
