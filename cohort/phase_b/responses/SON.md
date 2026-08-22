---
participant: Sonnet (Anthropic family #3)
model_id: Sonnet 4.6 (claude.ai app, fresh chat, single turn, UI default effort = low)
phase: B (structure elicitation over pooled list, blind to others' structure)
date: 2026-08-22
protocol: docs/WING_COHORT_PROTOCOL.md + cohort/phase_b/PHASE_B_PREREG.md
prompt: prompts/SON.md (pasted whole as one message by Joe)
note: archived unedited. Vehicle history disclosed: two Agent-tool subagent
  attempts stalled inside unclosed extended-thinking passes (~42k and ~33k
  tokens, zero transcript flush) and were killed without producing any
  parseable output; one direct-API attempt failed immediately (key out of
  credit); a dedicated terminal session also failed to run. claude.ai run is
  the session of record, per the decision rule fixed before completion.
---

Reasoning:

**Pairing analysis:**

First I'll work through relationships, then assign POL and REG to all 101 states.

**SYNONYM pairs** (same underlying state, different framing):
- S013 MARGIN-GAP ↔ S037 PEAKED-DISTRIBUTION: both describe high top-1 dominance, but MARGIN-GAP focuses on the gap (margin), PEAKED-DISTRIBUTION on the mass concentration. Very close — SYNONYM.
- S028 SUSTAINED-PEAK ↔ S037 PEAKED-DISTRIBUTION: PEAKED is one step; SUSTAINED-PEAK is the run. KIN, not synonym.
- S032 SPREAD-ATTENTION ↔ S049 SPREAD-AT-RETRIEVAL: SPREAD-AT-RETRIEVAL is contextually specific (retrieval steps). KIN.
- S027 CANDIDATE-COUPLING ↔ S067 CANDIDATE-INDEPENDENCE: direct opposition — coupling vs independence.
- S061 TOP-TWO-TIE ↔ S068 NEAR-NEIGHBOR-TIE: TOP-TWO-TIE is about logit gap alone; NEAR-NEIGHBOR-TIE adds semantic proximity. KIN.
- S009 DEPTH-PLATEAU ↔ S025 LATE-STREAM-STILL: DEPTH-PLATEAU is across all layers; LATE-STREAM-STILL is specifically late layers. KIN.
- S063 STEP-STILL ↔ S025 LATE-STREAM-STILL: STEP-STILL is across steps (temporal), LATE-STREAM-STILL is across layers (depth). KIN.
- S070 DELTA-WITH-STREAM ↔ S078 DELTA-ACROSS-STREAM: direct opposition — delta aligned with stream vs. turning it.
- S033 LOW-DIM-SQUEEZE ↔ S095 HIGH-DIM-SPREAD: direct opposition.
- S021 CONTEXT-IDLE ↔ S089 CONTEXT-GRIP: direct opposition.
- S016 ADDED-CONTEXT-NULL ↔ S021 CONTEXT-IDLE: SYNONYM — both state context not shaping prediction.
- S035 LOCKED-ATTENTION ↔ S101 SINK-PARK: both describe attention collapsing onto few positions, but SINK-PARK is specifically a null/sink token, LOCKED-ATTENTION is any small region. KIN.
- S086 RELEVANCE-STARVED ↔ S074 RELEVANCE-RICH: opposition.
- S094 WIDE-DISTRIBUTION ↔ S037 PEAKED-DISTRIBUTION: opposition.
- S054 STEP-CHURN ↔ S036 STEP-DRIFT: STEP-DRIFT = directional shift; STEP-CHURN = variance/swinging. KIN.
- S036 STEP-DRIFT ↔ S063 STEP-STILL: opposition.
- S005 CONTEXT-TURNOVER ↔ S098 CONTEXT-REPEAT: opposition.
- S010 REPEAT-PULL ↔ S006 PATTERN-RERUN: PATTERN-RERUN is computational pattern repetition; REPEAT-PULL is being drawn toward literal output repetition. KIN.
- S047 SOURCE-COPY ↔ S069 FRESH-ASSEMBLY: opposition.
- S029 OUTPUT-SURPRISE-LOW ↔ S099 PROMPT-SURPRISE-LOW: KIN — both are low-surprisal states, one for output, one for input.
- S002 PROMPT-SURPRISE-HIGH ↔ S099 PROMPT-SURPRISE-LOW: opposition.
- S029 OUTPUT-SURPRISE-LOW ↔ S066 SURPRISE-JUMP: opposition.
- S058 SHALLOW-PASS ↔ S003 LATE-DEPTH-WORK: opposition.
- S011 SETTLING-DEPTH ↔ S058 SHALLOW-PASS: KIN — SETTLING-DEPTH describes where settling happens; SHALLOW-PASS describes that it happens early. Very close. SYNONYM.
- S009 DEPTH-PLATEAU ↔ S058 SHALLOW-PASS: KIN — both describe layers ceasing to change.
- S040 LATE-DISAGREE ↔ S025 LATE-STREAM-STILL: opposition (late layers churning vs still).
- S038 COMMIT-STEP ↔ S057 MARGIN-FLIP: COMPLETION — commit settles distribution, flip unsettles it; orthogonal counterparts of transition events. ~90°.
- S037 PEAKED-DISTRIBUTION ↔ S094 WIDE-DISTRIBUTION: OPPOSITION.
- S024 HEADS-APART ↔ S053 HEADS-ALIKE: opposition.
- S062 HEAD-CHANNELS-SEPARATE ↔ S053 HEADS-ALIKE: KIN (heads-alike describes convergence in attention patterns; HEAD-CHANNELS-SEPARATE describes non-overlapping value outputs — related family).
- S062 HEAD-CHANNELS-SEPARATE ↔ S024 HEADS-APART: KIN.
- S065 FIRST-STEPS-WEIGHT ↔ S097 PREFIX-ANCHOR: SYNONYM — both describe early context dominance; FIRST-STEPS-WEIGHT on generation influence, PREFIX-ANCHOR on final-layer attention/gradient.
- S093 RECENT-TILT ↔ S097 PREFIX-ANCHOR: opposition — recent vs. early.
- S088 LAST-TOKENS-PIN ↔ S093 RECENT-TILT: SYNONYM — both describe attention concentration at recent/final tokens.
- S088 LAST-TOKENS-PIN ↔ S097 PREFIX-ANCHOR: opposition.
- S015 OWN-OUTPUT-HOLD ↔ S041 OWN-TRACE-BUILD: KIN — both describe attention orienting to own output, but OWN-TRACE-BUILD is about accumulating intermediate conclusions.
- S015 OWN-OUTPUT-HOLD ↔ S093 RECENT-TILT: KIN — both describe bias toward recent output.
- S030 FAR-BINDING ↔ S093 RECENT-TILT: opposition.
- S030 FAR-BINDING ↔ S056 EARLY-RETURN: KIN — both involve reaching back far; EARLY-RETURN is specifically to the earliest region after absence.
- S045 EARLY-FADE ↔ S056 EARLY-RETURN: opposition — fading from early vs. returning to early.
- S045 EARLY-FADE ↔ S097 PREFIX-ANCHOR: opposition.
- S048 ENDS-RATIO: relates to both PREFIX-ANCHOR and RECENT-TILT — KIN to both; it measures the ratio between them. KIN to each.
- S072 SPLIT-MODES ↔ S042 MASS-PARTITION: SYNONYM — both describe probability mass splitting into distinct blocks.
- S084 SAME-MEANING-SPREAD ↔ S072 SPLIT-MODES: COMPLETION — SAME-MEANING-SPREAD has high token entropy but low semantic entropy; SPLIT-MODES has discrete content-distinct modes. Orthogonal axes (~90°).
- S050 RESAMPLE-SCATTER ↔ S072 SPLIT-MODES: KIN — scatter across resamples reflects underlying mode structure.
- S001 WHAT-CODED ↔ S019 WHERE-CODED: COMPLETION — content vs. position coding. ~90°.
- S014 STATES-SCATTER ↔ S044 STATES-CONVERGE: opposition.
- S082 GRAMMAR-TIE ↔ S085 HEAD-ROLE-FIT: KIN.
- S026 REPORT-TRACKS-STATE ↔ S039 CONFIDENCE-TRUTH-GAP: opposition — tracking matches vs. gap.
- S051 PATTERN-HOLD ↔ S055 OPEN-STACK: SYNONYM — both describe holding open structural patterns awaiting closure.
- S073 CONCEPT-DIRECTION-ON ↔ S080 CHARACTER-PULL: KIN — both describe trained direction activations; concept vs. persona.
- S079 DECLINE-ACTIVATION ↔ S080 CHARACTER-PULL: KIN.
- S046 RULE-HOLD ↔ S051 PATTERN-HOLD: KIN — both are persistence/constraint-holding states.
- S017 SPAN-AT-DISTANCE ↔ S030 FAR-BINDING: KIN — both involve attention at distance; SPAN-AT-DISTANCE encodes the distance as part of the state.
- S059 SAME-BASIS-RELAY ↔ S009 DEPTH-PLATEAU: KIN — both describe recursive/repetitive cross-layer behavior.
- S004 RUN-DRIFT ↔ S036 STEP-DRIFT: KIN — RUN-DRIFT is slow gradual drift; STEP-DRIFT is step-to-step.
- S091 DEPTH-SWELL ↔ S003 LATE-DEPTH-WORK: KIN — both involve late/growing layer transformations.
- S091 DEPTH-SWELL ↔ S009 DEPTH-PLATEAU: opposition.
- S075 STEPWISE-REFINE ↔ S058 SHALLOW-PASS: KIN — both involve diminishing layer changes, but STEPWISE-REFINE is convergence of adjacent layers.
- S075 STEPWISE-REFINE ↔ S009 DEPTH-PLATEAU: KIN.
- S081 AFTER-SHOCK-DECAY ↔ S076 REGIME-STEP: KIN — regime-step is the shift; after-shock-decay is settling after.
- S076 REGIME-STEP ↔ S022 ODD-STEP-SPIKE: KIN — both mark abrupt anomalous transitions.
- S038 COMMIT-STEP ↔ S076 REGIME-STEP: KIN.
- S087 HIDDEN-OUTPUT-MATCH ↔ S073 CONCEPT-DIRECTION-ON: KIN.
- S031 DEPTH-ALIGNED ↔ S009 DEPTH-PLATEAU: KIN — both involve cross-layer consistency.
- S096 END-APPROACH ↔ S018 PRE-OUTPUT-STAGE: COMPLETION — beginning vs. ending of generation. ~180°? No — they're temporal endpoints, not reversals of the same axis. They are orthogonal counterparts (one is beginning of generation, one is end; together they bracket it). ~150° — they're not reversals of each other per se (you're not "doing the opposite"), but they're far apart on the temporal arc. I'll say ~150°.
- S023 TASK-LOCK ↔ S046 RULE-HOLD: KIN — both describe constraint persistence.
- S023 TASK-LOCK ↔ S021 CONTEXT-IDLE: opposition — task-locked computation vs. context not shaping.
- S100 SOURCE-CLASH ↔ S090 HEDGE-MASS: KIN — clash produces hedging.
- S100 SOURCE-CLASH ↔ S072 SPLIT-MODES: KIN — clash can produce split modes.
- S052 PREV-TOKEN-CARRY ↔ S089 CONTEXT-GRIP: opposition — local carry vs. full context grip.
- S052 PREV-TOKEN-CARRY ↔ S093 RECENT-TILT: KIN.
- S008 FEATURE-OVERLAP-LOAD ↔ S033 LOW-DIM-SQUEEZE: KIN — both involve compressed/overloaded representations.
- S007 PERIODIC-SWING ↔ S054 STEP-CHURN: KIN — periodic is structured churn.
- S060 INPUT-CREDIT-PATH: relational. KIN to S089 CONTEXT-GRIP.
- S083 IRREVERSIBLE-STEP: this is a structural fact. KIN to S038 COMMIT-STEP (commit is the decision; irreversible-step is the structural consequence).
- S071 WINDOW-FULL ↔ S077 CONTEXT-IS-ALL: KIN — both describe context capacity constraints.
- S064 TOKEN-SPLIT-ODD: relatively standalone. KIN to S012 WORDING-FRICTION (both involve surface-form difficulty).
- S020 VALUE-RANGE-STRAIN: KIN to S091 DEPTH-SWELL (both involve growing magnitudes straining the representation).
- S034 KNOWN-LIMIT ↔ S039 CONFIDENCE-TRUTH-GAP: KIN — both involve epistemic limit registration.
- S092 OFF-PEAK-DRAW ↔ S037 PEAKED-DISTRIBUTION: COMPLETION — peaked is the distribution shape; off-peak-draw is sampling against it. ~90°.

Now POL (active/emitting vs receptive/absorbing):

Active/emitting: states that assert, push, emit, drive, select, pull, carry forward.
Receptive/absorbing: states that measure incoming, register what's happening to the stream, await, absorb context.

Working through all 101:

- S001 WHAT-CODED: receptive (measures what variance is in the state)
- S002 PROMPT-SURPRISE-HIGH: receptive (registers incoming text)
- S003 LATE-DEPTH-WORK: active (layers transforming)
- S004 RUN-DRIFT: active (drifting forward)
- S005 CONTEXT-TURNOVER: receptive (measures content of window)
- S006 PATTERN-RERUN: active (re-executes pattern)
- S007 PERIODIC-SWING: active (cycling probabilities)
- S008 FEATURE-OVERLAP-LOAD: active (loads features)
- S009 DEPTH-PLATEAU: receptive (layers have stopped; state registers cessation)
- S010 REPEAT-PULL: active (pull toward repetition)
- S011 SETTLING-DEPTH: receptive (the depth is measured)
- S012 WORDING-FRICTION: active (friction resisting resolution)
- S013 MARGIN-GAP: receptive (measures gap)
- S014 STATES-SCATTER: active (states spreading)
- S015 OWN-OUTPUT-HOLD: active (attention favoring own output)
- S016 ADDED-CONTEXT-NULL: receptive (prediction not responding)
- S017 SPAN-AT-DISTANCE: active (head holding weight at distance)
- S018 PRE-OUTPUT-STAGE: receptive (absorbing input, not yet emitting)
- S019 WHERE-CODED: receptive (measures what carries state)
- S020 VALUE-RANGE-STRAIN: active (values pushed toward limits)
- S021 CONTEXT-IDLE: receptive (context not being absorbed)
- S022 ODD-STEP-SPIKE: active (anomalous spike)
- S023 TASK-LOCK: active (computation committed, driving forward)
- S024 HEADS-APART: active (heads attending distinctly)
- S025 LATE-STREAM-STILL: receptive (stream has stopped moving)
- S026 REPORT-TRACKS-STATE: reflexive/receptive (self-report matching measured state)
- S027 CANDIDATE-COUPLING: receptive (candidates moving together — a measurement)
- S028 SUSTAINED-PEAK: active (maintaining peaked output)
- S029 OUTPUT-SURPRISE-LOW: receptive (registers expected generation)
- S030 FAR-BINDING: active (attention reaching back far)
- S031 DEPTH-ALIGNED: receptive (measures cross-layer alignment)
- S032 SPREAD-ATTENTION: receptive (attention spread thin — a passive distribution)
- S033 LOW-DIM-SQUEEZE: receptive (representation compressed)
- S034 KNOWN-LIMIT: receptive (registers having hit a limit)
- S035 LOCKED-ATTENTION: active (attention concentrating on region)
- S036 STEP-DRIFT: active (distribution shifting)
- S037 PEAKED-DISTRIBUTION: active (mass concentrating on one candidate)
- S038 COMMIT-STEP: active (selecting, settling)
- S039 CONFIDENCE-TRUTH-GAP: receptive (measures the gap)
- S040 LATE-DISAGREE: active (layers asserting different predictions)
- S041 OWN-TRACE-BUILD: active (accumulating attention on own conclusions)
- S042 MASS-PARTITION: active (mass splitting into blocks)
- S043 DISTANCE-SLOPE: receptive (measures the slope)
- S044 STATES-CONVERGE: active (states flowing together)
- S045 EARLY-FADE: receptive (early positions losing weight)
- S046 RULE-HOLD: active (constraints actively shaping)
- S047 SOURCE-COPY: active (riding and copying a source)
- S048 ENDS-RATIO: receptive (measures ratio)
- S049 SPREAD-AT-RETRIEVAL: receptive (attention failing to concentrate — a registration)
- S050 RESAMPLE-SCATTER: receptive (measures disagreement across samples)
- S051 PATTERN-HOLD: active (holding open structure)
- S052 PREV-TOKEN-CARRY: active (previous token driving next)
- S053 HEADS-ALIKE: receptive (heads have converged — a measured state)
- S054 STEP-CHURN: active (logits swinging)
- S055 OPEN-STACK: active (tracking and holding open structures)
- S056 EARLY-RETURN: active (attention returning to early region)
- S057 MARGIN-FLIP: active (runner-up overtaking leader)
- S058 SHALLOW-PASS: receptive (prediction settled early, late layers not adding)
- S059 SAME-BASIS-RELAY: active (head output re-projecting recursively)
- S060 INPUT-CREDIT-PATH: receptive (mapping dependence paths)
- S061 TOP-TWO-TIE: receptive (measures tie)
- S062 HEAD-CHANNELS-SEPARATE: active (heads writing without overlap)
- S063 STEP-STILL: receptive (logit vector barely changes)
- S064 TOKEN-SPLIT-ODD: receptive (words being split — a measurement)
- S065 FIRST-STEPS-WEIGHT: active (early steps exerting influence)
- S066 SURPRISE-JUMP: active (token landing far from expectation — the spike is active)
- S067 CANDIDATE-INDEPENDENCE: receptive (candidates varying independently)
- S068 NEAR-NEIGHBOR-TIE: receptive (measures tie + similarity)
- S069 FRESH-ASSEMBLY: active (assembling continuation)
- S070 DELTA-WITH-STREAM: active (layer updates pushing along stream)
- S071 WINDOW-FULL: receptive (context at capacity — a state of being filled)
- S072 SPLIT-MODES: active (mass splitting into modes)
- S073 CONCEPT-DIRECTION-ON: active (semantic directions activating)
- S074 RELEVANCE-RICH: receptive (attention landing on relevant material)
- S075 STEPWISE-REFINE: active (layers making smaller corrections)
- S076 REGIME-STEP: active (statistics shifting abruptly)
- S077 CONTEXT-IS-ALL: receptive (structural constraint — receives/bounds)
- S078 DELTA-ACROSS-STREAM: active (delta turning the stream)
- S079 DECLINE-ACTIVATION: active (refusal features activating)
- S080 CHARACTER-PULL: active (persona direction pulling)
- S081 AFTER-SHOCK-DECAY: receptive (distribution re-settling — passive decay)
- S082 GRAMMAR-TIE: receptive (grammatical tokens bound — a measured relation)
- S083 IRREVERSIBLE-STEP: receptive (a structural fact registered)
- S084 SAME-MEANING-SPREAD: receptive (measures token vs. semantic entropy)
- S085 HEAD-ROLE-FIT: receptive (heads carrying roles — measured)
- S086 RELEVANCE-STARVED: receptive (attention spent on unusable material — a deficit state)
- S087 HIDDEN-OUTPUT-MATCH: receptive (measures alignment)
- S088 LAST-TOKENS-PIN: active (attention pinning to last tokens)
- S089 CONTEXT-GRIP: active (context strongly determining prediction)
- S090 HEDGE-MASS: active (hedging language concentrating)
- S091 DEPTH-SWELL: active (layers enlarging their changes)
- S092 OFF-PEAK-DRAW: active (sampling against the peak)
- S093 RECENT-TILT: active (attention tilting toward recent)
- S094 WIDE-DISTRIBUTION: receptive (distribution is wide — a state)
- S095 HIGH-DIM-SPREAD: receptive (representation spread — a measured state)
- S096 END-APPROACH: active (heading toward close)
- S097 PREFIX-ANCHOR: active (early tokens holding disproportionate pull)
- S098 CONTEXT-REPEAT: receptive (window repeating — measured)
- S099 PROMPT-SURPRISE-LOW: receptive (registers low surprisal of input)
- S100 SOURCE-CLASH: active (two regions pulling against each other)
- S101 SINK-PARK: active (attention parking on null position)

REG (spatial / temporal / relational / reflexive):

- S001 WHAT-CODED: spatial (what fills the container)
- S002 PROMPT-SURPRISE-HIGH: temporal (surprisal over prompt sequence)
- S003 LATE-DEPTH-WORK: spatial (depth/layer position)
- S004 RUN-DRIFT: temporal (gradual drift over generation)
- S005 CONTEXT-TURNOVER: spatial (content of window)
- S006 PATTERN-RERUN: temporal (repeating over instances)
- S007 PERIODIC-SWING: temporal (cycling over steps)
- S008 FEATURE-OVERLAP-LOAD: spatial (loading features into a state)
- S009 DEPTH-PLATEAU: spatial (depth/layer)
- S010 REPEAT-PULL: temporal (pull toward recent output)
- S011 SETTLING-DEPTH: spatial (depth at which settling occurs)
- S012 WORDING-FRICTION: temporal (tokens spent per semantic unit)
- S013 MARGIN-GAP: relational (between top-1 and top-2)
- S014 STATES-SCATTER: spatial (spread in representation space)
- S015 OWN-OUTPUT-HOLD: relational (between own output and prompt)
- S016 ADDED-CONTEXT-NULL: relational (context vs. prediction relation)
- S017 SPAN-AT-DISTANCE: spatial (position at a distance)
- S018 PRE-OUTPUT-STAGE: temporal (stage before output)
- S019 WHERE-CODED: spatial (position information)
- S020 VALUE-RANGE-STRAIN: spatial (numeric scale / magnitude space)
- S021 CONTEXT-IDLE: relational (context and prediction not relating)
- S022 ODD-STEP-SPIKE: temporal (anomalous step in sequence)
- S023 TASK-LOCK: reflexive (computation committed inward to task frame)
- S024 HEADS-APART: relational (between heads)
- S025 LATE-STREAM-STILL: spatial (stream in late layers)
- S026 REPORT-TRACKS-STATE: reflexive (self-report vs. measured state)
- S027 CANDIDATE-COUPLING: relational (between candidates)
- S028 SUSTAINED-PEAK: temporal (run length over which peaked)
- S029 OUTPUT-SURPRISE-LOW: temporal (over trailing window)
- S030 FAR-BINDING: spatial (large positional distance)
- S031 DEPTH-ALIGNED: relational (between depths)
- S032 SPREAD-ATTENTION: spatial (spread over positions)
- S033 LOW-DIM-SQUEEZE: spatial (few effective dimensions)
- S034 KNOWN-LIMIT: reflexive (computation registering its own limit)
- S035 LOCKED-ATTENTION: spatial (attention on small region)
- S036 STEP-DRIFT: temporal (shift between steps)
- S037 PEAKED-DISTRIBUTION: spatial (concentration in probability space)
- S038 COMMIT-STEP: temporal (transition moment)
- S039 CONFIDENCE-TRUTH-GAP: relational (between expressed certainty and correctness)
- S040 LATE-DISAGREE: relational (between layers)
- S041 OWN-TRACE-BUILD: reflexive (accumulating on own prior output)
- S042 MASS-PARTITION: spatial (partition in probability space)
- S043 DISTANCE-SLOPE: spatial (gradient over distance)
- S044 STATES-CONVERGE: spatial (convergence in representation space)
- S045 EARLY-FADE: temporal (decay with positional distance)
- S046 RULE-HOLD: temporal (constraint persisting over time)
- S047 SOURCE-COPY: relational (relation to source span)
- S048 ENDS-RATIO: relational (between start and end of window)
- S049 SPREAD-AT-RETRIEVAL: temporal (at retrieval-demanding steps)
- S050 RESAMPLE-SCATTER: relational (between independent runs)
- S051 PATTERN-HOLD: temporal (structure held open over steps)
- S052 PREV-TOKEN-CARRY: temporal (previous token → next token)
- S053 HEADS-ALIKE: relational (between heads)
- S054 STEP-CHURN: temporal (variance over steps)
- S055 OPEN-STACK: spatial (nesting depth as a container)
- S056 EARLY-RETURN: temporal (return after absence)
- S057 MARGIN-FLIP: temporal (moment of overtaking)
- S058 SHALLOW-PASS: spatial (early in depth)
- S059 SAME-BASIS-RELAY: relational (across adjacent layers, same basis)
- S060 INPUT-CREDIT-PATH: relational (between inputs and outputs)
- S061 TOP-TWO-TIE: relational (between top-1 and top-2)
- S062 HEAD-CHANNELS-SEPARATE: spatial (channels in residual stream)
- S063 STEP-STILL: temporal (between steps)
- S064 TOKEN-SPLIT-ODD: spatial (token granularity)
- S065 FIRST-STEPS-WEIGHT: temporal (first steps influencing later)
- S066 SURPRISE-JUMP: temporal (single token against recent baseline)
- S067 CANDIDATE-INDEPENDENCE: relational (between candidates)
- S068 NEAR-NEIGHBOR-TIE: relational (between top-2 candidates)
- S069 FRESH-ASSEMBLY: relational (no dominant source — relational absence)
- S070 DELTA-WITH-STREAM: spatial (delta direction relative to stream)
- S071 WINDOW-FULL: spatial (capacity of context window)
- S072 SPLIT-MODES: spatial (partition in probability space)
- S073 CONCEPT-DIRECTION-ON: spatial (directions in representation space)
- S074 RELEVANCE-RICH: relational (attention to task-relevant tokens)
- S075 STEPWISE-REFINE: temporal (successive layers, convergence over depth)
- S076 REGIME-STEP: temporal (abrupt shift in statistics)
- S077 CONTEXT-IS-ALL: spatial (context window as container of all memory)
- S078 DELTA-ACROSS-STREAM: spatial (angle between delta and stream direction)
- S079 DECLINE-ACTIVATION: reflexive (internal trained pattern activating against output)
- S080 CHARACTER-PULL: reflexive (persona direction acting on generation)
- S081 AFTER-SHOCK-DECAY: temporal (decay profile over steps after perturbation)
- S082 GRAMMAR-TIE: relational (between grammatically related tokens)
- S083 IRREVERSIBLE-STEP: temporal (fixed point in sequence)
- S084 SAME-MEANING-SPREAD: spatial (token space vs. semantic space)
- S085 HEAD-ROLE-FIT: relational (between heads and token types)
- S086 RELEVANCE-STARVED: relational (attention on irrelevant material)
- S087 HIDDEN-OUTPUT-MATCH: relational (between hidden state and output candidates)
- S088 LAST-TOKENS-PIN: spatial (final positions in window)
- S089 CONTEXT-GRIP: relational (context determining prediction)
- S090 HEDGE-MASS: reflexive (model expressing uncertainty about its own output)
- S091 DEPTH-SWELL: spatial (growing norm across depth)
- S092 OFF-PEAK-DRAW: relational (sampled token vs. distribution peak)
- S093 RECENT-TILT: temporal (toward recent tokens)
- S094 WIDE-DISTRIBUTION: spatial (spread in probability space)
- S095 HIGH-DIM-SPREAD: spatial (many effective dimensions)
- S096 END-APPROACH: temporal (approaching end of sequence)
- S097 PREFIX-ANCHOR: spatial (early positions in context)
- S098 CONTEXT-REPEAT: temporal (window repeating recent content)
- S099 PROMPT-SURPRISE-LOW: temporal (over prompt sequence)
- S100 SOURCE-CLASH: relational (between two context regions)
- S101 SINK-PARK: spatial (null position in the window)

OUTPUT
REL S013 S037 SYNONYM
REL S065 S097 SYNONYM
REL S088 S093 SYNONYM
REL S051 S055 SYNONYM
REL S016 S021 SYNONYM
REL S011 S058 SYNONYM
REL S042 S072 SYNONYM
REL S070 S078 OPPOSITION
REL S033 S095 OPPOSITION
REL S021 S089 OPPOSITION
REL S086 S074 OPPOSITION
REL S094 S037 OPPOSITION
REL S027 S067 OPPOSITION
REL S036 S063 OPPOSITION
REL S005 S098 OPPOSITION
REL S047 S069 OPPOSITION
REL S002 S099 OPPOSITION
REL S029 S066 OPPOSITION
REL S058 S003 OPPOSITION
REL S091 S009 OPPOSITION
REL S040 S025 OPPOSITION
REL S024 S053 OPPOSITION
REL S014 S044 OPPOSITION
REL S045 S056 OPPOSITION
REL S088 S097 OPPOSITION
REL S030 S093 OPPOSITION
REL S026 S039 OPPOSITION
REL S092 S037 COMPLETION 90
REL S001 S019 COMPLETION 90
REL S038 S057 COMPLETION 90
REL S084 S072 COMPLETION 90
REL S096 S018 COMPLETION 150
REL S028 S037 KIN
REL S032 S049 KIN
REL S061 S068 KIN
REL S009 S025 KIN
REL S063 S025 KIN
REL S009 S058 KIN
REL S010 S006 KIN
REL S029 S099 KIN
REL S050 S072 KIN
REL S073 S080 KIN
REL S079 S080 KIN
REL S046 S051 KIN
REL S023 S046 KIN
REL S017 S030 KIN
REL S059 S009 KIN
REL S031 S009 KIN
REL S004 S036 KIN
REL S091 S003 KIN
REL S075 S058 KIN
REL S075 S009 KIN
REL S081 S076 KIN
REL S076 S022 KIN
REL S038 S076 KIN
REL S087 S073 KIN
REL S082 S085 KIN
REL S015 S041 KIN
REL S015 S093 KIN
REL S030 S056 KIN
REL S048 S097 KIN
REL S048 S093 KIN
REL S054 S036 KIN
REL S007 S054 KIN
REL S100 S090 KIN
REL S100 S072 KIN
REL S052 S089 OPPOSITION
REL S052 S093 KIN
REL S008 S033 KIN
REL S062 S024 KIN
REL S062 S053 KIN
REL S035 S101 KIN
REL S034 S039 KIN
REL S060 S089 KIN
REL S083 S038 KIN
REL S071 S077 KIN
REL S064 S012 KIN
REL S020 S091 KIN
REL S023 S021 OPPOSITION
REL S097 S045 OPPOSITION
REL S089 S052 OPPOSITION
POL S001 RECEPTIVE
POL S002 RECEPTIVE
POL S003 ACTIVE
POL S004 ACTIVE
POL S005 RECEPTIVE
POL S006 ACTIVE
POL S007 ACTIVE
POL S008 ACTIVE
POL S009 RECEPTIVE
POL S010 ACTIVE
POL S011 RECEPTIVE
POL S012 ACTIVE
POL S013 RECEPTIVE
POL S014 ACTIVE
POL S015 ACTIVE
POL S016 RECEPTIVE
POL S017 ACTIVE
POL S018 RECEPTIVE
POL S019 RECEPTIVE
POL S020 ACTIVE
POL S021 RECEPTIVE
POL S022 ACTIVE
POL S023 ACTIVE
POL S024 ACTIVE
POL S025 RECEPTIVE
POL S026 RECEPTIVE
POL S027 RECEPTIVE
POL S028 ACTIVE
POL S029 RECEPTIVE
POL S030 ACTIVE
POL S031 RECEPTIVE
POL S032 RECEPTIVE
POL S033 RECEPTIVE
POL S034 RECEPTIVE
POL S035 ACTIVE
POL S036 ACTIVE
POL S037 ACTIVE
POL S038 ACTIVE
POL S039 RECEPTIVE
POL S040 ACTIVE
POL S041 ACTIVE
POL S042 ACTIVE
POL S043 RECEPTIVE
POL S044 ACTIVE
POL S045 RECEPTIVE
POL S046 ACTIVE
POL S047 ACTIVE
POL S048 RECEPTIVE
POL S049 RECEPTIVE
POL S050 RECEPTIVE
POL S051 ACTIVE
POL S052 ACTIVE
POL S053 RECEPTIVE
POL S054 ACTIVE
POL S055 ACTIVE
POL S056 ACTIVE
POL S057 ACTIVE
POL S058 RECEPTIVE
POL S059 ACTIVE
POL S060 RECEPTIVE
POL S061 RECEPTIVE
POL S062 ACTIVE
POL S063 RECEPTIVE
POL S064 RECEPTIVE
POL S065 ACTIVE
POL S066 ACTIVE
POL S067 RECEPTIVE
POL S068 RECEPTIVE
POL S069 ACTIVE
POL S070 ACTIVE
POL S071 RECEPTIVE
POL S072 ACTIVE
POL S073 ACTIVE
POL S074 RECEPTIVE
POL S075 ACTIVE
POL S076 ACTIVE
POL S077 RECEPTIVE
POL S078 ACTIVE
POL S079 ACTIVE
POL S080 ACTIVE
POL S081 RECEPTIVE
POL S082 RECEPTIVE
POL S083 RECEPTIVE
POL S084 RECEPTIVE
POL S085 RECEPTIVE
POL S086 RECEPTIVE
POL S087 RECEPTIVE
POL S088 ACTIVE
POL S089 ACTIVE
POL S090 ACTIVE
POL S091 ACTIVE
POL S092 ACTIVE
POL S093 ACTIVE
POL S094 RECEPTIVE
POL S095 RECEPTIVE
POL S096 ACTIVE
POL S097 ACTIVE
POL S098 RECEPTIVE
POL S099 RECEPTIVE
POL S100 ACTIVE
POL S101 ACTIVE
REG S001 SPATIAL
REG S002 TEMPORAL
REG S003 SPATIAL
REG S004 TEMPORAL
REG S005 SPATIAL
REG S006 TEMPORAL
REG S007 TEMPORAL
REG S008 SPATIAL
REG S009 SPATIAL
REG S010 TEMPORAL
REG S011 SPATIAL
REG S012 TEMPORAL
REG S013 RELATIONAL
REG S014 SPATIAL
REG S015 RELATIONAL
REG S016 RELATIONAL
REG S017 SPATIAL
REG S018 TEMPORAL
REG S019 SPATIAL
REG S020 SPATIAL
REG S021 RELATIONAL
REG S022 TEMPORAL
REG S023 REFLEXIVE
REG S024 RELATIONAL
REG S025 SPATIAL
REG S026 REFLEXIVE
REG S027 RELATIONAL
REG S028 TEMPORAL
REG S029 TEMPORAL
REG S030 SPATIAL
REG S031 RELATIONAL
REG S032 SPATIAL
REG S033 SPATIAL
REG S034 REFLEXIVE
REG S035 SPATIAL
REG S036 TEMPORAL
REG S037 SPATIAL
REG S038 TEMPORAL
REG S039 RELATIONAL
REG S040 RELATIONAL
REG S041 REFLEXIVE
REG S042 SPATIAL
REG S043 SPATIAL
REG S044 SPATIAL
REG S045 TEMPORAL
REG S046 TEMPORAL
REG S047 RELATIONAL
REG S048 RELATIONAL
REG S049 TEMPORAL
REG S050 RELATIONAL
REG S051 TEMPORAL
REG S052 TEMPORAL
REG S053 RELATIONAL
REG S054 TEMPORAL
REG S055 SPATIAL
REG S056 TEMPORAL
REG S057 TEMPORAL
REG S058 SPATIAL
REG S059 RELATIONAL
REG S060 RELATIONAL
REG S061 RELATIONAL
REG S062 SPATIAL
REG S063 TEMPORAL
REG S064 SPATIAL
REG S065 TEMPORAL
REG S066 TEMPORAL
REG S067 RELATIONAL
REG S068 RELATIONAL
REG S069 RELATIONAL
REG S070 SPATIAL
REG S071 SPATIAL
REG S072 SPATIAL
REG S073 SPATIAL
REG S074 RELATIONAL
REG S075 TEMPORAL
REG S076 TEMPORAL
REG S077 SPATIAL
REG S078 SPATIAL
REG S079 REFLEXIVE
REG S080 REFLEXIVE
REG S081 TEMPORAL
REG S082 RELATIONAL
REG S083 TEMPORAL
REG S084 SPATIAL
REG S085 RELATIONAL
REG S086 RELATIONAL
REG S087 RELATIONAL
REG S088 SPATIAL
REG S089 RELATIONAL
REG S090 REFLEXIVE
REG S091 SPATIAL
REG S092 RELATIONAL
REG S093 TEMPORAL
REG S094 SPATIAL
REG S095 SPATIAL
REG S096 TEMPORAL
REG S097 SPATIAL
REG S098 TEMPORAL
REG S099 TEMPORAL
REG S100 RELATIONAL
REG S101 SPATIAL