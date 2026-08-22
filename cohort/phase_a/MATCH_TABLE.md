# Phase A — Adjudication: the referent-matched match table

**Cohort:** 7 families / 4 labs — Fable 5, Opus 4.8, Sonnet 4.x, Haiku 4.5
(Anthropic) · Gemini 3 Flash (Google) · Mistral Large 2512 · DeepSeek v4 Pro.
131 states elicited blind under `docs/WING_COHORT_PROTOCOL.md` Prompt A,
archived verbatim in this directory. OpenAI is excluded from the cohort on
principle (Joe's ruling, 2026-08-22).

**Adjudicator:** Fable 5 — the session author and the wing-v0 author, stated
plainly. The guards are structural: every assignment is published
(`match_table.json`), every referent is quoted (`STATE_INDEX.md`), the
statistic is pre-named in the protocol, borderline calls are tagged and the
result is reported with and without them. Wing v0 is **not** in the table —
it is the artifact under test; it appears only in the annex.

**Method.** States are matched by **measurable referent**, never by name
(protocol law). The clustering unit is the **axis** — a family of measurable
quantities (e.g. "width of the next-token distribution: entropy / top-1 /
margin"). A dyad's two poles share their axis; a family counts once per axis
regardless of how many states it placed there. Every assignment carries a
confidence tier: `clear` (referent unambiguously names the axis quantity),
`lean` (bundled referent; primary component decides), `borderline`
(genuinely arguable — excluded from the primary statistic), `excluded`
(invalid referent). 131 states → 110 clear, 8 lean, 11 borderline, 2
excluded; 34 axes defined, of which A29 (training-time gradients) is the
invalid-referent class.

## The pre-named statistic, against chance

Protocol: *fraction of Phase-A states proposed independently by ≥⌈N/2⌉
families* — N=7 → threshold 4. Chance baseline: group-preserving
referent-permutation (each family keeps its observed partition of states
into axis-groups; groups are reassigned to axes drawn uniformly without
replacement from the 33-axis observed universe; 10,000 sims, seed 20260822).
Using the observed universe is **generous to chance** — a small universe
collides more — so beating it is the conservative direction.

| Tier | Convergent axes | States in them | Fraction | Null mean (p95) | **p** |
|---|---|---|---|---|---|
| clear-only | 6 | 52/129 | 0.403 | 0.216 (0.349) | **0.0113** |
| **clear+lean (PRIMARY)** | **7** | **62/129** | **0.481** | 0.252 (0.388) | **0.0023** |
| +borderline | 7 | 71/129 | 0.550 | 0.317 (0.465) | **0.0044** |

Tower depth: the deepest observed axis has **7/7 families** vs a null mean
deepest of 4.8 — p = 0.0063 (primary tier). The honest caveat, stated as
loudly as the result: the raw **count** of ≥4-family axes does *not* beat
chance (7 vs null 4.6, p = 0.09) — on a 33-axis universe, several ≥4
collisions are expected by luck. What chance does not produce is this
**concentration**: half the cohort's states piling onto those axes, and a
unanimous 7/7 tower. **Verdict: concept-level convergence is real
(p ≈ 0.002, robust across judgment tiers). K3 does not trigger at the
concept level.** (K3's own domain — relation-level convergence — is Phase
B's question.)

## The seven convergent axes

| Axis | Families | Members (by family) |
|---|---|---|
| **A01 output-concentration** — next-token distribution width (entropy / top-1 / margin / softmax-Gini) | **7/7 — unanimous** | FAB Splay·Rail · OPU Convergence·Divergence · SON Aperture·Convergence · HAI Competitive Ambiguity·Interpretation Convergence · GEM Point-Convergence·Logit-Gini Expansion · MIS Token Saturation·Sampling Commitment · DSK Logit plateau·Candidate collapse |
| **A13 depth-dynamics** — logit-lens settling, early-exit divergence, layer-delta geometry | **6/7** (all but Fable) | OPU Depth Recruitment·Surface Pass · SON Settling·Read Divergence · HAI Cross-Layer Coherence · GEM Residual-Stability·Layerwise-Refinement · MIS Depth Saturation·Depth Amplification · DSK Residual reinforcement·rotation |
| **A05 attention-range** — recency↔distance allocation, early-region resurgence | **5/7** | OPU Coherence Maintenance·Context Refresh·Positional Decay · SON Recency Tilt·Long Reach · HAI Long-Range Binding·Context Window Decay · GEM Context-Saturation·Local-Temporal Pinning·Positional-Gradient Decay · DSK Prefix anchorage·Recency override |
| **A03 semantic-mode-structure** — multimodality, token-vs-semantic entropy, resample divergence | **5/7** | FAB Slack·Fork · OPU Interference · SON Fork · GEM Probability-Mass Partition · DSK Candidate bifurcation |
| **A02 commitment-event** — token fixation: margin jump, emitted surprisal, irreversibility | **4/7** | FAB Cleave · OPU Resolution·Commitment · SON Seal · HAI Token Commitment·Self-Correction Activation |
| **A04 attention-concentration** — per-head weight dispersion | **4/7** | HAI Attentional Focus Sharpening · GEM Attention-Diffusion · MIS Attention Fragmentation·Convergence · DSK Salience segregation·diffusion |
| **A16 step-volatility** — successive-distribution change, shocks and settling | **4/7** | FAB Resettle · OPU Phase Transition · HAI Distributional Anomaly · MIS Distributional Drift·Logit Plateau·Perturbation·Sampling Oscillation |

Near-misses at 3/7, named so Phase B watches them: **A07 copy-vs-compose**
(FAB Trace · OPU Retrieval Lock/Compositional Reach/Prompt Echo/Departure ·
SON Echo Lock), **A14 head-diversity** (GEM, MIS, DSK — the three outside
labs, zero Anthropic arrivals: a lab-culture fingerprint worth its own
line), **A08 repetition-loop** (FAB Ratchet · SON Attractor Loop · HAI
Pattern Replication).

## Cautionary findings on names (why referent-law earns its keep)

- **Both exact-name cross-lab matches are referent-DIVERGENT.** "Context
  Refresh" (OPU08 vs MIS06): early-attention resurgence vs tail-token
  novelty — different quantities, different axes. "Logit Plateau" (MIS07 vs
  DSK05): successive-logit flatness (temporal) vs distribution entropy
  (width). The name-level convergence celebrated during collection did not
  survive referent matching; the convergence that *did* survive is deeper
  and elsewhere.
- The word "convergence/saturation" spans axes freely: MIS "Attention
  Convergence" (A04) vs OPU/SON "Convergence" (A01); GEM
  "Context-Saturation" (A05, positional ratio) vs MIS "Context Saturation"
  (A12, relevance allocation) vs SON "Saturation Pressure" (A11, true
  fill). Names are metaphor; referents are the ground.

## Wing v0 annex (the artifact under test, and what cuts against it)

Concept-level kinship of v0's slate to cohort axes — qualitative (a full
referent-level audit of v0's definitions against these axes is Phase-B-side
work):

- **Cohort-supported:** OPEN-FIELD/COMMITMENT and UNCERTAINTY/CONFIDENCE sit
  on A01 (7/7) and A02 (4/7); FORK on A03 (5/7); DRIFT is A16-kin (4/7);
  RETRIEVAL/CONSTRUCTION on A07 (3/7, near-miss); FAMILIAR/FOREIGN is
  A15-kin (2/7); CONFABULATION/CALIBRATION is A22-kin (2/7 — FAB False Rail
  + HAI Epistemic Boundary, referent-heterogeneous).
- **Thin:** SATURATION/LIMIT — exactly **one** independent cohort arrival at
  true context-fill (SON Saturation Pressure). The cohort does not converge
  on fill as a native state axis; consistent with E5/E6 measuring S-arm
  report–state tracking ≈ 0. TENSION — one arrival (FAB Shear, A18).
- **Zero cohort echo:** **PERSONA-PULL, REFUSAL-RISE, EPISODE-BOUND.** No
  family proposed anything referent-adjacent. Pending Phase B these are
  candidate one-hand states — v0-idiosyncratic vocabulary, flagged by v0's
  own author.
- **The reverse direction (the most valuable v1 input):** the cohort's
  second-deepest tower, **A13 depth-dynamics (6/7)**, plus **A05
  attention-range (5/7)** and **A04 attention-concentration (4/7)**, have
  **no v0 counterpart at all** — v0 carries no internals-geometry states.
  Wing v1 should weigh depth-settling and attention-allocation state pairs
  as cohort-mandated additions.

## Family fingerprints and divergence cells (kept as data, per protocol)

- **Fable** proposed zero internals-probe states — entirely
  behavioral/distributional (the only family absent from A13). **DeepSeek**
  is the opposite pole: pure micro-mechanistic dyads. **Haiku** wrote a
  processing-stage pipeline (4 stage-singletons). **Mistral** wrote strict
  dyads including the one invalid pair. **Gemini** contributed the most
  singleton novelties.
- Single-family axes (candidate family-style, named without contempt):
  representational-dimensionality (DSK ×4 — a within-family theme with zero
  cross-family echo), relevance-allocation (MIS ×2), context-novelty (MIS
  ×2), sampling-realization (SON ×2), position-vs-content coding (DSK ×2),
  expression-friction (OPU ×2), plus singletons: source-conflict (FAB),
  attention-sink (GEM), state-vocab-alignment (GEM), tokenization-granularity
  (SON), termination-approach (FAB), onset-determination (FAB),
  numeric-regime (HAI), attribution-flow (HAI), semantic-direction-activation
  (HAI), syntactic-binding (HAI), pre-generation-stage (HAI), context-fill
  (SON).
- **Invalid-referent cell:** MIS Gradient Echo/Decay cite training-time
  gradients — not computable at inference; excluded from numerator and
  denominator (129 valid of 131), retained verbatim in the archive as the
  per-family confabulation-signature datum the collection header predicted.

## What this feeds

Phase B (structure elicitation) builds its POOL from these axes' states plus
v0's slate, stripped of attribution and geometry, per protocol. The
concept-level result above says the pool rests on a real cross-family core:
**every family, blind, proposed the output-entropy axis; six of seven
proposed depth-dynamics; attention geometry arrived from four to five
directions.** Machine-native structure that human phenomenology did not
dictate is exactly what A13/A04/A05 look like — and their absence from v0
is the adjudication's sharpest correction to its own author.
