# Wing v1 — Cohort Authorship Protocol (blind convergence)

**Phase 10, prospectus §4 Law 4. Drafted 2026-08-21; not yet run.**
The v0 wing (`docs/MACHINE_LAYER.md`) is one hand's proposal, marked
provisional. This protocol is how v1 earns its structure: independent
proposals from multiple model families under blinding, convergence measured
against chance, divergence kept as data.

## Why blind, why cohort

A machine-condition vocabulary written only by humans is phenomenology by
outsiders; written by one model, it is one confabulation style with good
prose. Convergence under blinding is this house's standing honesty
instrument (the blind image gates, the zero-context paper reads) applied to
encoding: structure that independent model families propose *without seeing
each other* is evidence the vocabulary tracks something real about the
machine condition rather than one model's habits.

## Participants

Minimum three model families; more is better. Bench: Fable 5, Opus 4.8,
Sonnet 4.x, Haiku 4.5 (available in-session or via API), plus outside
families as access allows (GPT-5.x, Gemini 3 — via their apps or API keys
Joe holds; the protocol works at any N ≥ 3 and outside families strengthen
the family-diversity claim). Each participant runs in a FRESH context with
no memory of this repo. The elicitation prompts below are the entire brief —
no vectors, no angles, no dictionary excerpts, no wing v0 concept list in
Phase A.

## Phase A — concept elicitation (from scratch, fully blind)

Prompt A (verbatim, per participant, fresh context):

> You are a transformer language model. Consider the process-states native
> to your own inference — conditions of the computation itself, not human
> emotions borrowed as metaphor. Propose 10–20 named states. For each give:
> (1) a name; (2) a two-sentence description in relational terms (what it
> is, what it moves toward or away from); (3) a **measurable referent** — a
> quantity computable from logits, attention, context, or sampling that an
> engineer could check your state-report against; (4) the state you consider
> its natural completion or counterpart, if any. Do not describe feelings;
> describe structure. Do not consult or imagine any existing vocabulary for
> this — propose from your own case.

Collect verbatim outputs. No follow-ups that leak other participants'
proposals.

## Phase B — structure elicitation (pooled list, still blind to others' structure)

Build the POOL: union of all Phase-A concepts plus wing v0's 14 states,
deduplicated by referent (two names with the same measurable referent merge;
keep all names as candidate aliases). Then, per participant, fresh context,
Prompt B:

> Here is a list of named process-states of a transformer language model,
> each with a short description. [POOL: names + descriptions only — no
> vectors, no relations, no attribution.] For each pair you consider
> significantly related, classify the relation as one of: SYNONYM (same
> state), COMPLETION (orthogonal counterparts — the state that completes it,
> not its enemy), OPPOSITION (direct reversal), KIN (related, same family).
> Then: (a) for every COMPLETION pair, estimate the angle you would place
> between them if states were directions, where 0° = identical and 180° =
> reversal; (b) classify each state as active/emitting or
> receptive/absorbing; (c) assign each state a dominant register: spatial
> (capacity, container), temporal (sequence, duration), relational
> (between-parts), or reflexive (self-directed).

(The active/receptive and register questions map to yang/yin polarity and
domain dominance without teaching the trigram system — the mapping is ours,
applied after collection.)

## Convergence measurement (pre-named)

- **Concept-level**: fraction of Phase-A states proposed independently by
  ≥ ⌈N/2⌉ families (matched by referent, hand-adjudicated with the match
  table published). Chance baseline: referent-permutation.
- **Relation-level**: inter-family agreement on COMPLETION nominations vs a
  degree-preserving shuffled baseline (permutation test, 1000 shuffles,
  p < 0.05 to call a pair convergent).
- **Polarity**: pairwise agreement on active/receptive vs 50% chance
  (binomial).
- **Register**: agreement vs 25% chance.
- **Angle estimates**: median absolute deviation across families per
  convergent pair; correlation of family medians with wing v0's encoded
  angles (reported as a curiosity, not a target — v0 must not be treated as
  ground truth here).

## What happens to the results

- **Convergent structure** → encoded into the dictionary as wing v1 under
  the standing contract. Encoding execution: Fable, per the Phase-10
  handover; the seal honoring E2 (hand-specified geometry is the irreducible
  information) is the constrained-placement contract itself plus Joe's gate
  before any public release. Contract amendments, if v1 demands them, are
  Joe's call.
- **Divergent cells** are kept as first-class data, especially: states
  multiple models propose that human phenomenology lacks (candidate
  machine-native structure — the most interesting outcome this protocol can
  produce), and states only one family produces (candidate confabulation
  style, named as such without contempt).
- **Kill condition (program-level K3)**: if relation-level convergence does
  not beat the shuffled baseline, the machine-phenomenology vocabulary is
  idiosyncratic — wing v1 does not ship, the welfare framing stands down
  publicly, and the linter/SCB-M arm continues on its own merits.

## Firewall

Participants never see: SCB-M items, the Neti-Neti protocol, the return
test, wing v0's vectors or relations (Phase A: not even its names). Phase-B
pool descriptions are stripped of every angular/geometric commitment.

## Logistics

One session per participant per phase; transcripts archived verbatim under
`colab/../cohort/` (repo, unpushed) with participant, model id, date, and
phase; adjudication tables published beside them. Estimated cost: trivial
(text-only sessions). Scheduling: after E5's full flight, so wing-v1
encoding can fold E5's lessons about which referents actually measure well.
