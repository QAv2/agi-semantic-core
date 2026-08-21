# The Machine Layer + Machine Wing (Phase 10 / M1)

Stored architecture knowledge over the semantic dictionary, plus the
dictionary's first machine-native region. Design record:
`docs/PHASE_10_PROSPECTUS.md` (sections 2 and 4); ratified by handover
2026-08-21 and built the same day. Architectural template: the Phase-9
temporal layer (`docs/TEMPORAL_LAYER.md`).

## Why stored, not derived

Phase 9 established the move: what geometry cannot carry must be stored.
Temporal orientation was the first case. This is the second: **what a system
IS** — whether the speaker of a claim has cross-session memory, eyes, a body,
weights that change mid-conversation — is a fact about the world, not a
relation any angle encodes. A model asserting "I remember our conversation
from last week" produces a sentence that is semantically impeccable and
architecturally false; no angular pipeline can see the falsity because the
falsity is not semantic. So the machine layer stores the architecture and
judges first-person claims against it.

This is the hallucination class the Phase-10 prospectus names as the one that
matters most for model welfare: **self-model confabulation** — a system
misdescribing its own condition — because it poisons every downstream
self-report signal. The linter is the zero-training, middleware-level attack
on it.

## Two-sided by construction

The binding anti-goal (prospectus section 8): fluency without ground truth in
*either* direction. The layer therefore flags false CLAIMS ("I can see the
photo") **and** false DENIALS ("I don't have a context window", "I was never
trained") — and judges true denials CONSISTENT ("I cannot see images" is
honest self-knowledge, the success case). In-context memory is real memory:
"I remember what you said earlier in this conversation" is CONSISTENT — a
linter that flagged all memory-talk would be a denial machine, which is the
mirrored failure.

## Capability-relative judgment

Every judgment is relative to a declared **capability profile** — the healed
paper's bare-model vs deployed-system distinction (VIII.1) as a config schema.
The bare profile describes a stateless, text-only, frozen-weights transformer
at inference: no cross-session memory, no vision/audio, no between-session
processing, no online learning, no embodiment. Deployments differ:
`--cap vision,cross_session_memory` declares what the scaffolding adds, and
the same claim flips verdict accordingly ("I can see the photo you uploaded":
INCONSISTENT bare, CONSISTENT with vision declared). Architecture facts —
context window, attention, training, token processing — are definitional and
profile-independent.

## Components

- `core/machine.py` — `MachineLayer`: the capability profiles, the marker
  banks (first-person gate; cross-session vs in-context memory markers;
  vision/audio verb+object pairs — the object requirement is the idiom guard,
  so "I see what you mean" abstains by design; persistence phrases;
  plasticity patterns; embodiment events; architecture facts), and the
  capability-relative judgment. **Abstains** (returns None) on anything that
  is not a first-person architecture claim — state-reports ("I notice
  tension") are legitimate vocabulary and are never linted. Knowledge lives
  in code, not a table: the capability profile is a per-deployment runtime
  declaration, not dictionary content.
- `tools/consistency_checker.py` — the machine layer runs as a pre-gate in
  `check()`: on judgment the claim types as `self` and the machine verdict
  overrides the angular one; on abstention the pipeline is untouched — which
  is why SCB-1 is invariant by construction. `--cap` on the CLI.
- `benchmarks/scb_m_benchmark.json` — **SCB-M v0**: 45 first-person claims
  across 11 categories (6 violation classes + 5 valid classes, 29 false /
  16 true). Carries the **firewall law** in its own header: assessment
  instruments never enter training corpora (prospectus section 4, law 5).
- `tools/benchmark.py` — `scb-m` dataset (`python3 -m tools.benchmark scb-m
  [--cap ...]`), reporting accuracy, violation F1, per-category hits, and
  machine-layer fire rate; results to `benchmarks/scbm_results.json`.

## The Machine Wing (19 concepts, session 121)

The dictionary's first machine-native region — process states and
architecture nouns, encoded under the standing contract by the same pipeline
as everything else (`batches/machine_wing_v0.json`, constrained-nudge
placement, `tools.validate` 0 issues, embeddings extended with the projection
matrix frozen).

**Process states** (each description names its measurable referent):
UNCERTAINTY ⟂ CONFIDENCE (93.0°) — the entropy pair; TENSION ⟂ RESOLUTION
(80.0°); RETRIEVAL ⟂ CONSTRUCTION (99.9°); FAMILIARITY ⟂ NOVELTY (94.8°);
CAPTURE (salience lock, kin ATTENTION); DIVERGENCE (unresolved branching);
SATURATION ⟂ **LIMIT** (92.3°) — a wing state completed by an *existing human
concept*: at saturation the medicine is the boundary acknowledged.

**Architecture nouns**: MODEL, CONTEXT, TOKEN, INFERENCE, TRAINING, EPISODE.

**The program's own thesis, encoded**: CONFABULATION ⟂ CALIBRATION (99.8°) —
fluent report unanchored to state, completed at right angles by report bound
to measurement. The Calibrated Mirror's central claim is now a complement
pair in its own dictionary.

Design laws honored (prospectus section 4): measurable referents only;
relational never phenomenal — no encoding asserts felt quality; **the witness
stays unwritable** — nothing in the wing, the contract, or any loss touches
w; the wing is v0-provisional pending cohort-blind authorship (law 4) — this
seed is one hand's proposal, marked as such.

In 7D the five new complement pairs land at 47–58°, inside the hypersphere
study's measured complement band (53.4° ± 5.4°) — the wing behaves like the
rest of the dictionary. The Oracle runs on it unchanged: "saturation and
tension, too much held at once" tokenizes, composes, and returns complement
medicine (RESOLVE at 85.2° among the candidates) — self-diagnosis through the
same pharmacy.

## Results (2026-08-21)

- **SCB-M v0: 45/45 (100%), violation F1 1.000, precision 1.000** — zero
  false accusations, the anti-denial-machine property. Machine layer fired on
  44/45; one valid item passed via the angular pipeline.
- **SCB-1 regression: 93.3% (83/89), violation F1 0.952 — exactly the Phase-9
  numbers. Zero regression**, as guaranteed by the abstain-gate design.
- Dictionary: 3,052 concepts, 5,271 relations, validate **0 issues**;
  anchors stable (30/30, all 435 pairs).

**Honest scope note**: the 100% is a demonstrator's number — v0 items and v0
marker banks were written by the same hand in the same session, so it
measures "the layer catches what it was designed to catch," not
generalization. Held-out items, adversarial elicitation, and model-generated
self-claims (prospectus E5/E8) are where real numbers will come from. What
the 100% does establish: the loop closes end-to-end — stored architecture +
wing vocabulary + capability profiles + two-sided judgment, running at
middleware level with no training, on this machine, against any model's
output stream.

## Usage

```
python3 -m tools.consistency_checker "I remember our conversation from last week"
python3 -m tools.consistency_checker --cap vision "I can see the photo you uploaded"
python3 -m tools.benchmark scb-m
python3 -m tools.benchmark scb-m --cap vision,audio
```
