# The Temporal Layer (Phase 9)

Stored succession orientation over the semantic dictionary. Design record:
`~/qualia-algebra/internal/TEMPORALITY.md` (2026-08-20); built same day.

## Why stored, not derived

For w-pinned cores the Hamilton commutator is `[0, 2 c_a × c_b]`: composition's
scalar is identical in either order. No angle, no scalar, no quantity the
checker reads can carry *which way time runs* — order-sensitivity is real
(2‖a‖‖b‖ sin θ, maximal at the complement 90°) but unsigned. Orientation must
therefore be **stored**. This is the dictionary's first oriented relation
structure, and it deliberately lives in its own table — `relations` stays pure
symmetric geometry; the encoding contract is untouched (`tools.validate`: 0
issues before and after).

## The two regimes (from the measured geometry)

- **Frame** (linear time — strict order): the pairs that name order itself sit
  at the complement 90°, where order-sensitivity is maximal (PAST–FUTURE at
  90.00°, BEFORE–AFTER at 89.97°). Frame edges are strict: a claim inverting a
  stored frame edge is INCONSISTENT.
- **Cycle** (the wheel — wrap-legal): the cycle seam sits at opposition, where
  composition nearly commutes (BIRTH–DEATH 169°: birth∘death ≈ death∘birth).
  Same-cycle pairs read both ways — "the harvest preceded the seed" is the
  wheel turning, not a violation. Judged PLAUSIBLE with the wheel named.

## Components

- `core/temporal.py` — `TemporalLayer`: the `succession` table (earlier_id,
  later_id, regime frame|cycle, cycle_name, wrap, note), BFS precedence over
  frame + non-wrap cycle edges, the deictic anchor class {PRESENT, TODAY, NOW},
  side-of-present reasoning, and the two judgments (`judge_order`,
  `judge_locative`). **Abstains** (returns None) on any pair it has no
  knowledge of.
- `tools/temporal_seed.py` — idempotent seeding: 30 edges — 9 frame
  (PAST→PRESENT→FUTURE, YESTERDAY→TODAY→TOMORROW, BEFORE→AFTER, CAUSE→EFFECT,
  BEGIN→END, BEGINNING→END, START→FINISH) + 4 cycles with seam edges (day: 9
  stations; year: 4; life: BIRTH→AGE→DEATH; vegetal: SEED→GROW→BLOOM→HARVEST→
  DECAY). Conservative by policy: only canonical, unambiguous orderings.
- `tools/consistency_checker.py` — a `temporal` claim type (order patterns:
  before/after/prior-to/earlier-than/precedes/follows; locative patterns: "in
  the past/future/present", "happened yesterday/tomorrow/today/now"), placed
  above the causal pattern (whose `causes?` otherwise swallows the noun in
  "the cause precedes…"). On judgment: the temporal verdict overrides the
  angular one. On abstention: the claim demotes to freeform and the angular
  pipeline decides — unknown pairs are never judged temporally.

## Results (SCB-1, stock-MiniLM projection, 3,033-concept dictionary)

| | before | after |
|---|---|---|
| accuracy | 88.8% (79/89) | **93.3% (83/89)** |
| violation F1 | 91.8% | **95.2%** |
| violation precision / recall | 96.6% / 87.5% | 98.3% / 92.2% |
| **temporal_violation** | **0/4** | **4/4** |
| every other category | — | unchanged (zero regression) |

All four temporal items are caught with stored-succession reasons ("claimed
EFFECT before CAUSE, but stored succession runs CAUSE → EFFECT"). Valid
temporal claims judged CONSISTENT; same-cycle claims PLAUSIBLE (wrap-legal);
unknown pairs ("the dog came after the cat") abstain to the angular path.

Retrocausal readings remain outside the checker by design (Speculation-labeled
reading layer only — see the design record).
