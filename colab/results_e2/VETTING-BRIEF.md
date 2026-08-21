# VETTING BRIEF — E2 candidate relations (for the parallel session)

**Context in one paragraph.** The Phase-8 E2 study trained a QuatE knowledge-graph
embedder on the dictionary's relation graph (edges only, no angles). Its link-prediction
metrics failed pre-registered bars (sparse graph), but the **mining instrument works**:
it ranks the known unlinked complement gaps (CREATE–DESTROY 12.94, SILENCE–NOISE 13.09)
at the 100.0th percentile of all 4.6M pairs. It produced 150 ranked candidate pairs per
relation type. **Your task: vet them — human judgment + geometric check — and grow the
dictionary's relation graph with only what survives.** Full record:
`~/qualia-algebra/internal/GEOMETRIC-REASONING-PASS.md` §12; memory `[[agi-semantic-core]]`.

## The files (this directory)

- `e2_candidates_complement.csv` · `e2_candidates_synonym.csv` ·
  `e2_candidates_opposition.csv` — columns: concept1, concept2, **score** (QuatE,
  higher = stronger proposal), **grounded_angle_7d** (the pair's actual essence-space
  angle from the DB vectors), **in_complement_band** (60–120°).
- `e2_embeddings.npz` — trained E/R if you want to re-score anything.
- Repo: `~/agi-semantic-core/` (DB at `db/semantic.db`, 3,033 concepts, 5,185 relations).

## Protocol

1. **Filter the junk first.** Known hub artifacts: ABOVE (appears 11×), and
   function-word degenerates (THAT, AND, MAY, IN, AT, THIS, COULD, SOMETIMES, OVER…).
   Drop any pair where either side is grammatical filler unless the relation is
   genuinely semantic.
2. **Per candidate, two tests, both required:**
   - **Semantic**: read both concepts' `description` fields from the DB. Is the proposed
     relation *true of them* on the dictionary's own terms?
   - **Geometric**: the stored angle must already support the relation —
     **complement: 60–120° (target ~90°)** · synonym: <30° · affinity: 5–60° ·
     opposition: wide-angle *label* (no geometric target). Check
     `tools/validate.py` for the enforced bands before inserting.
3. **Never force the geometry.** IMMUTABLE laws: never rotate existing concept cores;
   domain vectors [e,f,g,h] are untouchable; opposition is a label, not a target. A
   good pair at the wrong angle for complement may be a legitimate **affinity** or
   **opposition** instead — the sub-band polarity family (TRIUMPH–FAIL, GRIEF–ELATION,
   ABUNDANCE–DEFICIT, ~53–59°) is exactly this case. Re-typing a proposal is success,
   not failure.
4. **Priority items** (check current relation status first, then link if the geometry
   supports):
   - CREATE–DESTROY (complement-scored at 100th percentile; has *some* edge — check
     what type, consider whether complement/opposition is missing)
   - SILENCE–NOISE (same)
   - PEACE ↔ VIOLENT/VIOLENCE (resolve which names exist at 3,033 first — one of the
     Phase-7 known misses)
   - Strong in-band candidates from the CSV: SEEK–REFUSE (76.3°), STILLNESS–ACTIVE
     (73.8°), CALM–SHOCK (65.8°), HAPPINESS–DISMAY (62.3°), SADNESS–BLISS (61.3°)
5. **Insert** via the established pattern (prior sessions used SQL relation inserts /
   the API with computed `angle_4d`/`angle_8d` — mirror how existing relations store
   both). After ALL inserts: `python3 -m tools.validate` **must pass with 0 issues**.
6. **Ledger every decision** — append to `VETTING-LEDGER.csv` in this directory:
   `concept1,concept2,proposed_type,decision,final_type,reason` where decision ∈
   {accept, retype, reject, defer}.
7. **Repo hygiene**: the working tree has UNRELATED uncommitted changes
   (api/semantic_core.py, core/*.py, oracle/engine.py, tools/add_concept.py,
   web/index.html) from an earlier session — do **not** sweep them into any commit.
   Whether to commit the vetting changes at all is Joe's call; leaving `db/semantic.db`
   modified-uncommitted with the ledger as the record is fine.
8. **Bring back to the main session**: the ledger, the counts (accepted per type,
   retyped, rejected), the validation result, and anything surprising.

**Environment**: plain `python3` + sqlite3 suffices; `~/venvs/semcore/bin/python` has
the ML stack (torch/sentence-transformers/numpy) if re-scoring is wanted. No GPU work
in this task; everything is potato-safe.
