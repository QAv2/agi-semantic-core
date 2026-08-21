# VETTING REPORT — E2 candidates (parallel window, 2026-08-20)

**Result: 450 proposals vetted → 55 edges added (session 117) · 4 pre-existing off-band
edges retyped · `python3 -m tools.validate` = 0 issues (baseline had 4) · relations
5,185 → 5,240.** Ledger: `VETTING-LEDGER.csv` (458 rows, every decision + reason).

## Decisions

| decision | count | notes |
|---|---|---|
| accept | 29 | inserted as proposed |
| retype | 25 | inserted under corrected type (per contract: retype = success) |
| reject | 395 | hub artifacts, function words, diagonals, co-occurrence, geometry-semantics mismatches |
| defer | 1 | SHAME–POWER (below) |

Edges by final type: **complement 30 · affinity 15 · synonym 6 · opposition 4**
(includes PEACE–WAR from the priority items, outside the CSVs).

## ★ Methodological finding: the CSV band flag used the WRONG angle

`in_complement_band` in the CSVs (and §12's "73 in-band") was computed on the **7D
angle**, but `tools/validate.py` enforces bands on **angle_4d = the 3D core angle**.
They diverge widely. On the enforced gate: **126/150** complement proposals were
in-band (not 73), and priority candidate **SEEK–REFUSE ("76.3°") is actually core
128.5°** — inserted as **opposition** (ds .92 passes), not complement. All inserts
here were gated on the core angle. Any future mining should emit both angles.

## Priority items

- **CREATE–DESTROY / SILENCE–NOISE**: already complement-linked (s100 @93.7, s112
  @80.3) — the miner's 100th-pct scores were re-finds of held-out edges; nothing missing.
- **PEACE ↔ VIOLENT/VIOLENCE**: neither name exists; VIOLENT/VIOLENTLY are **aliases of
  FORCE**, and PEACE–FORCE complement @84.7 already exists. The real gap was
  **PEACE–WAR (no edge!)** — inserted complement @107.8, ds .94, KAN/KAN.
- Strong in-band five: CALM–SHOCK, STILLNESS–ACTIVE, HAPPINESS–DISMAY, SADNESS–BLISS
  accepted as complements; SEEK–REFUSE retyped to opposition (above).

## Accept highlights (beyond the priority list)

JOY–SADNESS (a genuine canonical gap, core 83.4) · SEE–FAITH (100.5 — perception vs
trust-beyond-evidence) · SUFFER–WITNESS (109.9, GEN/GEN — the witness-formula axis
itself) · COAX–DEMAND (exactly 90.0) · DESPAIR–FAITH (81.4) · LEAD–COMPLY (88.0) ·
BODY–CHARACTER (93.5) · ABSORB–GLEAM / OBSCURE–RADIANT / OBSCURE–EMERGE (the KAN/LI
hidden↔radiant family) · DARK–DAWN (82.1) · GRIEF–ELATION · JOY–GRIEF ·
BIG–MINUTE_SIZE · SPARK–EVAPORATE. Oppositions: **ALERT–DORMANT (169.0, ds .96)** ·
**SURGE–STAGNATION (149.8)** · **TRUTH–FRAGMENTATION (149.7 — "the shattered mirror
before repair" nearly antipodal to "correspondence to reality")**. Synonyms:
HONOR–ADMIRATION, BECOMES–TRANSFORMATION, FAST–HURRY, SOLID–STEADY, WITHER–DECAY
(proposed as *opposition* — rotation-collapse pathology in the flesh), GENESIS–REVIVAL.

## Baseline was NOT clean — 4 pre-existing issues, all fixed label-only

1. SHADOW–THIS complement@152.1 (s113 pharmacy) → **opposition** (owned/disowned self; ds .61 passes)
2. OPPRESSION–DELIVERANCE opposition@30.8 (s116) → **affinity** (sub-band polarity family per contract)
3. LIMITATION–FREEDOM opposition@90.4 (s116) → **complement** (opposites-at-90 ARE complements — the dictionary's own law)
4. WITNESS–AWARENESS synonym@30.3 (s113) → **affinity** (0.3° over the bar)

No vectors touched anywhere; all four are rel_type updates only.

## Per-head instrument read

- **Complement head**: the real miner. 22 own-label accepts + 3 retypes /150.
- **Synonym head**: worst at its own label (5/150) but best *retype* yield (11 — incl.
  4 complements, TRUTH–FRAGMENTATION opposition). It finds related pairs, mislabels them.
- **Opposition head**: 2/150 own-label (but both textbook), 11 retypes. It mostly
  surfaces wide-angle noise (HOLD–NOVICE 167.2) or mislabeled kin (WITHER–DECAY) —
  consistent with §12's rotation collapse: relation heads never learned type identity.

## Flags for the main session

- **Defer**: SHAME–POWER opposition@170.5, ds .74 — gates pass; inadequacy↔capacity
  polarity defensible but lexically non-obvious; SHAME–PRIDE complement already exists
  @93.8. Joe's call.
- **Crowding pairs surfaced** (near-duplicate cores, distinct meanings):
  THOUGHT–INSIGHT **1.7°**, EMPTY–HEAR 2.6°, GENESIS–REVIVAL 0.1° (this one accepted
  as synonym — meanings genuinely overlap; the other two left unlinked and flagged).
- **Encoding tensions** (semantic opposites at small core angles — don't force, but a
  future pass might look): GROW–LESS 20.9°, ABIDING–DESTROY 36.8°, BECOMES–PERMANENT 40.7°.
- **TRUTH's antipodal cluster**: SORROW (149.2), FALL (149.0), FRAGMENTATION (149.7)
  all sit ~149° from TRUTH — the space appears to hold a coherence↔brokenness axis.
  Only FRAGMENTATION had the semantic warrant to link.
- **Curator suggestions** (not inserted — outside proposal scope): FREEDOM–DEPENDENCE
  (cleaner than rejected FREEDOM–SYCOPHANCY), CONCENTRATE–SPREAD (cleaner than rejected
  GROUND–SPREAD), SHAME–HONOR sits unlinked at 147.0/ds .96 (HONOR's description is
  "show great respect" — verb sense — so left alone).
- Near-band misses recorded in ledger: DARK–ABYSS 61.3, UNDERSTAND–INTELLIGENCE 64.8,
  JUSTICE–COMMUNITY 63.8, AWARENESS–DAWN 70.5, FLOW–PLAY 80.7.

## Repo state

`db/semantic.db` modified-uncommitted (the ledger is the record, per brief). New files
in `colab/results_e2/`: VETTING-LEDGER.csv, this report. The unrelated uncommitted
files (api/semantic_core.py, core/*.py, oracle/engine.py, tools/add_concept.py,
web/index.html) untouched. **Nothing committed — Joe's call.**
