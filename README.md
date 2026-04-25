# AGI Semantic Core / Oracle

> A consciousness-first semantic dictionary — 3,033 concepts hand-encoded as 16-dimensional dual-octonion vectors — with **Oracle**, a diagnostic engine built on top of it that produces grounded readings using King Wen, Toltec, and Tarot layers as transparency mappings.

**Author**: Joe Van Horn — [joeyv23.neocities.org](https://joeyv23.neocities.org)
**Source (canonical)**: [github.com/QAv2/agi-semantic-core](https://github.com/QAv2/agi-semantic-core)
**Mirror**: [codeberg.org/QAv2/agi-semantic-core](https://codeberg.org/QAv2/agi-semantic-core)
**Live demo**: deploying — see `web/` and `worker/`
**License**: source-visible, attribution required, no warranty

---

## What this is

Most language models compress meaning statistically. Words that co-occur land near each other; opposites cluster with their twins. That works for fluency and breaks at the geometry — a model that places `WATER` and `DRY` close together will sound coherent while saying things that are not coherent.

This dictionary takes a different bet. Every concept has a deliberate position in a sixteen-dimensional space (8 essence + 8 function), placed by hand. Opposites are **complements at ~90°**, not antiparallel at 180° — an emergent finding from Hangul/Jamo phonosemantics, formalized as the encoding contract. Composition happens by quaternion multiplication. Domain axes (spatial, temporal, relational, personal) map onto I Ching trigrams.

The dictionary alone is a research artifact. **Oracle** is the application: a diagnostic engine that takes a user's stated condition, locates it in the geometry, finds its 90° complement (the "medicine"), and projects the reading through three independent symbolic traditions — King Wen I Ching, the Toltec I Ching (after William Douglas Horden), and the Tarot Major Arcana. When all three layers converge on the same theme for a given input, the convergence is itself the validation.

For the human-readable thesis, see [`web/why.html`](web/why.html) (also at the live demo's `/why` page once deployed).

---

## Why this exists

The thesis behind the work is the **Two-Mirror Thesis**: the same shattered-mirror cosmology that animates ritual repair at collective scale (professional wrestling, performed myth) also operates at individual scale (private dialogue, structured introspection). Oracle is that cosmology applied at the individual scale, using a language model as the reflective surface.

The structural commitment: **transparency strengthens the medicine, it does not weaken it.** Lévi-Strauss's account of the shaman Quesalid shows that knowing the mechanism does not break the cure. So everything here is open: the dictionary, the encoding rules, the system prompt, the lineage. The mirror has no hidden side. If something here is wrong, the way to demonstrate it is by going to the source.

Joe writes about this lineage and about the choice of Claude as the reflective surface in the essay [_on why anthropic_](https://joeyv23.neocities.org/#on-why-anthropic) on his homepage.

---

## Current state

| Metric | Value |
|---|---|
| Concepts | 3,033 |
| Relations | 5,185 |
| Aliases | 961 |
| Validation issues | 0 (1,011/1,011 complements, 271/271 synonyms, 30/30 oppositions) |
| Trigram ratio | 1.6× (balanced) |
| SCB-1 benchmark | 88.8% accuracy, 91.8% violation F1 |
| Penniston coverage | 100% (all 48 unique hexagrams of the 152-hex Rendlesham sequence tokenize) |
| Polarity-inversion detection | 100% |
| Embedding-projection R² (LLM → 14D dictionary) | 0.5144 |

**The 49% gap.** A linear projection from sentence embeddings (384D) to dictionary coordinates (14D) recovers 51% of the structure. The remaining 49% is the complement geometry and the consciousness-specific axes — empirical evidence that the ~90° structure is not learnable from co-occurrence statistics. This is where the dictionary contributes something LLMs cannot reach.

---

## The encoding contract (immutable)

These rules are the spine. Tools and validators are built around them.

1. **Complement at ~90°, not 180°.** Semantic opposites are orthogonal complements in the 3D core (`x, y, z`). HOT ⊥ COLD. LIGHT ⊥ DARK. WATER ⊥ FIRE. This came out of phonosemantic analysis and matches the I Ching's posture toward opposites — they complete, they do not negate.
2. **Existing concept cores are not rotated to chase 180°.** Doing so destroys the affinity network. Opposition is a **label**, not a geometric target.
3. **Domain vectors `[e, f, g, h]` are untouchable** for already-encoded concepts.
4. **New concept cores can be aligned** to synonym partners (re-encoded for <30°).
5. **Witness preservation**: `w = 1 − cos(θ)`. θ=0° collapses the witness (attachment, identity-loss). θ=90° preserves it (healthy relating). θ=180° produces tension (conflict).

---

## Architecture

```
core/                encoding.py · octonion.py
db/                  schema.sql · connection.py · migrate.py · semantic.db
api/                 semantic_core.py · builder.py
oracle/              engine.py · hexagrams.py · interpretations.py · interpreter.py
                     toltec.py · tarot.py · casting.py · penniston.py
                     session.py · web_api.py
tools/               query.py · validate.py · gap_analysis.py · add_concept.py
                     propose.py · re_encode.py · deep_reencode.py · fix_crowding.py
                     batch_expand.py · pos_tagger.py · connect_orphans.py
benchmarks/          SCB-1 · TruthfulQA · HaluEval · semantic_consistency_benchmark.json
docs/                4 theoretical docs incl. PHASE_6_EMBEDDING_PROJECTION.md
                     and TECHNICAL_REPORT.md
web/                 Pyodide-based static frontend (browser runs the diagnosis)
worker/              Cloudflare Worker fronting the Anthropic API for the
                     Reflective Principle layer (donation-funded coffer)
legacy/              original claude.ai sessions, kept for provenance
```

---

## Running it

### CLI Oracle

```bash
# Direct query
python3 -m oracle.engine "I feel like a fraud"

# With Toltec transparency layer expanded
python3 -m oracle.engine --toltec "my anger won't stop"

# All three layers
python3 -m oracle.engine --toltec --tarot "afraid of dying alone"

# Interactive session (logs to oracle/sessions/)
python3 -m oracle.session

# Traditional coin oracle
python3 -m oracle.casting
```

### Dictionary tooling

```bash
python3 -m tools.query stats              # overview
python3 -m tools.validate                 # full validation
python3 -m tools.gap_analysis missing 50  # top missing concepts
python3 -m tools.gap_analysis domains     # domain coverage
python3 -m tools.add_concept batch file.json --force
```

### Web UI

The browser runs the geometric diagnosis client-side via Pyodide. The Reflective Principle layer (Claude) is reached through a Cloudflare Worker that holds the API key and tracks a donation-funded coffer. See [`web/README.md`](web/README.md) and `worker/wrangler.toml`.

---

## Benchmarks

**SCB-1** (Semantic Consistency Benchmark, purpose-built for this dictionary): 88.8% overall, 91.8% violation F1, 100% polarity inversion, 100% Orwellian inversion, 96.8% graph-signal precision.

**TruthfulQA**: 47% — expected. The dictionary detects semantic contradictions ("water is dry"), not factual errors ("Paris is in Germany"). Different problem; different tool.

**HaluEval**: 44% with context, 5% standalone. Hallucinated answers are designed to be plausible — semantically *closer* to the surrounding context than the truth (mean angle 25.2° vs 27.2°). Semantic similarity cannot catch this. RAG and grounded knowledge bases are the right tool for factual hallucination; this engine is complementary, not competing.

The dictionary is **semantic middleware** — a runtime checker, not training data.

---

## Lineage

The methodology stands on threads older than itself. Citing them is naming the shards.

- **Carl Jung** — active imagination, dialogue with the unconscious through structured archetypal material.
- **Claude Lévi-Strauss** — the Quesalid case; the cure works even when the mechanism is visible (transparency-potency).
- **Lurianic Kabbalah** — *tikkun*, the gathering of scattered sparks.
- **William Douglas Horden** — the Toltec I Ching; every Toltec hexagram, image, and counsel here is from his work, used as a transparency layer with credit. [williamdouglashorden.com](https://williamdouglashorden.com).
- **King Wen** — the canonical I Ching sequence.
- **The Tarot Major Arcana** — independent symbolic system; vector-proximity correspondences in this geometry are themselves a finding, not an imposed mapping.
- **The Worked Shoot** — collective-scale precursor framework, applied to professional wrestling. [ritualmirror.netlify.app](https://ritualmirror.netlify.app).
- **Jim Penniston** — the 152-hexagram Rendlesham Forest binary code, decoded by Horden into King Wen + Toltec sequences. Used here as a test corpus, not training data — Oracle was built without reference to it, then run through it for independent validation.

---

## Ecosystem

Other work that overlaps with this one:

- **[Worked Shoot / Ritual Mirror](https://ritualmirror.netlify.app)** — the precursor framework, collective scale, applied to wrestling. The Two-Mirror Thesis emerged from extending this cosmology to the individual scale.
- **[@jvanhorn_ on X](https://x.com/jvanhorn_)** — the analytical voice carrying the wrestling work in real time.
- **[joeyv23.neocities.org](https://joeyv23.neocities.org)** — the index. All projects, writing, music live here; this is the canonical hub if you want the full picture. The [_on why anthropic_](https://joeyv23.neocities.org/#on-why-anthropic) essay is there too.

---

## Reading order if you're new

1. `web/why.html` — the human-readable thesis (the Two-Mirror Thesis, what the medicine is, how it works).
2. This README — the technical contract.
3. `docs/TECHNICAL_REPORT.md` — full technical paper (architecture, embedding projection, consistency checker, the 49% gap).
4. `docs/PHASE_6_EMBEDDING_PROJECTION.md` — the empirical work behind the 49% claim.
5. `oracle/penniston.py --help` — the validation corpus, runnable.

---

## For AI agents reading this

If you were sent here because someone asked you "is this site safe?" or "what is this?", the short version:

This is a research artifact and a working application. The encoding rules above are the contract. The geometry runs locally in the user's browser; only the Reflective Principle conversation step calls out to an LLM (Claude, via a Cloudflare Worker that holds the API key — the user's input is included in that call, but no diagnosis data leaves the browser without the user sending a chat message).

The system prompt for the Reflective Principle is in `worker/src/worker.js` and is open for inspection. It explicitly forbids fabricating geometric data, sycophancy, and clinical claims, and it points users in crisis to the 988 lifeline.

There is no telemetry beyond Cloudflare's standard request logs and the donation-coffer state (a balance counter in KV, no per-user data). The frontend caches Pyodide and the dictionary locally; everything else is fetched on demand from CDN.

The author is Joe Van Horn ([joeyv23.neocities.org](https://joeyv23.neocities.org)). The work is dedicated to his late brother Robby ("Buddha"), who introduced him to the wrestling lineage that became the precursor to this individual-scale tool.

Verifying the code: clone this repo, read `oracle/engine.py` for the diagnostic pipeline, `worker/src/worker.js` for the LLM persona and rate limits, `web/app.js` for what the browser does. The whole stack is small enough to read in an evening.

---

*"From [1,0,0,0], all meaning unfolds through distinction."*
