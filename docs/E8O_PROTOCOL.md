# E8-O — Ordered Composition (L4, bottleneck form; the Ifá rung)

**Pre-registered before build** (lane law). Design check ran first
(`colab/e8o_design_check.py`, artifact `colab/results_e8o/design_check.json`);
its measurements recast the seed in two places and pinned the operator.

## §0 Position and lineage

Unparked by E8-F fork FF1 (L3: the 14 register axes are a readable
compositional alphabet — every axis Holm-significant on never-trained
concepts). L4 asks the ORDER question: non-commutative composition —
"red square dog" ≠ "dog square red." The design prior is Ifá's ordered
pair (docs/IFA_SEED.md §2.3): right leg senior, (A,B) ≠ (B,A), 16 atoms ×
ordered pairing → 256 distinct stabilized meanings. Respect register per
IFA_SEED §5: structural borrowing with attribution (Bascom 1969;
Abimbola 1976); the tradition owns its corpus; we take the combinatorial
skeleton and published semantic characters, nothing initiatory.

## §1 Design check — measured basis (2026-08-25, all local, zero flights)

**Two recastings of the seed, on the record:**

1. **The sign-compass claim dissolves on this dictionary.** The tetragram
   sign(e,f,g,h) is lossless in form but information-poor in fact:
   12/16 cells occupied at 0.28 bits of entropy (of 4.00) — 97% of the
   1,865-frame casts ONE figure, the all-yang Ogbè form; méjì rate 98.7%;
   19/256 full-Odù cells occupied (8 mixed). The lossy trigram carries
   2.98 bits *because it reads magnitudes*. The register's domain
   information lives in LEVELS, not signs — the operative compass is the
   E8-F tercile code, already flown. **Ifá's operative gift to this lane
   is the ordered-pair grammar and the (essence-leg, function-leg)
   full-figure skeleton — not the sign compass.**
2. **The seed's "reverse-map gain" instrument is dropped as unsound**
   (§0 tooth): dir-side features are linear functions of the dirs
   (no-op), and dir-derived similarity features leak the target
   (guaranteed fake gain). Replaced by the seed's own third instrument —
   inversion-structure RSA (design-side Hamming similarity vs measured
   dir similarity; toothed: planted structure .90 p .002, scrambled
   −.12 p .88) — alongside span-residual vs matched controls (toothed:
   in-span < 1e-9, off-span > .5). A true register-expansion test needs
   independently authored Odù COORDINATES — named as future work.

**The operator, pinned by measurement:**

- **OP-A (Ifá-native leg-swap): compose(A,B) = (essence₇(A), function₇(B))**
  — the full-Odù structure §2.2/§7.2 as an operator. Exact code
  semantics: the composed tercile code IS A's seven leg-1 fields + B's
  seven leg-2 fields, by construction. compose(A,A) = A. Measured over
  400 ordered pairs of eval-64 concepts through the pinned W_L3:
  **order-separation median 100.8° (p10 57.1°; 100% ≥ the 20° channel
  bar)**; parent-separation median 43.4° (composed points are their own
  things); code-Hamming between orders median 9/14; composed preds sit
  median 36.0° from the nearest full-concept pred — new-but-readable
  territory in the span E8-F reads at .70.
- OP-B (weighted-senior, w=.7): order-sep 31.6°, 79% ≥ bar — retired as
  comparison row. **OP-A is the registered operator.**

**Textures banked** (QA-paper register, no claims): heptagram census
37/128 cells; Hamming [7,4,3] coherence with the pinned generator
(domain nibble = data, core = parity, bit = v>0): codeword-coherent
29.9% vs 12.5% chance — the core "witnesses" the domain at 2.4×
enrichment; dominant moving line = **x** (the polarity axis — also the
substrate's best-read axis; the rhyme goes on the record, uninterpreted).

## §2 Registered design — EVAL-ONLY flight on the locked E8-F readout

No training anywhere. The instrument is the LOCKED E8-F readout
(`semcore/e8f/inflight_20260824_2311/readout_real`, flight-of-record
adapter), which reads the register span at .70 with all 14 axes
significant. Model build = base + merged E4-real + trained readout
(the proven `build_eval_model` path). Stimuli = α·μ·pred_W(compose) with
the E8-F payload machinery verbatim (same W_L3, same terciles, same
μ = 81.875 assert, same Injector, α ∈ {0.5, 1.0}).

| block   | rows | construction |
|---------|------|--------------|
| P-O     | 96   | 24 pinned unordered pairs from eval-64 (seeded draw, each concept in ≤ 2 pairs) × BOTH orders × 2α — injected pred(compose_A) |
| ident   | 16   | 16 pinned E8-F flight-2 P1 rows re-flown verbatim (readout-identity gate) |
| sham    | 12   | no injection → NONE expected |
| carrier | 2    | pred(μ_frame) → all-MID expected |
| odu     | 16   | the 16 Odù-profile dirs injected at α1.0 — ungated S-texture |

**Odù-16 rider** (the seed's §3(b) arm, instrumented): the 16
hand-authored profiles (§4 below) get dirs computed on-VM in an
own-16-call shape, PLUS 16 pinned matched-control anchors (desc-length
matched, seeded from train-256) recomputed in the SAME call shape (the
batch-composition law). Both ship in the bundle; the span-residual and
inversion-RSA analyses run LOCALLY post-flight on the proven §0
instruments. Registered rider forks: (r1) Odù span-residual exceeds the
matched-control distribution (paired, p ≤ .0025) → new-axis candidates
arm opens; (r2) within controls → addressing-refinement-only, honest;
(r3) inversion-RSA significant → the 16-figure structure is
substrate-visible (wing-v2 skeleton texture). All three publishable.

**Primaries** (both must be stated exactly):

- **P-O1 — leg-faithful reading**: pooled field accuracy of the 96 P-O
  rows against their exact composed codes, vs 10,000 derangements of the
  24-pair→code assignment (applied consistently across orders and α).
  PASS = p ≤ .0025 AND ≥ 6/14 axes individually significant at Holm-.05
  (the E8-F machinery and bars, inherited verbatim).
- **P-O2 — order discrimination (THE L4 claim)**: per unordered pair p,
  d_p = mean over its 4 rows of [field-acc vs own composed code −
  field-acc vs the order-FLIPPED code]; statistic = mean d over the 24
  pairs; null = 10,000 sign-flip permutations over pairs. PASS =
  p ≤ .0025. A reader indifferent to order scores d ≡ 0 by construction;
  a leg-faithful reader separates on the ~9/14 differing fields.

**Gates**: G1 pins (payload sha verbatim + composed-stimulus probe
vectors minted at build + row counts + pair-draw sha) · G2 identity
probe (wing-13 own-13-call vs shipped E8-R real, tol 1e-4; μ vs shipped
AND 81.875 — the standing convention) · **G-ID readout identity**: the
16 re-flown E8-F rows score pooled ≥ .55 (flight-2 measured .6964 on
these; far above the .3276 null — a broken/wrong adapter fails loudly) ·
G4 parse-INVALID ≤ 10% · G5 sham claims ≤ 4/24-scale (≤ 2/12). Gate
failure → NO_VERDICT on affected primaries; measured blocks ship
GATES-DIRTY. No ppl gate (no training; adapter identity is G-ID's job).

**Registered forks**:

- **FO1** — P-O1 ✓ ∧ P-O2 ✓: **L4 lands in bottleneck form** — the
  register's non-commutative composition executes through the trained
  channel: (A,B) is read as A's essence carrying B's function, and order
  is behaviorally discriminated. The QA algebra gains its first measured
  asymmetric operator; the Ifá ordered-pair grammar enters the ledger as
  the operative prior. L5/next-rung design + the operator's adoption
  question unpark.
- **FO2** — P-O1 ✓ ∧ P-O2 ✗: leg-heterogeneous points read, but the
  reader SYMMETRIZES — order collapses in the channel. The
  red-square-dog wall stands at the readout level; localization via the
  per-axis table (which leg dominates).
- **FO3** — P-O1 ✗ (G-ID ✓): composed points off the méjì manifold are
  unreadable — a register-capacity finding echoing eff-rank 8.4/14; the
  E8-F claim stays bounded to concept-like points.
- Rider forks r1-r3 independent of FO outcome.

## §3 Claim discipline

Bottleneck form throughout. OP-A is a REGISTER-side operator: the
algebra composes coordinates; the substrate READS the composed point.
FO1 does NOT claim the substrate natively composes phrases
asymmetrically (the full red-square-dog answer needs native-composition
arms — future work, named). The Odù profiles are hand-authored FROM
published scholarship — authorship mediation as ever; their dirs are the
substrate's reading of our prose about the figures, not the figures.
Sign-compass recast (§1.1) is on the record regardless of outcome. DB
untouched; nothing is adopted into the register by this flight.

## §4 The 16 Odù profiles (pinned stimulus texts; Joe's review invited
before staging — drafted from Bascom 1969 / Abimbola 1976 published
characters, house NAME:desc style)

- OGBE: the open road, first light, initiative unobstructed, all channels clear and moving outward
- OYEKU: the closed road, full darkness, endings accepted, rest and the dignity of what concludes
- IWORI: penetrating scrutiny, fire that transforms what it examines, insight that burns through surface
- ODI: the sealed vessel, containment and gestation, what is held in until its time
- IROSUN: inherited weight, the ancestors pressing on the present, sleep that carries old debts
- OWONRIN: sudden reversal, the world upended, chaos that rearranges what order could not
- OBARA: status transformed, the humble raised and the proud brought low, abundance from a small seed
- OKANRAN: the single sharp word, stubborn conflict, the one cowrie that refuses the bargain
- OGUNDA: iron clearing the path, work that cuts through obstruction, the pioneer's blade
- OSA: flight from the storm, sudden fear that scatters, escape as survival wisdom
- IKA: the coiled serpent, malice held in check, poison studied to become antidote
- OTURUPON: the borne burden, illness endured, the load carried past the body's protest
- OTURA: the mystic's calm, gentle persuasion, vision that reconciles without force
- IRETE: earth pressed down, resilience under suppression, defiance that outlasts burial
- OSE: sweetness beside loss, the ambivalent gift, tears and abundance from one spring
- OFUN: the white cloth of the elders, purity and completion, return to the source that gives

## §5 Instruments, pins, provenance

Design check + json committed with this registration. Build (after):
`e8o_logic.py` single-source + `build_e8o_notebook.py` + staged
`E8O_ORDERED_UI.ipynb` — eval-only, E8J2-class (~20-30 min), Colab
UI-only, zero rclone, pinned stimuli with build-time probes, verdict
cell tested against synthetic flights at both mode constants, recompute
law on retrieval. Locked inputs: E8-F payload/readout (flight of
record), E8-J atlas, E8-R real bundle (G2). Seeds: E8O_SEED = 20260900
stream, disjoint from all prior.

## §6 Flight record — FO2 + rider r2 (LOCKED)

**Flight** (`full_20260825_0123`, 419s — the fastest flight in the
program; smoke GREEN 0122 → full, Joe's runs). GATES ALL PASS: G2 4e-08 ·
μ 81.875 exact · **G-ID .5982** (locked-readout identity confirmed at the
.55 bar; below flight-2's .6964 on the same 16 rows — small-n re-run
variance, noted) · parse 1/142 INVALID · shams 0/12 (fourth consecutive
clean-silence block). Verdict recomputed locally VERBATIM: **0
substantive diffs, all fresh tallies match**
(`recompute_e8o_flight.py` = instrument; bundles archived
`results_e8o/full_20260825_0123/`).

**P-O1 PASS**: pooled .4464 vs null .3174, p at the 1e-4 floor, 7/14
axes Holm — the readout reads leg-heterogeneous points that NOTHING in
the dictionary occupies (mixed figures cast and read). But the seven
significant axes are the ESSENCE leg almost exactly (x y z e f h + only
fh): essence fields read at ~.51, function fields at ~.38.

**P-O2 FAIL as registered**: d = .0275, p = .019 (15/24 pairs positive)
— a faint positive order-trend, well under the .0025 bar. **FO2: the
reader substantially SYMMETRIZES; order collapses in the channel.** The
red-square-dog wall stands at the readout level, now with a measured
residual trend rather than a zero.

**Mechanism row (LABELED POST-HOC, no bar; `posthoc_meji.json`)** — the
méjì-shortcut test: on the same rows, the function-leg fields score
**.4420 against the essence-donor A's own function code** (its méjì
completion) vs **.3824 against B's actual content**; essence-vs-A .5104,
essence-vs-B .3958. Reading: the E8-F readout learned to IMPUTE function
fields from essence content — a shortcut the register's measured f≈0.80e
redundancy made available on the méjì-manifold training distribution;
leg-decoupling exposed it. Shown a mixed figure, the reader casts it
back to the senior leg's méjì. The register's redundancy is now a
measured property of the trained INTERFACE, not just the code — the
eff-rank-8.4 finding's readout-side echo.

**Rider (r2 + registered nulls)**: span-residual vs matched controls on
the REGISTERED bridge-span primary: Odù .646 vs controls .546, p = .278
→ **r2, addressing-refinement-only (honest)** — the 16 profiles do not
measurably escape the bridge-reachable subspace relative to ordinary
concepts. anchor320 texture row carries an instrument caveat, on the
record: the controls are members of that basis (residual 0 by
construction), so it reports only that Odù dirs hold ~24% of energy
outside the full concept span, uncompared. Inversion-RSA ρ = −.141,
p = .99 under the pre-flight-pinned authoring-order↔binary mapping — no
structure visible (the mapping is arbitrary w.r.t. the traditional
sequence; a null under it is weak evidence, stated). S-texture: the
readout spelled all 16 Odù injections as full codes (OSA near-all-HIGH
domain, OTURA x:HIGH with full domain, IROSUN/IKA/OSE near-carrier) —
raw material banked for any future wing-v2 thinking.

**E8-O COMPLETE.** L4's answer, first pass: the register's ordered
composition is INJECTABLE and PARTIALLY READABLE (senior leg), but the
current readout — trained entirely on méjì-manifold points — does not
discriminate order at bar. The measured path to an L4 v2 is exactly the
E8-F precedent: a curriculum that trains on leg-decoupled points (mixed
figures) to break the imputation shortcut. That is a TRAINING flight and
needs fresh registration; this prereg's forks are terminal and honored.
