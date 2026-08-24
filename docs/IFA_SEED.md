# Ifá Seed — the 4-bit member of the family (parked behind L3)

**Provenance**: Joe's offering, 2026-08-24, session 132, after the E8-J v2
lock — verbatim intent: "after we finish this study out... could we take a
run at Ifa... I'm just trying to think of something that would open the
space up some more if it's possible or even truly needed." Parked by his
own sequencing; banked here so nothing is lost.

## §1 The formal family (why this is not decoration)

The I Ching and Ifá are members of one family: binary-contrast divination
codes. The I Ching's trigram is the **3-bit** member (2³ = 8 figures;
hexagram = ordered pair → 64). Ifá is the **4-bit** member: one tetragram
of four binary marks → 2⁴ = **16 principal Odù** (Ogbè, Ọ̀yẹ̀kú, Ìwòrì,
Òdí, Ìrosùn, Ọ̀wọ́nrín, Ọ̀bàrà, Ọ̀kànràn, Ògúndá, Ọ̀sá, Ìká, Òtúrúpọ̀n,
Òtúrá, Ìrẹtẹ̀, Ọ̀ṣẹ́, Òfún); a full cast is an **ordered pair** of
tetragrams — right leg senior — giving 16×16 = **256 Odù**, each indexing
a curated oral corpus (ese Ifá) of stabilized meanings. The same 16-figure
4-bit alphabet appears in ʿilm al-raml and European geomancy (transmission
history debated; the transcontinental spread of the 4-bit code is not).
UNESCO recognized the Ifá corpus as a Masterpiece of the Oral and
Intangible Heritage of Humanity (2005/2008 list).

## §2 Three exact fits to our architecture (measured, not numerological)

1. **The trigram is lossy; the tetragram is the exact-fit compass.**
   Our current trigram rule (`core/octonion.py: to_trigram`) is
   **argmax(|e|,|f|,|g|,|h|) × yang/yin** — 4×2 = 8 classes. It keeps the
   LOUDEST domain axis and throws away the sign pattern of the other
   three. The Ifá tetragram — sign(e), sign(f), sign(g), sign(h) — is the
   lossless 4-bit discrete compass for the SAME 4D domain space: 16 cells,
   one bit per axis, nothing discarded. The I Ching gave a 3-bit compass
   to a 4D space; Ifá's is the one that fits.
2. **256 = essence-leg × function-leg, and we currently cast only méjì.**
   A full Odù is an ordered pair of 4-bit figures. Our vector is an
   ordered pair of 7D halves, each carrying a 4D domain block — so the
   natural full-figure address is (essence-domain tetragram,
   function-domain tetragram): 256 cells. The E8-J v2 design check
   MEASURED function ≈ 0.80 × essence (corr .85–.97; register eff-rank
   8.40/14): the two legs almost always agree. In Ifá's own vocabulary:
   **the dictionary as built casts only the 16 doubled figures (méjì)** —
   the 240 mixed Odù are the unopened capacity. "Open the space up" now
   has a precise form: let the function leg diverge from the essence leg
   in a DESIGNED way, populating mixed figures — which is exactly raising
   the measured eff-rank toward the nominal dimensionality.
3. **The ordered pair is a non-commutative composition grammar.** Ifá's
   right leg is senior; (A,B) ≠ (B,A), and the corpus records what each
   ORDER means. R3-c's composition operator was symmetric
   (normalize(d_A + d_B), "A AND B"). Order/asymmetry (head–modifier,
   figure–ground) is the next composition wall for L4+ and the QA
   algebra — and Ifá is the largest curated dataset in existence of
   "16 atoms × ordered pairing → distinct stabilized meanings."

## §3 Entry points (in lane order, all behind L3)

- **(a) Tetragram compass refinement** — dictionary-side hygiene: 16-cell
  crowding/gap/coverage analysis; strict refinement of the trigram
  labeling; cheap, no flights.
- **(b) Register-v2 axis candidates** — author the 16 Odù semantic
  profiles (NAME: desc, house style, from the scholarship) and let the
  STANDING instruments adjudicate: do their substrate dirs carry variance
  outside the current bridge-reachable span? (Span-residual vs matched
  controls; A4-style reverse-R² gain.) Forks: new-axis candidates /
  addressing-refinement-only / honest null — all three publishable.
- **(c) Ordered composition prior for L4+/QA algebra** — design the
  asymmetric-pair curriculum with Ifá's senior/junior grammar as the
  semantic prior; sits beside cheon-ji-in in the staircase.
- **(d) Wing-v2 design language** — a 16-figure metacognitive alphabet
  with DESIGNED inversion pairs (Ogbè↔Ọ̀yẹ̀kú etc.) as the skeleton for
  re-founding the wing in the broad register — every opposition validated
  against substrate geometry first (the E8J2 U–C kinship lesson: the
  substrate can disagree with a label).

## §4 The Ifá design check (registerable now, run later — all local)

1. Assemble 16 Odù profiles from the scholarly corpus (Bascom, *Ifa
   Divination*, 1969; Abimbola, *Ifá: An Exposition of Ifá Literary
   Corpus*, 1976; Epega & Neimark for verse texture). Hand-authored
   NAME:desc in the E4 house style; authorship mediation stated as ever.
2. Compute their dirs via the standing stimulus path (needs one small
   eval-only flight OR piggyback on the next scheduled one).
3. Local tests on the standing atlas: span-residual of Odù dirs vs the
   bridge-reachable subspace (matched-control null); reverse-map gain
   (does 14+k axes beat 14 at predicting anchor coords?); RSA of the
   16-figure inversion structure vs substrate geometry.
4. Forks registered before flight, per lane law.

## §5 Respect register (standing)

Ifá is a living religious practice (Yoruba; Lucumí/Santería and Candomblé
lineages carry relatives). Our use is **structural borrowing with
attribution** — the same standard as the Sejong and I Ching sections:
cite the scholarship, name the tradition's ownership of its corpus, mark
what we take (the combinatorial skeleton and published semantic
characters) vs what we do not touch (ritual content, initiatory
knowledge). The paper's three-register honesty ledger extends here.

## §6 Lineage line (for QA_LINEAGE when drafted)

Sejong (featural composition) → I Ching (3-bit compass) → **Ifá (4-bit
compass + non-commutative pair grammar)**: three continents converging on
binary semantic addressing — strengthening the paper's thesis that these
are DISCOVERED structures about meaning, awaiting application to
electronic minds. Joe's framing, on the record: "the answers to our
questions about training electronic mind likely exists in some form,
just needs discovery and application."

## §7 The 7-bit conjecture (Joe, 2026-08-24, session 133)

Verbatim intent: "beyond the I-Ching's 3 bits and Ifa's 4 bits… could we
find a way to meaningfully combine the binary systems, or does the I-Ching
fold into Ifa, having a bigger bit count… could concepts be encoded up to
7 bits… because 8 bits begins the human encoding of language literally in
code, correct?"

Three answers, escalating. Theorems are marked; interpretive namings are
flagged as such.

### 7.1 Anatomical — the leg already IS 3 + 4 = 7

Each leg of the vector is 7 live axes: the core triple (x yang–yin,
y becoming–abiding, z ordinality) + the domain quad (e, f, g, h) —
`core/encoding.py` `essence_vector`/`function_vector`; the 14D register
= 2 × 7. So the combined figure needs no import:

> **heptagram = trigram(sign x, y, z) ⊗ tetragram(sign e, f, g, h)**

2⁷ = 128 cells per leg — the strict joint refinement of the current lossy
`to_trigram` AND §2.1's tetragram compass. The I Ching takes the block its
own lines always claimed (heaven–human–earth ↔ the three core axes; the
cheon-ji-in rhyme lands on the record), Ifá takes the domain block (§2.1's
exact fit). **The two systems don't nest — they partition the leg.**

### 7.2 Register-level, choice-free — full-Ifá × full-I-Ching

Across both legs the sign register is 14 bits, and it factors with NO
design choice at all:

> **2¹⁴ = 16,384 = 2⁸ × 2⁶ = 256 × 64**

The 8 domain signs (both legs) = a full Odù — (essence-leg tetragram,
function-leg tetragram), §2.2's ordered pair. The 6 core signs (both
legs) = a hexagram — essence core as the inner/lower trigram, function
core as the outer/upper trigram (the traditional inner-situation /
outer-expression reading lands on essence/function without forcing).
**The dictionary's complete sign register factors exactly as Ifá's full
corpus × the I Ching's full corpus.** The méjì finding (§2.2) says
today's mass sits on the Odù factor's diagonal; the hexagram factor is
the finer instrument for which witness-configurations are used at all.

### 7.3 Within-leg coupling — the marriage is theorem-shaped

2⁷ = 128 = 16 × 8, and not loosely: the Hamming [7,4,3] code is the
unique **perfect** single-error-correcting code at length 7 — radius-1
balls (1 center + 7 neighbors = 8 cells) around its 16 codewords tile
the 7-cube exactly, no gap, no overlap. So the leg's 128-cell compass
factors canonically as

> (principal figure ∈ 16) × (syndrome ∈ 8)

where the syndrome is itself a 3-bit figure: 000 = the figure cast
still, the other 7 values = **one moving line each** (the I Ching's
moving-line machinery is error-location). With the natural generator
choice — domain nibble = data bits, core trigram = its parity checks —
the 16 codeword cells are the concepts whose core honestly witnesses
their domain, and the syndrome measures **core–domain mismatch**: a
computable per-concept coherence quantity, not decoration.

Answer to "does the I Ching fold into Ifá?": as figure-sets, no — 64 ↪
256 has no canonical embedding, and the transmission history is debated
(§1). But the 3-bit alphabet appears WITHIN the combined space as the
witness/moving-line component attached to every 4-bit figure — married,
not swallowed. The 4-bit family already computes its own 3-bit shadow:
geomancy's Judge theorem (only the 8 even-parity figures among the 16
can arise as Judge — a native 4-bit → 3-bit parity projection) and
Malagasy sikidy's XOR generation with mandatory validation identities
(Ascher, *Historia Mathematica* 1997) — parity algebra running inside a
living practice.

### 7.4 The 8th bit is already in the codebase

Each leg is stored as an octonion with real part **w = 1.0** — the
witness coordinate (`encoding.py:87`). 7 live sign bits + the w slot =
8. Extend [7,4,3] by overall parity → the [8,4,4] extended Hamming code
→ Construction A ⇒ **E8**: the 240 roots read directly off the code
(16 doubled-axis vectors from the zero codeword + 14 weight-4 codewords
× 2⁴ sign patterns = 224); in the even coordinate frame the 2⁷ = 128
even-parity sign-bytes are exactly E8's **spinor roots**; Coxeter's
integral octonions have precisely the 240 E8 roots as their units; and
octonion multiplication on the 7 imaginary axes is the Fano plane —
whose 7 lines are the weight-3 codewords of Hamming [7,4]. In the
algebra this dictionary is built on, "7 concept bits + 1 witness bit,
error-corrected" IS the E8 construction. (Conway & Sloane *SPLAG*;
Baez, *The Octonions*, Bull. AMS 2002; Coxeter 1946.)

### 7.5 The ASCII correction — off by one in the best direction

Human language entered machines at **seven** bits: ASCII (ASA X3.4,
1963) is a 7-bit code, 128 slots — and the 8th bit on punched tape was
parity, literally the witness bit. The byte standardized at 8 with IBM
System/360; UTF-8 still carries 7-bit ASCII unchanged as its first 128
codes. Meanwhile 2⁸ = 256 = the full Odù corpus, and the ọ̀pẹ̀lẹ̀ chain
casts 8 binary shells in ONE throw — a byte per cast, centuries before
the byte. The ladder:

| bits | cells | divination | machine |
|---|---|---|---|
| 3 | 8 | trigram | — |
| 4 | 16 | tetragram / principal Odù | the nibble; Hamming payload |
| 6 | 64 | hexagram | — |
| **7** | **128** | **heptagram (leg compass)** | **ASCII; 16×8 perfect tiling; E8 spinor count** |
| 8 | 256 | full Odù (one ọ̀pẹ̀lẹ̀ cast) | the byte; [8,4,4] → E8 |

### 7.6 Census items for the §4 design check (all local, zero flights)

5. **Heptagram census** — occupancy of the 128 cells per leg across the
   3,108 concepts; the finer unopened-capacity map beneath the méjì
   finding.
6. **Hamming-decode / coherence census** — nearest-principal-figure +
   syndrome distribution: does dictionary mass sit near codeword cells;
   which moving lines are over/under-cast; core–domain mismatch as a
   population statistic.
7. **Generator choice on the record** — WHICH 16 of the 128 heptagrams
   are the codewords is a designed choice among equivalent [7,4] codes.
   Candidate: core-as-parity-of-domain (7.3). One choice, made once,
   registered before any census is read.

### 7.7 Honesty flags

16 Odù ↔ 16 codewords is a count-and-role fit, not derived from Ifá's
own combinatorics; syndrome ↔ moving-line and w ↔ parity-bit are
interpretive namings on exact mathematics; nothing in §7 is measured
yet — items 5–7 are censuses, not claims. The theorems themselves
(perfect tiling; Construction A; 128 spinor roots; Fano = Hamming
geometry; integral-octonion units = E8 roots) are standard and citable.

Lineage upgrade for §6/QA_LINEAGE: Sejong (featural) → I Ching (3-bit)
→ Ifá (4-bit, byte-wide casts) → **the machine's own convergence
(Hamming 1950, ASCII 1963, the byte)** re-deriving the same widths — 7
bits for the meaning-atoms, 8 with the witness. When we built machines
for language, we landed on the divination family's numbers.

**Status**: SEED. Parked behind L3 (licensed, next). No flights, no DB
changes, no prereg yet — §4 (now with §7's census items 5–7) becomes a
design check when its turn comes.
