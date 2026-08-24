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

**Status**: SEED. Parked behind L3 (licensed, next). No flights, no DB
changes, no prereg yet — §4 becomes a design check when its turn comes.
