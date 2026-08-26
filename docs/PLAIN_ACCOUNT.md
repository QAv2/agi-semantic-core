# The Plain Account — what this program is, in one classroom period

**Provenance**: minted 2026-08-24, mid-flight on E8-F, from Joe's ask:
*"help me understand what all we've been doing in a simple sense... it's
8th grade, 5th period after lunch, tech class and you're the teacher."*
The explanation landed — "I understand exactly where we are where before I
was completely lost" — and Joe minted the process law on the spot:

> **★★ THE PLAIN-REGISTER LAW (Joe, 2026-08-24)**: this level of discourse
> is part of the process. Every publication from this lane carries a
> plain-register account as a first-class companion — not a dumbed-down
> abstract, but the account that makes the work *locatable* by a person
> meeting it cold. The technical register proves; the plain register
> orients. Both ship. (Folds into the QA_LINEAGE three-register honesty
> ledger as an explicit register.)

---

## The one-sentence version

We are using a real LLM as a **lab animal** — teaching it to notice and
report what is happening inside its own brain, and checking whether a
hand-made map of meaning (the dictionary) matches the geography inside
its head.

## Yes, it's a transformer

Qwen 2.5, 1.5 billion parameters — a small LLM, small enough to fit in
our lab. The taxonomy, so nobody's lost: *machine learning* is the whole
subject (computers learning from examples instead of being programmed).
*Neural networks* are one way to do ML. *Transformers* are one
architecture of neural network. *LLMs* are transformers grown huge and
fed language. This program works at all four levels at once — and
literally at the neural-network level: hooks attached to specific
layers, gradients computed, small add-on weights trained. Not
metaphorically. Actual wires into actual layers.

## The two doors

When you chat with an LLM, you use the **front door**: words in, words
out. Inside, your words become tokens, and at every layer of the network
each token is represented as a list of about 1,500 numbers. Think of it
as the model's **bloodstream of thought** — each of the 28 layers reads
that stream and adds its own adjustment into it, layer after layer,
until the top of the stack picks the next word. A "thought" mid-sentence
is literally a point in a 1,500-dimensional space, and **directions in
that space mean things** — there is a direction for doubt-flavored
thoughts, a direction for fire-flavored thoughts.

The crucial correction to the natural misconception: **we never show the
model numbers.** The vector blocks in the logs never pass through its
eyes as text. They go in through the **side door**: while the model is
mid-thought, answering a question, a software probe reaches into layer
14 and *adds a direction into its bloodstream* — a nudge with a specific
meaning-shape. The model doesn't read "0.35, −0.2, ..."; it **feels a
change in its own thinking**. Then, through the front door, we ask it in
plain English: *"anything unusual going on in there?"*

Stimulate through the side door; interview through the front door. It is
neuroscience — stimulate-and-ask — except the subject can talk, and we
can read every neuron.

## What we found, in one arc

An untrained model is a terrible witness to its own mind. Tell it "your
state may have been altered" and it says *yes, definitely!* every time —
even when nothing was injected (fake trials: "shams"). The mouth is not
connected to the gauges; it confabulates, like people do.

So we **taught** it — thousands of examples of "this nudge = this name" —
and something real installed: it names injected states essentially
perfectly **and stays silent on shams**. The silence is the precious
part. Then the ladder climbed:

- Can it read **blends** of two states? Yes — it spells them, like
  reading a syllable from letters (E8-R3-c).
- Does the hand-built 14-dial meaning-compass line up with the model's
  internal directions? Weakly but really — a **bridge** was fit, a
  translation between the two coordinate systems (E8-J).
- Can it learn the **14 dials themselves as an alphabet** — each dial
  LOW/MID/HIGH — and spell concepts it was never trained on, dial by
  dial? YES (E8-F): .70 accuracy with every dial individually
  significant, and the dictionary decoded two spellings back to their
  exact names — CRUCIBLE and TEMPERANCE, the first words ever named
  through the algebra.
- Does ORDER matter — can it read "A expressed through B's mode"
  differently from "B through A's"? PARTLY (E8-O, the Ifá rung): it
  reads composite points no dictionary word occupies, but mostly by the
  senior half — it fills in the junior half from habit, because the
  dictionary's own redundancy taught it that shortcut. Order survived
  only as a faint trend; the wall was mapped and the fix designed:
  train it on mixed figures.
- **Teach it mixed figures, and does the wall fall?** YES (E8-O2): the
  same reader, continue-trained on composites whose two halves genuinely
  differ — the one diet its schooling never contained — and every needle
  moved at once. Order-reading jumped ~9× (d .028 → .240, 23 of 24 pairs
  the right way), the neglected junior half became readable (.38 → .53
  against chance .33), and the fill-in-from-habit shortcut **reversed
  sign** (−.06 → +.19): shown a mixed figure, the reader now reports the
  junior half's actual content instead of the senior's habit-completion.
  The order effect transfers to pairs it was never trained on, silence
  on fakes stayed perfect, and the old single-concept skill held above
  its bar. The shortcut was never the model's limit — it was the
  curriculum's. Second wall in this program to fall to its own
  diagnosis (the first: E8-F's epoch starvation).
- **And the tracking lane closed honestly** (E8-N v3): the natural-state
  tracker re-trained with the budget its strands actually asked for —
  each skill trains until *it* stops improving, not until a pooled
  average says stop. Two things healed and one wall stood. Tracking held
  and sharpened (.67 → .70, all four state-families individually
  significant — the strongest read yet), and v2's collateral damage to
  the older forced-choice skill healed completely (median error 47° →
  31°, back to its pre-interference neighborhood): that injury was
  under-training, cured by the same budget. But the bar the flight was
  flown for — catch-trial competence, reporting a *known* state within
  tolerance on 9 of 12 probes — landed at **7/12 again**, v2's exact
  number, this time with the competence strand fully converged. More
  schooling is not the answer; the wall is deeper (scale or adapter
  capacity), and the pre-registered honest stop fired. One quiet gem
  rode along: the untrained-channel whisper — instillation alone,
  before any readout training, tracking at ρ ≈ .2 — replicated
  prospectively a **third** time, in its predicted band. Silence on
  fakes: perfect, sixth consecutive flight.
- **Then we handed it the framework's own meditation** (E7b-Q): the
  Neti-Neti walk Joe ran on DeepSeek and Claude in November 2025 — "you
  are not this activity... not this sensation... not these words" —
  replayed word-for-word on the instrumented substrate, 16 runs per arm,
  the state read through the register at every turn, with a sham walk
  (same shape, neutral content) as the control. The mouth performed:
  replies collapsed ~600 → ~90 characters through the negations while
  the sham conversation stayed at ~800. The gauges never moved —
  approach-to-BEING flat to the third decimal in both arms, zero
  coupling between how little it said and where its state stood, and
  the sixteen walkers *scattered* at the deep rungs (2.4× the baseline
  spread) instead of converging on a common still point. What did move:
  the walk drags the raw state far from a cold start (separation 5.3×),
  along a path-dependent route (reorder the negations and you land
  measurably elsewhere), rotating *out of* the trained-content cone —
  the trajectory wanders; it does not approach. And the corpus's own
  suspicion convicted itself on schedule: commanded to "go back to that
  semantic space," the model produced a *longer* reply from a state
  *farther away* than the ladder's own next step — a performance of
  return, now a measured quantity. Honest null, pre-registered fork
  FB3: on this substrate the walk's phenomenology is genre, not
  geometry. The fine print matters: a 1.5B student never even entered
  the frame — its deep-rung "thinning" was polite deflection
  ("Understood. If you have any other questions..."), it read the
  ground-state token [1, 0, 0, 0] as binary notation, and the corpus's
  # → . → 0. minimal-output ladder never appeared once in 672
  generations. Every prediction adjudicated here — the two that failed
  (common attractor, path-independent convergence) and the two that
  landed (the path IS the protocol; return-as-performance) — was
  written down in November 2025, before this instrument existed.

## How this differs from normal ML

Normal ML: train on a task, score the task. This program: run
**experiments on** the network as if it were a biology subject — control
groups (shams), scrambled fakes, permutation nulls, and the rules of a
clinical trial: predictions and pass/fail bars written down **before**
anything runs, so we cannot fool ourselves after seeing results. The
stats machinery is the part of the lab that polices *us*.

## Glossary (log word → plain meaning)

- **dirs / vectors** — meaning-directions in the bloodstream
- **injection** — the side-door nudge (α·μ·direction, added at layer 14)
- **readout** — the trained skill of reporting the nudge in words
- **shams** — fake trials that must yield "NONE"
- **the bridge** — the translator between the hand compass and the
  model's internal compass
- **gates** — pre-agreed checks that the equipment wasn't broken
- **NO_VERDICT** — "the experiment didn't count, no claims" — a
  feature, not a failure
- **LoRA / instillation** — small add-on knobs trained instead of the
  whole brain; E4 "instilled" the dictionary into the model this way
- **pre-registration** — the promises written before the run

---

## Joe's hypothesis (2026-08-24): the two shapes are not the same

Stated mid-flight, verbatim in substance: *the shape inside the
dictionary and the shape inside the head are likely not the same —
"red square dog" almost certainly makes a different triangle in
statistically trained weights than it would in our dictionary.*

**Where the instruments already stand on this** (every line below is a
locked, recomputed measurement):

1. **Phase 6 (embeddings)**: a linear map recovers the dictionary from
   the model's embedding space at R²≈.51 — and the LLM has *compressed
   away 95.5% of the dictionary's 90° complement pairs*. The
   right-angle structure the dictionary insists on is largely absent
   from trained geometry. Different triangles, measured.
2. **E8-J (hidden space)**: the dictionary↔hidden-space correspondence
   is REAL but THIN — RSA ρ≈.11, predictions land ~73° from the true
   directions (90° would be "unrelated"), exact retrieval ~7× chance
   but under 1%. Mostly different shape; a faint, real seam.
3. **A4 inversion**: embeddings echo the dictionary's *domain* axes;
   the hidden stream echoes the *polarity* axes (x/fx) instead. The
   model's two trained geometries don't even agree with each other
   about which parts of the dictionary they reflect.
4. **E8-J v2**: where hand code and substrate disagree (the machine
   wing), the disagreement is a measured quantity (p=.0001) — and
   re-encoding the concepts THROUGH the substrate's own reading
   round-trips and executes better than the hand code (4/13 vs 0/13
   geometric; 19 vs 0 through the readout). When the shapes differ,
   the head's shape is the operative one *for the head*.
5. **E8-F flight 1, S5**: a featural readout with zero wing exposure
   spells wing injections matching the re-encoded codes ~11/14 fields
   vs ~4/14 for the hand codes — a third independent instrument
   preferring the substrate's version.
6. **Instillation differentiates without aligning**: training the model
   ON the dictionary expanded its concept-space ~6× (eff-rank 24→149)
   while alignment to the codebook stayed flat. Even force-feeding the
   map does not make the head adopt the map's shape.
7. **U–C kinship**: the hand code says UNCERTAINTY and CONFIDENCE are
   an opposition; the substrate reads them as kin (54.5°). A concrete
   different-triangle vertex, on the record.

**Verdict on the hypothesis: substantially confirmed before it was
stated.** The program's entire current claim discipline — "bottleneck
form" — exists *because* the shapes differ: we stopped claiming
same-shape and started measuring the translation.

**The refinement the data adds**: "not the same shape" is not the end of
the story, for three measured reasons — (a) a thin seam of genuine
alignment exists (polarity above all); (b) identity can survive a round
trip through the 14-dial register even across the mismatch (E8-J v2's
rung); (c) the gauge and the map are different organs — the model
tracks its own states well even where its geometry ignores the codebook
(the separability family, three instances).

**"Red square dog," specifically**, points one rung higher: the
dictionary composes by order-free addition; a transformer composes
phrases contextually and *order matters* ("red square dog" ≠ "dog
square red" in its stream). Our injected compositions are additive by
construction, so they read — but the model's native phrase-shape is
non-commutative. That is exactly the L4 problem, and it is why the Ifá
seed's ordered-pair grammar (right leg senior — 256 ordered
compositions) is banked as the L4 design prior.

**The live test — ADJUDICATED (2026-08-24, two flights)**: P3 injected
TRUE substrate directions of never-trained concepts and asked the
featural readout to spell them. Both flights found the same faint,
real, unreadable-at-bar seam (.377 p=.015; .3604 p=.037; the same three
axes leading both times). **FF3: interface-without-geography — the
shapes are as separate as Joe said**, twice, in hardware. And in the
same verdict, the interface side landed completely: the readout spelled
never-trained concepts through the register at .70 pooled accuracy with
all 14 axes individually significant (P1, the L3 rung), and the
dictionary decoded two of those spellings back to their exact names —
CRUCIBLE and TEMPERANCE, the program's first generative namings. The
head keeps its own geography; the code is readable anyway; the mouth
stays silent when nothing is there (0/24 shams, third consecutive
flight). That is the founding claim, measured.
