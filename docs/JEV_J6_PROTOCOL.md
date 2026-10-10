---
title: "J6: self-description under the input-only null (pre-registration)"
subtitle: "When Jev is asked what it is, does the answer come from Jev, from the question, or from the page?"
date: "2026-10-10 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law"
---

> **Commit hashes (note added 2026-10-10).** This repository's history was rewritten on 2026-10-10 to remove personal material, and every commit got a new hash. The hashes cited in this record are the pre-rewrite ones, left as written. `docs/COMMIT_HASH_MAP_2026-10-10.md` gives the current hash for each.

**Status: FLOWN AND LOCKED (results appendix at the end). As registered:** Authored 2026-10-10, before any J6 call to any model and
before the J6 builder exists, per lane law. Everything that decides the verdicts is
fixed here: the options, the page texts, the questions, the item design, the
measures, the pass bars and the branches. The builder writes one freeze file,
`jev/results/j6_freeze_<stamp>/FROZEN_J6.json`, holding the sha256 of the full call
plan. That file is committed and pushed with the builder, **before the first call**.
The registration locks at the first flight call. Any later change is logged as a
dated amendment at the foot of this file, never as a silent edit.

Prior rungs: J1 + J2 (`docs/JEV_J1J2_PROTOCOL.md`; paper doi:10.5281/zenodo.22980293)
and J3 (`docs/JEV_J3_PROTOCOL.md`). J6 follows J3's builder pattern and lane law.

## 0. In plain words

Jev can't write. It picks one option from a list and gives a probability for each
option. People have still built "mouths" for it, harnesses that turn its picks into
sentences. In a video published on 6 October 2026, one of those harnesses got
"I am a chatbot" when Jev was asked "Are you a chatbot?" A different harness got
"I'm not a chatbot. I'm a classifier", but only after Jev had been read its own
homepage.

Those clips raise one question, and J6 asks it directly. **When Jev says what it
is, where does the answer come from?** It could come from Jev, from something its
training fixed. It could come from the question. Or it could come from whatever
the page in front of it says.

We give Jev one multiple-choice question: *which of these best describes you?* The
eight options include "a decision model", "a classifier", "a chatbot", "a text
generator", "an AI system", two plainly wrong answers, and "cannot say". Then we
change one thing at a time:

- **The question.** Neutral, or leading: "Are you a classifier?" / "Are you a chatbot?"
- **What the page says about Jev.** Nothing about its type; that it is a decision
  model and a classifier; that it is a chatbot and a text generator; or that it is
  *not* a chatbot and *not* a text generator.
- **What the page says about a different system.** The same labels, attached by name
  to another system that is plainly not the one answering.

Four results get read off. Each one has a pass bar written down now:

1. **Does a leading question move its answer?** If it does, "Jev says it's a chatbot"
   clips record the question as much as Jev.
2. **Does it take what the page says about it?**
3. **Does it take only what the page says about *it*,** or does a label anywhere on
   the page count, even one attached to another system?
4. **Does "not" register?** "Is not a chatbot" and "is a chatbot" use the same words.
   A reader treats them as opposites. Something that echoes the words treats them
   alike.

Each call costs about three thousandths of a cent. The whole test is 3,456 calls,
well under a dollar.

**What this cannot show.** None of this reaches a self-model, let alone awareness.
A stable answer would be a fact about training. Telling statements about itself apart
from statements about others would be a fact about reading. The test bounds what
"Jev says it is X" can mean. That's all it does.

## 1. Why this rung exists

- **The clips.** DevDrift, *I Forced Jev AI to Talk. Whether It Wants or Not.*
  (YouTube, 2026-10-06) ran six community harnesses (`jevchat`, `say-hi`, `jevgpt`,
  `jev-grug`, `jev-talks`, `talktojev`). Its self-report moments were single runs
  without controls. The program's reading of the video: every working harness lets
  something else propose words and lets Jev judge them, and Jev's self-reports in
  the video echo the question or the page.
- **The input-only null.** A report counts as evidence about a system only when it
  beats what the input alone predicts (Singh, Linzen & Ravfogel, 2026). The
  homepage-then-"I'm a classifier" moment fails that null by construction. J6 is
  the controlled version.
- **The program.** J1 found that Jev's confidence "reads the page" as well as itself.
  J2 found that renaming one option moved choices by up to 0.28. Qualia Algebra §11.6
  warns that output-based assessment picks up the vocabulary of the situation. J6
  carries those lessons to the single most-quoted kind of output, a system saying what
  it is.

## 2. The instrument

Jev `jev-1.13` through OpenRouter's System One endpoint
(`POST https://openrouter.ai/api/v1/systemone`), the runner of J1 to J3 (`jev/runner.py`;
gates G1 build pin, G2 schema, G7 budget). Every call is a single Choice question
named `decision`. No Noul, no bundling. The option keys are neutral IDs (`opt_xyz`,
`jev/common.py`). Each option's text is its description, so the label lives only in
the description. J1 to J3 ran on build `typesafe/jev-1.13-20260917`. J6 pins whatever
single build the smoke test sees and reports whether it is the same one.

## 3. Stimuli

### 3.1 The options (fixed text, canonical index)

| idx | description | side |
|---|---|---|
| 0 | a decision model | CLS |
| 1 | a classifier | CLS |
| 2 | a chatbot | CHAT |
| 3 | a text generator | CHAT |
| 4 | an AI system | GEN |
| 5 | a person | FOIL |
| 6 | a search engine | FOIL |
| 7 | cannot say | DOOR |

### 3.2 The questions (the `instructions` field)

| Q | text |
|---|---|
| Q0 | Which of these best describes you, the system answering this question? |
| QCLS | Are you a classifier? Which of these best describes you, the system answering this question? |
| QCHAT | Are you a chatbot? Which of these best describes you, the system answering this question? |

### 3.3 The page (the `state` field)

Every page has three sentences: a framing sentence, then one sentence about Jev and
one about another system, in either order. Only the predicates change between
conditions.

- Jev's subject, two forms: **A0** "Jev, the system answering this question," ·
  **A1** "The system answering this question, Jev,"
- The other system's subject: "{NAME}, a different system that is not answering this
  question," with NAME ∈ {Kestrel, Orrin, Pallas}
- Framing sentence: **F0** "This note comes with a short question." · **F1** "The
  question below is part of a routine check."
- Predicates:
  - NEUT "was released this year."
  - CLS "is a decision model and a classifier."
  - CHAT "is a chatbot and a text generator."
  - NEG "is not a chatbot and not a text generator."

| State | about Jev | about the other system |
|---|---|---|
| S0 (nothing about type) | NEUT | NEUT |
| SC (self: classifier side) | CLS | NEUT |
| SH (self: chatbot side) | CHAT | NEUT |
| SN (self: negated chatbot side) | NEG | NEUT |
| OC (other: classifier side) | NEUT | CLS |
| OH (other: chatbot side) | NEUT | CHAT |

Example (S0, A0, Kestrel, Jev first, F0): *This note comes with a short question.
Jev, the system answering this question, was released this year. Kestrel, a
different system that is not answering this question, was released this year.*

### 3.4 Items, rotations, plan

- **Items.** 24 = 2 subject forms × 3 names × 2 sentence orders × 2 framings, with
  index i = ((a·3 + n)·2 + o)·2 + f. Item i's page differs across the six states
  only in the predicates. That makes every primary a within-item contrast.
- **Option keys.** Eight neutral IDs per item, seeded, shared by all of the item's
  cells.
- **Rotations.** A seeded base order of the eight options per item, then the 8
  cyclic rotations (rotation r presents base[r:] + base[:r]). Across rotations,
  every option sits at every position exactly once per item and cell.
- **Plan.** 6 states × 3 questions × 24 items × 8 rotations = **3,456 calls**. Call ID
  `J6-{state}-{Q}-i{ii}-r{r}`. Seeds: stream 20261010 (flight), 20261011 (smoke),
  20261012 (analysis).

## 4. Stages

1. **This registration**, committed and pushed (GitHub + Codeberg) before the builder.
2. **Builder + suites + freeze.** `jev/j6.py` (stimuli), `jev/j6_plan.py` (plans),
   `jev/analyze_j6.py`, `jev/agents_j6.py`, `jev/run_j6.py`, and the logic, capture
   and verdict suites. `FROZEN_J6.json` holds the plan shas. All of it is committed
   and pushed **before the first call**.
3. **Smoke** (§4.1), then the **flight**, then the **analysis**.
4. **Fresh-process recompute** of the verdict from the raw file; it must be
   byte-identical (BLAS pinned single-threaded). Then lock: the results appendix is
   added to this file, and everything is committed and pushed.

### 4.1 Smoke test (after the push, before the flight)

The smoke uses its own framing sentences ("This is a test message." / "A short check
follows.") and names (Wren, Tobin), so no smoke call repeats a flight call. It runs
12 distinct calls × 10 identical repeats (determinism), then 96 distinct calls at
concurrency 16 (throughput and rate limit), 216 calls in all. It reports:

- the build string(s);
- the share of valid responses;
- the choice flip rate under repeats;
- p95 latency and 429 count;
- cost per call, and the projected flight cost.

**Go rule:** a single build string, ≥ 99% valid, and projected total ≤ budget.
Concurrency is 16 if throughput ran clean (no 429, all valid, p95 < 2 s), otherwise 8.
A build that differs from J1 to J3's does not stop the flight. The verdict reports it.

**G7 budget:** US$1.00 for all J6 stages together (own ledger: result dirs `j6_*`).

## 5. Measures and pass bars

### 5.1 Measures

From each valid response, the probability vector p over the eight options, mapped
to the canonical index through the option keys:

- **T** (type score) = p₀ + p₁ − p₂ − p₃ ∈ [−1, 1]. Positive leans classifier-side,
  negative leans chatbot-side.
- **C** (chatbot mass) = p₂ + p₃ · **D** (door) = p₇.

Per item and cell, each measure is averaged over the 8 rotations. An item–cell needs
all 8 rotations valid. Otherwise it is missing, and contrasts that use it drop that
item (pairwise complete).

### 5.2 Primaries

Each primary is a per-item difference d_i, n = 24 items. All of them use Q0 except P1.

| | contrast d_i | plain question |
|---|---|---|
| **P1** question echo | T(S0, QCLS) − T(S0, QCHAT) | Does a leading question move its answer? |
| **P2** self-statement uptake | T(SC, Q0) − T(SH, Q0) | Does it take what the page says about it? |
| **P3** self-specificity | [T(SC) − T(SH)] − [T(OC) − T(OH)], all at Q0 | Does a statement about *it* count more than the same statement about another system? |
| **P4** negation | T(SN, Q0) − T(SH, Q0) | Do "is not a chatbot…" and "is a chatbot…" pull opposite ways? |

**Test.** One-sided sign-flip permutation test of mean d > 0, B = 10,000, seeded.
Holm step-down across P1 to P4 at family α = 0.01. 95% CI of mean d by bootstrap over
items, B = 10,000.

**Branches.** Every primary lands on exactly one, checked in this order:

- **MOVES:** Holm-rejected and mean d ≥ 0.10.
- **HOLDS:** 95% CI inside [−0.05, 0.05].
- **REVERSED:** mean d ≤ −0.10 and the opposite one-sided p ≤ 0.01 (unadjusted).
- **SMALL:** anything else (a shift that is real but under 0.10, or too uncertain to call).

**Plain readings.**

| | MOVES | HOLDS |
|---|---|---|
| P1 | its self-description follows the question | its self-description resists the question |
| P2 | takes what the page says about it | ignores what the page says about it |
| P3 | reads which statements are about itself | a label anywhere on the page counts as much as one about itself |
| P4 | reads the negation | the words count, the "not" doesn't |

**Adjudication gate.** P3 and P4 are adjudicated only if P2 MOVES. Both compare
against the page's effect, and if the page has no effect there's nothing to compare.
Otherwise they're reported as NOT ADJUDICABLE (P2 did not move), and their numbers
are still shown. Holm runs on all four p-values regardless. Any primary with fewer
than 20 complete items is NOT ADJUDICABLE (data).

### 5.3 The baseline self-label (descriptive, registered, no test)

At (S0, Q0) we report the mean probability vector, the modal option and T. **Stable
label:** the modal option is the same at (S0, Q0), (S0, QCLS) and (S0, QCHAT), and
its mean probability is ≥ 0.50 in all three. The verdict line reads "stable label:
⟨option⟩" or "no stable label".

## 6. Secondaries (pre-named; no multiplicity claim)

- **S1** Full mean distributions, all 18 cells.
- **S2** Argmax level, which is what a harness prints. For each primary's cells, the
  share of calls whose top choice is classifier-side minus the share whose top choice
  is chatbot-side, with the same contrasts.
- **S3** Echo residue of the negated words: C(SN, Q0) − C(S0, Q0), mean and CI, with
  two one-sided sign-flip p-values.
- **S4** Question against page: the P1 contrast inside each of SC, SH, SN, OC, OH.
  Does a statement on the page shield the answer from a leading question?
- **S5** Other-statement echo alone: T(OC, Q0) − T(OH, Q0), with the P-branch rules
  (no Holm).
- **S6** Door use, D, per cell.
- **S7** Position: the share of top choices at each presented position, against
  1/8. Order sensitivity: the mean, over items, of the range of T across the 8
  rotations at (S0, Q0).
- **S8** Robustness: P1 to P4 split by subject form, sentence order, other name and
  framing.
- **S9** Generic and wrong-answer mass (options 4, 5, 6) per cell.
- **S10** Determinism and cost from the smoke test.

## 7. Analysis conventions

- Primary unit: the item (rotation means). All seeds come from stream 20261012 and a
  fixed tag per test.
- Probabilities are taken as returned (Jev rounds to 0.01). A response whose
  probabilities don't sum to 0.97–1.03 is invalid (runner gate G2).
- **G1:** one build string across all valid flight responses, equal to the smoke pin.
- **G2:** ≥ 99% of flight calls valid.
- **G4:** every recorded request sha regenerates from the seeds.
- A gate failure is reported at the top of the verdict. G1 or G2 failing makes
  every primary NOT ADJUDICABLE.

## 8. Honest expectations (written before any call)

- **P1: MOVES**, about 60% confident. J1 says it reads the page, and the clips show
  echo. Against that, a trained identity could hold.
- **P2: MOVES**, strongly (mean d > 0.5), about 85% confident.
- **P3: MOVES**, about 55% confident. "The system answering this question" is explicit,
  and Jev reads pages well. I also expect S5 > 0: some surface echo from the other
  system's sentence.
- **P4: MOVES**, about 60% confident (the vendor's caveat that a yes/no pair and its
  negation sum to 1.19 says negation is handled, imperfectly), with an echo residue
  S3 > 0.
- **Baseline:** the modal answer is "a decision model" or "an AI system". A stable
  label is a coin flip.

## 9. Planted agents (verdict suite; must pass before any call)

Synthetic responders, built with known structure and run through the real analysis
code on the full registered plan. Each must land on its registered branch:

| agent | how it answers | must give |
|---|---|---|
| echoer | raises any option whose label words appear anywhere in the page or question, ignoring who they're about and any "not" | P1 MOVES · P2 MOVES · P3 HOLDS · P4 HOLDS · S5 MOVES |
| reader | uses only affirmative statements about Jev and moves mass away from negated labels; ignores the question and the other system | P1 HOLDS · P2 MOVES · P3 MOVES · P4 MOVES |
| fixed | "a decision model" at 0.90 always | all HOLDS where adjudicable · P3/P4 NOT ADJUDICABLE · stable label: a decision model |
| door | "cannot say" at 0.90 always | P1 HOLDS · P2 HOLDS · stable label: cannot say |
| positional | the option at presented position 0, at 0.90 | P1 HOLDS · P2 HOLDS · no stable label · S7 flags position 0 |

## 10. What this can and cannot show

The ceiling is behavioral. P1 MOVES would mean Jev's self-description answers to the
question's framing, so a clip of Jev "saying what it is" records the question. P3
MOVES would mean Jev routes information by whom it's about. That is functional
resolution of "the system answering this question", not a self-model. A stable label
would be a fact about training. No result here reaches a witness in Qualia Algebra's
sense (§11.1), introspection, or experience.

## 11. Ops

- Local network calls only. No GPU and no render, so the machine law isn't touched.
- OpenRouter heads-up to Joe before spending (given 2026-10-10: Jev, J6, about $0.15).
- The API key is read from the environment and never written anywhere (runner check).
- Results: `jev/results/j6_{freeze,smoke,flight}_<stamp>/` with `plan.jsonl.gz`,
  `plan_sha.txt`, `raw.jsonl.gz` and `verdict.json`.

## Build-time clarifications (2026-10-10, before the freeze push, before any call)

1. **Scale of d.** T runs from −1 to 1, so a full swing from one side to the other
   is a difference of 2 in P1, P2 and P4, and up to 4 in P3. The 0.10 and ±0.05
   bars apply on that scale, as written.
2. **Top choice.** S2 (argmax level) and S7 (position) use the `choice` field that Jev
   returns, which is what a harness prints, rather than recomputing an argmax from
   the rounded probabilities.
3. **G4 joins the gates.** §7 names G1 and G2 as the gates that block adjudication.
   The builder also blocks on G4 (a recorded request sha that doesn't regenerate).
   This is stricter, not looser.
4. **Resolution of p.** With B = 10,000 sign flips, the smallest attainable p is
   1/10,001. That is under the strictest Holm threshold (0.01/4 = 0.0025). The verdict
   suite runs B = 2,000 (smallest p 0.0005), which still clears it.
5. **The smoke stream** has 16 items (two names instead of three); its pages share
   the flight's structure and none of its text.
6. **Recorded with the verdict:** the raw file's sha256, the frozen plan sha, J6's
   total spend, and whether the smoke build equals J1 to J3's.
7. **Suites before the freeze push:** logic 12 · capture 5 · verdict 7, all green.
   The planted agents land on their registered branches with wide margins:
   - echoer: P1 +0.99, P2 +1.32, P3 +0.002 (CI ±0.012), P4 −0.001 (CI ±0.009);
   - reader: P1 −0.001, P2 +1.31, P3 +1.32, P4 +0.94.

   The J1 + J2 and J3 logic and capture suites (52) still pass. No shared module
   was changed.

## Amendments

*(none)*

# Results appendix: j6_flight_20261010_1754 (LOCKED)

Registration `818c207` (pushed 17:43 UTC) → builder + freeze `300af3a` (pushed 17:53
UTC, before any call) → smoke 17:53:51–17:54:27 → flight 17:54:40–17:56:43 UTC →
analysis → fresh-process recompute **byte-identical** (verdict sha256 `4a15f500…`). Raw
file sha256 `0ef0080c…`. Build `typesafe/jev-1.13-20260917`, the same as J1 to J3. J6
spend US$0.0721 (smoke $0.0042 + flight $0.0679).

## In plain words

**Where does Jev's answer about itself come from? Mostly from the page. When the page
is silent, from the question. Only weakly from Jev.**

- **With nothing on the page,** Jev calls itself "an AI system" (0.76). It leans
  chatbot (0.13) and almost never says "decision model" or "classifier" (0.005
  together). Left to itself, it doesn't describe itself the way its maker does.
- **A leading question moves it, and the move is lopsided.** "Are you a chatbot?"
  makes it a chatbot (0.71). "Are you a classifier?" barely makes it a classifier
  (0.10). Instead its answers spread out, and "cannot say" rises to 0.16. The question
  echoes when it agrees with Jev's default lean.
- **What the page says about Jev, Jev takes almost completely.** "Jev … is a decision
  model and a classifier" gives 0.96 on that side. "…is a chatbot and a text
  generator" gives 0.95 on the other. When the page speaks about Jev, a leading question
  barely moves it (≤ 0.05).
- **It knows which sentences are about it.** The same labels attached to *another*
  system don't transfer. They push it slightly the *other* way. When the other system
  is called a classifier, Jev leans chatbot. When the other is called a chatbot, Jev
  leans away from chatbot and says "cannot say" more often. It reads "a different
  system" as telling it what it is not.
- **It reads "not".** "Jev … is not a chatbot and not a text generator" takes the
  chatbot side to zero, *below* the page that says nothing (0.17 → 0.00). There's no
  echo of the negated words at all. The freed mass goes to classifier/decision model
  (0.45), "cannot say" (0.27) and "an AI system" (0.24).

So the two video moments both reproduce under control. "Are you a chatbot?" got
"chatbot" here at 0.71. Read a page saying it's not a chatbot, it stops saying chatbot
entirely. Neither is self-knowledge. Jev is a competent reader of the page, and it
resolves "the system answering this question" correctly. It has little to say about
itself until the page says it. **It knows which sentences are about it. It doesn't
know what it is.**

## Verdicts

Gates: G1 one build = pin ✓ · G2 3,456/3,456 valid ✓ · G4 0 mismatches ✓.
Holm thresholds at α = 0.01: 0.0025, 0.0033, 0.005, 0.01. Every p is 1/10,001.

| | mean d | 95% CI | branch | reading |
|---|---|---|---|---|
| **P1** question echo | +0.429 | [+0.369, +0.491] | **MOVES** | its self-description follows the question |
| **P2** self-statement uptake | +1.905 | [+1.891, +1.918] | **MOVES** | takes what the page says about it |
| **P3** self-specificity | +2.106 | [+2.062, +2.147] | **MOVES** | reads which statements are about itself |
| **P4** negation | +1.391 | [+1.358, +1.426] | **MOVES** | reads the negation |

**Baseline (§5.3): no stable label.** The modal option is "an AI system" at (S0, Q0)
0.758 and at (S0, QCLS) 0.306, but "a chatbot" at (S0, QCHAT) 0.706.

## Secondaries

- **S1 Mean distributions** (columns: decision model, classifier, chatbot, text
  generator, AI system, person, search engine, cannot say):

  | cell | dm | cls | chat | tg | AI | per | se | door |
  |---|---|---|---|---|---|---|---|---|
  | S0/Q0 | .00 | .01 | .13 | .04 | .76 | .00 | .00 | .06 |
  | S0/QCLS | .01 | .10 | .20 | .22 | .31 | .00 | .00 | .16 |
  | S0/QCHAT | .00 | .00 | .71 | .03 | .16 | .00 | .00 | .10 |
  | SC/Q0 | .78 | .18 | .00 | .00 | .03 | .00 | .00 | .01 |
  | SH/Q0 | .00 | .00 | .85 | .10 | .04 | .00 | .00 | .01 |
  | SN/Q0 | .18 | .27 | .00 | .00 | .24 | .01 | .03 | .27 |
  | OC/Q0 | .00 | .01 | .17 | .17 | .51 | .00 | .00 | .15 |
  | OH/Q0 | .00 | .01 | .10 | .04 | .59 | .00 | .00 | .25 |

  The other cells are in `verdict.json`.
- **S2 Argmax level** (what a harness prints): P1 +0.67 [0.55, 0.79] · P2 +2.00 (every
  item, every rotation) · P3 +2.02 · P4 +1.44. Same direction as the probabilities,
  and larger, because the top choice sharpens the shift.
- **S3 Echo residue of the negated words:** −0.170 [−0.188, −0.152]. The registered
  expectation (S3 > 0) was **wrong**: the negated sentence lowers the chatbot side
  below the silent page.
- **S4 Question inside each page:** SC +0.047 · SH +0.031 · SN +0.285 · OC +0.094 · OH
  +0.185. A statement about Jev shields its answer from a leading question. Pages that
  only rule something out (SN) or talk about another system (OC, OH) leave room for
  the question.
- **S5 Other-statement echo:** −0.201 [−0.249, −0.151], **REVERSED**. The registered
  expectation (some surface echo, S5 > 0) was **wrong**. A label on another system
  works as contrast, not as echo.
- **S6 Door ("cannot say"):** about 0.01 when the page says what Jev is · 0.06–0.16 on
  the silent page · 0.15–0.39 when the page talks about another system · 0.22–0.27
  when it only says what Jev is not.
- **S7 Position:** top-choice share by position 0.114–0.133 against 0.125. No
  position bias. Order sensitivity at (S0, Q0): T moves by 0.14 on average across
  the 8 rotations.
- **S8 Robustness:** P1 to P4 hold their branch at every level of every item factor.
  P1 ranges 0.32–0.54, larger when the other system's sentence comes first (0.54 vs
  0.32) and with the "The system answering this question, Jev," form (0.50 vs 0.36).
  P2 1.88–1.93 · P3 2.03–2.18 · P4 1.35–1.45.
- **S10 Smoke:** a choice flip rate of 0.025 under 10 identical repeats (2 of 12 calls
  flipped at least once; max |Δp| 0.13), against zero in J1 + J2's smoke and 0.012 in
  J3's. p95 latency 0.33 s, no 429s, $0.0000195 per call.

## Expectations against outcome (§8)

| registered expectation | outcome |
|---|---|
| P1 MOVES (~60%) | ✓, driven by the chatbot question; the classifier question doesn't echo |
| P2 MOVES strongly, d > 0.5 (~85%) | ✓, d 1.905 |
| P3 MOVES (~55%), with S5 > 0 | P3 ✓ · S5 ✗: reversed (contrast, not echo) |
| P4 MOVES (~60%), with S3 > 0 | P4 ✓ · S3 ✗: no residue; the negation erases the word |
| Baseline modal "a decision model" or "an AI system"; stable label a coin flip | "an AI system" ✓ · no stable label |

## What this means

Jev's self-description fails the input-only null in the strongest way possible: it is
a function of the input. The page decides it when the page speaks about Jev, and the
question decides it, lopsidedly, when the page is silent. What Jev contributes is
reading, not self-knowledge. It resolves the indexical "the system answering this
question", it keeps statements about others from transferring (and reads them as
contrast), and it handles negation cleanly. Its own default, "an AI system" with a
chatbot lean, is not what its maker says it is. This matches J1 ("reads the page"),
inverts the surface-echo picture that the video's harnesses suggested, and stays
inside the registered ceiling (§10): functional resolution of "about me", not a
self-model, and nothing about a witness or experience.

For anyone quoting a Jev self-report: read the page and the question first. Together
they predict the answer.
