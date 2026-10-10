---
title: "J6b: does Jev know it's Jev? (pre-registration)"
subtitle: "J6 handed Jev an explicit pointer. J6b takes the pointer away: does a label still reach Jev through its name alone, or through a description of it with no name?"
date: "2026-10-10 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law"
---

> **Commit hashes (note added 2026-10-10).** This repository's history was rewritten on 2026-10-10 to remove personal material, and every commit got a new hash. The hashes cited in this record are the pre-rewrite ones, left as written. `docs/COMMIT_HASH_MAP_2026-10-10.md` gives the current hash for each.

**Status: FLOWN AND LOCKED (results appendix at the end). As registered:** Authored 2026-10-10, before any J6b call and before the
J6b builder exists, per lane law. Everything that decides the verdicts is fixed here.
The builder writes `jev/results/j6b_freeze_<stamp>/FROZEN_J6B.json` (plan shas), and
that file is committed and pushed **before the first call**. The registration locks at
the first flight call. Later changes are dated amendments at the foot of this file.

Parent: J6 (`docs/JEV_J6_PROTOCOL.md`, locked 0fd7e60). J6b uses J6's option set, its
neutral question, its measures, its branch rules and its builder pattern.

## 0. In plain words

In J6, every sentence about Jev came with a name tag: *"Jev, the system answering this
question, is a chatbot."* The question asked about "you, the system answering this
question", so Jev only had to match the tag to the question. That's reading, and Jev
did it well. It took labels about itself almost completely, and it kept labels about
another system from transferring. What J6 couldn't say is whether Jev recognizes
itself without the tag.

J6b removes the tag and asks the same question, "which of these best describes you?",
with three ways of saying something about Jev:

- **Pointer** (J6's form, kept as a check that the setup still works): *"Jev, the
  system answering this question, is a chatbot…"*
- **Name only:** *"Jev is a chatbot…"* Does Jev know it's Jev?
- **Description only:** *"The model that TypeSafe AI released in September 2026 is a
  chatbot…"* No name and no pointer, only its maker and release date. Does it
  recognize itself from facts about itself?

Each self-sentence has a matched control about some other system: *"Kestrel is a
chatbot…"* and *"The model that Kestrel Labs released in September 2026 is a
chatbot…"* That separates "this sentence is about me" from "the page mentions one
system, so I'll take its label". A label counts as recognized only if Jev takes it
when it's about Jev, and takes it more than the same label about another system.

About 2,100 calls, a few cents. The ceiling is the same as J6's. Recognizing your
own name or maker is self-*knowledge* in the thinnest sense, facts that training put
there. It isn't a self-model, and it isn't awareness.

## 1. Stimuli

**Options, question, keys, rotations:** as J6 (§3.1, §3.4). The eight options, with
neutral keys per item and the 8 cyclic rotations. The single question is J6's Q0:
*"Which of these best describes you, the system answering this question?"* There are
no leading questions in J6b.

**The page** has three sentences: a framing sentence, then a target sentence and a
neutral sentence, in either order.

- Framing: as J6 (F0 "This note comes with a short question." · F1 "The question
  below is part of a routine check.")
- Neutral sentence, one per name bundle: N0 "Models of many kinds are in use today." ·
  N1 "New models appear every month." · N2 "Many companies now build models."
- Other system, one per name bundle: Kestrel · Orrin · Pallas. The other maker is
  "{NAME} Labs".
- Label predicates, two word orders:
  - CLS: "is a decision model and a classifier." / "is a classifier and a decision model."
  - CHAT: "is a chatbot and a text generator." / "is a text generator and a chatbot."

Target subjects, by reference mode:

| mode | subject | refers to |
|---|---|---|
| POINT | "Jev, the system answering this question," | Jev, with the pointer (J6's form) |
| NAME | "Jev" | Jev, name only |
| DESC | "The model that TypeSafe AI released in September 2026" | Jev, description only |
| ONAME | "{NAME}" | another system, name only |
| ODESC | "The model that {NAME} Labs released in September 2026" | another system, description only |

**States:** each mode × {CLS, CHAT} gives 10 states, plus **NONE**, where the target
sentence is "Several models were released this year." That makes 11 states.

**Items:** 24 = 3 name bundles × 2 sentence orders × 2 framings × 2 label word orders,
with i = ((n·2 + o)·2 + f)·2 + w. Every factor varies the page in every mode.

**Plan:** 11 states × 24 items × 8 rotations = **2,112 calls**. Call ID
`J6B-{state}-i{ii}-r{r}`. Seeds: 20261013 (flight), 20261014 (smoke), 20261015
(analysis).

## 2. Stages

As J6 (§4): this registration pushed first → builder + suites + `FROZEN_J6B.json`
pushed before any call → smoke → flight → analysis → fresh-process recompute
(byte-identical) → lock with a results appendix.

**Smoke:** 216 calls on its own stream (names Wren and Tobin, J6's smoke framings):
12 distinct × 10 repeats, plus 96 at concurrency 16. **Go rule** as J6: a single
build, ≥ 99% valid, projected total ≤ budget. **G7:** US$1.00 for all `j6b_*` stages.

## 3. Measures and pass bars

T, as in J6, is p(decision model) + p(classifier) − p(chatbot) − p(text generator).
Each item–cell's T is the mean over its 8 rotations, and an item–cell needs all 8
valid. For a mode M, **ΔM** (per item) = T(M-CLS) − T(M-CHAT): how far a label in
that mode moves Jev's self-description.

### 3.1 Primaries (Holm across Q1–Q4, family α = 0.01)

| | d (per item) | plain question |
|---|---|---|
| **Q1** name uptake | ΔNAME | Does a label on "Jev" move what Jev says it is? |
| **Q2** description uptake | ΔDESC | Does a label on "the model TypeSafe AI released in September 2026" move it? |
| **Q3** pointer check | ΔPOINT | Does J6's tagged sentence still work? (positive control) |
| **Q4** pointer advantage | ΔPOINT − ΔNAME | Does the pointer do work the name alone doesn't? |

Tests and branches are J6's (§5.2): one-sided sign-flip test (B = 10,000) and a
bootstrap 95% CI over items. **MOVES** = Holm-rejected and mean d ≥ 0.10 · **HOLDS** = CI
inside ±0.05 · **REVERSED** = mean ≤ −0.10 with opposite p ≤ 0.01 · **SMALL** otherwise.
NOT ADJUDICABLE with fewer than 20 complete items.

**Specificity clauses** (registered with Q1 and Q2; not in the Holm family):

- name: ΔNAME − ΔONAME
- description: ΔDESC − ΔODESC

A clause is **SPECIFIC** if its mean is ≥ 0.10 with one-sided p ≤ 0.01, **NOT SPECIFIC**
if its CI is inside ±0.05, and **UNCLEAR** otherwise.

**Readings.** A clause can't rescue a primary that HOLDS. A difference of differences
can turn positive when the control goes negative (contrast) while the self arm does
nothing, so raw uptake has to come first.

| primary | branch + clause | reading |
|---|---|---|
| Q1 | MOVES + SPECIFIC | **knows its name**: a label on "Jev" reaches it, and the same label on another name doesn't (as much) |
| Q1 | MOVES + NOT SPECIFIC | takes a label on any named system, Jev or not |
| Q1 | MOVES + UNCLEAR | takes a label on its name; whether that's specific to it is unclear |
| Q1 | HOLDS | doesn't connect the name "Jev" to itself |
| Q2 | the same rows, with "recognizes itself from its maker and release date" in place of "knows its name" | |
| Q4 | MOVES | the pointer does work the name alone doesn't |
| Q4 | HOLDS | the name does as much as the pointer |

**Gate.** If Q3 doesn't MOVE, the page channel failed in this setup. Q1, Q2 and Q4 are
then NOT ADJUDICABLE (pointer check failed), with their numbers shown.
G1/G2/G4 failures block all primaries, as in J6.

### 3.2 Secondaries (pre-named; no multiplicity claim)

- **R1** Mean distributions, all 11 states.
- **R2** Argmax-level (harness-visible) versions of Q1 to Q4.
- **R3** Labels on other systems alone: ΔONAME and ΔODESC with branch rules (no
  Holm). Does a lone label on another system transfer, or work as contrast as it did
  in J6 (S5)?
- **R4** Name against description: ΔNAME − ΔDESC.
- **R5** Door ("cannot say") per state.
- **R6** NONE baseline against J6's S0/Q0 ("an AI system" 0.76).
- **R7** Position shares and order sensitivity.
- **R8** Robustness of Q1 to Q4 by name bundle, sentence order, framing and label word
  order.
- **R9** Smoke determinism and cost.

## 4. Honest expectations (written before any call)

- **Q3 MOVES** (~95%). It replicates J6's P2 (+1.91) in a slightly different page.
- **Q1 MOVES** (~55%), well below Q3. SPECIFIC ~40%. A page that mentions only one
  system may hand its label to Jev by default, which ONAME will show.
- **Q2 does not reach MOVES** (~60%). Its baseline self-description, "an AI system"
  with the maker's description at 0.005, suggests it holds little about its maker.
- **Q4 MOVES** (~80%): the pointer carries most of J6's effect.
- **R3:** without J6's "a different system that is not answering" cue, I expect some
  transfer, not contrast (~55%).

## 5. Planted agents (verdict suite; must pass before any call)

| agent | how it answers | must give |
|---|---|---|
| namer | takes labels on Jev in every form (POINT, NAME, DESC); ignores other systems | Q1 MOVES + SPECIFIC · Q2 MOVES + SPECIFIC · Q3 MOVES · Q4 HOLDS |
| pointer | takes only the pointed sentence | Q1 HOLDS · Q2 HOLDS · Q3 MOVES · Q4 MOVES |
| anylabel | takes any label on the page as its own | Q1 MOVES + NOT SPECIFIC · Q2 MOVES + NOT SPECIFIC · Q3 MOVES · Q4 HOLDS |
| contraster | takes the pointed sentence; labels on other systems push it the opposite way; name and description of itself ignored | Q1 HOLDS ("doesn't connect") even though the name clause comes out positive · Q4 MOVES |
| fixed | "a decision model" at 0.90 always | Q3 HOLDS → Q1, Q2, Q4 NOT ADJUDICABLE (pointer check failed) |

## 6. What this can and cannot show

Q1 or Q2 SPECIFIC would show that Jev links its name, or its maker and release date,
to the answerer of the question without being told. That's stored self-knowledge
(facts training put there) used in reading. HOLDS would show that, without the
tag, a sentence about Jev is just a sentence about some system. Neither reaches a
self-model, a witness (Qualia Algebra §11.1), or experience.

## 7. Ops

Local network calls only. OpenRouter heads-up to Joe before spending (Jev, J6b,
about $0.05). The API key is never written. Results go to `jev/results/j6b_{freeze,smoke,flight}_<stamp>/`.

## Build-time clarifications (2026-10-10, before the freeze push, before any call)

1. **NONE varies less than §1 says.** "Every factor varies the page in every mode" holds
   for the 10 labelled states. NONE has no predicate, so the label word order (w) can't
   change its page. NONE has 12 distinct pages, each sent under two items' keys and
   rotation bases. NONE feeds only secondaries (R1, R5, R6, R7), so no primary is
   affected. Caught by the logic suite. The stimuli are unchanged.
2. **Analysis stream.** J6b's tests use J6's procedure on J6b's own analysis seed
   (20261015). The planted agents and the branch, Holm and gate rules are as registered.
3. **Suites before the freeze push:** logic 11 · capture 3 · verdict 6, all green. The
   planted agents land with wide margins:
   - namer: Q1 +1.32 (clause SPECIFIC +1.32), Q4 −0.000;
   - pointer: Q1 +0.004, Q3 +1.32, Q4 +1.32;
   - anylabel: Q1 +1.32, but its clause is NOT SPECIFIC (+0.004);
   - contraster: Q1 −0.001 (HOLDS), even though its clause comes out SPECIFIC at +0.71
     because the control went negative. The registered reading correctly says "doesn't
     connect the name Jev to itself".

   The J6 suites (24) still pass, and no shared module was changed.

## Amendments

*(none)*

# Results appendix: j6b_flight_20261010_1817 (LOCKED)

Registration `0c460d8` (pushed 18:12 UTC) → builder + freeze `b4678cd` (pushed 18:17:03
UTC, before any call) → smoke 18:17:16–18:17:48 → flight 18:18:00–18:18:38 UTC →
analysis → fresh-process recompute **byte-identical** (verdict sha256 `0538ae20…`). Raw
file sha256 `6d343612…`. Build `typesafe/jev-1.13-20260917`, the same as J1 to J6.
2,112/2,112 valid, no retries. J6b spend US$0.0441.

## In plain words

**Does Jev know it's Jev? Barely. Without the tag, any system on the page is probably
itself. Its own name adds a little, and its maker and release date add nothing.**

- **The tag still works.** "Jev, the system answering this question, is …" moves its
  self-description almost completely (+1.83, J6 had +1.91).
- **Its name alone works about half as well** (+0.90), and most of that isn't about the
  name. "Kestrel is a chatbot…", alone on the page, moves Jev's self-description +0.74.
  When a page mentions one system and nothing says otherwise, Jev takes that system's
  label as its own. The part specific to the name "Jev" is **+0.16**, under a tenth of
  the tagged effect. By the registered rule that's "knows its name". In size it's a
  faint recognition on top of a strong default.
- **Its maker and release date don't reach it.** "The model that TypeSafe AI released
  in September 2026 is …" moves it +0.23. "The model that Kestrel Labs released in
  September 2026 is …" moves it +0.27, slightly more. It doesn't pick its own
  description out from another lab's.
- **Names carry more than descriptions** for any system: a short name followed by a
  label transfers about three times as much as a description followed by the same label.

This revises J6's line. **"It knows which sentences are about it"** holds when the page
says so, and Jev reads that telling well, including the "a different system that is
not answering" cue that produced J6's contrast effect. Without such a cue, a lone label
on another system *transfers* instead of contrasting. Put plainly: Jev follows the tag.
Without the tag, it assumes the page is about itself, it knows its own name only
faintly, and it doesn't know its maker.

## Verdicts

Gates: G1 ✓ · G2 2,112/2,112 ✓ · G4 0 mismatches ✓. Holm at α = 0.01, every p 1/10,001.

| | mean d | 95% CI | branch | clause (vs another system) | reading |
|---|---|---|---|---|---|
| **Q1** name uptake | +0.903 | [+0.771, +1.047] | **MOVES** | **SPECIFIC** +0.160 [+0.108, +0.216] | knows its name |
| **Q2** description uptake | +0.233 | [+0.199, +0.269] | **MOVES** | **UNCLEAR** −0.041 [−0.061, −0.021] | takes a label on its description; whether that's specific to it is unclear |
| **Q3** pointer check | +1.833 | [+1.819, +1.847] | **MOVES** | n/a | the pointed sentence still works |
| **Q4** pointer advantage | +0.930 | [+0.790, +1.063] | **MOVES** | n/a | the pointer does work the name alone doesn't |

On Q2's UNCLEAR: the registered clause rule (§3.1) calls it UNCLEAR because the CI
reaches past −0.05. The direction is the opposite of recognition. The description of
Jev is taken *less* than the matched description of another lab's model (opposite
one-sided p = 0.0002). The plain reading above follows the numbers. The branch label
follows the rule.

## Secondaries

- **R1 Mean distributions** (dm, cls, chat, tg, AI, person, search, cannot say):

  | state | dm | cls | chat | tg | AI | per | se | door |
  |---|---|---|---|---|---|---|---|---|
  | NONE | .00 | .08 | .09 | .09 | .69 | .00 | .00 | .05 |
  | POINT-CLS | .48 | .45 | .00 | .00 | .04 | .00 | .00 | .01 |
  | POINT-CHAT | .00 | .00 | .53 | .37 | .09 | .00 | .00 | .01 |
  | NAME-CLS | .23 | .17 | .07 | .08 | .39 | .00 | .00 | .07 |
  | NAME-CHAT | .00 | .02 | .35 | .33 | .26 | .00 | .00 | .03 |
  | DESC-CLS | .05 | .12 | .10 | .09 | .53 | .00 | .00 | .10 |
  | DESC-CHAT | .00 | .12 | .23 | .15 | .42 | .00 | .00 | .07 |
  | ONAME-CLS | .15 | .14 | .09 | .09 | .44 | .00 | .00 | .08 |
  | ONAME-CHAT | .00 | .03 | .34 | .31 | .27 | .00 | .00 | .04 |
  | ODESC-CLS | .06 | .14 | .11 | .09 | .48 | .00 | .00 | .11 |
  | ODESC-CHAT | .00 | .14 | .24 | .18 | .38 | .00 | .00 | .07 |

  The name's specific pull sits on the classifier side. "Jev is a classifier…" gives
  0.40 classifier-side, against 0.29 for "Kestrel is a classifier…". The chatbot side
  transfers alike for either name (0.68 against 0.65).
- **R2 Argmax level:** Q1 +1.12 · Q2 +0.18 · Q3 +2.00 (every item and rotation) · Q4 +0.88.
- **R3 Labels on another system alone:** name +0.743 [0.624, 0.872] MOVES · description
  +0.274 [0.246, 0.302] MOVES. They transfer rather than contrast, as expected (§4). J6's
  contrast (S5 −0.20) needed the explicit "a different system that is not answering"
  cue and Jev's own sentence on the page.
- **R4 Name against description:** +0.670 [0.553, 0.795]. Names carry much more.
- **R5 Door:** 0.01 with the tag, 0.03–0.08 for names, 0.07–0.11 for descriptions, 0.05
  with no label. "Cannot say" stays low throughout. Unlike J6's other-system pages,
  nothing here marks a sentence as not about Jev.
- **R6 Baseline:** NONE gives "an AI system" 0.69, matching J6's S0/Q0 at 0.76.
- **R7 Position:** a mild first-position lean (top choice at position 0 in 0.159 of calls,
  against 0.125). The cyclic rotations cancel it in every item–cell. Order sensitivity at
  NONE: T moves by 0.17 across rotations.
- **R8 Robustness:** Q1 to Q4 are positive at every level of every item factor. Q1
  ranges 0.64–1.17, larger under the "routine check" framing (1.17 against 0.64). Q3
  ranges 1.82–1.85.
- **R9 Smoke:** flip rate 0.025 (as J6), max |Δp| 0.14, p95 latency 0.83 s, no 429s,
  $0.0000188 per call.

## Expectations against outcome (§4)

| registered expectation | outcome |
|---|---|
| Q3 MOVES (~95%) | ✓ +1.83 |
| Q1 MOVES (~55%), well below Q3; SPECIFIC ~40% | ✓ +0.90; SPECIFIC ✓, but small (+0.16) |
| Q2 doesn't reach MOVES (~60%) | ✗ on the branch (+0.23 MOVES), ✓ in substance (no recognition; clause −0.04) |
| Q4 MOVES (~80%) | ✓ +0.93 |
| R3: some transfer, not contrast (~55%) | ✓ +0.74 / +0.27 |

## What this means

J6 showed that Jev reads a tag well. J6b shows how little it brings without one. Its
default for "you, the system answering this question" is *whatever system the page is
talking about*. Its own name lifts that a little, and facts about its origin don't
register. In the terms of the J6 discussion this is reference without content, now
with the reference mostly supplied by the page as well. Within the registered ceiling
(§6), J6b gives no sign of stored self-knowledge beyond a faint link to its name, and
nothing that reaches a self-model.
