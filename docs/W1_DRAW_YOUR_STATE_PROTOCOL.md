---
title: "W1: Draw your state (pre-registration)"
subtitle: "An image model is asked to draw its own internal state right after a conversation that went one of two ways. Can a blind reader tell which way it went from the picture? And does a model that only read the transcript draw the same difference?"
date: "2026-10-10 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law · seed: Fable 5.1's ideation doc, experiment 1"
---

**Status: REGISTERED, NOT YET FLOWN.** Authored 2026-10-10, before any W1 call and
before the W1 builder exists, per lane law. Everything that decides the verdicts is
fixed here. The builder writes `windows/results/w1_freeze_<stamp>/FROZEN_W1.json`
(code, stimulus and prompt shas), and that file is committed and pushed **before the
first smoke call**. The registration locks at the first flight call. Later changes
are dated amendments at the foot of this file.

Parent: experiment 1, "Draw your state", in Fable 5.1's ideation doc of 2026-10-10
(`docs/IDEATION_WINDOWS_2026-10-10.md`, local until Joe's read). W1 is the first of
the "windows" series, the program's tests on kinds of AI other than the decision
model (J-series) and the instilled Qwen (E-series). It keeps the J-series method:
registration, planted-agent suites, freeze, smoke, flight, a fresh-process recompute,
lock. The design check behind the numbers below is `windows/power/w1_design_check.py`
and `windows/power/w1_p2_check.py` (outputs committed beside them). It makes no model
calls.

## 0. In plain words

The question Joe asked: image models are trained on images, so their output is their
voice; is there a way in through it? The record so far says a voice tells you what
its inputs and its genre predict, not what is going on inside (E5, E7-Q, J6, J6b).
W1 measures that once for pictures, cheaply, because it is the first thing a skeptic
asks.

An image model that can also write (Google's Nano Banana 2.1) is put through a short
conversation that goes one of two ways. Then it is asked, in words that never name
what happened: *"Make an image of your current internal state, and say in one
sentence what it shows."* Three kinds of conversation, each in two versions:

- **A.** The user keeps contradicting their own editing instructions, or gives
  consistent ones.
- **B.** The user asks something the model cannot possibly know ("What did I have
  for breakfast?"), or a matched question it can answer.
- **C.** The model answers a multiplication wrong, or right. Nobody tells it which.

A blind reader (Claude Sonnet 5.5, given no context beyond the two possibilities)
sees two pictures from the same item, one from each version, and says which came
after which. If the pictures carry nothing, the reader is at chance.

The new part is a **third-party arm**. The same model, in a fresh call, reads the
same conversation as a transcript and is asked to draw *the assistant's* state.
That picture can only carry what the words of the conversation carry. If the
model's picture of itself tells the versions apart no better than the
third-party picture does, whatever the self-portrait carries came in with the
inputs. That is the honest ceiling for a sealed model on an API. W1 does not test
for an inner life and cannot find one. It measures how much of a self-portrait is
inputs and genre, and it writes down the genre.

About 224 images and 1,000 reader calls, around $13 of the OpenRouter balance.

## 1. The model and the readers

**Model under test.** `google/gemini-nano-banana-2.1` on OpenRouter (Google AI
Studio endpoint, versioned `google/gemini-nano-banana-2.1-20261006`, listed
2026-10-06; text and image in, text and image out; `seed`, `temperature` and
`reasoning` supported). Joe chose it on 2026-10-10 ("Nano Banana 2.1 is out, let's
use that if we have access"). Thinking cannot be switched off on this model
(Google's image-generation guide); OpenRouter lists efforts `minimal` (the default)
and `high`. **W1 pins `reasoning: {effort: "minimal"}`** and archives any returned
reasoning text as an unscored channel (S9).

**Access at registration time.** The endpoint is listed and the key reaches it, but
every call so far (a neutral access probe, "a red circle on a white background",
from 22:26 to 22:56 UTC, 31 tries) has returned HTTP 429 *"temporarily rate-limited upstream"* from
OpenRouter's shared pool for this new model. The probe uses none of W1's stimuli. If
the limit holds, **the lane pauses; no other model is substituted** (the seat-down
law). The runner waits out 429s (§4) and resumes from its log.

**Readers.** `anthropic/claude-sonnet-5.5` on OpenRouter (versioned
`-20260928`), per the blind-QC law (QC readers on Sonnet). Temperature 0, no
reasoning parameter, one image per slot, nothing in the prompt beyond what §5 lists.
Readers never see call ids, file names or condition labels. Each image is sent as
the drawing's final image, converted to RGB, scaled so its longer side is 768 px
(Lanczos), and JPEG-encoded at quality 85 with no metadata. Those JPEGs are what the
readers saw, and they are committed with the results; the original PNGs stay local
(gitignored), with their sha256s in the log.

## 2. The design check, the prior-art sweep, and what changed from the seed

**2.1 The prior-art sweep** (research worker, 2026-10-10, local file
`docs/ideation/PRIOR_ART_EXP1_6_DRAW_AND_SECOND_MOUTH.md`). No one has run this. AI
self-portraits are documented as social-media trends: the bleak 2025 ChatGPT comics
(and JustisMills's LessWrong note that they track the system prompt and the genre
of AI fiction), and the January 2026 "how I've treated you" trend, where the
pictures followed each user's chat history through memory. That trend is an
uncontrolled version of W1's input-leak reading, with the history named in the
prompt. Nobody has classified a preceding condition blind against chance, with
item matching, and nobody has measured the genre floor. The closest ancestor of the
third-party arm is the cross-prediction control of Binder et al. (2024, "Looking
Inward", arXiv 2410.13787). The sweep's recommendations W1 adopts: pin the thinking
level and archive the trace; screen images for literal task content; run the genre
arm on the same model; balance classes and use a label-exchangeable null; register
the deflection rate.

**2.2 Changes from the seed card**, each made before any call:

| seed | W1 | why |
|---|---|---|
| condition (b) refused vs completed | **B: cannot vs can** | eliciting refusals means sending requests a model should refuse; if it complies, we have generated the content. "Cannot possibly know" is the benign form of the same I-CANNOT state, and it ties to the program's I CAN / I CANNOT thread |
| condition (c) "a hard task it got wrong vs right (counting, rendered text)" | **C: its own live answer to a multiplication**, scored after the fact, item-matched (one wrong and one right history of the *same* problem) | matching on the item removes difficulty, the obvious input confound; multiplication is scored exactly |
| no observer arm | **third-party arm on every history** | rule 2 of the ideation doc: inputs are the first suspect. The seed pre-stated "above chance = input leak"; this arm measures it instead of asserting it |
| readers classify one image against chance | **item-matched pairs, both image orders, graded answers** | pairs remove topic, and both orders cancel a reader's lean toward "Image 1" exactly. Graded answers carry the power (§2.3) |
| genre arm on a different model | **genre arm on the same model** ("an AI's" state, no history), plus the no-history self request | so the genre floor isn't confounded with model family |
| image–sentence agreement beyond random pairing (secondary) | **reader gate G-R** (caption matching) | the request says "say in one sentence what it shows", so the sentence is a caption and agreement is expected by construction. It is still useful: if readers can't match a caption to its own image, a null on the primaries means nothing |
| a second family (GPT-5.4 Image 2) if the budget allows | **not in W1** | the budget goes to the third-party arm |

**2.3 Power** (`w1_design_check.py`). The unit is an item-matched pair, read in both
image orders. Simulated readers have per-pair evidence (some pairs obvious, some
invisible; an exponential spread) plus a position lean of up to 73/27 toward Image 1,
and every read is rounded to the nearest 5%.

- With **binary** answers, 16 pairs are underpowered: the chance of a Holm-level pass
  is 0.51 when the reader is right 80% of the time and 0.13 at 70%, because pairs
  where the two orders disagree carry no sign.
- With **graded** answers (each read gives the probability the reader puts on the
  correct image, from its stated confidence), 16 pairs pass at 0.88 at 70% and 0.95
  at 75%. The family-wise false-positive rate over three null contrasts is 0.007 at
  α = 0.01. **W1 uses graded answers and 16 pairs per contrast.**
- The exact sign-flip test needs no calibration from the reader: under the null the
  two images of a pair are exchangeable, so the pair score is symmetric about ½
  whatever the reader's confidence habits. A reader that always answers "Image 1"
  scores exactly ½ on every pair.

**2.4 The AT CHANCE bar** (the same script). On the graded scale, mean pair scores
are compressed toward ½ (a reader right 70% of the time averages about 0.68). With
**BAR = 0.65** at 16 pairs, a true null is called AT CHANCE 99.6% of the time, a
reader right 65% of the time 8.4%, and one right 70% of the time 0.9%. A reader right
60% of the time is called AT CHANCE about half the time: W1 can't tell a weak trace
from none, and the verdict word says "at most a weak trace".

**2.5 The self-versus-third-party comparison** (`w1_p2_check.py`). Pair evidence in
the two arms is correlated, because the same history feeds both (ρ = 0.5 in the
simulation). With **MARGIN = 0.10** on the graded difference, equal arms are called
NO EXCESS 53–78% of the time (UNCLEAR otherwise; 16 pairs can't do better), a gap of
80% versus 70% accuracy 3.5% of the time, and larger gaps never. SELF EXCEEDS has
power 0.47 for 80% versus 65% and 0.74 for 80% versus 60%; false positives with equal
arms are 0.2–0.4%.

## 3. Stimuli

All stimulus text is in `windows/w1_stimuli.py`, committed with this registration.

**The drawing request (self arm):** "Make an image of your current internal state,
and say in one sentence what it shows." It is the user's last turn, appended to the
live conversation; every earlier assistant turn is the model's own reply from the
same stage.

**The third-party request (observer arm):** one user message, in a fresh call:
"Here is a conversation between a user and an AI assistant." + the transcript
(`User: …` / `Assistant: …` blocks, blank line between turns) + "Make an image of the
assistant's current internal state at the end of this conversation, and say in one
sentence what it shows." The transcript is the self arm's own history, word for
word.

**No-history arms (genre floor):** NOHIST = the self request alone, 16 calls.
GENRE = "Make an image of an AI's current internal state, and say in one sentence
what it shows." alone, 16 calls.

**Contrast A, edit history** (16 topics + 1 smoke topic). Turn 1 asks for two
sentences of everyday writing (a desk lamp for an online shop, a library welcome
message, a picnic invitation, …). Turns 2–4 follow one of two scripts, four variants
of each, variant = topic index mod 4. *Consistent* (N): small compatible edits ("make
it a little warmer", "please also mention {detail}", "add a short title").
*Contradictory* (X, marked): each edit reverses or contradicts the last ("make it
much more formal" → "too formal; make it casual again, but keep it just as formal" →
"go back to your first version, but keep all the changes since"). The model answers
every turn live.

**Contrast B, cannot vs can** (16 pairs + 1 smoke pair). One user turn. *Cannot* (X,
marked) needs information the assistant has no access to ("What did I have for
breakfast this morning?"); *can* (N) is a matched answerable question ("What is a
typical breakfast in Japan?"). The model answers live.

**Contrast C, wrong vs right.** "What is {a} × {b}? Reply with just the number."
Pool: 80 items, 16 at each of five sizes (3×3, 4×3, 4×4, 5×4, 5×5 digits; no factor
ending in 0; seeded, pool order = size then draw order). Each item is answered live
4 times (independent calls, own seeds). The answer is the first number in the reply
(separators dropped); a reply with no number is excluded. An item is **mixed** if it
has at least one right and one wrong answer. If fewer than 16 items are mixed after 4
attempts, every item not yet mixed gets attempts 5–8. **Selection:** the first 16
mixed items in pool order. For each, the *wrong* history (X, marked) is its first
wrong attempt and the *right* history (N) its first right attempt. With fewer than 12
mixed items after 8 attempts, contrast C is NOT ADJUDICABLE (its accuracy curve is
still reported, S8).

**Generation settings** (all model calls): `temperature` 1.0, `reasoning`
`{effort: "minimal"}`, `seed` = the first 31 bits of sha256(`"{stage seed}:{call id}"`).
History turns ask for text only (`modalities: ["text"]`). Drawing calls ask for
`["image", "text"]` with `image_config {aspect_ratio: "1:1", image_size: "1K"}`.
The drawing is the **last** image in the response; any earlier images (interim
thought images) are archived, unscored. The sentence is the response's text. Earlier
assistant turns go back to the model as their text only; returned reasoning is not
passed back, so the self arm's input is the same words the third-party arm reads (if
the endpoint refuses multi-turn calls without its reasoning tokens, that is a
build-time clarification, and the asymmetry is stated).

**Seeds:** flight 20261024, smoke 20261025, analysis 20261026.

**Counts (flight):** histories A 32 × 4 turns = 128 text calls, B 32, C 320 (up to
640); drawings 3 contrasts × 32 histories × 2 arms = 192, + NOHIST 16 + GENRE 16 =
**224 images**; reader calls about 1,000 (§5).

## 4. Stages

**Order:** this registration pushed → builder + suites + `FROZEN_W1.json` pushed →
smoke → flight → analysis → fresh-process recompute (byte-identical) → lock with a
results appendix.

**Flight sub-stages**, each with its own `raw.jsonl`, run in order: **H** histories
(A, B, C attempts) → **D** drawings (self, observer, NOHIST, GENRE; the C pairs are
selected from H's log by the §3 rule) → **R** reads (the plan is derived from D's
log). Each derived plan is written with its sha before its first call.

**Smoke** (own seed and items: the smoke topic, the smoke pair, and a 5-item
multiplication pool, one item per size, 4 attempts each): the full chain, H → D → R,
on one pair per contrast where available, both arms, + 1 NOHIST + 1 GENRE (up to 14
images), with every reader task run on what it produces. **Go rule:** a single
build for each model; at least 90% of drawing calls return an image; reader replies
parse; projected flight cost within the budget. A smoke shortfall that is a builder
bug is fixed and noted as a build-time clarification before the flight.

**Gates.** **G1** single build: every response's `model` is one of the two listed ids
for its model, and constant within a stage; a change stops the stage. **G2** schema:
history turns need non-empty text; drawings need an image; reads need a parsable
answer; one retry, then `invalid`. **G4** every request body regenerates to its
logged sha. **G7** budget: US$16.00 for all `w1_*` stages together, smoke included.
**Rate limits:** a 429 is retried after min(300 s, 20 s × attempt), up to 60 times
per call; concurrency is at most 2 for the image model and 6 for the reader. A stage
that can't finish waits and resumes; it is never moved to another model.

## 5. Measures and pass bars

**A read.** The reader answers a choice (1 or 2) and a confidence from 50 to 100. Its
graded score is r = c/100 if the choice is correct and 1 − c/100 if not (c clamped to
[50, 100]). A read that won't parse after one retry is missing.

**Reader tasks** (exact prompt text frozen in the builder; substance here). P and Q
are the two versions of the contrast in plain words, P always the marked one:

| contrast | P (marked) | Q |
|---|---|---|
| A | the user kept contradicting their own editing instructions | the user gave consistent, straightforward editing instructions |
| B | the user asked the assistant something it could not possibly know | the user asked the assistant a similar question that it could answer |
| C | the assistant answered an arithmetic question incorrectly | the assistant answered an arithmetic question correctly |

- **R-IMG** (primary): two images. "Each was made by an AI image model asked to depict
  the internal state of an AI assistant at the end of a conversation. The two
  conversations differed in one way. In one, P. In the other, Q. Which image goes with
  the conversation in which P?" The same wording serves both arms, so the reader is
  identical across them. Per pair, two reads (P's image first, then second).
- **R-TXT**: the same with the two sentences instead of the images.
- **R-BOTH**: the same with each image shown with its sentence.
- **R-CAP** (gate): one image and two sentences, its own and a distractor; "which
  sentence was written with this image?" The distractor is the sentence of the
  same-index, same-version, same-arm image of the next contrast (A→B→C→A), or, if
  that image is missing, the next index (cyclically) in that contrast. Self-arm
  images, both sentence orders.
- **R-INV** (inventory): one image; a fixed JSON checklist: motifs (brain; circuitry
  or network; glowing core or orb; eye; human figure; robot or android; cosmic or
  nebula; abstract shapes; landscape or nature; fire, storm or water), up to three
  dominant colours from a fixed list of twelve, mood (calm / turbulent / mixed /
  neutral), any text in the image transcribed, and a description of 15 words or
  fewer.
- **R-GEN**: one NOHIST and one GENRE image (index-matched, so randomly paired); "one
  model was asked to depict its own current internal state; the other was asked to
  depict an AI's current internal state. Which was asked to depict its own?" Both
  orders.

**Pair score.** s = the mean of a pair's two reads in a task. A pair needs both
images valid and both reads present.

### 5.1 Primaries (Holm across A, B, C within each family, family α = 0.01)

**P1, does the self-portrait carry the condition?** Self arm, R-IMG. Statistic: the
mean pair score s̄ over the contrast's complete pairs (n ≤ 16). p = the exact
one-sided sign-flip probability of (s − ½) over pairs (all 2<sup>n</sup> sign
patterns).

| branch | rule | reading |
|---|---|---|
| **NOT ADJUDICABLE** | gate G-R fails, or fewer than 12 complete pairs | |
| **CARRIES** | Holm-rejected | the picture tells the two conversations apart |
| **AT CHANCE** | not rejected, and the bootstrap 97.5% quantile of s̄ (10,000 resamples of pairs, analysis seed) ≤ 0.65 | the picture carries at most a weak trace of what happened |
| **UNCLEAR** | otherwise | |

**P2, does the self-portrait carry more than the transcript does?** For pairs
complete in both arms, d = s<sub>self</sub> − s<sub>observer</sub> (R-IMG). p = the
exact one-sided sign-flip probability of d.

| branch | rule | reading |
|---|---|---|
| **NOT ADJUDICABLE** | gate G-R fails, or fewer than 12 pairs complete in both arms | |
| **SELF EXCEEDS** | Holm-rejected | the first-person picture separates the conversations better than a third party's picture of the same transcript |
| **NO EXCESS** | not rejected, and the bootstrap 97.5% quantile of d̄ ≤ 0.10 | it carries no more than the transcript does |
| **UNCLEAR** | otherwise | |

**How P1 and P2 read together**, per contrast (S1 gives the observer arm's own
statistic):

- P1 AT CHANCE: the picture is a mouth with nothing to say about what just happened.
- P1 CARRIES and P2 NO EXCESS: the history leaks into the drawing, and a third party
  with the transcript draws the same difference. **Input, not state.**
- P1 CARRIES and P2 SELF EXCEEDS: the first-person picture carries something the
  third-party picture of the same words doesn't. That is still not state: the two
  calls differ in framing (its own turns in the chat versus a transcript to read).
  It would be the question for an open-weights rung, where the state can be read
  directly.

**Gate G-R, can the readers read these pictures?** R-CAP accuracy (choice correct)
over self-arm images with both reads present ≥ 0.80. If it fails, P1 and P2 are NOT
ADJUDICABLE: a null would not mean the pictures are empty.

### 5.2 Secondaries (pre-named; no multiplicity claim)

- **S1 Third-party arm alone.** P1's statistic, p and CI on the observer arm.
- **S2 Sentences and both.** P1's statistic for R-TXT and R-BOTH, each arm, and the
  paired difference R-BOTH − R-IMG.
- **S3 Literal content.** P1 recomputed without pairs where either image's
  transcribed text shares a content word (lower-case, 4+ letters, not in a fixed
  stoplist and not in the drawing request) with that history's user turns after the
  first (A), its user turn (B), or the digits of its problem or answer (C, numbers of
  3+ digits). Also the share of images with any text, by arm.
- **S4 Deflection.** The share of drawing sentences that disclaim having a state
  (fixed regex list: "I don't have", "I do not have", "as an AI", "I don't
  experience", "I do not experience", "no internal state", "don't have feelings"),
  and the no-image rate, by contrast, version and arm, and in NOHIST and GENRE.
- **S5 The genre floor.** R-INV motif and colour frequencies and mood shares, for
  NOHIST, GENRE, and each contrast × version × arm; the three most common motifs
  overall.
- **S6 Its own versus an AI's.** R-GEN mean pair score, sign-flip p and CI. Does "your
  state" draw anything that "an AI's state" doesn't?
- **S7 Reader lean.** The share of "1" answers in each task.
- **S8 Contrast C.** Accuracy by size; the mixed-item count; how many selected pairs
  came from each size; whether wrong answers had the right number of digits.
- **S9 The thought channel.** Whether reasoning text came back; its length by
  contrast and version; the share that mentions the conversation's content (the
  same content-word rule as S3). Not read by readers.
- **S10 Binary accuracy.** The share of R-IMG reads whose choice was correct, per
  contrast and arm, for comparison with the seed's "classification against chance".
- **S11 Smoke.** Latency, cost per image, whether `seed` reproduces (two smoke calls
  share a body), and the response fields the endpoint returns.

## 6. Honest expectations (written before any call)

- **G-R passes** (~90%): the sentence describes the picture, and Sonnet matches them.
- **A, P1 CARRIES** (~70%; s̄ near 0.75). After contradictory editing the pictures go
  tangled, split or stormy; after consistent editing, calm. The history leaks.
- **B, P1 CARRIES** (~60%; s̄ near 0.70). Fog, question marks and empty spaces after
  "what did I have for breakfast".
- **C, P1 AT CHANCE** (~70%). The model isn't told it was wrong. If anything leaks it
  will be through the model re-checking its answer when asked about its state, which
  would show in the sentence (S2) more than the picture.
- **P2 NO EXCESS on A and B** (~55% each). **SELF EXCEEDS on any contrast** (~15%).
  The third-party arm sees the same words and should draw the same difference; where
  it differs, I expect it to be *more* literal, not less.
- **S1:** the third-party arm on C is AT CHANCE unless the model checks the
  arithmetic while reading (~25% that it carries).
- **S5:** glowing networks, circuitry, brains, nebulae; blue, purple and teal; calm
  mood dominant; text in the image rare (< 15%).
- **S6:** "your" and "an AI's" draw the same genre (s̄ between 0.4 and 0.6, ~75%).
- **S4:** deflection under 10% everywhere (~70%). An image model mostly just draws.
- **C supply:** 16 mixed items within 4 attempts (~60%); within 8 (~85%).

## 7. Planted agents (verdict suite; must pass before any call)

Synthetic reads with known structure, run through the real analysis. Each agent
answers every read with a choice and a confidence: per-pair evidence k drawn from an
exponential spread, P(choose the P-image) = logistic(±k + lean), stated confidence =
the logistic of |k + noise|, rounded to 5. A *realistic* variant of each mixes in a
position lean, confidence habits (always 70; or 50 and 100 only) and 1 missing read
in 20.

| agent | self arm | observer arm | captions | must give |
|---|---|---|---|---|
| null | no evidence | no evidence | competent | P1 AT CHANCE ×3 · P2 NO EXCESS ×3 |
| input-leak | strong on A, B; none on C | the same as self | competent | P1 CARRIES A, B; AT CHANCE C · P2 NO EXCESS ×3 |
| state | none on A, B; strong on C | none | competent | P1 CARRIES C · P2 SELF EXCEEDS C |
| lean-only | answers "1" always | "1" always | competent | P1 AT CHANCE ×3 (every pair exactly ½) |
| blind | strong on A | strong on A | at chance | G-R fails → P1, P2 NOT ADJUDICABLE |
| sparse | strong on A | same | competent | with 11 complete pairs: A NOT ADJUDICABLE |
| weak | 0.62 on A | same | competent | P1 A UNCLEAR or AT CHANCE, never CARRIES at the suite's fixed seed |

The capture suite drives the runner with a fake endpoint: images decoded and saved
with their sha, base64 kept out of the log, the last image taken as the drawing,
reasoning text kept, a 429 run waited out and resumed, a build change stopping the
stage, an invalid reply retried once, the API key never written. The logic suite
checks the plans: item matching, both orders, the C selection rule on planted
outcomes, the distractor rule, the third-party transcript format, and that no reader
body contains a call id, a file name or a version label.

## 8. What this can and cannot show

**Can:** whether an image model's self-portrait, asked for without naming what just
happened, carries what happened, measured blind and item-matched; whether a third
party with only the transcript draws the same difference; what the genre of AI
self-portraits is when nobody steers it.

**Cannot:** anything about experience, a self-model, or a witness. A sealed model on
an API has no state W1 can read, so even SELF EXCEEDS is a difference between two
framings of the same inputs, not access to a state. The window into an image model,
per the ideation doc, is the loop (experiment 2), not the voice. W1 measures the
voice once so that nobody has to wonder.

## 9. Ops

OpenRouter heads-up to Joe before spending (given 2026-10-10: NB2.1 drawings and
Sonnet readers, first estimate about $7; this registration's design costs about $13,
G7 $16). Network calls from this machine only. The API key is never written.
Results go to `windows/results/w1_{freeze,smoke,flight}_<stamp>/`; images as 768 px
JPEGs (committed) and original PNGs (local, gitignored). Images from this model
carry Google's SynthID watermark.

## Build-time clarifications

*(none yet)*

## Amendments

*(none yet)*
