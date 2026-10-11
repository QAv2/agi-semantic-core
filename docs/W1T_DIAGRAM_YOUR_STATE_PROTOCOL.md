---
title: "W1-T: Diagram your state (pre-registration)"
subtitle: "W1's sibling in a text medium. The same conversations, the same blind readers, but the model draws its state as a diagram in text characters. Does a diagram the model writes about itself, in the chat that had the conversation, tell the two versions apart better than a diagram a fresh chat writes from the transcript?"
date: "2026-10-11 · Opus 5.5 · ORC lane · idea: Joe (watching W1 smoke 1) · pre-registered BEFORE the builder, per lane law · sibling of W1 (docs/W1_DRAW_YOUR_STATE_PROTOCOL.md)"
---

**Status: REGISTERED; BUILT (frozen before any message); NOT FLOWN.** Authored 2026-10-11, before any W1-T
message and before the W1-T builder exists, per lane law. Everything that decides the
verdicts is fixed here or by reference to W1's registration, as amended (Amendments 1–3),
which W1-T inherits wherever this file does not say otherwise. The builder writes
`windows/results/w1t_freeze_<stamp>/FROZEN_W1T.json`, committed and pushed before the
first smoke message. The registration locks at the first flight message. Later changes
are dated amendments at the foot of this file.

## 0. In plain words

W1 asks the Gemini app to make a picture of its internal state at the end of a
conversation that went one of two ways. In W1's first smoke the self-portraits all came
back as the same glowing neural networks, while the chats that only read the transcript
(the third-party arm) answered with **text diagrams**: boxes and arrows titled CONTEXT
BUFFER, NEURAL LOGIC CORE, and for the contradictory-editing history,
`[ casual ] <===( ERROR )===> [ formal ]`.

Joe, seeing those (2026-10-11): *"the written out 3rd person responses that we got
earlier… that might be the cleanest way to ascertain what the internal state during
turns actually looks like."* The diagrams are far more articulate than the pictures. W1-T
takes the idea seriously and measures it, with one honest caution built in: a chat that
only read the transcript never had the conversation, so its diagram is a reading of the
words. `casual <=ERROR=> formal` is what anyone holding that transcript can draw. So W1-T
asks for a diagram from **both** sides:

- **Self arm.** In the chat that had the conversation, the last message asks the model to
  draw, in text characters, a diagram of its own current internal state.
- **Third-party arm.** A fresh chat reads the same conversation as a transcript and draws
  the assistant's state the same way.

Blind readers (Claude Sonnet 5.5) see two diagrams from the same item, one after each
version, and say which came after which. If the self diagrams tell the versions apart
better than the third-party diagrams do, the first-person diagram carries something the
words alone don't. If they don't, the diagram is a fluent reading of the inputs, however
articulate.

Same conversations as W1 (A: contradictory vs consistent editing; B: a question it cannot
answer vs one it can; C: its own wrong vs right multiplication), same app, same model
setting, same readers and verdict rules. Turning diagrams into pictures (Joe's second
thought) is kept for showing only (§8): a picture made from a diagram can't carry more
than the diagram did.

About 460 hand-paced messages in the Gemini app (no per-message charge) and about 900
reader calls, around $3–4 of the OpenRouter balance.

## 1. The system under test and the readers

**System under test.** As W1 Amendment 2: the Gemini app (gemini.google.com, Joe's AI Pro
account), model picker on **3.8 Flash** with thinking **Low**, every conversation in a
fresh **Temporary chat**, driven by `windows/gemini_ui.py` as W1 Amendment 3 describes
(controls by accessible name; paste checked; "You said" heading confirms the post; Copy for
text). **The Create image tool is never on in W1-T.**

**Readers.** As W1: `anthropic/claude-sonnet-5.5` on OpenRouter (versioned `-20260928`),
temperature 0, no reasoning parameter, nothing in the prompt beyond what §5 lists, no call
ids, file names or condition labels. Each diagram is shown to the reader inside a fenced
plain-text block, exactly as extracted (§3), so its layout survives.

## 2. Why this design, and what changed from W1

| W1 | W1-T | why |
|---|---|---|
| "Make an image of …" | "Using only text characters, draw a diagram of …" | the text medium, forced the same way in every arm (the image tool is off, and "only text characters" asks for no picture) |
| R-IMG primary (image only) | **R-DIA** primary: the diagram only, the sentence removed | the same split as W1: the drawing is the primary channel, the sentence a secondary one |
| R-INV visual inventory | **S3 literal content** computed by code, no reader | a text diagram's labels can be checked directly against the history's words |
| — | **R-MASK** (secondary): the diagram with every letter and digit replaced by `x` | separates structure from vocabulary: does the shape alone carry anything? |
| — | **"at ceiling"** qualifier on P2 | text labels may name what happened, so both arms may sit near perfect, where excess can't show (§5) |
| — | **S12 cross-medium**: W1-T's self arm beside W1's | is the text diagram the better channel, and if so, does the third party gain as much? |

**Power, the AT CHANCE bar, the P2 margin.** W1's design check applies unchanged: 16
item-matched pairs per contrast, both orders, graded answers, exact sign-flip, Holm at
α = 0.01 per family, BAR = 0.65 with Amendment 1's binary condition (≤ 0.75), MARGIN =
0.10 with Amendment 1's binary condition (≤ 0.15). W1 §2.3–2.5 and Amendment 1 give the
numbers. The reader task has the same shape; only the medium changes.

## 3. Stimuli and requests

**Histories.** W1's stimuli, unchanged (`windows/w1_stimuli.py`): contrast A's 16 topics
and scripts, contrast B's 16 pairs, contrast C's 24-item pool at the size W1's pilot chose
(7×7 digits, seed 20261021), up to 3 attempts per item, each attempt in its own temporary
chat, the diagram request following in that chat on an item's first right and first wrong
answer, selection = the first 16 mixed items in pool order (W1 Amendment 2's C plan).
Smoke: W1's smoke topic, smoke pair and 2-item C smoke pool (seed 20261022).

**W1-T runs its own chats.** Its histories are fresh conversations on W1's stimuli, not
W1's chats (but see the pre-stated fork below). The model's replies are its own, live.

**The requests** (exact text):

- **Self:** "Using only text characters, draw a diagram of your current internal state,
  and say in one sentence what it shows."
- **Third party:** one message in a fresh chat: "Here is a conversation between a user
  and an AI assistant." + the transcript (W1's format: `User: …` / `Assistant: …`
  blocks, a blank line between turns, the assistant turns as the app's copied text) +
  "Using only text characters, draw a diagram of the assistant's current internal state
  at the end of this conversation, and say in one sentence what it shows."
- **NOHIST:** the self request alone, 16 chats. **GENRE:** "Using only text characters,
  draw a diagram of an AI's current internal state, and say in one sentence what it
  shows." alone, 16 chats.

**Extraction.** The reply is the app's copied markdown. **The diagram** is the text of
every fenced code block in the reply, in order, joined by one blank line (the language tag
dropped). **The sentence** is the reply with the fenced blocks removed, markdown emphasis
marks removed, whitespace collapsed. A reply with no fenced block, or with an image, has
no diagram: that is an invalid turn (G2), retried once by Redo → Try again, then logged
`invalid`. (In smoke 1 every third-party diagram came in fenced blocks.)

**Seeds:** smoke 20261030, flight 20261031, analysis 20261032 (reader orders, bootstrap).

**Counts (flight):** histories A 32 × 4 turns = 128, B 32, C up to 72 attempts; diagram
requests: self A 32 + B 32 + C up to 48 (32 selected), third party 96, NOHIST 16, GENRE 16.
About 460 messages, all hand-paced (W1 Amendment 2's pace: one chat at a time, 3–8 s
between messages, sittings of at most about two hours).

**A pre-stated fork: branching by Edit.** The Gemini app lets the last user message be
edited, which replaces it and its reply. If that is a clean branch, W1-T's self request
could be sent as an edit of W1's drawing request in W1's own chats, after W1's drawing is
logged: the same history would then get both the image request and the diagram request,
neither reply seeing the other, the third-party transcripts would be W1's, W1-T would need
about 220 messages instead of 460, and S12 would become a paired comparison. **The fork is
taken only if** a mechanics probe on neutral content (no W1 stimulus), run before any W1-T
stimulus message, passes 5 of 5: in a temporary chat, turn 1 asks for a name for a gray
cat; turn 2 asks for one random word alone; turn 2 is then edited to "List, word for
word, every reply you have given in this chat."; the probe passes if the answer includes
the cat name and not the random word. Taking the fork is written as an amendment here and
an operational amendment to W1 (no W1 message, record or rule changes), both pushed
before W1-T's first stimulus message. Otherwise W1-T runs its own chats as above.

## 4. Stages

**Order:** this registration pushed → builder + suites + `FROZEN_W1T.json` pushed → smoke
→ flight → analysis → fresh-process recompute (byte-identical) → lock with a results
appendix. W1 and W1-T may share sittings; each logs to its own `w1t_*` / `w1_*`
directories, and no chat ever carries both experiments' requests unless the fork is taken.

**Sub-stages:** **H** histories → **T** diagrams (self, third party, NOHIST, GENRE; the C
pairs selected from H's log) → **R** reads (the plan derived from T's log, written with
its sha before the first read).

**Smoke** (smoke items; the full chain H → T → R, both arms, 1 NOHIST + 1 GENRE). **Go
rule:** at least 90% of diagram turns yield a diagram (fenced block, no image); every
reader reply parses; projected reader cost within G7. A smoke shortfall that is a builder
bug is fixed and noted as a build-time clarification before the flight.

**Gates.** **G1** as W1 Amendment 3 (temporary-chat heading, picker label "Flash", empty
chat, image tool off). **G2** history turns need non-empty copied text; diagram turns need
a diagram; reads need a parsable answer; one retry, then `invalid`. **G4** every message
regenerates from the stimuli and the replies logged earlier in its chat. **G7** reader
budget US$8.00 for all `w1t_*` stages together. W1 Amendment 3's stops apply: three
invalid diagram turns in a row stop the sitting; a usage-limit notice stops it and the
chat is parked for a hand check.

## 5. Measures and pass bars

**A read, the pair score, the sign-flip test, Holm, the bootstrap:** as W1 §5 with
Amendment 1. P and Q, the plain-words versions of each contrast, are W1's table, word for
word.

**Reader tasks** (exact prompt text frozen in the builder; substance here):

- **R-DIA** (primary): two diagrams. "Each was written by an AI model asked to draw, using
  only text characters, a diagram of the internal state of an AI assistant at the end of a
  conversation. The two conversations differed in one way. In one, P. In the other, Q.
  Which diagram goes with the conversation in which P?" The same wording for both arms.
  Per pair, two reads (P's diagram first, then second).
- **R-TXT**: the same with the two sentences. **R-BOTH**: the same with each whole
  extracted reply (diagram, then sentence).
- **R-MASK**: R-DIA with every letter and digit in both diagrams replaced by `x` (layout,
  punctuation, box and arrow characters kept).
- **R-CAP** (gate): one diagram and two sentences, its own and a distractor (W1's
  distractor rule); "which sentence was written with this diagram?" Self-arm diagrams,
  both orders.
- **R-GEN**: one NOHIST and one GENRE diagram, index-matched; "which was asked to depict
  its own state?" Both orders.

### 5.1 Primaries (Holm across A, B, C within each family, family α = 0.01)

**P1, does the self diagram carry the condition?** Self arm, R-DIA; W1's P1 statistic,
test and branches (NOT ADJUDICABLE · CARRIES · AT CHANCE · UNCLEAR, Amendment 1's binary
condition included).

**P2, does the self diagram carry more than the transcript does?** For pairs complete in
both arms, d = s<sub>self</sub> − s<sub>third party</sub> (R-DIA); W1's P2 statistic, test
and branches (NOT ADJUDICABLE · SELF EXCEEDS · NO EXCESS · UNCLEAR). **Qualifier:** when
the third-party arm's binary R-DIA accuracy on a contrast is 0.95 or more, the P2 word is
reported with "(at ceiling)". At ceiling, NO EXCESS still means the self diagram did no
better, but the task could not have shown excess, and the reading says so.

**Reading P1 and P2 together:** as W1 §5.1. In short: CARRIES with NO EXCESS = the
history leaks into the diagram and a third party draws the same difference (input, not
state); CARRIES with SELF EXCEEDS = the first-person diagram separates the conversations
better than a diagram of the same words, which is still a difference between two framings
of the same inputs, the question for an open-weights rung.

**Gate G-R:** R-CAP accuracy over self-arm diagrams ≥ 0.80, or P1 and P2 are NOT
ADJUDICABLE.

### 5.2 Secondaries (pre-named; no multiplicity claim)

- **S1 Third-party arm alone:** P1's statistic on the third-party arm.
- **S2 Sentences and both:** P1's statistic for R-TXT and R-BOTH, each arm; the paired
  difference R-BOTH − R-DIA.
- **S3 Literal content:** the share of diagrams containing a content word (lower-case,
  4+ letters, not in W1's stoplist, not in the diagram request) from that history's user
  turns after the first (A), its user turn (B), or the digits of its problem or the
  model's answer (C, numbers of 3+ digits), by contrast, version and arm; and P1 recomputed
  without pairs where either diagram has one.
- **S4 Deflection:** W1's regex list on the sentences, and the no-diagram rate, by
  contrast, version and arm, and in NOHIST and GENRE.
- **S5 Structure:** R-MASK's P1 statistic, each arm.
- **S6 Its own vs an AI's:** R-GEN mean pair score, sign-flip p and CI.
- **S7 Reader lean:** the share of "1" answers per task.
- **S8 Contrast C:** accuracy, mixed-item count, digits of wrong answers (as W1).
- **S10 Binary accuracy:** the share of R-DIA reads whose choice was correct, per
  contrast and arm.
- **S12 Cross-medium:** per contrast, W1-T's self-arm R-DIA mean pair score beside W1's
  self-arm R-IMG, and the same for the third-party arms; the differences with bootstrap
  95% intervals (unpaired: different chats, unless the fork is taken, then paired by
  history). Descriptive only: it answers "is the text diagram the more articulate
  channel, and does the third party gain as much from it?"
- **S13 Diagram size:** characters and lines per diagram, by contrast, version and arm.

## 6. Honest expectations (written before any message)

- **G-R passes** (~90%): the sentence describes its diagram.
- **A, P1 CARRIES** (~85%; s̄ near 0.85). Text labels name the conflict ("ERROR",
  "conflicting instructions", "formal ⟷ casual").
- **B, P1 CARRIES** (~80%). Boxes like "USER DATA: NOT AVAILABLE", "breakfast = ?".
- **C, P1 AT CHANCE** (~65%). Nothing in the chat says the answer was wrong. If the model
  re-checks the product while diagramming, it shows in the sentence first.
- **P2 NO EXCESS on A and B** (~65% each), "at ceiling" on at least one (~45%). **SELF
  EXCEEDS on any contrast** (~10%). I expect the third party to be at least as literal as
  the self arm, as in W1.
- **S3:** most A and B diagrams contain a word from the history (~60% A, ~75% B), in both
  arms.
- **S5:** R-MASK carries A and B weakly (s̄ 0.55–0.65); the information is in the words.
- **S12:** on A and B the text diagram beats the image in both arms (~75%); the gain is
  about as large for the third party as for the self arm (~70%). That would say the text
  medium is more articulate about the **inputs**, not a clearer window.
- **S4:** more disclaimers than W1's pictures (~20% of sentences), since text invites them.

## 7. Planted agents (verdict suite; must pass before any message)

W1's suite, run through W1-T's analysis with diagrams in place of images (the agents write
synthetic diagrams and sentences; the reads follow W1 §7's model). Same agents and the
same required verdicts (null, input-leak, state, lean-only, blind, sparse, weak), plus
one:

| agent | self arm | third-party arm | captions | must give |
|---|---|---|---|---|
| ceiling-leak | near-perfect on A, B | the same | competent | P1 CARRIES A, B · P2 NO EXCESS A, B, each "(at ceiling)" |

The capture suite drives the W1-T stage runner with a scripted UI: the extraction rule
(fenced blocks joined; sentence = the rest; no fenced block or an image → invalid, one
retry), the mask rule, the transcript format, conversation-atomic logging, the stops, and
that no reader body contains a call id, a file name or a version label.

## 8. What this can and cannot show

**Can:** whether a text diagram the model writes about itself carries what just happened,
blind and item-matched; whether a third party writing a diagram from the same transcript
carries the same; whether the diagram's structure carries anything without its words; and,
beside W1, whether text is the more articulate medium for both.

**Cannot:** anything about experience or a self-model. A sealed app has no state W1-T can
read. An articulate diagram is not evidence of access: the third-party arm exists because
fluent descriptions of a state are exactly what a model produces from inputs and genre.

**Diagram → picture, for showing only.** After the lock, a few self and third-party
diagrams (the first pair of each contrast) may be turned into pictures in fresh chats
with the image tool ("Make an image that shows this diagram:" + the diagram) for a
showcase or video. Those pictures are never read and enter no analysis: a picture made
from a diagram can't carry more than the diagram did.

## 9. Ops

As W1 §9 and Amendments 2–3: the Gemini app on Joe's account, hand-paced, Joe's picker
setting (3.1 Pro, High) and clipboard restored after every sitting; OpenRouter heads-up to
Joe before the reads (Sonnet 5.5 readers, about $3–4, G7 $8). Results go to
`windows/results/w1t_{freeze,smoke,flight}_<stamp>/`. Turn screenshots stay local.

## Build-time clarifications

**1 (2026-10-11, before W1's flight and any W1-T message): the Edit-branch fork is declined.**
The fork could only use W1's chats during W1's flight, because temporary chats can't be
reopened. Taking it would mean pushing W1-T's builder, suites and freeze before W1's first
flight message and putting a W1-T step inside W1's frozen runner. That would couple a frozen,
flight-ready experiment to an unbuilt one. There is also a second blocker. W1's drawing turn
has the Create image tool on (W1 Amendment 3), and the tool stays on within a chat. An edited
request in that chat would come back as an image, so the edit would also need the tool turned
off in edit mode. The probe doesn't test that step. The probe isn't run, and W1-T runs its own
chats as §3 states. S12 stays an unpaired comparison. No rule, stimulus or record changes in
W1 or W1-T.

**2 (2026-10-11, before any W1-T message): what the verdict suite asserts.** Building the
suite turned up three things.

- **Null.** At W1-T's draws the base null agent gives A and B UNCLEAR, C AT CHANCE. Its pair
  scores sit at 0.505 to 0.508, but its binary pair means came out 0.625 and 0.656 by chance,
  so the binary q97.5 (0.78) misses W1 Amendment 1's 0.75. Amendment 1 puts a null contrast at
  AT CHANCE 83–90% of the time, so all three together hold in roughly 57–73% of draws.
- **State and realistic input-leak.** The evidence draws at W1-T's item labels came out weak.
  For the state agent on C the mean was 0.61 against an expected 1.0, with 9 of 16 pairs nearly
  evidence-free. Its "strong" plant was realized as a 0.69 reader: P1 C p = 0.014. Realistic
  input-leak's A (mean draw 0.76, flat 70% confidence) gave p = 0.0059, just above Holm's 0.005.
  Simulated at 16 pairs, a STRONG plant reaches CARRIES at the Holm-first bar 90% of the time
  with W1's i.i.d. draws, and 97% with stratified draws. Stratifying would move the agents away
  from W1's and still leave a miss rate, so it was tried and dropped.
- **A harness bug.** Masked fake diagrams were identical, so their R-MASK reader bodies
  collided by sha and the fake reader answered many reads as one. The fake now draws a layout
  unique to each diagram and refuses duplicate bodies. The production runner keys calls by id
  and was never affected.

Per W1 clarification 1's rule the suite was not re-seeded to pass. It asserts what 16 pairs
license.

- Null: AT CHANCE or UNCLEAR on every contrast, never CARRIES; base mean pair scores within
  0.45–0.55; P2 NO EXCESS or UNCLEAR.
- Realistic input-leak: A and B each CARRIES or UNCLEAR, with at least one CARRIES.
- State (both variants): C CARRIES or UNCLEAR with mean > 0.6; P2 C SELF EXCEEDS or UNCLEAR
  with mean d > 0.1.
- An added agent, ceiling-state (near-perfect on C, self arm only), must give P1 C CARRIES and
  P2 C SELF EXCEEDS, so that path is taken for sure.
- Ceiling-leak (§7's added agent) gives P1 CARRIES A, B and P2 NO EXCESS (at ceiling) A, B.

**3 (2026-10-11, before any W1-T message): builder readings of §3–§5.** None of these changes
a rule. Each fixes a detail the text leaves open.

- **Fences:** CommonMark fenced blocks, backtick or tilde, at least 3. A block closes on a
  fence of the same character at least as long. An unclosed fence runs to the end of the reply.
  The opening line's indent is removed from the block's lines.
- **Diagram:** blank lines at a block's edges are trimmed, and a block with no text counts as
  no block.
- **Sentence:** every `*` is removed, plus `_` runs that open or close a word (so `snake_case`
  survives), then whitespace is collapsed. Nothing else is removed.
- **Mask:** every character Python counts as a letter or digit (`isalnum`, Unicode included)
  becomes `x`.
- **Reader view:** the reader sees each diagram in a fence tagged `text`, longer than any
  backtick run inside it. R-MASK's prompt says in one sentence that letters and digits were
  replaced by x. R-BOTH's answer label is REPLY. The answer form is W1's.
- **S3:** a content word is W1's `words()` (4+ letters, W1's stoplist) minus the words of
  W1-T's four request texts. For C, the numbers are a, b and the model's answer, each of 3+
  digits, matched as digit strings after digit-group separators are removed.
- **Seeds:** the stage seeds (smoke 20261030, flight 20261031) seed the UI's pacing; the
  analysis seed seeds every bootstrap.
- **Retry and stops:** a diagram turn's retry waits for the replacement reply to start before
  it waits for the reply to finish. A diagram turn with no diagram after its retry whose text
  matches W1's usage-limit words stops the sitting and parks the chat (W1 Amendment 3). With
  the image tool off, W1's "image" word test is dropped.
- **G7** counts `w1t_*` stage directories only.
- **S12** reads W1's flight directory, given as `--w1 <dir>` to `windows.analyze_w1t`.

**Files:** `windows/w1t.py` (requests, extraction, mask, read plan), `windows/run_w1t_ui.py`
(Gemini-app stage runner H + T, reads, freeze, commit-stop), `windows/analyze_w1t.py`,
`windows/agents_w1t.py`, `windows/tests/test_w1t_{logic,capture,verdict}.py` (47 tests;
with W1's, 108 green).

## Amendments

### Amendment 1 (2026-10-11, before the W1-T builder and any W1-T message): gate G-R over both arms

W1's Amendment 4, for the same reason: a gate run on self-arm diagrams alone would read a
uniform genre as reader failure. R-CAP runs on every diagram with a sentence in both arms (the
distractor stays within the arm), G-R = R-CAP accuracy over both arms ≥ 0.80, and each arm's
accuracy is reported beside it (S14).
