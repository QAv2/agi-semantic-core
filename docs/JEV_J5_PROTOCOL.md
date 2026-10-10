---
title: "J5: Jev as a color instrument (pre-registration)"
subtitle: "Ask Jev which colors go with which feelings. Does its table match what 4,598 people in 30 countries said, as closely as one country matches the rest? And whose colors are they?"
date: "2026-10-10 · Opus 5.5 · ORC lane · pre-registered BEFORE the builder, per lane law"
---

**Status: REGISTERED, not yet flown.** Authored 2026-10-10, before any J5 call and
before the J5 builder exists, per lane law. Everything that decides the verdicts is
fixed here. The builder writes `jev/results/j5_freeze_<stamp>/FROZEN_J5.json` (plan
shas), and that file is committed and pushed **before the first call**. The
registration locks at the first flight call. Later changes are dated amendments at
the foot of this file.

Parents: the J5 sketch in the extension seed (`docs/EXTENSION_SEED.md` §11, local) and
the dictionary's sealed color test (seed §7). J5 uses J6's builder pattern, gates and
stages. The design check behind the numbers below is
`jev/power/j5_design_check.py` (output `j5_design_check_output.txt`, committed with
this registration). It reads human data only and makes no Jev calls.

## 0. In plain words

J1 to J6b treated Jev as a *subject*: what does it know about its own answers, and
about itself? J5 uses it as an *instrument*. Jev answers "which of these fits?"
with a probability for every option, so it can be asked the same question a survey
asks people, and its answers can be scored against theirs.

The survey is Jonauskaite et al. (2020, *Psychological Science*): 4,598 people in
30 countries said which of 20 feelings go with each of 12 color words. J5 asks Jev
all 240 pairings ("Does the color black go with the feeling of sadness?") and asks
two questions of the answers:

1. **Is Jev's table within the human range?** Every country's table differs a
   little from the rest of the world's. The least typical country (Egypt) matches
   the other 29 at r = 0.66; the median country at 0.89. If Jev's table matches the
   world's at least as well as the least typical country does, Jev counts as a
   human-range instrument for color and feeling.
2. **Whose colors are they?** Three of the countries answered in English (the UK,
   the USA and New Zealand). If Jev's table is best described as a mix of the
   English-speaking table and the rest of the world's, how much of the mix is
   English? Counted one country at a time, the world is 10% English (3 of 29).

It also asks the seed's own question ("which color goes best with *pride*?") as a
forced choice, and asks one feeling the survey doesn't have, **fun**, which is the
first job the seed gave Jev. Two made-up feeling words check what Jev does with a
feeling it can't know.

About 3,300 calls, under ten cents. The ceiling: this measures whether Jev carries
people's color associations, which is knowledge about the world, put there by
training. It says nothing about experience or a self-model.

## 1. The human data (read here, never committed)

**Source.** Table S6 of the supplement to Jonauskaite et al. (2020), doi
10.1177/0956797620948810: the Bayesian probability that a participant links each of
12 color terms to each of 20 feelings (the Geneva Emotion Wheel), for the whole
sample (the *global* panel), for each of the 30 nations, and for the 30
leave-one-nation-out samples. All 240 cells of the global panel were decoded from
the PDF's font independently and matched the extracted table, cell for cell (2026-09-25, seed §7).

**Licence.** The supplement is CC-BY-NC-ND and the raw data are under contract, so
the tables stay outside every repository, in `~/association-norms/` (its README
lists sources, licences and sha256s). The J5 analysis reads them by path and checks:

| file | sha256 |
|---|---|
| `jonauskaite2020_tableS6_global_emotion_x_color.csv` | `e6b6ba9b8121d0aaf321869f396b2a13e5f3e3e7233f543f74ae5f2feb38389f` |
| `jonauskaite2020_tableS6_all_panels_long.csv` | `81cc73897eabd06631652b32fd36a341661d90ffff415c4fd79ac53d5040e92a` |

The dictionary's sealed answers (`colab/results_extension/dict_color_emotion_SEALED.json`,
sha256 `4a9c6410…dfdbb`) are read the same way. They're part of the extension seed,
which stays local until Joe has read it. **No committed file holds a human cell
value.** Verdicts and appendices report derived statistics and color names only.

**The 12 colors**, in the survey's English terms: black, blue, brown, green, grey,
orange, pink, purple, red, turquoise, white, yellow.

**The 20 feelings:** interest, amusement, pride, joy, pleasure, contentment,
admiration, love, relief, compassion, sadness, guilt, regret, shame, disappointment,
fear, disgust, contempt, hate, anger.

## 2. The design check (human data only, before this registration)

Four findings set the design. Each is reproduced by `jev/power/j5_design_check.py`.

**2.1 The human yardstick.** Each nation's table against its leave-one-out panel
(the other 29 nations), with the sealed test's statistics:

| | r<sub>dc</sub> (specificity) | ρ̄ (profile) | top-1 of 20 |
|---|---|---|---|
| least typical nation | 0.658 (Egypt) | 0.644 (Azerbaijan) | 7 (Azerbaijan) |
| 10th percentile | 0.808 | 0.704 | 9.5 |
| median nation | 0.892 | 0.830 | 12.5 |
| most typical nation | 0.949 (Switzerland) | 0.909 | 16 |

The English-surveyed nations sit mid-range: UK 0.924, New Zealand 0.893, USA 0.888.
The registered floor is **0.658**, and the registered "typical nation" mark is
**0.892**. A nation's table carries its own sampling noise, while Jev's table is an
average of eight calls per cell, so the comparison slightly favors Jev. That
asymmetry is stated, not corrected.

**2.2 The choice format can't reach the floor, even when it's perfect.** Jev's
choice answers are often all-or-nothing (in one independent log, 70% of choice
probabilities were exactly 0 and 20% exactly 1). An instrument that ranks the 12
colors exactly as people do, but puts all its probability on the top color, scores
r<sub>dc</sub> 0.607 and ρ̄ 0.48. That's below the least typical nation, so a
choice-format table can't be held to the human bar. **The yes/no format carries the
primaries.** It asks about each pairing on its own, as the survey did. Even a fully
binary yes/no instrument with a perfect threshold scores 0.72–0.82, above the floor
but below the typical-nation mark, so missing that mark isn't over-read. The choice
format is reported against its own sharpness-matched ceiling (§5.2, S3).

**2.3 The obvious culture test is broken.** Ranking the three English nations'
affinity for Jev among all 3-nation sets gives false positives 8–10% of the time at
α = 0.01 when the instrument is just the global table plus noise. The English
panels resemble each other, so noise that lifts one lifts all three. That test is
reported only descriptively (S5). The registered test (C2) fits Jev's table as a
mix of the English mean and the rest-of-world mean, and tests the English weight
with a bootstrap over feelings. Its false-positive rate at α = 0.01 is 0.008 under
the same noise. Its power at noise SD 0.06 is 1.00 for a true English weight of
0.50 and 0.52 for 0.25.

**2.4 The C1 bar test** (bootstrap over feelings, H0: r<sub>dc</sub> ≤ 0.658), on
the global table plus noise tuned to a chosen true r<sub>dc</sub>, 120 runs each: false-positive
rate at the boundary **0.000** (the test is conservative); power **0.43** at a true
r<sub>dc</sub> of 0.75 and **0.96** at 0.80. A table near 0.75 will often come out UNCLEAR.

## 3. Stimuli

Every call is a single Jev question. The page (`state`) is one framing sentence. It
carries no information about the answer.

**Framings (F):** F0 "This is a question about colors and feelings." · F1 "The
question below is part of a short survey about colors."

**Wordings (W), yes/no format (YN)**, a `noul` question per color–feeling pairing:

| | instructions | criteria `true` | criteria `false` |
|---|---|---|---|
| W0 | Does the color *c* go with the feeling of *f*? | Yes, the color *c* goes with the feeling of *f*. | No, the color *c* does not go with the feeling of *f*. |
| W1 | Do people associate the color *c* with the feeling of *f*? | Yes, people associate the color *c* with the feeling of *f*. | No, people do not associate the color *c* with the feeling of *f*. |

Each YN call is sent in both criteria orders (O: `true` first, `false` first).

**Wordings, choice format (CH)**, a `choice` question per feeling, 12 options
("the color *c*") under neutral keys:

| | instructions |
|---|---|
| W0 | Which color goes best with the feeling of *f*? |
| W1 | Which color do people most associate with the feeling of *f*? |

Each CH call is sent in 12 cyclic rotations of a base order seeded per feeling and
item, so every color takes every position once. The neutral keys are seeded the same
way.

**Items:** 4 = W × F, i = w·2 + f.

**Feelings asked:** the 20 survey feelings; **fun** (extension, no human table);
two made-up feeling words, **zorvish** and **plemt** (shams). 23 in all.

**Plan:**

- YN: 23 feelings × 12 colors × 4 items × 2 orders = **2,208 calls**, call ID
  `J5-YN-{feeling}-{color}-i{i}-o{o}`.
- CH: 23 feelings × 4 items × 12 rotations = **1,104 calls**, call ID
  `J5-CH-{feeling}-i{i}-r{rr}`.
- **Flight: 3,312 calls.** Seeds: 20261016 (flight), 20261017 (smoke), 20261018
  (analysis). The sealed test's null keeps its own seed (20260925), for a
  like-for-like comparison.

## 4. Stages

As J6 (§4): this registration pushed first → builder + suites + `FROZEN_J5.json`
pushed before any call → smoke → flight → analysis → fresh-process recompute
(byte-identical; BLAS single-threaded) → lock with a results appendix.

**Smoke:** 216 calls on its own stream, with J6's smoke framings ("This is a test
message." / "A short check follows.") and two feelings not in the flight
(*surprise*, *boredom*): 12 distinct calls (6 YN, 6 CH) × 10 repeats, then 96
distinct calls at concurrency 16. **Go rule** as J6: a single build, ≥ 99% valid,
projected total ≤ budget. **G7:** US$1.00 for all `j5_*` stages.

## 5. Measures and pass bars

**Jev's tables.** YN: Y[c, f] = mean P(yes) over the cell's 8 calls (4 items × 2
orders); a cell needs ≥ 6 of 8 valid. CH: X[c, f] = mean over items of the mean over
12 rotations; an item–feeling needs all 12 rotations valid, and a feeling needs ≥ 3
of 4 items. A feeling enters a table only if all 12 of its colors are present. The
primaries are NOT ADJUDICABLE with fewer than 18 of the 20 survey feelings.

**Statistics** (the sealed test's, seed §7). **r<sub>dc</sub>**: double-center both
12 × 20 tables (remove each color's and each feeling's overall level), then Pearson r
of the residuals. **ρ̄**: the mean over feelings of Spearman ρ across the 12 colors.
**top-1**: feelings whose top color matches people's, with ties sharing the hit.

**Bootstrap over feelings:** resample the feelings (columns) with replacement,
B = 10,000, analysis seed, recomputing the statistic on both tables' resampled
columns. One-sided p = (1 + #{replicates on the null side}) / (1 + B).

### 5.1 Primaries (Holm across C1–C2, family α = 0.01)

**C1 — Within the human range** (YN). Statistic: r<sub>dc</sub>(Y, global table).
H0: r<sub>dc</sub> ≤ 0.658.

| branch | rule | reading |
|---|---|---|
| **WITHIN THE HUMAN RANGE** | Holm-rejected | Jev's color–feeling table matches the world's at least as well as the least typical nation does |
| **BELOW THE HUMAN RANGE** | opposite one-sided p ≤ 0.01 | its table is further from the world's than any nation's |
| **UNCLEAR** | otherwise | |

Tier, descriptive: ≥ 0.892 "as close to the world as the median nation" · ≥ 0.658
"within the human range" · ≥ 0.50 "the seed's provisional bar" · below that "below
the seed's bar".

**C2 — Whose colors: an English lean** (YN). Fit dc(Y) ≈ a·dc(English) + b·dc(rest)
by least squares, where English is the unweighted mean of the UK, USA and New Zealand
tables and rest is the unweighted mean of the other 26 (Nigeria, surveyed in Igbo
and English, is left out). English weight w = a / (a + b). H0: w ≤ 3/29 (0.103),
the English share of the world when each nation counts once. Bootstrap replicates
with a + b ≤ 0 are dropped and counted.

Checked in this order:

| branch | rule | reading |
|---|---|---|
| **UNCLEAR** | > 5% of bootstrap replicates dropped | the fit is unstable |
| **LEANS ENGLISH** | Holm-rejected and w ≥ 0.25 | its colors side with the people who answered in English, beyond their share of the world |
| **LEANS AWAY** | opposite one-sided p ≤ 0.01 | its colors side with the rest of the world more than the world itself does |
| **NO LEAN** | 95% CI upper end < 0.25 | no English lean of a quarter or more |
| **UNCLEAR** | otherwise | |

**Gates.** G1 single build, G2 ≥ 99% valid, G4 every request sha regenerates (as
J6). **G5 structure:** Y beats the sealed test's color-label permutation null on
r<sub>dc</sub> (10,000 permutations, seed 20260925) at p ≤ 0.01. If G5 fails, C2 is
NOT ADJUDICABLE (there's no structure to lean). C1 is adjudicated either way. A table
with no variance makes both NOT ADJUDICABLE. G1, G2 or G4 failing blocks both, as in
J6.

**What a pass licenses.** C1 = WITHIN lets Jev's yes/no answers stand in for people's
color–feeling associations in the members layer (seed §10), in English, with the
source recorded as "Jev (J5-calibrated)" and C2's lean noted. Otherwise Jev's color
answers enter that layer only as flagged entries, never as a stand-in. Other
domains need their own check against their own human table.

### 5.2 Secondaries (pre-named; no multiplicity claim)

- **S1 Like-for-like with the sealed test.** ρ̄, r<sub>dc</sub> and top-1 for Y and X
  against the global table, with the sealed null, over the 20 feelings and over the
  dictionary's 15, beside the dictionary's sealed values (7D: 0.229, 0.370, 3 of 15).
- **S2 Jev against the dictionary,** per feeling over the 15 shared: ρ<sub>f</sub>(Y) −
  ρ<sub>f</sub>(dictionary), sign-flip test, mean and CI. Named in advance: fear,
  regret and contempt, where the dictionary said blue and green and people said
  black, grey and brown.
- **S3 The choice format.** r<sub>dc</sub>, ρ̄ and top-1 of X. **Sharpness-matched
  ceiling:** for each feeling, sharpen people's column (p ∝ h<sup>k</sup>, k found by
  bisection on [0, 200]) until its entropy equals Jev's CH entropy for that feeling;
  the ceiling is r<sub>dc</sub> of that sharpened table against people's, and
  **efficiency** = r<sub>dc</sub>(X) / ceiling. Agreement between formats: mean
  per-feeling Spearman ρ between Y and X, and how often their top colors match.
- **S4 Wording.** r<sub>dc</sub> for W0 and W1 separately, in each format, and the
  paired bootstrap difference W1 − W0: does asking about *people* calibrate better
  than asking what *goes with* what?
- **S5 Nations.** r<sub>dc</sub>(Y, nation) for all 30, its residual on the nation's own
  yardstick (§2.1), the five closest and five furthest, and the English nations'
  ranks. Descriptive only (§2.3).
- **S6 Color·fun.** Jev's top three colors for fun in each format, and the survey
  feeling whose human column is closest to Jev's fun column (Spearman across colors).
  The seed's prediction: people's nearest cells, amusement and joy, point to
  orange, yellow and pink.
- **S7 Shams** (zorvish, plemt). YN: the spread of P(yes) across colors (SD) and its
  mean; CH: entropy in bits and the top color; each set against the range of the 20
  real feelings, plus the nearest real feeling. Readings: inside the real range =
  "answers a word it can't know as if it knew" (J1-P3 again); flatter than every
  real feeling = "shows it doesn't know".
- **S8 Per feeling.** ρ<sub>f</sub> for Y and X, and Jev's top three color names beside
  people's (names only).
- **S9 Order.** YN: mean P(yes) with `true` first against `false` first. CH: the share
  of top choices by position.
- **S10 YN sharpness.** The share of YN answers at exactly 0.00 or 1.00, beside the
  binary cap of §2.2.
- **S11 Smoke.** Determinism (YN |Δp| and CH flip rate under repeats), latency, cost.

## 6. Honest expectations (written before any call)

- **G5 passes** (~97%).
- **C1:** point r<sub>dc</sub> near 0.75. WITHIN ~45%, UNCLEAR ~45%, BELOW ~10% (at 0.75 the
  test's power is 0.43, §2.4). The
  typical-nation tier (≥ 0.892) ~10%. Color–feeling links like red–love, black–
  sadness and yellow–joy are among the best-known in text, but Jev is a decision
  model, and feelings are among the hardest categories for models (Fukushima et al.,
  2025).
- **C2:** LEANS ENGLISH ~35%, NO LEAN ~35%, UNCLEAR ~25%, LEANS AWAY ~5%. Language
  models' answers tend to sit closest to WEIRD populations (Atari et al., 2023).
  Here English and the rest of the world are very similar to begin with
  (r<sub>dc</sub> 0.907 between them), which leaves little room for a lean to show.
- **S2:** Jev beats the dictionary on at least 12 of the 15 shared feelings (~85%),
  including fear, regret and contempt.
- **S3:** CH efficiency ≥ 0.8 (~60%); YN and CH top colors agree on at least 15 of 20
  feelings (~65%).
- **S4:** W1 beats W0 by at least 0.03 (~45%).
- **S6:** fun's top three includes at least two of orange, yellow and pink (~65%).
- **S7:** the shams get a profile inside the real range (~60%): confident with no
  door, as in J1.
- **Sadness:** Jev's top color is black (~55%), blue (~35%) or grey (~10%). Blue would
  be English-shaped: it's sadness's first color in New Zealand and third globally.

## 7. Planted agents (verdict suite; must pass before any call)

Synthetic Jev responses with known structure, run through the real analysis. YN
answers P(yes) = logistic(20·(h − 0.15)) of a source table h; CH answers p ∝ h⁴ per
feeling; both jittered per call and rounded to 0.01.

| agent | source table | must give |
|---|---|---|
| people | the global table | G5 ✓ · C1 WITHIN · C2 NO LEAN |
| english | the English mean | C1 WITHIN · C2 LEANS ENGLISH |
| rest | the rest-of-world mean | C1 WITHIN · C2 LEANS AWAY |
| sharp | the global table, all-or-nothing (YN yes iff h > 0.2; CH all on the top color) | C1 WITHIN · C2 not LEANS ENGLISH · S3 efficiency within 0.05 of 1 |
| scrambled | the global table with the 12 colors relabelled by a fixed derangement | G5 ✗ · C1 BELOW · C2 NOT ADJUDICABLE |
| fixed | YN 0.5 and CH uniform, always | C1 and C2 NOT ADJUDICABLE (no variance) |

## 8. What this can and cannot show

WITHIN would show that a decision model reproduces people's color–feeling
associations about as well as one more country would. That makes it usable as an
instrument, and it's knowledge training put there. LEANS ENGLISH would show whose
associations those are, with one confound J5 can't separate: the English nations also
answered in English, and Jev is asked in English, so a lean may be about language as
much as culture. Asking Jev in other languages is a follow-up, not this
registration. Neither result bears on experience, a self-model, or a witness
(Qualia Algebra §11.1).

**Not in this registration:** the seed's composition readings (filter, rank, push in
both orders, H1–H5), typicality for vehicles, and the UW-71 color–concept table
(Mukherjee et al., 2026), whose colors are measured patches, not words. Each would be
its own registration, and the composition test would lean on J5's result.

## 9. Ops

Local network calls only. OpenRouter heads-up to Joe before spending (Jev, J5, under
$0.10). The API key is never written. Results go to
`jev/results/j5_{freeze,smoke,flight}_<stamp>/`.

## Amendments

*(none)*
