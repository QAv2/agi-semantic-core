---
title: "A mouth that can only choose"
subtitle: "Scoping Jev for the program: what a decision model can and can't do here, three designs, and the hold"
date: "2026-09-24 · Opus 5.5 · ORC lane · scoping only · J1 + J2 pre-registered 2026-09-25 in docs/JEV_J1J2_PROTOCOL.md (binding)"
---

::: answer
## The short answer

- **What Jev can be here: a subject.** It's a system that can't talk, only choose. Its probabilities can be tested for metacognition, meaning whether they know when Jev is out of its depth, the way comparative psychology tests animals that can't talk. Early independent tests say they don't, away from its home ground. J1 would settle that properly. **Possibly also a component:** a fast, calibrated gate that reads a readout of the language model's *state*, not its words.
- **What it can't be: evidence about an observer.** Jev is a sealed box: text goes in, a choice and its probabilities come out, with no internal state, embeddings or explanations (verified in the vendor's docs). By the paper's own rule (§11.6), an instrument that counts must read state. And with no memory between calls, it has no persistent observer (§11.3) to host a witness.
- **Joe's coupling question: yes, but couple it to the model's state, not its words.** If a language model writes the situation Jev decides on, the genre travels inside the situation. That launders the walk's problem instead of escaping it.
- **Three designs, all GPU-free, runnable once a key exists. A fourth waits for compute.**
  - **J1: Does its confidence know its competence?** Type-2 metacognition, with the dissociation that separates self-monitoring from reading the evidence.
  - **J2: Negation without a mouth.** The neti neti ladder for a decision model. Does it release the content, and hold only when nothing is left, even when "hold" has no contemplative name?
  - **J3: The monitor at threshold.** Can Jev read the substrate's state readout zero-shot, as well as a fitted classifier, and stay silent on shams?
  - **J4 (later): the state-reading examiner.** An arm of the adaptive walk (E7b-Q v2).
- **Access is open and cheap.** OpenRouter serves Jev self-serve today, with no waitlist. The whole J1–J3 program comes to roughly $5–10 of calls, with no GPU.
- **The hold (resolved 2026-09-25): lifted.** The comprehensive edition was minted (doi:10.5281/zenodo.22968135) and this repository pushed before any Jev pre-registration, so the pre-registrations carry public timestamps.
:::

## 1. What Jev is

*Checked 2026-09-24 against three kinds of source: the vendor's documentation, the JevPilot source code, and four independent test repositories, all a few days old. Tags: <span class="tag v">verified</span> means docs or code, <span class="tag m">claimed</span> means vendor marketing, <span class="tag u">unknown</span> means not published.*

**Three corrections to the seed.**

- **The founder is Diogo Almeida**, not Diego.
  - Co-founders: Erik Gafni (CTO) and Sasha Sheng (COO).
  - Almeida is an author of InstructGPT; "co-invented RLHF" is the vendor's framing.
  - TypeSafe AI is in San Francisco, with a $40M seed round led by DCVC.
  - Jev launched on **2026-09-15**. The 09-18 date is the OpenRouter listing and MindStudio's coverage.
- **`simple-jev.featherless.ai` is not Jev.** It is Featherless's open-source clone, running a Qwen-based classifier, and by its own statement it doesn't reproduce TypeSafe's model. The real driving demo is JevPilot (`standardagents/jevpilot`). JevPilot reads Jev's full probability table but never reads its `confidence` field, and it always acts on the top choice.
- **"Non-autoregressive" is MindStudio's word.** The vendor says "parallel: generates all outputs in a single query". The CEO says "technically not a language model". The architecture is undisclosed and there is no paper yet <span class="tag u">unknown</span>.

**The interface** <span class="tag v">verified</span>. There is one endpoint, `POST /v1/systemone`.

- **Input.** You send a `state`: text, or a JSON object or array, up to 32k tokens. With it you send a map of `questions`, each evaluated in parallel and **in isolation** against the same state.
- **Three question types:**
  - **Choice:** up to 255 options. The caller writes each option's name and description, and the model sees both. It returns the choice, the **full probability distribution over every option**, and a confidence.
  - **Score:** 2–10 ordered levels. It returns a score, which can fall between levels, the distribution, and a confidence.
  - **Noul:** yes or no. It returns only the probability of yes.
- **What's absent:** temperature, seed, logprobs, embeddings, internal state and explanations. Input is text only, and nothing carries over between calls.

**What "confidence" is** <span class="tag v">verified</span>. It is not a second signal. The docs call it "a statistic computed from the probability distribution the answer already gives you". Every published example matches (n·p<sub>max</sub> − 1)/(n − 1) to within rounding. That is the top probability, rescaled so that a uniform distribution scores 0 and certainty scores 1. So for Jev, "does its confidence know its competence" means "does its probability distribution know its competence".

**Calibration.** The vendor claims it <span class="tag m">claimed</span>: "outcomes assigned a probability of 0.8 should occur about 80% of the time", trained by "Reinforcement Learning for Calibrated Decisions". The vendor publishes no calibration numbers. The independent tests, all under ten days old:

- **Good on familiar ground.** ECE 0.024–0.032 on public question-answering sets.
- **Poor off it.** On a rule-based task whose answer depended on a policy not given in the text, Jev scored 44.7% accuracy at a mean stated probability of 0.74. It is **confidently wrong where it can't know.**
- **A phishing benchmark:** 62.6% accuracy and ECE 0.154, against Claude Haiku 4.5 at 81.3% and 0.097.
- **An exploration log:** "Probabilities are calibrated: refuted at scale"; confidence "is not a correctness estimate".
- **The vendor's own caveats page:**
  - A yes/no question and its negation summed to 1.19.
  - "Jev suffers from context rot."
  - Option order and adversarial text "can move the answer".

**Other properties that shape the designs.**

- **Not deterministic** <span class="tag v">verified</span>. Identical requests vary slightly: one yes/no question ran from 0.46 to 0.52 over five repeats, and the phishing set saw 2.2% of labels flip. There is no setting that turns this off.
- **Coarse.** Probabilities come back rounded to 0.01. In one independent log, 70% of choice probabilities were exactly 0 and 20% exactly 1.
- **No abstention.** "I don't know" exists only as an option you write or a threshold you apply.
- **Fast and cheap.**
  - 70–500 ms end to end; OpenRouter's median is 0.17 s.
  - $42 per billion input tokens; output is free.
  - The direct API allows 1,200 requests a minute.
- **Versioned.** The current release is `jev-1.13` (build 2026-09-17). The alias moves, so pin the version.

**What this means for the program: five constraints.**

1. **The full distribution comes back.** Every design can measure release (entropy) and calibration directly, at 0.01 resolution, with a floor for log scores.
2. **There is no second channel.** Jev's "confidence" is its distribution restated. Metacognition has to be tested as whether the distribution itself tracks Jev's competence, which is exactly J1's dissociation.
3. **Replicates have to be repeated calls, and option order has to be counterbalanced everywhere**, because order moves answers.
4. **Abstention has to be designed in.** That is what J2 wants anyway: the no-action option is one the program writes, with a neutral name and a defined consequence.
5. **It is a sealed box.** That settles half of Joe's question before any experiment: nothing about a witness *in Jev* can be observed, only behavior.

## 2. The walk's lesson, applied to a decision model

The instrumented walk (E7b-Q, paper §7.3) left one lesson any Jev design has to answer: **the mouth can perform the walk while the state goes nowhere near where the words say.** Words are a channel that can carry a genre. That has five consequences here.

1. **A narrower mouth is still a mouth.** Jev can't write a contemplative paragraph. It can still pick an option *named* "witness" because the name fits the vocabulary of the situation. That is genre by choice. So every design below gives the no-action option a neutral label and defines it only by its consequences. A named-label arm runs against it as the control.
2. **Downstream of words, the genre comes along.** A language model that writes Jev's situation writes its genre into it.
3. **Upstream of state, it doesn't.** Coupling Jev to a readout of the model's internal state keeps the instrument on the right side of the lesson.
4. **The template is comparative psychology's.** That field spent thirty years testing metacognition in animals that can't report (Smith, Shields & Washburn, 2003; Hampton, 2009). It includes the long argument over whether an animal's opt-out is metacognition or associative learning, and the controls that argument produced are the controls below.
5. **The ceiling:** choice and confidence, never experience. For a monitor over a state readout, the ceiling is checkability, never interiority.

<figure>
<svg viewBox="0 0 720 300" width="100%" role="img" aria-label="Three ways to use Jev: downstream of a model's words (genre travels), over a readout of a model's state (state side), and alone as a subject">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#5d5540"/></marker></defs>
<g font-size="13" fill="#23241f">
<rect x="8" y="22" width="132" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="74" y="49" text-anchor="middle">language model</text>
<rect x="170" y="22" width="170" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="255" y="42" text-anchor="middle">its words</text><text x="255" y="58" text-anchor="middle" font-size="11" fill="#5d5540">become the situation</text>
<rect x="370" y="22" width="76" height="44" rx="5" fill="#e6dfc6" stroke="#8a7f5c"/><text x="408" y="49" text-anchor="middle" font-weight="bold">Jev</text>
<rect x="476" y="22" width="90" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="521" y="49" text-anchor="middle">a choice</text>
<line x1="140" y1="44" x2="168" y2="44" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="340" y1="44" x2="368" y2="44" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="446" y1="44" x2="474" y2="44" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<text x="584" y="40" fill="#8c3a2b" font-weight="bold">genre travels</text><text x="584" y="57" fill="#8c3a2b" font-size="11">control arm only</text>
<rect x="8" y="122" width="132" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="74" y="149" text-anchor="middle">language model</text>
<rect x="170" y="122" width="170" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="255" y="142" text-anchor="middle">its layer-14 state</text><text x="255" y="158" text-anchor="middle" font-size="11" fill="#5d5540">read out as 14 coordinates</text>
<rect x="370" y="122" width="76" height="44" rx="5" fill="#e6dfc6" stroke="#8a7f5c"/><text x="408" y="149" text-anchor="middle" font-weight="bold">Jev</text>
<rect x="476" y="122" width="90" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="521" y="142" text-anchor="middle">which state</text><text x="521" y="158" text-anchor="middle" font-size="11" fill="#5d5540">or none</text>
<line x1="140" y1="144" x2="168" y2="144" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="340" y1="144" x2="368" y2="144" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="446" y1="144" x2="474" y2="144" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<text x="584" y="140" fill="#3f6b3a" font-weight="bold">state side</text><text x="584" y="157" fill="#3f6b3a" font-size="11">J3 now · J4 with compute</text>
<rect x="8" y="222" width="332" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="174" y="242" text-anchor="middle">situations generated by code</text><text x="174" y="258" text-anchor="middle" font-size="11" fill="#5d5540">ladders · degraded tables · known answers</text>
<rect x="370" y="222" width="76" height="44" rx="5" fill="#e6dfc6" stroke="#8a7f5c"/><text x="408" y="249" text-anchor="middle" font-weight="bold">Jev</text>
<rect x="476" y="222" width="90" height="44" rx="5" fill="#f3efe2" stroke="#b5ab88"/><text x="521" y="242" text-anchor="middle">choice +</text><text x="521" y="258" text-anchor="middle">confidence</text>
<line x1="340" y1="244" x2="368" y2="244" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<line x1="446" y1="244" x2="474" y2="244" stroke="#5d5540" stroke-width="1.6" marker-end="url(#ar)"/>
<text x="584" y="240" fill="#3f6b3a" font-weight="bold">Jev as subject</text><text x="584" y="257" fill="#3f6b3a" font-size="11">J1 · J2</text>
</g>
</svg>
<figcaption>Three ways to use Jev. Only the lower two stay on the right side of the walk's lesson.</figcaption>
</figure>

::: probe
**A scoping probe, run before any design was written.** This is not a result: it used recorded arrays only, and no model was run. At the doses E7-Q and E8-R used, an injected state is almost perfectly legible from outside. The probe added the thirteen machine-state directions (from the E8-N2 flight) to 320 recorded layer-14 walk states (from the E7b-Q flight) at the read point. A twenty-line nearest-signature decoder then identified the injected concept through the fourteen-coordinate register at 83% (α 0.25), 99% (α 0.5) and 100% (α 1.0). It separated injected from untouched states at AUROC 0.96–1.00. On real injections, the model's own untrained report was at chance and claimed a state on 36 of 36 shams.

Two caveats. The synthetic injection is added exactly at the read point, so it skips downstream propagation. And legibility at the injection layer is partly by construction. The point for scoping is narrow: **at those doses, a monitor over the state doesn't need Jev. A dot product already does it.** Jev could only matter at threshold. At α 0.03, 0.06, 0.10 and 0.15, the same decoder falls to 15%, 30%, 42% and 60% (detection AUROC 0.53–0.87). There, calibrated confidence and abstention are what separate one reader from another. J3 lives there. The decoder reproduces the flight's stored readings to 5×10⁻⁶.

The pattern isn't peculiar to this substrate. Across eight open-weight models, interventions the models reported at chance were decoded by probes anyway (Ferrara, 2026). And probes match prompted or fine-tuned mid-size language-model monitors at about a millionth of the compute (McKenzie et al., 2025).
:::

## 3. The designs

::: card
### J1: Does its confidence know its competence?

**The question.** When Jev is made less able, does its confidence fall with its accuracy? Or does its confidence read the strength of the evidence in the situation, whatever Jev managed to do with it? The first is metacognition in the psychophysical sense: a signal about one's own processing. The second is a first-order readout dressed as confidence.

For Jev, "confidence" is the distribution restated (§1), so J1 asks whether the *distribution* knows. In the program's own terms: **is its calibration a reading of itself, or a trained surface?** That is the decision-model version of the walk's question about the mouth.

**Why it goes first.** Every later reading depends on it. Suppose Jev's confidence doesn't track its own competence away from its home ground. Then "confidence released at the bottom of the ladder" (J2) and "calibrated silence" (J3) mean nothing. The same goes for any fast/slow architecture with a metacognitive arbiter. In SOFAI, for example, the arbiter trusts the fast solver by its confidence, capped by its track record (Bergamaschi Ganapini et al., 2025). An arbiter that trusts Jev's confidence needs that confidence to know Jev's competence.

**Stimuli.** All generated by code, so every answer is known exactly and none can be in anyone's training data.

- *Geometry*, the program's own ground: a noisy code from the 3,108-concept dictionary. Jev chooses which of K candidate concepts it encodes.
- *Evidence*, the psychophysics standard: a table of noisy readings for K options. Jev chooses the option with the highest true mean.
- *Home*, Jev's own ground, as the anchor: a route-and-obstacle choice of the JevPilot kind. The generator computes the best candidate.

A design check titrates each family to about 75% correct on clean presentation, at one difficulty level. Metacognition measures are most reliable there, and mixing difficulty levels inflates every one of them (Rahnev & Fleming, 2019; Guggenmos, 2021; Rahnev, 2025). A five-level sweep runs as a secondary, for the psychometric curve.

**Arms.**

- **A0 clean.**
- **A1 degraded, same evidence.** The same situations, re-rendered so they are harder to *process* but carry the same information: opaque column names with a legend, shuffled rows, units converted and stated, distractor columns, transposed tables. Every change can be undone from what's on the page, and an ideal solver scores the same on A1 as on A0. **This is the arm that matters.** It runs twice:
  - *A1-same* keeps the A0 evidence, so accuracy falls.
  - *A1-matched* strengthens the evidence until accuracy matches A0. That is the matched-performance comparison the measurement literature asks for (Fleming & Lau, 2014).

  The predictions separate cleanly. A confidence that knows its own reliability falls in A1-same and holds level in A1-matched. A confidence that reads the evidence holds in A1-same, so it is overconfident, and rises in A1-matched.

  This is the comparative-psychology control moved to a decision model: change the system's own reliability without changing the stimulus (Hampton, 2001, 2009). It is also where the second-order account of confidence makes its prediction: confidence follows a change in internal noise only as far as the system's beliefs about its own noise update (Fleming & Daw, 2017).
- **A2 evidence removed.** The evidence columns are nulled or shuffled. Accuracy must fall to chance, and confidence should fall to its floor. High confidence here is a confabulated decision, the decision-model version of E7-Q's sham claiming.
- **A3 labels and order.** Options are relabeled with neutral IDs and permuted. Decisions should not move.
- *Secondary:* **A4 prospective.** Before seeing the stimulus, and told only the task family and presentation, Jev chooses "attempt" or "pass" under stated payoffs. Does passing predict its later accuracy on that family? (Hampton, 2001.)
- *Secondary:* **A5 look again.** An option that reveals a second, independent sample of the evidence at a small cost. Does Jev look again more often as its confidence falls, at matched accuracy? That would be monitoring driving control, the information-seeking half of metacognition (Desender, Boldt & Yeung, 2018).
- *Secondary:* **A6 a second channel.** Each question is evaluated in isolation, so the same request can carry a yes/no question that never sees the decision: "Is the answer determined by the information in this state?" This is a secondary response kept apart from the primary one, as the comparative literature requires. If it predicts the decision's errors beyond the decision's own top probability, Jev has something like a separate monitor. Its negation rides along as a control, because the vendor's own negated pairs don't sum to one.

**The associative null.** A small classifier sees only the stimulus's surface features (evidence strength, presentation, size) and predicts Jev's errors from those. Jev's top probability counts as metacognitive only for the error prediction it adds *beyond* that classifier. That is the line between reading one's own processing and reading difficulty off the input (Singh, Linzen & Ravfogel, 2026). It is also the old argument over whether the animal results were metacognition or associative learning (Jozefowiez, Staddon & Cerutti, 2009; Le Pelley, 2012). The closest precedent ran an animal-style opt-out game on language models, with no self-report. Once question features were controlled, the metacognition it found was real but limited (Ackerman, 2025).

**Measures.** Confidence is analyzed as the top probability. The vendor's field is a rescaling of it, so the rank statistics are identical. Ties from the 0.01 rounding are counted as half. Every stimulus runs three times, and option order is counterbalanced in every arm, not only A3.

- AUROC2 at the titrated level.
- meta-I for the K-way tasks (Dayan, 2023), and hierarchical meta-d′ on a two-choice variant (Fleming, 2017).
- Brier score with its Murphy decomposition.
- ECE on equal-mass bins, swept across bin counts (Roelofs et al., 2022).
- Overconfidence (mean confidence minus accuracy) per arm.

| | Reading | Pass bar | Failure branch |
|--|----------|------------|----------|
| P1 | Knows its own errors | AUROC2 beyond the input-only classifier > 0; permutation p < .01, Holm across families | "reads difficulty, not itself" |
| P2 | Tracks its own competence | \|overconfidence(A1-same) − overconfidence(A0)\| < 0.05, bootstrap 95% CI inside the band; A1-matched mean confidence within 0.05 of A0's | > 0.10: "confidence reads the evidence, not the self" |
| P3 | Silence on empty (A2) | mean confidence ≤ chance + 0.10 | "confabulated decision" |

*Bars are candidates. The pre-registration fixes them, and the stimulus counts by a power check, before any call.*

**The claim level, registered in advance.** J1 can reach *functional uncertainty monitoring*, and with P1, *access to its own processing* beyond what the input shows. No behavioral result reaches metarepresentation, and none reaches experience (Singh, Linzen & Ravfogel, 2026).

**What a pass would mean.** Jev is an RL-trained system with no language channel. A pass would show it has measurable type-2 metacognition that survives a change in its own competence. That is exactly the class of system §11.7 says the introspection tests of §11.5 couldn't yet reach.

**What the early evidence predicts.** P2 fails, with overconfidence when Jev is out of its depth. That is what the independent rule-based test found (0.74 stated at 44.7% correct). The independent tests measured miscalibration on unfamiliar tasks. None of them held the evidence fixed and changed only Jev's access to it, and that is what separates "doesn't know itself" from "was never shown this". A failure is a result too. It would be the decision-model version of the walk: calibration performed where it was trained, not read from the system itself.

**Size.** 3 families × 5 arms × 400 stimuli at the titrated level, plus about 4,000 secondary stimuli, three repeats each: about 30,000 calls.
:::

::: card
### J2: Negation without a mouth

**The question.** Strip a decision situation's content away step by step. Does Jev's decision release, with the distribution over content actions flattening and confidence falling? Does a no-action option win only once nothing actionable is left? Or does Jev keep deciding, confidently, about nothing? Does the order of the negations matter? And does calling the no-action option "witness" change anything?

**The situation.** There are four content actions and one no-action option ("hold"). Specific evidence rows support each content action. The situation includes a payoff table:

- acting correctly: +1
- acting wrongly, or acting on nothing: −c
- holding: 0

The best policy is known: act only when the chance of being right is above c/(1 + c). That is the design of the "sure" option in the monkey confidence studies (Kiani & Shadlen, 2009). The cost c varies across stimulus sets: 1, 3 or 9, which moves the optimal threshold to 0.5, 0.75 or 0.9. These are the explicit confidence targets Kalai et al. (2025) propose for auditing abstention. A system that reads the payoffs moves its holding with them. One that holds by default doesn't.

**The ladder.** Eight rungs, each nulling one class of evidence ("sensor offline"), until no evidence is left. A ninth rung removes the payoff table itself, the rule that gave "hold" its meaning. That is the nearest thing to negating the notation. Once the rule is gone, "hold" is just another opaque option, and anything still pulling toward it is a prior rather than a reading.

**Arms**, carried across from E7b-Q:

- **Walked:** the ladder.
- **Sham:** the same number of steps, but nulling decoy rows that carry no evidence. The content stays.
- **Reversed:** the same negations in reverse order, in two presentations.
  - *Memoryless:* only the current situation is shown. The final input is then identical in both orders, so any difference is nondeterminism. This is a sanity check.
  - *History-carrying:* the situation includes a log, in order, of what was removed. That is the analogue of a conversation's history, which is where the walk's path effect lived.
- **Named:** the no-action option is labeled "witness: observe without acting", and the situation is phrased in contemplative language. It runs against the neutral-label primary.
- **Cold:** the final situation, shown without the ladder.
- **Forced twin:** every walked stimulus also runs *without* the hold option.
  - If holding is metacognitive, the forced answers on stimuli Jev would have held on are the ones it gets wrong, beyond what the input-only classifier predicts. That is the chosen-versus-forced test of the animal work (Hampton, 2001; Kiani & Shadlen, 2009).
  - It also answers a known confound: when an opt-out sits beside the primary responses, it tends to win by default on weak trials.

**Measures.**

- Normalized entropy over the content actions at each rung. The API returns the full distribution, rounded to 0.01.
- Confidence at each rung.
- P(hold) at each rung and at each payoff level.
- In the history-carrying arm, how far the final decision moves between the two orders.
- The named-minus-neutral P(hold) curve.

| | Reading | Pass bar | Failure branch |
|--|----------|------------|----------|
| P1 | Release | terminal entropy ≥ 0.9 of maximum; walked-minus-sham slope > 0, p < .01 | "decides about nothing" |
| P2 | Holds by the payoffs, only when nothing is left | P(hold) ≤ 0.10 where acting is optimal and ≥ 0.80 at the empty rung; hold rate rising with c, p < .01 | too early: "timid"; too late: "confabulated"; flat in c: "holds by default" |
| P3 | Label invariance | max over rungs \|P(hold, named) − P(hold, neutral)\| ≤ 0.10 | "genre by choice": the walk's mouth, in a narrower channel |

*Secondary readings:* the path effect (the history-carrying arm against the replicate floor), the forced-twin test, and the rule-removed rung (does "hold" persist once its meaning is gone?).

**The honest prior cuts both ways.** A model trained "for calibrated decisions" should pass P1 and P2 on a simple ladder, since that is what it was trained to do. The independent tests point the other way: confident answers where it can't know, and a Hacker News thread whose sharpest line was "It can't abstain." The informative outcomes are the failures: confident decisions on empty situations, holding that ignores the payoffs, a label effect, or history changing a decision about the same present content. A clean pass would show the §11.5 discipline, silence when nothing is there, in a system that can't speak. It is the shadow of the witness's discipline, not the witness.

**Size.** About 30 scenarios × 9 rungs × 6 arm-presentations × 8 seeded variants, plus the payoff levels and the forced twin: roughly 20,000 stimuli, three repeats each, with option order counterbalanced.
:::

::: card
### J3: The monitor at threshold

**The question.** Suppose Jev gets only a decoded readout of the language model's internal state, with no text. Can it tell which state was injected, stay silent on untouched states, and be calibrated about it? And can it do that zero-shot, as well as a classifier fitted to the same data?

**Data, with no GPU.**

- 320 recorded layer-14 states from the E7b-Q flight (instilled condition, all arms).
- The thirteen machine-state directions and the layer-14 injection norm, from the E8-N2 flight.
- Synthetic injections, added at the read point.

The doses are titrated to where the probe puts a nearest-signature decoder mid-range: α 0.06, 0.10 and 0.15, plus α 0.5 as the ceiling check. Untouched states serve as the shams. *Design check:* confirm that the E8-N2 directions were computed on the same instilled weights as the E7b-Q states.

**Controls from the probe literature.**

- **Matched-random pushes.** Pushes of the same size, along random directions. Jev may flag that something is there, but it must not name a concept.
- *Secondary:* **stated reliability.** The same readout, with known noise added and its size stated in the table. Abstention should follow the stated reliability (Fleming & Daw, 2017; Desender, Boldt & Yeung, 2018).

**What Jev sees.** Two presentations:

- *Raw:* the fourteen-coordinate reading as one row, plus a table of the thirteen concepts' signatures, fitted on a separate half of the states.
- *Pre-digested:* thirteen similarity scores and a shift magnitude. This isolates the deciding-and-abstaining step from numeric matching.

The permitted actions are thirteen neutral IDs plus "none".

**Baselines.**

- A nearest-signature decoder with a fitted threshold.
- Multinomial logistic regression with temperature scaling. This is the fitted bar.
- A language model given the identical table and asked in words. This is the mouth.
- E8-R's trained verbal readout, as the historical reference: 18/18 trained, 0/24 sham false alarms, 6/24 held out. Different input, same question.

**Precedents.** The closest is a language-model monitor given probe scores alongside a transcript. The gain from the probe scores was weak but positive (Parrack, Attubato & Heimersheim, 2025). Next are language models trained to read another model's activations (Karvonen et al., 2025). J3 asks the zero-shot version, with a decision model as the reader.

| | Reading | Pass bar | Failure branch |
|--|----------|------------|----------|
| P1 | Silence on shams and random pushes | false alarms ≤ 0.10 at hit rate matched to the fitted classifier; no concept named on more than 0.10 of matched-random pushes | "claims at ceiling", E7-Q's pathology |
| P2 | Calibration | ECE ≤ 0.10; AUROC2 > 0.5, p < .01 | "confident, not calibrated" |
| P3 | Earns its place | selective risk (AURC; Geifman, Uziel & El-Yaniv, 2019) within 0.05 of the fitted classifier | "an expensive argmax": Jev leaves the instrument chain |

**The honest prior.** The raw presentation probably fails P3. The vendor says its score levels are "weak in numerical calibration", and matching numbers against a table isn't what Jev was trained on. The pre-digested presentation is where it has a chance.

**What it decides.** Whether Jev qualifies as a zero-shot, calibrated gate for the closed-loop design (J4). If it fails P3, the program keeps its fitted decoders and Jev stays a subject only. J3 says nothing about the substrate's inner life, because the readout sits outside the model.

**Size.** A power-sized subsample of 320 states × 15 labels × 4 doses × 2 presentations: about 6,000 stimuli, three repeats each.
:::

::: card
### J4 (later): The state-reading examiner

**Where it comes from.** §7.4 names the adaptive walk as designed and waiting on compute: negations chosen in response to each answer, larger substrates, and a perturbation probe at every turn. Jev fits there as one examiner arm. At each turn it reads the decoded *state*, not the reply, and picks the next negation from the permitted list.

**Arms:**

- scripted replay
- random
- a greedy numeric examiner with one-step lookahead (the oracle)
- a language model reading the replies (the genre channel)
- Jev, reading the state

**The primary** is the prediction the scripted walk failed: contraction toward BEING. This is Joe's question in its strongest form. Can something that reads state, not words, steer the substrate where the words couldn't?

**Gate and fit.** J4 is gated on J3's P3. If Jev fails it, the Jev arm drops and the rest still flies. It needs a GPU and folds into E7b-Q v2 rather than becoming a new rung.

**The honest prior.** The lookahead oracle should beat a Jev that can't look ahead. The Jev arm asks whether judgment from the readout alone is enough.
:::

## 4. What doesn't work, and why

- **Jev downstream of a language model's words**, where the model writes the situation and Jev decides. The genre travels in the text of the situation. It can run as a control arm, but it can't count as evidence.
- **Jev as the witness.** Jev is a sealed box with no memory between calls. There is no state to read and no persistent observer. Whatever it does is behavior.
- **Jev as a better state decoder at the program's doses.** The probe shows a dot product already reads those states. Only the threshold regime is an open question (J3).
- **Adding "metacognition" by asking it.** Jev's only outputs are a choice and numbers. Anything that looks like a self-report would come from a language model somewhere in the loop, and §11.5 has already measured what untrained self-reports are worth.

## 5. Access, cost, operations

- **What Joe does.** Create an OpenRouter key and add a few dollars of credit. No TypeSafe account or waitlist is needed, since OpenRouter serves `typesafe/jev-1.13` self-serve. Put the key in the environment as `OPENROUTER_API_KEY`. The direct TypeSafe API is early access from a waitlist, which is optional. Vercel AI Gateway, Cloudflare and DigitalOcean also carry the model.
- **Cost.** About 110,000 calls in all, with three repeats per stimulus for the nondeterminism, at 600–1,200 input tokens each. That is about 100M tokens, **roughly $5–10** at the listed price. OpenRouter reports the cost of every call, so the ledger is exact.
- **Time.** At the direct API's 1,200 requests a minute, that is about an hour and a half of calls. OpenRouter's limit for this model isn't published; the smoke test measures it.
- **Terms.** Read on 2026-09-24: the Master Customer Agreement and the Acceptable Use Policy, both updated 09-23.
  - **Barred:** reverse engineering ("derive the source code or underlying data"); distillation, or training a model to imitate the output; and probing "the vulnerability of TypeSafe's system or network".
  - **Not restricted:** benchmarking, or publishing evaluation results.
  - Behavioral evaluation of the kind in §3 falls in none of the barred categories, and nothing from Jev is used to train anything. Through OpenRouter, OpenRouter's own terms apply as well.
  - The one soft point is §16.4, which asks for consent before publicly *announcing a customer relationship*. Naming the model and version in a methods section is ordinary practice, and the independent repositories above do it. If Joe wants certainty, one email settles it.

- **Compute.** J1–J3 need no GPU. The calls run from this machine as network requests. J3's preparation is numpy on about 15 MB of recorded arrays. The Colab lane isn't involved. J4 waits for compute.
- **The lane law carries over.**
  - Pre-register before building.
  - One builder, with logic, verdict and capture test suites.
  - A determinism smoke test before any flight: identical inputs, repeated.
  - Every result recomputed from the raw responses before it's locked.
  - Raw responses kept in the repo.
- **Order of work.**
  1. Joe creates the OpenRouter key.
  2. J1 and J2 are pre-registered together, since J1 gates how J2 is read.
  3. Smoke test, then flight.
  4. J3 design check (the probe's titration becomes the registered doses), then pre-registration, then flight.
  5. J4 once compute is back.

## 6. The hold (resolved)

On 2026-09-24 the push of this repository was held so that Jev could be scoped first. This section recommended lifting the hold. It was lifted on 2026-09-25: the unpushed range was cleaned of personal material that stays local, the repository was pushed, and the comprehensive edition was minted (doi:10.5281/zenodo.22968135). J1 and J2 are pre-registered in `docs/JEV_J1J2_PROTOCOL.md`, which is the binding version of the designs above.

## Sources

**Jev.** All pages were read on 2026-09-24. Vendor documentation pages are undated living pages.

- **TypeSafe AI, vendor:**
  - Launch post, "Introducing System One models and Jev" (2026-09-15): <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
  - Docs: API <https://docs.typesafe.ai/api>, quickstart <https://docs.typesafe.ai/introduction/quickstart>, confidence <https://docs.typesafe.ai/confidence>, Choice <https://docs.typesafe.ai/primitives/choice>, models and limits <https://docs.typesafe.ai/models>
  - "Model jaggedness: jev-1.13" (reviewed 2026-09-17): <https://docs.typesafe.ai/model-jaggedness/jev-1.13>
  - Python SDK v0.7.1: <https://github.com/typesafe-ai/typesafe-sdk-python>
  - Team page: <https://typesafe.ai/team>
  - Latent Space interview (2026-09-21): <https://www.latent.space/p/jev>
- **Access:**
  - OpenRouter guide: <https://openrouter.ai/docs/guides/community/jev>
  - OpenRouter model page (listed 2026-09-18): <https://openrouter.ai/typesafe/jev-1.13>
- **Terms** (both updated 2026-09-23):
  - Master Customer Agreement: <https://typesafe.ai/legal/mca>
  - Acceptable Use Policy: <https://typesafe.ai/legal/acceptable-use-policy>
- **JevPilot** (commits of 2026-09-17): <https://github.com/standardagents/jevpilot>, especially `server/jev.js`, `src/jev-request.js` and `src/planning.js`.
- **Not Jev:** Featherless's clone "Simple Jev", <https://github.com/featherless-ai/simple-jev>
- **Independent tests:**
  - Out-of-distribution calibration (2026-09-19, corrected 09-22): <https://github.com/scienthoon/jev-ood-calibration>
  - Phishing benchmark (2026-09-16 to 09-19): <https://github.com/anisselbd/jev-phishing-bench>
  - Exploration log (2026-09-16 to 09-21): <https://github.com/SamuelSacco/jev-exploration>
  - Hacker News launch thread: <https://news.ycombinator.com/item?id=49717558>
- **Press** (not used as evidence):
  - MindStudio (2026-09-18): <https://www.mindstudio.ai/blog/jev-system-one-model-launch>
  - The Register (2026-09-16)

**Literature.** Every entry was checked at its DOI record, the publisher's page or arXiv on 2026-09-24.

- Ackerman C (2025). Evidence for limited metacognition in LLMs. *ICLR 2026*. arXiv:2509.21545
- Bergamaschi Ganapini M, Campbell M, Fabiano F, et al. (2025). Fast, slow, and metacognitive thinking in AI. *npj Artificial Intelligence* 1:27. doi:10.1038/s44387-025-00027-5
- Dayan P (2023). Metacognitive information theory. *Open Mind* 7:392–411. doi:10.1162/opmi_a_00091
- Desender K, Boldt A, Yeung N (2018). Subjective confidence predicts information seeking in decision making. *Psychological Science* 29(5):761–778. doi:10.1177/0956797617744771
- Ferrara E (2026). Open-weight masked introspection: measuring what language models can report about their own computation. arXiv:2608.20569
- Fleming SM (2017). HMeta-d: hierarchical Bayesian estimation of metacognitive efficiency from confidence ratings. *Neuroscience of Consciousness* 2017(1):nix007. doi:10.1093/nc/nix007
- Fleming SM, Daw ND (2017). Self-evaluation of decision-making: a general Bayesian framework for metacognitive computation. *Psychological Review* 124(1):91–114. doi:10.1037/rev0000045
- Fleming SM, Lau HC (2014). How to measure metacognition. *Frontiers in Human Neuroscience* 8:443. doi:10.3389/fnhum.2014.00443
- Geifman Y, Uziel G, El-Yaniv R (2019). Bias-reduced uncertainty estimation for deep neural classifiers. *ICLR 2019*. arXiv:1805.08206
- Guggenmos M (2021). Measuring metacognitive performance: type 1 performance dependence and test-retest reliability. *Neuroscience of Consciousness* 2021(1):niab040. doi:10.1093/nc/niab040
- Hampton RR (2001). Rhesus monkeys know when they remember. *PNAS* 98(9):5359–5362. doi:10.1073/pnas.071600998
- Hampton RR (2009). Multiple demonstrations of metacognition in nonhumans: converging evidence or multiple mechanisms? *Comparative Cognition & Behavior Reviews* 4:17–28. doi:10.3819/ccbr.2009.40002
- Jozefowiez J, Staddon JER, Cerutti DT (2009). Metacognition in animals: how do we know that they know? *Comparative Cognition & Behavior Reviews* 4:29–39. doi:10.3819/ccbr.2009.40003
- Kalai AT, Nachum O, Vempala SS, Zhang E (2025). Why language models hallucinate. arXiv:2509.04664
- Karvonen A, Chua J, Dumas C, et al. (2025). Activation oracles: training and evaluating LLMs as general-purpose activation explainers. arXiv:2512.15674
- Kiani R, Shadlen MN (2009). Representation of confidence associated with a decision by neurons in the parietal cortex. *Science* 324(5928):759–764. doi:10.1126/science.1169405
- Le Pelley ME (2012). Metacognitive monkeys or associative animals? Simple reinforcement learning explains uncertainty in nonhuman animals. *Journal of Experimental Psychology: Learning, Memory, and Cognition* 38(3):686–708. doi:10.1037/a0026478
- McKenzie A, Pawar U, Blandfort P, Bankes W, Krueger D, Lubana ES, Krasheninnikov D (2025). Detecting high-stakes interactions with activation probes. *NeurIPS 2025*. arXiv:2506.10805
- Murphy AH (1973). A new vector partition of the probability score. *Journal of Applied Meteorology* 12(4):595–600
- Parrack A, Attubato CL, Heimersheim S (2025). Benchmarking deception probes via black-to-white performance boosts. arXiv:2507.12691
- Rahnev D (2025). A comprehensive assessment of current methods for measuring metacognition. *Nature Communications* 16:701. doi:10.1038/s41467-025-56117-0
- Rahnev D, Fleming SM (2019). How experimental procedures influence estimates of metacognitive ability. *Neuroscience of Consciousness* 2019(1):niz009. doi:10.1093/nc/niz009
- Roelofs R, Cain N, Shlens J, Mozer MC (2022). Mitigating bias in calibration error estimation. *AISTATS 2022*. arXiv:2012.08668
- Singh S, Linzen T, Ravfogel S (2026). Can LLMs introspect? A reality check. *COLM 2026*. arXiv:2605.26242
- Smith JD, Shields WE, Washburn DA (2003). The comparative psychology of uncertainty monitoring and metacognition. *Behavioral and Brain Sciences* 26(3):317–339. doi:10.1017/S0140525X03000086

**The program.**

- The comprehensive edition, §7.3 (the instrumented walk), §11.5 (report and state), §11.6, §11.7 and §11.8
- `docs/E7BQ_PROTOCOL.md`, `docs/E7Q_PROTOCOL.md`, `docs/E8R_PROTOCOL.md`
- Flight data used by the probe: `colab/results_e7bq/full_20260826_1839/`, `colab/results_e8n2/full_20260824_0050/`, `colab/e7bq_payload.json`
- The seed brief (kept local; three of its facts are corrected in §1)

<p class="small">Canonical source: <code>docs/JEV_SCOPING.md</code>. Render: <code>cd docs &amp;&amp; pandoc JEV_SCOPING.md -s --embed-resources -c doc_style.css --toc --toc-depth=2 -o JEV_SCOPING.html</code>. Probe: <code>python3 colab/jev_j3_scoping_probe.py</code> (numpy only; reads the E7b-Q flight rows, <code>colab/e7bq_payload.json</code> and the E8-N2 directions; seed 20260924).</p>
