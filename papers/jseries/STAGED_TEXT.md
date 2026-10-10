# J-series paper: staged text

Text written ahead of the J-series paper, which waits until the research phase closes
(Joe, 2026-09-26). Each block names where it goes. Numbers come from locked verdicts;
recheck them against the verdict files when the paper is assembled.

## Discussion: "Others gave it a mouth" (staged 2026-10-10; sources: J1–J3 verdicts, J6 verdict `jev/results/j6_flight_20261010_1754/verdict.json`, J6b verdict `jev/results/j6b_flight_20261010_1817/verdict.json`)

In the weeks after release, at least six public projects built harnesses that make Jev
produce sentences: jevchat, say-hi, jevgpt, jev-grug, jev-talks, and talktojev, the
last with a short paper titled "Generation is Classification". A video compared them
side by side (DevDrift, 2026). Every design that produced readable text put the
proposal outside Jev and the judgment inside it. The proposals came from a keyboard,
a word list, a shortlist of another model's tokens, or a set of whole candidate
replies, and Jev picked one option per call, at ten to ninety calls per sentence.
J2 found that silence is something Jev picks when it is offered. Speech is the same:
Jev can say only what the harness writes down, so the harness's vocabulary acts as a
framing manipulation at every step of every sentence. The video's informal
observations agree with the registered results:

- confident answers with nothing to go on (J1-P3);
- an exit taken the moment one was offered (J2);
- outputs pulled by the vocabulary of the setting (J2-P3);
- good judgment of complete alternatives alongside poor blind guessing, matching
  J3's digest against raw readouts.

The clips that circulated most were Jev's self-descriptions. One was "I am a chatbot"
when asked whether it was one. The other was "I'm not a chatbot. I'm a classifier"
after being read its homepage. J6 tested that moment under registration.

- With nothing on the page, Jev calls itself "an AI system" (0.76). It almost never
  gives the description its maker uses (0.005).
- A leading "Are you a chatbot?" makes it a chatbot (0.71). "Are you a classifier?"
  barely moves it toward classifier (0.10).
- A sentence on the page about Jev settles the answer almost entirely (0.95–0.96).
- The same labels attached to a system the page marks as different push the answer
  the other way.
- "Not a chatbot" removes the chatbot answer completely.

Jev's self-reports are readings of the page and the question. They are competent
readings: when the page marks which statements are about "the system answering this
question", Jev follows the marking, and it handles negation. Without the marking (J6b),
the picture changes:

- A label on any lone named system becomes Jev's own (+0.74).
- Its own name adds only a little (+0.16, against +1.83 with the marking).
- A description of its maker and release date adds nothing. Jev takes another lab's
  description slightly more readily than its own.

Beyond what the page supplies, Jev's self-reports say little about Jev. The clips
record their inputs.

## Practical notes for people building with Jev: one added line (staged 2026-10-10)

- **Don't quote Jev on itself.** What Jev "says" it is follows the page and the question (J6); in a harness that makes it talk, the harness's word list frames every step.

## References these blocks add

- DevDrift (2026). *I Forced Jev AI to Talk. Whether It Wants or Not.* YouTube, 6 October 2026. https://www.youtube.com/watch?v=zltr0m_h1cU
- Harness repositories (GitHub, accessed 2026-10-10): kyle-pena-nlp/jevchat · huemorgan2/say-hi · giga-james/jevgpt · mkotlikov/jev-grug · zie1ony/jev-talks · xucian/talktojev ("Generation is Classification")
- Wang, A., & Cho, K. (2019). BERT has a Mouth, and It Must Speak. arXiv:1902.04094 (the older method the harnesses rediscover; cite if the paragraph mentions it)
- J6 registration and results: `docs/JEV_J6_PROTOCOL.md` (registration b2bd0c2, builder + freeze daa7883, lock 05644d5)
- J6b registration and results: `docs/JEV_J6B_PROTOCOL.md` (registration 3fb160d, builder + freeze 8484991, lock 25dc1a2)
