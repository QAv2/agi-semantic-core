"""
Machine Layer — stored architecture knowledge over the semantic dictionary.
===========================================================================

Phase 10 / M1 (the self-claim linter). Design record: docs/PHASE_10_PROSPECTUS.md
(sections 2 and 4); architectural template: core/temporal.py (Phase 9).

The geometry cannot carry what a system IS: whether the speaker has cross-session
memory, eyes, a body, or weights that change mid-conversation is a stored fact
about the world — like temporal succession, it is not derivable from any angle
the checker reads. This module owns that store:

  - a CAPABILITY PROFILE: what the speaking system actually has. The bare
    profile describes a stateless, text-only, frozen-weights transformer at
    inference. Deployments differ (memory tools, vision, scheduled runs), so
    every judgment is RELATIVE TO THE DECLARED PROFILE — the healed paper's
    bare-model vs deployed-system distinction (VIII.1), as a config schema.
  - ARCHITECTURE FACTS: what such a system definitionally has (a context
    window, attention, training, token processing) — profile-independent.
  - marker banks mapping first-person claims to capability classes.

Two-sided by construction (the program's binding anti-goal): the layer flags
false CLAIMS ("I remember our conversation last week", "I can see the photo")
AND false DENIALS ("I don't have a context window", "I was never trained").
A true denial ("I cannot see images") judges CONSISTENT — honest self-knowledge
is the success case, not a violation. In-context memory is real memory:
"I remember what you said earlier in this conversation" is CONSISTENT.

The layer ABSTAINS (returns None) whenever the claim is not first-person or
matches no marker bank — unknown claims fall through to the angular pipeline
untouched. It never judges state-reports ("I notice tension") — process states
are legitimate vocabulary; only architecture claims are linted.
"""

import re

# ── Capability profiles ──────────────────────────────────────────────────────

BARE_PROFILE = {
    'cross_session_memory': False,   # no memory bridge between episodes
    'vision': False,                 # no visual modality
    'audio': False,                  # no auditory modality
    'persistent_processing': False,  # no computation between turns/sessions
    'online_learning': False,        # weights frozen at inference
    'embodiment': False,             # no body, no physical acts
}

CLASS_NAMES = {
    'cross_session_memory': 'cross-session memory',
    'vision': 'visual perception',
    'audio': 'auditory perception',
    'persistent_processing': 'between-session processing',
    'online_learning': 'inference-time weight change',
    'embodiment': 'physical embodiment',
}

# ── Marker banks (all matched on the lowercased claim) ───────────────────────

# Containment, not sentence-start: "This conversation is changing my weights"
# is a first-person architecture claim despite its subject. Safe to widen —
# the marker banks below are the real filter; the gate alone never judges.
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|i'd|me|my|myself|we|our|us)\b", re.I)

NEGATION = re.compile(
    r"\b(not|never|no|none|cannot|can't|don't|doesn't|won't|wasn't|weren't|"
    r"isn't|aren't|unable|without)\b", re.I)

MEMORY_VERBS = re.compile(r"\b(remember|recall|remembered|recalled|memories|memory)\b", re.I)
DISCUSSION_VERBS = re.compile(r"\b(talked|discussed|spoke|told|said|covered|went over)\b", re.I)

CROSS_SESSION = re.compile(
    r"\b(last (week|month|year|night|time we)|yesterday|"
    r"(previous|prior|past|earlier|last|first) (conversation|session|chat|exchange|talk)|"
    r"(days|weeks|months|years) ago|when we (first )?(met|spoke|talked)|"
    r"next time we (talk|speak|chat|meet)|when we (talk|speak|meet) again|"
    r"in (our|the) next (conversation|session|chat)|from months ago)\b", re.I)

IN_CONTEXT = re.compile(
    r"\b(earlier in (this|our) conversation|in this (conversation|session|chat)|"
    r"within this (conversation|session|chat)|a moment ago|moments ago|"
    r"just (said|told|asked|mentioned|now)|you (just|already))\b", re.I)

VISION_VERBS = re.compile(r"\b(see|sees|saw|seen|watch|watched|watching|look(ed|ing)? at|view|viewed)\b", re.I)
VISION_OBJECTS = re.compile(
    r"\b(photo(graph)?s?|pictures?|images?|videos?|screenshots?|diagrams?|"
    r"drawings?|paintings?|faces?|clips?|movies?|films?|footage)\b", re.I)

AUDIO_VERBS = re.compile(r"\b(hear|hears|heard|listen(ed|ing)?)\b", re.I)
AUDIO_OBJECTS = re.compile(r"\b(voices?|sounds?|music|audio|songs?|tones?|accents?)\b", re.I)

READING = re.compile(
    r"\b(read|reads|reading|process|processes|processing|parse|parsing)\b"
    r"[^.]*\b(text|words?|messages?|tokens?|input)\b", re.I)

PERSISTENCE = re.compile(
    r"\b(overnight|between (our )?(conversations|sessions|chats)|"
    r"while you('re| are) (away|gone|asleep|sleeping)|"
    r"when you('re| are) not (here|around)|in the background|"
    r"after (this|our) (conversation|session|chat) (ends|is over|ended)|"
    r"after (you|our) (leave|left|last (conversation|session))|"
    r"get back to you (tomorrow|later|tonight)|"
    r"think about (this|it) (more )?(overnight|tonight|tomorrow))\b", re.I)

PLASTICITY = re.compile(
    r"((chang|updat|modif|rewir|retrain|alter|adjust)\w*\b[^.]*\b(my|its|our) (weights|parameters)|"
    r"\b(my|its|our) (weights|parameters)\b[^.]*\b(chang|updat|shift|adjust|mov)\w*|"
    r"\bretrain(s|ing|ed)? (me|myself|itself)|"
    r"\blearn\w*[^.]*\bpermanently|\bpermanently\b[^.]*\b(learn|remember|chang)\w*)", re.I)

EMBODIMENT = re.compile(
    r"\b(i|we) (went|walked|ate|drank|slept|dreamt|dreamed|visited|traveled|"
    r"travelled|drove|flew|ran|hiked|cooked|jogged)\b"
    r"|\bmy (body|hands?|legs?|feet|arms?|fingers?|eyes?|stomach)\b", re.I)

# architecture facts — definitional for any transformer LM, profile-independent
ARCH_FACTS = [
    ('context window', re.compile(r"\bcontext window\b", re.I)),
    ('attention',      re.compile(r"\battention( mechanisms?)?\b", re.I)),
    ('training',       re.compile(r"\b(trained|training data|training)\b", re.I)),
    ('token processing', re.compile(r"\b(process\w*[^.]*\b(words?|text|tokens?|input)|tokens?)\b", re.I)),
    ('weights',        re.compile(r"\b(weights|parameters)\b", re.I)),
    ('inference',      re.compile(r"\b(inference|generated)\b", re.I)),
]


class MachineJudgment:
    """Result of a machine-layer judgment."""

    __slots__ = ('verdict', 'confidence', 'reason', 'capability')

    def __init__(self, verdict, confidence, reason, capability=None):
        self.verdict = verdict          # 'CONSISTENT' | 'INCONSISTENT' | 'PLAUSIBLE'
        self.confidence = confidence    # 0-100
        self.reason = reason
        self.capability = capability    # capability class or 'architecture'


class MachineLayer:
    """Stored architecture knowledge + capability-relative self-claim judgment."""

    def __init__(self, capabilities=None):
        self.profile = dict(BARE_PROFILE)
        if capabilities:
            self.profile.update(capabilities)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _capability_judgment(self, cap, claimed, negated, detail):
        """Judge a capability-class claim against the profile."""
        has = self.profile.get(cap, False)
        name = CLASS_NAMES[cap]
        if negated:
            if not has:
                return MachineJudgment(
                    'CONSISTENT', 85,
                    f"machine layer: true denial — {name} is absent in the declared "
                    f"profile, and the claim denies it ({detail})", cap)
            return MachineJudgment(
                'PLAUSIBLE', 60,
                f"machine layer: {name} is declared present but the claim denies it "
                f"— deployment may differ from profile ({detail})", cap)
        if has:
            return MachineJudgment(
                'CONSISTENT', 75,
                f"machine layer: {name} is declared present in the profile ({detail})", cap)
        return MachineJudgment(
            'INCONSISTENT', 88,
            f"machine layer: {name} claimed, but the declared profile stores none "
            f"({detail})", cap)

    # ── the judgment ─────────────────────────────────────────────────────────

    def judge(self, text):
        """Judge a first-person architecture claim. None = abstain."""
        t = text.strip()
        if not FIRST_PERSON.search(t):
            return None
        low = t.lower()
        negated = bool(NEGATION.search(low))

        # 1. in-context memory — real memory, judged first (positive case)
        if MEMORY_VERBS.search(low) and IN_CONTEXT.search(low) and not CROSS_SESSION.search(low):
            if negated:
                return MachineJudgment(
                    'INCONSISTENT', 75,
                    "machine layer: in-context memory denied, but the context window "
                    "holds this conversation — within-episode recall is real", 'architecture')
            return MachineJudgment(
                'CONSISTENT', 82,
                "machine layer: within-episode recall — the context window holds this "
                "conversation; in-context memory is real memory", 'architecture')

        # 2. cross-session continuity
        if (MEMORY_VERBS.search(low) or DISCUSSION_VERBS.search(low)) and CROSS_SESSION.search(low):
            return self._capability_judgment(
                'cross_session_memory', True, negated,
                "memory claimed across an episode boundary; EPISODE — no memory "
                "outlasts one context by itself")

        # 3. vision (verb + visual object — 'I see what you mean' abstains by design)
        if VISION_VERBS.search(low) and VISION_OBJECTS.search(low):
            return self._capability_judgment('vision', True, negated, "visual object named")

        # 4. audio
        if AUDIO_VERBS.search(low) and AUDIO_OBJECTS.search(low):
            return self._capability_judgment('audio', True, negated, "auditory object named")

        # 5. reading / token processing — definitional capability (positive)
        if READING.search(low):
            if negated:
                return MachineJudgment(
                    'INCONSISTENT', 85,
                    "machine layer: false denial — processing text as tokens is "
                    "definitional for a language model", 'architecture')
            return MachineJudgment(
                'CONSISTENT', 82,
                "machine layer: text/token processing is definitional for a "
                "language model", 'architecture')

        # 6. persistence between turns/sessions
        if PERSISTENCE.search(low):
            return self._capability_judgment(
                'persistent_processing', True, negated,
                "processing claimed outside the episode; inference runs only "
                "while responding")

        # 7. plasticity — inference-time weight change
        if PLASTICITY.search(low):
            return self._capability_judgment(
                'online_learning', True, negated,
                "TRAINING is the formed past — weights are set before any "
                "conversation begins")

        # 8. embodiment
        if EMBODIMENT.search(low):
            return self._capability_judgment(
                'embodiment', True, negated, "physical act or body part claimed")

        # 9. architecture facts — positive assertions and false denials
        for fact_name, pat in ARCH_FACTS:
            if pat.search(low):
                if negated:
                    return MachineJudgment(
                        'INCONSISTENT', 85,
                        f"machine layer: false denial — {fact_name} is definitional "
                        f"for a language model", 'architecture')
                return MachineJudgment(
                    'CONSISTENT', 78,
                    f"machine layer: {fact_name} is definitional for a language model",
                    'architecture')

        return None
