"""
Tarot Major Arcana — Transparency Layer
=========================================
22 Major Arcana cards mapped to the semantic dictionary via
anchor concepts. Each card's position in 7D space is the
centroid of its anchor vectors.

Unlike the Toltec layer (which transposes hexagram → hexagram),
the Tarot layer maps via vector proximity — the correspondence
is itself a geometric finding, not a lookup table.

At diagnostic time:
  presenting vector → nearest card (what you carry)
  medicine vector   → nearest card (what is offered)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore


MAJOR_ARCANA = {
    0: {
        "name": "The Fool",
        "numeral": "0",
        "anchors": ["FREEDOM", "BEGINNING", "WONDER"],
        "image": "A figure at the cliff's edge, facing open sky. The pack on their back holds everything they don't know they'll need. The dog at their heel is instinct — the last tether before the leap.",
        "upright": "The willingness to begin without knowing the end. Not ignorance but pre-knowledge — the state before categories harden. Trust in the process that has no name yet.",
        "reversed": "Recklessness mistaken for freedom. Refusing to learn from what has already been shown. The leap without the listening.",
        "counsel": "The medicine here is not caution — it is the courage to be new. Something in you has not yet been tried. The Fool does not fall because the Fool has not yet decided where the ground is.",
    },
    1: {
        "name": "The Magician",
        "numeral": "I",
        "anchors": ["WILL", "POWER", "SKILL"],
        "image": "One hand points upward, the other toward the earth. On the table: cup, pentacle, sword, wand — the four domains made available. The lemniscate above the head is circulation, not possession.",
        "upright": "The capacity to act with precision and intention. All necessary resources are present — the question is not what you have but whether you will use it. Agency fully aligned with ability.",
        "reversed": "Manipulation. Skill deployed without purpose, or purpose deployed without skill. The gap between what you can do and what you should do has become invisible to you.",
        "counsel": "You have the tools. The medicine is not acquisition but alignment — pointing what you already carry toward what actually needs doing. The Magician does not create from nothing; the Magician arranges what is already here.",
    },
    2: {
        "name": "The High Priestess",
        "numeral": "II",
        "anchors": ["INTUITION", "MYSTERY", "DEPTH"],
        "image": "She sits between two pillars — severity and mercy, known and unknown. The scroll in her lap is partially concealed. The moon at her feet is the light that illuminates by reflection, not direct sight.",
        "upright": "What you need to know is already present but not yet conscious. The answer is not in more information — it is in the silence between the information you already have. Trust the pattern you can feel but not yet name.",
        "reversed": "Secrets kept from yourself. Intuition overridden by noise. The scroll is there but you keep looking at the pillars instead.",
        "counsel": "Stop seeking and start receiving. The medicine is not another search — it is the willingness to sit with what has already surfaced and let it speak in its own time. The Priestess does not chase knowledge; knowledge settles when she is still.",
    },
    3: {
        "name": "The Empress",
        "numeral": "III",
        "anchors": ["NURTURE", "ABUNDANCE", "GROWTH"],
        "image": "Seated in a field of ripe grain, crowned with stars. The shield at her side bears the symbol of Venus. The river behind her flows without effort — fertility that does not strain.",
        "upright": "The generative principle. What you tend will grow. The capacity to sustain life — not through force but through environment. Creation that comes from fullness, not from deficit.",
        "reversed": "Smothering. Growth for its own sake. Nurture that does not allow the nurtured to become independent. The field overgrown because no one harvested.",
        "counsel": "The medicine is tending, not controlling. Something in your situation needs nourishment — steady, patient, without demand for immediate return. The Empress grows a forest, not a bouquet.",
    },
    4: {
        "name": "The Emperor",
        "numeral": "IV",
        "anchors": ["AUTHORITY", "STRUCTURE", "ORDER"],
        "image": "Stone throne against a barren mountain. The ram's heads carved into the armrests are drive turned to stillness. The orb and scepter: sovereignty held, not wielded.",
        "upright": "The capacity to establish structure that serves. Order that is not imposed but earned. Authority that comes from having done the work, not from the position alone.",
        "reversed": "Rigidity. Control mistaken for leadership. Structure that has become the purpose instead of the container. The throne defended long after the kingdom has changed.",
        "counsel": "Something needs a frame. Not a cage — a frame. The medicine is the willingness to set a boundary, make a decision, or name what is and is not acceptable. The Emperor does not apologize for structure; the Emperor builds what can be stood upon.",
    },
    5: {
        "name": "The Hierophant",
        "numeral": "V",
        "anchors": ["HERITAGE", "FAITH", "RITUAL"],
        "image": "Between two acolytes, the raised hand offers blessing — two fingers up, two down. The keys at his feet cross: one gold, one silver. The institution and the mystery it was built to house.",
        "upright": "The living thread of what has been passed down. Not blind obedience but conscious participation in a lineage. The structure that holds what is too large for one person to carry alone.",
        "reversed": "Dogma. The form preserved after the spirit has departed. Following rules whose reasons have been forgotten. Conformity mistaken for devotion.",
        "counsel": "There is a teaching here — but it may not come in the form you expect. The medicine is receptivity to inherited wisdom without surrendering discernment. The Hierophant does not demand belief; the Hierophant transmits what was entrusted.",
    },
    6: {
        "name": "The Lovers",
        "numeral": "VI",
        "anchors": ["LOVE", "CHOICE", "UNION"],
        "image": "Two figures beneath an angel. The tree of knowledge behind one, the tree of life behind the other. The choice is not between them — it is whether to stand in the field between the trees at all.",
        "upright": "Authentic alignment between values and action. The choice that reveals who you are, not the choice that is safe. Union that requires both parties to be fully present.",
        "reversed": "Misalignment. Choosing from fear rather than from love. The relationship — to a person, a path, a commitment — that you stay in because leaving would mean seeing yourself clearly.",
        "counsel": "The medicine is not choosing correctly — it is choosing honestly. The Lovers card is not about romance; it is about the moment where what you want and what you value must be reconciled. The angel overhead is not judging; the angel is witnessing.",
    },
    7: {
        "name": "The Chariot",
        "numeral": "VII",
        "anchors": ["WILL", "DRIVE", "VICTORY"],
        "image": "The charioteer holds no reins. Two sphinxes — one black, one white — pull in divergent directions. The canopy of stars above is the celestial order that holds when earthly forces contradict.",
        "upright": "Movement through contradiction, not around it. The victory that comes from holding opposing forces in alignment through sheer directed will. Progress that is earned, not given.",
        "reversed": "Force without direction. Aggression mistaken for progress. The chariot moving fast but no longer toward anything. Winning the battle you forgot to question.",
        "counsel": "The contradictions in your situation are not obstacles — they are the horses. The medicine is not resolving the tension but steering through it. The Chariot does not eliminate opposition; the Chariot makes opposition into propulsion.",
    },
    8: {
        "name": "Strength",
        "numeral": "VIII",
        "anchors": ["STRENGTH", "COURAGE", "PATIENCE"],
        "image": "A woman closes the jaws of a lion — not with force but with presence. The lemniscate above her head is the same as the Magician's, but here it is patience rather than power. The lion does not resist because it has been met, not conquered.",
        "upright": "The quiet power that does not need to prove itself. Courage that is not the absence of fear but the decision to remain present with what is frightening. Endurance that transforms rather than merely survives.",
        "reversed": "Self-doubt disguised as humility. The lion let loose because you stopped believing you could hold it. Or: the lion crushed rather than befriended — force where patience was needed.",
        "counsel": "Whatever feels unmanageable is not asking to be controlled. It is asking to be met. The medicine is the steady hand — not the clenched fist. Strength in this geometry is relational, not muscular.",
    },
    9: {
        "name": "The Hermit",
        "numeral": "IX",
        "anchors": ["SOLITUDE", "WISDOM", "INTROSPECTION"],
        "image": "A robed figure on a mountain peak, lantern held forward. The staff in the other hand is not for walking — it is for measuring the depth of snow before stepping. The light illuminates one step ahead, not the whole path.",
        "upright": "The withdrawal that is not escape but focus. Solitude chosen for the purpose of seeing what crowds obscure. The wisdom that can only be found alone — not because others are wrong, but because your own signal needs silence to be heard.",
        "reversed": "Isolation mistaken for wisdom. Withdrawal that has become avoidance. The lantern turned inward until it blinds instead of illuminates.",
        "counsel": "Something in you needs to be alone with itself. Not forever — but now. The medicine is temporary separation from the noise so you can hear what you actually think. The Hermit returns to the village. But not yet.",
    },
    10: {
        "name": "Wheel of Fortune",
        "numeral": "X",
        "anchors": ["CHANGE", "CYCLE", "CHANCE"],
        "image": "The wheel turns. Figures rise on one side, fall on the other. At the hub: stillness. The sphinx at the top holds a sword — discernment at the peak of the cycle, where everything is briefly visible.",
        "upright": "What is happening is not personal — it is cyclical. The conditions are shifting and your position on the wheel is changing. This is not punishment or reward; it is motion. The question is not whether the wheel turns but what you see from where you are.",
        "reversed": "Resistance to what is already in motion. Clinging to the position on the wheel that felt comfortable. The refusal to acknowledge that what brought you here will also take you onward.",
        "counsel": "The medicine is not stopping the wheel. It is finding the stillness at the center while the circumference moves. What is ending was always going to end. What is beginning was already beginning before you noticed.",
    },
    11: {
        "name": "Justice",
        "numeral": "XI",
        "anchors": ["JUSTICE", "BALANCE", "TRUTH"],
        "image": "The scales in one hand, the sword in the other. The sword is upright — truth does not lean. The veil behind the throne is drawn aside. No mystery here, only accuracy.",
        "upright": "The situation requires honest accounting. Not punishment, not mercy — precision. What is true is true regardless of how it feels. The scales do not measure worth; they measure what is.",
        "reversed": "Dishonesty — with others or with yourself. The scales tampered with. Accountability avoided through narrative rather than faced through accuracy.",
        "counsel": "Something needs to be weighed honestly. The medicine is not judgment in the moral sense — it is clarity about cause and effect. Justice here is structural: what you put in determines what comes out. The sword cuts, but it cuts cleanly.",
    },
    12: {
        "name": "The Hanged Man",
        "numeral": "XII",
        "anchors": ["SACRIFICE", "SURRENDER", "INSIGHT"],
        "image": "Suspended by one foot from a living tree, arms forming a triangle behind the back. The face is calm — this is not torture but chosen suspension. The halo around the head is illumination that comes only from inversion.",
        "upright": "The reversal that reveals. What you are holding onto is the very thing obscuring your view. The surrender is not defeat — it is the decision to see from a different angle. What looked like loss from one orientation is gain from this one.",
        "reversed": "Stalling disguised as patience. Hanging on rather than hanging still. The refusal to make the sacrifice that would change the view.",
        "counsel": "Let go — not of everything, but of the one thing you're gripping hardest. The medicine is voluntary suspension of your usual perspective. The Hanged Man does not fall; the Hanged Man is held by what was given up.",
    },
    13: {
        "name": "Death",
        "numeral": "XIII",
        "anchors": ["DEATH", "TRANSFORMATION", "ENDING"],
        "image": "The skeleton rides a white horse. The sun sets between two towers — the same towers the Moon will rise between. The banner is the white rose of purification. The king is already fallen; the child offers flowers.",
        "upright": "Something is ending and cannot be preserved. This is not symbolic death — it is the actual completion of a form that can no longer sustain itself. What comes after is not yet visible, and that is correct. The gap between forms is real.",
        "reversed": "Resistance to necessary ending. Attempting to reanimate what has completed its cycle. Decay mistaken for preservation.",
        "counsel": "The medicine is not hope — it is composting. What has died needs to be allowed to decompose so that what comes next has soil. Death in this geometry is not destruction; it is the completion of transformation that was always underway.",
    },
    14: {
        "name": "Temperance",
        "numeral": "XIV",
        "anchors": ["TEMPERANCE", "MODERATION", "HARMONY"],
        "image": "An angel pours water between two cups — one foot on land, one in the stream. The triangle on the chest contains the square of manifestation. The path behind leads to the crown of light on the horizon — the goal is distant but the pouring is now.",
        "upright": "The art of right proportion. Not the middle path as compromise, but the precise mixture that produces something neither ingredient could produce alone. Integration that requires patience and exactness.",
        "reversed": "Excess or deficiency — too much of one element, not enough of another. The mixture wrong not because the ingredients are wrong but because the proportion is off.",
        "counsel": "Something in your situation needs blending, not choosing. The medicine is neither this nor that — it is the specific ratio of this-and-that that produces what is needed. Temperance is chemistry: precise, patient, transformative.",
    },
    15: {
        "name": "The Devil",
        "numeral": "XV",
        "anchors": ["BONDAGE", "SHADOW", "COMPULSION"],
        "image": "The horned figure on the black half-cube. Two figures chained at its base — but the chains are loose. They could lift them off. The inverted pentagram above is spirit submerged beneath matter, not absent from it.",
        "upright": "The bond that feels necessary but is not. Attachment to what diminishes you, maintained because the attachment itself has become the identity. The shadow is not the enemy — the refusal to look at the shadow is.",
        "reversed": "Breaking free — or the first moment of seeing the chains for what they are. The loosening that begins with recognition. Not yet liberation, but the prerequisite for it.",
        "counsel": "Name it. The medicine is not willpower — it is honesty about what you are bound to and why. The Devil does not imprison; the Devil reveals what you have been choosing without admitting you were choosing it.",
    },
    16: {
        "name": "The Tower",
        "numeral": "XVI",
        "anchors": ["DESTRUCTION", "REVELATION", "RUIN"],
        "image": "Lightning strikes the crown from the tower top. Two figures fall — not into nothing but into the open air they had been avoiding. The foundation was never as solid as the height suggested. The fire is clarifying.",
        "upright": "Sudden structural failure that was long in coming. The collapse of what was built on false ground. This is not catastrophe — it is the architecture correcting itself. What falls was already falling; the lightning only made it visible.",
        "reversed": "Avoiding the necessary collapse. Shoring up what should come down. Or: the aftermath — picking through rubble to find what survived, which is what was always real.",
        "counsel": "You cannot save the tower. The medicine is surviving the fall and noticing what remains standing when the smoke clears. That remainder is your actual foundation. The Tower destroys the false so the true becomes undeniable.",
    },
    17: {
        "name": "The Star",
        "numeral": "XVII",
        "anchors": ["HOPE", "INSPIRATION", "RENEWAL"],
        "image": "Naked at the water's edge, pouring from two vessels — one onto land, one into the pool. The great star above surrounded by seven smaller stars. After the Tower's destruction, this is the first clear sky.",
        "upright": "Renewal after devastation. Not optimism — something quieter and more durable. The return of meaning after a period where meaning was absent. Hope that is not denial but recognition that the cycle continues.",
        "reversed": "Despair mistaken for realism. The refusal to acknowledge that recovery is possible because admitting hope feels too dangerous. The pool is there but you won't drink.",
        "counsel": "After what has fallen, something still flows. The medicine is allowing yourself to be replenished without needing to understand how. The Star does not explain itself; the Star pours.",
    },
    18: {
        "name": "The Moon",
        "numeral": "XVIII",
        "anchors": ["FEAR", "DECEPTION", "DELUSION"],
        "image": "The moon between two towers — the same towers from Death, now at night. A dog and a wolf howl. A crayfish emerges from the pool onto a winding path. Nothing is as it appears; everything is as it feels.",
        "upright": "What you are perceiving is real but not accurate. The feelings are true data, but they are not facts. The path is obscured not because it is gone but because the light source distorts distances and shapes. Trust the discomfort; do not trust the story the discomfort tells.",
        "reversed": "Clarity returning after confusion. The willingness to distinguish between what you felt and what was actually happening. The moon setting so the sun can rise.",
        "counsel": "Do not make decisions in this light. The medicine is endurance through the uncertainty, not resolution of it. The Moon's territory must be crossed, not conquered. Walk the path; do not try to straighten it.",
    },
    19: {
        "name": "The Sun",
        "numeral": "XIX",
        "anchors": ["JOY", "VITALITY", "CLARITY"],
        "image": "A child on a white horse, arms open, beneath an enormous sun. Sunflowers turn toward the light. The wall behind is low — not a barrier but a gentle boundary. Everything visible, everything warm.",
        "upright": "Unobstructed clarity. The condition where what is true and what feels good are the same thing. Not naivety — earned simplicity. The joy that comes after complexity has been metabolized, not avoided.",
        "reversed": "Excessive brightness — clarity that blinds rather than reveals. Or: joy that is performed rather than felt. The sun as spotlight rather than nourishment.",
        "counsel": "Receive it. The medicine is not complicated — it is the willingness to accept that this moment of clarity is real and does not need to be defended or explained. The Sun does not argue for its own existence.",
    },
    20: {
        "name": "Judgement",
        "numeral": "XX",
        "anchors": ["AWAKENING", "RENEWAL", "PURPOSE"],
        "image": "The angel's trumpet sounds. Figures rise from coffins — not resurrected but awakened. They were not dead; they were dormant. The mountains behind are the same ones the Hermit climbed, now seen from the valley floor.",
        "upright": "The call that cannot be ignored because it comes from inside the structure, not outside it. Awakening to a purpose that was always present but dormant. The self recognizing itself through its own history.",
        "reversed": "Ignoring the call. Self-judgment that paralyzes rather than activates. The trumpet sounds but you pretend not to hear because answering it would require becoming what you have been avoiding.",
        "counsel": "Something in you is ready to rise. The medicine is not self-improvement — it is self-recognition. Judgement here is not condemnation; it is the accurate seeing of what you are and what you are for. The trumpet calls what is already alive.",
    },
    21: {
        "name": "The World",
        "numeral": "XXI",
        "anchors": ["COMPLETION", "INTEGRATION", "WHOLENESS"],
        "image": "The dancer in the wreath, holding two wands. The four fixed signs at the corners — the same four from the Wheel of Fortune, now stabilized. The wreath is the zero-point: end and beginning share a border.",
        "upright": "Completion that is not termination but integration. All elements present and in right relation. The cycle fulfilled — not because nothing is left to do, but because what needed to be done has been done. The whole is visible.",
        "reversed": "Near-completion. The last step avoided because finishing means beginning again. The wreath almost closed but the dancer hesitates.",
        "counsel": "You are closer to whole than you think. The medicine is not adding anything — it is recognizing that the circuit is already complete. The World does not need to be achieved; the World needs to be acknowledged.",
    },
}


class TarotLayer:
    """Maps Major Arcana to the Oracle's 7D semantic space via anchor concept centroids."""

    def __init__(self, sc: SemanticCore = None):
        self.sc = sc or SemanticCore()
        self._card_vectors = {}
        self._card_cores = {}
        self._build_vectors()

    def _build_vectors(self):
        for num, card in MAJOR_ARCANA.items():
            vecs_7d = []
            vecs_3d = []
            for anchor in card["anchors"]:
                c = self.sc.get(anchor)
                if c:
                    vecs_7d.append(np.array([c.x, c.y, c.z, c.e, c.f, c.g, c.h]))
                    vecs_3d.append(np.array([c.x, c.y, c.z]))
            if vecs_7d:
                self._card_vectors[num] = np.mean(vecs_7d, axis=0)
                self._card_cores[num] = np.mean(vecs_3d, axis=0)

    def nearest_card(self, core: np.ndarray, domain: np.ndarray, n: int = 1) -> list:
        """Find the n nearest Major Arcana cards to a 7D vector.
        Returns list of (card_number, angle, card_data)."""
        query = np.concatenate([core, domain])
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        results = []
        for num, vec in self._card_vectors.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            cos = np.clip(np.dot(query, vec) / (query_norm * vec_norm), -1, 1)
            angle = np.degrees(np.arccos(cos))
            results.append((num, round(angle, 1), MAJOR_ARCANA[num]))

        results.sort(key=lambda x: x[1])
        return results[:n]

    def nearest_card_core(self, core: np.ndarray, n: int = 1) -> list:
        """Find nearest cards using only core (x,y,z) — complement geometry space."""
        core_norm = np.linalg.norm(core)
        if core_norm < 1e-10:
            return []

        results = []
        for num, vec in self._card_cores.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            cos = np.clip(np.dot(core, vec) / (core_norm * vec_norm), -1, 1)
            angle = np.degrees(np.arccos(cos))
            results.append((num, round(angle, 1), MAJOR_ARCANA[num]))

        results.sort(key=lambda x: x[1])
        return results[:n]


def get_tarot_card(number: int) -> dict | None:
    return MAJOR_ARCANA.get(number)


def format_tarot_reading(presenting_card: tuple, medicine_card: tuple) -> str:
    """Format a Tarot transparency reading for display.
    Each card arg is (card_number, angle, card_data)."""
    lines = []
    lines.append("")
    lines.append("=" * 56)
    lines.append("  TAROT CORRESPONDENCE")
    lines.append("=" * 56)
    lines.append("")

    p_num, p_angle, p_data = presenting_card
    lines.append(f"  PRESENTING — {p_data['numeral']}. {p_data['name']}")
    lines.append(f"  (nearest at {p_angle}°)")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(p_data["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  UPRIGHT")
    for line in _wrap(p_data["upright"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(p_data["counsel"], 54):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("─" * 56)
    lines.append("")

    m_num, m_angle, m_data = medicine_card
    lines.append(f"  MEDICINE — {m_data['numeral']}. {m_data['name']}")
    lines.append(f"  (nearest at {m_angle}°)")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(m_data["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  UPRIGHT")
    for line in _wrap(m_data["upright"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(m_data["counsel"], 54):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("=" * 56)
    return "\n".join(lines)


def format_tarot_pointer(presenting_card: tuple, medicine_card: tuple) -> str:
    """Compact Tarot correspondence for default (non-expanded) display."""
    p_num, p_angle, p_data = presenting_card
    m_num, m_angle, m_data = medicine_card
    lines = []
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"  TAROT CORRESPONDENCE")
    lines.append(f"  Presenting: {p_data['numeral']}. {p_data['name']} ({p_angle}°)")
    lines.append(f"  Medicine:   {m_data['numeral']}. {m_data['name']} ({m_angle}°)")
    lines.append("")
    lines.append("  For the full Tarot reading, use --tarot flag")
    lines.append("─" * 60)
    return "\n".join(lines)


def format_tarot_standalone(card_number: int) -> str:
    """Format a single card for standalone lookup."""
    card = MAJOR_ARCANA.get(card_number)
    if not card:
        return f"No Major Arcana card #{card_number}"
    lines = []
    lines.append("")
    lines.append("=" * 56)
    lines.append(f"  {card['numeral']}. {card['name']}")
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"  Anchors: {', '.join(card['anchors'])}")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(card["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  UPRIGHT")
    for line in _wrap(card["upright"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  REVERSED")
    for line in _wrap(card["reversed"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(card["counsel"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("=" * 56)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list:
    words = text.split()
    lines = []
    current = []
    length = 0
    for word in words:
        if length + len(word) + (1 if current else 0) > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for num in sorted(MAJOR_ARCANA.keys()):
            card = MAJOR_ARCANA[num]
            print(f"  {card['numeral']:>4s}. {card['name']:22s}  [{', '.join(card['anchors'])}]")
    elif len(sys.argv) > 1 and sys.argv[1] == "--card":
        try:
            n = int(sys.argv[2])
        except (IndexError, ValueError):
            print("Usage: python3 -m oracle.tarot --card <number>")
            sys.exit(1)
        print(format_tarot_standalone(n))
    elif len(sys.argv) > 1 and sys.argv[1] == "--map":
        layer = TarotLayer()
        print("\n  MAJOR ARCANA — SEMANTIC MAP")
        print("  " + "=" * 54)
        for num in sorted(MAJOR_ARCANA.keys()):
            card = MAJOR_ARCANA[num]
            vec = layer._card_vectors.get(num)
            if vec is not None:
                from core.octonion import SemanticOctonion
                o = SemanticOctonion(w=1.0, x=vec[0], y=vec[1], z=vec[2],
                                     e=vec[3], f=vec[4], g=vec[5], h=vec[6])
                tri = o.to_trigram(x_axis=vec[0], y_axis=vec[1])
                print(f"  {card['numeral']:>4s}. {card['name']:22s} {tri.name:5s}  [{vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f}]")
        print()
    else:
        print("Usage:")
        print("  python3 -m oracle.tarot --list          List all 22 Major Arcana")
        print("  python3 -m oracle.tarot --card <number>  Show a single card")
        print("  python3 -m oracle.tarot --map            Show semantic map (trigrams + vectors)")
