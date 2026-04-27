"""
Elder Futhark Runes — Transparency Layer
==========================================
24 runes (three aetts of 8) mapped to the semantic dictionary
via anchor concepts. Each rune's position in 7D space is the
centroid of its anchor vectors.

Like the Tarot layer, the rune correspondence is a geometric
finding — proximity in 7D, not a lookup table.

At diagnostic time:
  presenting vector → nearest rune (what the moment carries)
  medicine vector   → nearest rune (what is offered)

Aetts (groupings of 8):
  Freyr's   (1-8)   manifestation, beginnings, social
  Hagal's   (9-16)  disruption, force, transformation
  Tyr's     (17-24) divine, ancestral, transcendence
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore


RUNES = {
    1: {
        "name": "Fehu",
        "glyph": "ᚠ",
        "aett": "Freyr",
        "anchors": ["WEALTH", "ABUNDANCE", "EXCHANGE"],
        "image": "Cattle in a stone-walled pen at dawn, breath visible in cold air. The herdsman counts not by number but by which ones can move, which can be traded, which would bear the journey. Wealth that has legs.",
        "meaning": "The resource that is yours because you can move it — not the field, but what walks across it. Mobile worth. Capital that still answers to your hand. The opposite of treasure buried for safekeeping; this is value in circulation.",
        "counsel": "Look at what you have that moves. Not the asset but the relationship — the trust, the reputation, the skill that follows you across rooms. The medicine is to spend it, not hoard it. Fehu does not appreciate by stillness; Fehu appreciates by exchange.",
    },
    2: {
        "name": "Uruz",
        "glyph": "ᚢ",
        "aett": "Freyr",
        "anchors": ["STRENGTH", "VITALITY", "COURAGE"],
        "image": "The aurochs in the river-shallows at first light. Horns wider than a man can stretch. It does not look up because it has no predator that matters. The water moves around its legs the way wind moves around stone.",
        "meaning": "Raw force that does not need permission. Vitality that is not earned through discipline but received from being alive at all. The body's intelligence — older than the mind that thinks it commands the body. What rises in you before strategy.",
        "counsel": "Stop asking the situation if you are allowed to be strong. The medicine is to feel where the strength already is — in the breath, in the spine, in the no that does not need a reason. Uruz does not negotiate its existence; Uruz drinks from the river and the river makes way.",
    },
    3: {
        "name": "Thurisaz",
        "glyph": "ᚦ",
        "aett": "Freyr",
        "anchors": ["THORN", "BOUNDARY", "RESISTANCE"],
        "image": "A single thorn on a long stem, oil-black at the tip. The plant has no other defense. It does not need one. Whatever brushes against it learns the lesson once and adjusts.",
        "meaning": "The sharp point that is not aggression but accuracy. The boundary that does not need to be argued because it does not need to be repeated. Force that is dangerous only to what insists on touching what it should not touch.",
        "counsel": "There is something here that will not yield, and it is correct that it does not yield. The medicine is not softening the thorn — it is recognizing that the wound it makes is the same wound the lesson requires. Thurisaz does not punish; Thurisaz informs.",
    },
    4: {
        "name": "Ansuz",
        "glyph": "ᚨ",
        "aett": "Freyr",
        "anchors": ["MESSAGE", "INSPIRATION", "REVELATION"],
        "image": "A figure at the cave's mouth, head tilted toward what cannot be seen. The breath visible. The mouth open but not yet speaking. The signal is arriving from somewhere that does not have a body.",
        "meaning": "The transmission that comes through, not from. The thought you did not assemble. The phrase that knew what it meant before you did. Communication from the layer of self that does not call itself you.",
        "counsel": "Listen for the line that did not come from your planning. The medicine is to receive it on the page or the breath before the editing mind catches up. Ansuz does not announce itself; Ansuz is the moment you realize you were already being spoken through.",
    },
    5: {
        "name": "Raidho",
        "glyph": "ᚱ",
        "aett": "Freyr",
        "anchors": ["JOURNEY", "MOVEMENT", "RHYTHM"],
        "image": "A horse and rider on a packed-earth road that goes around the hill, not over it. Dust rising in the slant light. The destination is not visible but the way is — the road remembers itself even when the rider forgets.",
        "meaning": "Right movement. Not motion for its own sake but the rhythm that fits the terrain. The journey that respects its own speed. Travel as practice — the rider becoming the road's shape over the days.",
        "counsel": "Find the pace the situation actually wants. The medicine is not faster; the medicine is the rhythm that the road and the body both consent to. Raidho is not about arriving; Raidho is about the gait that makes arrival inevitable.",
    },
    6: {
        "name": "Kenaz",
        "glyph": "ᚲ",
        "aett": "Freyr",
        "anchors": ["KNOWLEDGE", "INSIGHT", "CRAFT"],
        "image": "A torch held shoulder-high in a workshop after dark. The flame steady because the hand is steady. It illuminates one bench at a time, not the whole room. The shadows behind the lit area are not absent — they are waiting.",
        "meaning": "Knowledge that you can use, not knowledge you can recite. The lit area of competence — what you can actually do under pressure. Skill held warm. The understanding that comes from making the thing twice, then twice more.",
        "counsel": "Bring the torch to one bench. The medicine is not surveying everything; it is illuminating the work in front of you well enough to do it. Kenaz does not set the room on fire; Kenaz lights what your hands need to see.",
    },
    7: {
        "name": "Gebo",
        "glyph": "ᚷ",
        "aett": "Freyr",
        "anchors": ["GIFT", "EXCHANGE", "RECIPROCITY"],
        "image": "Two open hands offered toward each other, neither yet holding the other's gift. The space between them is not empty. The agreement is in the air before the objects arrive.",
        "meaning": "The bond that is sealed by exchange. The relationship in which what is given and what is received cannot be separated from the giving. Reciprocity not as accounting but as continuity — the cord between two parties that the gift braids.",
        "counsel": "Notice what you are giving and what you are accepting. The medicine is not balancing the ledger — it is allowing the flow to happen at all. Gebo does not measure; Gebo binds. What you receive will not look like what you gave, and that is correct.",
    },
    8: {
        "name": "Wunjo",
        "glyph": "ᚹ",
        "aett": "Freyr",
        "anchors": ["JOY", "HARMONY", "BELONGING"],
        "image": "A pennant held up over a clearing where many have gathered. Not victory yet — recognition. Each face in the crowd is known to at least one other face. The pennant is the proof that everyone in the clearing has come to the same place on purpose.",
        "meaning": "Joy that comes from fit. The relief of being among the right people, in the right room, at the right time. Belonging not as approval but as accuracy — the place where what you are is what is needed.",
        "counsel": "Stay in the clearing a moment longer. The medicine is not pursuing joy; it is recognizing the joy that has already arrived and refusing to leave it prematurely. Wunjo does not chase happiness; Wunjo notices when the gathering is the answer.",
    },
    9: {
        "name": "Hagalaz",
        "glyph": "ᚺ",
        "aett": "Hagal",
        "anchors": ["STORM", "CHANGE", "HAIL"],
        "image": "Hail on a wheat field at the height of summer. The stalks bend, then break. By morning the sun returns and the mud holds the imprint of every stone that fell. Nothing is the same, but the field is still there.",
        "meaning": "The disruption that arrives without warning because it was never personal. The weather that does not care what you were building. What survives the hail is what was built deeply enough to be worth keeping; what does not survive was always going to fail later, more expensively.",
        "counsel": "Stop trying to negotiate with the storm. The medicine is not preventing the disruption — it is recognizing that the disruption is information about what was structurally sound and what was held together by hope. Hagalaz does not destroy; Hagalaz reveals.",
    },
    10: {
        "name": "Nauthiz",
        "glyph": "ᚾ",
        "aett": "Hagal",
        "anchors": ["NEED", "CONSTRAINT", "SCARCITY"],
        "image": "A figure rubbing two sticks together in the cold. The friction is reluctant. The fire will come — but only because nothing else is available, and the body has decided to make the heat itself.",
        "meaning": "The need that becomes the teacher. Not the lack you can solve by acquisition but the constraint that forces the development of capacities you would never have grown otherwise. The skill that only the dry season trains.",
        "counsel": "Stop trying to escape the constraint. The medicine is to ask what the constraint is asking you to develop. Nauthiz is not punishment; Nauthiz is the curriculum disguised as deprivation. The fire you make under pressure is the fire you will know how to make for the rest of your life.",
    },
    11: {
        "name": "Isa",
        "glyph": "ᛁ",
        "aett": "Hagal",
        "anchors": ["STILLNESS", "PAUSE", "WAITING"],
        "image": "A river frozen in mid-flow. The shapes of the eddies still visible under the surface — what was moving became sculpture. No fish. No sound. The current is preserved in its last position, exact and silent.",
        "meaning": "The pause that is not death but suspension. Motion held in trust against a thaw that has not yet arrived. Patience that is not chosen — patience that is the condition itself. What cannot be hurried because the medium has decided.",
        "counsel": "Stop pushing on the ice. The medicine is to use the stillness for the work that only stillness allows — observing the shape of what was moving, naming what was about to happen, taking inventory before the current returns. Isa does not reward struggle; Isa rewards attention.",
    },
    12: {
        "name": "Jera",
        "glyph": "ᛃ",
        "aett": "Hagal",
        "anchors": ["HARVEST", "CYCLE", "COMPLETION"],
        "image": "A field at full grain, late summer, the air thick with the sound of insects. The farmer walks the rows, not yet cutting. The work was done in spring; the standing is the proof. The sickle waits.",
        "meaning": "The cycle completing on its own terms. The reward that arrives because the work was done at the right season, not because anyone hurried. What you planted in the right month is now what you have.",
        "counsel": "Trust the timeline that is already in motion. The medicine is not adding more effort — it is allowing what is ripening to ripen. Jera does not negotiate the calendar; Jera is the proof that what you set in motion months ago was correct, and now the only task is to receive it.",
    },
    13: {
        "name": "Eihwaz",
        "glyph": "ᛇ",
        "aett": "Hagal",
        "anchors": ["ENDURANCE", "TREE", "DEATH"],
        "image": "A yew tree that has stood through three generations. Its roots reach as deep as its branches reach high — the tree is a column running through every layer of soil and air. It survives by being equally present above and below.",
        "meaning": "The endurance that comes from being rooted in both directions — into what is visible and into what is buried. The tree that does not fall in the storm because it has already integrated death into its structure. Resilience that is not avoidance of difficulty but absorption of it.",
        "counsel": "Find your axis. The medicine is to remember where in you the depth is — what part of you reaches into the dark soil that nourishes the part of you that reaches toward the light. Eihwaz does not avoid the underworld; Eihwaz makes the underworld its foundation.",
    },
    14: {
        "name": "Perthro",
        "glyph": "ᛈ",
        "aett": "Hagal",
        "anchors": ["MYSTERY", "CHANCE", "HIDDEN"],
        "image": "A leather cup full of carved bones. The hand reaches in but does not yet draw. What you receive is not chosen — but the act of reaching is. The pattern that emerges is older than the moment of drawing.",
        "meaning": "The structure beneath chance. What appears random because the order is too large to perceive. Fate that does not feel like fate from inside the moment but is unmistakable in retrospect. The element that cannot be bargained with because it operates from a deeper layer than your bargaining mind can reach.",
        "counsel": "Stop trying to engineer the outcome. The medicine is not control but consent — the willingness to draw the lot and accept what you drew. Perthro does not promise good fortune; Perthro promises that what you draw is part of a pattern that includes you.",
    },
    15: {
        "name": "Algiz",
        "glyph": "ᛉ",
        "aett": "Hagal",
        "anchors": ["PROTECTION", "GUARDIAN", "SHIELD"],
        "image": "A sedge of tall grass with edges sharp enough to cut. Or seen another way: an elk with antlers raised, head turned to the wind. Either form announces the same thing — there is something here that does not want to be approached carelessly.",
        "meaning": "The protection that is not aggression but signaling. The boundary that warns rather than wounds. What stands between you and what would diminish you — sometimes a person, sometimes a practice, sometimes the part of yourself that knows when to leave.",
        "counsel": "Listen to what is signaling no on your behalf. The medicine is recognizing that the impulse to withdraw, refuse, or invoke help is not weakness — it is intelligence. Algiz does not ask permission to protect; Algiz is the antlers that were already there.",
    },
    16: {
        "name": "Sowilo",
        "glyph": "ᛋ",
        "aett": "Hagal",
        "anchors": ["SUN", "VICTORY", "CLARITY"],
        "image": "The sun at the height of midsummer, casting no shadow. The sails of a small ship full. The journey that began in dim weather has arrived in light so direct it cannot be argued with. Everything visible.",
        "meaning": "The clarity that wins. Not the victory of force but the victory of becoming undeniable. The work that shines because it was done thoroughly enough that it cannot be dimmed. The vitality that comes from being aligned with what is.",
        "counsel": "Receive the light without translating it into modesty. The medicine is to allow what worked to be acknowledged as having worked. Sowilo does not apologize for arrival; Sowilo is the sun that has decided where to stand.",
    },
    17: {
        "name": "Tiwaz",
        "glyph": "ᛏ",
        "aett": "Tyr",
        "anchors": ["JUSTICE", "SACRIFICE", "HONOR"],
        "image": "A god with one hand. The other was given to bind the wolf that could not be bound by anything else. The remaining hand holds the spear. The sacrifice is in the past; the bearing of it is the present.",
        "meaning": "The principle that holds because someone paid for it. Justice that is not abstract but personal — the cost was real, and the structure that the cost purchased is still standing. Honor as the discipline of remaining accountable to what you swore.",
        "counsel": "Examine what you are bound to and whether you would pay the price again. The medicine is not lighter commitments — it is committing to what is worth losing a hand for, and accepting that the loss is the proof of the commitment. Tiwaz does not regret the wolf; Tiwaz remembers why the wolf had to be bound.",
    },
    18: {
        "name": "Berkano",
        "glyph": "ᛒ",
        "aett": "Tyr",
        "anchors": ["GROWTH", "BEGINNING", "NURTURE"],
        "image": "A birch sapling pushing through the leaf-litter at the edge of a clearing. The bark white as paper. The tree will be slender for years before it is strong, and that is the correct sequence. Nothing about its smallness is failure.",
        "meaning": "The growth that is gentle because it has time. What is just beginning is allowed to be small, and the smallness is the protection. The mother principle — not as control but as sheltering of what cannot yet defend itself.",
        "counsel": "Protect what is just starting from your impatience. The medicine is to recognize that early growth needs concealment more than it needs encouragement. Berkano does not rush the sapling; Berkano lets the sapling become a tree by remaining a sapling long enough.",
    },
    19: {
        "name": "Ehwaz",
        "glyph": "ᛖ",
        "aett": "Tyr",
        "anchors": ["HORSE", "TRUST", "ALLIANCE"],
        "image": "Horse and rider crossing a stream side by side, neither in front. The horse trusts the rider's weight; the rider trusts the horse's footing. Neither is in command. Both are in agreement.",
        "meaning": "The bond that two beings build by moving together. Trust that is built mile by mile, not declared. The partnership in which the strength of each is multiplied because neither is forcing the other into the role.",
        "counsel": "Look at the alliance you are in. The medicine is to remember that partnership is built by the small adjustments — the rider shifting weight, the horse shortening stride — not by the grand gesture. Ehwaz does not announce the bond; Ehwaz is the bond that the journey reveals.",
    },
    20: {
        "name": "Mannaz",
        "glyph": "ᛗ",
        "aett": "Tyr",
        "anchors": ["SELF", "BEING", "AWARENESS"],
        "image": "A figure standing, arms slightly raised, looking neither down at the ground nor up at the sky but level — outward at other figures who are doing the same. Recognition flowing in both directions across the air.",
        "meaning": "The self that knows itself by being seen by other selves. Humanity not as species but as practice — the daily work of recognizing other consciousnesses as real. Awareness that includes itself in the picture.",
        "counsel": "Notice who is seeing you and let them see you. The medicine is not solitude — it is the willingness to be human among humans, which means being recognized in the ways you are also still becoming. Mannaz does not perform; Mannaz exists in the field of mutual recognition that other people make possible.",
    },
    21: {
        "name": "Laguz",
        "glyph": "ᛚ",
        "aett": "Tyr",
        "anchors": ["WATER", "FLOW", "INTUITION"],
        "image": "A river at night. The surface reflects what little light there is — the moon, a lantern from the far shore. The depth below the reflection is not visible, but the current is felt by anything that wades in. The water remembers its course even when the night does not show it.",
        "meaning": "The intuition that operates below the level of speech. Knowledge that arrives as feeling first and as language only afterward. The depths in you that the conscious mind has agreed not to name but continues to consult.",
        "counsel": "Trust the water before you translate it. The medicine is to let the felt sense lead, even when the words have not arrived. Laguz does not announce its depth; Laguz is the depth that keeps your decisions correct even when you cannot explain them.",
    },
    22: {
        "name": "Ingwaz",
        "glyph": "ᛜ",
        "aett": "Tyr",
        "anchors": ["SEED", "POTENTIAL", "GESTATION"],
        "image": "A seed buried in dark soil over winter. Nothing visible above ground. The seed is not dormant — it is doing the slow assembly that will make spring possible. The work is total and entirely invisible.",
        "meaning": "The gestation phase that the world cannot yet see and cannot help. The interior development that requires concealment. What is being made of you in the dark, before the visible form arrives, is the form. Patience as the medium of becoming.",
        "counsel": "Honor what is being made in you that has no outward form yet. The medicine is to stop trying to display the work before it has finished assembling itself. Ingwaz does not announce; Ingwaz becomes. The display was never the point — the becoming was.",
    },
    23: {
        "name": "Othala",
        "glyph": "ᛟ",
        "aett": "Tyr",
        "anchors": ["HOME", "ANCESTRY", "HERITAGE"],
        "image": "An old farmhouse with a stone foundation. The walls have been rebuilt twice but the foundation is original. The view from the porch has not changed. Underneath the floorboards, a cellar dug by hands that did not live to see this generation.",
        "meaning": "The inheritance that arrives whether or not you claim it. Ancestry as the ground you stand on — not what you chose but what made the choosing possible. Home not as place but as the soil that grew the part of you that knows what soil is.",
        "counsel": "Acknowledge what was given to you before you arrived. The medicine is not pride in the lineage — it is recognition that the foundation you stand on was poured by people whose names you may not know, and that you are now responsible for what is built on top of it. Othala does not romanticize the past; Othala is the practice of continuing what was begun.",
    },
    24: {
        "name": "Dagaz",
        "glyph": "ᛞ",
        "aett": "Tyr",
        "anchors": ["DAY", "AWAKENING", "LIGHT"],
        "image": "The horizon at the moment the sun's edge crosses it. The world is neither in night nor in day — it is at the threshold. The shadow side is still cool. The lit side is already warm. The line between them is a doorway that takes a single instant to cross.",
        "meaning": "The breakthrough that is not gradual but absolute. The threshold that is crossed in one step, even though the approach was long. Awakening as the moment when what was implicit becomes explicit. The change of state that cannot be undone because the night cannot be re-entered.",
        "counsel": "Recognize that you have already crossed. The medicine is not preparing for the breakthrough; the breakthrough has already occurred and is asking you to stop pretending you are still in the dark. Dagaz does not arrive; Dagaz is the moment you notice that day was already happening.",
    },
}


AETT_NAMES = {
    "Freyr": "Freyr's aett — manifestation, beginnings, social",
    "Hagal": "Hagal's aett — disruption, force, transformation",
    "Tyr":   "Tyr's aett — divine, ancestral, transcendence",
}


class RuneLayer:
    """Maps Elder Futhark runes to the Oracle's 7D semantic space via anchor concept centroids."""

    def __init__(self, sc: SemanticCore = None):
        self.sc = sc or SemanticCore()
        self._rune_vectors = {}
        self._rune_cores = {}
        self._build_vectors()

    def _build_vectors(self):
        for num, rune in RUNES.items():
            vecs_7d = []
            vecs_3d = []
            for anchor in rune["anchors"]:
                c = self.sc.get(anchor)
                if c:
                    vecs_7d.append(np.array([c.x, c.y, c.z, c.e, c.f, c.g, c.h]))
                    vecs_3d.append(np.array([c.x, c.y, c.z]))
            if vecs_7d:
                self._rune_vectors[num] = np.mean(vecs_7d, axis=0)
                self._rune_cores[num] = np.mean(vecs_3d, axis=0)

    def nearest_rune(self, core: np.ndarray, domain: np.ndarray, n: int = 1) -> list:
        """Find the n nearest runes to a 7D vector.
        Returns list of (rune_number, angle, rune_data)."""
        query = np.concatenate([core, domain])
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        results = []
        for num, vec in self._rune_vectors.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            cos = np.clip(np.dot(query, vec) / (query_norm * vec_norm), -1, 1)
            angle = np.degrees(np.arccos(cos))
            results.append((num, round(angle, 1), RUNES[num]))

        results.sort(key=lambda x: x[1])
        return results[:n]

    def nearest_rune_core(self, core: np.ndarray, n: int = 1) -> list:
        """Find nearest runes using only core (x,y,z) — complement geometry space."""
        core_norm = np.linalg.norm(core)
        if core_norm < 1e-10:
            return []

        results = []
        for num, vec in self._rune_cores.items():
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            cos = np.clip(np.dot(core, vec) / (core_norm * vec_norm), -1, 1)
            angle = np.degrees(np.arccos(cos))
            results.append((num, round(angle, 1), RUNES[num]))

        results.sort(key=lambda x: x[1])
        return results[:n]


def get_rune(number: int) -> dict | None:
    return RUNES.get(number)


def format_rune_reading(presenting_rune: tuple, medicine_rune: tuple) -> str:
    """Format a Rune transparency reading for display.
    Each rune arg is (rune_number, angle, rune_data)."""
    lines = []
    lines.append("")
    lines.append("=" * 56)
    lines.append("  RUNE CORRESPONDENCE")
    lines.append("=" * 56)
    lines.append("")

    p_num, p_angle, p_data = presenting_rune
    lines.append(f"  PRESENTING — {p_data['glyph']}  {p_num:02d}. {p_data['name']} ({p_data['aett']}'s aett)")
    lines.append(f"  (nearest at {p_angle}°)")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(p_data["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  MEANING")
    for line in _wrap(p_data["meaning"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(p_data["counsel"], 54):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("─" * 56)
    lines.append("")

    m_num, m_angle, m_data = medicine_rune
    lines.append(f"  MEDICINE — {m_data['glyph']}  {m_num:02d}. {m_data['name']} ({m_data['aett']}'s aett)")
    lines.append(f"  (nearest at {m_angle}°)")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(m_data["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  MEANING")
    for line in _wrap(m_data["meaning"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(m_data["counsel"], 54):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("=" * 56)
    return "\n".join(lines)


def format_rune_pointer(presenting_rune: tuple, medicine_rune: tuple) -> str:
    """Compact Rune correspondence for default (non-expanded) display."""
    p_num, p_angle, p_data = presenting_rune
    m_num, m_angle, m_data = medicine_rune
    lines = []
    lines.append("")
    lines.append("─" * 60)
    lines.append(f"  RUNE CORRESPONDENCE")
    lines.append(f"  Presenting: {p_data['glyph']}  {p_data['name']} ({p_angle}°)")
    lines.append(f"  Medicine:   {m_data['glyph']}  {m_data['name']} ({m_angle}°)")
    lines.append("")
    lines.append("  For the full Rune reading, use --runes flag")
    lines.append("─" * 60)
    return "\n".join(lines)


def format_rune_standalone(rune_number: int) -> str:
    """Format a single rune for standalone lookup."""
    rune = RUNES.get(rune_number)
    if not rune:
        return f"No rune #{rune_number}"
    lines = []
    lines.append("")
    lines.append("=" * 56)
    lines.append(f"  {rune['glyph']}  {rune_number:02d}. {rune['name']}  ({rune['aett']}'s aett)")
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"  Anchors: {', '.join(rune['anchors'])}")
    lines.append("")
    lines.append("  IMAGE")
    for line in _wrap(rune["image"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  MEANING")
    for line in _wrap(rune["meaning"], 54):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  COUNSEL")
    for line in _wrap(rune["counsel"], 54):
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
        for num in sorted(RUNES.keys()):
            r = RUNES[num]
            print(f"  {r['glyph']}  {num:02d}. {r['name']:10s} {r['aett']:6s}  [{', '.join(r['anchors'])}]")
    elif len(sys.argv) > 1 and sys.argv[1] == "--rune":
        try:
            n = int(sys.argv[2])
        except (IndexError, ValueError):
            print("Usage: python3 -m oracle.runes --rune <number>")
            sys.exit(1)
        print(format_rune_standalone(n))
    elif len(sys.argv) > 1 and sys.argv[1] == "--aett":
        try:
            aett = sys.argv[2].capitalize()
        except IndexError:
            print("Usage: python3 -m oracle.runes --aett {Freyr|Hagal|Tyr}")
            sys.exit(1)
        if aett not in AETT_NAMES:
            print(f"Unknown aett: {aett}. Use Freyr, Hagal, or Tyr.")
            sys.exit(1)
        print(f"\n  {AETT_NAMES[aett]}")
        print("  " + "=" * 54)
        for num in sorted(RUNES.keys()):
            if RUNES[num]["aett"] == aett:
                r = RUNES[num]
                print(f"  {r['glyph']}  {num:02d}. {r['name']:10s} [{', '.join(r['anchors'])}]")
        print()
    elif len(sys.argv) > 1 and sys.argv[1] == "--map":
        layer = RuneLayer()
        print("\n  ELDER FUTHARK — SEMANTIC MAP")
        print("  " + "=" * 54)
        for num in sorted(RUNES.keys()):
            r = RUNES[num]
            vec = layer._rune_vectors.get(num)
            if vec is not None:
                from core.octonion import SemanticOctonion
                o = SemanticOctonion(w=1.0, x=vec[0], y=vec[1], z=vec[2],
                                     e=vec[3], f=vec[4], g=vec[5], h=vec[6])
                tri = o.to_trigram(x_axis=vec[0], y_axis=vec[1])
                print(f"  {r['glyph']}  {num:02d}. {r['name']:10s} {tri.name:5s}  [{vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f}]")
        print()
    else:
        print("Usage:")
        print("  python3 -m oracle.runes --list             List all 24 runes")
        print("  python3 -m oracle.runes --rune <number>    Show a single rune")
        print("  python3 -m oracle.runes --aett <name>      List one aett (Freyr|Hagal|Tyr)")
        print("  python3 -m oracle.runes --map              Show semantic map (trigrams + vectors)")
