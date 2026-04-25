"""
Oracle Interpreter — Contextual Reading Composer
==================================================
Takes a DiagnosticResult and composes a prose reading
by weaving hexagram interpretation with diagnostic specifics:
presenting state, medicine concept, witness level, domain profile.
"""

from oracle.engine import DiagnosticResult
from oracle.interpretations import INTERPRETATIONS

TRIGRAM_QUALITIES = {
    "QIAN": ("creative force", "initiating", "the drive to act and create"),
    "KUN": ("receptive ground", "receiving", "the capacity to hold and sustain"),
    "ZHEN": ("arousing thunder", "activating", "the shock that sets things in motion"),
    "KAN": ("abysmal water", "flowing through danger", "the descent into depth"),
    "GEN": ("still mountain", "holding still", "the weight that does not need to move"),
    "XUN": ("gentle wind", "penetrating", "the persistence that enters every crack"),
    "LI": ("clinging fire", "illuminating", "the clarity that depends on what feeds it"),
    "DUI": ("joyous lake", "expressing", "the openness that connects through speech"),
}

WITNESS_READINGS = {
    "dissolution": "The boundary between your condition and its medicine is thin — almost too thin. There is a risk of dissolving into the cure rather than being transformed by it. The medicine works best when you remain a witness to the process, not a casualty of it.",
    "partial": "You hold some structure through this passage. Parts of you will change; parts will remain. The medicine does not require your total dissolution — it asks for yielding where you are rigid and firmness where you are formless.",
    "preserved": "The witness holds. You can receive the medicine without losing yourself to it. This is the healthiest geometry — complement without collapse. You and the medicine remain distinct, and the distinction is what makes the healing possible.",
    "tension": "There is significant tension between your presenting state and the medicine offered. This is not rejection — it is the friction of genuine difference. The medicine will not go down easy, but easy medicine rarely reaches the root.",
}

DOMAIN_READINGS = {
    "spatial": "The condition lives in space — in where you are, where you aren't, in boundaries and territories and the distance between things.",
    "temporal": "The condition lives in time — in what was, what is coming, in the gap between the pace you want and the pace that is real.",
    "relational": "The condition lives in relationship — in the space between you and others, in bonds that bind or bonds that are missing.",
    "personal": "The condition lives in the self — in identity, in the story you carry about who you are and who you should be.",
}


def interpret(d: DiagnosticResult) -> str:
    interp = INTERPRETATIONS.get(d.hexagram["number"])
    if not interp:
        return ""

    lines = []
    lines.append("─" * 56)
    lines.append("  READING")
    lines.append("─" * 56)
    lines.append("")

    p_quality, p_action, p_desc = TRIGRAM_QUALITIES.get(
        d.presenting_trigram, ("unknown", "unknown", "")
    )
    m_quality, m_action, m_desc = TRIGRAM_QUALITIES.get(
        d.medicine_trigram, ("unknown", "unknown", "")
    )

    lines.append(f"  Your presenting state carries the quality of {p_quality} —")
    lines.append(f"  {p_desc}. The medicine arrives as {m_quality} —")
    lines.append(f"  {m_desc}.")
    lines.append("")

    lines.append(f"  The specific medicine is {d.medicine_frame.source}.")
    if len(d.complement_concepts) >= 3:
        others = [name for name, _ in d.complement_concepts[1:3]]
        lines.append(f"  Also present in the field: {', '.join(others)}.")
    lines.append("")

    lines.append("  JUDGMENT")
    for line in _wrap(interp["judgment"], 54):
        lines.append(f"  {line}")
    lines.append("")

    lines.append("  IMAGE")
    for line in _wrap(interp["image"], 54):
        lines.append(f"  {line}")
    lines.append("")

    lines.append("  COUNSEL")
    for line in _wrap(interp["counsel"], 54):
        lines.append(f"  {line}")
    lines.append("")

    domain_text = DOMAIN_READINGS.get(d.domain_profile.get("dominant", ""), "")
    if domain_text:
        lines.append("  DOMAIN")
        for line in _wrap(domain_text, 54):
            lines.append(f"  {line}")
        lines.append("")

    if d.witness < 0.3:
        witness_key = "dissolution"
    elif d.witness < 0.7:
        witness_key = "partial"
    elif d.witness < 1.3:
        witness_key = "preserved"
    else:
        witness_key = "tension"

    lines.append("  WITNESS")
    for line in _wrap(WITNESS_READINGS[witness_key], 54):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("─" * 56)
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
