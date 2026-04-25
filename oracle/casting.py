"""
Oracle Casting — Traditional I Ching Coin Method
==================================================
Standalone I Ching reading via the coin toss method.
No semantic input needed — pure divination.

3 coins x 6 throws → hexagram with possible changing lines.
Changing lines produce a second (future) hexagram.
Displays both King Wen and Toltec readings.
"""

import random
import sys
from oracle.hexagrams import HEXAGRAM_NAMES, KING_WEN_TABLE, TRIGRAM_ORDER
from oracle.toltec import format_toltec_reading, LINE_STAGES, get_toltec_for_king_wen


TRIGRAM_SYMBOLS = {
    'QIAN': ('═══', '═══', '═══'),  # Heaven
    'KUN':  ('= =', '= =', '= ='),  # Earth
    'ZHEN': ('= =', '= =', '═══'),  # Thunder
    'KAN':  ('= =', '═══', '= ='),  # Water
    'GEN':  ('═══', '= =', '= ='),  # Mountain
    'XUN':  ('═══', '═══', '= ='),  # Wind
    'LI':   ('═══', '= =', '═══'),  # Fire
    'DUI':  ('= =', '═══', '═══'),  # Lake
}

TRIGRAM_NAMES = {
    'QIAN': 'Heaven', 'KUN': 'Earth', 'ZHEN': 'Thunder', 'KAN': 'Water',
    'GEN': 'Mountain', 'XUN': 'Wind', 'LI': 'Fire', 'DUI': 'Lake',
}

LINE_DISPLAY = {
    6: ('---x---', 'changing yin'),
    7: ('-------', 'yang'),
    8: ('---  ---', 'yin'),
    9: ('---o---', 'changing yang'),
}


def toss_coins() -> int:
    """Toss 3 coins. Heads=3, Tails=2. Returns sum (6-9)."""
    return sum(random.choice([2, 3]) for _ in range(3))


def cast_hexagram() -> list[int]:
    """Cast a full hexagram: 6 throws, bottom to top.
    Returns list of 6 values (6, 7, 8, or 9)."""
    return [toss_coins() for _ in range(6)]


def throws_to_trigram(throws: list[int]) -> str:
    """Convert 3 line values to a trigram name.
    Treats 6/8 as yin (broken), 7/9 as yang (solid)."""
    bits = tuple(0 if t in (6, 8) else 1 for t in throws)
    trigram_map = {
        (1, 1, 1): 'QIAN',
        (0, 0, 0): 'KUN',
        (1, 0, 0): 'ZHEN',
        (0, 1, 0): 'KAN',
        (0, 0, 1): 'GEN',
        (0, 1, 1): 'XUN',
        (1, 0, 1): 'LI',
        (1, 1, 0): 'DUI',
    }
    return trigram_map[bits]


def get_changing_lines(throws: list[int]) -> list[int]:
    """Return 1-indexed positions of changing lines (6s and 9s)."""
    return [i + 1 for i, t in enumerate(throws) if t in (6, 9)]


def resolve_changes(throws: list[int]) -> list[int]:
    """Resolve changing lines: 6→7, 9→8."""
    return [7 if t == 9 else 8 if t == 6 else t for t in throws]


def throws_to_hexagram_number(throws: list[int]) -> int:
    """Convert 6 throws to a King Wen hexagram number."""
    stable = [7 if t == 9 else 8 if t == 6 else t for t in throws]
    lower = throws_to_trigram(stable[:3])
    upper = throws_to_trigram(stable[3:])
    return KING_WEN_TABLE[(lower, upper)]


def format_hexagram_visual(throws: list[int], label: str = "") -> list[str]:
    """Format a hexagram as visual lines, top to bottom display."""
    lines = []
    if label:
        lines.append(f"  {label}")
    for i in range(5, -1, -1):
        t = throws[i]
        display, kind = LINE_DISPLAY[t]
        pos = i + 1
        stage_name = LINE_STAGES.get(pos, ("", ""))[0]
        marker = f"  <-- {stage_name}" if t in (6, 9) else ""
        lines.append(f"    {pos}  {display}  {kind}{marker}")
    return lines


def format_casting(throws: list[int]) -> str:
    """Format a complete casting result for terminal display."""
    out = []
    out.append("")
    out.append("=" * 60)
    out.append("  I CHING COIN ORACLE")
    out.append("=" * 60)

    # Present hexagram
    lower = throws_to_trigram([7 if t == 9 else 8 if t == 6 else t for t in throws[:3]])
    upper = throws_to_trigram([7 if t == 9 else 8 if t == 6 else t for t in throws[3:]])
    kw_num = KING_WEN_TABLE[(lower, upper)]
    name_data = HEXAGRAM_NAMES.get(kw_num, ("", "", ""))

    out.append("")
    out.append(f"  PRESENT HEXAGRAM")
    out.append(f"  #{kw_num:02d} — {name_data[1]}")
    out.append(f"  {TRIGRAM_NAMES[lower]} (below) + {TRIGRAM_NAMES[upper]} (above)")
    out.append("")
    out.extend(format_hexagram_visual(throws, ""))
    out.append("")

    changing = get_changing_lines(throws)

    if changing:
        resolved = resolve_changes(throws)
        lower2 = throws_to_trigram(resolved[:3])
        upper2 = throws_to_trigram(resolved[3:])
        kw_num2 = KING_WEN_TABLE[(lower2, upper2)]
        name_data2 = HEXAGRAM_NAMES.get(kw_num2, ("", "", ""))

        out.append(f"  Changing lines: {', '.join(str(c) for c in changing)}")
        out.append("")
        out.append(f"  FUTURE HEXAGRAM")
        out.append(f"  #{kw_num2:02d} — {name_data2[1]}")
        out.append(f"  {TRIGRAM_NAMES[lower2]} (below) + {TRIGRAM_NAMES[upper2]} (above)")
        out.append("")
        out.extend(format_hexagram_visual(resolved, ""))
        out.append("")
    else:
        out.append("  No changing lines — the present situation holds.")
        out.append("")

    out.append("─" * 60)

    # King Wen reading
    out.append("")
    out.append("  KING WEN READING")
    out.append(f"  #{kw_num:02d} {name_data[0]} — {name_data[1]}")
    out.append(f"  {name_data[2]}")

    # Toltec reading for present hexagram
    toltec_reading = format_toltec_reading(kw_num, show_lines=changing)
    if toltec_reading:
        out.append(toltec_reading)

    if changing and kw_num2 != kw_num:
        out.append("")
        out.append("─" * 60)
        out.append(f"  FUTURE — King Wen #{kw_num2:02d} {name_data2[1]}")
        out.append(f"  {name_data2[2]}")
        future_toltec = format_toltec_reading(kw_num2)
        if future_toltec:
            out.append(future_toltec)

    return "\n".join(out)


def run_coin_oracle():
    """Interactive coin oracle session."""
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║         I CHING COIN ORACLE                  ║")
    print("  ║  Traditional casting with Toltec readings     ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print("  Focus on your question or simply ask for a")
    print("  reflection of yourself at this time.")
    print()
    input("  Press Enter to cast the coins...")
    print()

    throws = cast_hexagram()
    print("  Casting...")
    for i, t in enumerate(throws):
        display, kind = LINE_DISPLAY[t]
        print(f"    Throw {i+1}: {display}  ({t} — {kind})")

    result = format_casting(throws)
    print(result)


def show_hexagram_reading(kw_number: int):
    """Display the full Toltec + King Wen reading for a specific hexagram."""
    name_data = HEXAGRAM_NAMES.get(kw_number)
    if not name_data:
        print(f"  Unknown hexagram number: {kw_number}")
        return

    print()
    print("=" * 60)
    print(f"  HEXAGRAM #{kw_number:02d}")
    print(f"  {name_data[0]} — {name_data[1]}")
    print(f"  {name_data[2]}")
    print("=" * 60)

    reading = format_toltec_reading(kw_number, show_lines=[1, 2, 3, 4, 5, 6])
    if reading:
        print(reading)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--hexagram' and len(sys.argv) > 2:
            try:
                num = int(sys.argv[2])
                show_hexagram_reading(num)
            except ValueError:
                print(f"  Invalid hexagram number: {sys.argv[2]}")
        elif sys.argv[1] == '--help':
            print()
            print("  I Ching Coin Oracle")
            print("  Usage:")
            print("    python3 -m oracle.casting              Cast coins (interactive)")
            print("    python3 -m oracle.casting --hexagram N  Show reading for hexagram #N")
            print()
        else:
            print(f"  Unknown option: {sys.argv[1]}")
            print("  Use --help for usage info")
    else:
        run_coin_oracle()
