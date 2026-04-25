"""
Penniston Binary Code — Test Corpus
=====================================
152-hexagram sequence decoded from Jim Penniston's Rendlesham Forest
binary (1980), as decoded by William Horden. Two parallel readings:
  - King Wen sequence (political-historical)
  - Toltec sequence (spiritual-cultural)

Same binary stream → two coherent narratives. Structural parallel
to dual-octonion encoding (8D essence + 8D function).

Used here as a validation corpus for the Oracle diagnostic engine.
"""

import sys
import os
import json
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.hexagrams import HEXAGRAM_NAMES, lookup_hexagram
from oracle.toltec import get_toltec_for_king_wen, KING_WEN_TO_TOLTEC, TOLTEC_HEXAGRAMS
from core.octonion import Trigram

KING_WEN_SEQUENCE = [
    49, 63, 28, 13, 55, 9, 8, 30, 13, 63,
    49, 58, 55, 53, 45, 19, 49, 53, 28, 63,
    55, 37, 5, 41, 30, 63, 49, 17, 60, 18,
    62, 17, 33, 25, 18, 23, 44, 21, 18, 49,
    40, 42, 18, 28, 57, 21, 18, 17, 57, 46,
    # Part II
    26, 58, 18, 25, 18, 17, 50, 17, 26, 31,
    56, 29, 6, 21, 27, 46, 38, 50, 30, 37,
    45, 63, 55, 39, 63, 55, 37, 17, 19, 37,
    48, 43, 61, 13, 37, 45, 5, 63, 9, 5,
    30, 22, 37, 5, 9, 63, 57, 5, 42, 11,
    # continued
    10, 57, 60, 26, 55, 64, 64, 47, 38, 50,
    13, 28, 47, 35, 27, 50, 58, 38,
    # Part III
    42, 44, 64, 50, 38, 28, 47, 35, 42, 64,
    58, 38, 42, 64, 47, 35, 6, 64, 64, 47,
    46, 10, 51, 5, 37, 5, 58, 14, 49, 37,
    17, 19, 37, 57,
]

TOLTEC_SEQUENCE = [
    15, 53, 7, 62, 33, 3, 61, 52, 62, 53,
    15, 2, 33, 63, 19, 45, 15, 63, 7, 53,
    33, 39, 42, 35, 52, 53, 15, 11, 20, 64,
    13, 11, 55, 34, 64, 38, 40, 26, 64, 15,
    40, 5, 64, 7, 29, 26, 64, 11, 29, 47,
    21, 2, 64, 34, 64, 11, 4, 11, 21, 36,
    22, 57, 32, 26, 14, 47, 23, 4, 52, 39,
    19, 53, 33, 37, 53, 33, 39, 11, 45, 39,
    59, 24, 8, 62, 39, 19, 42, 53, 3, 42,
    52, 56, 39, 42, 3, 53, 29, 42, 35, 17,
    16, 29, 20, 21, 33, 18, 18, 46, 23, 4,
    62, 7, 46, 41, 14, 4, 2, 23, 5, 40,
    18, 4, 23, 7, 46, 41, 5, 18, 2, 23,
    5, 18, 46, 41, 32, 18, 18, 23, 47, 16,
    1, 42, 39, 42, 2, 27, 15, 39, 11, 45,
    39, 29,
]

NUCLEAR_KW = [38, 54, 64, 29]
NUCLEAR_TOLTEC = [23, 12, 18, 57]

TRIGRAM_INFO = {
    "QIAN": ("Heaven", "Creative"),
    "KUN": ("Earth", "Receptive"),
    "ZHEN": ("Thunder", "Arousing"),
    "KAN": ("Water", "Abysmal"),
    "GEN": ("Mountain", "Stillness"),
    "XUN": ("Wind", "Penetrating"),
    "LI": ("Fire", "Clinging"),
    "DUI": ("Lake", "Joyous"),
}


def analyze_frequency(sequence, label):
    """Frequency analysis of hexagram appearances."""
    counts = Counter(sequence)
    total = len(sequence)
    unique = len(counts)
    missing = [n for n in range(1, 65) if n not in counts]

    print(f"\n  {label} — FREQUENCY ANALYSIS")
    print(f"  {'=' * 52}")
    print(f"  Total hexagrams: {total}")
    print(f"  Unique hexagrams: {unique}/64")
    print(f"  Missing hexagrams: {len(missing)}")
    print()

    print(f"  TOP REPEATERS")
    for num, count in counts.most_common(15):
        bar = "█" * count
        if label.startswith("King"):
            h = HEXAGRAM_NAMES.get(num, {})
            name = h.get("name", f"#{num}") if isinstance(h, dict) else str(h)
        else:
            t = TOLTEC_HEXAGRAMS.get(num)
            name = t["name"] if t else f"#{num}"
        print(f"    #{num:02d} {name:30s} {count:2d}× {bar}")

    print()
    print(f"  ABSENT ({len(missing)})")
    absent_names = []
    for num in missing:
        if label.startswith("King"):
            h = HEXAGRAM_NAMES.get(num, {})
            name = h.get("name", f"#{num}") if isinstance(h, dict) else str(h)
        else:
            t = TOLTEC_HEXAGRAMS.get(num)
            name = t["name"] if t else f"#{num}"
        absent_names.append(f"#{num:02d} {name}")
    for i in range(0, len(absent_names), 3):
        row = absent_names[i:i+3]
        print(f"    {'  |  '.join(row)}")


def analyze_trigrams(sequence, label):
    """Trigram distribution analysis using hexagram lower/upper trigrams."""
    lower_counts = Counter()
    upper_counts = Counter()

    for num in sequence:
        h = lookup_hexagram_by_number(num)
        if h:
            lower_counts[h["lower"]] += 1
            upper_counts[h["upper"]] += 1

    print(f"\n  {label} — TRIGRAM DISTRIBUTION")
    print(f"  {'=' * 52}")
    print(f"  {'Trigram':<8s} {'Lower':>6s} {'Upper':>6s} {'Total':>6s}  Visual")
    print(f"  {'-' * 52}")

    all_trigrams = sorted(set(list(lower_counts.keys()) + list(upper_counts.keys())))
    for tri in all_trigrams:
        lo = lower_counts.get(tri, 0)
        up = upper_counts.get(tri, 0)
        total = lo + up
        bar = "█" * (total // 2)
        print(f"  {tri:<8s} {lo:>6d} {up:>6d} {total:>6d}  {bar}")


def analyze_transitions(sequence, label):
    """Look for trigram phase transitions in the sequence."""
    transitions = Counter()
    prev_lower = None

    for num in sequence:
        h = lookup_hexagram_by_number(num)
        if h:
            if prev_lower is not None:
                transitions[(prev_lower, h["lower"])] += 1
            prev_lower = h["lower"]

    print(f"\n  {label} — TRIGRAM TRANSITIONS (lower→lower)")
    print(f"  {'=' * 52}")
    print(f"  Top 15 transitions:")
    for (fr, to), count in transitions.most_common(15):
        print(f"    {fr:5s} → {to:5s}  {count:3d}×")


def analyze_toltec_crossref():
    """Verify King Wen → Toltec correspondence for the full sequence."""
    print(f"\n  CROSS-REFERENCE: King Wen ↔ Toltec")
    print(f"  {'=' * 52}")

    matches = 0
    mismatches = 0
    for i, (kw, tol) in enumerate(zip(KING_WEN_SEQUENCE, TOLTEC_SEQUENCE)):
        expected_toltec = KING_WEN_TO_TOLTEC.get(kw)
        if expected_toltec == tol:
            matches += 1
        else:
            mismatches += 1
            kw_name = HEXAGRAM_NAMES.get(kw, {})
            kw_name = kw_name.get("name", f"#{kw}") if isinstance(kw_name, dict) else str(kw_name)
            t_actual = TOLTEC_HEXAGRAMS.get(tol, {})
            t_actual_name = t_actual.get("name", f"#{tol}") if t_actual else f"#{tol}"
            if expected_toltec:
                t_expect = TOLTEC_HEXAGRAMS.get(expected_toltec, {})
                t_expect_name = t_expect.get("name", f"#{expected_toltec}") if t_expect else f"#{expected_toltec}"
            else:
                t_expect_name = "NO MAPPING"
            print(f"  [{i+1:3d}] KW#{kw:02d} {kw_name}")
            print(f"        Expected Toltec #{expected_toltec}: {t_expect_name}")
            print(f"        Actual   Toltec #{tol}: {t_actual_name}")

    print()
    print(f"  Matches: {matches}/{len(KING_WEN_SEQUENCE)}")
    print(f"  Mismatches: {mismatches}")
    if mismatches == 0:
        print(f"  ✓ Transposition table VERIFIED across all 152 hexagrams")


def analyze_repeats(sequence, label):
    """Find consecutive repeats — structurally significant."""
    print(f"\n  {label} — CONSECUTIVE REPEATS")
    print(f"  {'=' * 52}")
    found = False
    for i in range(len(sequence) - 1):
        if sequence[i] == sequence[i+1]:
            found = True
            num = sequence[i]
            if label.startswith("King"):
                h = HEXAGRAM_NAMES.get(num, {})
                name = h.get("name", f"#{num}") if isinstance(h, dict) else str(h)
            else:
                t = TOLTEC_HEXAGRAMS.get(num)
                name = t["name"] if t else f"#{num}"
            print(f"  [{i+1:3d}→{i+2:3d}] #{num:02d} {name}")
    if not found:
        print(f"  (none)")


def analyze_oracle_spread():
    """Feed each unique hexagram through the Oracle to see medicine distribution."""
    from oracle.engine import OracleEngine
    from oracle.tarot import TarotLayer

    engine = OracleEngine()
    tarot = TarotLayer(engine.sc)

    kw_counts = Counter(KING_WEN_SEQUENCE)
    unique_hexagrams = sorted(kw_counts.keys())

    print(f"\n  ORACLE SPREAD — Hexagram Trigram Pairs")
    print(f"  {'=' * 60}")
    print(f"  {'#':>3s} {'Name':20s} {'Lower':>6s} {'Upper':>6s} {'Freq':>4s}")
    print(f"  {'-' * 60}")

    lower_total = Counter()
    upper_total = Counter()
    for num in KING_WEN_SEQUENCE:
        h = lookup_hexagram_by_number(num)
        if h:
            lower_total[h["lower"]] += 1
            upper_total[h["upper"]] += 1

    for num in unique_hexagrams:
        h = lookup_hexagram_by_number(num)
        if h:
            count = kw_counts[num]
            print(f"  {num:3d} {h['name']:20s} {h['lower']:>6s} {h['upper']:>6s} {count:4d}×")

    print()
    print(f"  WEIGHTED TRIGRAM BALANCE (presenting vs medicine)")
    print(f"  {'Trigram':<8s} {'Presenting':>10s} {'Medicine':>10s}")
    for tri in sorted(set(list(lower_total.keys()) + list(upper_total.keys()))):
        lo = lower_total.get(tri, 0)
        up = upper_total.get(tri, 0)
        print(f"  {tri:<8s} {lo:>10d} {up:>10d}")


def analyze_tarot_spread():
    """Feed the top hexagram themes through the Oracle as queries and find Tarot correspondences."""
    from oracle.engine import OracleEngine
    from oracle.tarot import TarotLayer

    engine = OracleEngine()
    tarot = TarotLayer(engine.sc)

    print(f"\n  TAROT CORRESPONDENCES — Penniston Sequence")
    print(f"  {'=' * 60}")

    kw_counts = Counter(KING_WEN_SEQUENCE)

    presenting_cards = Counter()
    medicine_cards = Counter()
    hex_tarot_map = {}

    for num in sorted(set(KING_WEN_SEQUENCE)):
        h = lookup_hexagram_by_number(num)
        if not h:
            continue

        query = h.get("title", h.get("name", ""))
        result = engine.diagnose(query)
        if result.tokens_found:
            p = tarot.nearest_card(result.input_frame.core, result.input_frame.domain, n=1)
            m = tarot.nearest_card(result.medicine_frame.core, result.medicine_frame.domain, n=1)
            if p and m:
                count = kw_counts[num]
                hex_tarot_map[num] = (h["name"], p[0], m[0], count)
                presenting_cards[p[0][2]["name"]] += count
                medicine_cards[m[0][2]["name"]] += count

    print(f"\n  Per-Hexagram Tarot Mapping (via title as Oracle query)")
    print(f"  {'#':>3s} {'Hexagram':17s} {'Freq':>4s}  {'Presenting':>22s}  {'Medicine':>22s}")
    print(f"  {'-' * 75}")
    for num in sorted(hex_tarot_map.keys()):
        name, p, m, count = hex_tarot_map[num]
        pname = f"{p[2]['name']} ({p[1]}°)"
        mname = f"{m[2]['name']} ({m[1]}°)"
        print(f"  {num:3d} {name:17s} {count:4d}×  {pname:>22s}  {mname:>22s}")

    print(f"\n  Weighted Presenting Card Distribution")
    for name, count in presenting_cards.most_common():
        bar = "█" * (count // 2)
        print(f"    {name:22s} {count:3d}× {bar}")

    print(f"\n  Weighted Medicine Card Distribution")
    for name, count in medicine_cards.most_common():
        bar = "█" * (count // 2)
        print(f"    {name:22s} {count:3d}× {bar}")


def lookup_hexagram_by_number(num):
    """Look up hexagram data by King Wen number."""
    data = HEXAGRAM_NAMES.get(num)
    if data and isinstance(data, dict):
        return data
    for lower_name in TRIGRAM_INFO:
        for upper_name in TRIGRAM_INFO:
            h = lookup_hexagram(lower_name, upper_name)
            if h and h.get("number") == num:
                return h
    return None


def build_hexagram_lookup():
    """Build a num→hexagram lookup from all trigram pairs."""
    lookup = {}
    trigrams = ["QIAN", "KUN", "ZHEN", "KAN", "GEN", "XUN", "LI", "DUI"]
    for lower in trigrams:
        for upper in trigrams:
            h = lookup_hexagram(lower, upper)
            if h:
                lookup[h["number"]] = h
    return lookup

_HEX_LOOKUP = None

def get_hex(num):
    global _HEX_LOOKUP
    if _HEX_LOOKUP is None:
        _HEX_LOOKUP = build_hexagram_lookup()
    return _HEX_LOOKUP.get(num)


def lookup_hexagram_by_number(num):
    return get_hex(num)


def run_full_analysis():
    print()
    print("=" * 60)
    print("  PENNISTON BINARY CODE — ORACLE CORPUS ANALYSIS")
    print("  152 hexagrams from the Rendlesham Forest transmission")
    print("=" * 60)

    analyze_frequency(KING_WEN_SEQUENCE, "King Wen")
    analyze_frequency(TOLTEC_SEQUENCE, "Toltec")
    analyze_repeats(KING_WEN_SEQUENCE, "King Wen")
    analyze_repeats(TOLTEC_SEQUENCE, "Toltec")
    analyze_trigrams(KING_WEN_SEQUENCE, "King Wen")
    analyze_toltec_crossref()
    analyze_oracle_spread()

    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--tarot" in args:
        analyze_tarot_spread()
    elif "--freq" in args:
        analyze_frequency(KING_WEN_SEQUENCE, "King Wen")
        analyze_frequency(TOLTEC_SEQUENCE, "Toltec")
    elif "--trigrams" in args:
        analyze_trigrams(KING_WEN_SEQUENCE, "King Wen")
    elif "--crossref" in args:
        analyze_toltec_crossref()
    elif "--transitions" in args:
        analyze_transitions(KING_WEN_SEQUENCE, "King Wen")
    elif "--repeats" in args:
        analyze_repeats(KING_WEN_SEQUENCE, "King Wen")
        analyze_repeats(TOLTEC_SEQUENCE, "Toltec")
    else:
        run_full_analysis()
