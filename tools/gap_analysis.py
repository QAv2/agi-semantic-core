#!/usr/bin/env python3
"""
Gap Analysis — Find missing definition words and measure self-containment.

Usage:
    python -m tools.gap_analysis                  # Full analysis
    python -m tools.gap_analysis missing 50       # Top 50 missing words
    python -m tools.gap_analysis containment      # Self-containment % only
    python -m tools.gap_analysis domains           # Domain coverage check
"""

import sys
import os
import re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore

# Common English stop words — not meaningful for self-containment
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'which',
    'who', 'whom', 'this', 'these', 'those', 'am', 'it', 'its', 'i', 'me',
    'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'they', 'them', 'their', 'what', 'about', 'up', 'also', 'e', 'g',
    'etc', 'vs', 'ie', 'eg', 'like', 'without', 'within', 'toward',
    'towards', 'upon', 'across', 'along', 'among', 'around',
}

# Words that are artifacts of encoding descriptions, not semantic content
NOISE_WORDS = {
    'yang', 'yin', 'pure', 'session', 'fix', 'adjusted', 'calibrated',
    'orthogonal', 'complement', 'encoded', 'encoding', 'domain', 'axis',
    'component', 'vector', 'trigram', 'hexagram', 'qian', 'kun',
    'number', 'fold', 'symmetry', 'geometry', 'shape', 'polygon',
}


def extract_content_words(text: str) -> list:
    """Extract meaningful content words from a description."""
    # Lowercase, split on non-alpha
    words = re.findall(r'[a-z]+', text.lower())
    # Filter stop words, noise, and very short words
    return [w for w in words if w not in STOP_WORDS
            and w not in NOISE_WORDS and len(w) > 2]


def run_self_containment(sc: SemanticCore, verbose: bool = True):
    """Measure what % of definition words are themselves defined."""
    all_names = set(n.upper() for n in sc.all_concepts())

    # Also include aliases
    alias_rows = sc.conn.execute("SELECT alias FROM aliases").fetchall()
    for r in alias_rows:
        all_names.add(r['alias'].upper())

    rows = sc.conn.execute("SELECT name, description FROM concepts").fetchall()

    all_def_words = Counter()
    missing_words = Counter()
    total_words = 0
    found_words = 0

    for row in rows:
        words = extract_content_words(row['description'])
        for w in words:
            all_def_words[w] += 1
            total_words += 1
            if w.upper() in all_names:
                found_words += 1
            else:
                missing_words[w] += 1

    unique_def_words = len(all_def_words)
    unique_found = unique_def_words - len(missing_words)
    containment_pct = 100 * unique_found / unique_def_words if unique_def_words > 0 else 0
    token_containment_pct = 100 * found_words / total_words if total_words > 0 else 0

    if verbose:
        print("=" * 60)
        print("  SELF-CONTAINMENT ANALYSIS")
        print("=" * 60)
        print(f"  Concepts in dictionary:   {len(rows)}")
        print(f"  Unique definition words:  {unique_def_words}")
        print(f"  Unique words defined:     {unique_found}")
        print(f"  Unique words missing:     {len(missing_words)}")
        print(f"  Self-containment (type):  {containment_pct:.1f}%")
        print(f"  Self-containment (token): {token_containment_pct:.1f}%")

    return containment_pct, missing_words


def show_missing(sc: SemanticCore, n: int = 50):
    """Show top N missing definition words."""
    _, missing = run_self_containment(sc, verbose=True)

    print()
    print(f"  TOP {n} MISSING DEFINITION WORDS")
    print("  " + "-" * 50)
    print(f"  {'WORD':25s} {'COUNT':>6s}  CONTEXT")

    for word, count in missing.most_common(n):
        # Find a concept that uses this word
        row = sc.conn.execute(
            "SELECT name FROM concepts WHERE description LIKE ? LIMIT 1",
            (f"%{word}%",)
        ).fetchone()
        context = row['name'] if row else ""
        print(f"  {word:25s} {count:6d}  (in {context})")


def check_domains(sc: SemanticCore):
    """Check coverage across semantic domains."""
    # Domain word lists (representative, not exhaustive)
    domains = {
        'Animals': ['dog', 'cat', 'bird', 'fish', 'horse', 'wolf', 'lion', 'eagle',
                     'snake', 'bear', 'deer', 'rabbit', 'mouse', 'whale', 'shark',
                     'butterfly', 'bee', 'ant', 'spider', 'frog', 'owl', 'crow',
                     'elephant', 'tiger', 'monkey'],
        'Materials': ['wood', 'metal', 'iron', 'glass', 'gold', 'silver', 'paper',
                       'cloth', 'silk', 'cotton', 'wool', 'leather', 'plastic',
                       'copper', 'bronze', 'steel', 'clay', 'stone', 'sand', 'lead'],
        'Body parts': ['head', 'hand', 'heart', 'eye', 'ear', 'nose', 'mouth',
                        'arm', 'leg', 'foot', 'finger', 'bone', 'blood', 'brain',
                        'skin', 'hair', 'face', 'neck', 'chest', 'back', 'shoulder',
                        'knee', 'tooth', 'tongue', 'lung'],
        'Nature': ['tree', 'river', 'mountain', 'sky', 'sun', 'moon', 'star',
                    'ocean', 'lake', 'forest', 'valley', 'island', 'desert',
                    'cloud', 'rain', 'snow', 'wind', 'storm', 'flower', 'grass',
                    'rock', 'hill', 'cave', 'cliff', 'shore'],
        'Social roles': ['mother', 'father', 'child', 'friend', 'teacher', 'king',
                          'queen', 'soldier', 'priest', 'doctor', 'judge', 'servant',
                          'master', 'sister', 'brother', 'husband', 'wife', 'neighbor',
                          'stranger', 'warrior', 'elder', 'leader', 'artist', 'farmer',
                          'merchant', 'student', 'healer', 'guardian', 'ancestor'],
        'Food': ['bread', 'meat', 'fruit', 'milk', 'egg', 'salt', 'sugar', 'rice',
                  'grain', 'wine', 'honey', 'cheese', 'fish', 'soup', 'cake',
                  'butter', 'oil', 'tea', 'coffee', 'vegetable'],
        'Time': ['today', 'tomorrow', 'yesterday', 'morning', 'evening', 'night',
                  'noon', 'midnight', 'hour', 'minute', 'second', 'week', 'month',
                  'year', 'century', 'season', 'dawn', 'dusk', 'moment', 'era'],
        'Conjunctions': ['and', 'or', 'but', 'if', 'because', 'although', 'while',
                          'since', 'unless', 'until', 'whether', 'whereas', 'however',
                          'therefore', 'moreover', 'nevertheless', 'furthermore',
                          'meanwhile', 'otherwise', 'hence', 'thus', 'yet'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown',
                    'grey', 'purple', 'orange', 'pink', 'gold', 'silver'],
        'Weather': ['rain', 'snow', 'storm', 'wind', 'cloud', 'fog', 'thunder',
                     'lightning', 'frost', 'hail', 'drought', 'flood', 'breeze',
                     'hurricane', 'tornado'],
    }

    print("=" * 60)
    print("  DOMAIN COVERAGE")
    print("=" * 60)

    total_have = 0
    total_need = 0

    for domain, words in sorted(domains.items()):
        have = []
        missing = []
        for w in words:
            if sc.exists(w) or sc.exists(w.upper()):
                have.append(w)
            else:
                missing.append(w)

        total_have += len(have)
        total_need += len(words)
        pct = 100 * len(have) / len(words) if words else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {domain:18s} {len(have):3d}/{len(words):3d} ({pct:4.0f}%) [{bar}]")
        if missing and len(missing) <= 10:
            print(f"    missing: {', '.join(missing)}")
        elif missing:
            print(f"    missing: {', '.join(missing[:8])}... (+{len(missing)-8})")

    overall = 100 * total_have / total_need if total_need > 0 else 0
    print()
    print(f"  Overall: {total_have}/{total_need} ({overall:.1f}%)")


def run_full(sc: SemanticCore, n: int = 50):
    """Full gap analysis."""
    show_missing(sc, n=n)
    print()
    check_domains(sc)

    # Quick summary of what to encode next
    _, missing = run_self_containment(sc, verbose=False)
    print()
    print("=" * 60)
    print("  RECOMMENDED NEXT BATCH (top 20 by self-containment impact)")
    print("=" * 60)
    for word, count in missing.most_common(20):
        print(f"  {word:25s} (used in {count} definitions)")


def main():
    sc = SemanticCore()
    try:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            if cmd == "missing":
                n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
                show_missing(sc, n=n)
            elif cmd == "containment":
                run_self_containment(sc, verbose=True)
            elif cmd == "domains":
                check_domains(sc)
            else:
                run_full(sc)
        else:
            run_full(sc)
    finally:
        sc.close()


if __name__ == "__main__":
    main()
