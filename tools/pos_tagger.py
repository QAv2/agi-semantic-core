#!/usr/bin/env python3
"""
Auto-tag part of speech for untagged concepts.

Uses level heuristics + description keyword patterns.

Usage:
    python -m tools.pos_tagger              # Preview (dry run)
    python -m tools.pos_tagger --apply      # Apply tags to DB
    python -m tools.pos_tagger --stats      # Show POS distribution
"""

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_connection


def guess_pos(name: str, level: int, description: str) -> str:
    """Guess POS from level and description patterns."""
    desc = description.lower()

    # Already-disambiguated names
    if '_VERB' in name or '_ACTION' in name:
        return 'verb'
    if '_NOUN' in name or '_FRUIT' in name or '_GARMENT' in name:
        return 'noun'
    if '_COLOR' in name or '_SIZE' in name or '_WEIGHT' in name:
        return 'adj'

    # Level-based heuristics
    if level == 6:  # VERB
        return 'verb'
    if level == 8:  # INTERROGATIVE
        return 'pron'

    # Description pattern matching
    verb_starters = [
        'to ', 'act of ', 'process of ', 'make ', 'cause ', 'bring ',
        'move ', 'change ', 'give ', 'take ', 'put ', 'set ', 'get ',
        'turn ', 'come ', 'go ', 'send ', 'push ', 'pull ', 'carry ',
    ]
    if any(desc.startswith(p) for p in verb_starters):
        return 'verb'

    adj_patterns = [
        'quality of', 'state of being', 'characterized by', 'having the',
        'full of', 'lacking', 'without', 'opposite of', 'degree of',
        r'^(very |extremely |somewhat )',
        r'(ful|ous|ive|able|ible|ant|ent|ic|al|ary|ory)$',  # adjective suffixes in description
    ]
    # Check if the concept name ends with common adjective suffixes
    name_lower = name.lower()
    adj_suffixes = ('ful', 'ous', 'ive', 'able', 'ible', 'ant', 'ent', 'ic', 'al')
    if any(name_lower.endswith(s) for s in adj_suffixes) and level == 4:
        return 'adj'

    # Level 4 (QUALITY) — mostly adjectives
    if level == 4:
        # But some qualities are noun-like
        noun_quality_words = ['abundance', 'absence', 'presence', 'distance', 'silence']
        if any(w in name_lower for w in noun_quality_words):
            return 'noun'
        return 'adj'

    # Level 7 (ABSTRACT) — mostly nouns
    if level == 7:
        return 'noun'

    # Level 0-3 (UNITY, DYAD, TRIAD, TETRAD) — fundamental concepts = noun
    if level <= 3:
        return 'noun'

    # Level 5 (DERIVED) — could be noun or other
    # Check for verb-like descriptions
    if 'action' in desc or 'process' in desc or 'act of' in desc:
        return 'verb'

    # Preposition check
    prep_names = {'above', 'below', 'beside', 'between', 'beneath', 'beyond',
                  'inside', 'outside', 'across', 'along', 'among', 'around',
                  'behind', 'before', 'after', 'through', 'toward', 'towards',
                  'against', 'within', 'without', 'upon', 'under', 'over',
                  'near', 'far', 'next_to', 'into', 'onto', 'until', 'since'}
    if name_lower in prep_names:
        return 'prep'

    # Adverb check
    adv_names = {'quickly', 'slowly', 'gently', 'softly', 'loudly', 'quietly',
                 'always', 'never', 'sometimes', 'often', 'rarely', 'seldom',
                 'soon', 'now', 'then', 'here', 'there', 'already', 'still'}
    if name_lower in adv_names:
        return 'adv'

    # Default for DERIVED: noun
    return 'noun'


def run_tagger(apply: bool = False):
    """Tag all untagged concepts."""
    conn = get_connection()

    # Get untagged concepts
    rows = conn.execute(
        "SELECT id, name, level, description, pos FROM concepts WHERE pos = '' OR pos IS NULL"
    ).fetchall()

    print(f"Untagged concepts: {len(rows)}")
    print()

    tags = {}
    uncertain = []

    for row in rows:
        pos = guess_pos(row['name'], row['level'], row['description'])
        tags[row['id']] = (row['name'], pos)

    # Show preview
    pos_counts = {}
    for cid, (name, pos) in tags.items():
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    print("POS distribution for untagged:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos:8s} {count:4d}")

    if not apply:
        print()
        print("DRY RUN — showing first 30:")
        for i, (cid, (name, pos)) in enumerate(tags.items()):
            if i >= 30:
                print(f"  ... and {len(tags) - 30} more")
                break
            print(f"  {name:25s} -> {pos}")
        print()
        print("Run with --apply to write to database.")
    else:
        for cid, (name, pos) in tags.items():
            conn.execute("UPDATE concepts SET pos=? WHERE id=?", (pos, cid))
        conn.commit()
        print(f"\nApplied {len(tags)} POS tags.")

    conn.close()


def show_stats():
    """Show POS distribution across all concepts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT pos, COUNT(*) as cnt FROM concepts GROUP BY pos ORDER BY cnt DESC"
    ).fetchall()

    print("POS Distribution:")
    for r in rows:
        pos = r['pos'] if r['pos'] else '(untagged)'
        print(f"  {pos:12s} {r['cnt']:4d}")

    conn.close()


def main():
    if '--stats' in sys.argv:
        show_stats()
    elif '--apply' in sys.argv:
        run_tagger(apply=True)
    else:
        run_tagger(apply=False)


if __name__ == "__main__":
    main()
