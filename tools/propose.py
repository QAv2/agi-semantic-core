#!/usr/bin/env python3
"""
Vector Proposal Tool — Semi-automated encoding for new concepts.

Given a word and optional description, proposes a candidate 16D vector
by analyzing semantic context from existing dictionary entries.

Usage:
    python -m tools.propose WORD "description of the concept"
    python -m tools.propose WOLF "Predatory canine, pack hunter"
    python -m tools.propose STATE "Condition or mode of being"
    python -m tools.propose --batch words.txt     # One word per line
    python -m tools.propose --json WOLF "..."     # Output as insertable JSON
"""

import sys
import os
import re
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore
from core.encoding import ConceptLevel


def extract_context_words(description: str) -> list:
    """Extract meaningful words from a description."""
    stop = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'and', 'or', 'but', 'not', 'no', 'that', 'which', 'who', 'this',
        'it', 'its', 'very', 'more', 'most', 'also', 'just', 'than',
    }
    words = re.findall(r'[a-z]+', description.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def guess_level(word: str, description: str) -> ConceptLevel:
    """Heuristic guess at ontological level."""
    desc_lower = description.lower()
    # Verb indicators
    verb_patterns = ['to ', 'act of ', 'process of ', 'making ', 'causing ']
    if any(desc_lower.startswith(p) or f' {p}' in desc_lower for p in verb_patterns):
        return ConceptLevel.VERB

    # Quality/adjective indicators
    qual_patterns = ['quality', 'state of', 'condition', 'degree of', 'manner',
                     'having', 'characterized by', 'full of', 'lacking']
    if any(p in desc_lower for p in qual_patterns):
        return ConceptLevel.QUALITY

    # Abstract
    abstract_patterns = ['concept', 'principle', 'idea', 'belief', 'theory',
                         'philosophy', 'virtue', 'value', 'ideal']
    if any(p in desc_lower for p in abstract_patterns):
        return ConceptLevel.ABSTRACT

    return ConceptLevel.DERIVED


def guess_pos(level: ConceptLevel, description: str) -> str:
    """Heuristic guess at POS."""
    if level == ConceptLevel.VERB:
        return 'verb'
    elif level == ConceptLevel.QUALITY:
        return 'adj'
    elif level == ConceptLevel.ABSTRACT:
        return 'noun'
    elif level == ConceptLevel.INTERROGATIVE:
        return 'pron'
    return 'noun'


def propose_vector(sc: SemanticCore, word: str, description: str = "",
                   verbose: bool = True) -> dict:
    """
    Propose a 16D vector for a new concept.

    Strategy:
    1. Find existing concepts mentioned in the description
    2. Find concepts whose descriptions mention this word
    3. Weight these context concepts by relevance
    4. Interpolate their vectors to propose a candidate
    5. Show nearest neighbors for human review
    """
    word_upper = word.upper()

    # Check if already exists
    existing = sc.get(word_upper)
    if existing:
        if verbose:
            print(f"  NOTE: {word_upper} already exists in dictionary")
            print(f"  {existing}")
        return None

    # Strategy 1: Find defined concepts in the description
    desc_words = extract_context_words(description) if description else []
    desc_concepts = []
    for w in desc_words:
        c = sc.get(w.upper())
        if c:
            desc_concepts.append(c)

    # Strategy 2: Find concepts whose descriptions mention this word
    mention_rows = sc.conn.execute(
        "SELECT name FROM concepts WHERE description LIKE ? LIMIT 20",
        (f"% {word.lower()} %",)
    ).fetchall()
    mention_concepts = [sc.get(r['name']) for r in mention_rows if sc.get(r['name'])]

    # Strategy 3: Find nearest by name substring
    name_rows = sc.conn.execute(
        "SELECT name FROM concepts WHERE name LIKE ? LIMIT 10",
        (f"%{word_upper}%",)
    ).fetchall()
    name_concepts = [sc.get(r['name']) for r in name_rows if sc.get(r['name'])]

    # Combine all context concepts with weights
    weighted = []
    seen = set()
    for c in desc_concepts:
        if c.name not in seen:
            weighted.append((c, 3.0))  # Description words get highest weight
            seen.add(c.name)
    for c in mention_concepts:
        if c.name not in seen:
            weighted.append((c, 1.5))  # Mentioned-in gets medium weight
            seen.add(c.name)
    for c in name_concepts:
        if c.name not in seen:
            weighted.append((c, 1.0))  # Name similarity gets lowest weight
            seen.add(c.name)

    if not weighted:
        if verbose:
            print(f"  No context found for '{word}'. Cannot propose vector.")
            print(f"  Try providing a description with words already in the dictionary.")
        return None

    # Weighted interpolation
    total_weight = sum(w for _, w in weighted)
    essence = np.zeros(7)
    function = np.zeros(7)
    for c, w in weighted:
        essence += w * c.essence_vector()
        function += w * c.function_vector()
    essence /= total_weight
    function /= total_weight

    # Guess level and POS
    level = guess_level(word, description)
    pos = guess_pos(level, description)

    # Build proposal
    proposal = {
        'name': word_upper,
        'x': round(float(essence[0]), 3),
        'y': round(float(essence[1]), 3),
        'z': round(float(essence[2]), 3),
        'e': round(float(essence[3]), 3),
        'f': round(float(essence[4]), 3),
        'g': round(float(essence[5]), 3),
        'h': round(float(essence[6]), 3),
        'fx': round(float(function[0]), 3),
        'fy': round(float(function[1]), 3),
        'fz': round(float(function[2]), 3),
        'fe': round(float(function[3]), 3),
        'ff': round(float(function[4]), 3),
        'fg': round(float(function[5]), 3),
        'fh': round(float(function[6]), 3),
        'level': level.value,
        'pos': pos,
        'description': description,
    }

    # Compute trigram
    from core.octonion import SemanticOctonion
    octo = SemanticOctonion(w=1.0, x=proposal['x'], y=proposal['y'], z=proposal['z'],
                            e=proposal['e'], f=proposal['f'], g=proposal['g'], h=proposal['h'])
    try:
        trigram = octo.to_trigram(x_axis=proposal['x'], y_axis=proposal['y']).name
    except Exception:
        trigram = "?"

    # Find nearest neighbors of proposal
    prop_vec = np.array([proposal['x'], proposal['y'], proposal['z'],
                         proposal['e'], proposal['f'], proposal['g'], proposal['h']])
    nearest = sc.search_by_vector(prop_vec, n=8)

    # Dedup check
    too_close = [(n, a) for n, a in nearest if a < 15.0]

    if verbose:
        print("=" * 60)
        print(f"  PROPOSAL: {word_upper}")
        print("=" * 60)
        print(f"  Description: {description}")
        print(f"  Level:       {level.name} (guessed)")
        print(f"  POS:         {pos}")
        print(f"  Trigram:     {trigram}")
        print()
        print(f"  Core:     [{proposal['x']:+.3f}, {proposal['y']:+.3f}, {proposal['z']:+.3f}]")
        print(f"  Domain:   [{proposal['e']:.3f}, {proposal['f']:.3f}, {proposal['g']:.3f}, {proposal['h']:.3f}]")
        print(f"  Function: [{proposal['fx']:+.3f}, {proposal['fy']:+.3f}, {proposal['fz']:+.3f}]")
        print(f"  Fn Dom:   [{proposal['fe']:.3f}, {proposal['ff']:.3f}, {proposal['fg']:.3f}, {proposal['fh']:.3f}]")
        print()
        print(f"  Context concepts ({len(weighted)}):")
        for c, w in weighted[:10]:
            print(f"    {c.name:20s} (weight={w:.1f})")
        print()
        print(f"  Nearest neighbors of proposal:")
        for name, angle in nearest:
            flag = " ** TOO CLOSE" if angle < 15.0 else ""
            print(f"    {name:20s} {angle:5.1f}°{flag}")

        if too_close:
            print()
            print(f"  WARNING: {len(too_close)} concepts within 15° — may need adjustment")

    return proposal


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    sc = SemanticCore()

    try:
        if sys.argv[1] == '--json':
            word = sys.argv[2]
            desc = sys.argv[3] if len(sys.argv) > 3 else ""
            proposal = propose_vector(sc, word, desc, verbose=False)
            if proposal:
                print(json.dumps(proposal, indent=2))
        elif sys.argv[1] == '--batch':
            filepath = sys.argv[2]
            with open(filepath) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            proposals = []
            for line in lines:
                parts = line.split('|', 1)
                word = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                p = propose_vector(sc, word, desc, verbose=False)
                if p:
                    proposals.append(p)
                    print(f"  {word:20s} -> [{p['x']:+.3f},{p['y']:+.3f},{p['z']:+.3f}]")
                else:
                    print(f"  {word:20s} -> SKIP (no context or already exists)")
            # Write batch JSON
            out_path = filepath.rsplit('.', 1)[0] + '_proposals.json'
            with open(out_path, 'w') as f:
                json.dump(proposals, f, indent=2)
            print(f"\n  Wrote {len(proposals)} proposals to {out_path}")
        else:
            word = sys.argv[1]
            desc = sys.argv[2] if len(sys.argv) > 2 else ""
            proposal = propose_vector(sc, word, desc)
            if proposal:
                print()
                print("  JSON (for batch insert):")
                print(f"  {json.dumps(proposal)}")
    finally:
        sc.close()


if __name__ == "__main__":
    main()
