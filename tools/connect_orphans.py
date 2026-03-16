"""Connect 38 orphaned concepts (no relations) to the semantic graph.

Session 103 — Infrastructure hardening.
Manually curated semantic relations based on meaning, not vector proximity.
Uses the builder's add_relation to compute proper angles.
"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.builder import ConceptBuilder
from core.encoding import RelationType

# Curated semantic connections: (concept1, concept2)
# Relation type is determined by the actual geometric angle:
#   <15° → synonym, 15-45° → affinity, 45-75° → adjacent, 80-105° → complement, >150° → opposition
# This avoids mismatch between semantic intent and vector reality.
CONNECTIONS = [
    # Prepositions/Spatial
    ("ACROSS", "MOVE"), ("ACROSS", "BESIDE"),
    ("AROUND", "NEAR"), ("AROUND", "BESIDE"),
    ("BEHIND", "BACK"), ("BEHIND", "FRONT"),
    ("NEXT_TO", "BESIDE"), ("NEXT_TO", "NEAR"),
    ("CENTER", "EDGE"), ("CENTER", "CLOSE"),
    # Animals
    ("BIRD", "FLY"),
    ("FISH", "SWIM"), ("FISH", "WATER"),
    # Household/Objects
    ("BOOK", "SURFACE"),
    ("CEILING_LIGHT", "CEILING"), ("CEILING_LIGHT", "LIGHT"),
    ("CLOSET", "ROOM"), ("CLOSET", "CLOSE"),
    ("FAUCET", "WATER"),
    ("OUTLET", "POWER"),
    ("SCREEN", "SURFACE"), ("SCREEN", "SEE"),
    ("SIDEWALK", "PATH"), ("SIDEWALK", "WALK"),
    ("STORE", "PLACE"),
    ("TELEPHONE", "VOICE"),
    # Food
    ("CHEESE", "FOOD"),
    ("SNACK", "EAT"), ("SNACK", "FOOD"),
    # Qualities/States
    ("BLOCKED", "OPEN"),
    ("BROKEN", "WORK"),
    ("BUMPY", "ROUGH"), ("BUMPY", "SMOOTH"),
    ("CAREFUL", "GENTLE"),
    ("FUZZY", "SOFT"),
    ("LIGHT_WEIGHT", "SMALL"),
    ("READY", "OPEN"),
    ("REACHABLE", "NEAR"), ("REACHABLE", "REACH"),
    ("SMOOTH", "ROUGH"), ("SMOOTH", "SOFT"),
    # Verbs
    ("COLOR_VERB", "DRAW"),
    ("CONNECT", "CLOSE"),
    ("DRAW", "HAND"),
    ("HIDE", "CONCEAL"), ("HIDE", "SEE"),
    ("PET_VERB", "GENTLE"), ("PET_VERB", "HAND"),
    ("POINT", "FINGER"), ("POINT", "SEE"),
    ("SQUEEZE", "HAND"),
    ("STOP", "END"), ("STOP", "MOVE"),
    ("YAWN", "TIRED"), ("YAWN", "MOUTH"), ("YAWN", "SLEEP"),
    # Nature
    ("BUSH", "TREE"), ("BUSH", "SMALL"),
    # Places
    ("CHURCH", "PLACE"),
]


def classify_angle(angle):
    """Determine relation type from geometric angle.
    Matches validation thresholds: synonym <30°, complement 80-105°."""
    if angle < 15:
        return RelationType.SYNONYM
    elif angle < 45:
        return RelationType.AFFINITY
    elif angle < 80:
        return RelationType.ADJACENT
    elif angle <= 105:
        return RelationType.COMPLEMENT
    elif angle > 150:
        return RelationType.OPPOSITION
    else:
        return RelationType.ADJACENT  # 105-150° gray zone → adjacent


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    builder = ConceptBuilder("db/semantic.db")

    print("=" * 60)
    print("  ORPHAN CONNECTOR — Session 103")
    print("  Mode:", "DRY RUN" if dry_run else "LIVE")
    print("=" * 60)
    print()

    added = 0
    failed = 0
    skipped = 0

    # First pass: compute angles to determine relation types
    import numpy as np

    conn = sqlite3.connect("db/semantic.db")
    conn.row_factory = sqlite3.Row

    for n1, n2 in CONNECTIONS:
        r1 = conn.execute("SELECT x,y,z FROM concepts WHERE name=? COLLATE NOCASE", (n1,)).fetchone()
        r2 = conn.execute("SELECT x,y,z FROM concepts WHERE name=? COLLATE NOCASE", (n2,)).fetchone()
        if not r1 or not r2:
            print(f"  SKIP: {n1} or {n2} not found")
            failed += 1
            continue

        v1 = np.array([r1["x"], r1["y"], r1["z"]], dtype=float)
        v2 = np.array([r2["x"], r2["y"], r2["z"]], dtype=float)
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            angle = 0.0
        else:
            cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_t))

        rel_type = classify_angle(angle)

        ok, msg = builder.add_relation(n1, n2, rel_type, session=103)

        if "already exists" in msg.lower():
            skipped += 1
            continue

        if ok:
            if dry_run:
                print(f"  WOULD ADD: {n1:18s} ↔ {n2:18s} {rel_type.value:12s} {angle:5.1f}°")
            else:
                print(f"  {n1:18s} ↔ {n2:18s} {rel_type.value:12s} {angle:5.1f}°")
            added += 1
        else:
            print(f"  FAIL: {msg}")
            failed += 1

    conn.close()

    if dry_run:
        print(f"\n  DRY RUN: {added} would be added, {failed} failed, {skipped} already exist")
    else:
        print(f"\n  {added} added, {failed} failed, {skipped} already existed")

    # Check remaining orphans
    conn = sqlite3.connect("db/semantic.db")
    remaining = conn.execute('''
        SELECT c.name FROM concepts c
        WHERE c.id NOT IN (SELECT concept1_id FROM relations)
        AND c.id NOT IN (SELECT concept2_id FROM relations)
        ORDER BY c.name
    ''').fetchall()
    conn.close()

    if remaining:
        print(f"\n  Still orphaned ({len(remaining)}):")
        for r in remaining:
            print(f"    {r[0]}")
    else:
        print(f"\n  All concepts connected!")

    builder.close()


if __name__ == "__main__":
    main()
