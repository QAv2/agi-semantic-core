"""
Migrate legacy ExtendedDictionary (1,336 concepts) into SQLite.

Usage:
    cd ~/agi-semantic-core && python -m db.migrate
"""

import os
import sys
import sqlite3
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — legacy modules import from `octonion` directly, so we need
# both the project root (for `legacy.extended_dictionary`) and the legacy dir
# itself (so `from octonion import ...` resolves inside the legacy package).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = PROJECT_ROOT / "legacy"
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "semantic.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEGACY_DIR))

from legacy.extended_dictionary import ExtendedDictionary, ExtendedConcept

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_angles(c1: ExtendedConcept, c2: ExtendedConcept):
    """Return (angle_4d, angle_8d) between two concepts."""
    return c1.angle_4d(c2), c1.angle_8d(c2)


def run_migration():
    # ------------------------------------------------------------------
    # 1. Instantiate the legacy dictionary (takes a few seconds)
    # ------------------------------------------------------------------
    print("Loading legacy ExtendedDictionary ...")
    d = ExtendedDictionary()
    print(f"  Loaded {len(d.concepts)} dict entries, {len(d.relations)} relations")

    # ------------------------------------------------------------------
    # 2. Deduplicate concepts: only canonical entries (key == concept.name)
    # ------------------------------------------------------------------
    canonical = {}
    for key, concept in d.concepts.items():
        if key == concept.name:
            canonical[concept.name] = concept

    print(f"  Canonical concepts: {len(canonical)}")

    # ------------------------------------------------------------------
    # 3. Create / reset the SQLite database
    # ------------------------------------------------------------------
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Removed existing {DB_PATH.name}")

    schema_sql = SCHEMA_PATH.read_text()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(schema_sql)
    conn.execute("PRAGMA foreign_keys=ON")
    print(f"  Created {DB_PATH.name} with schema")

    # ------------------------------------------------------------------
    # 4. Insert concepts
    # ------------------------------------------------------------------
    name_to_id = {}
    trigram_dist = Counter()
    total_aliases = 0

    insert_concept_sql = """
        INSERT INTO concepts (
            name, w, x, y, z,
            e, f, g, h,
            fx, fy, fz, fe, ff, fg, fh,
            level, description, hexagram_ref, trigram, session
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    insert_alias_sql = """
        INSERT INTO aliases (alias, concept_id) VALUES (?, ?)
    """

    cur = conn.cursor()

    for concept in canonical.values():
        # Compute trigram
        try:
            tri = concept.trigram()
            tri_name = tri.name
        except Exception:
            tri_name = ""

        trigram_dist[tri_name] += 1

        cur.execute(insert_concept_sql, (
            concept.name,
            concept.w, concept.x, concept.y, concept.z,
            concept.e, concept.f, concept.g, concept.h,
            concept.fx, concept.fy, concept.fz,
            concept.fe, concept.ff, concept.fg, concept.fh,
            concept.level.value,
            concept.description,
            concept.hexagram_ref,
            tri_name,
            100,  # session — all legacy concepts built over 100 sessions
        ))

        concept_id = cur.lastrowid
        name_to_id[concept.name] = concept_id

        # Insert aliases
        for alias in concept.aliases:
            cur.execute(insert_alias_sql, (alias, concept_id))
            total_aliases += 1

    conn.commit()

    # ------------------------------------------------------------------
    # 5. Insert relations
    # ------------------------------------------------------------------
    insert_relation_sql = """
        INSERT OR IGNORE INTO relations (
            concept1_id, concept2_id, rel_type, angle_4d, angle_8d, session
        ) VALUES (?, ?, ?, ?, ?, ?)
    """

    # Build alias→canonical lookup so relations using alias names resolve
    alias_to_canonical = {}
    for key, concept in d.concepts.items():
        if key != concept.name:
            alias_to_canonical[key] = concept.name

    skipped = 0
    inserted = 0

    for name1, name2, rel_type, _legacy_angle in d.relations:
        # Resolve aliases to canonical names
        n1 = alias_to_canonical.get(name1, name1)
        n2 = alias_to_canonical.get(name2, name2)
        id1 = name_to_id.get(n1)
        id2 = name_to_id.get(n2)

        if id1 is None or id2 is None:
            skipped += 1
            continue

        # Compute fresh angles from the concept objects
        c1 = canonical[n1]
        c2 = canonical[n2]
        angle_4d, angle_8d = compute_angles(c1, c2)

        cur.execute(insert_relation_sql, (
            id1, id2,
            rel_type.value,  # e.g. "synonym", "complement"
            round(angle_4d, 4),
            round(angle_8d, 4),
            100,  # session
        ))
        inserted += 1

    conn.commit()

    if skipped:
        print(f"  WARNING: {skipped} relations skipped (concept not found)")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total_concepts = cur.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    total_aliases_db = cur.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    total_relations_db = cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    print()
    print("=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"  Concepts:   {total_concepts}")
    print(f"  Aliases:    {total_aliases_db}")
    print(f"  Relations:  {total_relations_db}")
    print()
    print("Trigram Distribution:")
    for tri_name, count in trigram_dist.most_common():
        label = tri_name if tri_name else "(none)"
        print(f"  {label:10s}  {count}")
    print()
    print(f"Database: {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    run_migration()
