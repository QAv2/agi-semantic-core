"""Fix crowded vector pairs and broken angle relations.

Session 103 — Infrastructure hardening (v2: constraint-aware).
Grid-searched optimal vectors that satisfy ALL existing relation constraints
(complements 80-105°, synonyms <30°) while separating identical/crowded pairs.

Changes:
- 3 identical-vector pairs differentiated (GRAVITY, SOLSTICE, ENHANCEMENT)
- 5 near-identical pairs differentiated (TRICKLE, COLOR_VERB, BE, DEFERENCE, INSPIRED)
- FEAR complement angle corrected (76.6° → ~80°)
- VANISH↔FADE: relation reclassified synonym → affinity (they were never <30°;
  original VANISH had correct complement to ABOVE at 80.9°)
"""

import sqlite3
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.octonion import SemanticOctonion


def angle_3d(v1, v2):
    """Angle between two 3D vectors in degrees."""
    v1, v2 = np.array(v1), np.array(v2)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-10 or m2 < 1e-10:
        return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def angle_7d(v1, v2):
    """Angle between two 7D vectors [x,y,z,e,f,g,h] in degrees."""
    v1, v2 = np.array(v1), np.array(v2)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-10 or m2 < 1e-10:
        return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def compute_trigram(x, y, z, e, f, g, h):
    """Compute trigram name for a vector."""
    octo = SemanticOctonion(w=1.0, x=x, y=y, z=z, e=e, f=f, g=g, h=h)
    return octo.to_trigram(x_axis=x, y_axis=y).name


# =========================================================================
# CORRECTIONS — constraint-satisfying vectors (v2)
# Grid-searched to satisfy ALL existing complement/synonym angles.
# =========================================================================

CORRECTIONS = {
    # --- Identical vector pairs (0°) ---

    # GRAVITY ↔ SOLEMNITY (anchor stays at [-0.35,-0.10,+0.70])
    # Grid search: must keep complements GLEE(104.8°), ABOVE(98.8°), PLAYFUL(104.7°) in [80,105]
    "GRAVITY": {
        "essence": [-0.50, -0.15, +0.55, +0.50, +0.40, +0.35, +0.50],
        "function": [-0.40, -0.10, +0.45, +0.40, +0.30, +0.30, +0.40],
        "reason": "Was identical to SOLEMNITY. Conservative shift: 16° separation, all 3 complements preserved (GLEE/ABOVE/PLAYFUL)."
    },

    # SOLSTICE ↔ GENESIS (anchor stays at [+0.416,-0.398,-0.398])
    # Grid search: must keep complement PAST in [80,105]
    "SOLSTICE": {
        "essence": [+0.10, -0.60, -0.43, +0.75, +0.85, +0.20, +0.15],
        "function": [-0.05, -0.45, -0.30, +0.50, +0.60, +0.15, +0.10],
        "reason": "Was identical to GENESIS. Shifted cyclical/temporal: 30° from GENESIS, PAST complement preserved (80.7°)."
    },

    # ENHANCEMENT ↔ EMERGENCE (anchor stays at [+0.41,-0.41,-0.392])
    # No complement constraints broken in v1 — keep that correction.
    "ENHANCEMENT": {
        "essence": [+0.50, -0.15, -0.20, +0.30, +0.45, +0.65, +0.55],
        "function": [-0.15, +0.30, +0.30, +0.25, +0.35, +0.55, +0.45],
        "reason": "Was identical to EMERGENCE. Shifted relational/personal: 28° separation, no broken relations."
    },

    # --- Near-identical pairs (<2°) ---

    # TRICKLE ↔ SMOKE (anchor) — no broken relations in v1
    "TRICKLE": {
        "essence": [-0.50, +0.30, +0.15, +0.65, +0.70, +0.40, +0.25],
        "function": [-0.45, +0.25, +0.10, +0.55, +0.60, +0.35, +0.20],
        "reason": "Was identical core to SMOKE. Shifted spatial/yin: 24° separation, no broken relations."
    },

    # COLOR_VERB ↔ DRAW (anchor) — no broken relations in v1
    "COLOR_VERB": {
        "essence": [+0.30, +0.15, +0.10, +0.65, +0.30, +0.45, +0.70],
        "function": [+0.25, +0.10, +0.05, +0.55, +0.25, +0.40, +0.60],
        "reason": "Was 1.1° from DRAW. Shifted less-becoming/spatial: 22° separation, stays in DUI."
    },

    # BE ↔ BEING (anchor = zero vector)
    # Constraint: synonym AM [0,0,0.01] (<30°), complement BECOME [+0.10,+0.80,+0.10] (80-105°)
    # Near-zero core [0,0,0.03] satisfies both: AM=0°, BECOME=82.9°
    # Domain provides real differentiation (temporal/relational)
    "BE": {
        "essence": [0.00, 0.00, +0.03, +0.30, +0.60, +0.45, +0.35],
        "function": [0.00, 0.00, +0.02, +0.25, +0.55, +0.40, +0.30],
        "reason": "Was near-zero, same direction as BEING. Tiny core (z=0.03) preserves AM synonym + BECOME complement. Domain differentiates."
    },

    # DEFERENCE ↔ ATTRACTION (anchor at [-0.398,+0.415,-0.398])
    # Constraint: synonyms MODESTY/SIMPLICITY (<30°)
    # Conservative shift: 10° from MODESTY/SIMPLICITY (synonym ✓), 10° from ATTRACTION (differentiated)
    "DEFERENCE": {
        "essence": [-0.45, +0.35, -0.30, +0.20, +0.25, +0.75, +0.55],
        "function": [-0.35, +0.25, -0.20, +0.15, +0.20, +0.65, +0.45],
        "reason": "Was 0.7° from ATTRACTION. Conservative shift: 10° from synonyms MODESTY/SIMPLICITY, 10° from ATTRACTION."
    },

    # INSPIRED ↔ VISIONARY (anchor) — no broken relations in v1
    "INSPIRED": {
        "essence": [+0.40, +0.55, +0.20, +0.35, +0.40, +0.80, +0.65],
        "function": [+0.35, +0.50, +0.15, +0.30, +0.35, +0.75, +0.55],
        "reason": "Was 1.8° from VISIONARY. Shifted receptive/becoming: 19° separation, no broken relations."
    },

    # --- Broken angle pairs ---

    # VANISH: REVERT to original vector. The synonym relation with FADE was wrong
    # (original was 80.9° from ABOVE = correct complement, 105.6° from FADE = not a synonym).
    # Relation fix handled separately below.
    "VANISH": {
        "essence": [-0.40, +0.60, +0.25, +0.40, +0.76, +0.30, +0.35],
        "function": [-0.35, +0.52, +0.20, +0.35, +0.62, +0.25, +0.30],
        "reason": "REVERTED to original. FADE synonym was misclassified (105.6°). Original correctly complements ABOVE (80.9°)."
    },

    # FEAR: minimal adjustment from original [-0.70,+0.60,+0.65]
    # Grid search: REST(80.1°), LOVE(80.3°), HOPE(89.2°), ANGER(92.8°) — all in [80,105]
    "FEAR": {
        "essence": [-0.76, +0.62, +0.59, +0.40, +0.50, +0.30, +0.80],
        "function": [-0.20, +0.30, -0.40, -0.40, +0.40, -0.30, -0.40],
        "reason": "Minimal shift from original. All 4 complements in range (REST/LOVE/HOPE/ANGER)."
    },
}

# Pair angle checks
PAIR_CHECKS = [
    ("GRAVITY", "SOLEMNITY", "affinity", 10, 45),
    ("SOLSTICE", "GENESIS", "affinity", 10, 45),
    ("ENHANCEMENT", "EMERGENCE", "affinity", 15, 45),
    ("TRICKLE", "SMOKE", "affinity", 10, 40),
    ("COLOR_VERB", "DRAW", "affinity", 15, 45),
    ("BE", "BEING", None, 0, 180),
    ("DEFERENCE", "ATTRACTION", "affinity", 5, 45),
    ("INSPIRED", "VISIONARY", "affinity", 15, 30),
    ("VANISH", "ABOVE", "complement", 75, 105),  # Check ABOVE complement, not FADE synonym
    ("FEAR", "LOVE", "complement", 80, 105),
]

# Relation reclassifications (applied after vector updates)
RELATION_FIXES = [
    # VANISH↔FADE: was synonym, should be affinity (never was <30°)
    ("VANISH", "FADE", "synonym", "affinity"),
]


def get_concept_vector(conn, name):
    """Get [x,y,z,e,f,g,h] for a concept."""
    row = conn.execute(
        "SELECT x, y, z, e, f, g, h FROM concepts WHERE name=?", (name,)
    ).fetchone()
    if row:
        return list(row)
    return None


def check_neighborhood_crowding(conn, name, new_vec, threshold=5.0):
    """Check if new vector crowds any existing non-synonym concept.
    Excludes BEING (zero vector by design) and other zero-magnitude concepts."""
    new_core = new_vec[:3]
    rows = conn.execute(
        "SELECT c.name, c.x, c.y, c.z FROM concepts c WHERE c.name != ? AND c.name != 'BEING'", (name,)
    ).fetchall()

    crowded = []
    for r in rows:
        other_core = [r[1], r[2], r[3]]
        if np.linalg.norm(other_core) < 1e-10:
            continue
        ang = angle_3d(new_core, other_core)
        if ang < threshold:
            is_syn = conn.execute("""
                SELECT 1 FROM relations r
                JOIN concepts c1 ON r.concept1_id = c1.id
                JOIN concepts c2 ON r.concept2_id = c2.id
                WHERE ((c1.name=? AND c2.name=?) OR (c1.name=? AND c2.name=?))
                AND r.rel_type = 'synonym'
            """, (name, r[0], r[0], name)).fetchone()
            if not is_syn:
                crowded.append((r[0], ang))

    return sorted(crowded, key=lambda x: x[1])


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    conn = sqlite3.connect("db/semantic.db")
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("  CROWDING FIX v2 — Session 103 (constraint-aware)")
    print("  Mode:", "DRY RUN" if dry_run else "LIVE (will modify database)")
    print("=" * 70)
    print()

    all_ok = True
    updates = []

    for name, correction in CORRECTIONS.items():
        new_essence = correction["essence"]
        new_function = correction["function"]
        reason = correction["reason"]

        cur = conn.execute(
            "SELECT x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh, trigram FROM concepts WHERE name=?",
            (name,)
        ).fetchone()
        if not cur:
            print(f"  ERROR: {name} not found!")
            all_ok = False
            continue

        old_core = [cur["x"], cur["y"], cur["z"]]
        new_core = new_essence[:3]
        old_trigram = cur["trigram"]
        new_trigram = compute_trigram(*new_essence)

        pair_info = next((p for p in PAIR_CHECKS if p[0] == name), None)
        if pair_info:
            _, partner, rel_type, min_ang, max_ang = pair_info
            partner_vec = get_concept_vector(conn, partner)
            if partner_vec:
                new_pair_angle = angle_3d(new_core, partner_vec[:3])
                old_pair_angle = angle_3d(old_core, partner_vec[:3])
                angle_ok = min_ang <= new_pair_angle <= max_ang
            else:
                new_pair_angle = old_pair_angle = 0
                angle_ok = True
        else:
            new_pair_angle = old_pair_angle = 0
            angle_ok = True

        if not angle_ok:
            all_ok = False

        crowded = check_neighborhood_crowding(conn, name, new_essence, threshold=5.0)

        trigram_change = f" → {new_trigram}" if old_trigram != new_trigram else ""

        print(f"  {name}")
        print(f"    {reason}")
        if pair_info:
            print(f"    {pair_info[1]}: {old_pair_angle:.1f}° → {new_pair_angle:.1f}°"
                  f"  [target: {pair_info[3]}-{pair_info[4]}°] {'✓' if angle_ok else '✗'}")
        print(f"    Trigram: {old_trigram}{trigram_change}")
        if crowded:
            print(f"    Nearby: {', '.join(f'{n}({a:.1f}°)' for n, a in crowded[:3])}")
        print()

        updates.append((name, new_essence, new_function, new_trigram))

    # Show relation fixes
    print("  RELATION RECLASSIFICATIONS:")
    for n1, n2, old_type, new_type in RELATION_FIXES:
        print(f"    {n1}↔{n2}: {old_type} → {new_type}")
    print()

    if not all_ok:
        print("  ANGLE TARGET MISSES — review above.")
        if not dry_run:
            resp = input("  Continue anyway? [y/N] ")
            if resp.lower() != 'y':
                conn.close()
                return

    if dry_run:
        print(f"  DRY RUN complete. {len(updates)} vector corrections + {len(RELATION_FIXES)} relation fixes.")
        conn.close()
        return

    # Apply vector updates
    print("-" * 70)
    print("  APPLYING VECTOR CORRECTIONS...")
    for name, essence, function, trigram in updates:
        x, y, z, e, f, g, h = essence
        fx, fy, fz, fe, ff, fg, fh = function
        conn.execute("""
            UPDATE concepts SET
                x=?, y=?, z=?, e=?, f=?, g=?, h=?,
                fx=?, fy=?, fz=?, fe=?, ff=?, fg=?, fh=?,
                trigram=?, session=103, updated_at=datetime('now')
            WHERE name=?
        """, (x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh, trigram, name))
        print(f"    {name} [{trigram}]")

    # Apply relation reclassifications
    print()
    print("  APPLYING RELATION FIXES...")
    for n1, n2, old_type, new_type in RELATION_FIXES:
        r = conn.execute("""
            SELECT r.id FROM relations r
            JOIN concepts c1 ON r.concept1_id = c1.id
            JOIN concepts c2 ON r.concept2_id = c2.id
            WHERE ((c1.name=? AND c2.name=?) OR (c1.name=? AND c2.name=?))
            AND r.rel_type=?
        """, (n1, n2, n2, n1, old_type)).fetchone()
        if r:
            conn.execute("UPDATE relations SET rel_type=?, updated_at=datetime('now') WHERE id=?",
                         (new_type, r["id"]))
            print(f"    {n1}↔{n2}: {old_type} → {new_type}")
        else:
            print(f"    WARNING: {n1}↔{n2} ({old_type}) not found!")

    # Recalculate ALL relation angles for affected concepts
    print()
    print("  RECALCULATING RELATION ANGLES...")
    affected_names = [u[0] for u in updates]
    for name in affected_names:
        concept = conn.execute(
            "SELECT id FROM concepts WHERE name=?", (name,)
        ).fetchone()
        if not concept:
            continue

        rels = conn.execute("""
            SELECT r.id,
                   c1.x as x1, c1.y as y1, c1.z as z1, c1.e as e1, c1.f as f1, c1.g as g1, c1.h as h1,
                   c2.x as x2, c2.y as y2, c2.z as z2, c2.e as e2, c2.f as f2, c2.g as g2, c2.h as h2
            FROM relations r
            JOIN concepts c1 ON r.concept1_id = c1.id
            JOIN concepts c2 ON r.concept2_id = c2.id
            WHERE r.concept1_id=? OR r.concept2_id=?
        """, (concept["id"], concept["id"])).fetchall()

        for rel in rels:
            v1_3d = [rel["x1"], rel["y1"], rel["z1"]]
            v2_3d = [rel["x2"], rel["y2"], rel["z2"]]
            v1_7d = [rel["x1"], rel["y1"], rel["z1"], rel["e1"], rel["f1"], rel["g1"], rel["h1"]]
            v2_7d = [rel["x2"], rel["y2"], rel["z2"], rel["e2"], rel["f2"], rel["g2"], rel["h2"]]

            conn.execute(
                "UPDATE relations SET angle_4d=?, angle_8d=?, updated_at=datetime('now') WHERE id=?",
                (round(angle_3d(v1_3d, v2_3d), 4), round(angle_7d(v1_7d, v2_7d), 4), rel["id"])
            )

        if rels:
            print(f"    {name}: {len(rels)} relations")

    conn.commit()
    conn.close()

    print()
    print(f"  DONE. {len(updates)} vectors + {len(RELATION_FIXES)} relations corrected.")
    print("  Run: python3 -m tools.validate")


if __name__ == "__main__":
    main()
