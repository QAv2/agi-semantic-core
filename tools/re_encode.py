"""Re-encode legacy vectors to fix 3D core synonym angle violations.

Session 106 — Legacy vector correction.
Moves the 3D core [x,y,z] of unconstrained Session 0 concepts toward
their synonym partners. Preserves domain [e,f,g,h] (operational character).

The validation tool checks angle_4d (3D core angle) for synonym < 30°.
Many legacy concepts have similar domains but divergent cores.

Usage:
    python -m tools.re_encode --dry-run     # Preview all corrections
    python -m tools.re_encode               # Apply corrections
"""

import sqlite3
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.octonion import SemanticOctonion


def angle_3d(v1, v2):
    """Angle between two 3D vectors in degrees."""
    v1, v2 = np.asarray(v1[:3], dtype=float), np.asarray(v2[:3], dtype=float)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-10 or m2 < 1e-10:
        return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def angle_7d(v1, v2):
    """Angle between two 7D vectors in degrees."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-10 or m2 < 1e-10:
        return 0.0
    cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def compute_trigram(vec7):
    """Compute trigram name for a 7D vector."""
    octo = SemanticOctonion(w=1.0, x=vec7[0], y=vec7[1], z=vec7[2],
                            e=vec7[3], f=vec7[4], g=vec7[5], h=vec7[6])
    return octo.to_trigram(x_axis=vec7[0], y_axis=vec7[1]).name


def find_optimal_core(target_name, syn_cores, comp_cores, all_cores, all_names,
                      related_set, original_core):
    """Find optimal 3D core [x,y,z] satisfying synonym < 25° and complement 80-105°.

    Strategy: search near centroid of synonym partners' cores.
    """
    if len(syn_cores) == 0:
        return None, float('inf')

    syn_arr = np.array([c[:3] for c in syn_cores])
    comp_arr = np.array([c[:3] for c in comp_cores]) if comp_cores else np.empty((0, 3))

    # Centroid of synonym partner cores
    centroid = np.mean(syn_arr, axis=0)

    # Precompute for crowding checks
    all_core_arr = np.array([c[:3] for c in all_cores.values()])
    all_core_names = list(all_cores.keys())
    norms = np.linalg.norm(all_core_arr, axis=1)

    unrelated_mask = np.array([n not in related_set for n in all_core_names])
    valid_mask = norms > 1e-10
    check_mask = unrelated_mask & valid_mask

    best_core = centroid.copy()
    best_score = float('inf')

    for trial in range(80):
        if trial == 0:
            v = centroid.copy()
        elif trial < 20:
            v = centroid + np.random.randn(3) * 0.05
        elif trial < 50:
            v = centroid + np.random.randn(3) * 0.12
        else:
            # Try along individual synonym partner directions
            idx = trial % len(syn_arr)
            v = syn_arr[idx] + np.random.randn(3) * 0.08

        # Gradient descent refinement
        for step in range(50):
            mv = np.linalg.norm(v)
            if mv < 1e-10:
                v = centroid + np.random.randn(3) * 0.05
                continue

            grad = np.zeros(3)

            # Synonym: want <25° (soft target 10-20°)
            for sv in syn_arr:
                ang = angle_3d(v, sv)
                if ang > 20:
                    direction = sv / np.linalg.norm(sv) - v / mv
                    grad += direction * (ang - 20) * 0.4
                elif ang < 5:
                    direction = v / mv - sv / np.linalg.norm(sv)
                    grad += direction * (5 - ang) * 0.2

            # Complement: want 80-105°
            for cv in comp_arr:
                ang = angle_3d(v, cv)
                if ang < 80:
                    direction = v / mv - cv / np.linalg.norm(cv)
                    grad += direction * (80 - ang) * 0.8
                elif ang > 105:
                    direction = cv / np.linalg.norm(cv) - v / mv
                    grad += direction * (ang - 105) * 0.8

            # Crowding: want >5° from unrelated
            if check_mask.any():
                dots = all_core_arr[check_mask] @ v
                dn = norms[check_mask] * mv
                safe_dn = np.where(dn > 1e-10, dn, 1.0)
                cos_a = np.clip(dots / safe_dn, -1.0, 1.0)
                angles = np.degrees(np.arccos(cos_a))
                crowded = np.where(angles < 5.0)[0]
                for ci in crowded:
                    oi = np.where(check_mask)[0][ci]
                    direction = v / mv - all_core_arr[oi] / norms[oi]
                    grad += direction * (5.0 - angles[ci]) * 0.3

            if np.linalg.norm(grad) < 1e-6:
                break

            step_size = 0.08 / (1 + step * 0.2)
            v = v + grad * step_size

        # Score
        score = 0.0
        for sv in syn_arr:
            ang = angle_3d(v, sv)
            if ang > 30:
                score += (ang - 30) ** 2 * 100
            elif ang > 20:
                score += (ang - 20) ** 2 * 2

        for cv in comp_arr:
            ang = angle_3d(v, cv)
            if ang < 80:
                score += (80 - ang) ** 2 * 50
            elif ang > 105:
                score += (ang - 105) ** 2 * 50

        # Crowding
        mv = np.linalg.norm(v)
        if mv > 1e-10 and check_mask.any():
            dots = all_core_arr[check_mask] @ v
            dn = norms[check_mask] * mv
            safe_dn = np.where(dn > 1e-10, dn, 1.0)
            cos_a = np.clip(dots / safe_dn, -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_a))
            n_crowded = int(np.sum(angles < 3.0))
            score += n_crowded * 30

        # Prefer keeping similar magnitude to original
        mag_diff = abs(np.linalg.norm(v) - np.linalg.norm(original_core))
        score += mag_diff * 2

        if score < best_score:
            best_score = score
            best_core = v.copy()

    return np.round(best_core, 2), best_score


# Processing order — later entries see earlier corrections
RE_ENCODE_ORDER = [
    # Wave 1: Single-partner (move toward anchored s100/s104)
    "FREE",          # → FREEDOM (3 comps)
    "THIN",          # → NARROW (3 comps)
    "WORLD",         # → EARTH (4 comps)
    "REQUIREMENT",   # → NEED (1 comp)
    "ARRANGEMENT",   # → ORDER (3 comps)
    "FINE",          # → SUBTLE (2 comps)
    "SHAPE",         # → FORM
    "DENSE",         # → THICK
    "DOMAIN",        # → AREA (s104)
    "CONDITION",     # → STATE (s104)
    "LEVEL",         # → DEGREE

    # Wave 2: Multi-partner
    "FAITH",         # → TRUST (move FAITH toward TRUST to shrink gap for BELIEF)
    "BELIEF",        # → TRUST (1 comp), FAITH (moved, 1 comp)
    "NOTHING",       # → VOID (1 comp), ABSENCE
    "DESIRE",        # → WANT (2 comps)
    "WISH",          # → DESIRE (after move)
    "MATTER",        # → MATERIAL, SUBSTANCE

    # Wave 3: Core cluster — move CENTER toward ESSENCE first, then converge
    "CHARACTER",     # → NATURE
    "CENTER",        # → ESSENCE (no comps, free to move — bring it closer)
    "CORE",          # → CENTER (moved), ESSENCE
    "ESSENCE",       # → CORE (moved), NATURE
    "NATURE",        # → ESSENCE (moved), CHARACTER (moved)

    # Wave 4: Stability cluster
    "STEADY",        # → STABLE (3 comps), FIRM
    "FIRM",          # → SOLID (2 comps), STEADY (moved)

    # Wave 5: Purpose cluster — move REASON toward INTENTION first, then converge
    "REASON",        # → INTENTION (bring them closer first)
    "INTENTION",     # → REASON (moved), PURPOSE
    "PURPOSE",       # → REASON (moved), INTENTION (moved)
]

SYN_TARGETS = {
    "FREE": ["FREEDOM"],
    "THIN": ["NARROW"],
    "WORLD": ["EARTH"],
    "REQUIREMENT": ["NEED"],
    "ARRANGEMENT": ["ORDER"],
    "FINE": ["SUBTLE"],
    "SHAPE": ["FORM"],
    "DENSE": ["THICK"],
    "DOMAIN": ["AREA"],
    "CONDITION": ["STATE"],
    "LEVEL": ["DEGREE"],
    "FAITH": ["TRUST"],             # Bring FAITH closer to TRUST
    "BELIEF": ["TRUST", "FAITH"],
    "NOTHING": ["VOID", "ABSENCE"],
    "DESIRE": ["WANT"],
    "WISH": ["DESIRE"],
    "MATTER": ["MATERIAL", "SUBSTANCE"],
    "CHARACTER": ["NATURE"],
    "CENTER": ["ESSENCE"],          # Move CENTER toward ESSENCE first
    "CORE": ["CENTER", "ESSENCE"],
    "ESSENCE": ["CORE", "NATURE"],
    "NATURE": ["ESSENCE", "CHARACTER"],
    "STEADY": ["STABLE", "FIRM"],
    "FIRM": ["SOLID", "STEADY"],
    "REASON": ["INTENTION"],         # Move REASON toward INTENTION first
    "INTENTION": ["REASON", "PURPOSE"],
    "PURPOSE": ["REASON", "INTENTION"],
}


def main():
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    np.random.seed(42)

    conn = sqlite3.connect("db/semantic.db")
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("  LEGACY CORE RE-ENCODING — Session 106")
    print("  Mode:", "DRY RUN" if dry_run else "LIVE")
    print("  Target: fix 3D core angles for synonym pairs (angle_4d < 30°)")
    print("=" * 70)
    print()

    # Load all concept vectors: name -> [x,y,z,e,f,g,h]
    all_rows = conn.execute(
        "SELECT name, x, y, z, e, f, g, h FROM concepts"
    ).fetchall()
    vec_map = {}
    for r in all_rows:
        vec_map[r[0]] = np.array([r[1], r[2], r[3], r[4], r[5], r[6], r[7]])

    # Load all relations: name -> [(partner, type, angle_4d)]
    rel_rows = conn.execute('''
        SELECT c1.name, c2.name, r.rel_type, r.angle_4d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
    ''').fetchall()
    rel_map = {}
    for r in rel_rows:
        rel_map.setdefault(r[0], []).append((r[1], r[2], r[3]))
        rel_map.setdefault(r[1], []).append((r[0], r[2], r[3]))

    updates = []
    resolved = 0
    partial = 0

    for concept_name in RE_ENCODE_ORDER:
        if concept_name not in vec_map:
            print(f"  SKIP: {concept_name} not found")
            continue

        cur_vec = vec_map[concept_name]
        partner_names = SYN_TARGETS.get(concept_name, [])

        # Check if still needs fix (3D core angle)
        needs_fix = False
        for pn in partner_names:
            if pn in vec_map:
                a3 = angle_3d(cur_vec, vec_map[pn])
                if a3 > 30:
                    needs_fix = True
                    break

        if not needs_fix:
            continue

        # Gather synonym partner full vectors (for core extraction)
        syn_vecs = [vec_map[pn] for pn in partner_names if pn in vec_map]

        # Gather complement partner full vectors
        comp_vecs = []
        rels = rel_map.get(concept_name, [])
        related_set = {concept_name} | {r[0] for r in rels}
        for partner, rtype, _ in rels:
            if rtype == 'complement' and partner in vec_map:
                comp_vecs.append(vec_map[partner])

        # Optimize 3D core only
        new_core, cost = find_optimal_core(
            concept_name, syn_vecs, comp_vecs,
            vec_map, list(vec_map.keys()), related_set,
            cur_vec[:3]
        )

        if new_core is None:
            print(f"  {concept_name}: FAILED")
            continue

        # Build new full vector: new core + original domain
        new_vec = np.concatenate([new_core, cur_vec[3:7]])
        new_trigram = compute_trigram(new_vec)
        old_trigram = compute_trigram(cur_vec)

        # Evaluate
        all_ok = True
        trigram_note = f" (was {old_trigram})" if new_trigram != old_trigram else ""
        print(f"  {concept_name} [{new_core[0]:+.2f},{new_core[1]:+.2f},{new_core[2]:+.2f}] → {new_trigram}{trigram_note}")

        for pn in partner_names:
            if pn not in vec_map:
                continue
            old_a = angle_3d(cur_vec, vec_map[pn])
            new_a = angle_3d(new_vec, vec_map[pn])
            ok = new_a <= 30
            if not ok:
                all_ok = False
            print(f"    syn  {pn:20s}: {old_a:5.1f}° → {new_a:5.1f}° {'OK' if ok else 'MISS'}")

        for partner, rtype, old_a in rels:
            if rtype == 'complement' and partner in vec_map:
                new_a = angle_3d(new_vec, vec_map[partner])
                ok = 80 <= new_a <= 105
                if not ok:
                    all_ok = False
                print(f"    comp {partner:20s}: {old_a:5.1f}° → {new_a:5.1f}° {'OK' if ok else 'MISS'}")

        if all_ok:
            resolved += 1
        else:
            partial += 1

        # Update vec_map for downstream
        vec_map[concept_name] = new_vec
        updates.append((concept_name, new_vec, new_trigram))
        print()

    # Second pass: retry any concepts that had MISS results
    if partial > 0:
        print("  --- REFINEMENT PASS (retry partial fixes) ---")
        print()
        retry_names = []
        for concept_name, new_vec, _ in updates:
            partner_names = SYN_TARGETS.get(concept_name, [])
            for pn in partner_names:
                if pn in vec_map:
                    if angle_3d(new_vec, vec_map[pn]) > 30:
                        retry_names.append(concept_name)
                        break

        for concept_name in retry_names:
            cur_vec = vec_map[concept_name]
            partner_names = SYN_TARGETS.get(concept_name, [])
            syn_vecs = [vec_map[pn] for pn in partner_names if pn in vec_map]
            comp_vecs = []
            rels = rel_map.get(concept_name, [])
            related_set = {concept_name} | {r[0] for r in rels}
            for partner, rtype, _ in rels:
                if rtype == 'complement' and partner in vec_map:
                    comp_vecs.append(vec_map[partner])

            # Use more trials
            np.random.seed(12345)
            new_core, cost = find_optimal_core(
                concept_name, syn_vecs, comp_vecs,
                vec_map, list(vec_map.keys()), related_set,
                cur_vec[:3]
            )
            if new_core is None:
                continue

            new_vec = np.concatenate([new_core, cur_vec[3:7]])
            new_trigram = compute_trigram(new_vec)

            all_ok = True
            print(f"  {concept_name} (retry) [{new_core[0]:+.2f},{new_core[1]:+.2f},{new_core[2]:+.2f}] → {new_trigram}")
            for pn in partner_names:
                if pn not in vec_map:
                    continue
                new_a = angle_3d(new_vec, vec_map[pn])
                ok = new_a <= 30
                if not ok:
                    all_ok = False
                print(f"    syn  {pn:20s}: {new_a:5.1f}° {'OK' if ok else 'MISS'}")

            for partner, rtype, _ in rels:
                if rtype == 'complement' and partner in vec_map:
                    new_a = angle_3d(new_vec, vec_map[partner])
                    ok = 80 <= new_a <= 105
                    if not ok:
                        all_ok = False
                    print(f"    comp {partner:20s}: {new_a:5.1f}° {'OK' if ok else 'MISS'}")

            # Update in updates list and vec_map
            vec_map[concept_name] = new_vec
            for i, (n, v, t) in enumerate(updates):
                if n == concept_name:
                    updates[i] = (concept_name, new_vec, new_trigram)
                    break
            print()

    # Recount after refinement
    final_issues = 0
    for concept_name, new_vec, _ in updates:
        partner_names = SYN_TARGETS.get(concept_name, [])
        for pn in partner_names:
            if pn in vec_map and angle_3d(vec_map[concept_name], vec_map[pn]) > 30:
                final_issues += 1
                break

    print("=" * 70)
    print(f"  Concepts re-encoded: {len(updates)}")
    print(f"  Remaining issues: {final_issues}")
    print()

    if dry_run:
        print("  DRY RUN complete. No changes applied.")
        conn.close()
        return

    # Apply
    print("  APPLYING CORE CORRECTIONS...")
    for name, vec, trigram in updates:
        x, y, z = vec[:3]
        # Keep original domain
        e, f, g, h = vec[3:7]
        # Update function vector core proportionally, keep function domain
        old_row = conn.execute(
            "SELECT fx, fy, fz, fe, ff, fg, fh, x as ox, y as oy, z as oz FROM concepts WHERE name=?",
            (name,)
        ).fetchone()

        # Scale function core same way as essence core
        old_core_mag = np.linalg.norm([old_row["ox"], old_row["oy"], old_row["oz"]])
        new_core_mag = np.linalg.norm([x, y, z])

        if old_core_mag > 1e-10 and new_core_mag > 1e-10:
            # Rotate function core to match new essence direction
            scale = new_core_mag / old_core_mag
            old_fn_core_mag = np.linalg.norm([old_row["fx"], old_row["fy"], old_row["fz"]])
            if old_fn_core_mag > 1e-10:
                # Point function core in same direction as new essence core, with original magnitude
                fn_dir = np.array([x, y, z]) / new_core_mag
                fx, fy, fz = fn_dir * old_fn_core_mag
            else:
                fx, fy, fz = 0.0, 0.0, 0.0
        else:
            fx = old_row["fx"]
            fy = old_row["fy"]
            fz = old_row["fz"]

        fe, ff, fg, fh = old_row["fe"], old_row["ff"], old_row["fg"], old_row["fh"]

        conn.execute("""
            UPDATE concepts SET
                x=?, y=?, z=?,
                fx=?, fy=?, fz=?,
                trigram=?, session=106, updated_at=datetime('now')
            WHERE name=?
        """, (round(x, 4), round(y, 4), round(z, 4),
              round(fx, 4), round(fy, 4), round(fz, 4),
              trigram, name))
        print(f"    {name} [{trigram}]")

    # Recalculate relation angles
    print()
    print("  RECALCULATING RELATION ANGLES...")
    affected = {u[0] for u in updates}
    recalc = 0
    for name in affected:
        cid = conn.execute("SELECT id FROM concepts WHERE name=?", (name,)).fetchone()
        if not cid:
            continue
        rels = conn.execute("""
            SELECT r.id,
                   c1.x as x1, c1.y as y1, c1.z as z1,
                   c1.e as e1, c1.f as f1, c1.g as g1, c1.h as h1,
                   c2.x as x2, c2.y as y2, c2.z as z2,
                   c2.e as e2, c2.f as f2, c2.g as g2, c2.h as h2
            FROM relations r
            JOIN concepts c1 ON r.concept1_id = c1.id
            JOIN concepts c2 ON r.concept2_id = c2.id
            WHERE r.concept1_id=? OR r.concept2_id=?
        """, (cid["id"], cid["id"])).fetchall()

        for rel in rels:
            a3 = angle_3d(
                [rel["x1"], rel["y1"], rel["z1"]],
                [rel["x2"], rel["y2"], rel["z2"]]
            )
            a7 = angle_7d(
                [rel["x1"], rel["y1"], rel["z1"], rel["e1"], rel["f1"], rel["g1"], rel["h1"]],
                [rel["x2"], rel["y2"], rel["z2"], rel["e2"], rel["f2"], rel["g2"], rel["h2"]]
            )
            conn.execute(
                "UPDATE relations SET angle_4d=?, angle_8d=?, updated_at=datetime('now') WHERE id=?",
                (round(a3, 4), round(a7, 4), rel["id"])
            )
            recalc += 1

    print(f"    {recalc} relation angles recalculated")
    conn.commit()
    conn.close()
    print()
    print(f"  DONE. Run: python3 -m tools.validate")


if __name__ == "__main__":
    main()
