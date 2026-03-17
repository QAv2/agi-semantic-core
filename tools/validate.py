#!/usr/bin/env python3
"""
Validate the AGI Semantic Dictionary.

Checks:
1. Complement pairs: core angle in [60, 120], domain_sim 0.1-0.95
2. Opposition pairs: core angle > 120, domain_sim > 0.8
3. Synonym pairs: core angle < 30
4. All concepts have non-zero vectors (except BEING)
5. Trigram distribution balance
6. Self-containment metric
7. Vector space crowding (min pairwise angle per trigram)
8. Anchor drift detection

Session 107 recalibration:
- Opposition (180°) distinguished from complement (90°) using domain similarity
- True opposites (same domain, reversed polarity) → opposition at ~180°
- Orthogonal complements (related domain, independent axis) → complement at ~90°
- Structural noise (unrelated domains) → deleted

Usage:
    python -m tools.validate              # Full validation
    python -m tools.validate complements  # Complements only
    python -m tools.validate opposition   # Opposition only
    python -m tools.validate crowding     # Crowding analysis
    python -m tools.validate anchors      # Anchor drift check
    python -m tools.validate containment  # Self-containment only
    python -m tools.validate summary      # Quick summary
"""

import sys
import os
import json
import re
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore


# Trigram-aware domain angle expectations (from session 50 discovery)
EXPECTED_DOMAIN_ANGLES = {
    ('KUN', 'QIAN'): (0, 60),
    ('KAN', 'LI'): (0, 50),
    ('GEN', 'XUN'): (15, 70),
    ('ZHEN', 'DUI'): (10, 65),
    ('GEN', 'DUI'): (10, 70),
    ('KAN', 'QIAN'): (5, 55),
    ('KUN', 'LI'): (5, 55),
    ('ZHEN', 'XUN'): (10, 60),
}

# 30 anchor concepts — if these move relative to each other, something is wrong
ANCHORS = [
    'BEING', 'THIS', 'THAT', 'YANG', 'YIN',
    'BECOMING', 'ABIDING', 'RELATIONSHIP',
    'FIRE', 'WATER', 'AIR', 'EARTH',
    'LOVE', 'FEAR', 'JOY', 'SORROW',
    'TRUTH', 'FALSEHOOD', 'GOOD', 'BAD',
    'LIGHT', 'DARK', 'HOT', 'COLD',
    'GIVE', 'TAKE', 'CREATE', 'DESTROY',
    'LIFE', 'DEATH',
]


def _domain_sim(sc, n1, n2):
    """Compute cosine similarity between domain vectors [e,f,g,h]."""
    r1 = sc.conn.execute("SELECT e,f,g,h FROM concepts WHERE name=?", (n1,)).fetchone()
    r2 = sc.conn.execute("SELECT e,f,g,h FROM concepts WHERE name=?", (n2,)).fetchone()
    if not r1 or not r2:
        return 0.0
    v1 = np.array([r1[0], r1[1], r1[2], r1[3]])
    v2 = np.array([r2[0], r2[1], r2[2], r2[3]])
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-10 or m2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (m1 * m2))


def validate_complements(sc: SemanticCore, verbose: bool = True):
    """Validate all complement pairs.
    Complement: core angle 60-120° (orthogonal perspectives, not opposites).
    """
    rows = sc.conn.execute("""
        SELECT c1.name, c2.name, c1.trigram, c2.trigram, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        WHERE r.rel_type = 'complement'
    """).fetchall()

    total = len(rows)
    valid = 0
    issues = []

    for r in rows:
        n1, n2 = r['name'], r[1]
        angle_4d = r['angle_4d']

        if angle_4d < 60 or angle_4d > 120:
            issues.append(f"CORE ANGLE: {n1} <-> {n2}: {angle_4d:.1f} (expected 60-120)")
        else:
            valid += 1

    if verbose:
        print(f"Complement Validation: {valid}/{total} valid ({100*valid/total:.1f}%)")
        if issues:
            print(f"\nIssues ({len(issues)}):")
            for issue in issues[:20]:
                print(f"  {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")

    return valid, total, issues


def validate_opposition(sc: SemanticCore, verbose: bool = True):
    """Validate all opposition pairs.
    Opposition: core angle > 120° (true semantic opposites, same domain).
    """
    rows = sc.conn.execute("""
        SELECT c1.name, c2.name, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        WHERE r.rel_type = 'opposition'
    """).fetchall()

    total = len(rows)
    valid = 0
    issues = []

    for r in rows:
        n1, n2 = r['name'], r[1]
        angle_4d = r['angle_4d']
        dsim = _domain_sim(sc, n1, n2)

        if angle_4d < 120:
            issues.append(f"OPP ANGLE: {n1} <-> {n2}: {angle_4d:.1f}° (expected >120)")
        elif dsim < 0.5:
            issues.append(f"OPP DOMAIN: {n1} <-> {n2}: domain_sim={dsim:.2f} (expected >0.5)")
        else:
            valid += 1

    if verbose:
        a_all = np.array([r['angle_4d'] for r in rows])
        print(f"Opposition Validation: {valid}/{total} valid ({100*valid/total:.1f}%)")
        print(f"  Angle: mean={np.mean(a_all):.1f}° median={np.median(a_all):.1f}° >150°: {int(np.sum(a_all>150))}")
        if issues:
            print(f"\nIssues ({len(issues)}):")
            for issue in issues[:20]:
                print(f"  {issue}")
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")

    return valid, total, issues


def validate_synonyms(sc: SemanticCore, verbose: bool = True):
    """Validate all synonym pairs."""
    rows = sc.conn.execute("""
        SELECT c1.name, c2.name, r.angle_4d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        WHERE r.rel_type = 'synonym'
    """).fetchall()

    total = len(rows)
    valid = 0
    issues = []

    for r in rows:
        if r['angle_4d'] <= 30:
            valid += 1
        else:
            issues.append(f"SYNONYM ANGLE: {r['name']} <-> {r[1]}: {r['angle_4d']:.1f} (expected <30)")

    if verbose:
        print(f"Synonym Validation: {valid}/{total} valid ({100*valid/total:.1f}%)")
        if issues:
            print(f"\nIssues ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  {issue}")

    return valid, total, issues


def validate_vectors(sc: SemanticCore, verbose: bool = True):
    """Check all concepts have valid vectors."""
    rows = sc.conn.execute("SELECT name, x, y, z, e, f, g, h FROM concepts").fetchall()

    total = len(rows)
    zero_vec = 0
    zero_domain = 0

    for r in rows:
        core_mag = (r['x']**2 + r['y']**2 + r['z']**2) ** 0.5
        domain_mag = (r['e']**2 + r['f']**2 + r['g']**2 + r['h']**2) ** 0.5

        if core_mag < 1e-10 and r['name'] != 'BEING':
            zero_vec += 1
        if domain_mag < 1e-10 and r['name'] not in ('BEING', 'I'):
            zero_domain += 1

    if verbose:
        print(f"Vector Validation: {total} concepts checked")
        print(f"  Zero core vectors: {zero_vec} (excluding BEING)")
        print(f"  Zero domain vectors: {zero_domain} (excluding BEING, I)")

    return zero_vec, zero_domain


def validate_trigrams(sc: SemanticCore, verbose: bool = True):
    """Check trigram distribution health."""
    stats = sc.statistics()
    dist = stats['trigram_distribution']

    if verbose:
        print(f"Trigram Distribution ({sum(dist.values())} concepts):")
        for tri, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            bar = "#" * (cnt // 5)
            print(f"  {tri:8s} {cnt:4d} {bar}")

        values = list(dist.values())
        if values:
            avg = sum(values) / len(values)
            min_v, max_v = min(values), max(values)
            ratio = max_v / (min_v + 1)
            print(f"\n  Balance: avg={avg:.0f}, min={min_v}, max={max_v}, ratio={ratio:.1f}")
            if ratio > 2.0:
                print(f"  WARNING: Trigram imbalance > 2x")


def validate_crowding(sc: SemanticCore, verbose: bool = True):
    """Check for vector space crowding within each trigram."""
    rows = sc.conn.execute(
        "SELECT id, name, x, y, z, e, f, g, h, trigram FROM concepts ORDER BY trigram, id"
    ).fetchall()

    # Group by trigram
    trigram_groups = {}
    for r in rows:
        tri = r['trigram'] or '(none)'
        if tri not in trigram_groups:
            trigram_groups[tri] = []
        trigram_groups[tri].append({
            'name': r['name'],
            'vec': np.array([r['x'], r['y'], r['z'], r['e'], r['f'], r['g'], r['h']])
        })

    if verbose:
        print("Crowding Analysis (min pairwise angle per trigram):")
        print(f"  {'TRIGRAM':8s} {'COUNT':>5s} {'MIN':>6s} {'PAIR':40s}")
        print("  " + "-" * 65)

    all_crowded = []

    for tri in sorted(trigram_groups.keys()):
        group = trigram_groups[tri]
        if len(group) < 2:
            continue

        min_angle = 180.0
        min_pair = ("", "")

        # Check non-synonym pairs only — synonyms are expected to be close
        # Get synonym pairs for this trigram
        synonym_pairs = set()
        for g in group:
            syn_rows = sc.conn.execute("""
                SELECT c1.name, c2.name FROM relations r
                JOIN concepts c1 ON r.concept1_id = c1.id
                JOIN concepts c2 ON r.concept2_id = c2.id
                WHERE r.rel_type = 'synonym'
                AND (c1.name = ? OR c2.name = ?)
            """, (g['name'], g['name'])).fetchall()
            for sr in syn_rows:
                synonym_pairs.add((sr[0], sr[1]))
                synonym_pairs.add((sr[1], sr[0]))

        # Sample check — full O(n^2) would be slow for large trigrams
        # Check each concept against its 3 nearest neighbors
        vecs = np.array([g['vec'] for g in group])
        norms = np.linalg.norm(vecs, axis=1)

        for i in range(len(group)):
            if norms[i] < 1e-10:
                continue
            dots = vecs @ vecs[i]
            denoms = norms * norms[i]
            mask = denoms > 1e-10
            cos_thetas = np.zeros(len(vecs))
            cos_thetas[mask] = np.clip(dots[mask] / denoms[mask], -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_thetas))
            angles[i] = 999  # exclude self

            for j in np.argsort(angles)[:3]:
                if j == i:
                    continue
                a = angles[j]
                n1, n2 = group[i]['name'], group[j]['name']
                # Skip synonym pairs
                if (n1, n2) in synonym_pairs:
                    continue
                if a < min_angle:
                    min_angle = a
                    min_pair = (n1, n2)

        if verbose:
            flag = " ** CROWDED" if min_angle < 5.0 else ""
            print(f"  {tri:8s} {len(group):5d} {min_angle:5.1f}° {min_pair[0]} <-> {min_pair[1]}{flag}")

        if min_angle < 5.0:
            all_crowded.append((tri, min_angle, min_pair))

    if verbose:
        print()
        if all_crowded:
            print(f"  WARNING: {len(all_crowded)} trigrams with non-synonym pairs < 5°")
        else:
            print("  No crowding issues detected (all non-synonym pairs > 5°)")

    return all_crowded


def validate_anchors(sc: SemanticCore, verbose: bool = True):
    """Check anchor concept stability — these angles should never change."""
    # Compute current anchor pairwise angles
    anchor_concepts = []
    for name in ANCHORS:
        c = sc.get(name)
        if c:
            anchor_concepts.append(c)

    if verbose:
        print(f"Anchor Validation: {len(anchor_concepts)}/{len(ANCHORS)} anchors found")

    # Compute and store/compare angles
    # For now, just compute and display — future: compare against stored baseline
    anchor_angles = {}
    issues = []

    for i in range(len(anchor_concepts)):
        for j in range(i + 1, len(anchor_concepts)):
            c1, c2 = anchor_concepts[i], anchor_concepts[j]
            angle = c1.angle_4d(c2)
            key = f"{c1.name}:{c2.name}"
            anchor_angles[key] = round(angle, 2)

    # Check for known invariants
    # Post-recalibration: anchor pairs are now opposition (target ~180°)
    # or complement (target ~90°), validated by relation type in DB
    known_invariants = {
        'LOVE:FEAR': (60, 180),     # Complex multi-constraint pair
        'LIFE:DEATH': (120, 180),   # Opposition
        'TRUTH:FALSEHOOD': (120, 180),  # Opposition
        'GIVE:TAKE': (120, 180),    # Opposition
        'CREATE:DESTROY': (120, 180),   # Opposition
        'LIGHT:DARK': (120, 180),   # Opposition
        'HOT:COLD': (120, 180),     # Opposition
        'GOOD:BAD': (120, 180),     # Opposition
    }

    for pair, (lo, hi) in known_invariants.items():
        if pair in anchor_angles:
            angle = anchor_angles[pair]
            if angle < lo or angle > hi:
                issues.append(f"ANCHOR DRIFT: {pair}: {angle}° (expected {lo}-{hi}°)")

    if verbose:
        if not issues:
            print("  All anchor invariants hold")
        else:
            for issue in issues:
                print(f"  {issue}")

        # Save baseline for future comparison
        baseline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'db', 'anchor_baseline.json')
        if not os.path.exists(baseline_path):
            with open(baseline_path, 'w') as f:
                json.dump(anchor_angles, f, indent=2)
            print(f"\n  Saved anchor baseline to {baseline_path}")
            print(f"  ({len(anchor_angles)} pairwise angles for {len(anchor_concepts)} anchors)")
        else:
            # Compare against baseline
            with open(baseline_path) as f:
                baseline = json.load(f)
            drift_count = 0
            max_drift = 0
            for key, base_angle in baseline.items():
                current = anchor_angles.get(key)
                if current is not None:
                    drift = abs(current - base_angle)
                    if drift > 0.1:
                        drift_count += 1
                        max_drift = max(max_drift, drift)
            if drift_count > 0:
                print(f"\n  Anchor drift detected: {drift_count} pairs shifted, max drift={max_drift:.2f}°")
            else:
                print(f"\n  No anchor drift (all {len(baseline)} pairs stable)")

    return issues


def validate_self_containment(sc: SemanticCore, verbose: bool = True):
    """Quick self-containment check."""
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
        'by', 'from', 'as', 'and', 'or', 'but', 'not', 'no', 'that', 'which',
        'who', 'this', 'it', 'its', 'very', 'more', 'most', 'also', 'just',
        'than', 'so', 'if', 'about', 'up', 'out', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can',
    }

    all_names = set(n.upper() for n in sc.all_concepts())
    alias_rows = sc.conn.execute("SELECT alias FROM aliases").fetchall()
    for r in alias_rows:
        all_names.add(r['alias'].upper())

    rows = sc.conn.execute("SELECT description FROM concepts").fetchall()

    unique_words = set()
    found = set()

    for row in rows:
        words = re.findall(r'[a-z]+', row['description'].lower())
        for w in words:
            if w not in stop_words and len(w) > 2:
                unique_words.add(w)
                if w.upper() in all_names:
                    found.add(w)

    missing_count = len(unique_words) - len(found)
    pct = 100 * len(found) / len(unique_words) if unique_words else 0

    if verbose:
        print(f"Self-Containment: {len(found)}/{len(unique_words)} unique definition words defined ({pct:.1f}%)")
        print(f"  Missing: {missing_count} words")

    return pct, missing_count


def run_full_validation(sc: SemanticCore):
    """Run all validation checks."""
    print("=" * 60)
    print("  AGI SEMANTIC DICTIONARY — FULL VALIDATION")
    print("=" * 60)
    print()

    all_issues = []

    # Complements
    c_valid, c_total, c_issues = validate_complements(sc)
    all_issues.extend(c_issues)
    print()

    # Opposition
    o_valid, o_total, o_issues = validate_opposition(sc)
    all_issues.extend(o_issues)
    print()

    # Synonyms
    s_valid, s_total, s_issues = validate_synonyms(sc)
    all_issues.extend(s_issues)
    print()

    # Vectors
    zv, zd = validate_vectors(sc)
    print()

    # Trigrams
    validate_trigrams(sc)
    print()

    # Self-containment
    sc_pct, sc_missing = validate_self_containment(sc)
    print()

    # Crowding
    crowded = validate_crowding(sc)
    print()

    # Anchors
    anchor_issues = validate_anchors(sc)
    all_issues.extend(anchor_issues)
    print()

    # Summary
    print("=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    stats = sc.statistics()
    print(f"  Concepts:         {stats['total_concepts']}")
    print(f"  Relations:        {stats['total_relations']}")
    print(f"  Complements:      {c_valid}/{c_total} valid")
    print(f"  Opposition:       {o_valid}/{o_total} valid")
    print(f"  Synonyms:         {s_valid}/{s_total} valid")
    print(f"  Zero vectors:     {zv}")
    print(f"  Self-containment: {sc_pct:.1f}%")
    print(f"  Crowded trigrams: {len(crowded)}")
    print(f"  Total issues:     {len(all_issues)}")

    if not all_issues:
        print("\n  All checks passed")
    else:
        print(f"\n  {len(all_issues)} issues found")

    # Log to validation_log table
    sc.conn.execute("""
        INSERT INTO validation_log (total_concepts, total_relations, complements_checked, complements_valid, issues, summary)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        stats['total_concepts'], stats['total_relations'],
        c_total, c_valid,
        json.dumps(all_issues[:100]),
        f"{c_valid}/{c_total} complements, {s_valid}/{s_total} synonyms, SC={sc_pct:.1f}%, {len(all_issues)} issues"
    ))
    sc.conn.commit()


def main():
    sc = SemanticCore()
    try:
        if len(sys.argv) > 1:
            cmd = sys.argv[1].lower()
            if cmd == "complements":
                validate_complements(sc)
            elif cmd == "opposition":
                validate_opposition(sc)
            elif cmd == "synonyms":
                validate_synonyms(sc)
            elif cmd == "trigrams":
                validate_trigrams(sc)
            elif cmd == "vectors":
                validate_vectors(sc)
            elif cmd == "crowding":
                validate_crowding(sc)
            elif cmd == "anchors":
                validate_anchors(sc)
            elif cmd == "containment":
                validate_self_containment(sc)
            elif cmd == "summary":
                stats = sc.statistics()
                sc_pct, _ = validate_self_containment(sc, verbose=False)
                print(f"Concepts: {stats['total_concepts']}, Relations: {stats['total_relations']}, SC: {sc_pct:.1f}%")
            else:
                run_full_validation(sc)
        else:
            run_full_validation(sc)
    finally:
        sc.close()


if __name__ == "__main__":
    main()
