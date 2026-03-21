"""
Deep re-encoding pipeline for shallow-vector concepts.

Uses the same semantic interpolation logic as the propose tool to derive
meaningful vectors from each concept's description + the existing dictionary.
Then nudges for 7D proximity and updates the database.

Usage:
    python3 -m tools.deep_reencode --dry-run      # Preview changes
    python3 -m tools.deep_reencode                 # Apply changes
    python3 -m tools.deep_reencode --session 112   # Only re-encode session 112
"""

import sys, os, re, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_connection
from core.octonion import SemanticOctonion
from tools.propose import extract_context_words, guess_level, guess_pos


def load_all_concepts(conn):
    """Load all concepts as dicts keyed by name."""
    rows = conn.execute(
        'SELECT id, name, x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh, '
        'description, level, pos, session, trigram FROM concepts'
    ).fetchall()
    concepts = {}
    for r in rows:
        concepts[r['name']] = dict(r)
    return concepts


def essence_vec(c):
    return np.array([c['x'], c['y'], c['z'], c['e'], c['f'], c['g'], c['h']])


def function_vec(c):
    return np.array([c['fx'], c['fy'], c['fz'], c['fe'], c['ff'], c['fg'], c['fh']])


def compute_trigram(vec7):
    octo = SemanticOctonion(w=1.0, x=vec7[0], y=vec7[1], z=vec7[2],
                            e=vec7[3], f=vec7[4], g=vec7[5], h=vec7[6])
    return octo.to_trigram(x_axis=vec7[0], y_axis=vec7[1]).name


def semantic_interpolate_core(name, description, all_concepts):
    """
    Derive a semantically meaningful CORE vector (x,y,z) from the concept's
    description using weighted interpolation of related concepts.

    Only interpolates the 3D core + function core — preserves domain (e,f,g,h)
    to maintain trigram assignment.
    """
    desc_words = extract_context_words(description) if description else []

    weighted = []
    seen = {name}  # Don't include self

    # Strategy 1: Concepts mentioned in the description (weight 3.0)
    for w in desc_words:
        w_upper = w.upper()
        if w_upper in all_concepts and w_upper not in seen:
            weighted.append((all_concepts[w_upper], 3.0))
            seen.add(w_upper)

    # Strategy 2: Concepts whose descriptions mention this word (weight 1.5)
    word_lower = name.lower().replace('_', ' ')
    for cname, c in all_concepts.items():
        if cname in seen:
            continue
        if c['description'] and f' {word_lower} ' in f' {c["description"].lower()} ':
            weighted.append((c, 1.5))
            seen.add(cname)
            if len(weighted) > 25:
                break

    # Strategy 3: Name substring matches (weight 1.0)
    for cname, c in all_concepts.items():
        if cname in seen:
            continue
        if name in cname or cname in name:
            weighted.append((c, 1.0))
            seen.add(cname)

    if not weighted:
        return None, None, 0

    # Weighted interpolation — CORE ONLY (x, y, z)
    total_weight = sum(w for _, w in weighted)
    core = np.zeros(3)
    func_core = np.zeros(3)
    for c, w in weighted:
        core += w * np.array([c['x'], c['y'], c['z']])
        func_core += w * np.array([c['fx'], c['fy'], c['fz']])
    core /= total_weight
    func_core /= total_weight

    return core, func_core, len(weighted)


def min_angle_7d_excluding(vec7, ex_mat, ex_normed, ex_names, exclude_idx):
    """Min angle to all existing concepts except one (the concept itself)."""
    n = np.linalg.norm(vec7)
    if n < 1e-10:
        return 0, ""
    v_normed = vec7 / n
    cos_vals = np.clip(ex_normed @ v_normed, -1, 1)
    angles = np.degrees(np.arccos(cos_vals))
    # Exclude self
    angles[exclude_idx] = 999.0
    idx = np.argmin(angles)
    return angles[idx], ex_names[idx]


def nudge_7d(vec7, func7, ex_mat, ex_normed, ex_names, exclude_idx,
             max_trials=2000, lock_domain=False, preserve_trigram=None):
    """Nudge vector to clear 15° from all neighbors (excluding self).

    If lock_domain=True, only perturbs core (x,y,z), keeping domain (e,f,g,h) fixed.
    If preserve_trigram is set, rejects candidates that would change trigram.
    """
    best_vec = vec7.copy()
    best_func = func7.copy()
    best_min = 0

    ma, _ = min_angle_7d_excluding(vec7, ex_mat, ex_normed, ex_names, exclude_idx)
    if ma >= 15:
        return vec7, func7, ma

    for trial in range(max_trials):
        candidate = vec7.copy()
        candidate[:3] += np.random.randn(3) * 0.10
        if not lock_domain:
            candidate[3:] += np.random.randn(4) * 0.08
            candidate[3:] = np.clip(candidate[3:], 0.05, 0.95)

        # Check trigram preservation
        if preserve_trigram:
            try:
                new_tri = compute_trigram(candidate)
                if new_tri != preserve_trigram:
                    continue
            except Exception:
                continue

        ma2, _ = min_angle_7d_excluding(candidate, ex_mat, ex_normed, ex_names, exclude_idx)
        if ma2 > best_min:
            best_min = ma2
            best_vec = candidate.copy()
            # Scale function core proportionally
            best_func = func7.copy()
            if np.linalg.norm(vec7[:3]) > 1e-10:
                ratio_core = candidate[:3] / (vec7[:3] + 1e-10)
                best_func[:3] = func7[:3] * np.where(np.abs(ratio_core) < 5, ratio_core, 1.0)
            if best_min >= 16:
                break

    return best_vec, best_func, best_min


def main():
    dry_run = '--dry-run' in sys.argv
    target_session = None
    for i, arg in enumerate(sys.argv):
        if arg == '--session' and i + 1 < len(sys.argv):
            target_session = int(sys.argv[i + 1])

    conn = get_connection()
    all_concepts = load_all_concepts(conn)

    # Identify concepts to re-encode
    if target_session:
        targets = {n: c for n, c in all_concepts.items() if c['session'] == target_session}
    else:
        # Default: session 112 (expansion concepts)
        targets = {n: c for n, c in all_concepts.items() if c['session'] == 112}

    print(f"Re-encoding {len(targets)} concepts (session={target_session or 112})")
    print(f"Total dictionary: {len(all_concepts)} concepts")
    if dry_run:
        print("[DRY RUN — no changes will be applied]")

    # Build matrix for proximity checks
    all_names = list(all_concepts.keys())
    all_mat = np.array([essence_vec(all_concepts[n]) for n in all_names])
    all_norms = np.linalg.norm(all_mat, axis=1, keepdims=True)
    all_normed = all_mat / np.where(all_norms > 1e-10, all_norms, 1)

    # Index lookup
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    stats = {'total': 0, 'reencoded': 0, 'no_context': 0, 'kept': 0, 'nudged': 0}
    trigram_changes = 0

    np.random.seed(42)

    for name, concept in sorted(targets.items()):
        stats['total'] += 1
        desc = concept['description'] or ''
        old_vec = essence_vec(concept)
        old_trigram = concept['trigram']
        idx = name_to_idx[name]

        # Compute semantically-derived CORE (x,y,z) only — preserve domain
        new_core, new_func_core, n_context = semantic_interpolate_core(
            name, desc, all_concepts
        )

        if new_core is None:
            stats['no_context'] += 1
            continue

        # Preserve yang/yin polarity to maintain trigram
        # Trigram yang: x > 0.2 or (|x| <= 0.2 and y > 0)
        old_x, old_y = concept['x'], concept['y']
        old_is_yang = old_x > 0.2 or (abs(old_x) <= 0.2 and old_y > 0)
        new_is_yang = new_core[0] > 0.2 or (abs(new_core[0]) <= 0.2 and new_core[1] > 0)

        if old_is_yang != new_is_yang:
            # Flip x to preserve polarity
            new_core[0] = -new_core[0]
            new_func_core[0] = -new_func_core[0]
            # If still wrong (near zero x), adjust more
            check_yang = new_core[0] > 0.2 or (abs(new_core[0]) <= 0.2 and new_core[1] > 0)
            if check_yang != old_is_yang:
                # Force x to clear the threshold
                new_core[0] = 0.3 if old_is_yang else -0.3
                new_func_core[0] = 0.24 if old_is_yang else -0.24

        # Build full 7D vector: new core + preserved domain
        new_essence = np.array([
            new_core[0], new_core[1], new_core[2],
            concept['e'], concept['f'], concept['g'], concept['h']
        ])
        new_func = np.array([
            new_func_core[0], new_func_core[1], new_func_core[2],
            concept['fe'], concept['ff'], concept['fg'], concept['fh']
        ])

        # Final trigram check — if still wrong after polarity fix, force x harder
        check_trigram = compute_trigram(new_essence)
        if check_trigram != old_trigram:
            # Force x sign more aggressively
            if old_is_yang:
                new_essence[0] = abs(new_essence[0]) + 0.25
            else:
                new_essence[0] = -(abs(new_essence[0]) + 0.25)
            new_func[0] = new_essence[0] * 0.8

        # Check proximity
        ma, closest = min_angle_7d_excluding(new_essence, all_mat, all_normed, all_names, idx)

        if ma < 15:
            new_essence, new_func, ma = nudge_7d(
                new_essence, new_func, all_mat, all_normed, all_names, idx,
                lock_domain=True, preserve_trigram=old_trigram
            )
            if ma >= 15:
                stats['nudged'] += 1
            else:
                stats['kept'] += 1
                continue  # Can't find valid position, keep original

        # Compute new trigram (should be same since domain preserved)
        new_trigram = compute_trigram(new_essence)

        # Calculate how much the core vector changed
        old_core_3d = np.array([concept['x'], concept['y'], concept['z']])
        new_core_3d = new_essence[:3]
        old_n = np.linalg.norm(old_core_3d)
        new_n = np.linalg.norm(new_core_3d)
        angle_change = 0
        if old_n > 1e-10 and new_n > 1e-10:
            cos = np.clip(np.dot(old_core_3d, new_core_3d) / (old_n * new_n), -1, 1)
            angle_change = math.degrees(math.acos(cos))

        if new_trigram != old_trigram:
            trigram_changes += 1

        stats['reencoded'] += 1

        if stats['reencoded'] <= 20 or stats['reencoded'] % 50 == 0:
            print(f"  {name:25s}: {n_context:2d} ctx, Δ={angle_change:5.1f}°, "
                  f"min={ma:5.1f}°, {old_trigram}→{new_trigram}")

        if not dry_run:
            # Update database — core + function core, preserve domain + function domain
            conn.execute("""
                UPDATE concepts SET
                    x=?, y=?, z=?,
                    fx=?, fy=?, fz=?,
                    trigram=?
                WHERE id=?
            """, (
                round(float(new_essence[0]), 4), round(float(new_essence[1]), 4),
                round(float(new_essence[2]), 4),
                round(float(new_func[0]), 4), round(float(new_func[1]), 4),
                round(float(new_func[2]), 4),
                new_trigram,
                concept['id']
            ))

            # Update the matrix for subsequent proximity checks
            all_mat[idx] = new_essence
            n = np.linalg.norm(new_essence)
            all_normed[idx] = new_essence / n if n > 1e-10 else new_essence

    if not dry_run:
        conn.commit()

        # Recompute relation angles for affected concepts
        print("\nRecomputing relation angles...")
        for name in targets:
            cid = all_concepts[name]['id']
            rels = conn.execute(
                'SELECT id, concept1_id, concept2_id FROM relations '
                'WHERE concept1_id=? OR concept2_id=?', (cid, cid)
            ).fetchall()
            for rel in rels:
                c1 = conn.execute('SELECT x,y,z,e,f,g,h FROM concepts WHERE id=?',
                                  (rel['concept1_id'],)).fetchone()
                c2 = conn.execute('SELECT x,y,z,e,f,g,h FROM concepts WHERE id=?',
                                  (rel['concept2_id'],)).fetchone()
                if c1 and c2:
                    v1_3 = np.array([c1['x'], c1['y'], c1['z']])
                    v2_3 = np.array([c2['x'], c2['y'], c2['z']])
                    m1, m2 = np.linalg.norm(v1_3), np.linalg.norm(v2_3)
                    if m1 > 1e-10 and m2 > 1e-10:
                        cos_t = np.clip(np.dot(v1_3, v2_3) / (m1 * m2), -1, 1)
                        a4d = round(math.degrees(math.acos(cos_t)), 4)
                    else:
                        a4d = 0.0
                    v1_7 = np.array([c1['x'],c1['y'],c1['z'],c1['e'],c1['f'],c1['g'],c1['h']])
                    v2_7 = np.array([c2['x'],c2['y'],c2['z'],c2['e'],c2['f'],c2['g'],c2['h']])
                    m1f, m2f = np.linalg.norm(v1_7), np.linalg.norm(v2_7)
                    if m1f > 1e-10 and m2f > 1e-10:
                        cos_tf = np.clip(np.dot(v1_7, v2_7) / (m1f * m2f), -1, 1)
                        a8d = round(math.degrees(math.acos(cos_tf)), 4)
                    else:
                        a8d = 0.0
                    conn.execute('UPDATE relations SET angle_4d=?, angle_8d=? WHERE id=?',
                                 (a4d, a8d, rel['id']))
        conn.commit()

    conn.close()

    print(f"\n{'='*60}")
    print(f"  DEEP RE-ENCODE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total targets:    {stats['total']}")
    print(f"  Re-encoded:       {stats['reencoded']}")
    print(f"  Nudged:           {stats['nudged']}")
    print(f"  No context:       {stats['no_context']}")
    print(f"  Kept (unfixable): {stats['kept']}")
    print(f"  Trigram changes:  {trigram_changes}")
    print(f"  {'[DRY RUN]' if dry_run else '[APPLIED]'}")


if __name__ == '__main__':
    main()
