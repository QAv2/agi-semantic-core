"""
Batch expansion pipeline for AGI Semantic Dictionary.

Usage:
    python3 -m tools.batch_expand concepts.txt [--force] [--dry-run]

Input format (one per line):
    CONCEPT_NAME | description | TARGET_TRIGRAM

TARGET_TRIGRAM is optional (GEN, XUN, KAN, DUI, LI, ZHEN, QIAN, KUN).
If omitted, propose tool picks natural landing.

Examples:
    GLACIER | A slow-moving mass of ice | XUN
    LAUGHTER | Spontaneous vocal expression of joy | DUI
    SOLITUDE | The state of being alone | GEN
"""

import sys, json, math, os
import numpy as np
from db.connection import get_connection
from api.builder import ConceptBuilder
from core.encoding import RelationType

np.random.seed(int.from_bytes(os.urandom(4), 'big'))

# Trigram domain recipes: (dominant_dim, x_sign)
# Domain order: e=Spatial, f=Temporal, g=Relational, h=Personal
TRIGRAM_DOMAINS = {
    "QIAN": {"e": 0.70, "f": 0.35, "g": 0.30, "h": 0.25, "x_sign": +1},
    "KUN":  {"e": 0.70, "f": 0.35, "g": 0.30, "h": 0.25, "x_sign": -1},
    "ZHEN": {"e": 0.30, "f": 0.70, "g": 0.35, "h": 0.25, "x_sign": +1},
    "XUN":  {"e": 0.30, "f": 0.70, "g": 0.35, "h": 0.25, "x_sign": -1},
    "LI":   {"e": 0.30, "f": 0.25, "g": 0.70, "h": 0.35, "x_sign": +1},
    "KAN":  {"e": 0.30, "f": 0.25, "g": 0.70, "h": 0.35, "x_sign": -1},
    "DUI":  {"e": 0.25, "f": 0.30, "g": 0.35, "h": 0.70, "x_sign": +1},
    "GEN":  {"e": 0.25, "f": 0.30, "g": 0.35, "h": 0.70, "x_sign": -1},
}


def load_existing_7d():
    """Load all existing concept vectors as numpy arrays."""
    conn = get_connection()
    rows = conn.execute('SELECT name, x, y, z, e, f, g, h FROM concepts').fetchall()
    conn.close()
    names = [r[0] for r in rows]
    mat = np.array([[r[1],r[2],r[3],r[4],r[5],r[6],r[7]] for r in rows])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.where(norms > 1e-10, norms, 1)
    return names, mat, normed


def min_angle_7d(vec7, ex_normed, ex_names):
    """Vectorized minimum angle to all existing concepts."""
    n = np.linalg.norm(vec7)
    if n < 1e-10: return 0, ""
    v_normed = vec7 / n
    cos_vals = np.clip(ex_normed @ v_normed, -1, 1)
    angles = np.degrees(np.arccos(cos_vals))
    idx = np.argmin(angles)
    return angles[idx], ex_names[idx]


def generate_vector(name, description, target_trigram, ex_names, ex_mat, ex_normed):
    """Generate a 7D vector for a concept, targeting a specific trigram."""
    recipe = TRIGRAM_DOMAINS.get(target_trigram, TRIGRAM_DOMAINS["QIAN"])
    x_sign = recipe["x_sign"]

    # Generate core vector
    # x: controlled by trigram (yang/yin)
    x = x_sign * (0.25 + np.random.rand() * 0.35)
    # y: random with slight bias toward becoming (+) for active concepts
    y = (np.random.rand() - 0.45) * 0.8
    # z: random with slight upward bias
    z = (np.random.rand() - 0.3) * 0.7

    # Domain values from recipe with jitter
    e = np.clip(recipe["e"] + np.random.randn() * 0.12, 0.05, 0.95)
    f = np.clip(recipe["f"] + np.random.randn() * 0.12, 0.05, 0.95)
    g = np.clip(recipe["g"] + np.random.randn() * 0.12, 0.05, 0.95)
    h = np.clip(recipe["h"] + np.random.randn() * 0.12, 0.05, 0.95)

    vec7 = np.array([x, y, z, e, f, g, h])

    # Nudge to clear 15° from all existing
    ma, closest = min_angle_7d(vec7, ex_normed, ex_names)
    if ma < 15:
        best_vec, best_min = vec7.copy(), ma
        for trial in range(3000):
            perturb = np.zeros(7)
            perturb[:3] = np.random.randn(3) * 0.12
            perturb[3:] = np.random.randn(4) * 0.10
            candidate = vec7 + perturb
            candidate[3:] = np.clip(candidate[3:], 0.05, 0.95)
            # Preserve x sign for trigram
            if x_sign > 0 and candidate[0] < 0.15:
                continue
            if x_sign < 0 and candidate[0] > -0.15:
                continue

            ma2, _ = min_angle_7d(candidate, ex_normed, ex_names)
            if ma2 > best_min:
                best_min = ma2
                best_vec = candidate.copy()
                if best_min >= 18:
                    break

        vec7 = best_vec
        ma = best_min

    return vec7, ma


def parse_input(filepath):
    """Parse input file into list of (name, description, trigram)."""
    concepts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            name = parts[0].upper().replace(' ', '_')
            desc = parts[1] if len(parts) > 1 else name.lower().replace('_', ' ')
            trigram = parts[2].upper() if len(parts) > 2 else None
            concepts.append((name, desc, trigram))
    return concepts


def infer_trigram(description):
    """Simple heuristic to pick a trigram from description keywords."""
    d = description.lower()

    # Personal/internal → GEN (yin) or DUI (yang)
    personal = any(w in d for w in ['feeling', 'emotion', 'inner', 'self', 'personal',
                                      'solitary', 'alone', 'quiet', 'still', 'introvert',
                                      'shame', 'guilt', 'regret', 'melanchol', 'nostalg'])
    if personal:
        return "GEN"

    expressive = any(w in d for w in ['joy', 'celebrat', 'laugh', 'express', 'charm',
                                        'humor', 'wit', 'entertain', 'cheerful', 'delight'])
    if expressive:
        return "DUI"

    # Temporal/process → XUN (yin) or ZHEN (yang)
    slow_temporal = any(w in d for w in ['gradual', 'slow', 'decay', 'erode', 'fade',
                                           'wane', 'diminish', 'age', 'wither', 'dissolve'])
    if slow_temporal:
        return "XUN"

    fast_temporal = any(w in d for w in ['sudden', 'erupt', 'burst', 'emerge', 'spark',
                                           'ignite', 'explode', 'surge', 'rapid', 'instant'])
    if fast_temporal:
        return "ZHEN"

    # Relational → KAN (yin) or LI (yang)
    relational_yin = any(w in d for w in ['bond', 'obligat', 'depend', 'submit',
                                            'hierarchy', 'serve', 'owe', 'debt', 'trap'])
    if relational_yin:
        return "KAN"

    relational_yang = any(w in d for w in ['connect', 'lead', 'inspir', 'teach',
                                             'command', 'govern', 'unite', 'rally'])
    if relational_yang:
        return "LI"

    # Spatial → QIAN (yang) or KUN (yin)
    spatial = any(w in d for w in ['place', 'space', 'physical', 'body', 'material',
                                     'land', 'earth', 'terrain', 'structure', 'build'])
    if spatial:
        return "KUN" if any(w in d for w in ['receiv', 'absorb', 'yield', 'soft']) else "QIAN"

    # Default: pick from underrepresented
    return np.random.choice(["GEN", "XUN", "KAN", "DUI"], p=[0.35, 0.30, 0.20, 0.15])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    filepath = sys.argv[1]
    force = '--force' in sys.argv
    dry_run = '--dry-run' in sys.argv

    concepts_in = parse_input(filepath)
    print(f"Loaded {len(concepts_in)} concepts from {filepath}")

    # Load existing
    ex_names, ex_mat, ex_normed = load_existing_7d()
    print(f"Existing: {len(ex_names)} concepts")

    # Check for name collisions first
    conn = get_connection()
    skip = set()
    for name, desc, tri in concepts_in:
        row = conn.execute('SELECT name FROM concepts WHERE UPPER(name)=?', (name,)).fetchone()
        alias = conn.execute('SELECT a.alias, c.name FROM aliases a JOIN concepts c ON a.concept_id=c.id WHERE UPPER(a.alias)=?', (name,)).fetchone()
        if row:
            print(f"  SKIP {name}: already exists")
            skip.add(name)
        elif alias:
            print(f"  SKIP {name}: alias for {alias[1]}")
            skip.add(name)
    conn.close()

    concepts_in = [(n, d, t) for n, d, t in concepts_in if n not in skip]
    if not concepts_in:
        print("Nothing to add.")
        return

    # Generate vectors
    batch = []
    trigram_counts = {}
    for name, desc, trigram in concepts_in:
        if trigram is None:
            trigram = infer_trigram(desc)

        vec7, min_a = generate_vector(name, desc, trigram, ex_names, ex_mat, ex_normed)

        concept = {
            "name": name,
            "x": round(float(vec7[0]), 3),
            "y": round(float(vec7[1]), 3),
            "z": round(float(vec7[2]), 3),
            "e": round(float(vec7[3]), 3),
            "f": round(float(vec7[4]), 3),
            "g": round(float(vec7[5]), 3),
            "h": round(float(vec7[6]), 3),
            "fx": round(float(vec7[0]) * 0.8, 3),
            "fy": round(float(vec7[1]) * 0.8, 3),
            "fz": round(float(vec7[2]) * 0.8, 3),
            "fe": round(float(vec7[3]) * 0.8, 3),
            "ff": round(float(vec7[4]) * 0.8, 3),
            "fg": round(float(vec7[5]) * 0.8, 3),
            "fh": round(float(vec7[6]) * 0.8, 3),
            "level": 5,
            "pos": "noun",
            "description": desc,
            "session": 112,
        }

        status = "✓" if min_a >= 15 else f"⚠ {min_a:.1f}°"
        trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
        batch.append(concept)

        # Update existing for next concept's proximity check
        ex_names.append(name)
        new_row = vec7.reshape(1, -1)
        ex_mat = np.vstack([ex_mat, new_row])
        n = np.linalg.norm(new_row)
        ex_normed = np.vstack([ex_normed, new_row / n if n > 1e-10 else new_row])

        print(f"  {name:20s} → {trigram:5s}  min={min_a:5.1f}° {status}")

    print(f"\nTrigram targets: {dict(sorted(trigram_counts.items()))}")

    if dry_run:
        print("\n[DRY RUN] Would insert {len(batch)} concepts.")
        outpath = filepath.replace('.txt', '_proposals.json')
        with open(outpath, 'w') as f:
            json.dump(batch, f, indent=2)
        print(f"Wrote proposals to {outpath}")
        return

    # Write batch and insert
    batch_path = filepath.replace('.txt', '_batch.json')
    with open(batch_path, 'w') as f:
        json.dump(batch, f, indent=2)

    # Insert using add_concept CLI
    run_batch_insert(batch_path, force=force)

    print(f"\nDone. Run: python3 -m tools.validate")


def run_batch_insert(filepath, force=False):
    """Insert batch via add_concept CLI."""
    import subprocess
    cmd = ['python3', '-m', 'tools.add_concept', 'batch', filepath]
    if force:
        cmd.append('--force')
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


if __name__ == '__main__':
    main()
