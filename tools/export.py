#!/usr/bin/env python3
"""
Export the AGI Semantic Dictionary to various formats.

Usage:
    python -m tools.export json                 # Full JSON export
    python -m tools.export json --compact        # Vectors only
    python -m tools.export numpy                 # NumPy .npz (for ML training)
    python -m tools.export names                 # Just concept names (one per line)
    python -m tools.export csv                   # CSV with all fields
    python -m tools.export dedup-manifest        # Names + aliases for dedup checking
"""

import sys
import os
import json
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from api.semantic_core import SemanticCore

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')


def ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)


def export_json(sc: SemanticCore, compact: bool = False):
    """Export full dictionary as JSON."""
    ensure_export_dir()

    rows = sc.conn.execute("SELECT * FROM concepts ORDER BY id").fetchall()
    concepts = []

    for r in rows:
        if compact:
            entry = {
                'name': r['name'],
                'essence': [r['x'], r['y'], r['z'], r['e'], r['f'], r['g'], r['h']],
                'function': [r['fx'], r['fy'], r['fz'], r['fe'], r['ff'], r['fg'], r['fh']],
                'level': r['level'],
                'trigram': r['trigram'],
            }
        else:
            # Get aliases
            aliases = sc.conn.execute(
                "SELECT alias FROM aliases WHERE concept_id=?", (r['id'],)
            ).fetchall()

            entry = {
                'id': r['id'],
                'name': r['name'],
                'core': {'w': r['w'], 'x': r['x'], 'y': r['y'], 'z': r['z']},
                'domain': {'e': r['e'], 'f': r['f'], 'g': r['g'], 'h': r['h']},
                'function': {
                    'fx': r['fx'], 'fy': r['fy'], 'fz': r['fz'],
                    'fe': r['fe'], 'ff': r['ff'], 'fg': r['fg'], 'fh': r['fh']
                },
                'level': r['level'],
                'description': r['description'],
                'trigram': r['trigram'],
                'hexagram_ref': r['hexagram_ref'],
                'aliases': [a['alias'] for a in aliases],
                'session': r['session'],
            }
        concepts.append(entry)

    suffix = '-compact' if compact else ''
    path = os.path.join(EXPORT_DIR, f'semantic_dictionary{suffix}.json')
    with open(path, 'w') as f:
        json.dump(concepts, f, indent=2 if not compact else None)

    print(f"Exported {len(concepts)} concepts to {path}")
    print(f"  Size: {os.path.getsize(path) / 1024:.1f} KB")


def export_numpy(sc: SemanticCore):
    """Export as NumPy arrays for ML training."""
    ensure_export_dir()

    rows = sc.conn.execute(
        "SELECT name, x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh, level FROM concepts ORDER BY id"
    ).fetchall()

    names = [r['name'] for r in rows]
    essence = np.array([[r['x'], r['y'], r['z'], r['e'], r['f'], r['g'], r['h']] for r in rows])
    function = np.array([[r['fx'], r['fy'], r['fz'], r['fe'], r['ff'], r['fg'], r['fh']] for r in rows])
    full_16d = np.hstack([essence, function])
    levels = np.array([r['level'] for r in rows])

    path = os.path.join(EXPORT_DIR, 'semantic_vectors.npz')
    np.savez(path,
             names=names,
             essence_7d=essence,
             function_7d=function,
             full_14d=full_16d,
             levels=levels)

    print(f"Exported {len(names)} concepts to {path}")
    print(f"  essence_7d:  {essence.shape}")
    print(f"  function_7d: {function.shape}")
    print(f"  full_14d:    {full_16d.shape}")
    print(f"  levels:      {levels.shape}")
    print(f"  Size: {os.path.getsize(path) / 1024:.1f} KB")


def export_names(sc: SemanticCore):
    """Export just concept names."""
    ensure_export_dir()
    names = sc.all_concepts()
    path = os.path.join(EXPORT_DIR, 'concept_names.txt')
    with open(path, 'w') as f:
        for name in sorted(names):
            f.write(name + '\n')
    print(f"Exported {len(names)} names to {path}")


def export_csv(sc: SemanticCore):
    """Export as CSV."""
    ensure_export_dir()
    rows = sc.conn.execute("SELECT * FROM concepts ORDER BY id").fetchall()
    path = os.path.join(EXPORT_DIR, 'semantic_dictionary.csv')

    fields = ['id', 'name', 'w', 'x', 'y', 'z', 'e', 'f', 'g', 'h',
              'fx', 'fy', 'fz', 'fe', 'ff', 'fg', 'fh',
              'level', 'description', 'hexagram_ref', 'trigram', 'session']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in rows:
            writer.writerow([r[field] for field in fields])

    print(f"Exported {len(rows)} concepts to {path}")


def export_dedup_manifest(sc: SemanticCore):
    """Export names + aliases for dedup checking by external tools."""
    ensure_export_dir()

    rows = sc.conn.execute("SELECT id, name FROM concepts ORDER BY name").fetchall()
    aliases = sc.conn.execute("SELECT alias, concept_id FROM aliases").fetchall()
    alias_map = {}
    for a in aliases:
        alias_map.setdefault(a['concept_id'], []).append(a['alias'])

    path = os.path.join(EXPORT_DIR, 'dedup_manifest.json')
    manifest = []
    for r in rows:
        entry = {'name': r['name']}
        if r['id'] in alias_map:
            entry['aliases'] = alias_map[r['id']]
        manifest.append(entry)

    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

    total_names = len(rows) + sum(len(v) for v in alias_map.values())
    print(f"Exported {len(rows)} concepts + {total_names - len(rows)} aliases to {path}")
    print(f"Total unique names: {total_names}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    sc = SemanticCore()
    cmd = sys.argv[1].lower()

    try:
        if cmd == "json":
            compact = "--compact" in sys.argv
            export_json(sc, compact=compact)
        elif cmd == "numpy":
            export_numpy(sc)
        elif cmd == "names":
            export_names(sc)
        elif cmd == "csv":
            export_csv(sc)
        elif cmd == "dedup-manifest":
            export_dedup_manifest(sc)
        else:
            print(__doc__)
    finally:
        sc.close()


if __name__ == "__main__":
    main()
