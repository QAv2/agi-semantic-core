#!/usr/bin/env python3
"""
Query the AGI Semantic Dictionary.

Usage:
    python -m tools.query describe FIRE
    python -m tools.query neighbors LOVE 10
    python -m tools.query angle FIRE WATER
    python -m tools.query trigram QIAN
    python -m tools.query level TETRAD
    python -m tools.query domain g 0.7          # relational > 0.7
    python -m tools.query search "transform"
    python -m tools.query complements LOVE
    python -m tools.query relations FIRE
    python -m tools.query compose FIRE WATER
    python -m tools.query coherence LOVE TRUTH BEAUTY
    python -m tools.query stats
    python -m tools.query exists QUANTUM
    python -m tools.query count
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore
from core.encoding import ConceptLevel


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    sc = SemanticCore()

    try:
        if cmd == "describe" and args:
            result = sc.describe(args[0])
            print(result if result else f"Not found: {args[0]}")

        elif cmd == "neighbors" and args:
            n = int(args[1]) if len(args) > 1 else 10
            mode = args[2] if len(args) > 2 else "7d"
            results = sc.nearest_neighbors(args[0], n=n, mode=mode)
            if results:
                print(f"Nearest neighbors to {args[0]} ({mode}):")
                for name, angle in results:
                    print(f"  {name:25s} {angle:6.1f}°")
            else:
                print(f"Not found: {args[0]}")

        elif cmd == "angle" and len(args) >= 2:
            mode = args[2] if len(args) > 2 else "4d"
            angle = sc.angle(args[0], args[1], mode=mode)
            if angle is not None:
                print(f"{args[0]} <-> {args[1]}: {angle:.2f}° ({mode})")
            else:
                print(f"Concept not found")

        elif cmd == "trigram" and args:
            concepts = sc.by_trigram(args[0])
            print(f"{args[0]}: {len(concepts)} concepts")
            for c in concepts:
                print(f"  {c.name:25s} [{c.x:+.2f},{c.y:+.2f},{c.z:+.2f}] L{c.level.value} {c.description[:50]}")

        elif cmd == "level" and args:
            try:
                level = ConceptLevel[args[0].upper()]
            except KeyError:
                print(f"Invalid level. Choose: {', '.join(l.name for l in ConceptLevel)}")
                return
            concepts = sc.by_level(level)
            print(f"{level.name}: {len(concepts)} concepts")
            for c in concepts:
                print(f"  {c.name:25s} [{c.x:+.2f},{c.y:+.2f},{c.z:+.2f}] {c.trigram:5s} {c.description[:50]}")

        elif cmd == "domain" and len(args) >= 1:
            domain = args[0]
            threshold = float(args[1]) if len(args) > 1 else 0.5
            domain_labels = {'e': 'Spatial', 'f': 'Temporal', 'g': 'Relational', 'h': 'Personal'}
            concepts = sc.by_domain(domain, threshold)
            print(f"{domain_labels.get(domain, domain)} >= {threshold}: {len(concepts)} concepts")
            for c in concepts[:50]:
                val = getattr(c, domain)
                print(f"  {c.name:25s} {domain}={val:.2f} {c.trigram:5s} {c.description[:45]}")
            if len(concepts) > 50:
                print(f"  ... and {len(concepts) - 50} more")

        elif cmd == "search" and args:
            concepts = sc.search(args[0])
            print(f"Search '{args[0]}': {len(concepts)} results")
            for c in concepts:
                print(f"  {c.name:25s} {c.trigram:5s} {c.description[:50]}")

        elif cmd == "complements" and args:
            pairs = sc.complements_of(args[0])
            if pairs:
                print(f"Complements of {args[0]}:")
                for name, angle in pairs:
                    print(f"  {name:25s} {angle:6.1f}°")
            else:
                print(f"No complements found for {args[0]}")

        elif cmd == "relations" and args:
            rel_type = args[1] if len(args) > 1 else None
            rels = sc.relations_of(args[0], rel_type=rel_type)
            if rels:
                print(f"Relations of {args[0]}:" + (f" (type={rel_type})" if rel_type else ""))
                for r in sorted(rels, key=lambda x: x['angle_4d']):
                    print(f"  {r['type']:12s} {r['other']:25s} 4D={r['angle_4d']:5.1f}° 8D={r['angle_8d']:5.1f}°")
            else:
                print(f"No relations found for {args[0]}")

        elif cmd == "compose" and len(args) >= 2:
            mode = args[2] if len(args) > 2 else "slerp"
            result = sc.compose(args[0], args[1], mode=mode)
            if result:
                print(f"{args[0]} + {args[1]} ({mode}):")
                print(f"  Witness: {result['witness']:.4f}")
                print(f"  Nearest matches:")
                for name, angle in result['nearest']:
                    print(f"    {name:25s} {angle:6.1f}°")
            else:
                print("Concept not found")

        elif cmd == "coherence" and args:
            score = sc.coherence_score(*args)
            if score is not None:
                print(f"Coherence({', '.join(args)}) = {score:.4f}")
                if score < 0.5:
                    print("  → Low coherence (dissolution)")
                elif score < 1.2:
                    print("  → Healthy coherence")
                else:
                    print("  → High tension")
            else:
                print("Concept not found")

        elif cmd == "stats":
            stats = sc.statistics()
            print("=" * 50)
            print("  AGI SEMANTIC DICTIONARY STATISTICS")
            print("=" * 50)
            print(f"  Concepts:   {stats['total_concepts']}")
            print(f"  Relations:  {stats['total_relations']}")
            print(f"  Aliases:    {stats['total_aliases']}")
            print()
            print("  Trigram Distribution:")
            for tri, cnt in sorted(stats['trigram_distribution'].items(), key=lambda x: -x[1]):
                bar = "█" * (cnt // 5)
                print(f"    {tri:8s} {cnt:4d} {bar}")
            print()
            print("  Level Distribution:")
            for lvl, cnt in stats['level_distribution'].items():
                print(f"    {lvl:15s} {cnt:4d}")
            print()
            print("  Relation Types:")
            for rt, cnt in stats['relation_distribution'].items():
                print(f"    {rt:12s} {cnt:4d}")

        elif cmd == "exists" and args:
            print(f"{args[0]}: {'YES' if sc.exists(args[0]) else 'NO'}")

        elif cmd == "count":
            print(f"Total concepts: {sc.count()}")

        else:
            print(__doc__)

    finally:
        sc.close()


if __name__ == "__main__":
    main()
