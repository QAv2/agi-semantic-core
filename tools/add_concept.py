#!/usr/bin/env python3
"""
Add concepts to the AGI Semantic Dictionary with dedup protection.

Usage:
    python -m tools.add_concept                     # Interactive mode
    python -m tools.add_concept check NAME x y z    # Dedup check only
    python -m tools.add_concept batch file.json     # Batch from JSON
    python -m tools.add_concept batch file.json --force  # Skip dedup
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.builder import ConceptBuilder
from core.encoding import ConceptLevel, RelationType


def interactive_add(builder: ConceptBuilder):
    """Interactive concept addition."""
    print("=" * 50)
    print("  ADD CONCEPT (interactive)")
    print("=" * 50)
    print()

    name = input("  Name: ").strip().upper()
    if not name:
        print("  Cancelled.")
        return

    print()
    print("  Core 4D (Yang/Yin, Becoming/Abiding, Ordinality):")
    x = float(input("    x (yang+/yin-): "))
    y = float(input("    y (becoming+/abiding-): "))
    z = float(input("    z (ascending+/descending-): "))

    print()
    print("  Domain 4D (0-1 range):")
    e = float(input("    e (spatial):    ") or "0.0")
    f = float(input("    f (temporal):   ") or "0.0")
    g = float(input("    g (relational): ") or "0.0")
    h = float(input("    h (personal):   ") or "0.0")

    print()
    print("  Function 8D (how it operates):")
    fx = float(input("    fx: ") or "0.0")
    fy = float(input("    fy: ") or "0.0")
    fz = float(input("    fz: ") or "0.0")
    fe = float(input("    fe: ") or "0.0")
    ff = float(input("    ff: ") or "0.0")
    fg = float(input("    fg: ") or "0.0")
    fh = float(input("    fh: ") or "0.0")

    print()
    level_str = input(f"  Level ({', '.join(l.name for l in ConceptLevel)}): ").strip().upper() or "DERIVED"
    try:
        level = ConceptLevel[level_str]
    except KeyError:
        print(f"  Invalid level, using DERIVED")
        level = ConceptLevel.DERIVED

    description = input("  Description: ").strip()
    aliases_str = input("  Aliases (comma-separated, or blank): ").strip()
    aliases = tuple(a.strip().upper() for a in aliases_str.split(",") if a.strip()) if aliases_str else ()
    session = int(input("  Session number: ") or "0")

    # Dedup check first
    print()
    result = builder.check_dedup(name, x, y, z, e, f, g, h, aliases)
    if result.is_duplicate:
        print(f"  ⚠ DUPLICATE DETECTED: {result.reason}")
        if result.existing_match:
            print(f"    Matches: {result.existing_match}")
        if result.angle is not None:
            print(f"    Angle: {result.angle}°")
        force = input("  Force add anyway? (y/N): ").strip().lower() == 'y'
        if not force:
            print("  Cancelled.")
            return
    else:
        force = False

    success, msg = builder.add_concept(
        name=name, x=x, y=y, z=z,
        level=level, description=description,
        e=e, f=f, g=g, h=h,
        fx=fx, fy=fy, fz=fz, fe=fe, ff=ff, fg=fg, fh=fh,
        aliases=aliases, session=session, force=force or result.is_duplicate
    )
    print(f"  {'✓' if success else '✗'} {msg}")


def check_concept(builder: ConceptBuilder, args: list):
    """Just check dedup without adding."""
    if len(args) < 4:
        print("Usage: check NAME x y z [e f g h]")
        return

    name = args[0].upper()
    x, y, z = float(args[1]), float(args[2]), float(args[3])
    e = float(args[4]) if len(args) > 4 else 0.0
    f = float(args[5]) if len(args) > 5 else 0.0
    g = float(args[6]) if len(args) > 6 else 0.0
    h = float(args[7]) if len(args) > 7 else 0.0

    result = builder.check_dedup(name, x, y, z, e, f, g, h)
    if result.is_duplicate:
        print(f"DUPLICATE: {result.reason}")
        if result.existing_match:
            print(f"  Matches: {result.existing_match}")
        if result.angle is not None:
            print(f"  Angle: {result.angle}°")
    else:
        print(f"CLEAN: {name} can be added")


def batch_add(builder: ConceptBuilder, filepath: str, force: bool = False):
    """Add concepts from a JSON file."""
    with open(filepath, 'r') as f:
        concepts = json.load(f)

    if not isinstance(concepts, list):
        print("Error: JSON must be a list of concept objects")
        return

    print(f"Loading {len(concepts)} concepts from {filepath}")
    print()

    # Convert string levels to enum
    for c in concepts:
        if 'level' in c and isinstance(c['level'], str):
            c['level'] = ConceptLevel[c['level'].upper()]
        if 'aliases' in c and isinstance(c['aliases'], list):
            c['aliases'] = tuple(c['aliases'])

    session = concepts[0].get('session', 0) if concepts else 0
    result = builder.add_batch(concepts, session=session, force=force)

    print(f"Submitted: {result.submitted}")
    print(f"Inserted:  {result.inserted}")
    print(f"Rejected:  {result.rejected}")

    if result.duplicates:
        print()
        print("Duplicates detected:")
        for d in result.duplicates:
            msg = f"  {d.name}: {d.reason}"
            if d.existing_match:
                msg += f" (matches {d.existing_match})"
            if d.angle is not None:
                msg += f" at {d.angle}°"
            print(msg)

    if result.inserted_names:
        print()
        print("Inserted:")
        for name in result.inserted_names:
            print(f"  {name}")


def main():
    builder = ConceptBuilder()

    try:
        if len(sys.argv) < 2:
            interactive_add(builder)
        elif sys.argv[1] == "check":
            check_concept(builder, sys.argv[2:])
        elif sys.argv[1] == "batch":
            if len(sys.argv) < 3:
                print("Usage: batch file.json [--force]")
                return
            force = "--force" in sys.argv
            batch_add(builder, sys.argv[2], force=force)
        else:
            print(__doc__)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
