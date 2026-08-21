"""
Seed the succession table — Phase 9 temporal layer.
====================================================

Idempotent: creates the table if absent, then syncs it to exactly the seed set
below (delete-and-reinsert inside one transaction). Prints a ledger.

Seeds are conservative: only canonical, unambiguous orderings. Frame = linear
time, strict. Cycles = named wheels with ordered stations; the wrap edge closes
the seam (last -> first) and is excluded from the linear-precedence graph.

Design record: ~/qualia-algebra/internal/TEMPORALITY.md.

Usage: python3 -m tools.temporal_seed
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_connection
from core.temporal import SCHEMA

# frame edges: (earlier, later, note)
FRAME = [
    ('PAST', 'PRESENT', 'the deictic frame'),
    ('PRESENT', 'FUTURE', 'the deictic frame'),
    ('YESTERDAY', 'TODAY', 'the deictic day-line'),
    ('TODAY', 'TOMORROW', 'the deictic day-line'),
    ('BEFORE', 'AFTER', 'order itself'),
    ('CAUSE', 'EFFECT', 'causal order'),
    ('BEGIN', 'END', 'event frame'),
    ('BEGINNING', 'END', 'event frame'),
    ('START', 'FINISH', 'event frame'),
]

# cycles: name -> ordered stations (wrap edge auto-added last -> first)
CYCLES = {
    'day': ['DAWN', 'SUNRISE', 'MORNING', 'NOON', 'EVENING', 'SUNSET', 'DUSK',
            'NIGHT', 'MIDNIGHT'],
    'year': ['SPRING', 'SUMMER', 'AUTUMN', 'WINTER'],
    'life': ['BIRTH', 'AGE', 'DEATH'],
    'vegetal': ['SEED', 'GROW', 'BLOOM', 'HARVEST', 'DECAY'],
}


def main():
    conn = get_connection()
    conn.execute(SCHEMA)

    ids = {row[1].upper(): row[0]
           for row in conn.execute("SELECT id, name FROM concepts")}

    rows = []
    missing = []

    def add(e, l, regime, cyc, wrap, note):
        if e not in ids:
            missing.append(e)
            return
        if l not in ids:
            missing.append(l)
            return
        rows.append((ids[e], ids[l], regime, cyc, wrap, note))

    for e, l, note in FRAME:
        add(e, l, 'frame', None, 0, note)
    for cyc, stations in CYCLES.items():
        for a, b in zip(stations, stations[1:]):
            add(a, b, 'cycle', cyc, 0, f'{cyc} cycle')
        add(stations[-1], stations[0], 'cycle', cyc, 1, f'{cyc} cycle — the seam')

    if missing:
        print("MISSING CONCEPTS (not seeded):", sorted(set(missing)))

    with conn:
        conn.execute("DELETE FROM succession")
        conn.executemany(
            "INSERT INTO succession (earlier_id, later_id, regime, cycle_name, wrap, note) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows)

    n_frame = sum(1 for r in rows if r[2] == 'frame')
    n_cycle = len(rows) - n_frame
    n_wrap = sum(1 for r in rows if r[4])
    print(f"succession seeded: {len(rows)} edges "
          f"({n_frame} frame, {n_cycle} cycle incl. {n_wrap} wrap/seam)")
    name_of = {v: k for k, v in ids.items()}
    for e, l, regime, cyc, wrap, _ in rows:
        tag = f"{regime}" + (f":{cyc}" if cyc else "") + (" [SEAM]" if wrap else "")
        print(f"  {name_of[e]:12s} → {name_of[l]:12s}  {tag}")


if __name__ == '__main__':
    main()
