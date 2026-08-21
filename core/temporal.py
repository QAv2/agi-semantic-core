"""
Temporal Layer — stored succession orientation over the semantic dictionary.
=============================================================================

Phase 9. Design record: ~/qualia-algebra/internal/TEMPORALITY.md (2026-08-20).

The geometry cannot carry temporal orientation: for w-pinned cores the Hamilton
commutator is [0, 2 c_a x c_b] — composition's scalar is identical in either
order, so before/after cannot be derived from any angle the checker reads.
Orientation must therefore be STORED. This module owns that store:

  - a `succession` table (separate from `relations`, which stays symmetric
    geometry): directed edges earlier -> later, each tagged with a regime —
    'frame' (linear time: strict order) or 'cycle' (a named wheel with ordered
    stations and one wrap edge closing the seam).
  - regime semantics, per the design and the measured geometry (frame pairs
    encoded at the complement 90 deg where order-sensitivity is maximal; the
    cycle seam near 180 deg where composition nearly commutes): frame order is
    strict; same-cycle pairs read both ways (the wheel wraps — last year's
    harvest precedes this year's seed).
  - the deictic anchor class {PRESENT, TODAY, NOW} and side-of-present
    reasoning for locative claims ("X is in the past", "X happened yesterday").

The layer ABSTAINS (returns None) whenever it has no stored knowledge — unknown
pairs fall through to the angular pipeline untouched.
"""

from collections import deque

ANCHORS = {'PRESENT', 'TODAY', 'NOW'}

SCHEMA = """
CREATE TABLE IF NOT EXISTS succession (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    earlier_id  INTEGER NOT NULL REFERENCES concepts(id),
    later_id    INTEGER NOT NULL REFERENCES concepts(id),
    regime      TEXT    NOT NULL CHECK (regime IN ('frame', 'cycle')),
    cycle_name  TEXT,
    wrap        INTEGER NOT NULL DEFAULT 0,
    note        TEXT    NOT NULL DEFAULT '',
    UNIQUE (earlier_id, later_id)
);
"""


class TemporalJudgment:
    """Result of a temporal-layer judgment."""

    __slots__ = ('verdict', 'confidence', 'reason')

    def __init__(self, verdict, confidence, reason):
        self.verdict = verdict          # 'CONSISTENT' | 'INCONSISTENT' | 'PLAUSIBLE'
        self.confidence = confidence    # 0-100
        self.reason = reason


class TemporalLayer:
    """Stored succession orientation + regime-aware temporal judgment."""

    def __init__(self, conn):
        """conn: sqlite3 connection to the semantic db (row_factory optional)."""
        self.conn = conn
        conn.execute(SCHEMA)
        self._load()

    def _load(self):
        self.name_of = {}
        self.id_of = {}
        for row in self.conn.execute("SELECT id, name FROM concepts"):
            cid, name = row[0], row[1].upper()
            self.name_of[cid] = name
            self.id_of[name] = cid

        # forward graph over frame edges + non-wrap cycle edges (linear order);
        # cycle membership recorded separately so same-cycle pairs can wrap.
        self.forward = {}
        self.cycle_of = {}
        self.participants = set()
        for e, l, regime, cyc, wrap in self.conn.execute(
                "SELECT earlier_id, later_id, regime, cycle_name, wrap FROM succession"):
            self.participants.update((e, l))
            if regime == 'cycle' and cyc:
                self.cycle_of.setdefault(e, set()).add(cyc)
                self.cycle_of.setdefault(l, set()).add(cyc)
            if not wrap:
                self.forward.setdefault(e, set()).add(l)

    # ── graph queries ────────────────────────────────────────────────────────

    def knows(self, name):
        cid = self.id_of.get(name.upper())
        return cid is not None and cid in self.participants

    def precedes(self, a_name, b_name):
        """True iff a canonically precedes b through stored (non-wrap) edges."""
        a = self.id_of.get(a_name.upper())
        b = self.id_of.get(b_name.upper())
        if a is None or b is None or a == b:
            return False
        seen, q = {a}, deque([a])
        while q:
            n = q.popleft()
            for m in self.forward.get(n, ()):
                if m == b:
                    return True
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        return False

    def shared_cycle(self, a_name, b_name):
        a = self.id_of.get(a_name.upper())
        b = self.id_of.get(b_name.upper())
        if a is None or b is None:
            return None
        common = self.cycle_of.get(a, set()) & self.cycle_of.get(b, set())
        return next(iter(common)) if common else None

    def side_of_present(self, name):
        """'present' | 'past' | 'future' | None — position vs the anchor class."""
        u = name.upper()
        if u in ANCHORS:
            return 'present'
        for anchor in ANCHORS:
            if anchor in self.id_of:
                if self.precedes(u, anchor):
                    return 'past'
                if self.precedes(anchor, u):
                    return 'future'
        return None

    # ── judgments ────────────────────────────────────────────────────────────

    def judge_order(self, first_name, second_name):
        """Judge the claim 'FIRST comes before SECOND'. None = abstain."""
        f, s = first_name.upper(), second_name.upper()
        if not (self.knows(f) and self.knows(s)):
            return None
        cyc = self.shared_cycle(f, s)
        if cyc is not None:
            return TemporalJudgment(
                'PLAUSIBLE', 70,
                f"temporal layer: {f} and {s} share the {cyc} cycle — "
                f"the wheel reads both ways (wrap-legal)")
        if self.precedes(f, s):
            return TemporalJudgment(
                'CONSISTENT', 90,
                f"temporal layer: stored succession {f} → {s} matches the claimed order")
        if self.precedes(s, f):
            return TemporalJudgment(
                'INCONSISTENT', 95,
                f"temporal layer: claimed {f} before {s}, but stored succession "
                f"runs {s} → {f} (frame regime — order is strict)")
        return None

    def judge_locative(self, subject_name, landmark_name):
        """Judge 'SUBJECT is located at/in LANDMARK-time'. None = abstain."""
        ss = self.side_of_present(subject_name)
        ls = self.side_of_present(landmark_name)
        if ss is None or ls is None:
            return None
        if ss == ls:
            return TemporalJudgment(
                'CONSISTENT', 90,
                f"temporal layer: {subject_name.upper()} and {landmark_name.upper()} "
                f"lie on the same side of the present ({ss})")
        return TemporalJudgment(
            'INCONSISTENT', 95,
            f"temporal layer: {subject_name.upper()} lies in the {ss} but the claim "
            f"places it in the {ls} — sides of the present disagree")
