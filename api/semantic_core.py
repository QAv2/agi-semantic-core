"""
SemanticCore — Main public API for the AGI Semantic Dictionary
===============================================================
DB-backed replacement for the legacy in-memory SemanticCore.
Provides: concept lookup, angle computation, nearest neighbors,
search by trigram/level/domain, composition, coherence checking.
"""

import sqlite3
import numpy as np
from typing import List, Tuple, Optional, Dict
from core import Concept, ConceptLevel, RelationType
from core.octonion import SemanticOctonion, DualOctonion, Trigram, semantic_compose
from db.connection import get_connection, DB_PATH


class SemanticCore:
    """Main API for the AGI Semantic Dictionary."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.conn = get_connection(self.db_path)
        self._build_index()

    def _build_index(self):
        """Load vectors into numpy arrays for fast similarity search."""
        rows = self.conn.execute(
            "SELECT id, name, x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh FROM concepts ORDER BY id"
        ).fetchall()

        self._ids = []
        self._names = []
        self._vectors_7d = []  # [x,y,z,e,f,g,h] for nearest-neighbor
        self._vectors_3d = []  # [x,y,z] for core angle

        for row in rows:
            self._ids.append(row['id'])
            self._names.append(row['name'])
            self._vectors_7d.append([row['x'], row['y'], row['z'], row['e'], row['f'], row['g'], row['h']])
            self._vectors_3d.append([row['x'], row['y'], row['z']])

        self._vectors_7d = np.array(self._vectors_7d)
        self._vectors_3d = np.array(self._vectors_3d)
        self._norms_7d = np.linalg.norm(self._vectors_7d, axis=1)
        self._norms_3d = np.linalg.norm(self._vectors_3d, axis=1)
        self._name_to_idx = {name: i for i, name in enumerate(self._names)}

    def _row_to_concept(self, row) -> Concept:
        """Convert a database row to a Concept object."""
        return Concept(
            id=row['id'], name=row['name'],
            w=row['w'], x=row['x'], y=row['y'], z=row['z'],
            e=row['e'], f=row['f'], g=row['g'], h=row['h'],
            fx=row['fx'], fy=row['fy'], fz=row['fz'],
            fe=row['fe'], ff=row['ff'], fg=row['fg'], fh=row['fh'],
            level=ConceptLevel(row['level']),
            description=row['description'],
            hexagram_ref=row['hexagram_ref'],
            trigram=row['trigram'],
            pos=row['pos'] if 'pos' in row.keys() else '',
            session=row['session']
        )

    # ----- Core Access -----

    def get(self, name: str) -> Optional[Concept]:
        """Look up a concept by name or alias (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE name=? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            return self._row_to_concept(row)
        # Try alias lookup
        alias_row = self.conn.execute(
            "SELECT concept_id FROM aliases WHERE alias=? COLLATE NOCASE", (name,)
        ).fetchone()
        if alias_row:
            row = self.conn.execute(
                "SELECT * FROM concepts WHERE id=?", (alias_row['concept_id'],)
            ).fetchone()
            if row:
                return self._row_to_concept(row)
        return None

    def exists(self, name: str) -> bool:
        """Check if a concept exists (by name or alias)."""
        return self.get(name) is not None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    def all_concepts(self) -> List[str]:
        """Return all concept names."""
        return list(self._names)

    # ----- Geometric Operations -----

    def angle(self, name1: str, name2: str, mode: str = "4d") -> Optional[float]:
        """Angle between two concepts. mode='4d' (core) or '8d' (full)."""
        c1 = self.get(name1)
        c2 = self.get(name2)
        if not c1 or not c2:
            return None
        if mode == "4d":
            return c1.angle_4d(c2)
        else:
            return c1.angle_8d(c2)

    def nearest_neighbors(self, name: str, n: int = 10, mode: str = "7d") -> List[Tuple[str, float]]:
        """Find n nearest concepts by angle. Returns [(name, angle_degrees), ...]."""
        concept = self.get(name)
        if not concept:
            return []

        if mode == "3d":
            query = np.array([concept.x, concept.y, concept.z])
            vectors = self._vectors_3d
            norms = self._norms_3d
        else:
            query = np.array([concept.x, concept.y, concept.z, concept.e, concept.f, concept.g, concept.h])
            vectors = self._vectors_7d
            norms = self._norms_7d

        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        # Vectorized angle computation
        dots = vectors @ query
        denoms = norms * query_norm
        mask = denoms > 1e-10
        cos_thetas = np.full(len(vectors), 0.0)
        cos_thetas[mask] = np.clip(dots[mask] / denoms[mask], -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_thetas))

        # Sort by angle, exclude self
        indices = np.argsort(angles)
        results = []
        for idx in indices:
            if self._names[idx].upper() != concept.name.upper() and len(results) < n:
                results.append((self._names[idx], round(float(angles[idx]), 2)))

        return results

    def search_by_vector(self, vector: np.ndarray, n: int = 10) -> List[Tuple[str, float]]:
        """Find nearest concepts to an arbitrary 7D vector."""
        query_norm = np.linalg.norm(vector)
        if query_norm < 1e-10:
            return []
        dots = self._vectors_7d @ vector
        denoms = self._norms_7d * query_norm
        mask = denoms > 1e-10
        cos_thetas = np.full(len(self._vectors_7d), 0.0)
        cos_thetas[mask] = np.clip(dots[mask] / denoms[mask], -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_thetas))
        indices = np.argsort(angles)[:n]
        return [(self._names[idx], round(float(angles[idx]), 2)) for idx in indices]

    # ----- Filtered Search -----

    def by_trigram(self, trigram: str) -> List[Concept]:
        """Get all concepts with a given trigram (e.g. 'QIAN', 'KUN')."""
        rows = self.conn.execute(
            "SELECT * FROM concepts WHERE trigram=? COLLATE NOCASE ORDER BY name", (trigram,)
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def by_level(self, level: ConceptLevel) -> List[Concept]:
        """Get all concepts at a given ontological level."""
        rows = self.conn.execute(
            "SELECT * FROM concepts WHERE level=? ORDER BY name", (level.value,)
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def by_domain(self, domain: str, threshold: float = 0.5) -> List[Concept]:
        """
        Get concepts where a domain axis exceeds threshold.
        domain: 'e' (spatial), 'f' (temporal), 'g' (relational), 'h' (personal)
        """
        if domain not in ('e', 'f', 'g', 'h'):
            raise ValueError(f"Invalid domain: {domain}. Use e/f/g/h.")
        rows = self.conn.execute(
            f"SELECT * FROM concepts WHERE {domain} >= ? ORDER BY {domain} DESC", (threshold,)
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    def search(self, query: str) -> List[Concept]:
        """Search concepts by name or description substring."""
        rows = self.conn.execute(
            "SELECT * FROM concepts WHERE name LIKE ? OR description LIKE ? ORDER BY name",
            (f"%{query}%", f"%{query}%")
        ).fetchall()
        return [self._row_to_concept(r) for r in rows]

    # ----- Relations -----

    def relations_of(self, name: str, rel_type: str = None) -> List[Dict]:
        """Get all relations for a concept. Optional filter by type."""
        concept = self.get(name)
        if not concept:
            return []

        if rel_type:
            rows = self.conn.execute("""
                SELECT c1.name as n1, c2.name as n2, r.rel_type, r.angle_4d, r.angle_8d
                FROM relations r
                JOIN concepts c1 ON r.concept1_id = c1.id
                JOIN concepts c2 ON r.concept2_id = c2.id
                WHERE (r.concept1_id=? OR r.concept2_id=?) AND r.rel_type=?
            """, (concept.id, concept.id, rel_type)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT c1.name as n1, c2.name as n2, r.rel_type, r.angle_4d, r.angle_8d
                FROM relations r
                JOIN concepts c1 ON r.concept1_id = c1.id
                JOIN concepts c2 ON r.concept2_id = c2.id
                WHERE r.concept1_id=? OR r.concept2_id=?
            """, (concept.id, concept.id)).fetchall()

        results = []
        for r in rows:
            other = r['n2'] if r['n1'] == concept.name else r['n1']
            results.append({
                'other': other,
                'type': r['rel_type'],
                'angle_4d': r['angle_4d'],
                'angle_8d': r['angle_8d']
            })
        return results

    def complements_of(self, name: str) -> List[Tuple[str, float]]:
        """Get complement pairs for a concept. Returns [(name, angle), ...]."""
        rels = self.relations_of(name, 'complement')
        return [(r['other'], r['angle_4d']) for r in rels]

    # ----- Composition -----

    def compose(self, name1: str, name2: str, mode: str = "slerp") -> Optional[Dict]:
        """
        Compose two concepts. Returns dict with result vector, witness, nearest match.
        mode: 'slerp' (smooth blend) or 'average'
        """
        c1 = self.get(name1)
        c2 = self.get(name2)
        if not c1 or not c2:
            return None

        v1 = np.array([c1.x, c1.y, c1.z, c1.e, c1.f, c1.g, c1.h])
        v2 = np.array([c2.x, c2.y, c2.z, c2.e, c2.f, c2.g, c2.h])

        # Witness preservation
        core1 = np.array([c1.x, c1.y, c1.z])
        core2 = np.array([c2.x, c2.y, c2.z])
        m1, m2 = np.linalg.norm(core1), np.linalg.norm(core2)
        if m1 > 1e-10 and m2 > 1e-10:
            cos_theta = np.clip(np.dot(core1, core2) / (m1 * m2), -1.0, 1.0)
            witness = 1.0 - cos_theta
        else:
            witness = 1.0

        if mode == "average":
            result = (v1 + v2) / 2
        elif mode == "slerp":
            # Spherical linear interpolation at t=0.5
            t = 0.5
            dot = np.dot(v1, v2)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_t = np.clip(dot / (n1 * n2), -1.0, 1.0)
                theta = np.arccos(cos_t)
                if abs(theta) < 1e-10:
                    result = v1
                else:
                    sin_t = np.sin(theta)
                    result = (np.sin((1 - t) * theta) / sin_t) * v1 + (np.sin(t * theta) / sin_t) * v2
            else:
                result = (v1 + v2) / 2
        else:
            raise ValueError(f"Unknown compose mode: {mode!r}. Use 'slerp' or 'average'.")

        # Find nearest existing concept
        nearest = self.search_by_vector(result, n=3)

        return {
            'vector': result.tolist(),
            'witness': round(witness, 4),
            'nearest': nearest,
            'components': [name1, name2],
            'mode': mode
        }

    # ----- Coherence -----

    def coherence_score(self, *names: str) -> Optional[float]:
        """
        Compute witness-based coherence for a sequence of concepts.
        Returns average witness preservation (1.0 = fully coherent).
        """
        concepts = [self.get(n) for n in names]
        if any(c is None for c in concepts) or len(concepts) < 2:
            return None

        witnesses = []
        for i in range(len(concepts) - 1):
            c1, c2 = concepts[i], concepts[i+1]
            core1 = np.array([c1.x, c1.y, c1.z])
            core2 = np.array([c2.x, c2.y, c2.z])
            m1, m2 = np.linalg.norm(core1), np.linalg.norm(core2)
            if m1 > 1e-10 and m2 > 1e-10:
                cos_theta = np.clip(np.dot(core1, core2) / (m1 * m2), -1.0, 1.0)
                witnesses.append(1.0 - cos_theta)
            else:
                witnesses.append(1.0)

        return round(sum(witnesses) / len(witnesses), 4)

    # ----- Statistics -----

    def statistics(self) -> Dict:
        """Return comprehensive dictionary statistics."""
        total = self.count()
        trigrams = self.conn.execute(
            "SELECT trigram, COUNT(*) as cnt FROM concepts GROUP BY trigram ORDER BY cnt DESC"
        ).fetchall()
        levels = self.conn.execute(
            "SELECT level, COUNT(*) as cnt FROM concepts GROUP BY level ORDER BY level"
        ).fetchall()
        rel_types = self.conn.execute(
            "SELECT rel_type, COUNT(*) as cnt FROM relations GROUP BY rel_type ORDER BY cnt DESC"
        ).fetchall()

        return {
            'total_concepts': total,
            'total_relations': self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
            'total_aliases': self.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0],
            'trigram_distribution': {r['trigram']: r['cnt'] for r in trigrams},
            'level_distribution': {ConceptLevel(r['level']).name: r['cnt'] for r in levels},
            'relation_distribution': {r['rel_type']: r['cnt'] for r in rel_types},
        }

    def describe(self, name: str) -> Optional[str]:
        """Human-readable description of a concept and its relationships."""
        c = self.get(name)
        if not c:
            return None

        lines = [
            f"{'='*60}",
            f"  {c.name}",
            f"{'='*60}",
            f"  Level:       {c.level.name} ({c.level.value})",
            f"  Description: {c.description}",
            f"  Trigram:     {c.trigram}",
            f"  Core:        [{c.x:+.3f}, {c.y:+.3f}, {c.z:+.3f}]",
            f"  Domain:      [{c.e:.2f}, {c.f:.2f}, {c.g:.2f}, {c.h:.2f}]  ({c.domain_profile()})",
            f"  Function:    [{c.fx:+.3f}, {c.fy:+.3f}, {c.fz:+.3f}]",
            f"  Fn Domain:   [{c.fe:.2f}, {c.ff:.2f}, {c.fg:.2f}, {c.fh:.2f}]",
        ]

        if c.hexagram_ref:
            lines.append(f"  Hexagram:    #{c.hexagram_ref}")

        # Relations
        rels = self.relations_of(name)
        if rels:
            lines.append(f"\n  Relations ({len(rels)}):")
            for r in sorted(rels, key=lambda x: x['angle_4d']):
                lines.append(f"    {r['type']:12s} {r['other']:20s} {r['angle_4d']:5.1f}")

        # Nearest neighbors
        nn = self.nearest_neighbors(name, n=5)
        if nn:
            lines.append(f"\n  Nearest Neighbors:")
            for n_name, angle in nn:
                lines.append(f"    {n_name:20s} {angle:5.1f}")

        return "\n".join(lines)

    def close(self):
        self.conn.close()
