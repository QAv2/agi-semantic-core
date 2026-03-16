"""
Concept Builder with Dedup Gate
================================
Ensures no duplicate concepts enter the dictionary.
Three-layer dedup check:
1. Exact name match (case-insensitive)
2. Alias collision (new name matches existing alias or vice versa)
3. Vector proximity (angle < 15° to any existing concept = likely duplicate)
"""

import sqlite3
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from core import Concept, ConceptLevel, RelationType
from db.connection import get_connection, DB_PATH


@dataclass
class DedupResult:
    """Result of a dedup check for a single concept."""
    name: str
    is_duplicate: bool
    reason: str  # "clean", "name_exists", "alias_collision", "vector_proximity"
    existing_match: Optional[str] = None  # Name of the concept it collides with
    angle: Optional[float] = None  # Angle to closest match (for vector proximity)


@dataclass
class BatchResult:
    """Result of a batch insertion attempt."""
    submitted: int
    inserted: int
    rejected: int
    duplicates: List[DedupResult]
    inserted_names: List[str]


class ConceptBuilder:
    """Add concepts to the dictionary with automatic dedup and validation."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.conn = get_connection(self.db_path)
        self._load_index()

    def _load_index(self):
        """Load all concept names, aliases, and vectors into memory for fast dedup."""
        # Load all concept names (case-insensitive set)
        rows = self.conn.execute("SELECT id, name, x, y, z, e, f, g, h FROM concepts").fetchall()
        self.names = {}  # uppercase name → id
        self.vectors = {}  # id → numpy array [x,y,z,e,f,g,h]
        self.id_to_name = {}  # id → canonical name

        for row in rows:
            self.names[row['name'].upper()] = row['id']
            self.id_to_name[row['id']] = row['name']
            self.vectors[row['id']] = np.array([
                row['x'], row['y'], row['z'],
                row['e'], row['f'], row['g'], row['h']
            ])

        # Load all aliases
        self.alias_map = {}  # uppercase alias → concept_id
        alias_rows = self.conn.execute("SELECT alias, concept_id FROM aliases").fetchall()
        for row in alias_rows:
            self.alias_map[row['alias'].upper()] = row['concept_id']

    def check_dedup(self, name: str, x: float, y: float, z: float,
                    e: float = 0.0, f: float = 0.0, g: float = 0.0, h: float = 0.0,
                    aliases: Tuple[str, ...] = (),
                    proximity_threshold: float = 15.0) -> DedupResult:
        """
        Check if a concept would be a duplicate.

        Three checks in order:
        1. Exact name match (case-insensitive) against concepts AND aliases
        2. Any of the new concept's aliases collide with existing names/aliases
        3. Vector proximity: angle < proximity_threshold to any existing concept

        Returns DedupResult with is_duplicate=True if any check fails.
        """
        upper_name = name.upper()

        # Check 1: Name already exists as concept
        if upper_name in self.names:
            return DedupResult(name=name, is_duplicate=True, reason="name_exists",
                               existing_match=self.id_to_name[self.names[upper_name]])

        # Check 1b: Name exists as an alias
        if upper_name in self.alias_map:
            cid = self.alias_map[upper_name]
            return DedupResult(name=name, is_duplicate=True, reason="alias_collision",
                               existing_match=self.id_to_name[cid])

        # Check 2: Any alias collisions
        for alias in aliases:
            upper_alias = alias.upper()
            if upper_alias in self.names:
                return DedupResult(name=name, is_duplicate=True, reason="alias_collision",
                                   existing_match=self.id_to_name[self.names[upper_alias]])
            if upper_alias in self.alias_map:
                cid = self.alias_map[upper_alias]
                return DedupResult(name=name, is_duplicate=True, reason="alias_collision",
                                   existing_match=self.id_to_name[cid])

        # Check 3: Vector proximity
        new_vec = np.array([x, y, z, e, f, g, h])
        new_mag = np.linalg.norm(new_vec)

        if new_mag > 1e-10:  # Skip proximity check for zero vectors
            closest_name = None
            closest_angle = 180.0

            for cid, vec in self.vectors.items():
                mag = np.linalg.norm(vec)
                if mag < 1e-10:
                    continue
                cos_theta = np.clip(np.dot(new_vec, vec) / (new_mag * mag), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                if angle < closest_angle:
                    closest_angle = angle
                    closest_name = self.id_to_name[cid]

            if closest_angle < proximity_threshold:
                return DedupResult(name=name, is_duplicate=True, reason="vector_proximity",
                                   existing_match=closest_name, angle=round(closest_angle, 2))

        return DedupResult(name=name, is_duplicate=False, reason="clean")

    def add_concept(self, name: str, x: float, y: float, z: float,
                    level: ConceptLevel = ConceptLevel.DERIVED,
                    description: str = "",
                    e: float = 0.0, f: float = 0.0, g: float = 0.0, h: float = 0.0,
                    fx: float = 0.0, fy: float = 0.0, fz: float = 0.0,
                    fe: float = 0.0, ff: float = 0.0, fg: float = 0.0, fh: float = 0.0,
                    hexagram_ref: int = 0,
                    aliases: Tuple[str, ...] = (),
                    pos: str = "",
                    session: int = 0,
                    force: bool = False) -> Tuple[bool, str]:
        """
        Add a single concept. Returns (success, message).
        Set force=True to bypass dedup (for corrections/re-encodings).
        """
        if not force:
            result = self.check_dedup(name, x, y, z, e, f, g, h, aliases)
            if result.is_duplicate:
                msg = f"BLOCKED: {name} — {result.reason}"
                if result.existing_match:
                    msg += f" (matches {result.existing_match})"
                if result.angle is not None:
                    msg += f" at {result.angle}°"
                return False, msg

        # Compute trigram
        from core.octonion import SemanticOctonion
        octo = SemanticOctonion(w=1.0, x=x, y=y, z=z, e=e, f=f, g=g, h=h)
        try:
            tri_name = octo.to_trigram(x_axis=x, y_axis=y).name
        except Exception:
            tri_name = ""

        # Insert
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO concepts (name, w, x, y, z, e, f, g, h,
                                  fx, fy, fz, fe, ff, fg, fh,
                                  level, description, hexagram_ref, trigram, pos, session)
            VALUES (?, 1.0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, x, y, z, e, f, g, h, fx, fy, fz, fe, ff, fg, fh,
              level.value, description, hexagram_ref, tri_name, pos, session))

        concept_id = cur.lastrowid

        # Insert aliases
        for alias in aliases:
            cur.execute("INSERT INTO aliases (alias, concept_id) VALUES (?, ?)", (alias, concept_id))

        self.conn.commit()

        # Update in-memory index
        self.names[name.upper()] = concept_id
        self.id_to_name[concept_id] = name
        self.vectors[concept_id] = np.array([x, y, z, e, f, g, h])
        for alias in aliases:
            self.alias_map[alias.upper()] = concept_id

        return True, f"Added: {name} [{x:+.3f},{y:+.3f},{z:+.3f}] {tri_name}"

    def add_batch(self, concepts: List[Dict], session: int = 0,
                  force: bool = False) -> BatchResult:
        """
        Add a batch of concepts with dedup checking.

        Each dict in concepts should have at minimum: name, x, y, z
        Optional: level, description, e, f, g, h, fx-fh, hexagram_ref, aliases

        Returns BatchResult with full report.
        """
        duplicates = []
        inserted_names = []

        # First pass: check all for dedup
        if not force:
            for c in concepts:
                result = self.check_dedup(
                    c['name'], c['x'], c['y'], c['z'],
                    c.get('e', 0.0), c.get('f', 0.0), c.get('g', 0.0), c.get('h', 0.0),
                    c.get('aliases', ())
                )
                if result.is_duplicate:
                    duplicates.append(result)

        # If any duplicates found, report without inserting
        if duplicates and not force:
            return BatchResult(
                submitted=len(concepts),
                inserted=0,
                rejected=len(duplicates),
                duplicates=duplicates,
                inserted_names=[]
            )

        # Second pass: insert all clean concepts
        for c in concepts:
            level = c.get('level', ConceptLevel.DERIVED)
            if isinstance(level, int):
                level = ConceptLevel(level)

            success, msg = self.add_concept(
                name=c['name'], x=c['x'], y=c['y'], z=c['z'],
                level=level,
                description=c.get('description', ''),
                e=c.get('e', 0.0), f=c.get('f', 0.0),
                g=c.get('g', 0.0), h=c.get('h', 0.0),
                fx=c.get('fx', 0.0), fy=c.get('fy', 0.0), fz=c.get('fz', 0.0),
                fe=c.get('fe', 0.0), ff=c.get('ff', 0.0),
                fg=c.get('fg', 0.0), fh=c.get('fh', 0.0),
                hexagram_ref=c.get('hexagram_ref', 0),
                aliases=c.get('aliases', ()),
                pos=c.get('pos', ''),
                session=session,
                force=force
            )
            if success:
                inserted_names.append(c['name'])

        return BatchResult(
            submitted=len(concepts),
            inserted=len(inserted_names),
            rejected=len(concepts) - len(inserted_names),
            duplicates=duplicates,
            inserted_names=inserted_names
        )

    def add_relation(self, name1: str, name2: str, rel_type: RelationType,
                     session: int = 0) -> Tuple[bool, str]:
        """Add a relation between two existing concepts."""
        row1 = self.conn.execute(
            "SELECT id, x, y, z, e, f, g, h FROM concepts WHERE name=? COLLATE NOCASE",
            (name1,)
        ).fetchone()
        row2 = self.conn.execute(
            "SELECT id, x, y, z, e, f, g, h FROM concepts WHERE name=? COLLATE NOCASE",
            (name2,)
        ).fetchone()

        if not row1:
            return False, f"Concept not found: {name1}"
        if not row2:
            return False, f"Concept not found: {name2}"

        # Compute angles
        v1 = np.array([row1['x'], row1['y'], row1['z']])
        v2 = np.array([row2['x'], row2['y'], row2['z']])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 > 1e-10 and m2 > 1e-10:
            cos_t = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            angle_4d = round(np.degrees(np.arccos(cos_t)), 4)
        else:
            angle_4d = 0.0

        v1f = np.array([row1['x'], row1['y'], row1['z'], row1['e'], row1['f'], row1['g'], row1['h']])
        v2f = np.array([row2['x'], row2['y'], row2['z'], row2['e'], row2['f'], row2['g'], row2['h']])
        m1f, m2f = np.linalg.norm(v1f), np.linalg.norm(v2f)
        if m1f > 1e-10 and m2f > 1e-10:
            cos_tf = np.clip(np.dot(v1f, v2f) / (m1f * m2f), -1.0, 1.0)
            angle_8d = round(np.degrees(np.arccos(cos_tf)), 4)
        else:
            angle_8d = 0.0

        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO relations (concept1_id, concept2_id, rel_type, angle_4d, angle_8d, session)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row1['id'], row2['id'], rel_type.value, angle_4d, angle_8d, session))
            self.conn.commit()
            return True, f"Relation: {name1} <-> {name2} ({rel_type.value}) {angle_4d:.1f}°"
        except sqlite3.IntegrityError as e:
            return False, f"Relation already exists: {e}"

    def close(self):
        self.conn.close()
