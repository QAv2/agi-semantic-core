"""Core data structures for semantic encoding."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from core.octonion import SemanticOctonion, DualOctonion, Trigram


class ConceptLevel(Enum):
    UNITY = 0           # [1,0,0,0]
    DYAD = 1            # First distinction
    TRIAD = 2           # Second distinction
    TETRAD = 3          # Third distinction (elements)
    QUALITY = 4         # Qualities
    DERIVED = 5         # Derived concepts
    VERB = 6            # Verbs
    ABSTRACT = 7        # Abstract concepts
    INTERROGATIVE = 8   # Questions


class RelationType(Enum):
    SYNONYM = "synonym"           # ~0°
    AFFINITY = "affinity"         # 15-45°
    ADJACENT = "adjacent"         # 45-75°
    COMPLEMENT = "complement"     # 80-100°
    OPPOSITION = "opposition"     # >150°
    HOMONYM = "homonym"           # Same word, different meaning (angle varies)
    ISOMORPHIC = "isomorphic"     # Same principle, different domain (e.g. THREE↔TRIANGLE↔TRIAD)


@dataclass
class Concept:
    """A concept with full 4D/8D/16D encoding."""
    id: int = 0
    name: str = ""

    # Core 4D
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    # Domain 4D
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0

    # Function 8D
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    fe: float = 0.0
    ff: float = 0.0
    fg: float = 0.0
    fh: float = 0.0

    # Metadata
    level: ConceptLevel = ConceptLevel.DERIVED
    description: str = ""
    hexagram_ref: int = 0
    trigram: str = ""
    pos: str = ""  # Part of speech: noun/verb/adj/adv/prep/conj/det/pron/num/shape
    session: int = 0
    aliases: Tuple[str, ...] = ()

    # -------------------------------------------------------------------------
    # Core accessors (from legacy ExtendedConcept)
    # -------------------------------------------------------------------------

    def quaternion(self) -> Tuple[float, float, float, float]:
        """Return 4D quaternion [w, x, y, z]."""
        return (self.w, self.x, self.y, self.z)

    def octonion(self) -> SemanticOctonion:
        """Return 8D octonion."""
        return SemanticOctonion(
            w=self.w, x=self.x, y=self.y, z=self.z,
            e=self.e, f=self.f, g=self.g, h=self.h
        )

    def dual_octonion(self) -> DualOctonion:
        """Return 16D dual octonion."""
        return DualOctonion(
            essence=self.octonion(),
            function=SemanticOctonion(
                w=1.0, x=self.fx, y=self.fy, z=self.fz,
                e=self.fe, f=self.ff, g=self.fg, h=self.fh
            )
        )

    # -------------------------------------------------------------------------
    # Magnitude and angle methods
    # -------------------------------------------------------------------------

    def vector_magnitude(self) -> float:
        """Magnitude of 3D vector part."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def domain_magnitude(self) -> float:
        """Magnitude of domain part [e,f,g,h]."""
        return np.sqrt(self.e**2 + self.f**2 + self.g**2 + self.h**2)

    def angle_4d(self, other: 'Concept') -> float:
        """Angle between 3D vector parts in degrees."""
        v1 = np.array([self.x, self.y, self.z])
        v2 = np.array([other.x, other.y, other.z])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def angle_8d(self, other: 'Concept') -> float:
        """Angle in full 8D space (excluding w)."""
        v1 = np.array([self.x, self.y, self.z, self.e, self.f, self.g, self.h])
        v2 = np.array([other.x, other.y, other.z, other.e, other.f, other.g, other.h])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    # -------------------------------------------------------------------------
    # Trigram and domain profile
    # -------------------------------------------------------------------------

    def compute_trigram(self) -> str:
        """Compute trigram name from domain components using octonion.to_trigram()."""
        tri = self.octonion().to_trigram(x_axis=self.x, y_axis=self.y)
        return tri.name

    def domain_profile(self) -> str:
        """Human-readable domain summary."""
        domains = {'S': self.e, 'T': self.f, 'R': self.g, 'P': self.h}
        active = [k for k, v in domains.items() if abs(v) > 0.3]
        if not active:
            return "General"
        return "+".join(active)

    # -------------------------------------------------------------------------
    # Convenience vector accessors
    # -------------------------------------------------------------------------

    def essence_vector(self) -> np.ndarray:
        """Return 7D essence vector [x,y,z,e,f,g,h]."""
        return np.array([self.x, self.y, self.z, self.e, self.f, self.g, self.h])

    def function_vector(self) -> np.ndarray:
        """Return 7D function vector [fx,fy,fz,fe,ff,fg,fh]."""
        return np.array([self.fx, self.fy, self.fz, self.fe, self.ff, self.fg, self.fh])

    def full_vector(self) -> np.ndarray:
        """Return all 14D (essence + function)."""
        return np.concatenate([self.essence_vector(), self.function_vector()])

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"{self.name}: 4D[{self.x:+.2f},{self.y:+.2f},{self.z:+.2f}] "
                f"8D[{self.e:+.2f},{self.f:+.2f},{self.g:+.2f},{self.h:+.2f}]")
