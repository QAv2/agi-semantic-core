"""
Octonion and Dual Octonion Infrastructure for Semantic Encoding
================================================================

Provides:
- SemanticOctonion: 8D encoding [w,x,y,z | e,f,g,h]
- DualOctonion: 16D encoding (essence + function)
- Trigram: I Ching trigram mapping

8D Domain Components:
- e: Spatial (physical extension, location)
- f: Temporal (time-related, process duration)
- g: Relational (connection, meaning, social)
- h: Personal (subjective intensity, inner experience)

Trigram Mapping (to e,f,g,h):
- QIAN ☰ (Heaven/Creative): All yang [+++]
- KUN ☷ (Earth/Receptive): All yin [---]
- ZHEN ☳ (Thunder/Arousing): e+
- KAN ☵ (Water/Abysmal): f+  
- GEN ☶ (Mountain/Keeping Still): g+
- XUN ☴ (Wind/Gentle): e-
- LI ☲ (Fire/Clinging): f-, (or h+)
- DUI ☱ (Lake/Joyous): h+
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class Trigram(Enum):
    """I Ching trigrams with their properties."""
    QIAN = ("☰", "Heaven", "Creative", (1, 1, 1))     # All Yang
    KUN = ("☷", "Earth", "Receptive", (-1, -1, -1))   # All Yin
    ZHEN = ("☳", "Thunder", "Arousing", (-1, -1, 1))  # Yang at bottom
    KAN = ("☵", "Water", "Abysmal", (-1, 1, -1))      # Yang in middle
    GEN = ("☶", "Mountain", "Keeping Still", (1, -1, -1))  # Yang at top
    XUN = ("☴", "Wind", "Gentle", (1, 1, -1))         # Yin at bottom
    LI = ("☲", "Fire", "Clinging", (1, -1, 1))        # Yin in middle
    DUI = ("☱", "Lake", "Joyous", (-1, 1, 1))         # Yin at top
    
    @property
    def symbol(self) -> str:
        return self.value[0]
    
    @property
    def element(self) -> str:
        return self.value[1]
    
    @property
    def meaning(self) -> str:
        return self.value[2]
    
    @property
    def lines(self) -> Tuple[int, int, int]:
        """Bottom, middle, top lines (1=yang, -1=yin)."""
        return self.value[3]


@dataclass
class SemanticOctonion:
    """
    8D semantic encoding combining quaternion core with domain extensions.
    
    Core 4D (quaternion-like):
    - w: Witness/scalar (preserved at 1.0)
    - x: Yang-Yin axis
    - y: Becoming-Abiding axis
    - z: Ordinality/hierarchy axis
    
    Domain 4D (extensions):
    - e: Spatial domain
    - f: Temporal domain
    - g: Relational domain
    - h: Personal domain
    """
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0
    
    def __post_init__(self):
        """Ensure w=1.0 for semantic meaning."""
        if self.w != 1.0:
            self.w = 1.0
    
    @classmethod
    def from_arrays(cls, core: np.ndarray, domain: np.ndarray) -> 'SemanticOctonion':
        """Create from 4D core and 4D domain arrays."""
        return cls(
            w=1.0, x=core[0], y=core[1], z=core[2],
            e=domain[0], f=domain[1], g=domain[2], h=domain[3]
        )
    
    def core_array(self) -> np.ndarray:
        """Return core [x, y, z] as array."""
        return np.array([self.x, self.y, self.z])
    
    def domain_array(self) -> np.ndarray:
        """Return domain [e, f, g, h] as array."""
        return np.array([self.e, self.f, self.g, self.h])
    
    def full_array(self) -> np.ndarray:
        """Return all 8 components (excluding w)."""
        return np.array([self.x, self.y, self.z, self.e, self.f, self.g, self.h])
    
    def core_magnitude(self) -> float:
        """Magnitude of 3D core."""
        return np.linalg.norm(self.core_array())
    
    def domain_magnitude(self) -> float:
        """Magnitude of domain components."""
        return np.linalg.norm(self.domain_array())
    
    def total_magnitude(self) -> float:
        """Total 7D magnitude (excluding w)."""
        return np.linalg.norm(self.full_array())
    
    def normalize_core(self) -> 'SemanticOctonion':
        """Return copy with normalized core (||[x,y,z]|| = 1)."""
        mag = self.core_magnitude()
        if mag < 1e-10:
            return SemanticOctonion(w=1.0, x=0, y=0, z=0,
                                   e=self.e, f=self.f, g=self.g, h=self.h)
        return SemanticOctonion(
            w=1.0, x=self.x/mag, y=self.y/mag, z=self.z/mag,
            e=self.e, f=self.f, g=self.g, h=self.h
        )
    
    def dot_core(self, other: 'SemanticOctonion') -> float:
        """Dot product of core components."""
        return np.dot(self.core_array(), other.core_array())
    
    def dot_domain(self, other: 'SemanticOctonion') -> float:
        """Dot product of domain components."""
        return np.dot(self.domain_array(), other.domain_array())
    
    def dot_full(self, other: 'SemanticOctonion') -> float:
        """Dot product of all 7 components."""
        return np.dot(self.full_array(), other.full_array())
    
    def angle_core(self, other: 'SemanticOctonion') -> float:
        """Angle between cores in degrees."""
        v1, v2 = self.core_array(), other.core_array()
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    def angle_full(self, other: 'SemanticOctonion') -> float:
        """Angle in full 7D space."""
        v1, v2 = self.full_array(), other.full_array()
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    def to_trigram(self, x_axis: float = 0.0, y_axis: float = 0.0) -> Trigram:
        """
        Convert domain components to I Ching trigram.
        
        Uses dominant domain + yang/yin polarity:
        - S (Spatial, e) + Yang → QIAN (Heaven/Creative)
        - S (Spatial, e) + Yin  → KUN (Earth/Receptive)
        - T (Temporal, f) + Yang → ZHEN (Thunder/Arousing)
        - T (Temporal, f) + Yin  → XUN (Wind/Gentle)
        - R (Relational, g) + Yang → LI (Fire/Clinging)
        - R (Relational, g) + Yin  → KAN (Water/Abysmal)
        - P (Personal, h) + Yang → DUI (Lake/Joyous)
        - P (Personal, h) + Yin  → GEN (Mountain/Stillness)
        
        Args:
            x_axis: The x-component (yang/yin polarity) of the concept
            y_axis: The y-component (used as tiebreaker for neutral x)
        """
        # Find dominant domain
        domains = {'S': abs(self.e), 'T': abs(self.f), 'R': abs(self.g), 'P': abs(self.h)}
        total = sum(domains.values()) + 0.01
        dominant = max(domains, key=domains.get)
        
        # Determine yang/yin from x-axis (or y-axis if neutral)
        is_yang = x_axis > 0.2 or (abs(x_axis) <= 0.2 and y_axis > 0)
        
        # Mapping table
        mapping = {
            ('S', True):  Trigram.QIAN,   # Spatial + Yang = Heaven
            ('S', False): Trigram.KUN,    # Spatial + Yin = Earth
            ('T', True):  Trigram.ZHEN,   # Temporal + Yang = Thunder
            ('T', False): Trigram.XUN,    # Temporal + Yin = Wind
            ('R', True):  Trigram.LI,     # Relational + Yang = Fire
            ('R', False): Trigram.KAN,    # Relational + Yin = Water
            ('P', True):  Trigram.DUI,    # Personal + Yang = Lake
            ('P', False): Trigram.GEN,    # Personal + Yin = Mountain
        }
        
        return mapping[(dominant, is_yang)]
    
    def compose(self, other: 'SemanticOctonion') -> 'SemanticOctonion':
        """
        Compose two octonions using quaternion-like multiplication for core
        and weighted average for domains.
        """
        # Core: quaternion multiplication (simplified - just cross product for now)
        a = self.core_array()
        b = other.core_array()
        
        # For sentence composition: w_result = 1 - cos(theta)
        cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        w_result = 1.0 - cos_theta
        
        # Simple average for core
        c = (a + b) / 2
        
        # Domain: weighted average based on magnitudes
        d1, d2 = self.domain_array(), other.domain_array()
        m1, m2 = np.linalg.norm(d1) + 0.1, np.linalg.norm(d2) + 0.1
        d = (d1 * m1 + d2 * m2) / (m1 + m2)
        
        return SemanticOctonion(
            w=w_result, x=c[0], y=c[1], z=c[2],
            e=d[0], f=d[1], g=d[2], h=d[3]
        )
    
    def __repr__(self) -> str:
        return (f"O8[{self.x:+.2f},{self.y:+.2f},{self.z:+.2f} | "
                f"{self.e:+.2f},{self.f:+.2f},{self.g:+.2f},{self.h:+.2f}]")


@dataclass
class DualOctonion:
    """
    16D dual octonion: essence + εfunction
    
    Separates WHAT something IS (essence) from HOW it OPERATES (function).
    
    Example:
    - WATER essence: [cold, flowing, descending]
    - WATER function: [nurturing, cleansing, dissolving]
    """
    essence: SemanticOctonion
    function: SemanticOctonion
    
    def __init__(self, essence: SemanticOctonion = None, 
                 function: SemanticOctonion = None):
        self.essence = essence or SemanticOctonion()
        self.function = function or SemanticOctonion()
    
    @classmethod
    def from_concept(cls, x, y, z, e, f, g, h,
                    fx, fy, fz, fe, ff, fg, fh) -> 'DualOctonion':
        """Create from essence and function components."""
        return cls(
            essence=SemanticOctonion(w=1.0, x=x, y=y, z=z, e=e, f=f, g=g, h=h),
            function=SemanticOctonion(w=1.0, x=fx, y=fy, z=fz, e=fe, f=ff, g=fg, h=fh)
        )
    
    def full_array(self) -> np.ndarray:
        """Return all 14 components (excluding both w's)."""
        return np.concatenate([self.essence.full_array(), self.function.full_array()])
    
    def angle_essence(self, other: 'DualOctonion') -> float:
        """Angle between essences."""
        return self.essence.angle_full(other.essence)
    
    def angle_function(self, other: 'DualOctonion') -> float:
        """Angle between functions."""
        return self.function.angle_full(other.function)
    
    def angle_full(self, other: 'DualOctonion') -> float:
        """Angle in full 14D space."""
        v1, v2 = self.full_array(), other.full_array()
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    def compose(self, other: 'DualOctonion') -> 'DualOctonion':
        """
        Compose dual octonions.
        
        Essence composition: what the combined concept IS
        Function composition: how the combined concept OPERATES
        """
        return DualOctonion(
            essence=self.essence.compose(other.essence),
            function=self.function.compose(other.function)
        )
    
    def __repr__(self) -> str:
        return f"DO16[E:{self.essence} F:{self.function}]"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quaternion_multiply(q1: Tuple[float, float, float, float],
                       q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Standard quaternion multiplication.
    
    q = (w, x, y, z) = w + xi + yj + zk
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return (w, x, y, z)


def semantic_compose(verb: SemanticOctonion, 
                    object: SemanticOctonion) -> Tuple[SemanticOctonion, float]:
    """
    Compose verb with object semantically.
    
    Uses the formula: w = 1 - cos(θ)
    
    Returns:
        result: Composed semantic octonion
        witness: Witness preservation factor (1.0 = fully preserved)
    """
    # Get core vectors
    v1 = verb.core_array()
    v2 = object.core_array()
    
    m1 = np.linalg.norm(v1)
    m2 = np.linalg.norm(v2)
    
    if m1 < 1e-10 or m2 < 1e-10:
        # One is Unity - preserve the other
        witness = 1.0
        result_core = v1 + v2
    else:
        # Compute angle and witness
        cos_theta = np.dot(v1, v2) / (m1 * m2)
        witness = 1.0 - cos_theta  # At 90°: w=1, at 0°: w=0, at 180°: w=2
        
        # Cross product for result direction
        cross = np.cross(v1, v2)
        result_core = cross
    
    # Domain composition: weighted average
    d1, d2 = verb.domain_array(), object.domain_array()
    result_domain = (d1 + d2) / 2
    
    result = SemanticOctonion(
        w=witness,
        x=result_core[0], y=result_core[1], z=result_core[2],
        e=result_domain[0], f=result_domain[1], 
        g=result_domain[2], h=result_domain[3]
    )
    
    return result, witness


def trigram_from_components(e: float, f: float, g: float, h: float,
                           threshold: float = 0.3) -> Trigram:
    """
    Map domain components to I Ching trigram.
    
    Mapping:
    - Bottom line: e (spatial) > threshold → Yang
    - Middle line: f (temporal) > threshold → Yang  
    - Top line: (g+h) (relational+personal) > threshold → Yang
    """
    octo = SemanticOctonion(e=e, f=f, g=g, h=h)
    return octo.to_trigram(threshold)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OCTONION INFRASTRUCTURE TEST")
    print("=" * 60)
    
    # Create some test octonions
    fire = SemanticOctonion(x=0.7, y=-0.5, z=0.25, e=0.5, f=0.7, g=0.3, h=0.0)
    water = SemanticOctonion(x=-0.7, y=-0.5, z=0.35, e=0.6, f=0.5, g=0.0, h=0.3)
    
    print(f"\nFIRE: {fire}")
    print(f"WATER: {water}")
    print(f"\nCore angle: {fire.angle_core(water):.1f}°")
    print(f"Full angle: {fire.angle_full(water):.1f}°")
    
    print(f"\nFIRE trigram: {fire.to_trigram().symbol} {fire.to_trigram().name}")
    print(f"WATER trigram: {water.to_trigram().symbol} {water.to_trigram().name}")
    
    # Test composition
    print("\n" + "=" * 60)
    print("COMPOSITION TEST")
    print("=" * 60)
    
    hot = SemanticOctonion(x=0.7, y=0.7, z=-0.15, e=0.5, f=0.5, g=0.0, h=0.5)
    cold = SemanticOctonion(x=-0.7, y=0.7, z=-0.15, e=0.5, f=0.5, g=0.0, h=0.5)
    
    print(f"\nHOT: {hot}")
    print(f"COLD: {cold}")
    print(f"Angle: {hot.angle_core(cold):.1f}° (should be ~90°)")
    
    result, witness = semantic_compose(hot, water)
    print(f"\nHOT + WATER composition:")
    print(f"Result: {result}")
    print(f"Witness preservation: {witness:.3f}")
    
    # Trigram test
    print("\n" + "=" * 60)
    print("TRIGRAM DISTRIBUTION")
    print("=" * 60)
    
    for tri in Trigram:
        print(f"  {tri.symbol} {tri.name}: {tri.meaning} {tri.lines}")
