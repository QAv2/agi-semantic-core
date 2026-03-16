# Mathematical Structures for Semantic Encoding

## Overview

This document describes the mathematical structures available for encoding semantic meaning geometrically.

## 1. Quaternions (4D) â€” Primary Encoding

### Structure
```
Q = [w, x, y, z] = w + xi + yj + zk

Where:
- w: scalar (witness/existence component)
- x, y, z: vector (content components)
- iÂ² = jÂ² = kÂ² = ijk = -1
```

### Properties
- **Associative**: (AB)C = A(BC) âœ“
- **Non-commutative**: AB â‰  BA (order matters!)
- **Division algebra**: Every non-zero quaternion has an inverse
- **Norm**: ||Q|| = âˆš(wÂ² + xÂ² + yÂ² + zÂ²)

### Semantic Interpretation

| Component | Semantic Meaning |
|-----------|------------------|
| w | Witness/existence intensity |
| x | Yang-Yin axis (active-passive, expansive-contractive) |
| y | External-Internal axis (manifest-hidden, surface-depth) |
| z | Process-State axis (dynamic-static, becoming-being) |

### Operations

**Multiplication** (composition):
```
Qâ‚ Ã— Qâ‚‚ = [wâ‚wâ‚‚ - vâ‚Â·vâ‚‚, wâ‚vâ‚‚ + wâ‚‚vâ‚ + vâ‚Ã—vâ‚‚]
```
Non-commutative: "hot water" â‰  "water hot" (different emphasis)

**Distance** (semantic difference):
```
d(Qâ‚, Qâ‚‚) = ||Qâ‚ - Qâ‚‚||
```

**Angle** (relationship type):
```
cos(Î¸) = (Qâ‚Â·Qâ‚‚) / (||Qâ‚|| ||Qâ‚‚||)
```
- Î¸ â‰ˆ 0Â°: synonyms
- Î¸ â‰ˆ 90Â°: complements
- Î¸ â‰ˆ 180Â°: opposites (rare in semantics)

**SLERP** (interpolation):
```
slerp(Qâ‚, Qâ‚‚, t) = smooth path between meanings
```

## 2. Clifford Algebra Cl(n) â€” Extended Structure

### Why Clifford?
- **Associative** (unlike octonions)
- **Arbitrary dimension** (unlike Hurwitz algebras)
- **Grade structure** (scalars, vectors, bivectors, etc.)
- **Matches I Ching**: Cl(6) = 64 dimensions

### Cl(6) for Hexagram Encoding

Dimension: 2â¶ = 64

Grade structure:
```
Grade 0: 1 scalar (Unity)
Grade 1: 6 vectors (6 lines of hexagram)
Grade 2: 15 bivectors (line pairs)
Grade 3: 20 trivectors (line triplets)
Grade 4: 15 4-vectors
Grade 5: 6 5-vectors
Grade 6: 1 pseudoscalar (complete hexagram)
```

Each hexagram = specific multivector in Cl(6)

### Geometric Product
```
ab = aÂ·b + aâˆ§b
   = (symmetric inner) + (antisymmetric outer)
```

The product naturally decomposes into:
- How much concepts overlap (aÂ·b)
- What new relationship they create (aâˆ§b)

## 3. Encoding Conventions

### Normalization
All concept vectors are normalized: ||[x,y,z]|| = 1

This ensures:
- Distance comparisons are meaningful
- Angles are well-defined
- No concept has "more existence" than another

### Unity at Origin
[1,0,0,0] = BEING/IS/ONE = Pure witness, undifferentiated

All concepts are displacements from Unity.

### Axis Semantics (Quaternion)

**X-axis (i)**: Yang-Yin polarity
- +x: Yang, active, expansive, creative, bright
- -x: Yin, passive, contractive, receptive, dark

**Y-axis (j)**: External-Internal polarity
- +y: External, manifest, surface, public
- -y: Internal, hidden, depth, private

**Z-axis (k)**: Process-State polarity
- +z: Process, dynamic, becoming, changing
- -z: State, static, being, stable

### Complementary Pairs

For complements A and B:
```
AÂ·B â‰ˆ 0 (orthogonal)
Î¸(A,B) â‰ˆ 90Â°
```

Example encoding:
```
HOT:  [1, +0.7, +0.7, 0.0]
COLD: [1, -0.7, +0.7, 0.0]

Dot product: 0.7Ã—(-0.7) + 0.7Ã—0.7 + 0Ã—0 = -0.49 + 0.49 = 0 âœ“
```

They differ on Yang-Yin but share External-Internal and Process-State.

## 4. Relationship Types

### Identity (Î¸ â‰ˆ 0Â°)
- Synonyms: big/large
- Translations: water/agua/ë¬¼
- d â‰ˆ 0

### Complementarity (Î¸ â‰ˆ 90Â°)
- Hot/Cold, Light/Dark, Yang/Yin
- d â‰ˆ âˆš2
- Orthogonal in semantic space

### Golden Ratio (Î¸ â‰ˆ 37Â°)
- Part/Whole relationships
- Specific/General (dog/animal)
- Ï† â‰ˆ 1.618 appears naturally

### Octant (Î¸ â‰ˆ 45Â°)
- "Related but distinct"
- Partial overlap
- Neither identical nor complementary

## 5. Composition Rules

### Quaternion Product (for phrases)
```
"hot water" = Q(hot) Ã— Q(water)
"water hot" = Q(water) Ã— Q(hot) â‰  "hot water"
```

Non-commutativity captures modifier-head asymmetry.

### Clifford Product (for complex compositions)
```
Hexagram = Trigramâ‚ âˆ§ Trigramâ‚‚
```

Outer product creates higher-grade structure.

### Addition (for blending)
```
"lukewarm" â‰ˆ 0.5Ã—Q(hot) + 0.5Ã—Q(cold)
```

Linear combination for intermediate concepts.

## 6. Validation Checks

For any encoding, verify:

1. **Complementary pairs**: Î¸ within 80-100Â°
2. **Synonyms**: Î¸ < 15Â°
3. **Hierarchies**: child "between" Unity and parent
4. **Compositions**: products give sensible results
5. **Distances**: scale appropriately with semantic difference

## 7. Implementation Notes

### Python Quaternion
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SemanticQuaternion:
    w: float  # witness
    x: float  # yang-yin
    y: float  # external-internal
    z: float  # process-state
    
    def normalize(self):
        mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag > 0:
            return SemanticQuaternion(
                self.w, self.x/mag, self.y/mag, self.z/mag
            )
        return self
    
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def angle_degrees(self, other):
        cos_theta = self.dot(other)
        return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

### Verification Function
```python
def verify_complement(q1, q2, name1, name2):
    angle = q1.angle_degrees(q2)
    status = "âœ“" if 80 <= angle <= 100 else "âœ—"
    print(f"{name1}/{name2}: {angle:.1f}Â° {status}")
```

---

## Summary Table

| Structure | Dimension | Associative | Best For |
|-----------|-----------|-------------|----------|
| Quaternion | 4 | âœ“ | Basic concept encoding |
| Octonion | 8 | âœ— | Trigram structure |
| Cl(3) | 8 | âœ“ | â‰… Quaternions |
| Cl(6) | 64 | âœ“ | Hexagram composition |
| Cl(8) | 256 | âœ“ | IfÃ¡ / deep structure |

**Recommendation**: Use quaternions for core encoding, Clifford Cl(6) for compositional operations.
