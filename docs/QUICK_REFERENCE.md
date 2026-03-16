# Geometric Semantics Quick Reference

## The 16D Encoding at a Glance

```
Concept = [w, x, y, z | e, f, g, h] + ε·[1, fx, fy, fz | fe, ff, fg, fh]
          └─────────────────────┘     └───────────────────────────────┘
               8D ESSENCE                      8D FUNCTION
           (what it IS)                    (how it OPERATES)
```

## Core Axes (x, y, z)

| Axis | + Direction | - Direction | Example Pairs |
|------|-------------|-------------|---------------|
| **x** | Yang/Active | Yin/Passive | CREATE/DESTROY, GIVE/TAKE |
| **y** | Becoming/Process | Abiding/State | FLOW/STILLNESS, CHANGE/STABLE |
| **z** | Ascending/Prior | Descending/Post | UP/DOWN, BEGIN/END |

## Domain Axes (e, f, g, h)

| Axis | Domain | High Values | Low Values |
|------|--------|-------------|------------|
| **e** | Spatial | body, place, physical | mental, abstract |
| **f** | Temporal | time, duration, process | eternal, atemporal |
| **g** | Relational | social, meaning, connection | isolated, private |
| **h** | Personal | emotion, subjective, inner | objective, outer |

## Trigram Mapping

| Trigram | Symbol | Element | Domain+Polarity |
|---------|--------|---------|-----------------|
| QIAN | ☰ | Heaven | e+ x+ (Spatial-Yang) |
| KUN | ☷ | Earth | e+ x- (Spatial-Yin) |
| ZHEN | ☳ | Thunder | f+ x+ (Temporal-Yang) |
| XUN | ☴ | Wind | f+ x- (Temporal-Yin) |
| LI | ☲ | Fire | g+ x+ (Relational-Yang) |
| KAN | ☵ | Water | g+ x- (Relational-Yin) |
| DUI | ☱ | Lake | h+ x+ (Personal-Yang) |
| GEN | ☶ | Mountain | h+ x- (Personal-Yin) |

## Key Formulas

### Witness Preservation
```
w = 1 - cos(θ)

θ=0°:   w=0   → DISSOLUTION (attachment)
θ=90°:  w=1   → PRESERVATION (healthy)
θ=180°: w=2   → TENSION (conflict)
```

### Complementarity
```
Complements are ORTHOGONAL (90°), not opposite (180°)

HOT ⊥ COLD  (they complete, not negate)
LIGHT ⊥ DARK
YANG ⊥ YIN
```

### Validation Thresholds
```
Complement pair: 80° ≤ core angle ≤ 105°
Affinity pair:   15° ≤ core angle ≤ 45°
Synonym:         core angle < 15°
```

## Ontological Levels

| Level | Number | Description | Examples |
|-------|--------|-------------|----------|
| UNITY | 0 | Undifferentiated | BEING, IS, ONE, TAO |
| DYAD | 1 | First distinction | THIS/THAT, YANG/YIN |
| TRIAD | 2 | Relationship | BECOMING, PROCESS |
| TETRAD | 3 | Elements | FIRE, WATER, AIR, EARTH |
| QUALITY | 4 | Qualities | HOT, COLD, LIGHT, DARK |
| DERIVED | 5 | Derived | (most concepts) |
| VERB | 6 | Actions | GIVE, TAKE, CREATE |
| ABSTRACT | 7 | Abstract | TRUTH, BEAUTY |
| INTERROGATIVE | 8 | Questions | WHAT, WHY, HOW |

## Essence-Function Categories

| Category | Angle | Meaning | Examples |
|----------|-------|---------|----------|
| ALIGNED | <30° | IS = DOES | GIVE, CREATE, LOVE |
| OBLIQUE | 30-60° | Partial alignment | THIS, SELF |
| PERPENDICULAR | 60-90° | Orthogonal operation | FIRE, DARK |
| REVERSED | >90° | DOES opposes IS | DOWN, END, HIDDEN |

## Quick Python Usage

```python
from semantic_api import SemanticCore

core = SemanticCore()

# Get concept
water = core.get("WATER")
print(water.trigram())  # ☵ KAN

# Find angle
print(core.angle("HOT", "COLD"))  # ~90°

# Compose
result = core.compose("I", "LOVE", "WATER")
print(result.witness)  # ~1.0 (preserved)

# Find neighbors
neighbors = core.nearest_neighbors("LOVE", n=5)
```

## The Consciousness Circuit

```
I AM     [1,0,0,0]  → Pure witness
I HAVE   [1,x,0,0]  → First distinction (content)
I CAN    [1,x,y,0]  → Possibilities (binding)
I DO     [1,x,y,z]  → Action (choice)
I CANNOT [limit]    → Constraint → back to I AM
```

## Current Statistics

- **1,316** unique concepts
- **3,264** semantic relations
- **983** complement pairs (100% validated)
- **2,010** affinity relations
- **100** development sessions

---

*"From [1,0,0,0], all meaning unfolds through distinction."*
