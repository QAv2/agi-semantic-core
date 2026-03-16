"""
AGI Integration Examples
=========================

Demonstrates how the geometric semantic core can serve as an
"unconscious" foundation for various AGI capabilities.

Examples:
1. Symbol Grounding - Concepts have intrinsic geometric meaning
2. Semantic Coherence - Validate statements mathematically
3. Analogical Reasoning - Navigate semantic space geometrically
4. Compositional Semantics - Build sentence meaning algebraically
5. Emotional Valence - Detect affective content via domain axes
"""

from semantic_api import SemanticCore
from composition_engine import CompositionEngine
import numpy as np


def example_symbol_grounding():
    """
    SYMBOL GROUNDING
    
    Unlike statistical embeddings, geometric semantics provides
    intrinsic meaning based on consciousness differentiation.
    
    Each concept's position derives from:
    - Its place in the ontological tree (Unity → Dyad → Triad → ...)
    - Its polarity on semantic axes (Yang/Yin, Becoming/Abiding, etc.)
    - Its domain profile (Spatial, Temporal, Relational, Personal)
    """
    print("=" * 60)
    print("1. SYMBOL GROUNDING")
    print("=" * 60)
    
    core = SemanticCore()
    
    # Compare water across its semantic profile
    water = core.describe("WATER")
    print("\nWATER is intrinsically grounded:")
    print(f"  Trigram: {water['trigram']} (Abysmal - depth, danger, hidden meaning)")
    print(f"  Core polarity: {water['core']}")
    print(f"    x=-0.68 → Yin (passive, receptive, flowing)")
    print(f"    y=-0.55 → Abiding (state, depth, stillness within)")
    print(f"    z=+0.35 → Slightly ascending (rises to its level)")
    print(f"  Domain: {water['domain_profile']}")
    print(f"  Complements: {water['complements']}")
    
    # Show how grounding enables inference
    print("\nGrounding enables inference:")
    similar = core.nearest_neighbors("WATER", n=5)
    print(f"  WATER's nearest concepts: {[n for n, _ in similar]}")
    
    # The geometric meaning is not arbitrary
    fire = core.describe("FIRE")
    print(f"\nFIRE as WATER's complement:")
    print(f"  Trigram: {fire['trigram']} (Clinging - clarity, consciousness)")
    print(f"  Core polarity: {fire['core']}")
    print(f"    x=+0.70 → Yang (active, consuming, transforming)")
    print(f"    y=-0.50 → Abiding (maintains its nature)")
    print(f"  Angle to WATER: {core.angle('FIRE', 'WATER', core_only=True):.1f}° (complement)")


def example_semantic_coherence():
    """
    SEMANTIC COHERENCE CHECKING
    
    The witness preservation formula mathematically encodes
    whether statements are coherent, contradictory, or tense.
    
    w = 1 - cos(θ)
    - w ≈ 0: Dissolution (identity collapse, attachment)
    - w ≈ 1: Preservation (healthy composition)
    - w ≈ 2: Tension (conflict, opposition)
    """
    print("\n" + "=" * 60)
    print("2. SEMANTIC COHERENCE")
    print("=" * 60)
    
    engine = CompositionEngine()
    
    test_statements = [
        "I LOVE WATER",       # Coherent
        "I LOVE HATE",        # Tension (complements composed)
        "HOT COLD",           # Complementary (orthogonal)
        "LOVE LOVE",          # Dissolution (self-attachment)
        "I AM BEING",         # Near-identity (spiritual truth)
        "FIRE WATER EARTH",   # Multi-element (tension?)
    ]
    
    print("\nCoherence Analysis:")
    for stmt in test_statements:
        coh = engine.sentence_coherence(stmt)
        result = engine.compose_sentence(stmt)
        
        status = "✓ COHERENT" if coh['coherent'] else "✗ INCOHERENT"
        print(f"\n  '{stmt}'")
        print(f"    {status} (score: {coh['score']:.2f})")
        
        if coh['issues']:
            print(f"    Issues: {'; '.join(coh['issues'])}")
        
        # Interpret the score
        if coh['score'] < 0.3:
            print(f"    → Identity collapse: concepts too similar")
        elif coh['score'] > 1.5:
            print(f"    → Semantic tension: concepts in conflict")
        else:
            print(f"    → Healthy composition: witness preserved")


def example_analogical_reasoning():
    """
    ANALOGICAL REASONING
    
    Geometric semantics enables analogy as vector arithmetic:
    - A is to B as C is to ?
    - Solved by: ? = C + (B - A)
    
    This works because relationships are encoded as
    angular distances and vector differences.
    """
    print("\n" + "=" * 60)
    print("3. ANALOGICAL REASONING")
    print("=" * 60)
    
    core = SemanticCore()
    
    def analogy(a: str, b: str, c: str, n: int = 5):
        """A is to B as C is to ?"""
        va = core.vector(a)
        vb = core.vector(b)
        vc = core.vector(c)
        
        if va is None or vb is None or vc is None:
            return []
        
        # Vector arithmetic: ? = C + (B - A)
        target = vc + (vb - va)
        
        # Find nearest concepts to target
        return core.search_by_vector(target, n=n)
    
    analogies = [
        ("HOT", "COLD", "LIGHT"),      # HOT:COLD :: LIGHT:?
        ("LOVE", "HATE", "GIVE"),       # LOVE:HATE :: GIVE:?
        ("FIRE", "WATER", "UP"),        # FIRE:WATER :: UP:?
        ("I", "SELF", "YOU"),           # I:SELF :: YOU:?
        ("BEGIN", "END", "BIRTH"),      # BEGIN:END :: BIRTH:?
    ]
    
    print("\nAnalogical Reasoning (A:B :: C:?):")
    for a, b, c in analogies:
        results = analogy(a, b, c, n=3)
        answers = [r[0] for r in results]
        print(f"\n  {a} : {b} :: {c} : ?")
        print(f"    Top candidates: {answers}")
        
        # Verify the relationship
        ab_angle = core.angle(a, b, core_only=True)
        if results:
            cd_angle = core.angle(c, results[0][0], core_only=True)
            print(f"    {a}-{b} angle: {ab_angle:.1f}°, {c}-{results[0][0]} angle: {cd_angle:.1f}°")


def example_compositional_semantics():
    """
    COMPOSITIONAL SEMANTICS
    
    Complex meaning emerges from geometric composition:
    - SLERP for smooth blending
    - PRODUCT for ordered composition
    - Different structures yield different meanings
    """
    print("\n" + "=" * 60)
    print("4. COMPOSITIONAL SEMANTICS")
    print("=" * 60)
    
    engine = CompositionEngine()
    core = SemanticCore()
    
    # Demonstrate order matters
    print("\nOrder Sensitivity (quaternion product):")
    
    result1 = engine.compose_sequential(["HOT", "WATER"])
    result2 = engine.compose_sequential(["WATER", "HOT"])
    
    print(f"\n  'HOT WATER' (hot modifies water):")
    print(f"    Core: [{', '.join(f'{v:.2f}' for v in result1.core)}]")
    
    print(f"\n  'WATER HOT' (water modifies hot):")
    print(f"    Core: [{', '.join(f'{v:.2f}' for v in result2.core)}]")
    
    # Compare cores
    diff = np.linalg.norm(result1.core - result2.core)
    print(f"\n  Difference: {diff:.3f} (order changes meaning)")
    
    # Demonstrate blending vs composition
    print("\n\nBlending vs Sequential:")
    
    words = ["LOVE", "FIRE", "WATER"]
    
    seq_result = engine.compose_sentence(' '.join(words), mode='sequential')
    blend_result = engine.compose_sentence(' '.join(words), mode='blended')
    avg_result = engine.compose_sentence(' '.join(words), mode='average')
    
    print(f"\n  '{' '.join(words)}':")
    print(f"    Sequential: core=[{', '.join(f'{v:.2f}' for v in seq_result.core)}], w={seq_result.witness:.2f}")
    print(f"    Blended:    core=[{', '.join(f'{v:.2f}' for v in blend_result.core)}], w={blend_result.witness:.2f}")
    print(f"    Average:    core=[{', '.join(f'{v:.2f}' for v in avg_result.core)}], w={avg_result.witness:.2f}")
    
    # Find what the composite meanings are closest to
    seq_nearest = core.search_by_vector(np.concatenate([seq_result.core, [0,0,0,0]]))
    print(f"\n    Sequential result nearest to: {seq_nearest[0][0]}")


def example_emotional_valence():
    """
    EMOTIONAL VALENCE
    
    The personal domain (h-axis) and core polarity encode
    affective content, enabling emotion detection.
    """
    print("\n" + "=" * 60)
    print("5. EMOTIONAL VALENCE")
    print("=" * 60)
    
    core = SemanticCore()
    
    def analyze_emotion(name: str):
        """Analyze emotional content of a concept."""
        c = core.get(name)
        if not c:
            return None
        
        # Valence from x-axis (yang=positive, yin=negative)
        valence = "POSITIVE" if c.x > 0.1 else "NEGATIVE" if c.x < -0.1 else "NEUTRAL"
        
        # Arousal from domain magnitude
        arousal = np.sqrt(c.e**2 + c.f**2 + c.g**2 + c.h**2)
        arousal_level = "HIGH" if arousal > 0.7 else "LOW" if arousal < 0.3 else "MEDIUM"
        
        # Personal intensity from h-axis
        personal = "INTENSE" if abs(c.h) > 0.3 else "MILD"
        
        return {
            'valence': valence,
            'valence_score': c.x,
            'arousal': arousal_level,
            'arousal_score': arousal,
            'personal': personal,
            'personal_score': c.h,
            'trigram': c.trigram().symbol
        }
    
    emotions = [
        "LOVE", "HATE", "JOY", "FEAR", "ANGER",
        "PEACE", "DESIRE", "HOPE", "DESPAIR", "CALM"
    ]
    
    print("\nEmotional Analysis:")
    print(f"{'Concept':10} {'Valence':10} {'Arousal':8} {'Personal':10} {'Trigram':6}")
    print("-" * 50)
    
    for name in emotions:
        e = analyze_emotion(name)
        if e:
            print(f"{name:10} {e['valence']:10} {e['arousal']:8} {e['personal']:10} {e['trigram']:6}")
    
    # Emotional transition analysis
    print("\n\nEmotional Transitions (witness = change difficulty):")
    transitions = [
        ("ANGER", "PEACE"),
        ("FEAR", "COURAGE"),
        ("DESPAIR", "HOPE"),
        ("LOVE", "HATE"),
    ]
    
    engine = CompositionEngine()
    for start, end in transitions:
        analysis = engine.analyze_pair(start, end)
        witness = analysis.get('witness', 0)
        
        # Interpret witness as transition difficulty
        if witness < 0.5:
            difficulty = "EASY (similar states)"
        elif witness < 1.5:
            difficulty = "MODERATE (complement)"
        else:
            difficulty = "HARD (opposition)"
        
        print(f"  {start} → {end}: w={witness:.2f} ({difficulty})")


def example_concept_creation():
    """
    CONCEPT CREATION
    
    New concepts can be created by geometric operations
    on existing concepts, enabling semantic inference.
    """
    print("\n" + "=" * 60)
    print("6. CONCEPT CREATION (Semantic Inference)")
    print("=" * 60)
    
    core = SemanticCore()
    engine = CompositionEngine()
    
    # Create compound concepts
    compounds = [
        ("FIRE", "WATER", "steam-like"),      # Tension creates transformation
        ("LOVE", "TRUTH", "authentic love"),   # Composition
        ("SELF", "OTHER", "relationship"),     # Complement synthesis
        ("UP", "DOWN", "vertical"),            # Dimensional concept
    ]
    
    print("\nCompound Concept Creation:")
    for c1, c2, expected in compounds:
        result = engine.slerp(
            engine.get_frame(c1),
            engine.get_frame(c2),
            t=0.5
        )
        
        if result:
            # Find what this new concept is closest to
            nearest = core.search_by_vector(
                np.concatenate([result.core, result.domain])
            )
            
            print(f"\n  {c1} ⊕ {c2} (blend):")
            print(f"    Expected meaning: {expected}")
            print(f"    Core: [{', '.join(f'{v:.2f}' for v in result.core)}]")
            print(f"    Nearest existing: {nearest[0][0]} ({nearest[0][1]:.1f}°)")
            
            # Also show the product (ordered composition)
            product = engine.product(
                engine.get_frame(c1),
                engine.get_frame(c2)
            )
            product_nearest = core.search_by_vector(
                np.concatenate([product.core, product.domain])
            )
            print(f"    Product ({c1} × {c2}) nearest: {product_nearest[0][0]}")


def run_all_examples():
    """Run all integration examples."""
    example_symbol_grounding()
    example_semantic_coherence()
    example_analogical_reasoning()
    example_compositional_semantics()
    example_emotional_valence()
    example_concept_creation()
    
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print("""
The geometric semantic core provides:

1. SYMBOL GROUNDING - Intrinsic, non-arbitrary meaning
2. COHERENCE CHECKING - Mathematical validity of statements
3. ANALOGICAL REASONING - Vector arithmetic for inference
4. COMPOSITIONAL SEMANTICS - Algebraic meaning combination
5. EMOTIONAL VALENCE - Affective content from domain axes
6. CONCEPT CREATION - Geometric inference of new meanings

These capabilities emerge from the consciousness-structured
encoding, not from statistical patterns in text.

"From [1,0,0,0], all meaning unfolds through distinction."
""")


if __name__ == "__main__":
    run_all_examples()
