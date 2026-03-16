"""
Semantic Composition Engine
============================

Computes meaning of phrases and sentences through geometric operations.

Core Principles:
1. Meaning combines via quaternion-like multiplication
2. Witness preservation measures coherence
3. Complementary concepts (90°) preserve meaning
4. Identical concepts (0°) cause witness dissolution

Operations:
- SLERP: Spherical linear interpolation (smooth blending)
- PRODUCT: Quaternion-style multiplication (order matters)
- AVERAGE: Simple weighted mean (commutative)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from extended_dictionary import ExtendedDictionary, ExtendedConcept
from octonion import SemanticOctonion, quaternion_multiply


@dataclass
class SemanticFrame:
    """
    A semantic representation in the geometric space.
    
    Can represent a concept, phrase, or sentence.
    """
    core: np.ndarray          # [x, y, z] semantic polarity
    domain: np.ndarray        # [e, f, g, h] semantic field
    witness: float            # Preservation factor (0-2)
    source: str               # Description of what this represents
    
    @classmethod
    def from_concept(cls, concept: ExtendedConcept) -> 'SemanticFrame':
        """Create frame from a dictionary concept."""
        return cls(
            core=np.array([concept.x, concept.y, concept.z]),
            domain=np.array([concept.e, concept.f, concept.g, concept.h]),
            witness=1.0,
            source=concept.name
        )
    
    @property
    def full_vector(self) -> np.ndarray:
        """Combined 7D vector."""
        return np.concatenate([self.core, self.domain])
    
    @property
    def quaternion(self) -> Tuple[float, float, float, float]:
        """As quaternion [w, x, y, z]."""
        return (self.witness, self.core[0], self.core[1], self.core[2])
    
    def normalize_core(self) -> 'SemanticFrame':
        """Return copy with normalized core."""
        mag = np.linalg.norm(self.core)
        if mag < 1e-10:
            return SemanticFrame(
                core=np.array([0.0, 0.0, 0.0]),
                domain=self.domain.copy(),
                witness=self.witness,
                source=self.source
            )
        return SemanticFrame(
            core=self.core / mag,
            domain=self.domain.copy(),
            witness=self.witness,
            source=self.source
        )
    
    def angle_to(self, other: 'SemanticFrame', core_only: bool = False) -> float:
        """Angle to another frame in degrees."""
        if core_only:
            v1, v2 = self.core, other.core
        else:
            v1, v2 = self.full_vector, other.full_vector
        
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        
        cos_theta = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    def __repr__(self) -> str:
        return f"Frame[{self.source}, w={self.witness:.2f}]"


class CompositionEngine:
    """
    Engine for composing semantic frames into complex meanings.
    
    Supports multiple composition strategies:
    - SEQUENTIAL: Apply concepts left-to-right (order matters)
    - HIERARCHICAL: Subject-verb-object structure
    - BLENDING: Smooth interpolation (SLERP)
    """
    
    def __init__(self, dictionary: Optional[ExtendedDictionary] = None):
        """Initialize with dictionary."""
        self.dictionary = dictionary or ExtendedDictionary()
    
    def get_frame(self, name: str) -> Optional[SemanticFrame]:
        """Get semantic frame for a concept."""
        concept = self.dictionary.concepts.get(name.upper())
        if concept:
            return SemanticFrame.from_concept(concept)
        return None
    
    # =========================================================================
    # WITNESS PRESERVATION
    # =========================================================================
    
    def witness_between(self, frame1: SemanticFrame, frame2: SemanticFrame) -> float:
        """
        Compute witness preservation between two frames.
        
        Formula: w = 1 - cos(θ)
        
        Results:
            θ=0°:   w=0   → Complete dissolution (attachment/fusion)
            θ=90°:  w=1   → Full preservation (healthy complement)
            θ=180°: w=2   → Maximum tension (opposition)
        """
        # Use core vectors only
        v1, v2 = frame1.core, frame2.core
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if m1 < 1e-10 or m2 < 1e-10:
            return 1.0  # Unity preserves
        
        cos_theta = np.dot(v1, v2) / (m1 * m2)
        return 1.0 - cos_theta
    
    # =========================================================================
    # COMPOSITION OPERATIONS
    # =========================================================================
    
    def slerp(self, frame1: SemanticFrame, frame2: SemanticFrame, 
              t: float = 0.5) -> SemanticFrame:
        """
        Spherical linear interpolation between frames.
        
        Creates smooth blending:
        - t=0: Pure frame1
        - t=0.5: Midpoint
        - t=1: Pure frame2
        
        Best for: Conceptual blending, metaphor, gradual transitions
        """
        # Normalize cores
        v1, v2 = frame1.core.copy(), frame2.core.copy()
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if m1 < 1e-10:
            v1_norm = np.array([0.0, 0.0, 0.0])
        else:
            v1_norm = v1 / m1
        
        if m2 < 1e-10:
            v2_norm = np.array([0.0, 0.0, 0.0])
        else:
            v2_norm = v2 / m2
        
        # Compute angle
        cos_omega = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        omega = np.arccos(cos_omega)
        
        if omega < 1e-10:
            # Nearly identical - linear interpolation
            core = (1 - t) * v1 + t * v2
        else:
            # SLERP
            sin_omega = np.sin(omega)
            core = (np.sin((1 - t) * omega) / sin_omega) * v1 + \
                   (np.sin(t * omega) / sin_omega) * v2
        
        # Linear interpolation for domain
        domain = (1 - t) * frame1.domain + t * frame2.domain
        
        # Witness: average, weighted by t
        witness = (1 - t) * frame1.witness + t * frame2.witness
        
        return SemanticFrame(
            core=core,
            domain=domain,
            witness=witness,
            source=f"slerp({frame1.source}, {frame2.source}, {t:.2f})"
        )
    
    def product(self, frame1: SemanticFrame, frame2: SemanticFrame) -> SemanticFrame:
        """
        Quaternion-like product of frames.
        
        Non-commutative: product(A, B) ≠ product(B, A)
        
        Models how modifier + head combine:
        - "HOT WATER" ≠ "WATER HOT"
        - First frame modifies second
        
        Best for: Adjective-noun, verb-object, ordered composition
        """
        # Quaternion multiplication of cores
        q1 = (frame1.witness, frame1.core[0], frame1.core[1], frame1.core[2])
        q2 = (frame2.witness, frame2.core[0], frame2.core[1], frame2.core[2])
        
        w, x, y, z = quaternion_multiply(q1, q2)
        
        # Domain: weighted average (frame2 dominates as "head")
        domain = 0.3 * frame1.domain + 0.7 * frame2.domain
        
        # Compute witness between original frames
        pairwise_witness = self.witness_between(frame1, frame2)
        
        return SemanticFrame(
            core=np.array([x, y, z]),
            domain=domain,
            witness=pairwise_witness,
            source=f"({frame1.source} × {frame2.source})"
        )
    
    def average(self, *frames: SemanticFrame) -> SemanticFrame:
        """
        Simple average of frames.
        
        Commutative: order doesn't matter.
        
        Best for: Bag-of-words, topic summarization
        """
        if not frames:
            return SemanticFrame(
                core=np.zeros(3),
                domain=np.zeros(4),
                witness=1.0,
                source="empty"
            )
        
        core_sum = np.zeros(3)
        domain_sum = np.zeros(4)
        witness_sum = 0.0
        
        for f in frames:
            core_sum += f.core
            domain_sum += f.domain
            witness_sum += f.witness
        
        n = len(frames)
        
        return SemanticFrame(
            core=core_sum / n,
            domain=domain_sum / n,
            witness=witness_sum / n,
            source=f"avg({', '.join(f.source for f in frames)})"
        )
    
    # =========================================================================
    # SENTENCE COMPOSITION
    # =========================================================================
    
    def compose_sequential(self, names: List[str]) -> Optional[SemanticFrame]:
        """
        Compose concepts sequentially (left-to-right product).
        
        Example: "I LOVE WATER" → ((I × LOVE) × WATER)
        """
        frames = []
        for name in names:
            frame = self.get_frame(name)
            if frame:
                frames.append(frame)
        
        if not frames:
            return None
        
        result = frames[0]
        for frame in frames[1:]:
            result = self.product(result, frame)
        
        return result
    
    def compose_svo(self, subject: str, verb: str, obj: str) -> Optional[SemanticFrame]:
        """
        Subject-Verb-Object composition.
        
        Structure: (SUBJECT × (VERB × OBJECT))
        
        The verb-object relation is computed first, then subject relates to it.
        """
        s = self.get_frame(subject)
        v = self.get_frame(verb)
        o = self.get_frame(obj)
        
        if not all([s, v, o]):
            return None
        
        # Verb-object first
        vo = self.product(v, o)
        
        # Subject to verb-object
        result = self.product(s, vo)
        result.source = f"SVO({subject}, {verb}, {obj})"
        
        return result
    
    def compose_blended(self, names: List[str]) -> Optional[SemanticFrame]:
        """
        Blend all concepts equally using SLERP chain.
        
        Produces smooth interpolation through semantic space.
        """
        frames = []
        for name in names:
            frame = self.get_frame(name)
            if frame:
                frames.append(frame)
        
        if not frames:
            return None
        
        if len(frames) == 1:
            return frames[0]
        
        # Pairwise SLERP with t=0.5
        result = frames[0]
        for frame in frames[1:]:
            result = self.slerp(result, frame, 0.5)
        
        return result
    
    def compose_sentence(self, sentence: str, 
                        mode: str = 'sequential') -> Optional[SemanticFrame]:
        """
        Compose a sentence (space-separated concepts).
        
        Args:
            sentence: Space-separated concept names
            mode: 'sequential', 'average', or 'blended'
        """
        words = sentence.upper().split()
        
        if mode == 'sequential':
            return self.compose_sequential(words)
        elif mode == 'average':
            frames = [f for f in [self.get_frame(w) for w in words] if f]
            return self.average(*frames) if frames else None
        elif mode == 'blended':
            return self.compose_blended(words)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    
    def analyze_pair(self, name1: str, name2: str) -> Dict:
        """Analyze the semantic relationship between two concepts."""
        f1 = self.get_frame(name1)
        f2 = self.get_frame(name2)
        
        if not f1 or not f2:
            return {}
        
        core_angle = f1.angle_to(f2, core_only=True)
        full_angle = f1.angle_to(f2, core_only=False)
        witness = self.witness_between(f1, f2)
        
        # Categorize relationship
        if core_angle < 15:
            relation = "SYNONYM"
        elif core_angle < 45:
            relation = "AFFINITY"
        elif core_angle < 75:
            relation = "ADJACENT"
        elif core_angle < 105:
            relation = "COMPLEMENT"
        else:
            relation = "OPPOSITION"
        
        # Witness interpretation
        if witness < 0.3:
            witness_meaning = "dissolution"
        elif witness < 0.7:
            witness_meaning = "partial_preservation"
        elif witness < 1.3:
            witness_meaning = "full_preservation"
        else:
            witness_meaning = "tension"
        
        return {
            'concept1': name1,
            'concept2': name2,
            'core_angle': core_angle,
            'full_angle': full_angle,
            'witness': witness,
            'relation': relation,
            'witness_meaning': witness_meaning
        }
    
    def sentence_coherence(self, sentence: str) -> Dict:
        """Analyze coherence of a sentence."""
        words = sentence.upper().split()
        frames = [f for f in [self.get_frame(w) for w in words] if f]
        
        if len(frames) < 2:
            return {'coherent': True, 'score': 1.0, 'issues': []}
        
        # Check pairwise witnesses
        pairwise = []
        issues = []
        
        for i in range(len(frames) - 1):
            w = self.witness_between(frames[i], frames[i+1])
            pairwise.append(w)
            
            if w < 0.3:
                issues.append(f"{frames[i].source}-{frames[i+1].source}: dissolution ({w:.2f})")
            elif w > 1.5:
                issues.append(f"{frames[i].source}-{frames[i+1].source}: tension ({w:.2f})")
        
        avg_witness = np.mean(pairwise)
        coherent = avg_witness > 0.3 and avg_witness < 1.7
        
        return {
            'coherent': coherent,
            'score': avg_witness,
            'pairwise': pairwise,
            'issues': issues
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compose(*concepts: str, mode: str = 'sequential') -> Optional[SemanticFrame]:
    """Quick composition."""
    engine = CompositionEngine()
    return engine.compose_sentence(' '.join(concepts), mode=mode)


def analyze(concept1: str, concept2: str) -> Dict:
    """Quick pair analysis."""
    engine = CompositionEngine()
    return engine.analyze_pair(concept1, concept2)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COMPOSITION ENGINE TEST")
    print("=" * 60)
    
    engine = CompositionEngine()
    
    print("\n1. PAIR ANALYSIS")
    print("-" * 60)
    
    pairs = [
        ("HOT", "COLD"),
        ("LOVE", "HATE"),
        ("FIRE", "WATER"),
        ("LIGHT", "DARK"),
        ("I", "LOVE"),
        ("LOVE", "LOVE")  # Self-composition
    ]
    
    for a, b in pairs:
        result = engine.analyze_pair(a, b)
        if result:
            print(f"\n{a} <-> {b}:")
            print(f"  Angle: {result['core_angle']:.1f}°")
            print(f"  Relation: {result['relation']}")
            print(f"  Witness: {result['witness']:.2f} ({result['witness_meaning']})")
    
    print("\n\n2. SENTENCE COMPOSITION")
    print("-" * 60)
    
    sentences = [
        "I LOVE WATER",
        "HOT WATER",
        "COLD FIRE",
        "LIGHT DARK",
        "I AM LOVE"
    ]
    
    for sent in sentences:
        result = engine.compose_sentence(sent, mode='sequential')
        if result:
            print(f"\n'{sent}':")
            print(f"  Core: [{', '.join(f'{v:.2f}' for v in result.core)}]")
            print(f"  Witness: {result.witness:.2f}")
    
    print("\n\n3. COHERENCE ANALYSIS")
    print("-" * 60)
    
    for sent in sentences:
        coh = engine.sentence_coherence(sent)
        status = "✓" if coh['coherent'] else "✗"
        print(f"\n'{sent}': {status}")
        print(f"  Score: {coh['score']:.2f}")
        if coh['issues']:
            print(f"  Issues: {', '.join(coh['issues'])}")
    
    print("\n\n4. COMPOSITION MODES")
    print("-" * 60)
    
    test_sentence = "LOVE FIRE WATER"
    
    for mode in ['sequential', 'average', 'blended']:
        result = engine.compose_sentence(test_sentence, mode=mode)
        if result:
            print(f"\n'{test_sentence}' [{mode}]:")
            print(f"  Core: [{', '.join(f'{v:.2f}' for v in result.core)}]")
            print(f"  Witness: {result.witness:.2f}")
