"""
Semantic API for AGI Integration
=================================

High-level interface to the consciousness-structured semantic core.
Provides geometric semantics as an "unconscious" foundation for AGI systems.

Usage:
    from semantic_api import SemanticCore
    
    core = SemanticCore()
    
    # Get concept
    water = core.get("WATER")
    
    # Compose concepts
    result = core.compose("HOT", "WATER")
    
    # Find relationships
    neighbors = core.nearest_neighbors("LOVE", n=5)
    
    # Check coherence
    is_valid = core.is_coherent("I LOVE WATER")
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from extended_dictionary import ExtendedDictionary, ExtendedConcept, RelationType, ConceptLevel
from octonion import SemanticOctonion, DualOctonion, Trigram, semantic_compose


@dataclass
class CompositionResult:
    """Result of composing concepts."""
    essence: np.ndarray        # 8D essence vector
    function: np.ndarray       # 8D function vector
    witness: float             # Witness preservation (0=dissolved, 1=preserved, 2=tension)
    trigram: Trigram           # Resulting I Ching trigram
    components: List[str]      # Input concept names
    
    def __repr__(self):
        return f"Composition[w={self.witness:.2f}, {self.trigram.symbol}]"


class SemanticCore:
    """
    High-level interface to the geometric semantic dictionary.
    
    Provides:
    - Concept lookup and metadata
    - Geometric composition (quaternion-like multiplication)
    - Relationship analysis (angles, distances)
    - Semantic inference (nearest neighbors, paths)
    - Coherence checking
    """
    
    def __init__(self):
        """Initialize the semantic core."""
        self.dictionary = ExtendedDictionary()
        self._concept_vectors = None
        self._concept_names = None
        self._build_index()
    
    def _build_index(self):
        """Build numpy arrays for fast similarity search."""
        unique_concepts = {name: c for name, c in self.dictionary.concepts.items() 
                          if name == c.name}
        
        self._concept_names = list(unique_concepts.keys())
        
        # Build 7D vectors (excluding w)
        vectors = []
        for name in self._concept_names:
            c = unique_concepts[name]
            v = np.array([c.x, c.y, c.z, c.e, c.f, c.g, c.h])
            vectors.append(v)
        
        self._concept_vectors = np.array(vectors)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._concept_vectors, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        self._normalized_vectors = self._concept_vectors / norms
    
    # =========================================================================
    # CORE ACCESS
    # =========================================================================
    
    def get(self, name: str) -> Optional[ExtendedConcept]:
        """Get a concept by name (case-insensitive)."""
        return self.dictionary.concepts.get(name.upper())
    
    def exists(self, name: str) -> bool:
        """Check if a concept exists."""
        return name.upper() in self.dictionary.concepts
    
    def all_concepts(self) -> List[str]:
        """Get all unique concept names."""
        return self._concept_names.copy()
    
    def count(self) -> int:
        """Total number of unique concepts."""
        return len(self._concept_names)
    
    # =========================================================================
    # GEOMETRIC OPERATIONS
    # =========================================================================
    
    def vector(self, name: str, include_domain: bool = True) -> Optional[np.ndarray]:
        """
        Get the semantic vector for a concept.
        
        Args:
            name: Concept name
            include_domain: If True, return 7D [x,y,z,e,f,g,h]. If False, return 3D [x,y,z].
        
        Returns:
            numpy array or None if not found
        """
        c = self.get(name)
        if not c:
            return None
        
        if include_domain:
            return np.array([c.x, c.y, c.z, c.e, c.f, c.g, c.h])
        else:
            return np.array([c.x, c.y, c.z])
    
    def angle(self, name1: str, name2: str, core_only: bool = False) -> float:
        """
        Compute angle between two concepts in degrees.
        
        Args:
            name1, name2: Concept names
            core_only: If True, use only [x,y,z]. If False, use full 7D.
        
        Returns:
            Angle in degrees (0-180)
        """
        c1, c2 = self.get(name1), self.get(name2)
        if not c1 or not c2:
            return 0.0
        
        if core_only:
            return c1.angle_4d(c2)
        else:
            return c1.angle_8d(c2)
    
    def distance(self, name1: str, name2: str) -> float:
        """Euclidean distance between concept vectors."""
        v1, v2 = self.vector(name1), self.vector(name2)
        if v1 is None or v2 is None:
            return float('inf')
        return np.linalg.norm(v1 - v2)
    
    def dot(self, name1: str, name2: str) -> float:
        """Dot product of concept vectors."""
        v1, v2 = self.vector(name1), self.vector(name2)
        if v1 is None or v2 is None:
            return 0.0
        return np.dot(v1, v2)
    
    # =========================================================================
    # COMPOSITION
    # =========================================================================
    
    def compose(self, *names: str) -> Optional[CompositionResult]:
        """
        Compose multiple concepts into a single semantic result.
        
        Uses the witness preservation formula:
            w = 1 - cos(θ)
        
        where θ is the angle between consecutive concepts.
        
        Args:
            *names: Concept names to compose (minimum 2)
        
        Returns:
            CompositionResult with essence, function, and witness preservation
        """
        if len(names) < 2:
            if len(names) == 1:
                c = self.get(names[0])
                if c:
                    return CompositionResult(
                        essence=np.array([c.x, c.y, c.z, c.e, c.f, c.g, c.h]),
                        function=np.array([c.fx, c.fy, c.fz, c.fe, c.ff, c.fg, c.fh]),
                        witness=1.0,
                        trigram=c.trigram(),
                        components=list(names)
                    )
            return None
        
        # Get all concepts
        concepts = [self.get(name) for name in names]
        if None in concepts:
            missing = [n for n, c in zip(names, concepts) if c is None]
            raise ValueError(f"Unknown concepts: {missing}")
        
        # Sequential composition with witness tracking
        total_witness = 0.0
        current_essence = concepts[0].octonion()
        
        for i in range(1, len(concepts)):
            prev = concepts[i-1].octonion()
            curr = concepts[i].octonion()
            
            # Compute pairwise witness
            result, witness = semantic_compose(prev, curr)
            total_witness += witness
            
            # Update current (weighted average for multi-concept)
            current_essence = result
        
        # Average witness across pairs
        avg_witness = total_witness / (len(concepts) - 1)
        
        # Final composition of functions (simple average)
        func_sum = np.zeros(7)
        for c in concepts:
            func_sum += np.array([c.fx, c.fy, c.fz, c.fe, c.ff, c.fg, c.fh])
        func_avg = func_sum / len(concepts)
        
        # Determine resulting trigram
        result_octonion = SemanticOctonion(
            w=1.0,
            x=current_essence.x,
            y=current_essence.y,
            z=current_essence.z,
            e=current_essence.e,
            f=current_essence.f,
            g=current_essence.g,
            h=current_essence.h
        )
        
        return CompositionResult(
            essence=result_octonion.full_array(),
            function=func_avg,
            witness=avg_witness,
            trigram=result_octonion.to_trigram(x_axis=result_octonion.x),
            components=list(names)
        )
    
    def compose_sentence(self, sentence: str) -> Optional[CompositionResult]:
        """
        Compose a simple sentence (space-separated concepts).
        
        Example:
            core.compose_sentence("I LOVE WATER")
        """
        words = sentence.upper().split()
        valid_words = [w for w in words if self.exists(w)]
        
        if len(valid_words) < 2:
            return None
        
        return self.compose(*valid_words)
    
    # =========================================================================
    # RELATIONSHIP ANALYSIS
    # =========================================================================
    
    def relationship(self, name1: str, name2: str) -> Tuple[RelationType, float]:
        """
        Determine the relationship type between two concepts.
        
        Returns:
            (RelationType, angle_degrees)
        """
        angle = self.angle(name1, name2, core_only=True)
        
        if angle < 15:
            return (RelationType.SYNONYM, angle)
        elif angle < 45:
            return (RelationType.AFFINITY, angle)
        elif angle < 75:
            return (RelationType.ADJACENT, angle)
        elif angle < 105:
            return (RelationType.COMPLEMENT, angle)
        else:
            return (RelationType.OPPOSITION, angle)
    
    def is_complement(self, name1: str, name2: str) -> bool:
        """Check if two concepts are complements (80-105° core angle)."""
        angle = self.angle(name1, name2, core_only=True)
        return 80 <= angle <= 105
    
    def complements_of(self, name: str) -> List[Tuple[str, float]]:
        """Find all known complements of a concept."""
        results = []
        for rel in self.dictionary.relations:
            if rel[2] == RelationType.COMPLEMENT:
                if rel[0].upper() == name.upper():
                    results.append((rel[1], rel[3]))
                elif rel[1].upper() == name.upper():
                    results.append((rel[0], rel[3]))
        return sorted(results, key=lambda x: x[1])
    
    def affinities_of(self, name: str) -> List[Tuple[str, float]]:
        """Find all known affinities of a concept."""
        results = []
        for rel in self.dictionary.relations:
            if rel[2] == RelationType.AFFINITY:
                if rel[0].upper() == name.upper():
                    results.append((rel[1], rel[3]))
                elif rel[1].upper() == name.upper():
                    results.append((rel[0], rel[3]))
        return sorted(results, key=lambda x: x[1])
    
    # =========================================================================
    # SEMANTIC SEARCH
    # =========================================================================
    
    def nearest_neighbors(self, name: str, n: int = 5, 
                         core_only: bool = False) -> List[Tuple[str, float]]:
        """
        Find the n nearest concepts to a given concept.
        
        Args:
            name: Query concept
            n: Number of neighbors to return
            core_only: If True, match only on [x,y,z]
        
        Returns:
            List of (concept_name, angle) tuples, sorted by angle
        """
        query = self.get(name)
        if not query:
            return []
        
        if core_only:
            query_vec = np.array([query.x, query.y, query.z])
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            
            # Rebuild core-only normalized vectors
            core_vecs = self._concept_vectors[:, :3]
            norms = np.linalg.norm(core_vecs, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            normalized = core_vecs / norms
        else:
            query_vec = np.array([query.x, query.y, query.z, 
                                 query.e, query.f, query.g, query.h])
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            normalized = self._normalized_vectors
        
        # Cosine similarity
        similarities = np.dot(normalized, query_vec)
        
        # Convert to angles
        angles = np.degrees(np.arccos(np.clip(similarities, -1, 1)))
        
        # Sort and filter
        indices = np.argsort(angles)
        
        results = []
        for idx in indices:
            concept_name = self._concept_names[idx]
            if concept_name != name.upper():
                results.append((concept_name, angles[idx]))
                if len(results) >= n:
                    break
        
        return results
    
    def search_by_vector(self, vector: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts nearest to an arbitrary vector.
        
        Args:
            vector: 7D vector [x,y,z,e,f,g,h] or 3D [x,y,z]
            n: Number of results
        
        Returns:
            List of (concept_name, distance) tuples
        """
        if len(vector) == 3:
            # Pad to 7D
            vector = np.concatenate([vector, np.zeros(4)])
        
        query = vector / (np.linalg.norm(vector) + 1e-10)
        similarities = np.dot(self._normalized_vectors, query)
        angles = np.degrees(np.arccos(np.clip(similarities, -1, 1)))
        
        indices = np.argsort(angles)[:n]
        return [(self._concept_names[i], angles[i]) for i in indices]
    
    def concepts_by_trigram(self, trigram: str) -> List[str]:
        """Get all concepts belonging to a trigram."""
        trigram = trigram.upper()
        results = []
        for name in self._concept_names:
            c = self.get(name)
            if c and c.trigram().name == trigram:
                results.append(name)
        return results
    
    def concepts_by_level(self, level: ConceptLevel) -> List[str]:
        """Get all concepts at an ontological level."""
        results = []
        for name in self._concept_names:
            c = self.get(name)
            if c and c.level == level:
                results.append(name)
        return results
    
    # =========================================================================
    # COHERENCE CHECKING
    # =========================================================================
    
    def is_coherent(self, sentence: str, threshold: float = 0.3) -> bool:
        """
        Check if a sentence is semantically coherent.
        
        A sentence is coherent if witness preservation stays above threshold.
        
        Args:
            sentence: Space-separated concept names
            threshold: Minimum witness value (default 0.3)
        
        Returns:
            True if coherent, False if contains semantic conflict
        """
        result = self.compose_sentence(sentence)
        if result is None:
            return False
        return result.witness > threshold
    
    def coherence_score(self, sentence: str) -> float:
        """
        Get the coherence score (witness preservation) of a sentence.
        
        Returns:
            Float from 0 (completely incoherent) to 2 (maximum tension)
            Values near 1.0 indicate healthy semantic composition
        """
        result = self.compose_sentence(sentence)
        if result is None:
            return 0.0
        return result.witness
    
    # =========================================================================
    # INTROSPECTION
    # =========================================================================
    
    def describe(self, name: str) -> Dict[str, Any]:
        """Get full description of a concept."""
        c = self.get(name)
        if not c:
            return {}
        
        return {
            'name': c.name,
            'level': c.level.name,
            'trigram': c.trigram().symbol + ' ' + c.trigram().element,
            'core': [c.x, c.y, c.z],
            'domain': [c.e, c.f, c.g, c.h],
            'function': [c.fx, c.fy, c.fz, c.fe, c.ff, c.fg, c.fh],
            'domain_profile': c.domain_profile(),
            'description': c.description,
            'complements': [r[0] if r[1] == name.upper() else r[1] 
                           for r in self.dictionary.relations 
                           if r[2] == RelationType.COMPLEMENT and 
                           (r[0] == name.upper() or r[1] == name.upper())][:5]
        }
    
    def statistics(self) -> Dict[str, Any]:
        """Get dictionary statistics."""
        trigram_counts = {}
        level_counts = {}
        
        for name in self._concept_names:
            c = self.get(name)
            if c:
                t = c.trigram().name
                trigram_counts[t] = trigram_counts.get(t, 0) + 1
                l = c.level.name
                level_counts[l] = level_counts.get(l, 0) + 1
        
        complement_count = sum(1 for r in self.dictionary.relations 
                              if r[2] == RelationType.COMPLEMENT)
        affinity_count = sum(1 for r in self.dictionary.relations 
                            if r[2] == RelationType.AFFINITY)
        
        return {
            'total_concepts': len(self._concept_names),
            'total_relations': len(self.dictionary.relations),
            'complement_pairs': complement_count,
            'affinity_relations': affinity_count,
            'by_trigram': trigram_counts,
            'by_level': level_counts
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_core() -> SemanticCore:
    """Create and return a SemanticCore instance."""
    return SemanticCore()


def quick_compose(*names: str) -> Optional[CompositionResult]:
    """Quick composition without explicit core creation."""
    core = SemanticCore()
    return core.compose(*names)


def quick_angle(name1: str, name2: str, core_only: bool = True) -> float:
    """Quick angle computation."""
    core = SemanticCore()
    return core.angle(name1, name2, core_only=core_only)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC CORE API TEST")
    print("=" * 60)
    
    core = SemanticCore()
    
    print(f"\nLoaded {core.count()} concepts")
    print("\nStatistics:")
    for key, val in core.statistics().items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val}")
    
    print("\n" + "=" * 60)
    print("CONCEPT EXAMPLES")
    print("=" * 60)
    
    for name in ["WATER", "FIRE", "LOVE", "I"]:
        desc = core.describe(name)
        if desc:
            print(f"\n{name}:")
            print(f"  Trigram: {desc['trigram']}")
            print(f"  Level: {desc['level']}")
            print(f"  Core: {desc['core']}")
            print(f"  Domain: {desc['domain_profile']}")
            print(f"  Complements: {desc['complements']}")
    
    print("\n" + "=" * 60)
    print("COMPOSITION EXAMPLES")
    print("=" * 60)
    
    examples = [
        ["HOT", "WATER"],
        ["I", "LOVE", "WATER"],
        ["FIRE", "WATER"],
        ["LOVE", "HATE"]
    ]
    
    for names in examples:
        result = core.compose(*names)
        if result:
            print(f"\n{' + '.join(names)}:")
            print(f"  Witness: {result.witness:.2f}")
            print(f"  Trigram: {result.trigram.symbol} {result.trigram.element}")
    
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBORS")
    print("=" * 60)
    
    for query in ["LOVE", "WATER", "FIRE"]:
        neighbors = core.nearest_neighbors(query, n=5)
        print(f"\n{query} neighbors:")
        for name, angle in neighbors:
            print(f"  {name}: {angle:.1f}°")
