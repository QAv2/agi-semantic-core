"""
Oracle Diagnostic Engine
=========================
Input is symptom. Output is medicine.
The diagnosis uses quaternion-encoded semantic geometry
with I-Ching trigram structure as the change-state taxonomy.

Pipeline:
  User text → tokenize → dictionary lookup → compose → diagnostic vector
  → trigram reading (presenting state) → complement (medicine)
  → hexagram (composed reading) → structured diagnosis
"""

import sys
import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.semantic_core import SemanticCore
from core.encoding import Concept
from core.octonion import SemanticOctonion, Trigram, quaternion_multiply
from oracle.hexagrams import lookup_hexagram, HEXAGRAM_NAMES

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
    'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
    'really', 'like', 'feel', 'feeling', 'think', 'know', 'going',
    'want', 'need', 'get', 'got', 'would', 'could', 'much',
    'lot', 'thing', 'things', 'way', 'even', 'still', 'also',
    'always', 'never', 'sometimes', 'everything', 'anything', 'something',
    'nothing', 'im', "i'm", "don't", "can't", "it's", "i've",
}

SYNONYM_MAP = {
    'stuck': 'STAGNATION',
    'lonely': 'LONELINESS',
    'alone': 'SOLITUDE',
    'anxious': 'ANXIETY',
    'depressed': 'DEPRESSION',
    'stressed': 'STRESS',
    'guilty': 'GUILT',
    'bitter': 'BITTERNESS',
    'confused': 'CONFUSION',
    'disconnected': 'ISOLATION',
    'trapped': 'CONFINEMENT',
    'lost': 'LOSS',
    'hurt': 'WOUND',
    'broken': 'FRAGMENTATION',
    'tired': 'EXHAUSTION',
    'empty': 'VOID',
    'numb': 'APATHY',
    'scared': 'FEAR',
    'afraid': 'FEAR',
    'worried': 'ANXIETY',
    'hopeless': 'DESPAIR',
    'restless': 'RESTLESSNESS',
    'angry': 'ANGER',
    'sad': 'SADNESS',
    'jealous': 'ENVY',
    'ashamed': 'SHAME',
    'weak': 'WEAKNESS',
    'overwhelmed': 'OVERWHELM',
    'unforgiven': 'RESENTMENT',
    'abandoned': 'ABANDONMENT',
    'rejected': 'REJECTION',
    'grateful': 'GRATITUDE',
    'thankful': 'GRATITUDE',
    'perfect': 'COMPLETENESS',
    'complete': 'COMPLETENESS',
    'peaceful': 'PEACE',
    'joyful': 'JOY',
    'happy': 'JOY',
    'loving': 'LOVE',
    'creative': 'CREATIVE',
    'powerful': 'POWER',
    'free': 'FREEDOM',
    'connected': 'CONNECTION',
    'inspired': 'INSPIRATION',
    'curious': 'CURIOSITY',
    'patient': 'PATIENCE',
}

# Words to try stripping suffixes from for dictionary lookup
SUFFIX_PATTERNS = [
    (r'ing$', ''),
    (r'ed$', ''),
    (r'ness$', ''),
    (r'ment$', ''),
    (r'tion$', ''),
    (r'sion$', ''),
    (r'ful$', ''),
    (r'less$', ''),
    (r'ly$', ''),
    (r'er$', ''),
    (r'est$', ''),
    (r'ity$', ''),
    (r'ous$', ''),
    (r'ive$', ''),
    (r'able$', ''),
    (r'ible$', ''),
    (r's$', ''),
]


@dataclass
class SemanticFrame:
    """A point in 7D semantic space with witness preservation."""
    core: np.ndarray       # [x, y, z]
    domain: np.ndarray     # [e, f, g, h]
    witness: float = 1.0
    source: str = ""

    @classmethod
    def from_concept(cls, c: Concept) -> 'SemanticFrame':
        return cls(
            core=np.array([c.x, c.y, c.z]),
            domain=np.array([c.e, c.f, c.g, c.h]),
            witness=1.0,
            source=c.name,
        )

    @property
    def full_vector(self) -> np.ndarray:
        return np.concatenate([self.core, self.domain])

    def to_octonion(self) -> SemanticOctonion:
        return SemanticOctonion(
            w=self.witness,
            x=self.core[0], y=self.core[1], z=self.core[2],
            e=self.domain[0], f=self.domain[1],
            g=self.domain[2], h=self.domain[3],
        )

    def trigram(self) -> Trigram:
        return self.to_octonion().to_trigram(x_axis=self.core[0], y_axis=self.core[1])

    def core_angle(self, other: 'SemanticFrame') -> float:
        v1, v2 = self.core, other.core
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)))

    def full_angle(self, other: 'SemanticFrame') -> float:
        v1, v2 = self.full_vector, other.full_vector
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 1e-10 or m2 < 1e-10:
            return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)))


@dataclass
class DiagnosticResult:
    """Full Oracle diagnostic output."""
    input_text: str
    tokens_found: List[str]
    tokens_missed: List[str]
    input_frame: SemanticFrame
    presenting_trigram: str
    presenting_element: str
    presenting_meaning: str
    complement_concepts: List[Tuple[str, float]]  # (name, angle)
    medicine_frame: SemanticFrame
    medicine_trigram: str
    medicine_element: str
    medicine_meaning: str
    hexagram: dict
    witness: float
    domain_profile: dict


class OracleEngine:
    """The diagnostic layer. Input is symptom, output is medicine."""

    def __init__(self):
        self.sc = SemanticCore()
        self._level_weights = np.array([
            self.sc.get(name).level.value / 7.0 if self.sc.get(name) else 0.0
            for name in self.sc._names
        ])
        self._level_weights = np.clip(self._level_weights, 0.3, 1.0)

    def tokenize(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract semantically meaningful tokens from input text.
        Returns (found_tokens, missed_tokens)."""
        cleaned = re.sub(r'[^\w\s\'-]', ' ', text.lower())
        raw_words = cleaned.split()

        found = []
        missed = []

        for word in raw_words:
            word = word.strip("'-")
            if not word or word in STOPWORDS:
                continue

            concept = self.sc.get(word.upper())
            if concept:
                found.append(word.upper())
                continue

            # Try synonym map
            if word in SYNONYM_MAP:
                syn = SYNONYM_MAP[word]
                if self.sc.get(syn):
                    found.append(syn)
                    continue

            # Try morphological reduction
            matched = False
            for pattern, replacement in SUFFIX_PATTERNS:
                stem = re.sub(pattern, replacement, word)
                if stem and len(stem) >= 3 and stem != word:
                    concept = self.sc.get(stem.upper())
                    if concept:
                        found.append(stem.upper())
                        matched = True
                        break
            if not matched:
                missed.append(word)

        return found, missed

    def compose(self, token_names: List[str]) -> SemanticFrame:
        """Compose a list of concept names into a single semantic frame."""
        if not token_names:
            return SemanticFrame(
                core=np.zeros(3), domain=np.zeros(4),
                witness=1.0, source="<empty>",
            )

        frames = []
        for name in token_names:
            c = self.sc.get(name)
            if c:
                frames.append(SemanticFrame.from_concept(c))

        if not frames:
            return SemanticFrame(
                core=np.zeros(3), domain=np.zeros(4),
                witness=1.0, source="<no matches>",
            )

        if len(frames) == 1:
            return frames[0]

        # Sequential quaternion composition (order-preserving)
        result = frames[0]
        for frame in frames[1:]:
            q1 = (result.witness, result.core[0], result.core[1], result.core[2])
            q2 = (frame.witness, frame.core[0], frame.core[1], frame.core[2])
            w, x, y, z = quaternion_multiply(q1, q2)

            # Domain: weighted average, later frames accumulate
            domain = 0.4 * result.domain + 0.6 * frame.domain

            # Witness between pair: w = 1 - cos(θ)
            pair_witness = 1.0 - np.clip(
                np.dot(result.core, frame.core) /
                (np.linalg.norm(result.core) * np.linalg.norm(frame.core) + 1e-10),
                -1, 1,
            )

            result = SemanticFrame(
                core=np.array([x, y, z]),
                domain=domain,
                witness=pair_witness,
                source=f"({result.source} × {frame.source})",
            )

        return result

    def find_complement(self, frame: SemanticFrame, n: int = 5) -> Tuple[SemanticFrame, List[Tuple[str, float]]]:
        """Find the ~90° complement of a frame in the dictionary.
        Searches for orthogonality in core space (x,y,z), then ranks
        candidates by domain similarity — medicine operates in the same
        domain but from a different direction.
        Returns (medicine_frame, [(concept_name, angle), ...])."""
        query_core = frame.core
        query_domain = frame.domain
        core_norm = np.linalg.norm(query_core)
        if core_norm < 1e-10:
            return frame, []

        # Core angles (3D: x,y,z) — where the 90° rule lives
        dots = self.sc._vectors_3d @ query_core
        denoms = self.sc._norms_3d * core_norm
        mask = denoms > 1e-10
        cos_thetas = np.full(len(self.sc._vectors_3d), 0.0)
        cos_thetas[mask] = np.clip(dots[mask] / denoms[mask], -1, 1)
        core_angles = np.degrees(np.arccos(cos_thetas))

        # Domain similarity (4D: e,f,g,h) — cosine similarity for ranking
        domain_norm = np.linalg.norm(query_domain)
        domains = self.sc._vectors_7d[:, 3:]  # columns 3-6 are e,f,g,h
        domain_norms = np.linalg.norm(domains, axis=1)
        domain_dots = domains @ query_domain
        domain_denom = domain_norms * domain_norm + 1e-10
        domain_sim = np.clip(domain_dots / domain_denom, -1, 1)

        # Composite score: 90° proximity + domain alignment + semantic depth
        angular_deviation = np.abs(core_angles - 90.0)
        candidates = angular_deviation < 30.0  # within 60°-120° band
        scores = np.full(len(self.sc._names), -np.inf)
        scores[candidates] = (
            domain_sim[candidates] * 0.5
            + self._level_weights[candidates] * 0.3
            - (angular_deviation[candidates] / 90.0) * 0.2
        )

        indices = np.argsort(-scores)

        complement_concepts = []
        for idx in indices[:n]:
            complement_concepts.append((
                self.sc._names[idx],
                round(float(core_angles[idx]), 1),
            ))

        best_idx = indices[0]
        best_concept = self.sc.get(self.sc._names[best_idx])
        medicine = SemanticFrame.from_concept(best_concept)

        return medicine, complement_concepts

    def domain_profile(self, frame: SemanticFrame) -> dict:
        """Human-readable domain analysis."""
        labels = ['spatial', 'temporal', 'relational', 'personal']
        values = frame.domain
        total = np.sum(np.abs(values)) + 0.01
        profile = {}
        for label, val in zip(labels, values):
            profile[label] = round(float(val), 2)
        dominant_idx = np.argmax(np.abs(values))
        profile['dominant'] = labels[dominant_idx]
        return profile

    def diagnose(self, text: str) -> DiagnosticResult:
        """Full diagnostic pipeline: text → structured reading."""
        # 1. Tokenize
        found, missed = self.tokenize(text)

        # 2. Compose
        input_frame = self.compose(found)

        # 3. Presenting trigram
        presenting = input_frame.trigram()

        # 4. Find complement (medicine)
        medicine_frame, complement_concepts = self.find_complement(input_frame)
        medicine_tri = medicine_frame.trigram()

        # 5. Hexagram
        hexagram = lookup_hexagram(presenting.name, medicine_tri.name)

        # 6. Domain profile
        profile = self.domain_profile(input_frame)

        # 7. Witness between input and medicine
        witness = 1.0 - np.clip(
            np.dot(input_frame.core, medicine_frame.core) /
            (np.linalg.norm(input_frame.core) * np.linalg.norm(medicine_frame.core) + 1e-10),
            -1, 1,
        )

        return DiagnosticResult(
            input_text=text,
            tokens_found=found,
            tokens_missed=missed,
            input_frame=input_frame,
            presenting_trigram=presenting.name,
            presenting_element=presenting.element,
            presenting_meaning=presenting.meaning,
            complement_concepts=complement_concepts,
            medicine_frame=medicine_frame,
            medicine_trigram=medicine_tri.name,
            medicine_element=medicine_tri.element,
            medicine_meaning=medicine_tri.meaning,
            hexagram=hexagram,
            witness=round(witness, 3),
            domain_profile=profile,
        )

    def format_diagnosis(self, d: DiagnosticResult, show_toltec: bool = False, show_tarot: bool = False) -> str:
        """Format a diagnosis for terminal display."""
        lines = []
        lines.append("=" * 60)
        lines.append("  ORACLE DIAGNOSTIC")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Input: \"{d.input_text}\"")
        lines.append(f"  Tokens: {', '.join(d.tokens_found)}")
        if d.tokens_missed:
            lines.append(f"  Missed: {', '.join(d.tokens_missed)}")
        lines.append("")

        lines.append("  PRESENTING STATE")
        lines.append(f"  Trigram: {d.presenting_trigram} — {d.presenting_element} ({d.presenting_meaning})")
        lines.append(f"  Domain: {d.domain_profile['dominant']} dominant")
        for k in ['spatial', 'temporal', 'relational', 'personal']:
            bar = "█" * int(d.domain_profile[k] * 20)
            lines.append(f"    {k:11s} {d.domain_profile[k]:+.2f} {bar}")
        lines.append(f"  Vector: [{', '.join(f'{v:+.2f}' for v in d.input_frame.core)}]")
        lines.append("")

        lines.append("  MEDICINE (complement)")
        lines.append(f"  Trigram: {d.medicine_trigram} — {d.medicine_element} ({d.medicine_meaning})")
        lines.append(f"  Concept: {d.medicine_frame.source}")
        lines.append(f"  Complement candidates:")
        for name, angle in d.complement_concepts:
            lines.append(f"    {name:20s} {angle:5.1f}°")
        lines.append("")

        lines.append("  HEXAGRAM")
        h = d.hexagram
        lines.append(f"  #{h['number']:02d} — {h['name']} ({h['title']})")
        lines.append(f"  {h['lower']} (below) + {h['upper']} (above)")
        lines.append(f"  {h['reading']}")
        lines.append("")

        lines.append(f"  Witness: {d.witness:.3f}")
        if d.witness < 0.3:
            lines.append("  ⟶ Dissolution — input dissolves into medicine (fusion/attachment)")
        elif d.witness < 0.7:
            lines.append("  ⟶ Partial preservation — some structure holds")
        elif d.witness < 1.3:
            lines.append("  ⟶ Full preservation — healthy complement, witness intact")
        else:
            lines.append("  ⟶ Tension — input and medicine in opposition")

        lines.append("")
        lines.append("=" * 60)

        from oracle.interpreter import interpret
        reading = interpret(d)
        if reading:
            lines.append("")
            lines.append(reading)

        from oracle.toltec import format_toltec_reading, get_toltec_for_king_wen
        toltec = get_toltec_for_king_wen(h['number'])
        if toltec:
            if show_toltec:
                toltec_text = format_toltec_reading(h['number'])
                if toltec_text:
                    lines.append(toltec_text)
            else:
                lines.append("")
                lines.append("─" * 60)
                lines.append(f"  TOLTEC CORRESPONDENCE")
                lines.append(f"  Toltec #{toltec['toltec_number']:02d}: {toltec['name']}")
                lines.append(f"  (King Wen #{h['number']}: {toltec['trad_name']})")
                lines.append("")
                lines.append("  For the full Toltec reading, use --toltec flag")
                lines.append("─" * 60)

        from oracle.tarot import TarotLayer, format_tarot_reading, format_tarot_pointer
        tarot_layer = TarotLayer(self.sc)
        p_cards = tarot_layer.nearest_card(d.input_frame.core, d.input_frame.domain, n=1)
        m_cards = tarot_layer.nearest_card(d.medicine_frame.core, d.medicine_frame.domain, n=1)
        if p_cards and m_cards:
            if show_tarot:
                lines.append(format_tarot_reading(p_cards[0], m_cards[0]))
            else:
                lines.append(format_tarot_pointer(p_cards[0], m_cards[0]))

        return "\n".join(lines)


if __name__ == '__main__':
    show_toltec = '--toltec' in sys.argv
    show_tarot = '--tarot' in sys.argv
    args = [a for a in sys.argv[1:] if a not in ('--toltec', '--tarot')]
    engine = OracleEngine()
    if args:
        text = ' '.join(args)
        result = engine.diagnose(text)
        print()
        print(engine.format_diagnosis(result, show_toltec=show_toltec, show_tarot=show_tarot))
    else:
        from oracle.session import run_interactive
        run_interactive(show_toltec=show_toltec, show_tarot=show_tarot)
