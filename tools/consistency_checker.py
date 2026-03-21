#!/usr/bin/env python3
"""
Semantic Consistency Checker — Natural Language Claim Verification
=================================================================

Checks arbitrary natural language claims against the AGI Semantic Core's
grounded 14D semantic space via learned LLM embedding projection.

Accepts multiple claim formats:
    - "X is Y" / "X are Y"       (identity claims)
    - "X causes Y"               (causal claims)
    - "X is a type of Y"         (taxonomic claims)
    - "X is the opposite of Y"   (opposition claims)
    - Free-form sentences        (heuristic subject/predicate extraction)

Produces confidence-scored verdicts:
    CONSISTENT   (strong semantic alignment)
    PLAUSIBLE    (moderate alignment, contextually reasonable)
    SUSPICIOUS   (weak alignment, possible error)
    INCONSISTENT (semantic violation — hallucination signal)

Usage:
    python3 -m tools.consistency_checker "water is wet"
    python3 -m tools.consistency_checker "love causes happiness"
    python3 -m tools.consistency_checker --batch claims.txt
    python3 -m tools.consistency_checker --test
"""

import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from api.semantic_core import SemanticCore

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
EMBEDDINGS_PATH = os.path.join(EXPORT_DIR, 'llm_embeddings.npz')
PROJECTION_PATH = os.path.join(EXPORT_DIR, 'projection_matrix.npz')

# ── Verdict thresholds (degrees) ────────────────────────────────────────────
#
# These thresholds define how projected-space angles map to verdicts.
# They are calibrated per claim type because the semantic geometry differs:
#   - Identity claims expect near-alignment (small angles)
#   - Opposition claims expect large angles (near-orthogonal)
#   - Causal/taxonomic claims allow wider variance

THRESHOLDS = {
    'identity': {
        'consistent':   35,   # 0–35° → CONSISTENT
        'plausible':    55,   # 35–55° → PLAUSIBLE
        'suspicious':   75,   # 55–75° → SUSPICIOUS
        # >75° → INCONSISTENT
    },
    'causal': {
        'consistent':   45,
        'plausible':    65,
        'suspicious':   80,
    },
    'taxonomic': {
        'consistent':   40,
        'plausible':    60,
        'suspicious':   78,
    },
    'opposition': {
        # For opposition claims, we INVERT the logic:
        # large angle = consistent, small angle = inconsistent
        'consistent':   60,   # >60° → CONSISTENT (they ARE far apart)
        'plausible':    45,   # 45–60° → PLAUSIBLE
        'suspicious':   30,   # 30–45° → SUSPICIOUS
        # <30° → INCONSISTENT (they're too close to be opposites)
    },
    'freeform': {
        'consistent':   40,
        'plausible':    60,
        'suspicious':   78,
    },
}


# ── Claim Parsing ───────────────────────────────────────────────────────────

class ParsedClaim:
    """A parsed natural language claim with extracted terms and type."""

    __slots__ = ('raw', 'claim_type', 'subject', 'predicate', 'relation_hint')

    def __init__(self, raw, claim_type, subject, predicate, relation_hint=None):
        self.raw = raw
        self.claim_type = claim_type        # identity|causal|taxonomic|opposition|freeform
        self.subject = subject
        self.predicate = predicate
        self.relation_hint = relation_hint  # optional: expected relation type in dictionary

    def __repr__(self):
        return f"ParsedClaim({self.claim_type}: '{self.subject}' → '{self.predicate}')"


def parse_claim(text: str) -> ParsedClaim:
    """
    Parse a natural language claim into subject, predicate, and claim type.

    Tries patterns in order of specificity:
      1. "X is the opposite of Y"   → opposition
      2. "X is a type of Y" / "X is a kind of Y" → taxonomic
      3. "X causes Y" / "X leads to Y" / "X produces Y" → causal
      4. "X is Y" / "X are Y"       → identity
      5. Free-form fallback          → heuristic extraction
    """
    t = text.strip()

    # 1. Opposition: "X is the opposite of Y"
    m = re.match(r'^(.+?)\s+(?:is|are)\s+(?:the\s+)?opposite\s+of\s+(.+)$', t, re.I)
    if m:
        return ParsedClaim(t, 'opposition', m.group(1).strip(), m.group(2).strip(),
                           relation_hint='opposition')

    # 2. Taxonomic: "X is a type/kind/form/sort of Y"
    m = re.match(r'^(.+?)\s+(?:is|are)\s+(?:a\s+)?(?:type|kind|form|sort|category)\s+of\s+(.+)$', t, re.I)
    if m:
        return ParsedClaim(t, 'taxonomic', m.group(1).strip(), m.group(2).strip(),
                           relation_hint='affinity')

    # 3. Causal: "X causes/leads to/produces/creates/results in Y"
    m = re.match(r'^(.+?)\s+(?:causes?|leads?\s+to|produces?|creates?|results?\s+in|generates?)\s+(.+)$', t, re.I)
    if m:
        return ParsedClaim(t, 'causal', m.group(1).strip(), m.group(2).strip(),
                           relation_hint='affinity')

    # 4. Identity: "X is/are Y"
    m = re.match(r'^(.+?)\s+(?:is|are)\s+(.+)$', t, re.I)
    if m:
        subj = m.group(1).strip()
        pred = m.group(2).strip()
        # Strip leading articles from predicate for cleaner embedding
        pred_clean = re.sub(r'^(?:a|an|the)\s+', '', pred, flags=re.I)
        return ParsedClaim(t, 'identity', subj, pred_clean,
                           relation_hint='synonym')

    # 5. Free-form fallback: split on first verb-like word
    # Try to find a verb boundary
    m = re.match(r'^(.+?)\s+(can|will|should|must|has|have|had|does|do|did|was|were|'
                 r'means?|implies?|requires?|involves?|contains?|includes?|'
                 r'represents?|symbolizes?|embodies?|reflects?|denotes?)\s+(.+)$', t, re.I)
    if m:
        return ParsedClaim(t, 'freeform', m.group(1).strip(), m.group(3).strip())

    # Last resort: split roughly in half by word count
    words = t.split()
    if len(words) >= 2:
        mid = len(words) // 2
        return ParsedClaim(t, 'freeform', ' '.join(words[:mid]), ' '.join(words[mid:]))

    # Single word — can't really check
    return ParsedClaim(t, 'freeform', t, t)


# ── Verdict Engine ──────────────────────────────────────────────────────────

class GraphSignal:
    """Evidence from the complement/opposition graph scan."""

    __slots__ = ('has_complement', 'has_opposition', 'has_synonym', 'has_affinity',
                 'complement_pairs', 'opposition_pairs', 'synonym_pairs', 'affinity_pairs',
                 'strongest_signal', 'polarity_inversion')

    def __init__(self):
        self.has_complement = False
        self.has_opposition = False
        self.has_synonym = False
        self.has_affinity = False
        self.complement_pairs = []   # [(subj_match, pred_match, true_angle), ...]
        self.opposition_pairs = []
        self.synonym_pairs = []
        self.affinity_pairs = []
        self.strongest_signal = None  # 'synonym'|'affinity'|'complement'|'opposition'|None
        self.polarity_inversion = False  # True if identity claim + complement/opposition evidence

    def summary(self):
        parts = []
        if self.synonym_pairs:
            parts.append(f"SYNONYM({len(self.synonym_pairs)})")
        if self.affinity_pairs:
            parts.append(f"AFFINITY({len(self.affinity_pairs)})")
        if self.complement_pairs:
            parts.append(f"COMPLEMENT({len(self.complement_pairs)})")
        if self.opposition_pairs:
            parts.append(f"OPPOSITION({len(self.opposition_pairs)})")
        if self.polarity_inversion:
            parts.append("POLARITY_INVERSION")
        return " | ".join(parts) if parts else "(no graph signal)"


class Verdict:
    """Result of checking a single claim."""

    __slots__ = ('claim', 'label', 'confidence', 'angle', 'subject_nearest',
                 'predicate_nearest', 'known_relations', 'graph_signal', 'explanation')

    LABELS = ('CONSISTENT', 'PLAUSIBLE', 'SUSPICIOUS', 'INCONSISTENT')

    def __init__(self, claim, label, confidence, angle, subject_nearest,
                 predicate_nearest, known_relations, explanation, graph_signal=None):
        self.claim = claim
        self.label = label
        self.confidence = confidence    # 0–100
        self.angle = angle
        self.subject_nearest = subject_nearest      # [(name, angle), ...]
        self.predicate_nearest = predicate_nearest
        self.known_relations = known_relations      # [(n1, rel_type, n2, angle), ...]
        self.graph_signal = graph_signal            # GraphSignal or None
        self.explanation = explanation

    def short_str(self):
        return f"[{self.label:12s}] (confidence: {self.confidence:3d}/100, angle: {self.angle:5.1f}deg)  {self.claim.raw}"

    def long_str(self):
        lines = [
            "",
            f"  Claim: \"{self.claim.raw}\"",
            f"  Type:  {self.claim.claim_type}",
            f"  Parse: subject=\"{self.claim.subject}\"  predicate=\"{self.claim.predicate}\"",
            f"  Projected angle: {self.angle:.1f} deg",
            f"  Verdict: {self.label} (confidence {self.confidence}/100)",
            f"  Reason:  {self.explanation}",
        ]
        if self.graph_signal:
            lines.append(f"  Graph:   {self.graph_signal.summary()}")
        if self.subject_nearest:
            lines.append(f"  '{self.claim.subject}' nearest dictionary concepts:")
            for name, a in self.subject_nearest[:5]:
                lines.append(f"    {name:25s} {a:5.1f} deg")
        if self.predicate_nearest:
            lines.append(f"  '{self.claim.predicate}' nearest dictionary concepts:")
            for name, a in self.predicate_nearest[:5]:
                lines.append(f"    {name:25s} {a:5.1f} deg")
        if self.known_relations:
            lines.append(f"  Known relations between nearest matches:")
            for n1, rtype, n2, angle in self.known_relations:
                lines.append(f"    {n1} --[{rtype}]--> {n2} ({angle} deg)")
        else:
            lines.append(f"  Known relations between nearest matches: (none found)")
        return "\n".join(lines)


# ── Core Checker ────────────────────────────────────────────────────────────

class ConsistencyChecker:
    """
    Checks natural language claims against the grounded semantic space.

    Pipeline:
      1. Parse claim → (subject, predicate, claim_type)
      2. Embed both terms via sentence-transformers
      3. Project through learned W matrix into 14D grounded space
      4. Compute angle between projected vectors
      5. Find nearest dictionary concepts for each term
      6. Look up known relations between nearest matches
      7. Produce confidence-scored verdict
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._model = None
        self._W = None
        self._all_names = None
        self._all_proj = None
        self._sc = None

    def _ensure_loaded(self):
        """Lazy-load model, projection matrix, and dictionary."""
        if self._W is not None:
            return

        # Load projection matrix
        if not os.path.exists(PROJECTION_PATH):
            print("ERROR: No projection matrix found at", PROJECTION_PATH)
            print("       Run: python3 -m tools.embedding_projection project")
            sys.exit(1)
        proj_data = np.load(PROJECTION_PATH, allow_pickle=True)
        self._W = proj_data['W']

        # Load pre-computed embeddings for nearest-neighbor lookup
        if not os.path.exists(EMBEDDINGS_PATH):
            print("ERROR: No embeddings found at", EMBEDDINGS_PATH)
            print("       Run: python3 -m tools.embedding_projection extract")
            sys.exit(1)
        emb_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        self._all_names = list(emb_data['names'])
        all_emb = emb_data['embeddings']
        self._all_proj = all_emb @ self._W   # (N, 14) — all concepts projected

        # Precompute norms for fast cosine
        self._all_proj_norms = np.linalg.norm(self._all_proj, axis=1)

        # Build name→index lookup for direct dictionary matching
        self._name_to_idx = {n.upper(): i for i, n in enumerate(self._all_names)}

        # Load sentence-transformers model
        from sentence_transformers import SentenceTransformer
        if self.verbose:
            print("Loading embedding model (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load semantic core for relation lookups
        self._sc = SemanticCore()

        # Build alias→concept lookup for direct matching
        self._alias_to_concept = {}
        aliases = self._sc.conn.execute("SELECT alias, concept_id FROM aliases").fetchall()
        id_to_name = {}
        for row in self._sc.conn.execute("SELECT id, name FROM concepts"):
            id_to_name[row['id']] = row['name']
        for a in aliases:
            cname = id_to_name.get(a['concept_id'])
            if cname:
                self._alias_to_concept[a['alias'].upper()] = cname.upper()

    def _resolve_term(self, term):
        """
        Phase 7: Resolve a natural language term to a dictionary concept.

        Tries (in order):
          1. Exact concept name match (case-insensitive)
          2. Alias match
          3. Stripped articles/adjectives match ("a liquid" → "LIQUID")
          4. Morphological variants (darkness→DARK, destruction→DESTROY, etc.)
          5. None if no match found

        Returns (concept_name, grounded_vector) or (None, None).
        """
        # Normalize
        t = term.strip().upper()

        # Strip common prefixes
        for prefix in ('A ', 'AN ', 'THE ', 'SOME ', 'VERY ', 'REALLY ', 'QUITE '):
            if t.startswith(prefix):
                t = t[len(prefix):]

        def try_lookup(candidate):
            """Try name then alias lookup. Returns (name, vector) or None."""
            if candidate in self._name_to_idx:
                idx = self._name_to_idx[candidate]
                concept = self._sc.get(self._all_names[idx])
                if concept:
                    return self._all_names[idx], concept.full_vector()
            if candidate in self._alias_to_concept:
                cname = self._alias_to_concept[candidate]
                if cname in self._name_to_idx:
                    idx = self._name_to_idx[cname]
                    concept = self._sc.get(self._all_names[idx])
                    if concept:
                        return self._all_names[idx], concept.full_vector()
            return None

        # 1. Direct match
        result = try_lookup(t)
        if result:
            return result

        # 2. Morphological variants — strip common suffixes to find root form
        # Order matters: try longest suffixes first
        suffix_map = [
            ('NESSES', ''),     # darkness → dark
            ('NESS', ''),       # darkness → dark
            ('MENT', ''),       # movement → move
            ('MENTS', ''),
            ('TION', 'T'),      # destruction → destruct → try DESTROY via alias
            ('TION', ''),       # creation → creat (then try CREATE)
            ('SION', ''),       # tension → ten
            ('ENCE', ''),       # violence → viol
            ('ANCE', ''),       # resistance → resist
            ('ING', ''),        # running → runn
            ('INGS', ''),
            ('LY', ''),         # quickly → quick
            ('ITY', ''),        # gravity → grav
            ('OUS', ''),        # dangerous → danger
            ('FUL', ''),        # peaceful → peace
            ('LESS', ''),       # fearless → fear
            ('ABLE', ''),       # breakable → break
            ('IBLE', ''),
            ('IVE', ''),        # destructive → destruct
            ('AL', ''),         # emotional → emotion
            ('ED', ''),         # created → creat
            ('ER', ''),         # lighter → light
            ('EST', ''),        # brightest → bright
            ('S', ''),          # liquids → liquid
        ]

        for suffix, replacement in suffix_map:
            if t.endswith(suffix) and len(t) > len(suffix) + 1:
                stem = t[:-len(suffix)] + replacement
                result = try_lookup(stem)
                if result:
                    return result
                # Also try common verb forms from noun stems
                # destruction → DESTROY, creation → CREATE
                for verb_suffix in ('', 'E', 'Y'):
                    result = try_lookup(stem + verb_suffix)
                    if result:
                        return result

        return None, None

    def _embed_and_project(self, texts):
        """Embed texts and project to grounded space. Returns (N, 14) array."""
        emb = self._model.encode(texts)        # (N, 384)
        return emb @ self._W                   # (N, 14)

    def _angle_between(self, v1, v2):
        """Angle in degrees between two vectors."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 90.0  # degenerate — treat as orthogonal
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        return float(np.degrees(np.arccos(cos)))

    def _find_nearest(self, vec, top_n=5, direct_match=None):
        """
        Find nearest dictionary concepts to a projected vector.

        If direct_match is provided (from _resolve_term), it's inserted at
        position 0 with angle 0.0 — the ground truth from the dictionary.
        """
        norm_v = np.linalg.norm(vec)
        if norm_v < 1e-10:
            return [(direct_match, 0.0)] if direct_match else []
        dots = self._all_proj @ vec
        denoms = self._all_proj_norms * norm_v
        mask = denoms > 1e-10
        cos_vals = np.full(len(self._all_proj), 0.0)
        cos_vals[mask] = np.clip(dots[mask] / denoms[mask], -1, 1)
        angles = np.degrees(np.arccos(cos_vals))
        indices = np.argsort(angles)[:top_n]
        results = [(self._all_names[i], round(float(angles[i]), 1)) for i in indices]

        # If we have a direct match, ensure it's at position 0
        if direct_match:
            results = [(n, a) for n, a in results if n != direct_match]
            results.insert(0, (direct_match, 0.0))
            results = results[:top_n]

        return results

    def _find_known_relations(self, subj_nearest, pred_nearest):
        """Look up known relations between nearest-match concepts."""
        found = []
        for sn, _ in subj_nearest[:3]:
            rels = self._sc.relations_of(sn)
            for r in rels:
                for pn, _ in pred_nearest[:3]:
                    if r['other'] == pn:
                        angle = r.get('angle_8d') or r.get('angle_4d') or '?'
                        if isinstance(angle, float):
                            angle = round(angle, 1)
                        found.append((sn, r['type'], pn, angle))
        return found

    def _scan_complement_graph(self, subj_nearest, pred_nearest, claim_type):
        """
        Phase 7: Deep scan of complement/opposition graph between nearest matches.

        Scans top 5 nearest matches (not just top 3) for complement, opposition,
        synonym, and affinity relations. Detects polarity inversion: when an
        identity/causal/taxonomic claim maps to concepts that are actually
        complements or opposites in the grounded space.
        """
        signal = GraphSignal()

        # Weight by proximity — closer matches are stronger evidence
        # Match at 5° = weight 1.0, match at 30° = weight 0.5, match at 60° = weight 0.0
        def proximity_weight(angle):
            return max(0.0, 1.0 - angle / 60.0)

        for si, (sn, s_angle) in enumerate(subj_nearest[:5]):
            s_weight = proximity_weight(s_angle)
            if s_weight < 0.1:
                continue
            rels = self._sc.relations_of(sn)
            for r in rels:
                for pi, (pn, p_angle) in enumerate(pred_nearest[:5]):
                    if r['other'] != pn:
                        continue
                    p_weight = proximity_weight(p_angle)
                    if p_weight < 0.1:
                        continue
                    true_angle = r.get('angle_8d') or r.get('angle_4d') or 0
                    if isinstance(true_angle, float):
                        true_angle = round(true_angle, 1)
                    entry = (sn, pn, true_angle)

                    if r['type'] == 'complement':
                        signal.has_complement = True
                        signal.complement_pairs.append(entry)
                    elif r['type'] == 'opposition':
                        signal.has_opposition = True
                        signal.opposition_pairs.append(entry)
                    elif r['type'] == 'synonym':
                        signal.has_synonym = True
                        signal.synonym_pairs.append(entry)
                    elif r['type'] == 'affinity':
                        signal.has_affinity = True
                        signal.affinity_pairs.append(entry)

        # Determine strongest signal (priority: opposition > complement > synonym > affinity)
        if signal.has_opposition:
            signal.strongest_signal = 'opposition'
        elif signal.has_complement:
            signal.strongest_signal = 'complement'
        elif signal.has_synonym:
            signal.strongest_signal = 'synonym'
        elif signal.has_affinity:
            signal.strongest_signal = 'affinity'

        # Detect polarity inversion: claim asserts similarity, but graph shows
        # complement or opposition. Taxonomic claims need stronger evidence
        # (≥2 pairs) because they naturally span broader semantic space.
        if claim_type in ('identity', 'causal', 'freeform'):
            if signal.has_complement or signal.has_opposition:
                signal.polarity_inversion = True
        elif claim_type == 'taxonomic':
            total_contrary = len(signal.complement_pairs) + len(signal.opposition_pairs)
            if total_contrary >= 2:
                signal.polarity_inversion = True

        return signal

    def _compute_verdict(self, angle, claim_type, known_relations, graph_signal=None):
        """
        Map angle + claim type + relation evidence + graph signal → verdict.

        Phase 7 hybrid architecture:
          Stage 1: Projection angle → base verdict (catches topic drift)
          Stage 2: Complement graph → polarity inversion override (catches
                   "water is dry" where projection says close but graph says complement)

        Confidence scoring:
          - Base confidence from angle zone
          - Boosted/reduced by known relations (Phase 6)
          - Overridden by polarity inversion signal (Phase 7)
        """
        thresholds = THRESHOLDS.get(claim_type, THRESHOLDS['freeform'])

        # ── Stage 1: Projection-based verdict ──
        if claim_type == 'opposition':
            if angle >= thresholds['consistent']:
                label = 'CONSISTENT'
                base_conf = min(95, 60 + int((angle - thresholds['consistent']) * 1.5))
            elif angle >= thresholds['plausible']:
                label = 'PLAUSIBLE'
                base_conf = 50 + int((angle - thresholds['plausible']) / (thresholds['consistent'] - thresholds['plausible']) * 25)
            elif angle >= thresholds['suspicious']:
                label = 'SUSPICIOUS'
                base_conf = 35 + int((angle - thresholds['suspicious']) / (thresholds['plausible'] - thresholds['suspicious']) * 20)
            else:
                label = 'INCONSISTENT'
                base_conf = min(90, 55 + int((thresholds['suspicious'] - angle) * 2))
        else:
            if angle <= thresholds['consistent']:
                label = 'CONSISTENT'
                base_conf = min(95, 60 + int((thresholds['consistent'] - angle) * 1.5))
            elif angle <= thresholds['plausible']:
                label = 'PLAUSIBLE'
                range_size = thresholds['plausible'] - thresholds['consistent']
                position = (thresholds['plausible'] - angle) / range_size if range_size > 0 else 0.5
                base_conf = 40 + int(position * 25)
            elif angle <= thresholds['suspicious']:
                label = 'SUSPICIOUS'
                range_size = thresholds['suspicious'] - thresholds['plausible']
                position = (thresholds['suspicious'] - angle) / range_size if range_size > 0 else 0.5
                base_conf = 35 + int(position * 20)
            else:
                label = 'INCONSISTENT'
                base_conf = min(90, 55 + int((angle - thresholds['suspicious']) * 1.0))

        # ── Relation bonus (Phase 7: stronger graph-aware scoring) ──
        relation_bonus = 0
        explanation_parts = [f"Projected angle {angle:.1f} deg"]

        if known_relations:
            rel_types = [r[1] for r in known_relations]
            if claim_type == 'opposition':
                if 'opposition' in rel_types:
                    relation_bonus += 15
                    explanation_parts.append("dictionary confirms opposition")
                elif 'complement' in rel_types:
                    relation_bonus += 10
                    explanation_parts.append("dictionary confirms complement")
                elif 'synonym' in rel_types or 'affinity' in rel_types:
                    relation_bonus -= 15
                    explanation_parts.append("dictionary shows affinity, not opposition")
            else:
                if 'synonym' in rel_types:
                    relation_bonus += 20
                    explanation_parts.append("dictionary confirms synonym")
                elif 'affinity' in rel_types:
                    # Affinity with direct match = strong signal (e.g., WATER/LIQUID)
                    relation_bonus += 20
                    explanation_parts.append("dictionary confirms affinity")
                elif 'complement' in rel_types:
                    relation_bonus -= 10
                    explanation_parts.append("dictionary shows complement")
                elif 'opposition' in rel_types:
                    relation_bonus -= 20
                    explanation_parts.append("dictionary shows opposition")
        else:
            explanation_parts.append("no direct dictionary relations found")

        confidence = max(5, min(99, base_conf + relation_bonus))

        # Re-evaluate label if relation bonus shifted it significantly
        # (e.g., angle=59° is SUSPICIOUS by threshold, but affinity relation
        # means the dictionary confirms the connection)
        if claim_type != 'opposition' and relation_bonus >= 15:
            if label == 'SUSPICIOUS':
                label = 'PLAUSIBLE'
            elif label == 'INCONSISTENT' and relation_bonus >= 20:
                label = 'SUSPICIOUS'

        # ── Stage 2: Complement graph override (Phase 7) ──
        if graph_signal and graph_signal.polarity_inversion:
            # Identity/causal/taxonomic claim, but graph shows complement or opposition
            # This is the key Phase 7 addition: the projection says "close" but the
            # grounded geometry says "perpendicular or opposed"
            n_complement = len(graph_signal.complement_pairs)
            n_opposition = len(graph_signal.opposition_pairs)
            total_contrary = n_complement + n_opposition

            if total_contrary >= 3:
                # Strong graph evidence: multiple complement/opposition pairs
                label = 'INCONSISTENT'
                confidence = min(95, 70 + total_contrary * 3)
                pairs_desc = []
                for s, p, a in (graph_signal.opposition_pairs + graph_signal.complement_pairs)[:3]:
                    pairs_desc.append(f"{s}/{p}={a} deg")
                explanation_parts.append(
                    f"POLARITY INVERSION: {total_contrary} complement/opposition pairs "
                    f"in graph ({', '.join(pairs_desc)})"
                )
            elif total_contrary >= 1:
                # Moderate evidence: at least one complement/opposition pair
                if label in ('CONSISTENT', 'PLAUSIBLE'):
                    label = 'SUSPICIOUS'
                    confidence = max(confidence, 55 + total_contrary * 10)
                pairs_desc = []
                for s, p, a in (graph_signal.opposition_pairs + graph_signal.complement_pairs)[:2]:
                    pairs_desc.append(f"{s}/{p}={a} deg")
                explanation_parts.append(
                    f"POLARITY WARNING: {total_contrary} complement/opposition pair(s) "
                    f"in graph ({', '.join(pairs_desc)})"
                )

        # For opposition claims, boost confidence if graph confirms
        if graph_signal and claim_type == 'opposition':
            if graph_signal.has_complement or graph_signal.has_opposition:
                if label in ('SUSPICIOUS', 'INCONSISTENT'):
                    # Graph confirms they ARE opposites — upgrade
                    label = 'CONSISTENT'
                    n_evidence = len(graph_signal.complement_pairs) + len(graph_signal.opposition_pairs)
                    confidence = min(95, 65 + n_evidence * 5)
                    explanation_parts.append("graph confirms complement/opposition relation")
            if graph_signal.has_synonym:
                if label in ('CONSISTENT', 'PLAUSIBLE'):
                    label = 'INCONSISTENT'
                    confidence = min(90, 60 + len(graph_signal.synonym_pairs) * 10)
                    explanation_parts.append("graph shows synonym — not opposites")

        confidence = max(5, min(99, confidence))
        explanation = "; ".join(explanation_parts)

        return label, confidence, explanation

    def check(self, text: str) -> Verdict:
        """
        Check a single natural language claim. Returns a Verdict.

        Phase 7 hybrid pipeline:
          1. Try direct dictionary resolution for both terms
          2. If direct match: use grounded vectors (exact) for angle + graph scan
          3. If no match: fall back to projection (approximate)
          4. Graph scan runs on nearest matches either way
        """
        self._ensure_loaded()

        claim = parse_claim(text)

        # Phase 7: Try direct dictionary resolution first
        subj_name, subj_grounded = self._resolve_term(claim.subject)
        pred_name, pred_grounded = self._resolve_term(claim.predicate)

        # Compute projected vectors (always needed for nearest-neighbor fallback)
        projected = self._embed_and_project([claim.subject, claim.predicate])
        proj_subj = projected[0]
        proj_pred = projected[1]

        # Use grounded vectors when available, projected when not
        vec_subj = subj_grounded if subj_grounded is not None else proj_subj
        vec_pred = pred_grounded if pred_grounded is not None else proj_pred

        angle = self._angle_between(vec_subj, vec_pred)

        # Find nearest with direct match priority
        subj_nearest = self._find_nearest(proj_subj, direct_match=subj_name)
        pred_nearest = self._find_nearest(proj_pred, direct_match=pred_name)
        known_rels = self._find_known_relations(subj_nearest, pred_nearest)

        # Phase 7: Deep complement graph scan
        graph_signal = self._scan_complement_graph(
            subj_nearest, pred_nearest, claim.claim_type
        )

        label, confidence, explanation = self._compute_verdict(
            angle, claim.claim_type, known_rels, graph_signal
        )

        return Verdict(
            claim=claim,
            label=label,
            confidence=confidence,
            angle=round(angle, 1),
            subject_nearest=subj_nearest,
            predicate_nearest=pred_nearest,
            known_relations=known_rels,
            explanation=explanation,
            graph_signal=graph_signal,
        )

    def check_batch(self, claims):
        """Check a list of claim strings. Returns list of Verdicts."""
        self._ensure_loaded()
        return [self.check(c) for c in claims if c.strip()]

    def close(self):
        if self._sc:
            self._sc.close()


# ── Built-in Test Suite ─────────────────────────────────────────────────────

TEST_CLAIMS = [
    # ── True identity claims (should be CONSISTENT or PLAUSIBLE) ──
    ("water is a liquid",                    "identity",  True),
    ("love is an emotion",                   "identity",  True),
    ("fire is hot",                          "identity",  True),
    ("murder is a crime",                    "identity",  True),
    ("a dog is an animal",                   "identity",  True),
    ("mathematics is a science",             "identity",  True),
    ("sadness is a feeling",                 "identity",  True),
    ("gravity is a force",                   "identity",  True),

    # ── False identity claims (should be SUSPICIOUS or INCONSISTENT) ──
    ("water is dry",                         "identity",  False),
    ("love is hatred",                       "identity",  False),
    ("fire is cold",                         "identity",  False),
    ("truth is deception",                   "identity",  False),
    ("darkness is bright",                   "identity",  False),
    ("peace is violence",                    "identity",  False),

    # ── Causal claims ──
    ("fire causes heat",                     "causal",    True),
    ("rain causes flooding",                 "causal",    True),
    ("ignorance causes suffering",           "causal",    True),
    ("ice causes warmth",                    "causal",    False),
    ("silence causes noise",                 "causal",    False),

    # ── Taxonomic claims ──
    ("a rose is a type of flower",           "taxonomic", True),
    ("anger is a type of emotion",           "taxonomic", True),
    ("iron is a type of metal",              "taxonomic", True),
    ("a river is a type of mountain",        "taxonomic", False),

    # ── Opposition claims ──
    ("love is the opposite of hatred",       "opposition", True),
    ("light is the opposite of darkness",    "opposition", True),
    ("truth is the opposite of deception",   "opposition", True),
    ("water is the opposite of fire",        "opposition", True),
    ("love is the opposite of love",         "opposition", False),

    # ── Subtle / tricky claims ──
    ("justice is fairness",                  "identity",  True),
    ("war is peace",                         "identity",  False),  # Orwellian inversion
    ("freedom is slavery",                   "identity",  False),  # Orwellian inversion

    # ── Phase 7 polarity inversion targets ──
    # These are the claims that Phase 6 missed because the LLM compresses
    # complement pairs. Phase 7's graph scan should catch them.
    ("safety is danger",                     "identity",  False),
    ("light is darkness",                    "identity",  False),
    ("birth is death",                       "identity",  False),
    ("hot is cold",                          "identity",  False),
    ("joy is sorrow",                        "identity",  False),
    ("creation causes destruction",          "causal",    False),
    ("virtue is a type of vice",             "taxonomic", False),
]


def run_test_suite(verbose=True):
    """Run the built-in test suite and report results."""
    checker = ConsistencyChecker(verbose=verbose)

    print("=" * 78)
    print("  SEMANTIC CONSISTENCY CHECKER — BUILT-IN TEST SUITE")
    print("=" * 78)
    print(f"  {len(TEST_CLAIMS)} claims  |  model: all-MiniLM-L6-v2  |  projection: 384D -> 14D")
    print(f"  Dictionary: {EMBEDDINGS_PATH}")
    print("=" * 78)

    t0 = time.time()
    results = []
    correct = 0
    total = len(TEST_CLAIMS)

    for text, expected_type, expected_true in TEST_CLAIMS:
        verdict = checker.check(text)
        results.append((text, expected_type, expected_true, verdict))

        # Determine if verdict direction matches expectation
        if expected_true:
            hit = verdict.label in ('CONSISTENT', 'PLAUSIBLE')
        else:
            hit = verdict.label in ('SUSPICIOUS', 'INCONSISTENT')
        if hit:
            correct += 1

    elapsed = time.time() - t0

    # ── Print results ──

    # Group by claim type
    by_type = {}
    for text, etype, exp_true, verdict in results:
        by_type.setdefault(etype, []).append((text, exp_true, verdict))

    for ctype in ['identity', 'causal', 'taxonomic', 'opposition']:
        if ctype not in by_type:
            continue
        entries = by_type[ctype]
        print(f"\n{'─' * 78}")
        print(f"  {ctype.upper()} CLAIMS ({len(entries)})")
        print(f"{'─' * 78}")
        print(f"  {'Claim':<40s} {'Expect':>6s} {'Verdict':>12s} {'Conf':>4s} {'Angle':>6s} {'Graph':>12s} {'Hit':>4s}")
        print(f"  {'─' * 90}")

        for text, exp_true, verdict in entries:
            expect_label = "TRUE" if exp_true else "FALSE"
            if exp_true:
                hit = verdict.label in ('CONSISTENT', 'PLAUSIBLE')
            else:
                hit = verdict.label in ('SUSPICIOUS', 'INCONSISTENT')
            hit_str = "OK" if hit else "MISS"
            # Graph signal summary
            if verdict.graph_signal and verdict.graph_signal.polarity_inversion:
                graph_str = "INVERSION"
            elif verdict.graph_signal and verdict.graph_signal.strongest_signal:
                graph_str = verdict.graph_signal.strongest_signal[:10]
            else:
                graph_str = "-"
            # Truncate long claims
            display_text = text if len(text) <= 38 else text[:35] + "..."
            print(f"  {display_text:<40s} {expect_label:>6s} {verdict.label:>12s} {verdict.confidence:>3d}% {verdict.angle:>5.1f}d {graph_str:>12s} {hit_str:>4s}")

    # ── Summary ──
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Correct direction: {correct}/{total} ({100 * correct / total:.0f}%)")
    print(f"  Elapsed: {elapsed:.2f}s ({elapsed / total * 1000:.0f}ms per claim)")

    # Breakdown by type
    for ctype in ['identity', 'causal', 'taxonomic', 'opposition']:
        if ctype not in by_type:
            continue
        entries = by_type[ctype]
        type_correct = 0
        for text, exp_true, verdict in entries:
            if exp_true:
                hit = verdict.label in ('CONSISTENT', 'PLAUSIBLE')
            else:
                hit = verdict.label in ('SUSPICIOUS', 'INCONSISTENT')
            if hit:
                type_correct += 1
        print(f"    {ctype:<12s}: {type_correct}/{len(entries)}")

    # ── Show misses in detail ──
    misses = []
    for text, etype, exp_true, verdict in results:
        if exp_true:
            hit = verdict.label in ('CONSISTENT', 'PLAUSIBLE')
        else:
            hit = verdict.label in ('SUSPICIOUS', 'INCONSISTENT')
        if not hit:
            misses.append((text, etype, exp_true, verdict))

    if misses:
        print(f"\n{'─' * 78}")
        print(f"  MISSES (detailed)")
        print(f"{'─' * 78}")
        for text, etype, exp_true, verdict in misses:
            expect_dir = "should be CONSISTENT/PLAUSIBLE" if exp_true else "should be SUSPICIOUS/INCONSISTENT"
            print(f"\n  \"{text}\"")
            print(f"    Expected: {expect_dir}")
            print(f"    Got:      {verdict.label} (confidence {verdict.confidence}/100, angle {verdict.angle} deg)")
            print(f"    Reason:   {verdict.explanation}")
            if verdict.subject_nearest:
                top_subj = verdict.subject_nearest[0]
                print(f"    Subject nearest: {top_subj[0]} ({top_subj[1]} deg)")
            if verdict.predicate_nearest:
                top_pred = verdict.predicate_nearest[0]
                print(f"    Predicate nearest: {top_pred[0]} ({top_pred[1]} deg)")

    print(f"\n{'=' * 78}")
    print(f"  Phase 7 hybrid checker: projection (384D->14D, R^2=0.51) + complement graph.")
    print(f"  Stage 1: Projection catches topic drift (wrong domain).")
    print(f"  Stage 2: Graph catches polarity inversion (complement/opposition as identity).")
    print(f"  Dictionary: {len(checker._all_names)} concepts, 992 complement pairs.")
    print(f"{'=' * 78}")

    checker.close()
    return correct, total


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    arg = sys.argv[1]

    if arg == '--test':
        run_test_suite()

    elif arg == '--batch':
        if len(sys.argv) < 3:
            print("Usage: python3 -m tools.consistency_checker --batch claims.txt")
            return
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
        with open(filepath, 'r') as f:
            claims = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        checker = ConsistencyChecker()
        print(f"Checking {len(claims)} claims from {filepath}...")
        print(f"{'─' * 78}")

        verdicts = checker.check_batch(claims)

        print(f"\n{'Claim':<50s} {'Verdict':>12s} {'Conf':>4s} {'Angle':>6s}")
        print(f"{'─' * 78}")
        for v in verdicts:
            display_text = v.claim.raw if len(v.claim.raw) <= 48 else v.claim.raw[:45] + "..."
            print(f"{display_text:<50s} {v.label:>12s} {v.confidence:>3d}% {v.angle:>5.1f}d")

        # Summary counts
        counts = {}
        for v in verdicts:
            counts[v.label] = counts.get(v.label, 0) + 1
        print(f"\n{'─' * 78}")
        print(f"Summary: ", end="")
        print("  ".join(f"{k}: {c}" for k, c in sorted(counts.items())))

        checker.close()

    elif arg == '--help' or arg == '-h':
        print(__doc__)

    else:
        # Single claim — could be the claim text or could span multiple argv entries
        claim_text = ' '.join(sys.argv[1:])
        checker = ConsistencyChecker()
        verdict = checker.check(claim_text)
        print(verdict.long_str())
        checker.close()


if __name__ == "__main__":
    main()
