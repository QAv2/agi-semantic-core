"""
Oracle Web API — JSON-serializable diagnostic results.
=======================================================
Wraps OracleEngine for browser/Pyodide use. Returns structured
dicts instead of formatted terminal output, so the frontend can
render however it wants.
"""

import json
import numpy as np

from oracle.engine import OracleEngine, DiagnosticResult
from oracle.interpretations import INTERPRETATIONS
from oracle.interpreter import TRIGRAM_QUALITIES, WITNESS_READINGS, DOMAIN_READINGS
from oracle.toltec import get_toltec_for_king_wen
from oracle.tarot import TarotLayer


_engine = None
_tarot = None


def _bootstrap():
    global _engine, _tarot
    if _engine is None:
        _engine = OracleEngine()
        _tarot = TarotLayer(_engine.sc)
    return _engine, _tarot


def _witness_key(w: float) -> str:
    if w < 0.3:
        return "dissolution"
    if w < 0.7:
        return "partial"
    if w < 1.3:
        return "preserved"
    return "tension"


def _serialize_diagnosis(d: DiagnosticResult, tarot: TarotLayer) -> dict:
    p_quality = TRIGRAM_QUALITIES.get(d.presenting_trigram, ("", "", ""))
    m_quality = TRIGRAM_QUALITIES.get(d.medicine_trigram, ("", "", ""))
    interp = INTERPRETATIONS.get(d.hexagram["number"], {})
    wkey = _witness_key(d.witness)

    presenting = {
        "trigram": d.presenting_trigram,
        "element": d.presenting_element,
        "meaning": d.presenting_meaning,
        "quality": p_quality[0],
        "action": p_quality[1],
        "description": p_quality[2],
        "vector": [round(float(v), 3) for v in d.input_frame.core],
        "domain": {
            k: float(d.domain_profile[k])
            for k in ("spatial", "temporal", "relational", "personal")
        },
        "dominant": d.domain_profile["dominant"],
    }

    medicine = {
        "concept": d.medicine_frame.source,
        "trigram": d.medicine_trigram,
        "element": d.medicine_element,
        "meaning": d.medicine_meaning,
        "quality": m_quality[0],
        "action": m_quality[1],
        "description": m_quality[2],
        "candidates": [
            {"name": name, "angle": float(angle)}
            for name, angle in d.complement_concepts
        ],
    }

    hexagram = {
        "number": int(d.hexagram["number"]),
        "name": d.hexagram["name"],
        "title": d.hexagram["title"],
        "lower": d.hexagram["lower"],
        "upper": d.hexagram["upper"],
        "reading": d.hexagram["reading"],
        "judgment": interp.get("judgment", ""),
        "image": interp.get("image", ""),
        "counsel": interp.get("counsel", ""),
    }

    domain_text = DOMAIN_READINGS.get(d.domain_profile.get("dominant", ""), "")
    witness_text = WITNESS_READINGS.get(wkey, "")

    toltec_data = get_toltec_for_king_wen(d.hexagram["number"])
    toltec = None
    if toltec_data:
        toltec = {
            "number": int(toltec_data["toltec_number"]),
            "name": toltec_data["name"],
            "trad_name": toltec_data["trad_name"],
            "symbol": toltec_data.get("symbol", ""),
            "image": toltec_data.get("image", ""),
            "interp": toltec_data.get("interp", ""),
            "action": toltec_data.get("action", ""),
            "intent": toltec_data.get("intent", ""),
            "summary": toltec_data.get("summary", ""),
        }

    p_cards = tarot.nearest_card(d.input_frame.core, d.input_frame.domain, n=1)
    m_cards = tarot.nearest_card(d.medicine_frame.core, d.medicine_frame.domain, n=1)
    tarot_out = None
    if p_cards and m_cards:
        p_num, p_angle, p_data = p_cards[0]
        m_num, m_angle, m_data = m_cards[0]
        tarot_out = {
            "presenting": {
                "number": int(p_num),
                "numeral": p_data["numeral"],
                "name": p_data["name"],
                "angle": float(p_angle),
                "image": p_data["image"],
                "upright": p_data["upright"],
                "reversed": p_data["reversed"],
                "counsel": p_data["counsel"],
                "anchors": p_data["anchors"],
            },
            "medicine": {
                "number": int(m_num),
                "numeral": m_data["numeral"],
                "name": m_data["name"],
                "angle": float(m_angle),
                "image": m_data["image"],
                "upright": m_data["upright"],
                "reversed": m_data["reversed"],
                "counsel": m_data["counsel"],
                "anchors": m_data["anchors"],
            },
        }

    return {
        "input": d.input_text,
        "tokens": list(d.tokens_found),
        "missed": list(d.tokens_missed),
        "presenting": presenting,
        "medicine": medicine,
        "hexagram": hexagram,
        "witness": float(d.witness),
        "witness_state": wkey,
        "witness_text": witness_text,
        "domain_text": domain_text,
        "toltec": toltec,
        "tarot": tarot_out,
    }


def diagnose(text: str) -> str:
    """Run a full diagnosis and return a JSON string.
    JSON is used for the boundary because Pyodide's PyProxy
    serialization of nested structures has edge cases — JSON is
    bulletproof and trivial to parse on the JS side."""
    engine, tarot = _bootstrap()
    result = engine.diagnose(text)
    return json.dumps(_serialize_diagnosis(result, tarot))


def warmup() -> str:
    """Pre-load the SemanticCore. Returns a JSON status object."""
    engine, _ = _bootstrap()
    return json.dumps({
        "ready": True,
        "concepts": len(engine.sc._names),
    })
