"""
Oracle Session Manager
=======================
Interactive diagnostic sessions with logging and history.

Usage:
  python3 -m oracle.session                  # start interactive session
  python3 -m oracle.session --history        # list past sessions
  python3 -m oracle.session --history 5      # last 5 sessions
  python3 -m oracle.session --last           # replay most recent session
  python3 -m oracle.session --replay <file>  # replay a specific session
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from oracle.engine import OracleEngine, DiagnosticResult

SESSIONS_DIR = Path(__file__).parent / "sessions"


def result_to_dict(d: DiagnosticResult) -> dict:
    return {
        "input_text": d.input_text,
        "tokens_found": d.tokens_found,
        "tokens_missed": d.tokens_missed,
        "presenting": {
            "trigram": d.presenting_trigram,
            "element": d.presenting_element,
            "meaning": d.presenting_meaning,
            "vector": [round(float(v), 4) for v in d.input_frame.core],
        },
        "medicine": {
            "concept": d.medicine_frame.source,
            "trigram": d.medicine_trigram,
            "element": d.medicine_element,
            "meaning": d.medicine_meaning,
        },
        "complement_candidates": [
            {"name": name, "angle": angle}
            for name, angle in d.complement_concepts
        ],
        "hexagram": d.hexagram,
        "witness": d.witness,
        "domain_profile": d.domain_profile,
    }


def run_interactive(show_toltec: bool = False, show_tarot: bool = False, show_runes: bool = False):
    SESSIONS_DIR.mkdir(exist_ok=True)

    engine = OracleEngine()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_file = SESSIONS_DIR / f"{timestamp}.json"
    queries = []

    print("=" * 60)
    print("  ORACLE — Interactive Session")
    print("  Type your condition. Type 'exit' or 'quit' to end.")
    layers = []
    if show_toltec:
        layers.append("Toltec")
    if show_tarot:
        layers.append("Tarot")
    if show_runes:
        layers.append("Runes")
    if layers:
        print(f"  Transparency layers: {', '.join(layers)}")
    print("=" * 60)
    print()

    try:
        while True:
            try:
                text = input(f"  [{len(queries) + 1}] > ").strip()
            except EOFError:
                break

            if not text:
                continue
            if text.lower() in ("exit", "quit", "q"):
                break

            result = engine.diagnose(text)
            print()
            print(engine.format_diagnosis(result, show_toltec=show_toltec, show_tarot=show_tarot, show_runes=show_runes))
            print()

            queries.append({
                "n": len(queries) + 1,
                "timestamp": datetime.now().isoformat(),
                "result": result_to_dict(result),
            })

    except KeyboardInterrupt:
        print()

    if queries:
        session = {
            "session": timestamp,
            "started": queries[0]["timestamp"],
            "ended": datetime.now().isoformat(),
            "count": len(queries),
            "queries": queries,
        }
        session_file.write_text(json.dumps(session, indent=2))
        print(f"\n  Session saved: {session_file.name} ({len(queries)} queries)")
    else:
        print("\n  No queries — session not saved.")


def show_history(n: int = 10):
    if not SESSIONS_DIR.exists():
        print("  No session history.")
        return

    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    if not files:
        print("  No session history.")
        return

    files = files[:n]
    print("=" * 60)
    print("  ORACLE — Session History")
    print("=" * 60)
    for f in files:
        try:
            data = json.loads(f.read_text())
            count = data.get("count", 0)
            started = data.get("started", "?")[:19]
            first_input = ""
            if data.get("queries"):
                first_input = data["queries"][0]["result"]["input_text"]
                if len(first_input) > 40:
                    first_input = first_input[:37] + "..."
            print(f"  {f.name}  {count} queries  {started}")
            if first_input:
                print(f"    → \"{first_input}\"")
        except (json.JSONDecodeError, KeyError):
            print(f"  {f.name}  (corrupt)")
    print()
    print(f"  Replay with: python3 -m oracle.session --replay <filename>")


def replay_session(path: Path):
    if not path.exists():
        if (SESSIONS_DIR / path.name).exists():
            path = SESSIONS_DIR / path.name
        else:
            print(f"  Not found: {path}")
            return

    data = json.loads(path.read_text())
    print("=" * 60)
    print(f"  ORACLE — Replay: {path.name}")
    print(f"  {data.get('count', '?')} queries, {data.get('started', '?')[:19]}")
    print("=" * 60)

    for q in data.get("queries", []):
        r = q["result"]
        h = r["hexagram"]
        print()
        print(f"  [{q['n']}] \"{r['input_text']}\"")
        print(f"      Tokens: {', '.join(r['tokens_found'])}")
        print(f"      Presenting: {r['presenting']['trigram']} — {r['presenting']['element']}")
        print(f"      Medicine: {r['medicine']['concept']} ({r['medicine']['trigram']} — {r['medicine']['element']})")
        print(f"      Hexagram: #{h['number']:02d} {h['name']} — {h['title']}")
        print(f"      Witness: {r['witness']}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    show_toltec = "--toltec" in args
    show_tarot = "--tarot" in args
    show_runes = "--runes" in args
    args = [a for a in args if a not in ("--toltec", "--tarot", "--runes")]

    if "--history" in args:
        idx = args.index("--history")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 10
        show_history(n)
    elif "--last" in args:
        if SESSIONS_DIR.exists():
            files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
            if files:
                replay_session(files[0])
            else:
                print("  No sessions found.")
        else:
            print("  No session history.")
    elif "--replay" in args:
        idx = args.index("--replay")
        if idx + 1 < len(args):
            replay_session(Path(args[idx + 1]))
        else:
            print("  Usage: --replay <filename>")
    else:
        run_interactive(show_toltec=show_toltec, show_tarot=show_tarot, show_runes=show_runes)
