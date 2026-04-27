#!/usr/bin/env bash
# Build the Oracle Pyodide bundle.
# Packages all Python modules + the semantic db into a single tar.gz
# that Pyodide can unpack with pyodide.unpackArchive().
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB="$ROOT/web"
STAGE="$WEB/.stage"
OUT="$WEB/assets/oracle-bundle.tar.gz"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# Copy Python source — only the modules needed for diagnose()
for pkg in core api db; do
  mkdir -p "$STAGE/$pkg"
  cp "$ROOT/$pkg"/*.py "$STAGE/$pkg/" 2>/dev/null || true
done

# oracle/ — skip CLI-only modules (session, casting, penniston)
mkdir -p "$STAGE/oracle"
for f in __init__.py engine.py hexagrams.py interpreter.py interpretations.py tarot.py toltec.py runes.py web_api.py; do
  cp "$ROOT/oracle/$f" "$STAGE/oracle/$f"
done

# Database
cp "$ROOT/db/semantic.db" "$STAGE/db/semantic.db"

# Strip __pycache__ if any snuck in
find "$STAGE" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Build the archive
mkdir -p "$WEB/assets"
tar -czf "$OUT" -C "$STAGE" .

# Report
SIZE=$(du -h "$OUT" | cut -f1)
COUNT=$(tar -tzf "$OUT" | wc -l)
echo "built $OUT — $SIZE, $COUNT files"

rm -rf "$STAGE"
