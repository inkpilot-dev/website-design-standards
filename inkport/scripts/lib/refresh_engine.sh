#!/usr/bin/env bash
# Developer shortcut: reinstall only the engine package into an assembled payload (no torch re-download).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAYLOAD="${1:?usage: refresh_engine.sh <payload-dir>}"
PY="$PAYLOAD/python/bin/python3"; [ -x "$PY" ] || PY="$PAYLOAD/python/python.exe"
OUT="$(mktemp -d)"
uv build -q --wheel --out-dir "$OUT" "$ROOT"
uv pip install -q --python "$PY" --break-system-packages --no-deps --reinstall "$OUT"/inkport_engine-*.whl
"$PY" -m compileall -q --invalidation-mode unchecked-hash "$PAYLOAD/python"
rm -rf "$OUT"
echo "engine refreshed in $PAYLOAD"
