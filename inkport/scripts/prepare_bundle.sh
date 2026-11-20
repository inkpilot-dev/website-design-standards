#!/usr/bin/env bash
# Stage release resources: the engine wheel (installed into a private venv on first run) and
# the uv binary that creates that venv. Runs automatically from `pnpm tauri build`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RES="$ROOT/src-tauri/resources"
rm -f "$RES"/engine/*.whl
uv build --wheel --out-dir "$RES/engine" "$ROOT" >/dev/null
rm -f "$RES/engine/.gitignore"
UV_BIN="${BARNRAISE_UV:-$(command -v uv)}"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) cp "$UV_BIN" "$RES/bin/uv.exe" ;; *) cp "$UV_BIN" "$RES/bin/uv" ;; esac
echo "staged: $(ls "$RES/engine"/*.whl | xargs -n1 basename), uv $("$UV_BIN" --version | cut -d' ' -f2)"
