#!/usr/bin/env bash
# Inkport end-to-end verification (Linux / macOS).
#
# Proves the layers work together through the real Rust supervisor:
#   Rust CLI -> Python engine (NDJSON) -> ingest + hidden-test split -> LoRA training on a tiny
#   offline model -> before-vs-after gate -> promotion -> Proof Arena -> local OpenAI-compatible
#   API (auth, Host/Origin checks, streaming) -> signed adapter export -> tamper-rejected import.
#
# Usage: scripts/verify_e2e.sh            (creates .e2e/venv with CPU PyTorch)
#        INKPORT_PYTHON=/path/to/python scripts/verify_e2e.sh   (reuse an environment)
#        SKIP_FRONTEND=1 / SKIP_UNIT=1    (skip those stages)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${INKPORT_E2E_DIR:-$ROOT/.e2e}"
PORT="${INKPORT_E2E_PORT:-11535}"
rm -rf "$WORK/home" "$WORK/run"
mkdir -p "$WORK/run"
export INKPORT_HOME="$WORK/home"
LOG="$WORK/run/e2e.log"
: >"$LOG"
SERVE_PID=""
START=$(date +%s)

cleanup() { if [ -n "$SERVE_PID" ]; then kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
pass() { printf '    \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '    \033[31mFAIL\033[0m %s\n    (full log: %s)\n' "$*" "$LOG"; exit 1; }
result() { grep '^RESULT ' | tail -1 | cut -c8-; }
jget() { "$PY" -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'d': d}))" "$1"; }

# --------------------------------------------------------------------------- 1. Python
step "Python engine environment"
if [ -z "${INKPORT_PYTHON:-}" ]; then
  command -v uv >/dev/null || fail "uv is required (https://docs.astral.sh/uv/) or set INKPORT_PYTHON"
  if [ ! -x "$WORK/venv/bin/python" ]; then
    uv venv -q --python 3.12 "$WORK/venv"
    uv pip install -q --python "$WORK/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cpu >>"$LOG" 2>&1 \
      || uv pip install -q --python "$WORK/venv/bin/python" torch >>"$LOG" 2>&1
  fi
  uv pip install -q --python "$WORK/venv/bin/python" -e "$ROOT[dev]" >>"$LOG" 2>&1 || fail "engine install"
  export INKPORT_PYTHON="$WORK/venv/bin/python"
fi
PY="$INKPORT_PYTHON"
"$PY" -c "import engine, torch, transformers, peft; print('    engine', engine.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

if [ -z "${SKIP_UNIT:-}" ]; then
  step "Python unit + contract tests"
  (cd "$ROOT" && "$PY" -m pytest -q -m "not slow" -p no:cacheprovider) >>"$LOG" 2>&1 && pass "pytest" || fail "pytest"
fi

# --------------------------------------------------------------------------- 2. Rust
step "Rust supervisor: tests + headless CLI build"
(cd "$ROOT/src-tauri" && cargo test -q --no-default-features) >>"$LOG" 2>&1 && pass "cargo test" || fail "cargo test"
(cd "$ROOT/src-tauri" && cargo build -q --no-default-features --bin inkport-cli) >>"$LOG" 2>&1 || fail "cargo build"
CLI="$ROOT/target/debug/inkport-cli"
cli() { "$CLI" "$@" 2>>"$LOG"; }

# --------------------------------------------------------------------------- 3. Fixtures
step "Offline fixtures (tiny model + synthetic small-business data)"
if [ ! -f "$WORK/model/config.json" ]; then
  "$PY" -m engine smoke-model --out "$WORK/model" >>"$LOG" 2>&1 || fail "smoke model"
fi
"$PY" -m engine smoke-data --out "$WORK/data" >>"$LOG" 2>&1 || fail "smoke data"
pass "model + data in $WORK"

# --------------------------------------------------------------------------- 4. Pipeline
step "Hardware detection + runtime status (Rust)"
HW=$(cli hardware | result); echo "    tier: $(echo "$HW" | jget "d['recommendation']['tier']") | $(echo "$HW" | jget "d['summary'][0]")"
ST=$(cli status | result); [ "$(echo "$ST" | jget "d['ready']")" = "True" ] && pass "engine reachable through the supervisor" || fail "runtime not ready: $ST"

step "Create project + ingest (dedup, hidden test split, verifiers, rules)"
cli create-project --name "Expense Sorter" --template business_rules --base-id smoke-tiny --base-path "$WORK/model" --backend torch --quant fp32 | result >/dev/null || fail "create project"
ING=$(cli ingest --project expense-sorter --inputs "$WORK/data/expenses.csv" "$WORK/data/policy.md" --rules '["category must be one of fuel, meals, software, office_supplies, travel", "amount must be positive"]' | result)
[ "$(echo "$ING" | jget "d['leakage_free']")" = "True" ] || fail "ingest: $ING"
echo "    split: $(echo "$ING" | jget "d['counts']")"
[ "$(echo "$ING" | jget "d['counts']['test'] >= 30")" = "True" ] && pass "zero train/test leakage, >=30 hidden test items" || fail "too few test items"

step "Train (LoRA SFT + outcome-filtered refinement) + before-vs-after gate"
TR=$(cli train --project expense-sorter --preset smoke | result)
DEC=$(echo "$TR" | jget "d.get('decision')")
echo "    $(echo "$TR" | jget "d.get('explanation', d)")"
[ "$DEC" = "PROMOTE" ] && pass "gate promoted $(echo "$TR" | jget "d.get('version')")" || fail "gate decision: $DEC"

step "Proof Arena (baseline vs trained, dev questions)"
AR=$(cli arena --project expense-sorter --n 3 | result)
echo "    wins: $(echo "$AR" | jget "d['wins']")"
[ "$(echo "$AR" | jget "d['cases']")" = "3" ] && pass "arena" || fail "arena: $AR"

# --------------------------------------------------------------------------- 5. Local API
step "Local OpenAI-compatible API on 127.0.0.1:$PORT"
"$CLI" serve --project expense-sorter --port "$PORT" >"$WORK/run/serve.out" 2>>"$LOG" &
SERVE_PID=$!
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"model_loaded":true'; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"model_loaded":true' || fail "API did not come up"
TOKEN=$(cat "$INKPORT_HOME/api_token")
API="http://127.0.0.1:$PORT/v1"
[ "$(curl -s -o /dev/null -w '%{http_code}' "$API/models")" = "401" ] && pass "missing API key -> 401" || fail "auth not enforced"
[ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H 'Host: evil.example' "$API/models")" = "403" ] && pass "DNS-rebinding Host -> 403" || fail "host check"
[ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" -H 'Origin: https://evil.example' "$API/models")" = "403" ] && pass "foreign web Origin -> 403" || fail "origin check"
MODELS=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API/models")
echo "$MODELS" | grep -q 'inkport/expense-sorter' && pass "/v1/models lists inkport/expense-sorter" || fail "models: $MODELS"
CHAT=$(curl -sf -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' "$API/chat/completions" \
  -d '{"model":"inkport/expense-sorter","messages":[{"role":"user","content":"Paid $42 at Shell for the delivery van"}],"max_tokens":40}')
CONTENT=$(echo "$CHAT" | jget "d['choices'][0]['message']['content']")
echo "    model answered: $CONTENT"
echo "$CONTENT" | "$PY" -c "import json,sys; o=json.loads(sys.stdin.read()); assert 'category' in o and 'amount' in o" && pass "chat completion returns the trained JSON format" || fail "chat: $CHAT"
curl -sfN -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' "$API/chat/completions" \
  -d '{"model":"inkport/expense-sorter","stream":true,"messages":[{"role":"user","content":"Zoom receipt $15 monthly plan"}],"max_tokens":30}' | grep -q '\[DONE\]' && pass "SSE streaming" || fail "streaming"
kill "$SERVE_PID"; wait "$SERVE_PID" 2>/dev/null || true; SERVE_PID=""

# --------------------------------------------------------------------------- 6. Adapters
step "Signed adapter export / import"
EX=$(cli export --project expense-sorter --out "$WORK/run/v1.inkpatch" | result)
echo "    sha256 $(echo "$EX" | jget "d['sha256']")"
"$PY" - "$WORK/run/v1.inkpatch" "$WORK/run/tampered.inkpatch" <<'PYEOF'
import sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
    for i in zin.infolist():
        data = zin.read(i.filename)
        if i.filename == "adapter/adapter_model.safetensors":
            data = data[:-4] + b"\x00\x00\x80\x7f"  # flip the last weight to +inf
        zout.writestr(i.filename, data)
PYEOF
TI=$(cli import --project expense-sorter --file "$WORK/run/tampered.inkpatch" --preset smoke | result || true)
echo "$TI" | grep -q 'SHA-256 mismatch' && pass "tampered package rejected (SHA-256)" || fail "tampered import not rejected: $TI"
OK=$(cli import --project expense-sorter --file "$WORK/run/v1.inkpatch" --preset smoke | result)
echo "    re-import gate: $(echo "$OK" | jget "d['gate']['decision']") (an identical adapter must not be promoted twice)"
[ "$(echo "$OK" | jget "d['import']['signature_verified']")" = "True" ] && pass "signature + hashes + tensors verified, then gated" || fail "import: $OK"

# --------------------------------------------------------------------------- 7. Frontend
if [ -z "${SKIP_FRONTEND:-}" ] && command -v pnpm >/dev/null; then
  step "Frontend typecheck + production build"
  (cd "$ROOT" && pnpm install --frozen-lockfile >>"$LOG" 2>&1 && pnpm build >>"$LOG" 2>&1) && pass "vite build" || fail "frontend build"
fi

printf '\n\033[1;32mAll end-to-end checks passed in %ss.\033[0m\n' "$(( $(date +%s) - START ))"
