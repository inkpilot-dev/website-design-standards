#!/usr/bin/env bash
# Opens the real Inkport window on Linux (on a virtual display) and uses it the way a person would,
# with the real engine behind it. The window is driven through WebKitWebDriver by scripts/gui_e2e.cjs.
#
#   scripts/window_check.sh build/installers/linux-x86_64-cpu/Inkport_0.1.0_amd64.AppImage [out-dir]
#   scripts/window_check.sh target/debug/inkport out-dir        # a bare binary; set INKPORT_BUNDLE_DIR
#
#   WINDOW_PHASES="smoke goals"    start a run, see progress, stop it; then every goal's sample data
#                                  with every suggested rule (a few minutes; the default)
#   WINDOW_PHASES="first reopen"   a full run to a verdict, then restart the app and check it is all still there
#   INKPORT_TEST_PRESET / INKPORT_TEST_OVERRIDES keep training short (see src-tauri/src/app.rs)
#   WINDOW_GOAL=docs_legal WINDOW_EXPECT=any   run the first/smoke phases with another goal; "any" accepts
#                                  a verdict that the trained model was not better
#
# Needs on the test machine only: Xvfb, WebKitWebDriver (package webkit2gtk-driver), dbus-run-session, node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${1:?pass an AppImage or the inkport binary}"
OUT="${2:-window-check}"
PHASES="${WINDOW_PHASES:-smoke goals}"
PORT="${WINDOW_DRIVER_PORT:-4455}"
DISPLAY_NUM="${WINDOW_DISPLAY:-95}"
for tool in Xvfb WebKitWebDriver dbus-run-session node; do
  command -v "$tool" >/dev/null || { echo "missing test tool: $tool" >&2; exit 2; }
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
ART="$(cd "$(dirname "$ART")" && pwd)/$(basename "$ART")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/inkport-window.XXXXXX")"
# Spaces and non-Latin characters, as in a home folder named after its owner.
DATA="$WORK/home of José 日本語/Inkport"
mkdir -p "$DATA" "$WORK/xdg"
if [[ "$ART" == *.AppImage ]]; then
  (cd "$WORK" && chmod +x "$ART" && "$ART" --appimage-extract >/dev/null)
  APP="$WORK/squashfs-root/AppRun"
else
  APP="$ART"
fi
pass() { printf '    \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '    \033[31mFAIL\033[0m %s\n' "$*"; FAILED=1; }
FAILED=0
XVFB_PID=""; DRIVER_PID=""
cleanup() {
  [ -n "$DRIVER_PID" ] && kill "$DRIVER_PID" 2>/dev/null || true
  # $DRIVER_PID is dbus-run-session; the driver itself is its child and would outlive it.
  pkill -f "WebKitWebDriver --port=$PORT" 2>/dev/null || true
  [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true
  pkill -f "$WORK" 2>/dev/null || true
  rm -rf "$WORK" 2>/dev/null || true   # a desktop-portal mount inside it may refuse; harmless
}
trap cleanup EXIT

echo "    machine: $(nproc) processors, $(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo) GB of memory, $(awk '/SwapTotal/{printf "%.1f", $2/1048576}' /proc/meminfo) GB of swap"
# If the window stops answering, this is run while it is still open: what is the app doing?
cat >"$WORK/diagnose.sh" <<DIAG
#!/usr/bin/env bash
echo "--- the machine when the window stopped answering ---"
echo "load: \$(cat /proc/loadavg)"; free -m
echo "--- Inkport's processes (state, cpu, memory, running time) ---"
ps -eo pid,ppid,stat,pcpu,pmem,etime,args | grep -E "inkport|WebKit|python.* -m engine|llama-server" | grep -v -E "grep|gui_e2e|window_check|diagnose" | cut -c1-220
for f in "$DATA/logs/app-crash.log" "$DATA/logs/last-engine-error.log"; do [ -f "\$f" ] && { echo "--- \$(basename "\$f") ---"; tail -30 "\$f"; }; done
echo "--- the app's own output (last lines) ---"; tail -25 "$OUT/webdriver.log" | cut -c1-300
if command -v gdb >/dev/null; then
  for pid in \$(pgrep -f "bin/inkport\$|/inkport\$|AppRun" | head -2) \$(pgrep -f WebKitWebProcess | head -1); do
    echo "--- where process \$pid (\$(cat /proc/\$pid/comm 2>/dev/null)) is, thread by thread ---"
    # The main thread first and in full (a frozen window means it is stuck somewhere), then the top of every thread.
    G=(-p "\$pid" -batch -ex "echo == main thread ==\\n" -ex "thread 1" -ex "bt 28" -ex "echo == every thread, top frames ==\\n" -ex "thread apply all bt 4")
    (sudo -n gdb "\${G[@]}" 2>/dev/null || gdb "\${G[@]}" 2>/dev/null) | grep -E "^==|^Thread|^#" | sed -E 's/ \\([^)]*\\) / /' | cut -c1-150 | head -130
  done
fi
(dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null || true) | grep -iE "out of memory|killed process|oom-kill" | tail -5
DIAG
chmod +x "$WORK/diagnose.sh"
export GUI_E2E_ON_FAILURE="$WORK/diagnose.sh" RUST_BACKTRACE=1

Xvfb ":$DISPLAY_NUM" -screen 0 1240x900x24 -nolisten tcp >/dev/null 2>&1 &
XVFB_PID=$!
sleep 1
export DISPLAY=":$DISPLAY_NUM" INKPORT_HOME="$DATA" XDG_DATA_HOME="$WORK/xdg/data" XDG_CONFIG_HOME="$WORK/xdg/config" XDG_CACHE_HOME="$WORK/xdg/cache"
export TAURI_WEBVIEW_AUTOMATION=true TAURI_AUTOMATION=true
# A virtual display has no GPU; keep WebKit on software rendering.
export WEBKIT_DISABLE_COMPOSITING_MODE=1 WEBKIT_DISABLE_DMABUF_RENDERER=1 LIBGL_ALWAYS_SOFTWARE=1
# A driver left over from an earlier run would answer on the same port with that run's settings.
pkill -f "WebKitWebDriver --port=$PORT" 2>/dev/null && sleep 1 || true
dbus-run-session -- WebKitWebDriver "--port=$PORT" --host=127.0.0.1 >"$OUT/webdriver.log" 2>&1 &
DRIVER_PID=$!
for _ in $(seq 1 40); do
  node -e "require('http').get('http://127.0.0.1:$PORT/status',r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))" 2>/dev/null && break
  sleep 0.5
done
if grep -q "Unable to listen" "$OUT/webdriver.log" 2>/dev/null || ! kill -0 "$DRIVER_PID" 2>/dev/null; then
  echo "the test driver could not start on port $PORT:" >&2; tail -3 "$OUT/webdriver.log" >&2; exit 2
fi

for phase in $PHASES; do
  printf '\n\033[1;34m==> Window test: %s\033[0m\n' "$phase"
  if node "$ROOT/scripts/gui_e2e.cjs" --mode=webdriver "--driver=http://127.0.0.1:$PORT" "--app=$APP" "--out=$OUT" "--phase=$phase" "--train-timeout=${WINDOW_TRAIN_TIMEOUT:-3600}" "--goal=${WINDOW_GOAL:-business_rules}" "--expect=${WINDOW_EXPECT:-promote}" 2>&1 | sed 's/^/    /'; then
    pass "window test ($phase), data folder with spaces and non-Latin characters"
  else
    fail "window test ($phase); see the lines above and the screenshots in $OUT"
    for f in "$DATA/logs/last-engine-error.log" "$DATA/logs/llama-server.log"; do
      [ -f "$f" ] && { echo "    --- $(basename "$f") ---"; tail -20 "$f" | sed 's/^/    /'; }
    done
    break
  fi
  sleep 3
  if pgrep -f "llama-server|python.* -m engine" >/dev/null 2>&1 && pgrep -af "llama-server|python.* -m engine" | grep -q "$WORK\|$DATA"; then
    fail "helper processes are still running after the app was closed"
  else
    pass "the app closed and left no helper process behind"
  fi
done
[ "$FAILED" = 0 ] && printf '\nWindow check passed. Screenshots are in %s\n' "$OUT" || { printf '\nWindow check FAILED.\n'; exit 1; }
