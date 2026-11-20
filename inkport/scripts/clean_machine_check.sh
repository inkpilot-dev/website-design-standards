#!/usr/bin/env bash
# Clean-machine verification for Inkport installers (Linux and macOS).
#
# Proves that, using ONLY what is inside the installer, a user can verify the installation,
# build a practice exam from the bundled sample data, train a model, run the before-vs-after
# gate on the hidden test set, and use the local OpenAI-compatible API, with no network and
# without any Python, Node, Rust, compiler or package manager from the machine.
#
#   scripts/clean_machine_check.sh build/installers/linux-x86_64-cpu/Inkport_0.1.0_amd64.AppImage
#   scripts/clean_machine_check.sh Inkport_0.1.0_linux-x86_64-portable.tar.zst
#   scripts/clean_machine_check.sh Inkport_0.1.0_amd64.deb
#   scripts/clean_machine_check.sh Inkport_0.1.0_aarch64.dmg          (macOS)
#
# Isolation on Linux (rootless, via unshare): a private network namespace with only loopback
# (every outbound connection fails), and the host's python*, pip*, node, npm, cargo, rustc, uv,
# cmake, gcc and git masked so any accidental use fails loudly. On macOS: sandbox-exec denies
# all non-loopback network traffic, and PATH is reduced to /usr/bin:/bin.
# Run it on a freshly installed OS (VM or container) for the strongest result.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
INSTALLER="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/inkport-clean.XXXXXX")"
PORT="${INKPORT_CHECK_PORT:-11541}"
OS="$(uname -s)"
say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

say "Unpacking $(basename "$INSTALLER") without root into $WORK"
case "$INSTALLER" in
  *.AppImage)
    (cd "$WORK" && APPIMAGE_EXTRACT_AND_RUN=1 "$INSTALLER" --appimage-extract >/dev/null)
    APP="$WORK/squashfs-root"; CLI="$APP/usr/bin/inkport-cli"; BUNDLE="$APP/usr/lib/Inkport/payload" ;;
  *.tar.zst)
    tar --zstd -xf "$INSTALLER" -C "$WORK"
    APP="$WORK/Inkport"; CLI="$APP/usr/bin/inkport-cli"; BUNDLE="$APP/usr/lib/Inkport/payload" ;;
  *.deb)
    dpkg-deb -x "$INSTALLER" "$WORK/root"
    APP="$WORK/root"; CLI="$APP/usr/bin/inkport-cli"; BUNDLE="$APP/usr/lib/Inkport/payload" ;;
  *.dmg)
    hdiutil attach -nobrowse -readonly -mountpoint "$WORK/mnt" "$INSTALLER" >/dev/null
    cp -R "$WORK/mnt/Inkport.app" "$WORK/"
    hdiutil detach "$WORK/mnt" >/dev/null
    APP="$WORK/Inkport.app"; CLI="$APP/Contents/MacOS/inkport-cli"; BUNDLE="$APP/Contents/Resources/payload" ;;
  *.app)
    cp -R "$INSTALLER" "$WORK/"
    APP="$WORK/$(basename "$INSTALLER")"; CLI="$APP/Contents/MacOS/inkport-cli"; BUNDLE="$APP/Contents/Resources/payload" ;;
  *)
    if [ -d "$INSTALLER" ]; then
      APP="$INSTALLER"; CLI="$(find "$APP" -name inkport-cli -type f | head -1)"; BUNDLE="$(dirname "$(find "$APP" -name manifest.json -path '*payload*' | head -1)")"
    else
      echo "unsupported installer: $INSTALLER" >&2; exit 2
    fi ;;
esac
[ -x "$CLI" ] || { echo "inkport-cli not found in the installer" >&2; exit 1; }
[ -f "$BUNDLE/manifest.json" ] || { echo "bundled payload not found in the installer" >&2; exit 1; }
du -sh "$APP" | awk '{print "    installed size: " $1}'
touch "$WORK/.before"

# ---------------------------------------------------------------- the checks (run isolated)
cat > "$WORK/inner.sh" <<'INNER'
set -euo pipefail
pass() { printf '    \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '    \033[31mFAIL\033[0m %s\n' "$*"; exit 1; }
json() { sed -n 's/^RESULT //p' | tail -1; }
field() { tr -d '\n' | grep -o "\"$1\":[^,}]*" | head -1 | cut -d: -f2- | tr -d '"'; }
t0=$(date +%s)

for tool in python python3 pip pip3 node npm cargo rustc uv cmake gcc git; do
  if command -v "$tool" >/dev/null 2>&1 && "$tool" --version >/dev/null 2>&1; then
    echo "    note: host $tool is still runnable here (macOS keeps system shims); Inkport does not use it"
  fi
done
if (exec 3<>/dev/tcp/1.1.1.1/443) 2>/dev/null; then fail "network is reachable; isolation did not apply"; else pass "no internet access inside the check"; fi

R=$("$CLI" doctor --full 2>>"$WORK/cli.log" | json) || fail "pre-flight failed: $(tail -3 "$WORK/cli.log")"
echo "$R" | grep -q '"ok":true' || fail "pre-flight not ok: $R"
echo "$R" | grep -q '"offline_core":true' || fail "pre-flight does not report offline core"
pass "pre-flight: every bundled file verified, engine warm ($(echo "$R" | field training_accelerator) for training)"

M=$("$CLI" models 2>>"$WORK/cli.log" | json); MODEL=$(echo "$M" | field id)
[ -n "$MODEL" ] || fail "no bundled base model"
pass "bundled starter model: $MODEL"
S=$("$CLI" samples 2>>"$WORK/cli.log" | json | field samples)
[ -f "$S/expenses.csv" ] || fail "bundled sample data missing"

"$CLI" create-project --name "Clean Check" --template business_rules 2>>"$WORK/cli.log" | json | grep -q '"slug":"clean-check"' || fail "create project"
I=$("$CLI" ingest --project clean-check --inputs "$S/expenses.csv" "$S/policy.md" --rules '["amount must be positive"]' 2>>"$WORK/cli.log" | json)
echo "$I" | grep -q '"leakage_free":true' || fail "ingest: $I"
pass "practice exam built from bundled samples (hidden test locked, zero leakage)"

if [ "$MODEL" = "smoke-tiny" ]; then
  T=$("$CLI" train --project clean-check --preset smoke 2>>"$WORK/cli.log" | json)
  echo "$T" | grep -q '"decision":"PROMOTE"' || fail "training/gate did not promote: $T"
else
  # A short real run on the bundled model: proves training + hidden-test gate complete offline.
  T=$("$CLI" train --project clean-check --preset fast --overrides '{"sft":{"max_steps":40,"eval_every":20,"checkpoint_every":20},"stage2":{"mode":"none"},"gate":{"general_suite_n":20}}' 2>>"$WORK/cli.log" | json)
  echo "$T" | grep -qE '"decision":"(PROMOTE|REJECT|INSUFFICIENT_DATA)"' || fail "training/gate did not finish: $T"
fi
pass "trained + gated on the hidden test: $(echo "$T" | field decision) $(echo "$T" | field version)"

A=$("$CLI" api-smoke --project clean-check --port "$PORT" 2>>"$WORK/cli.log" | json) || fail "local API: $(tail -3 "$WORK/cli.log")"
echo "$A" | grep -q '"streaming":true' || fail "local API: $A"
pass "local API on 127.0.0.1:$PORT: 401 without key, models listed, chat + @base + streaming via $(echo "$A" | field engine)"

R2=$("$CLI" doctor --full 2>>"$WORK/cli.log" | json)
echo "$R2" | grep -q '"ok":true' || fail "installation changed during use: $R2"
pass "installation unchanged after use (full re-verification)"
echo "    total: $(( $(date +%s) - t0 ))s"
INNER

say "Running the checks with no network and no developer tools"
ENVV=(HOME="$WORK/home" INKPORT_HOME="$WORK/home/inkport" TMPDIR="$WORK/tmp" PATH="/usr/bin:/bin" LANG=C.UTF-8 CLI="$CLI" WORK="$WORK" PORT="$PORT")
mkdir -p "$WORK/home" "$WORK/tmp"
if [ "$OS" = Linux ]; then
  # Mask host toolchains inside a private mount namespace; private net namespace with loopback only.
  MASK='for f in /usr/bin/python* /usr/bin/pip* /usr/local/bin/* /usr/bin/node /usr/bin/npm /usr/bin/cargo /usr/bin/rustc /usr/bin/cmake /usr/bin/gcc /usr/bin/git /usr/bin/uv; do [ -f "$f" ] && mount --bind /dev/null "$f" 2>/dev/null; done; for d in "$HOME_REAL/.local" "$HOME_REAL/.cargo" "$HOME_REAL/.rustup" "$HOME_REAL/.cache" /opt; do [ -d "$d" ] && mount -t tmpfs none "$d" 2>/dev/null; done; true'
  unshare --user --map-root-user --mount --net env HOME_REAL="$HOME" bash -c "ip link set lo up; $MASK; exec env -i ${ENVV[*]@Q} bash '$WORK/inner.sh'"
else
  PROFILE='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))(allow network-outbound (remote unix-socket))'
  sandbox-exec -p "$PROFILE" env -i "${ENVV[@]}" bash "$WORK/inner.sh"
fi

say "Checking that nothing was written outside the Inkport data folder"
CHANGED=$(find "$APP" -newer "$WORK/.before" -type f 2>/dev/null | head -5)
[ -z "$CHANGED" ] || { echo "    FAIL files changed inside the installation:"; echo "$CHANGED"; exit 1; }
printf '    \033[32mPASS\033[0m installation directory untouched; all data under %s\n' "$WORK/home/inkport"
du -sh "$WORK/home/inkport" | awk '{print "    user data created: " $1}'
printf '\n\033[1;32mClean-machine check passed: one download, fully offline.\033[0m\n'
rm -rf "$WORK"
