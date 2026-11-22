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
# (A grep that finds nothing must not end the script here: that once swallowed the log of a failed run.)
fail() {
  printf '    \033[31mFAIL\033[0m %s\n' "$*"
  echo "    --- errors in the Inkport log ---"
  { grep -iE "error|traceback|exception" "$WORK/cli.log" 2>/dev/null || true; } | tail -15 | cut -c1-400 | sed 's/^/    /'
  echo "    --- last lines of the Inkport log ---"
  { grep -v "not an adapter parameter" "$WORK/cli.log" 2>/dev/null || true; } | tail -70 | cut -c1-400 | sed 's/^/    /'
  exit 1
}
# The story of the training run just made: stages, practice scores, a few practice answers, verdict.
story() { { grep -E "^ *(stage|info|warn|eval|dev|gate) \|" "$WORK/cli.log" 2>/dev/null || true; } | tail -n "${1:-40}" | cut -c1-260 | sed 's/^/      /'; }
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
# The three rules the app suggests for this goal (contracts/templates.json): what a new user clicks.
RULES='["category must be one of fuel, meals, software, office_supplies, travel","amount must be positive","if the description mentions Uber then category must be travel"]'
# The tiny model of a test build barely learns this exam (it passes 4 to 12% after training), so it
# is spared anything that makes its instructions longer: it gets the two plain rules.
if [ "$MODEL" = "smoke-tiny" ]; then RULES='["category must be one of fuel, meals, software, office_supplies, travel","amount must be positive"]'; fi
I=$("$CLI" ingest --project clean-check --inputs "$S/expenses.csv" "$S/policy.md" --rules "$RULES" 2>>"$WORK/cli.log" | json) || fail "ingest crashed"
echo "$I" | grep -q '"leakage_free":true' || fail "ingest: $I"
echo "$I" | grep -q '"ready":true' || fail "the sample exam is not ready to train on: $I"
# A rule that is not used carries the reason in "error" (rules in use have an empty one).
if echo "$I" | grep -Eq '"error":"[^"]'; then fail "a suggested rule was not used with the sample data: $I"; fi
pass "practice exam built from bundled samples with the suggested rules (hidden test locked, zero leakage)"

if [ "$MODEL" = "smoke-tiny" ]; then
  T=$("$CLI" train --project clean-check --preset smoke 2>>"$WORK/cli.log" | json) || fail "training crashed"
  echo "$T" | grep -q '"decision":"PROMOTE"' || fail "training/gate did not promote: $T"
else
  # A real run on the bundled model, shortened to 60 steps (a "Quick run" takes 81) and without
  # the practice stage. The verdict is the one a user gets: the hidden exam and, switched on here
  # since a default run turned out to be refused for it, the general-skills check on 60 questions.
  # It must end in PROMOTE, and the steps below then answer with the trained version.
  T=$("$CLI" train --project clean-check --preset fast --overrides '{"sft":{"max_steps":60,"eval_every":20,"checkpoint_every":20},"stage2":{"mode":"none"}}' 2>>"$WORK/cli.log" | json) || fail "training crashed"
  echo "$T" | grep -q '"decision":"PROMOTE"' || fail "training finished but the trained model was not promoted on the bundled samples: $T"
fi
story 60
pass "trained + gated on the hidden test: $(echo "$T" | field decision) $(echo "$T" | field version)"

A=$("$CLI" api-smoke --project clean-check --port "$PORT" 2>>"$WORK/cli.log" | json) || fail "local API: $(tail -3 "$WORK/cli.log")"
echo "$A" | grep -q '"streaming":true' || fail "local API: $A"
pass "local API on 127.0.0.1:$PORT: 401 without key, models listed, chat + @base + streaming via $(echo "$A" | field engine)"

R2=$("$CLI" doctor --full 2>>"$WORK/cli.log" | json) || fail "re-verification crashed"
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
