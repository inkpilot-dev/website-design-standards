#!/usr/bin/env bash
# Inkport all-in-one installer build for macOS (Apple Silicon) and Linux (x86_64).
#
# Produces installers that contain everything needed to train, verify and serve models offline:
# a relocatable CPython with the engine and its libraries pre-installed, llama.cpp with this OS's
# GPU backends, and a 4-bit starter model. Users never install Python, CUDA, compilers or tools.
#
#   scripts/build_all_in_one.sh                     # release build for this machine's platform
#   scripts/build_all_in_one.sh --flavor cuda       # Linux: NVIDIA training build (larger)
#   scripts/build_all_in_one.sh --smoke-model       # bundle the 1.4M-param test model (CI)
#   scripts/build_all_in_one.sh --debug             # faster debug Rust build
#   scripts/build_all_in_one.sh --skip-payload      # reuse build/payload-<platform>
#
# Build-machine requirements (never needed by users): uv, Rust (cargo), Node 22 + pnpm, CMake,
# a C++ compiler, and on Linux: patchelf, the Vulkan headers + glslc, and the Tauri Linux deps
# (libwebkit2gtk-4.1-dev, libayatana-appindicator3-dev, librsvg2-dev). macOS: Xcode CLT.
# Network on the build machine: PyPI, the pinned python-build-standalone release on GitHub, the
# PyTorch wheel index for the chosen flavor, and huggingface.co for the starter model.
#
# Signing (optional, recommended):
#   macOS: APPLE_SIGNING_IDENTITY="Developer ID Application: …", and for notarization either
#          APPLE_ID + APPLE_PASSWORD + APPLE_TEAM_ID or APPLE_API_KEY + APPLE_API_ISSUER + APPLE_API_KEY_PATH
#   Linux: SIGN_KEY (GPG key id) to detach-sign the artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLAVOR=""
SMOKE=0
DEBUG=0
SKIP_PAYLOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --flavor) FLAVOR="$2"; shift 2 ;;
    --smoke-model) SMOKE=1; shift ;;
    --debug) DEBUG=1; shift ;;
    --skip-payload) SKIP_PAYLOAD=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) PLATFORM=linux-x86_64; TRIPLE=x86_64-unknown-linux-gnu; FLAVOR="${FLAVOR:-cpu}" ;;
  Darwin-arm64) PLATFORM=macos-aarch64; TRIPLE=aarch64-apple-darwin; FLAVOR="${FLAVOR:-mlx}" ;;
  *) echo "Run this on Linux x86_64 or macOS on Apple Silicon (Windows: scripts/build_all_in_one.ps1)." >&2; exit 2 ;;
esac

PAYLOAD="$ROOT/build/payload-$PLATFORM"
OUT="$ROOT/build/installers/$PLATFORM-$FLAVOR"
VERSION="$(python3 -c "import json;print(json.load(open('src-tauri/tauri.conf.json'))['version'])")"
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing build tool: $1 ($2)" >&2; exit 2; }; }
PY() { uv run --no-project --python 3.12 python "$@"; }

step "Inkport $VERSION all-in-one build: $PLATFORM ($FLAVOR)"
need uv "https://docs.astral.sh/uv/"
need cargo "https://rustup.rs"
need pnpm "npm install -g pnpm"
need cmake "cmake.org"
if [ "$PLATFORM" = linux-x86_64 ]; then need patchelf "apt install patchelf"; need glslc "apt install glslc libvulkan-dev"; need zstd "apt install zstd"; fi

step "1/7 Frontend"
pnpm install --frozen-lockfile
pnpm build

if [ "$SKIP_PAYLOAD" = 0 ]; then
  step "2/7 Offline Python runtime (engine + libraries, pruned, precompiled)"
  PY scripts/lib/assemble_payload.py --platform "$PLATFORM" --flavor "$FLAVOR" --payload "$PAYLOAD"

  step "3/7 llama.cpp for $PLATFORM"
  LLAMA_ARGS=()
  if [ "$FLAVOR" = cuda ]; then
    need nvcc "CUDA toolkit on the build machine (users only need the NVIDIA driver)"
    LLAMA_ARGS+=(--cuda)
  fi
  PY scripts/lib/build_llamacpp.py --platform "$PLATFORM" --out "$PAYLOAD/llama" "${LLAMA_ARGS[@]}"

  step "4/7 Starter model"
  if [ "$SMOKE" = 1 ]; then
    PY scripts/lib/starter_model.py --platform "$PLATFORM" --payload "$PAYLOAD" --smoke
  else
    PY scripts/lib/starter_model.py --platform "$PLATFORM" --payload "$PAYLOAD"
  fi
fi

if [ "$PLATFORM" = macos-aarch64 ] && [ -n "${APPLE_SIGNING_IDENTITY:-}" ]; then
  step "5/7 Signing every Mach-O inside the payload (hardened runtime)"
  # Tauri signs the app and seals Contents/Resources; nested executables and libraries must carry
  # their own Developer ID signatures for notarization.
  while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then
      if [ "$APPLE_SIGNING_IDENTITY" = "-" ]; then
        codesign --force --sign - "$f"   # ad-hoc (unsigned test builds): Apple Silicon requires a signature
      else
        codesign --force --timestamp --options runtime --entitlements src-tauri/entitlements.plist --sign "$APPLE_SIGNING_IDENTITY" "$f"
      fi
    fi
  done < <(find "$PAYLOAD" -type f \( -perm -u+x -o -name "*.so" -o -name "*.dylib" \) -print0)
else
  step "5/7 Signing nested binaries: skipped (APPLE_SIGNING_IDENTITY not set or not macOS)"
fi

step "6/7 Integrity manifest"
PY scripts/lib/finalize_payload.py --platform "$PLATFORM" --flavor "$FLAVOR" --payload "$PAYLOAD"
export INKPORT_MANIFEST_SHA256="$(cat "$ROOT/build/payload-$PLATFORM.manifest.sha256")"

step "7/7 Installer (Tauri)"
DBG=(); [ "$DEBUG" = 1 ] && DBG=(--debug)
PROFILE=$([ "$DEBUG" = 1 ] && echo debug || echo release)
BUNDLE_DIR="$ROOT/target/$TRIPLE/$PROFILE/bundle"
rm -rf "$OUT" "$BUNDLE_DIR"
mkdir -p "$OUT"
# The bundle configs ship the payload; plain `cargo build` / `tauri dev` never copy it.
if [ "$PLATFORM" = linux-x86_64 ]; then
  # .deb: payload copied verbatim to /usr/lib/Inkport/payload.
  NO_STRIP=1 pnpm tauri build --config src-tauri/bundle/linux-x86_64.json --target "$TRIPLE" --bundles deb "${DBG[@]}"
  cp "$BUNDLE_DIR"/deb/*.deb "$OUT/"
  rm -rf "$ROOT/target/$TRIPLE/$PROFILE/payload"   # the resource copy made for the .deb
  # AppImage: linuxdeploy rewrites the RPATH of (and strips) every ELF it finds under usr/lib,
  # which would break the payload checksums and trip over driver-provided CUDA libraries. So the
  # AppImage is built without the payload, then the untouched payload is added and the image is
  # repacked with the same appimage plugin Tauri used.
  APPIMAGE_EXTRACT_AND_RUN=1 NO_STRIP=1 pnpm tauri build --config src-tauri/bundle/linux-x86_64-appimage.json --target "$TRIPLE" --bundles appimage "${DBG[@]}"
  APPDIR="$BUNDLE_DIR/appimage/Inkport.AppDir"
  rm -f "$BUNDLE_DIR"/appimage/*.AppImage
  mkdir -p "$APPDIR/usr/lib/Inkport"
  rm -rf "$APPDIR/usr/lib/Inkport/payload"
  cp -a "$PAYLOAD" "$APPDIR/usr/lib/Inkport/payload"
  PLUGIN="$(ls "$HOME"/.cache/tauri/linuxdeploy-plugin-appimage*.AppImage | head -1)"
  (cd "$BUNDLE_DIR/appimage" && ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 LDAI_OUTPUT="$OUT/Inkport_${VERSION}_amd64.AppImage" OUTPUT="$OUT/Inkport_${VERSION}_amd64.AppImage" "$PLUGIN" --appdir "$APPDIR" >/dev/null)
  [ -f "$OUT/Inkport_${VERSION}_amd64.AppImage" ] || mv "$BUNDLE_DIR"/appimage/*.AppImage "$OUT/Inkport_${VERSION}_amd64.AppImage"
  chmod +x "$OUT/Inkport_${VERSION}_amd64.AppImage"
  # Portable build: the same AppDir (bundled WebKit and all) as a tarball: runs on any
  # glibc 2.35+ distribution without FUSE, root, or a package manager.
  STAGE="$(mktemp -d)"
  mv "$APPDIR" "$STAGE/Inkport"
  cat > "$STAGE/Inkport/inkport" <<'LAUNCH'
#!/bin/sh
# Inkport portable launcher: no installation, no root. Run ./inkport (or ./inkport-cli doctor).
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/AppRun" "$@"
LAUNCH
  cat > "$STAGE/Inkport/inkport-cli" <<'LAUNCH'
#!/bin/sh
HERE="$(cd "$(dirname "$0")" && pwd)"
INKPORT_BUNDLE_DIR="$HERE/usr/lib/Inkport/payload" exec "$HERE/usr/bin/inkport-cli" "$@"
LAUNCH
  chmod +x "$STAGE/Inkport/inkport" "$STAGE/Inkport/inkport-cli"
  tar -C "$STAGE" -cf - Inkport | zstd -T0 -10 -q -o "$OUT/Inkport_${VERSION}_linux-x86_64-portable.tar.zst"
  rm -rf "$STAGE"
else
  pnpm tauri build --config "src-tauri/bundle/$PLATFORM.json" --target "$TRIPLE" "${DBG[@]}"
  cp "$BUNDLE_DIR"/dmg/*.dmg "$OUT/"
  (cd "$BUNDLE_DIR/macos" && ditto -c -k --keepParent Inkport.app "$OUT/Inkport_${VERSION}_macos-aarch64.app.zip")
fi

if command -v sha256sum >/dev/null; then (cd "$OUT" && sha256sum ./* > SHA256SUMS); else (cd "$OUT" && shasum -a 256 ./* > SHA256SUMS); fi
if [ -n "${SIGN_KEY:-}" ]; then
  gpg --batch --yes --local-user "$SIGN_KEY" --detach-sign --armor "$OUT/SHA256SUMS"
fi

step "Done"
du -h "$OUT"/* | sed 's/^/    /'
echo "    payload: $(du -sh "$PAYLOAD" | cut -f1) uncompressed, manifest $INKPORT_MANIFEST_SHA256"
echo "    verify on a clean machine: scripts/clean_machine_check.sh <installer>"
