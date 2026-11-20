"""Build llama.cpp's `llama-server` (and the build-time `llama-quantize`) for one platform.

Accelerators, chosen per platform so nothing foreign ships:
  linux-x86_64   CPU (all x86 micro-architecture variants, loaded at runtime) + Vulkan (+ CUDA with --cuda)
  windows-x86_64 CPU (all variants) + Vulkan (+ CUDA with --cuda)
  macos-aarch64  CPU + Metal (shaders embedded in the library)
With GGML_BACKEND_DL the server probes backends at startup and silently falls back to CPU when no
GPU, driver or Vulkan loader is present, so the same files run on every machine of that OS.

Output layout (copied into payload/llama/):
  llama-server[.exe]  + all ggml/llama shared libraries next to it (rpath $ORIGIN / @loader_path)
  convert/            convert_lora_to_gguf.py, convert_hf_to_gguf.py, conversion/, gguf-py/   (adapter export)
  LICENSE
Build-only tools (llama-quantize) go to build/tools/ and are NOT shipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from common import CACHE, ROOT, host_platform, llama_source, log, rmtree, run

SHARED_SUFFIXES = (".so", ".dylib", ".dll")


def cmake_args(plat: str, cuda: bool, vulkan: bool = True) -> list[str]:
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_OPENSSL=OFF",  # no HTTPS client inside the server: it can never download anything
        "-DGGML_NATIVE=OFF",
        # No embedded web UI: Inkport runs llama-server with --no-webui, and the prebuilt-UI
        # download from a Hugging Face bucket is non-hermetic (it broke CI with partial assets).
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
    ]
    if plat == "macos-aarch64":
        args += [
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_BLAS=ON",
            "-DGGML_BLAS_VENDOR=Apple",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3",
            "-DCMAKE_INSTALL_RPATH=@loader_path",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_MACOSX_RPATH=ON",
        ]
    else:
        args += ["-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON", f"-DGGML_VULKAN={'ON' if vulkan else 'OFF'}"]
        if cuda:
            args += ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75-virtual;80-virtual;86-real;89-real;90-virtual"]
        if plat == "linux-x86_64":
            args += ["-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"]
    return args


def collect_libraries(bindir: Path, dest: Path, plat: str) -> list[str]:
    """Copy each shared library once, under the name the loader asks for.

    Linux: real file libX.so.1.2.3 is installed as its soname libX.so.1 (no symlink chains, so
    installers never triple-copy). Backend plugins (libggml-cpu-*.so, libggml-vulkan.so) keep
    their names because ggml loads them by name at runtime. macOS/Windows: copied as built.
    """
    import re

    names = []
    for p in sorted(bindir.iterdir()):
        if not p.is_file() or not any(s in p.name for s in SHARED_SUFFIXES):
            continue
        if plat == "linux-x86_64":
            if p.is_symlink():
                continue
            m = re.match(r"^(lib.+\.so\.\d+)\.\d+\.\d+$", p.name)
            name = m.group(1) if m else p.name
        else:
            name = p.name
        shutil.copy2(p, dest / name, follow_symlinks=True)
        names.append(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=host_platform())
    ap.add_argument("--out", required=True, help="payload/llama directory")
    ap.add_argument("--tools-out", default=str(ROOT / "build" / "tools"))
    ap.add_argument("--cuda", action="store_true", help="also build the CUDA backend (needs the CUDA toolkit on the build machine)")
    ap.add_argument("--no-vulkan", action="store_true", help="skip the Vulkan backend (CPU-only inference on Windows/Linux)")
    ap.add_argument("--jobs", default=str(os.cpu_count() or 4))
    a = ap.parse_args()
    if a.platform != host_platform():
        raise SystemExit("llama.cpp is compiled natively: run this on the target OS")

    src = llama_source()
    build = CACHE / f"llama-build-{a.platform}{'-cuda' if a.cuda else ''}"
    run(["cmake", "-S", src, "-B", build, *cmake_args(a.platform, a.cuda, not a.no_vulkan)])
    run(["cmake", "--build", build, "--config", "Release", "-j", a.jobs, "--target", "llama-server", "llama-quantize"])
    # Single-config generators put everything in bin/; MSVC puts it in bin/Release/.
    bindir = build / "bin" / "Release" if (build / "bin" / "Release").is_dir() else build / "bin"

    out, tools = Path(a.out), Path(a.tools_out)
    rmtree(out)
    out.mkdir(parents=True)
    tools.mkdir(parents=True, exist_ok=True)
    exe = ".exe" if a.platform.startswith("windows") else ""
    copied = collect_libraries(bindir, out, a.platform)
    shutil.copy2(bindir / f"llama-server{exe}", out / f"llama-server{exe}")
    shutil.copy2(bindir / f"llama-quantize{exe}", tools / f"llama-quantize{exe}")
    collect_libraries(bindir, tools, a.platform)  # llama-quantize needs the same libraries at build time
    log(f"collected {len(copied)} shared libraries")
    conv = out / "convert"
    conv.mkdir()
    for f in ("convert_lora_to_gguf.py", "convert_hf_to_gguf.py"):
        shutil.copy2(src / f, conv / f)
    shutil.copytree(src / "conversion", conv / "conversion", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(src / "gguf-py" / "gguf", conv / "gguf", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(src / "LICENSE", out / "LICENSE")
    if a.platform == "linux-x86_64":
        # Every ELF looks next to itself first, then prove nothing resolves outside the payload
        # except the C/C++ runtime and (optionally, at runtime) the system Vulkan loader.
        if shutil.which("patchelf"):
            for p in out.iterdir():
                if ".so" in p.name or p.name == "llama-server":
                    run(["patchelf", "--set-rpath", "$ORIGIN", p])
        import subprocess

        ldd = subprocess.run(["ldd", str(out / "llama-server")], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}).stdout
        missing = [ln.strip() for ln in ldd.splitlines() if "not found" in ln]
        if missing:
            raise SystemExit("llama-server has unresolved libraries: " + "; ".join(missing))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"llama.cpp for {a.platform}: {len(list(out.iterdir()))} entries, {size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
