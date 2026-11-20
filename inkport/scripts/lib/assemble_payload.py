"""Assemble the offline Python runtime for one platform inside build/payload-<platform>/python.

1. Download the pinned python-build-standalone CPython (SHA-256 verified) and unpack it. It is
   relocatable: it finds its standard library relative to its own executable, so it runs from
   wherever the installer puts it, with no system Python involved.
2. Install the engine and its dependencies *into that interpreter* (not a venv). PyTorch comes
   from the flavor's index (CPU wheels for the core installer, CUDA wheels for the NVIDIA
   flavor); Apple Silicon gets MLX and no PyTorch at all.
3. Prune dead weight (headers, import libs, tests, Tk, pip, JIT tooling: see packaging/prune.json).
4. Precompile every module with `--invalidation-mode unchecked-hash`, so imports are fast on
   first launch and never try to write .pyc files into a read-only, signed app bundle.

When the build host differs from the target platform, wheels are installed with uv's
--python-platform/--target (pure download, nothing executed), which lets CI assemble the
Windows and macOS runtimes from Linux. Native code (llama.cpp, the Tauri app) is still built
on each target OS.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import CACHE, LOCK, ROOT, TRIPLES, die, download, extract_tar, host_platform, llama_source, log, payload_python, rmtree, run

FLAVORS = {"linux-x86_64": ("cpu", "cuda"), "windows-x86_64": ("cpu", "cuda"), "macos-aarch64": ("mlx",)}


def site_packages(py_root: Path, plat: str) -> Path:
    return py_root / ("Lib/site-packages" if plat.startswith("windows") else "lib/python3.12/site-packages")


def uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        die("uv is required on the build machine (https://docs.astral.sh/uv/)")
    return exe


def install(plat: str, py_root: Path, args: list[str], cross: bool) -> None:
    base = [uv(), "pip", "install", "--link-mode", "copy", "--no-progress"]
    if cross:
        base += ["--target", str(site_packages(py_root, plat)), "--python-platform", TRIPLES[plat], "--python-version", "3.12"]
    else:
        base += ["--python", str(payload_python(py_root.parent, plat)), "--break-system-packages"]
    run(base + args)


def prune(py_root: Path, plat: str) -> int:
    rules = json.loads((ROOT / "packaging" / "prune.json").read_text())
    patterns = rules["common"] + rules.get(plat, [])
    removed = 0
    paths = sorted(py_root.rglob("*"), key=lambda p: len(p.parts))
    for p in paths:
        if not p.exists() and not p.is_symlink():
            continue
        rel = p.relative_to(py_root).as_posix()
        if any(fnmatch.fnmatchcase(rel, pat) for pat in patterns):  # case-sensitive: Windows must not match Lib/threading.py against lib/thread*
            if p.is_symlink():
                size = 0
            elif p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
            else:
                size = p.stat().st_size
            rmtree(p)
            removed += size
    return removed


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=host_platform())
    ap.add_argument("--flavor", help="cpu | cuda (Windows/Linux), mlx (macOS)")
    ap.add_argument("--payload", required=True)
    a = ap.parse_args()
    plat = a.platform
    flavor = a.flavor or FLAVORS[plat][0]
    if flavor not in FLAVORS[plat]:
        die(f"flavor {flavor} is not available for {plat} (choose {FLAVORS[plat]})")
    cross = plat != host_platform()
    payload = Path(a.payload)
    py_root = payload / "python"

    spec = LOCK["python"]["archives"][plat]
    archive = download(f"{LOCK['python']['base_url']}/{spec['file'].replace('+', '%2B')}", CACHE / spec["file"], spec["sha256"], spec["size"])
    rmtree(py_root)
    tmp = CACHE / f"python-unpack-{plat}"
    rmtree(tmp)
    extract_tar(archive, tmp)
    shutil.move(str(tmp / "python"), str(py_root))
    log(f"CPython {LOCK['python']['version']} for {plat} unpacked ({tree_size(py_root) / 1e6:.0f} MB)")

    req = ROOT / "packaging" / "requirements" / f"{plat}-{flavor}.txt"
    if flavor in ("cpu", "cuda"):
        index = LOCK["torch_index"][flavor][plat]
        install(plat, py_root, ["torch>=2.6", "--index-url", index], cross)
    install(plat, py_root, ["-r", str(req)], cross)
    # The GGUF reader/writer used by transformers *and* by the bundled converters, from the same
    # pinned llama.cpp source so they always agree.
    install(plat, py_root, [str(llama_source() / "gguf-py")], cross)
    wheel_dir = CACHE / "engine-wheel"
    rmtree(wheel_dir)
    run([uv(), "build", "--wheel", "--out-dir", str(wheel_dir), str(ROOT)])
    install(plat, py_root, ["--no-deps", "--reinstall", str(next(wheel_dir.glob("inkport_engine-*.whl")))], cross)

    before = tree_size(py_root)
    removed = prune(py_root, plat)
    log(f"pruned {removed / 1e6:.0f} MB of unused files ({before / 1e6:.0f} -> {tree_size(py_root) / 1e6:.0f} MB)")

    compile_py = payload_python(payload, plat) if not cross else Path(subprocess.check_output([uv(), "python", "find", "3.12"], text=True).strip())
    run([compile_py, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash", str(py_root)])

    freeze: list[str] = []
    if not cross:
        py = payload_python(payload, plat)
        check = "import engine, sys; import " + ("mlx.core, mlx_lm" if flavor == "mlx" else "torch, transformers, peft") + "; print(sys.version.split()[0])"
        run([py, "-I", "-c", check])
        freeze = subprocess.check_output([uv(), "pip", "freeze", "--python", str(py)], text=True).split()
    info = {"python": LOCK["python"]["version"], "platform": plat, "flavor": flavor, "cross_assembled": cross, "packages": freeze}
    (py_root / "inkport-runtime.json").write_text(json.dumps(info, indent=2))
    log(f"runtime ready: {tree_size(py_root) / 1e6:.0f} MB, {sum(1 for _ in py_root.rglob('*'))} entries")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
