"""Shared helpers for the all-in-one build (standard library only; runs on the build machine).

Every download is resume-safe (HTTP Range into a .part file) and verified against a SHA-256
before it is used. Nothing here ever runs on a user's machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads((ROOT / "packaging" / "lock.json").read_text())
CACHE = Path(os.environ.get("INKPORT_BUILD_CACHE", ROOT / "build" / "cache"))
PLATFORMS = ("linux-x86_64", "macos-aarch64", "windows-x86_64")
TRIPLES = {"linux-x86_64": "x86_64-unknown-linux-gnu", "macos-aarch64": "aarch64-apple-darwin", "windows-x86_64": "x86_64-pc-windows-msvc"}


def log(msg: str) -> None:
    print(f"[inkport-build] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[inkport-build] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def host_platform() -> str:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "linux-x86_64"
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "macos-aarch64"
    if system == "Windows" and machine in ("amd64", "x86_64"):
        return "windows-x86_64"
    die(f"unsupported build host {system}/{machine} (supported: {', '.join(PLATFORMS)})")
    raise SystemExit(1)


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(url: str, dest: Path, sha256: Optional[str] = None, size: Optional[int] = None, headers: Optional[dict] = None, retries: int = 5) -> Path:
    """Resume-safe download. Verifies sha256 (if given) before moving into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (sha256 is None or sha256_file(dest) == sha256):
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "inkport-build/1.0", **(headers or {})})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0  # server ignored Range; start over
                total = int(r.headers.get("Content-Length") or 0) + have
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    last = time.monotonic()
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if time.monotonic() - last > 5 and total:
                            log(f"  {dest.name}: {have / 1e6:.0f}/{total / 1e6:.0f} MB")
                            last = time.monotonic()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                die(f"download failed after {retries} attempts: {url} ({exc})")
            log(f"  retry {attempt}/{retries} for {dest.name}: {exc}")
            time.sleep(min(30, 2**attempt))
    if size is not None and part.stat().st_size != size:
        die(f"{dest.name}: size {part.stat().st_size} != expected {size}")
    if sha256 is not None:
        got = sha256_file(part)
        if got != sha256:
            part.unlink()
            die(f"{dest.name}: SHA-256 mismatch (expected {sha256}, got {got})")
    os.replace(part, dest)
    return dest


def extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12
            t.extractall(dest)


def run(cmd: Iterable[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if proc.returncode != 0:
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def rmtree(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    elif p.exists() or p.is_symlink():
        p.unlink()


def payload_python(payload: Path, plat: str) -> Path:
    return payload / "python" / ("python.exe" if plat.startswith("windows") else "bin/python3")


def llama_source() -> Path:
    """Download (pinned + verified) and extract the llama.cpp sources; returns the source root."""
    spec = LOCK["llama_cpp"]
    if os.environ.get("INKPORT_LLAMA_SRC"):
        return Path(os.environ["INKPORT_LLAMA_SRC"])
    archive = download(spec["url"], CACHE / Path(spec["url"]).name, spec["sha256"], spec["size"])
    out = CACHE / f"llama-src-{spec['version']}"
    if not (out / spec["subdir"] / "CMakeLists.txt").is_file():
        rmtree(out)
        extract_tar(archive, out)
    return out / spec["subdir"]
