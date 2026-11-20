"""Finish a payload: sample data, optional-pack catalog, licenses, and the integrity manifest.

manifest.json lists every file with its size and SHA-256, grouped into components. Its own
SHA-256 is written to build/payload-<platform>.manifest.sha256, which the Rust build embeds into
the signed app binary (src-tauri/build.rs). At launch, bootstrap.rs checks the manifest against
that embedded hash and every file against the manifest, so a damaged or tampered install is
caught before anything runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import LOCK, ROOT, host_platform, log, payload_python, rmtree, run, sha256_file

COMPONENTS = ("python", "llama", "models", "samples", "packs", "licenses")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=host_platform())
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--payload", required=True)
    ap.add_argument("--app-version", default=json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text())["version"])
    a = ap.parse_args()
    payload = Path(a.payload)

    samples = payload / "samples"
    rmtree(samples)
    py = payload_python(payload, a.platform)
    gen = py if a.platform == host_platform() else Path(sys.executable)
    run([gen, "-m", "engine", "smoke-data", "--out", samples], env={"PYTHONPATH": str(ROOT)} if gen != py else None)

    catalog = json.loads((ROOT / "packaging" / "packs" / "catalog.json").read_text())
    catalog["packs"] = [p for p in catalog["packs"] if a.platform in p["platforms"] and (not p.get("flavors") or a.flavor in p["flavors"])]
    (payload / "packs").mkdir(exist_ok=True)
    (payload / "packs" / "catalog.json").write_text(json.dumps(catalog, indent=2))

    lic = payload / "licenses"
    rmtree(lic)
    lic.mkdir()
    shutil.copy2(payload / "python" / ("LICENSE.txt" if (payload / "python" / "LICENSE.txt").is_file() else "lib/python3.12/LICENSE.txt"), lic / "CPython-LICENSE.txt")
    if (payload / "llama" / "LICENSE").is_file():
        shutil.copy2(payload / "llama" / "LICENSE", lic / "llama.cpp-LICENSE.txt")
    for f in (payload / "models" / "starter").glob("LICENSE*"):
        shutil.copy2(f, lic / f"starter-model-{f.name}.txt")
    (lic / "NOTICE.txt").write_text(
        "Inkport bundles: CPython (PSF License), llama.cpp (MIT), PyTorch (BSD-3) or MLX (MIT), "
        "Hugging Face transformers/tokenizers/safetensors (Apache-2.0), PEFT (Apache-2.0), and the "
        f"starter model {LOCK['starter_model']['hf_repo']} ({LOCK['starter_model']['license']}).\n"
        "Per-package license files are inside python/**/site-packages/*.dist-info/.\n"
    )

    # Bytecode outside the precompiled runtime is never shipped (e.g. left by a local test run).
    for cache in [d for d in payload.rglob("__pycache__") if d.is_dir() and not d.relative_to(payload).as_posix().startswith("python/")]:
        rmtree(cache)
    files = sorted(p for p in payload.rglob("*") if p.is_file() and not p.is_symlink() and p.name != "manifest.json")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        digests = list(pool.map(sha256_file, files))
    entries = {p.relative_to(payload).as_posix(): [p.stat().st_size, d] for p, d in zip(files, digests)}
    components = {}
    for comp in COMPONENTS:
        items = {k: v for k, v in entries.items() if k.split("/", 1)[0] == comp}
        h = hashlib.sha256("".join(f"{k}\0{v[0]}\0{v[1]}\n" for k, v in sorted(items.items())).encode()).hexdigest()
        components[comp] = {"files": len(items), "bytes": sum(v[0] for v in items.values()), "tree_sha256": h}
    exe = ".exe" if a.platform.startswith("windows") else ""
    # Point at the real interpreter file, not the python3 -> python3.12 symlink (installers copy
    # symlinks as plain files, and the manifest only lists regular files).
    py_rel = "python/python.exe" if exe else (payload / "python" / "bin" / "python3").resolve().relative_to(payload.resolve()).as_posix()
    if a.platform.startswith("windows"):
        # NSIS installs per-user to %LOCALAPPDATA%\Programs\Inkport (~45 chars); keep every path
        # comfortably below the classic 260-character Windows limit.
        longest = max(entries, key=len)
        print(f"[inkport-build] longest payload path: {len(longest)} chars ({longest})", flush=True)
        too_long = [k for k in entries if len(k) > 190]
        if too_long:
            raise SystemExit(f"{len(too_long)} payload paths are too long for Windows, e.g. {too_long[0]}")
    starter = json.loads((payload / "models" / "starter" / "model.json").read_text())
    manifest = {
        "format": "inkport-payload",
        "version": 1,
        "app_version": a.app_version,
        "platform": a.platform,
        "flavor": a.flavor,
        "created_at": int(time.time()),
        "python": {"exe": py_rel, "version": LOCK["python"]["version"]},
        "llama_server": f"llama/llama-server{exe}" if (payload / "llama" / f"llama-server{exe}").is_file() else None,
        "llama_convert": "llama/convert" if (payload / "llama" / "convert").is_dir() else None,
        "starter_model": {"dir": "models/starter", **starter},
        "samples": "samples",
        "packs_catalog": "packs/catalog.json",
        "components": components,
        "critical": [k for k in entries if k == py_rel or k.endswith(("llama-server", "llama-server.exe", "model.json")) or (k.startswith("python/") and "/site-packages/engine/" in k and k.endswith(".py"))],
        "files": entries,
    }
    raw = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    (payload / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (payload.parent / f"{payload.name}.manifest.sha256").write_text(digest + "\n")
    total = sum(v[0] for v in entries.values())
    log(f"manifest: {len(entries)} files, {total / 1e9:.2f} GB, hashed in {time.monotonic() - t0:.1f}s, sha256 {digest[:16]}…")
    for comp, c in components.items():
        log(f"  {comp:9s} {c['files']:6d} files {c['bytes'] / 1e6:9.1f} MB")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
