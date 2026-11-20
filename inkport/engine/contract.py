"""Shared contract loading and project file layout.

Contracts (tiers, templates, JSON schemas) live in <repo>/contracts and are bundled into the
wheel as engine/_contracts, so the same files drive Rust, Python and the UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


@lru_cache(maxsize=1)
def contracts_dir() -> Path:
    env = os.environ.get("INKPORT_CONTRACTS")
    candidates = [Path(env)] if env else []
    here = Path(__file__).resolve().parent
    candidates += [here / "_contracts", here.parent / "contracts"]
    for c in candidates:
        if (c / "tiers.json").is_file():
            return c
    raise FileNotFoundError("Inkport contracts not found (looked in: " + ", ".join(map(str, candidates)) + ")")


@lru_cache(maxsize=None)
def load_contract(name: str) -> Any:
    return json.loads((contracts_dir() / name).read_text(encoding="utf-8"))


def template_meta(template_id: str) -> dict:
    for t in load_contract("templates.json")["templates"]:
        if t["id"] == template_id:
            return t
    raise KeyError(f"unknown template: {template_id}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Stable digest of a directory: sha256 over sorted (relative path, file sha256) pairs."""
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(b"\0")
        h.update(sha256_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def project_json(self) -> Path:
        return self.root / "project.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    def split(self, name: str) -> Path:
        return self.data / f"{name}.jsonl"

    @property
    def verifiers(self) -> Path:
        return self.data / "verifiers.json"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.json"

    @property
    def test_manifest(self) -> Path:
        return self.data / "test_manifest.json"

    @property
    def tables_db(self) -> Path:
        return self.data / "tables.sqlite"

    @property
    def tables_db_perturbed(self) -> Path:
        return self.data / "tables_perturbed.sqlite"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def versions(self) -> Path:
        return self.root / "versions"

    @property
    def active(self) -> Path:
        return self.root / "active.json"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id


def load_project(root: Path) -> dict:
    paths = ProjectPaths(root)
    if not paths.project_json.is_file():
        raise FileNotFoundError(f"not a Inkport project (missing project.json): {root}")
    return json.loads(paths.project_json.read_text(encoding="utf-8"))


def save_project(root: Path, project: dict) -> None:
    atomic_write_json(ProjectPaths(root).project_json, project)


def active_version(root: Path) -> dict | None:
    p = ProjectPaths(root).active
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def iter_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    """Expand files and folders (recursively), skipping hidden files and VCS folders."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode"}
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_file():
            yield p
        elif p.is_dir():
            for sub in sorted(p.rglob("*")):
                if any(part in skip_dirs or part.startswith(".") for part in sub.relative_to(p).parts):
                    continue
                if sub.is_file():
                    yield sub
