"""Load config.yaml and resolve its paths."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml

_LOCK = threading.Lock()


def load(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["root"] = path.parent
    cfg["data_dir"] = path.parent / cfg["dataset"]["dir"]
    cfg["system_prompt"] = f"{cfg['identity'].strip()} {cfg['instruction'].strip()}"
    return cfg


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # a half-written last line means the program was killed mid-write; the rest is intact
    return rows


def write_jsonl(path: Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row and flush, so an interrupted run loses nothing already paid for."""
    with _LOCK:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
