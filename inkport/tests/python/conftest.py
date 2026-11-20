import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """A business_rules project whose base model path is a stub (ingestion never loads it)."""
    monkeypatch.setenv("BARNRAISE_HOME", str(tmp_path / "home"))
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"id": "p1", "slug": "p1", "name": "p1", "template": "business_rules", "base_model": {"id": "stub", "path": str(tmp_path / "model"), "chat_kwargs": {}}, "backend": "torch", "quant": "fp32"}))
    return root
