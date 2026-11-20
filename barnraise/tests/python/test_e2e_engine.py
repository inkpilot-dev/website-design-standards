"""Slow: builds the tiny model and runs ingest -> train -> gate -> promote -> export -> import."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(*args, env):
    out = subprocess.run([sys.executable, "-m", "engine", *args], cwd=ROOT, capture_output=True, text=True, env=env, timeout=1800)
    events = [json.loads(ln) for ln in out.stdout.splitlines() if ln.startswith("{")]
    done = events[-1]
    assert done["type"] == "done", out.stderr[-2000:]
    return done, events


@pytest.mark.slow
def test_full_engine_pipeline(tmp_path):
    import os

    env = {**os.environ, "BARNRAISE_HOME": str(tmp_path / "home")}
    done, _ = run("smoke-model", "--out", str(tmp_path / "model"), env=env)
    assert done["ok"]
    run("smoke-data", "--out", str(tmp_path / "data"), env=env)
    done, _ = run("create-project", "--root", str(tmp_path / "p"), "--name", "Expense Sorter", "--template", "business_rules", "--base-id", "smoke-tiny", "--base-path", str(tmp_path / "model"), "--backend", "torch", "--quant", "fp32", env=env)
    proj = done["result"]["project_dir"]
    done, _ = run("ingest", "--project", proj, "--inputs", str(tmp_path / "data" / "expenses.csv"), str(tmp_path / "data" / "policy.md"), "--rules", json.dumps(["category must be one of fuel, meals, software, office_supplies, travel", "amount must be positive"]), env=env)
    assert done["ok"] and done["result"]["leakage_free"]
    done, events = run("train", "--project", proj, "--preset", "smoke", env=env)
    assert done["ok"], done
    assert done["result"]["decision"] == "PROMOTE", done["result"]["explanation"]
    assert (Path(proj) / "active.json").is_file()
    done, _ = run("export", "--project", proj, "--out", str(tmp_path / "v1.brpatch"), env=env)
    assert done["ok"] and len(done["result"]["sha256"]) == 64
