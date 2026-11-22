import json
from pathlib import Path

import pytest

from engine.contract import read_jsonl
from engine.pipeline.ingest_and_verify import run_ingest
from engine.smoke import write_smoke_data
from engine.verify.core import Context, load_verifier_set, score_item


ROOT = Path(__file__).resolve().parents[2]


def _retemplate(project, template):
    p = json.loads((project / "project.json").read_text())
    p["template"] = template
    (project / "project.json").write_text(json.dumps(p))


def test_code_sql_reference_answers_pass_their_own_verifiers(project, tmp_path):
    _retemplate(project, "code_sql")
    data = tmp_path / "d"
    write_smoke_data(data)
    run_ingest(project, [str(data / "orders.csv"), str(data / "customers.csv"), str(data / "helpers.py")], [])
    vset, ctx = load_verifier_set(project), Context(root=project)
    items = read_jsonl(project / "data" / "train.jsonl") + read_jsonl(project / "data" / "test.jsonl")
    assert any(i["meta"].get("gold_sql") for i in items) and any(i["meta"].get("tests") for i in items)
    for it in items[:60]:
        r = score_item(it, it["reference"], vset, ctx)
        assert r["pass"], (it["messages"][-1]["content"], r["verdicts"])
    sql_item = next(i for i in items if i["meta"].get("gold_sql"))
    assert not score_item(sql_item, '{"sql": "SELECT 12345"}', vset, ctx)["pass"]


def test_docs_reference_answers_pass_citation_check(project, tmp_path):
    _retemplate(project, "docs_legal")
    data = tmp_path / "d"
    write_smoke_data(data)
    summary = run_ingest(project, [str(data / "lease.md")], [])
    assert sum(summary["counts"].values()) >= 5
    vset, ctx = load_verifier_set(project), Context(root=project)
    for split in ("train", "dev", "test"):
        for it in read_jsonl(project / "data" / f"{split}.jsonl"):
            assert score_item(it, it["reference"], vset, ctx)["pass"]
            fake = json.dumps({"answer": it["meta"]["answer"], "quote": "This sentence is not in the lease."})
            assert not score_item(it, fake, vset, ctx)["pass"]


SAMPLE_FILES = {"business_rules": ["expenses.csv", "policy.md"], "code_sql": ["orders.csv", "customers.csv", "helpers.py"], "docs_legal": ["lease.md"]}


@pytest.mark.parametrize("template", sorted(SAMPLE_FILES))
def test_try_the_sample_data_gives_a_trainable_exam_with_every_suggested_rule(project, tmp_path, template):
    """The first thing a new user does: pick a goal, press "Try the sample data", add the suggested
    rules and build. That path has to end in an exam that can be trained on, with every suggested
    rule in use (two suggestions once contradicted the sample they were shown next to, and the
    sample lease was too short to hold an exam at all)."""
    _retemplate(project, template)
    data = tmp_path / "samples"
    write_smoke_data(data)
    suggested = next(t for t in json.loads((ROOT / "contracts" / "templates.json").read_text(encoding="utf-8"))["templates"] if t["id"] == template)["example_rules"]
    assert suggested
    summary = run_ingest(project, [str(data / name) for name in SAMPLE_FILES[template]], suggested)
    assert summary["ready"] and summary["blocker"] is None, summary
    assert summary["counts"]["test"] >= 30, summary["counts"]
    assert [r["ok"] for r in summary["rules"]] == [True] * len(suggested), summary["rules"]
    assert not any(r.get("warning") for r in summary["rules"]), summary["rules"]
    assert not summary["warnings"], summary["warnings"]
