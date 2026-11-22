"""Rules are gates, so a rule has to agree with the user's own examples before it is used."""

import io
import json

import pytest

from engine import events
from engine.contract import read_jsonl
from engine.pipeline.ingest_and_verify import rules_preview, run_ingest
from engine.verify.core import run_verifier, Context

BASE = [("Paid $42 at Shell for the delivery van", "fuel"), ("Zoom monthly plan", "software"), ("Staples printer paper", "office_supplies"), ("Lunch with client at Chipotle", "meals"), ("Uber to the airport", "travel")]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(events, "_stream", io.StringIO())


def _csv(tmp_path, header, rows, name="expenses.csv"):
    f = tmp_path / name
    f.write_text(header + "\n" + "\n".join(",".join(str(c) for c in row) for row in rows), encoding="utf-8")
    return str(f)


def _rows(n):
    return [(f"{BASE[i % 5][0]} #{i}", BASE[i % 5][1], 10 + i) for i in range(n)]


def _items(project):
    return [it for s in ("train", "dev", "test") for it in read_jsonl(project / "data" / f"{s}.jsonl")]


def _report(summary, text):
    return next(r for r in summary["rules"] if r["text"] == text)


def test_rules_find_columns_whatever_their_capitals_and_spaces(project, tmp_path):
    # Spreadsheet headers are rarely typed the way a rule is.
    f = _csv(tmp_path, "Description,Expense Category,Total Amount ($)", _rows(120))
    rules = ["total amount must be positive", "Expense category must be one of fuel, software, office_supplies, meals, travel", "expense categories must not be empty"]
    summary = run_ingest(project, [f], rules)
    assert [r["ok"] for r in summary["rules"]] == [True, True, True], summary["rules"]
    assert not any(r.get("warning") for r in summary["rules"])
    items = _items(project)
    assert len(items) == 120
    # The reference answers pass their own gates, and a wrong answer does not.
    specs = {v["id"]: v for v in json.loads((project / "data" / "verifiers.json").read_text())["verifiers"]}
    rule_ids = [i for i in specs if i.startswith("rule:")]
    assert len(rule_ids) == 3
    it = items[0]
    ctx = Context(project)
    good = it["reference_json"]
    bad = {**good, "Total Amount ($)": -5}
    for rid in rule_ids:
        assert run_verifier(specs[rid], it, json.dumps(good), good, ctx)["pass"] is True
    assert run_verifier(specs["rule:1"], it, json.dumps(bad), bad, ctx)["pass"] is False


def test_rule_about_a_column_that_does_not_exist_is_not_used(project, tmp_path):
    f = _csv(tmp_path, "description,category,amount", _rows(120))
    summary = run_ingest(project, [f], ["price must be positive", "amount must be positive"])
    bad, good = _report(summary, "price must be positive"), _report(summary, "amount must be positive")
    assert not bad["ok"] and bad["depends_on_data"]
    assert "no column called “price”" in bad["error"] and "category, amount" in bad["error"]
    assert good["ok"]
    assert any("price must be positive" in w for w in summary["warnings"])
    specs = json.loads((project / "data" / "verifiers.json").read_text())["verifiers"]
    assert [v["params"]["text"] for v in specs if v["kind"] == "rule"] == ["amount must be positive"]
    # The model is told the rules as well: only the one in use may appear in its instructions.
    instructions = {it["messages"][0]["content"] for it in _items(project)}
    assert len(instructions) == 1
    assert "amount must be positive" in next(iter(instructions)) and "price" not in next(iter(instructions))
    assert all(sum(v.startswith("rule:") for v in it["verifiers"]) == 1 for it in _items(project))


def test_rule_that_the_examples_do_not_follow_is_not_used(project, tmp_path):
    rows = _rows(120)
    rows = [(d, c, -a if i % 3 == 0 else a) for i, (d, c, a) in enumerate(rows)]  # a third are refunds
    f = _csv(tmp_path, "description,category,amount", rows)
    summary = run_ingest(project, [f], ["amount must be positive", "amount must be negative and under -5000", "amount must be reasonably small"])
    many, none, vague = summary["rules"]
    assert not many["ok"] and "40 of your 120 examples break this rule" in many["error"] and "for instance “" in many["error"]
    assert not none["ok"] and "None of your 120 examples follow this rule" in none["error"]
    assert not vague["ok"] and "reasonably small" in vague["error"]
    assert len(_items(project)) == 120  # nothing was thrown away for rules that were not used
    assert summary["ready"]


def test_a_few_examples_that_break_a_rule_are_left_out_with_a_warning(project, tmp_path):
    rows = _rows(120)
    rows[7] = ("Refund for order 1182", "software", -30)
    rows[50] = ("Chargeback from the bank", "meals", -12)
    f = _csv(tmp_path, "description,category,amount", rows)
    summary = run_ingest(project, [f], ["amount must be positive"])
    rule = summary["rules"][0]
    assert rule["ok"]
    assert "2 of your 120 examples break this rule and were left out" in rule["warning"]
    assert "Refund for order 1182" in rule["warning"]
    prompts = {it["messages"][-1]["content"] for it in _items(project)}
    assert len(prompts) == 118 and "Refund for order 1182" not in prompts
    assert any("left out" in w for w in summary["warnings"])


def test_condition_on_the_question_column(project, tmp_path):
    f = _csv(tmp_path, "description,category,amount", _rows(120))
    rules = ["if the description mentions uber then category must be travel", "if category is fuel, amount must be under 500", "amount must be positive unless category is refund"]
    summary = run_ingest(project, [f], rules)
    assert [r["ok"] for r in summary["rules"]] == [True, True, True], summary["rules"]
    assert summary["rules"][0]["expr"].startswith("(not (mentions(input,")
    specs = {v["id"]: v for v in json.loads((project / "data" / "verifiers.json").read_text())["verifiers"]}
    uber = next(it for it in _items(project) if "Uber" in it["messages"][-1]["content"])
    ctx = Context(project)
    wrong = {**uber["reference_json"], "category": "meals"}
    assert run_verifier(specs["rule:1"], uber, json.dumps(uber["reference_json"]), uber["reference_json"], ctx)["pass"] is True
    assert run_verifier(specs["rule:1"], uber, json.dumps(wrong), wrong, ctx)["pass"] is False
    other = next(it for it in _items(project) if "Zoom" in it["messages"][-1]["content"])
    assert run_verifier(specs["rule:1"], other, json.dumps(other["reference_json"]), other["reference_json"], ctx)["pass"] is True


def test_column_rule_on_plain_text_answers_is_explained(project, tmp_path):
    cfg = json.loads((project / "project.json").read_text())
    cfg["template"] = "custom"
    (project / "project.json").write_text(json.dumps(cfg))
    f = tmp_path / "qa.csv"
    f.write_text("question,answer\n" + "\n".join(f"What is {i} plus {i}?,It is {2 * i}. Thank you for asking" for i in range(80)), encoding="utf-8")
    summary = run_ingest(project, [str(f)], ["amount must be positive", "no more than 12 words", 'must mention "thank you"', "must not mention sorry or apologies"])
    column, words, thanks, sorry = summary["rules"]
    assert not column["ok"] and "plain text" in column["error"]
    assert words["ok"] and thanks["ok"] and sorry["ok"]


def test_rules_typed_after_the_exam_is_built_are_checked_on_the_examples(project, tmp_path):
    f = _csv(tmp_path, "Description,Category,Amount", _rows(120))
    before = rules_preview(project, ["price must be positive"])["rules"][0]
    assert before["ok"]  # nothing to compare with yet: only the wording can be judged
    run_ingest(project, [f], [])
    after = rules_preview(project, ["price must be positive", "AMOUNT must be at least 10", "if description mentions zoom then category must be software"])["rules"]
    assert not after[0]["ok"] and "no column called “price”" in after[0]["error"]
    assert after[1]["ok"] and after[2]["ok"]
    assert rules_preview(None, ["amount must be positive", "make it nice"])["rules"][1]["ok"] is False
