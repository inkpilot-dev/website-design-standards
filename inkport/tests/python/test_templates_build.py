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
    # Every split: the sample has only two questions about Python functions, and where they land is
    # drawn per project (reading train and test alone made this fail about once in a hundred runs).
    items = [it for split in ("train", "dev", "test") for it in read_jsonl(project / "data" / f"{split}.jsonl")]
    assert any(i["meta"].get("gold_sql") for i in items) and any(i["meta"].get("tests") for i in items)
    items.sort(key=lambda i: not i["meta"].get("tests"))  # the function questions first: they are among those checked below
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


SAMPLE_FILES = {"business_rules": ["expenses.csv", "policy.md"], "code_sql": ["orders.csv", "customers.csv", "helpers.py"], "docs_legal": ["lease.md"], "custom": ["support.csv"]}


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


def test_document_headings_are_context_not_quiz_sentences(project, tmp_path):
    from engine.pipeline.ingest_and_verify import chunk_document, cloze_items, preview

    text = "# Lease\n\n## 1. Parties and premises\n\nThis lease is made on February 12, 2026 between Harborview Property Group and Maria Okafor. The rent is $1,450 per month.\n\n## 2. Term\nThe term is 12 months.\n"
    chunks = chunk_document(text)
    assert chunks == ["# Lease\n## 1. Parties and premises\nThis lease is made on February 12, 2026 between Harborview Property Group and Maria Okafor. The rent is $1,450 per month.\n## 2. Term\nThe term is 12 months."]
    items = cloze_items({"path": "lease.md", "text": text})
    quotes = [it["reference_json"]["quote"] for it in items]
    # A heading used to be glued to the sentence after it, and was quizzed as a sentence itself.
    assert quotes and all(not q.startswith("#") and "Parties and premises" not in q and "Term" not in q.split()[0] for q in quotes), quotes
    assert quotes[0].startswith("This lease is made on")
    # Text without blank lines (PDF, Word) is still packed sentence by sentence into several excerpts.
    long = " ".join(f"Clause {i} sets a fee of ${i * 10} payable within {i + 3} days." for i in range(1, 120))
    assert len(chunk_document(long)) > 5 and all(len(c) <= 760 for c in chunk_document(long))

    _retemplate(project, "docs_legal")
    data = tmp_path / "s"
    write_smoke_data(data)
    run_ingest(project, [str(data / "lease.md")], [])
    cards = preview(project, 8)["items"]
    assert all(c["question"].startswith("Fill in the blank") and c["source"] == "lease.md" for c in cards), cards[:2]
    assert not any(c["reference"].startswith('{"answer": "Lease Agreement"') for c in cards)


def test_generated_sql_questions_read_naturally(project, tmp_path):
    from engine.pipeline.ingest_and_verify import preview

    _retemplate(project, "code_sql")
    data = tmp_path / "s"
    write_smoke_data(data)
    run_ingest(project, [str(data / "orders.csv"), str(data / "customers.csv"), str(data / "helpers.py")], [])
    prompts = [it["messages"][-1]["content"] for split in ("train", "dev", "test") for it in read_jsonl(project / "data" / f"{split}.jsonl")]
    assert not any("total total" in p for p in prompts)
    assert any("highest total?" in p for p in prompts)
    sources = {c["source"] for c in preview(project, 12)["items"]}
    assert sources <= {"your tables", "helpers.py"}, sources


STYLE = [
    ("How are you today?", "I am as merry as a lark at dawn, good friend. And how dost thou fare this day?"),
    ("Good night.", "Good night, sweet friend. May soft sleep wrap thee till thou wakest with the lark."),
    ("I am tired.", "Then lay thee down and rest. Sleep knits up the ravelled sleeve of care, and thou hast earned it."),
    ("What is love?", "Love is a fire that warms yet may burn. Guard it well, and thou shalt never want for light."),
]


def _style_file(tmp_path, n=60):
    import csv

    path = tmp_path / "style.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for i in range(n):
            q, a = STYLE[i % len(STYLE)]
            w.writerow([f"{q} (asked on day {i})", f"{a} So say I on day {i}, and thou mayst mark it well."])
    return path


def test_a_style_is_judged_by_the_rules_not_by_the_wording_of_one_expected_answer(project, tmp_path):
    """Seen with 120 Shakespeare-style examples and the rules 'must mention "thou"' and 'no more
    than 50 words': every answer of the trained model was marked wrong, also "Good morrow, thou
    hast a kind heart. May the day be fair to thee.", because it shared too few words with the one
    expected answer. The verdict was 0% against 0%, whatever was trained."""
    _retemplate(project, "custom")
    summary = run_ingest(project, [str(_style_file(tmp_path))], ['must mention "thou"', "no more than 50 words"])
    assert summary["ready"] and all(r["ok"] for r in summary["rules"]), summary
    vset, ctx = load_verifier_set(project), Context(root=project)
    item = next(iter(read_jsonl(project / "data" / "dev.jsonl")))
    assert score_item(item, "Good morrow, thou hast a kind heart. May the day be fair to thee.", vset, ctx)["pass"]
    assert not score_item(item, "Hello! How can I assist you today?", vset, ctx)["pass"]
    assert not score_item(item, "thou " * 60, vset, ctx)["pass"]


def test_a_style_without_any_rule_says_that_a_rule_is_needed(project, tmp_path):
    _retemplate(project, "custom")
    summary = run_ingest(project, [str(_style_file(tmp_path))], [])
    assert any("Add at least one rule" in w for w in summary["warnings"]), summary["warnings"]


def test_set_replies_asked_in_several_ways_still_have_to_say_the_same_thing(project, tmp_path):
    # The bundled customer-replies sample: a handful of answers, each asked in many ways.
    _retemplate(project, "custom")
    data = tmp_path / "d"
    write_smoke_data(data)
    run_ingest(project, [str(data / "support.csv")], ["no more than 50 words"])
    vset, ctx = load_verifier_set(project), Context(root=project)
    item = next(iter(read_jsonl(project / "data" / "dev.jsonl")))
    assert score_item(item, item["reference"], vset, ctx)["pass"]
    assert not score_item(item, "Thank you, we will look into it.", vset, ctx)["pass"]
