"""Ingestion must cope with real-world files and always produce a usable split, or say why not."""

import io
import json
import zipfile

import pytest

from engine import events
from engine.contract import read_jsonl
from engine.pipeline.ingest_and_verify import MIN_TEST_ITEMS, preview, run_ingest

BASE = [("Paid $42 at Shell for the delivery van", "fuel"), ("Zoom monthly plan", "software"), ("Staples printer paper", "office_supplies"), ("Lunch with client at Chipotle", "meals"), ("Hilton two nights", "travel")]


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(events, "_stream", io.StringIO())


def _set_template(project, template):
    cfg = json.loads((project / "project.json").read_text())
    cfg["template"] = template
    (project / "project.json").write_text(json.dumps(cfg))


def _rows(n):
    return [(f"{BASE[i % 5][0]} #{i}", BASE[i % 5][1], 10 + i) for i in range(n)]


def _splits(project):
    return {s: read_jsonl(project / "data" / f"{s}.jsonl") for s in ("train", "dev", "test")}


def test_european_excel_csv_semicolons_decimal_commas_cp1252(project, tmp_path):
    lines = ["Beschreibung;Kategorie;Betrag"] + [f"{d} – Café;{c};{a},50" for d, c, a in _rows(120)]
    f = tmp_path / "ausgaben.csv"
    f.write_bytes("\r\n".join(lines).encode("cp1252"))
    summary = run_ingest(project, [str(f)], [])
    assert summary["ready"] and summary["blocker"] is None
    assert summary["template_info"]["fields"] == ["Kategorie", "Betrag"]
    assert summary["template_info"]["numeric_fields"] == ["Betrag"]
    items = [it for rows in _splits(project).values() for it in rows]
    assert len(items) == 120
    by_prompt = {it["messages"][-1]["content"]: it["reference_json"] for it in items}
    assert by_prompt["Zoom monthly plan #1 – Café"] == {"Kategorie": "software", "Betrag": 11.5}


def test_templated_examples_still_get_a_balanced_split(project, tmp_path):
    # 200 rows that all read alike used to form a handful of "near-duplicate" groups, and whole
    # groups landed in one split (once: 0 training examples, 120 in dev).
    f = tmp_path / "expenses.csv"
    f.write_text("description,category,amount\n" + "\n".join(f"{d},{c},{a}" for d, c, a in _rows(200)))
    summary = run_ingest(project, [str(f)], [])
    assert summary["counts"] == {"train": 120, "dev": 20, "test": 60}
    assert summary["ready"]
    splits = _splits(project)
    assert not ({it["group"] for it in splits["train"]} & {it["group"] for it in splits["test"]})
    assert not ({it["item_id"] for it in splits["train"]} & {it["item_id"] for it in splits["test"]})
    # Same files again: the same questions stay in the hidden exam.
    before = {it["item_id"] for it in splits["test"]}
    run_ingest(project, [str(f)], [])
    assert {it["item_id"] for it in _splits(project)["test"]} == before


def test_too_few_examples_is_reported_before_training(project, tmp_path):
    f = tmp_path / "few.csv"
    f.write_text("description,category,amount\n" + "\n".join(f"{d},{c},{a}" for d, c, a in _rows(12)))
    summary = run_ingest(project, [str(f)], [])
    assert summary["ready"] is False
    assert "12 different examples" in summary["blocker"] and "at least 30" in summary["blocker"]
    assert summary["counts"]["test"] < MIN_TEST_ITEMS
    again = preview(project, 4)
    assert again["ready"] is False and again["blocker"] == summary["blocker"]


def test_forty_examples_are_enough_to_train(project, tmp_path):
    f = tmp_path / "forty.csv"
    f.write_text("description,category,amount\n" + "\n".join(f"{d},{c},{a}" for d, c, a in _rows(40)))
    summary = run_ingest(project, [str(f)], [])
    assert summary["ready"] and summary["counts"]["test"] >= MIN_TEST_ITEMS and summary["counts"]["train"] >= 12
    assert any("preliminary" in w for w in summary["warnings"])


def test_numbers_as_descriptions_and_stray_cells(project, tmp_path):
    f = tmp_path / "codes.csv"
    f.write_text("description,category,amount\n" + "\n".join(f"{1000 + i},{BASE[i % 5][1]},{i}" for i in range(60)) + "\nstray,row,1,extra,cells\n")
    summary = run_ingest(project, [str(f)], [])
    assert summary["template_info"]["fields"] == ["category", "amount"]  # one row's extra cells do not add fields
    assert sum(summary["counts"].values()) == 61


def test_problems_with_files_are_part_of_the_error(project, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    header = tmp_path / "header.csv"
    header.write_text("description,category,amount\n")
    old = tmp_path / "book.xls"
    old.write_bytes(b"\xd0\xcf\x11\xe0 old excel")
    with pytest.raises(ValueError) as err:
        run_ingest(project, [str(empty), str(header), str(old), str(tmp_path / "gone.csv")], [])
    msg = str(err.value)
    assert "empty.csv is empty" in msg and "header.csv has no rows" in msg and "old Excel format" in msg and "gone.csv was not found" in msg


def test_preview_remembers_what_the_exam_was_built_from(project, tmp_path):
    f = tmp_path / "expenses.csv"
    f.write_text("description,category,amount\n" + "\n".join(f"{d},{c},{a}" for d, c, a in _rows(80)))
    run_ingest(project, [str(f)], ["amount must be positive"])
    pv = preview(project, 3)
    assert pv["inputs"] == [str(f)]
    assert [r["text"] for r in pv["rules"]] == ["amount must be positive"]
    assert pv["ready"] and pv["leakage_free"] and len(pv["items"]) == 3


def test_custom_examples_in_the_shapes_people_have(project, tmp_path):
    _set_template(project, "custom")
    qa = [(f"What is the code for item {i}?", f"CODE-{i}") for i in range(60)]
    chat = tmp_path / "chat.jsonl"
    chat.write_text("﻿" + "\n".join(json.dumps({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]}) for q, a in qa) + "\n{not json}\n", encoding="utf-8")
    summary = run_ingest(project, [str(chat)], [])
    assert summary["ready"] and sum(summary["counts"].values()) == 60
    assert any("1 line(s)" in w for w in summary["warnings"])
    first = _splits(project)["train"][0]
    assert first["messages"][-1]["content"].startswith("What is the code") and first["reference"].startswith("CODE-")

    for header in ("Q,A", "instruction,output", "Frage,Antwort"):
        f = tmp_path / "qa.csv"
        f.write_text(header + "\n" + "\n".join(f'"{q}",{a}' for q, a in qa))
        summary = run_ingest(project, [str(f)], [])
        assert sum(summary["counts"].values()) == 60, header
    assert any("was used as the question" in w for w in summary["warnings"])  # unnamed columns: first is the question

    braces = tmp_path / "braces.csv"
    braces.write_text("question,answer\n" + "\n".join(f'"{q}","{{curly}} {a}"' for q, a in qa))
    summary = run_ingest(project, [str(braces)], [])
    assert summary["template_info"]["answer_style"] == "short"  # "{curly} ..." is text, not broken JSON


def test_word_document_is_read(project, tmp_path):
    _set_template(project, "docs_legal")
    facts = [f"Clause {i}: the fee of ${100 + i} is due within {i + 2} days of March {i % 27 + 1}, 2026. " for i in range(60)]
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    path = tmp_path / "contract.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f"<w:document {w}><w:body>" + "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in facts) + "</w:body></w:document>")
    summary = run_ingest(project, [str(path)], [])
    assert sum(summary["counts"].values()) >= 30 and summary["ready"]


def test_documents_without_blank_lines_are_cut_into_excerpts():
    from engine.pipeline.ingest_and_verify import chunk_document

    # PDF and Word text: one newline per line or paragraph, never a blank line.
    text = "\n".join(f"Clause {i}: the fee of ${100 + i} is due within {i + 2} days." for i in range(120))
    chunks = chunk_document(text)
    assert len(chunks) > 5 and max(len(c) for c in chunks) <= 760
    assert " ".join(chunks) == " ".join(text.split())  # nothing lost, nothing repeated
    # A wall of text with no sentence punctuation is still cut, at spaces.
    wall = chunk_document("word " * 2000)
    assert len(wall) > 10 and max(len(c) for c in wall) <= 760
    # Normal prose with blank lines keeps whole paragraphs together.
    prose = chunk_document("First paragraph. It has two sentences.\n\nSecond paragraph.")
    assert prose == ["First paragraph. It has two sentences.\nSecond paragraph."]  # paragraphs keep their own line
