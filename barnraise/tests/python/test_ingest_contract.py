import json
from pathlib import Path

import jsonschema

from engine.contract import contracts_dir, load_contract, read_jsonl
from engine.hardware_profile import select_tier
from engine.pipeline.ingest_and_verify import merkle_root, run_ingest
from engine.smoke import write_smoke_data

SCHEMAS = contracts_dir() / "schemas"


def _schema(name):
    return json.loads((SCHEMAS / name).read_text())


def test_tier_cases_match_contract():
    for case in json.loads((contracts_dir() / "fixtures" / "tier_cases.json").read_text())["cases"]:
        assert select_tier(case)["tier"] == case["expect"], case["name"]


def test_event_fixtures_validate():
    schema = _schema("event.schema.json")
    for line in (contracts_dir() / "fixtures" / "events.ndjson").read_text().splitlines():
        jsonschema.validate(json.loads(line), schema)


def test_ingest_no_leakage_and_schemas(project, tmp_path, monkeypatch):
    import io

    from engine import events

    buf = io.StringIO()
    monkeypatch.setattr(events, "_stream", buf)
    data = tmp_path / "data"
    write_smoke_data(data)
    summary = run_ingest(project, [str(data / "expenses.csv"), str(data / "policy.md")], ["category must be one of fuel, meals, software, office_supplies, travel", "amount must be positive"])
    assert summary["leakage_free"]
    splits = {s: read_jsonl(project / "data" / f"{s}.jsonl") for s in ("train", "dev", "test")}
    ids = {s: {i["item_id"] for i in rows} for s, rows in splits.items()}
    groups = {s: {i["group"] for i in rows} for s, rows in splits.items()}
    assert not (ids["train"] & ids["test"]) and not (groups["train"] & groups["test"]) and not (groups["dev"] & groups["test"])
    assert len(splits["test"]) >= 30
    item_schema = _schema("item.schema.json")
    for rows in splits.values():
        for it in rows:
            jsonschema.validate(it, item_schema)
    jsonschema.validate(json.loads((project / "data" / "verifiers.json").read_text()), _schema("verifier.schema.json"))
    manifest = json.loads((project / "data" / "test_manifest.json").read_text())
    assert manifest["merkle_root"] == merkle_root([i["item_id"] for i in splits["test"]])
    # Every event emitted during ingestion conforms to the shared contract.
    out = buf.getvalue()
    schema = _schema("event.schema.json")
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    assert lines
    for ln in lines:
        jsonschema.validate(json.loads(ln), schema)


def test_split_is_deterministic(project, tmp_path):
    data = tmp_path / "data"
    write_smoke_data(data)
    run_ingest(project, [str(data / "expenses.csv")], [])
    first = [i["item_id"] for i in read_jsonl(project / "data" / "test.jsonl")]
    run_ingest(project, [str(data / "expenses.csv")], [])
    assert first == [i["item_id"] for i in read_jsonl(project / "data" / "test.jsonl")]


def test_near_duplicates_share_a_split(project, tmp_path):
    rows = ["description,category,amount"]
    for i in range(80):
        rows.append(f"Paid ${i + 10} at Shell for the delivery van,fuel,{i + 10}")
        rows.append(f"Paid ${i + 10} at Shell for the delivery van.,fuel,{i + 10}")
    f = tmp_path / "dupes.csv"
    f.write_text("\n".join(rows))
    run_ingest(project, [str(f)], [])
    by_group = {}
    for s in ("train", "dev", "test"):
        for it in read_jsonl(project / "data" / f"{s}.jsonl"):
            by_group.setdefault(it["group"], set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())


def test_templates_contract_lists_four_cards():
    assert [t["id"] for t in load_contract("templates.json")["templates"]] == ["business_rules", "code_sql", "docs_legal", "custom"]
