"""Data ingestion and auto-verifier studio.

raw files (CSV/TSV, Excel, PDF, Word, Markdown, TXT, JSON/JSONL, Python; see engine/readers.py)
  -> parsed sources (records, tables, documents, code)
  -> template-specific items (prompt + reference + verifier ids)
  -> exact dedup (SHA-256 of normalized prompt) + near-duplicate grouping (MinHash LSH)
  -> group-level train / dev / test split by HMAC-SHA256(project salt, group)
  -> leakage proof (no shared item hash or group across splits) + Merkle root of the test set
  -> verifiers.json (deterministic checks + compiled plain-English rules + self-tests)

Every file this module writes conforms to contracts/schemas/item.schema.json and
contracts/schemas/verifier.schema.json.
"""

from __future__ import annotations

import ast
import csv
import doctest
import hashlib
import io
import json
import random
import re
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from engine import events, readers
from engine.contract import ProjectPaths, atomic_write_json, iter_files, load_project, read_jsonl, save_project, sha256_file, write_jsonl
from engine.templates import CODE_SYSTEM, CUSTOM_SYSTEM, DOCS_SYSTEM, SQL_SYSTEM, business_system
from engine.textnorm import near_duplicate_groups, normalize, prompt_hash, split_fraction, word_count
from engine.verify import rules as rules_mod
from engine.verify import sql as sql_mod

MAX_FILE_BYTES = 50 * 1024 * 1024
INPUT_KEYS = ["text", "message", "description", "input", "prompt", "question", "email", "memo", "details", "transaction", "note", "query"]
OUTPUT_KEYS = ["answer", "output", "response", "completion", "reply", "target", "label"]
ID_KEYS = {"id", "row_id", "index", "idx", "uuid", "#"}
# Question/answer column names people use in example files (the Custom Skill template only).
QUESTION_KEYS = INPUT_KEYS + ["q", "instruction", "request", "user", "ask", "input_text"]
ANSWER_KEYS = OUTPUT_KEYS + ["a", "assistant", "expected", "expected_answer", "expected_output", "result", "ground_truth", "gold", "output_text"]
# The gate cannot decide anything on fewer hidden questions than this (see Trainer.gate).
MIN_TEST_ITEMS = 15
MIN_TRAIN_ITEMS = 8

# ------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------


@dataclass
class Sources:
    records: list[tuple[str, dict]] = field(default_factory=list)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    documents: list[dict] = field(default_factory=list)
    code: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _table_name(path: Path, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_]", "_", path.stem.lower()).strip("_") or "table"
    if base[0].isdigit():
        base = "t_" + base
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


def _json_rows(obj: Any) -> list:
    """Records inside a .json file: a list, or a list stored under a key such as "examples" or "rows"."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("examples", "data", "rows", "items", "records", "entries"):
            if isinstance(obj.get(key), list):
                return obj[key]
        for value in obj.values():
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                return value
    return []


def parse_sources(paths: list[str], template: str) -> Sources:
    src = Sources()
    for raw in paths:
        if not Path(raw).expanduser().exists():
            src.warnings.append(f"{Path(raw).name} was not found. It may have been moved, renamed or deleted after you added it.")
    files = list(iter_files(paths))
    prog = events.Progress("parse", max(len(files), 1), 2, 40)
    for i, path in enumerate(files):
        ext = path.suffix.lower()
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                src.warnings.append(f"{path.name} was skipped because it is larger than 50 MB.")
                continue
            if size == 0:
                src.warnings.append(f"{path.name} is empty.")
                continue
            digest = sha256_file(path)
            kind = "other"
            if ext in (".csv", ".tsv", ".tab", ".xlsx", ".xlsm"):
                rows, warns = readers.read_table(path)
                src.warnings += [f"{path.name}: {w}" for w in warns]
                if not rows:
                    src.warnings.append(f"{path.name} has no rows of data (only a header row, or an empty sheet).")
                if template == "code_sql":
                    if rows:
                        src.tables[_table_name(path, set(src.tables))] = rows
                    kind = "table"
                else:
                    src.records += [(f"{path.name}:{n + 2}", r) for n, r in enumerate(rows)]
                    kind = "records"
            elif ext in (".jsonl", ".ndjson"):
                bad = 0
                for n, line in enumerate(readers.read_text(path).splitlines()):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if isinstance(obj, dict):
                        src.records.append((f"{path.name}:{n + 1}", obj))
                    else:
                        bad += 1
                if bad:
                    src.warnings.append(f"{path.name}: {bad} line(s) are not valid JSON objects and were skipped.")
                kind = "records"
            elif ext == ".json":
                try:
                    obj = json.loads(readers.read_text(path))
                except ValueError as exc:
                    raise readers.UnreadableFile(f"is not valid JSON ({exc}).") from exc
                rows = _json_rows(obj)
                if not any(isinstance(r, dict) for r in rows):
                    src.warnings.append(f"{path.name} does not contain a list of examples.")
                src.records += [(f"{path.name}:{n + 1}", r) for n, r in enumerate(rows) if isinstance(r, dict)]
                kind = "records"
            elif ext == ".pdf":
                pages = readers.read_pdf(path)
                src.documents.append({"path": path.name, "text": "\n".join(pages), "pages": len(pages)})
                kind = "document"
                if sum(len(p) for p in pages) < 50 * max(len(pages), 1):
                    src.warnings.append(f"{path.name} has almost no text that can be selected; it looks like a scan (pictures of pages). Inkport cannot read scans yet. If you have the original document, add that instead.")
            elif ext == ".docx":
                src.documents.append({"path": path.name, "text": readers.read_docx(path), "pages": None})
                kind = "document"
            elif ext in (".md", ".markdown", ".txt", ".rst"):
                src.documents.append({"path": path.name, "text": readers.read_text(path), "pages": None})
                kind = "document"
            elif ext == ".py":
                src.code.append({"path": path.name, "source": readers.read_text(path)})
                kind = "code"
            elif ext in readers.LEGACY_HINTS:
                src.warnings.append(f"{path.name} {readers.LEGACY_HINTS[ext]}")
                continue
            else:
                src.warnings.append(f"{path.name} was skipped: Inkport cannot read '{ext or 'this kind of'}' files.")
                continue
            src.files.append({"path": str(path), "name": path.name, "sha256": digest, "bytes": size, "kind": kind})
        except readers.UnreadableFile as exc:
            src.warnings.append(f"{path.name} {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingestion
            src.warnings.append(f"{path.name} could not be read ({type(exc).__name__}: {exc}).")
        prog.update(i + 1, {"files": float(i + 1)})
    return src


# ------------------------------------------------------------------------------------------
# Template builders. Each returns (items, verifiers, info).
# Items here are pre-split: {"messages", "reference", "reference_json", "verifiers", "meta", "group_hint"?}
# ------------------------------------------------------------------------------------------


def _vj(vid: str, kind: str, role: str, weight: float = 1.0, description: str = "", **params: Any) -> dict:
    return {"id": vid, "kind": kind, "role": role, "weight": weight, "description": description, "params": params}


def _pick_key(keys: list[str], preferred: list[str]) -> Optional[str]:
    lower = {k.lower(): k for k in keys}
    for p in preferred:
        if p in lower:
            return lower[p]
    return None


def build_business(src: Sources, rule_texts: list[str]) -> tuple[list[dict], list[dict], dict]:
    if not src.records:
        raise ValueError("No examples were found. Add a spreadsheet of past examples (Excel or CSV) with a header row, for example: description, category, amount.")
    keys = list(dict.fromkeys(k for _, r in src.records for k in r.keys()))
    input_key = _pick_key(keys, INPUT_KEYS)
    if input_key is None:
        str_keys = [k for k in keys if all(isinstance(r.get(k), str) for _, r in src.records[:50] if r.get(k) is not None)]
        if not str_keys:
            raise ValueError("Could not find a text column to learn from (for example 'description').")
        input_key = max(str_keys, key=lambda k: sum(len(str(r.get(k) or "")) for _, r in src.records[:200]))
    fields = [k for k in keys if k != input_key and k.lower() not in ID_KEYS]
    if not fields:
        raise ValueError("Your examples need at least one column to fill in besides the text (for example 'category').")
    numeric = {f for f in fields if all(isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool) for _, r in src.records if r.get(f) is not None)}
    enums: dict[str, list[str]] = {}
    for f in fields:
        if f in numeric:
            continue
        values = [str(r[f]) for _, r in src.records if r.get(f) is not None]
        distinct = sorted(set(values))
        if 1 < len(distinct) <= 50 and len(distinct) <= 0.5 * len(values):
            enums[f] = distinct
    policy = "\n\n".join(d["text"] for d in src.documents)[:1500]
    system = business_system(fields, enums, rule_texts, policy)
    schema = {
        "type": "object",
        "required": fields,
        "properties": {f: ({"type": "number"} if f in numeric else {"type": ["string", "null"]}) for f in fields},
    }
    verifiers = [
        _vj("json_valid", "json_valid", "gate", description="Reply is a JSON object"),
        _vj("json_schema", "json_schema", "gate", description="Has every field with the right type", schema=schema),
    ]
    for f in fields:
        verifiers.append(_vj(f"field:{f}", "field_match", "reward", 1.0, f"{f} matches your record", field=f, mode="numeric" if f in numeric else "exact", tol_abs=0.01))
    items = []
    for source, r in src.records:
        text = r.get(input_key)
        if isinstance(text, (int, float)) and not isinstance(text, bool):
            text = str(text)  # a description that happens to be a number is still a description
        if not isinstance(text, str) or not text.strip():
            continue
        ref = {f: r.get(f) for f in fields}
        items.append(
            {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text.strip()}],
                "reference": json.dumps(ref, ensure_ascii=False),
                "reference_json": ref,
                "verifiers": [v["id"] for v in verifiers],
                "meta": {"source": source, "origin": "user_examples"},
            }
        )
    if not items:
        raise ValueError(f"None of the rows has text in the '{input_key}' column, so there is nothing to learn from.")
    return items, verifiers, {"input_field": input_key, "fields": fields, "numeric_fields": sorted(numeric), "choices": enums}


# --- code & SQL ------------------------------------------------------------------------------


def _schema_text(schema: dict[str, list[tuple[str, str]]], tables: dict[str, list[dict]]) -> str:
    parts = []
    for t, cols in schema.items():
        parts.append(f'CREATE TABLE {t} ({", ".join(f"{c} {ty}" for c, ty in cols)});')
        for row in tables[t][:2]:
            parts.append(f"-- example row: {json.dumps(row, ensure_ascii=False)}")
    return "\n".join(parts)


def _q(name: str) -> str:
    return name if re.fullmatch(r"[a-z_][a-z0-9_]*", name) else f'"{name}"'


def _lit(v: Any) -> str:
    return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "'" + str(v).replace("'", "''") + "'"


def sql_questions(schema: dict[str, list[tuple[str, str]]], tables: dict[str, list[dict]], seed: int = 0, per_table: int = 80) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []

    def add(key: str, phrasings: list[str], gold: str) -> None:
        for p in phrasings:
            out.append({"question": p, "sql": gold, "group_hint": key})

    for t, cols in schema.items():
        rows = tables[t]
        nums = [c for c, ty in cols if ty in ("INTEGER", "REAL") and c.lower() not in ID_KEYS and not c.lower().endswith("_id")]
        cats = []
        for c, ty in cols:
            if ty != "TEXT":
                continue
            vals = [r.get(c) for r in rows if r.get(c) is not None]
            distinct = sorted(set(map(str, vals)))
            if 1 < len(distinct) <= 30 and len(distinct) < len(vals):
                cats.append((c, distinct))
        T = _q(t)
        add(f"{t}:count", [f"How many rows are in the {t} table?", f"Count all records in {t}."], f"SELECT COUNT(*) FROM {T}")
        budget = per_table
        for c, distinct in cats:
            C = _q(c)
            add(f"{t}:distinct:{c}", [f"How many different {c} values appear in {t}?"], f"SELECT COUNT(DISTINCT {C}) FROM {T}")
            for v in rng.sample(distinct, min(len(distinct), 6)):
                add(f"{t}:countwhere:{c}={v}", [f"How many {t} rows have {c} equal to '{v}'?", f"Count the {t} where {c} is {v}."], f"SELECT COUNT(*) FROM {T} WHERE {C} = {_lit(v)}")
                for n in nums[:3]:
                    N = _q(n)
                    add(f"{t}:sumwhere:{n}:{c}={v}", [f"What is the {_total(n)} for {c} '{v}'?", f"Add up {n} where {c} is {v}."], f"SELECT SUM({N}) FROM {T} WHERE {C} = {_lit(v)}")
                    add(f"{t}:maxwhere:{n}:{c}={v}", [f"What is the largest {n} where {c} is '{v}'?"], f"SELECT MAX({N}) FROM {T} WHERE {C} = {_lit(v)}")
                    budget -= 3
            for n in nums[:3]:
                N = _q(n)
                add(f"{t}:groupsum:{n}:{c}", [f"List each {c} with its {_total(n)}, highest first.", f"Show {_total(n)} per {c}, sorted from highest to lowest."], f"SELECT {C}, SUM({N}) AS total FROM {T} GROUP BY {C} ORDER BY total DESC")
                add(f"{t}:top:{n}:{c}", [f"Which {c} has the highest {_total(n)}?"], f"SELECT {C} FROM {T} GROUP BY {C} ORDER BY SUM({N}) DESC LIMIT 1")
                add(f"{t}:groupavg:{n}:{c}", [f"What is the average {n} for each {c}?"], f"SELECT {C}, AVG({N}) FROM {T} GROUP BY {C}")
            if budget <= 0:
                break
        for n in nums[:4]:
            N = _q(n)
            add(f"{t}:avg:{n}", [f"What is the average {n} in {t}?"], f"SELECT AVG({N}) FROM {T}")
            add(f"{t}:sum:{n}", [f"What is the {_total(n)} across all {t}?"], f"SELECT SUM({N}) FROM {T}")
            add(f"{t}:top3:{n}", [f"Show the 3 {t} rows with the highest {n}."], f"SELECT * FROM {T} ORDER BY {N} DESC LIMIT 3")
            vals = sorted(r[n] for r in rows if isinstance(r.get(n), (int, float)))
            if vals:
                thr = vals[len(vals) // 2]
                add(f"{t}:countgt:{n}", [f"How many {t} have {n} greater than {thr}?"], f"SELECT COUNT(*) FROM {T} WHERE {N} > {_lit(thr)}")
    # Joins: tables sharing a column name ending in _id.
    names = list(schema)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = {c for c, _ in schema[a]} & {c for c, _ in schema[b]}
            keys = [c for c in shared if c.lower().endswith("_id") or c.lower() == "id"]
            if not keys:
                continue
            k = keys[0]
            a_nums = [c for c, ty in schema[a] if ty in ("INTEGER", "REAL") and c not in shared and not c.lower().endswith("_id")]
            b_cats = [c for c, ty in schema[b] if ty == "TEXT" and c not in shared]
            for n in a_nums[:2]:
                for c in b_cats[:2]:
                    add(f"join:{a}:{b}:{n}:{c}", [f"What is the {_total(n)} from {a} for each {c} in {b}?"], f"SELECT {_q(b)}.{_q(c)}, SUM({_q(a)}.{_q(n)}) FROM {_q(a)} JOIN {_q(b)} ON {_q(a)}.{_q(k)} = {_q(b)}.{_q(k)} GROUP BY {_q(b)}.{_q(c)}")
    return out


def doctest_items(code_files: list[dict]) -> list[dict]:
    parser = doctest.DocTestParser()
    out = []
    for cf in code_files:
        try:
            tree = ast.parse(cf["source"])
        except SyntaxError:
            continue
        lines = cf["source"].splitlines()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            doc = ast.get_docstring(node) or ""
            examples = [e for e in parser.get_examples(doc) if e.want.strip()]
            if not examples:
                continue
            tests = [{"call": e.source.strip(), "expect": e.want.strip()} for e in examples]
            shown = examples[: max(1, len(examples) // 2)]
            desc = doc.split(">>>")[0].strip()
            sig = lines[node.lineno - 1].strip()
            shown_txt = "\n".join(f">>> {e.source.strip()}\n{e.want.strip()}" for e in shown)
            prompt = f"Implement this function.\n\n{sig}\n    \"\"\"{desc}\n\n    Examples:\n{shown_txt}\n    \"\"\""
            body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            out.append({"prompt": prompt, "reference": f"```python\n{body}\n```", "tests": tests, "entry": node.name, "source": f"{cf['path']}:{node.lineno}"})
    return out


def _total(column: str) -> str:
    """"total quantity", but a column that is itself called "total" is not "total total"."""
    return column if re.search(r"(?:^|[\s_])(?:total|sum)(?:$|[\s_])", column, re.I) else f"total {column}"


def build_code_sql(src: Sources, paths: ProjectPaths, rule_texts: list[str]) -> tuple[list[dict], list[dict], dict]:
    items: list[dict] = []
    verifiers: list[dict] = [_vj("json_valid", "json_valid", "gate", description="Reply is a JSON object")]
    info: dict[str, Any] = {}
    sql_system = ""
    if src.tables:
        schema = sql_mod.build_database(src.tables, paths.tables_db)
        sql_mod.build_database(src.tables, paths.tables_db_perturbed, perturb_seed=17)
        system = sql_system = SQL_SYSTEM.format(schema=_schema_text(schema, src.tables))
        verifiers += [
            _vj("sql_exec", "sql_exec", "reward", 1.0, "The query's result matches the correct result"),
            _vj("sql_exec_perturbed", "sql_exec", "audit", 0.0, "Still correct on scrambled data (catches hard-coded answers)", db="perturbed"),
        ]
        for q in sql_questions(schema, src.tables):
            items.append(
                {
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": q["question"]}],
                    "reference": json.dumps({"sql": q["sql"]}),
                    "reference_json": {"sql": q["sql"]},
                    "verifiers": ["json_valid", "sql_exec", "sql_exec_perturbed"],
                    "meta": {"source": "generated from your tables", "origin": "programmatic", "gold_sql": q["sql"]},
                    "group_hint": q["group_hint"],
                }
            )
        info["tables"] = {t: [c for c, _ in cols] for t, cols in schema.items()}
    code_items = doctest_items(src.code)
    if code_items:
        verifiers.append(_vj("unit_tests", "python_tests", "reward", 1.0, "The function passes your examples (run in a sandbox)", timeout_s=6.0))
        for c in code_items:
            items.append(
                {
                    "messages": [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": c["prompt"]}],
                    "reference": c["reference"],
                    "reference_json": None,
                    "verifiers": ["unit_tests"],
                    "meta": {"source": c["source"], "origin": "doctest", "tests": c["tests"], "entry": c["entry"]},
                }
            )
        info["functions"] = len(code_items)
    for source, r in src.records:
        q = r.get(_pick_key(list(r), ["question", "prompt", "input"]) or "")
        s = r.get(_pick_key(list(r), ["sql", "query", "answer"]) or "")
        if isinstance(q, str) and isinstance(s, str) and sql_system:
            items.append(
                {
                    "messages": [{"role": "system", "content": sql_system}, {"role": "user", "content": q}],
                    "reference": json.dumps({"sql": s}),
                    "reference_json": {"sql": s},
                    "verifiers": ["json_valid", "sql_exec", "sql_exec_perturbed"],
                    "meta": {"source": source, "origin": "user_examples", "gold_sql": s},
                }
            )
    if not items:
        raise ValueError("No practice questions could be made. Add at least one spreadsheet (Excel or CSV) with a header row and some rows of data, or Python files whose docstrings contain >>> examples.")
    return items, verifiers, info


# --- documents --------------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_SPAN_PATTERNS = [
    re.compile(r"[$€£]\s?\d[\d,.]*\d|[$€£]\s?\d"),
    re.compile(r"\b\d[\d.,]*\s?(?:€|EUR|USD|GBP|£)"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b\d+\s+(?:business\s+)?(?:days?|weeks?|months?|years?|hours?)\b", re.I),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"(?<=[a-z,;] )(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"),
    re.compile(r"[\"“]([A-Z][^\"”]{2,40})[\"”]"),
]


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def chunk_document(text: str, target: int = 700) -> list[str]:
    """Excerpts of roughly `target` characters, made of whole paragraphs or whole sentences.

    Text taken from PDFs and Word files usually has no blank lines at all (every line or paragraph
    ends in a single newline), so a "paragraph" can be the entire document. Long paragraphs are
    therefore packed sentence by sentence; without that, a 20-page contract became one excerpt
    with six questions.

    Paragraphs and headings each keep their own line inside an excerpt. Run together, a heading
    reads as the start of the sentence after it ("1. Parties This lease is made on ...").
    """
    chunks: list[str] = []
    cur = ""

    def push(piece: str, sep: str = "\n") -> None:
        nonlocal cur
        if cur and (len(cur) + len(piece) + 1 > target or (_HEADING.match(piece) and len(cur) > target * 0.6)):
            chunks.append(cur)
            cur = piece
        else:
            cur = (cur + sep + piece) if cur else piece

    def push_paragraph(para: str) -> None:
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            return
        if len(para) <= target:
            push(para)
            return
        sep = "\n"
        for sent in _SENT.split(para):
            while len(sent) > target:  # no sentence punctuation (lists, tables): cut at a space
                cut = sent.rfind(" ", 0, target)
                cut = cut if cut > target // 2 else target
                push(sent[:cut].strip(), sep)
                sep = " "
                sent = sent[cut:].strip()
            if sent:
                push(sent, sep)
                sep = " "

    for block in re.split(r"\n\s*\n", text):
        lines: list[str] = []
        for line in block.split("\n"):
            if _HEADING.match(line):
                push_paragraph(" ".join(lines))
                lines = []
                push(re.sub(r"\s+", " ", line).strip())
            else:
                lines.append(line)
        push_paragraph(" ".join(lines))
    if cur:
        chunks.append(cur)
    return chunks


def cloze_items(doc: dict, max_per_chunk: int = 6) -> list[dict]:
    out = []
    for ci, chunk in enumerate(chunk_document(doc["text"])):
        made = 0
        for sent in (x for line in chunk.split("\n") if not _HEADING.match(line) for x in _SENT.split(line)):
            if made >= max_per_chunk:
                break
            if not (30 <= len(sent) <= 320):
                continue
            seen = set()
            for pat in _SPAN_PATTERNS:
                m = pat.search(sent)
                if not m:
                    continue
                span = (m.group(1) if m.groups() else m.group(0)).strip()
                if len(span) < 2 or span in seen:
                    continue
                seen.add(span)
                masked = sent.replace(span, "_____", 1)
                prompt = f'Excerpt from {doc["path"]}:\n"""\n{chunk}\n"""\n\nFill in the blank using the excerpt: "{masked}"'
                out.append(
                    {
                        "messages": [{"role": "system", "content": DOCS_SYSTEM}, {"role": "user", "content": prompt}],
                        "reference": json.dumps({"answer": span, "quote": sent}, ensure_ascii=False),
                        "reference_json": {"answer": span, "quote": sent},
                        "verifiers": ["json_valid", "citation"],
                        "meta": {"source": f"{doc['path']}#chunk{ci}", "origin": "doc_extract", "source_text": chunk, "answer": span},
                        "group_hint": f"{doc['path']}#chunk{ci}#{normalize(sent)[:60]}",
                    }
                )
                made += 1
                break
    return out


def build_docs(src: Sources, rule_texts: list[str]) -> tuple[list[dict], list[dict], dict]:
    if not src.documents:
        raise ValueError("No readable documents were found. Add the documents you want your AI to know (PDF, Word, Markdown or text).")
    items = [it for d in src.documents for it in cloze_items(d)]
    if not items:
        raise ValueError("No facts to quiz on were found in the documents. The practice questions are made from dates, amounts of money, percentages, durations and names written in English; add longer documents that contain such details.")
    verifiers = [
        _vj("json_valid", "json_valid", "gate", description="Reply is a JSON object with answer and quote"),
        _vj("citation", "citation", "reward", 1.0, "Quotes the document word for word and gives the right answer"),
    ]
    return items, verifiers, {"documents": len(src.documents)}


# --- custom ----------------------------------------------------------------------------------


def _chat_pair(messages: Any) -> Optional[tuple[str, Any]]:
    """(last user turn, last assistant turn) of an OpenAI-style chat example."""

    def text(content: Any) -> str:
        if isinstance(content, list):  # [{"type": "text", "text": "..."}]
            return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return "" if content is None else str(content)

    if not isinstance(messages, list):
        return None
    user = next((text(m.get("content")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"), "")
    reply = next((text(m.get("content")) for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), "")
    return (user, reply) if user.strip() and reply.strip() else None


def _json_object(answer: Any) -> Optional[dict]:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str) and answer.strip().startswith("{"):
        try:
            value = json.loads(answer)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None


def build_custom(src: Sources, rule_texts: list[str]) -> tuple[list[dict], list[dict], dict]:
    pairs = []
    for source, r in src.records:
        chat = _chat_pair(r.get("messages"))
        if chat:
            pairs.append((source, chat[0], chat[1]))
            continue
        qk = _pick_key(list(r), QUESTION_KEYS)
        ak = _pick_key([k for k in r if k != qk], ANSWER_KEYS)
        if qk and ak and r.get(qk) is not None and r.get(ak) is not None:
            pairs.append((source, str(r[qk]), r[ak]))
    if not pairs and src.records:
        # A two-column table without recognisable names: question first, answer second.
        columns = list(dict.fromkeys(k for _, r in src.records for k in r))
        if len(columns) == 2:
            qk, ak = columns
            pairs = [(source, str(r[qk]), r[ak]) for source, r in src.records if r.get(qk) is not None and r.get(ak) is not None]
            if pairs:
                src.warnings.append(f"The columns were not named, so '{qk}' was used as the question and '{ak}' as the expected answer.")
    if not pairs:
        if not src.records:
            raise ValueError("No examples were found. Add a file with one example per row: a question column and an answer column.")
        raise ValueError("Your examples need a question column and an answer column. Name them, for example, 'question' and 'answer' (also understood: prompt/input/instruction and output/response/completion).")
    parsed = [_json_object(a) for _, _, a in pairs]
    json_answers = all(p is not None for p in parsed)
    avg_words = sum(word_count(str(a)) for _, _, a in pairs) / len(pairs)
    verifiers: list[dict]
    if json_answers:
        verifiers = [_vj("json_valid", "json_valid", "gate"), _vj("exact", "exact_match", "reward", 1.0, "Matches your expected JSON exactly")]
    elif avg_words <= 12:
        verifiers = [_vj("exact", "exact_match", "reward", 1.0, "Matches your expected answer")]
    else:
        verifiers = [
            _vj("overlap", "token_f1", "reward", 1.0, "Covers the same content as your expected answer", threshold=0.6),
            _vj("judge", "pairwise_judge", "eval_only", 0.0, "A local judge model prefers it over the original model"),
        ]
    items = []
    for (source, q, a), ref_json in zip(pairs, parsed):
        if not json_answers:
            ref_json = None
        if not q.strip():
            continue
        ref = json.dumps(ref_json, ensure_ascii=False) if ref_json is not None else str(a)
        items.append(
            {
                "messages": [{"role": "system", "content": CUSTOM_SYSTEM}, {"role": "user", "content": q}],
                "reference": ref,
                "reference_json": ref_json,
                "verifiers": [v["id"] for v in verifiers],
                "meta": {"source": source, "origin": "user_examples"},
            }
        )
    return items, verifiers, {"examples": len(items), "answer_style": "json" if json_answers else ("short" if avg_words <= 12 else "long")}


# ------------------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------------------


def _snippet(text: str, limit: int = 60) -> str:
    one_line = " ".join(str(text).split())
    return "“" + (one_line if len(one_line) <= limit else one_line[: limit - 1].rstrip() + "…") + "”"


def compile_rules(rule_texts: list[str], items: list[dict], info: Optional[dict] = None) -> tuple[list[dict], list[dict], set[int]]:
    """Returns (verifier specs, per-rule reports, positions of the examples to leave out).

    A rule is a gate: an answer that breaks it fails, whatever else it gets right. So a rule that
    the user's own examples do not follow cannot be kept as it is, or no answer could ever pass.
    A rule that names a column the examples do not have, or that most examples break, is not
    used (and the report says why); when only a few examples break it, those examples are left
    out instead, with a warning naming them."""
    from engine.verify.core import parse_json_output  # noqa: WPS433

    refs = []
    for it in items:
        parsed = it.get("reference_json") if it.get("reference_json") is not None else parse_json_output(it.get("reference") or "")
        question = next((m["content"] for m in reversed(it.get("messages", [])) if m["role"] == "user"), "")
        refs.append((parsed, it.get("reference") or "", question))
    names = list(dict.fromkeys(k for parsed, _, _ in refs if isinstance(parsed, dict) for k in parsed))
    columns = {rules_mod.field_key(k) for k in names}
    input_field = rules_mod.field_key((info or {}).get("input_field") or "") or None
    specs, reports, left_out = [], [], set()
    for i, text in enumerate(t for t in rule_texts if t.strip()):
        compiled = rules_mod.compile_rule(text, columns if refs else None, input_field)
        report = compiled.to_dict()
        reports.append(report)
        if not compiled.ok:
            continue
        if refs:
            problem = None
            unknown = [f for f in rules_mod.fields_used(compiled.expr) if f not in columns]
            if unknown and not columns:
                problem = f"This rule is about “{unknown[0].replace('_', ' ')}”, but the answers in your examples are plain text without separate columns. Rules about the whole answer work here, for example “no more than 50 words” or “must not mention X”."
            elif unknown:
                problem = f"There is no column called “{unknown[0].replace('_', ' ')}” in your examples. The columns your AI fills in are: {', '.join(names[:12])}."
            else:
                broken = rules_mod.breaking(compiled.expr, refs)
                shown = ", ".join(_snippet(refs[b][2] or refs[b][1]) for b in broken[:2])
                if broken and len(broken) == len(refs):
                    problem = f"None of your {len(refs)} examples follow this rule (it was read as: {compiled.explanation}). Reword the rule, or check your examples."
                elif len(broken) > max(2, len(refs) // 20):
                    problem = f"{len(broken)} of your {len(refs)} examples break this rule (it was read as: {compiled.explanation}), for instance {shown}. Change the rule, or correct those examples."
                elif broken:
                    report["warning"] = f"{len(broken)} of your {len(refs)} examples {'breaks' if len(broken) == 1 else 'break'} this rule and {'was' if len(broken) == 1 else 'were'} left out: {shown}."
                    left_out.update(broken)
            if problem:
                # Not a problem with the wording: the same rule may be fine once the files change.
                report.update(ok=False, error=problem, depends_on_data=True)
                continue
            report["self_test"] = rules_mod.self_test(compiled.expr, refs[:200])
        specs.append(_vj(f"rule:{i + 1}", "rule", "gate", 1.0, compiled.explanation, expr=compiled.expr, text=text))
    return specs, reports, left_out


# ------------------------------------------------------------------------------------------
# Dedup, grouping, split, leakage proof
# ------------------------------------------------------------------------------------------


def merkle_root(hashes: list[str]) -> str:
    level = sorted(h.split(":", 1)[-1] for h in hashes)
    if not level:
        return hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest() for i in range(0, len(level), 2)]
    return level[0]


def split_items(items: list[dict], salt: str, template: str) -> tuple[dict[str, list[dict]], dict]:
    # 1) exact dedup on normalized user prompt
    seen: set[str] = set()
    unique = []
    for it in items:
        user = next(m["content"] for m in it["messages"] if m["role"] == "user")
        h = prompt_hash(user)
        if h in seen:
            continue
        seen.add(h)
        it["item_id"] = h
        it["_user"] = user
        unique.append(it)
    dropped_dupes = len(items) - len(unique)
    # 2) near-duplicate clusters (plus template group hints, e.g. paraphrases of one SQL question)
    clusters = near_duplicate_groups([it["_user"] for it in unique])
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for it, c in zip(unique, clusters):
        union("lsh:" + str(c), it["item_id"])
        if it.get("group_hint"):
            union("hint:" + it["group_hint"], it["item_id"])
    families: dict[str, list[dict]] = defaultdict(list)
    for it in unique:
        families[find(it["item_id"])].append(it)
    n = len(unique)
    # A "near-duplicate group" that swallows a large share of the data is not a set of duplicates:
    # it is the shape of the task (templated questions, records that all read alike). Kept whole it
    # would land in a single split, leaving nothing to train on or nothing to test. Such families go
    # back to their hinted sub-groups (paraphrases of one question stay together) or single items.
    family_limit = max(4, n // 10)
    for root_id, members in families.items():
        for it in members:
            if len(members) > family_limit:
                key = ("hint:" + it["group_hint"]) if it.get("group_hint") else it["item_id"]
            else:
                key = root_id
            it["group"] = "g:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in unique:
        groups[it["group"]].append(it)
    n_groups = len(groups)
    # 3) split: groups in a fixed pseudo-random order (HMAC of the project salt, so the order does
    # not change when files are added later) fill the hidden test set, then dev, the rest trains.
    # Keep at least min(30, n/2) test items whenever possible.
    test_frac = 0.3 if n * 0.3 >= 30 else min(0.5, 30 / max(n, 1))
    dev_frac = 0.1
    targets = {"test": round(n * test_frac), "dev": round(n * dev_frac) if n >= 20 else 0}
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for g in sorted(groups, key=lambda name: split_fraction(salt, name)):
        members = groups[g]
        split = "train"
        for name in ("test", "dev"):
            short = targets[name] - len(splits[name])
            # Take the group if the split is still short and taking it brings it closer to its
            # target (a big group may overshoot a little, never wildly).
            if short > 0 and len(members) - short < short:
                split = name
                break
        for it in members:
            it["split"] = split
            it["task"] = template
            splits[split].append(it)
    # 4) leakage proof
    ids = {s: {it["item_id"] for it in rows} for s, rows in splits.items()}
    groups = {s: {it["group"] for it in rows} for s, rows in splits.items()}
    leakage = {
        "item_overlap_train_test": len(ids["train"] & ids["test"]),
        "item_overlap_dev_test": len(ids["dev"] & ids["test"]),
        "group_overlap_train_test": len(groups["train"] & groups["test"]),
        "group_overlap_dev_test": len(groups["dev"] & groups["test"]),
    }
    if any(leakage.values()):
        raise AssertionError(f"leakage detected: {leakage}")
    for rows in splits.values():
        for it in rows:
            it.pop("_user", None)
            it.pop("group_hint", None)
    stats = {"unique": n, "duplicates_removed": dropped_dupes, "groups": n_groups, "test_frac": round(test_frac, 3), "leakage": leakage, "leakage_free": True}
    return splits, stats


def _ordered(it: dict) -> dict:
    keys = ["item_id", "task", "group", "split", "messages", "reference", "reference_json", "verifiers", "meta"]
    return {k: it[k] for k in keys}


# ------------------------------------------------------------------------------------------
# Entry points
# ------------------------------------------------------------------------------------------


def run_ingest(root: Path, inputs: list[str], rule_texts: list[str]) -> dict:
    t0 = time.monotonic()
    project = load_project(root)
    template = project["template"]
    paths = ProjectPaths(root)
    paths.data.mkdir(parents=True, exist_ok=True)
    if not project.get("split_salt"):
        project["split_salt"] = secrets.token_hex(16)
        save_project(root, project)

    events.stage("parse", "started", f"Reading {len(inputs)} location(s)")
    src = parse_sources(inputs, template)
    for w in src.warnings:
        events.log(w, "warn")
    events.stage("parse", "completed", f"{len(src.files)} file(s) read")

    events.stage("build_items", "started", "Writing practice questions from your material")
    builders = {"business_rules": lambda: build_business(src, rule_texts), "code_sql": lambda: build_code_sql(src, paths, rule_texts), "docs_legal": lambda: build_docs(src, rule_texts), "custom": lambda: build_custom(src, rule_texts)}
    if template not in builders:
        raise ValueError(f"unknown template {template}")
    try:
        items, verifiers, info = builders[template]()
        if not items:
            raise ValueError("No practice questions could be made from these files.")
    except ValueError as exc:
        # The reason is usually in the files themselves; say what was wrong with each.
        if src.warnings:
            raise ValueError(f"{exc} " + " ".join(src.warnings[:6])) from exc
        raise
    events.Progress("build_items", 1, 40, 70).update(1, {"items": float(len(items))}, force=True)
    events.stage("build_items", "completed", f"{len(items)} questions drafted")

    events.stage("rules", "started", f"Checking {len(rule_texts)} rule(s)")
    rule_specs, rule_reports, left_out = compile_rules(rule_texts, items, info)
    if left_out:
        items = [it for i, it in enumerate(items) if i not in left_out]
    for it in items:
        it["verifiers"] = it["verifiers"] + [s["id"] for s in rule_specs]
    verifiers += rule_specs
    rule_warnings = []
    for rep in rule_reports:
        events.log(("Rule ok: " + rep["explanation"]) if rep["ok"] else ("Rule not used: " + rep["text"] + " - " + rep["error"]), "info" if rep["ok"] else "warn")
        if not rep["ok"]:
            rule_warnings.append(f"The rule “{rep['text']}” was not used. {rep['error']}")
        elif rep.get("warning"):
            rule_warnings.append(f"Rule “{rep['text']}”: {rep['warning']}")
    events.stage("rules", "completed")

    events.stage("split", "started", "Locking away a hidden final exam")
    splits, split_stats = split_items(items, project["split_salt"], template)
    events.Progress("split", 1, 70, 95).update(1, {k: float(len(v)) for k, v in splits.items()}, force=True)
    for name, rows in splits.items():
        write_jsonl(paths.split(name), [_ordered(it) for it in rows])
    test_hashes = sorted(it["item_id"] for it in splits["test"])
    atomic_write_json(paths.test_manifest, {"version": 1, "n": len(test_hashes), "merkle_root": merkle_root(test_hashes), "item_ids": test_hashes, "groups": sorted({it["group"] for it in splits["test"]})})
    vset = {"version": 1, "pass_threshold": 1.0, "verifiers": verifiers}
    atomic_write_json(paths.verifiers, vset)
    events.stage("split", "completed")

    counts = {k: len(v) for k, v in splits.items()}
    warnings = list(src.warnings) + rule_warnings
    blocker = readiness_blocker(counts, split_stats["unique"], template)
    if blocker is None:
        if counts["test"] < 30:
            warnings.append(f"Only {counts['test']} hidden final-exam questions. That is enough to train, but the result will be marked preliminary; add more examples (aim for 100 or more in the exam) for a result you can trust.")
        if counts["train"] < 20:
            warnings.append("Very few training examples; results may be weak.")
    manifest = {
        "version": 1,
        "template": template,
        "created_at": time.time(),
        "sources": src.files,
        "inputs": [str(p) for p in inputs],
        "rule_texts": list(rule_texts),
        "ready": blocker is None,
        "blocker": blocker,
        "counts": counts,
        "split": split_stats,
        "test_merkle_root": merkle_root(test_hashes),
        "template_info": info,
        "rules": rule_reports,
        "warnings": warnings,
        "verifiers": [{"id": v["id"], "kind": v["kind"], "role": v["role"], "description": v.get("description", "")} for v in verifiers],
    }
    atomic_write_json(paths.manifest, manifest)
    events.emit("artifact", kind="dataset", path=str(paths.data), sha256=hashlib.sha256(json.dumps(counts, sort_keys=True).encode() + merkle_root(test_hashes).encode()).hexdigest())
    summary = {"counts": counts, "leakage_free": True, "duplicates_removed": split_stats["duplicates_removed"], "warnings": warnings, "rules": rule_reports, "template_info": info, "seconds": round(time.monotonic() - t0, 2), "verifiers": manifest["verifiers"], "ready": blocker is None, "blocker": blocker}
    return summary


def readiness_blocker(counts: dict, unique: int, template: str) -> Optional[str]:
    """Why training cannot give a verdict with this exam (None when it can)."""
    if counts["test"] >= MIN_TEST_ITEMS and counts["train"] >= MIN_TRAIN_ITEMS:
        return None
    one = unique == 1
    if template == "docs_legal":
        return f"Only {unique} quiz question{'' if one else 's'} could be written from your documents (they are made from facts such as dates, amounts, durations and names). Inkport needs about 30 to hold a fair final exam; add longer or more documents and build the exam again."
    if template == "code_sql":
        return f"Only {unique} practice question{'' if one else 's'} could be made from these files. Inkport needs about 30 to hold a fair final exam; add bigger tables or more functions with examples and build the exam again."
    return f"Only {unique} different example{' was' if one else 's were'} found. Inkport needs at least 30 to set aside a fair final exam (100 or more gives a result you can trust). Add more examples and build the exam again."


def _card_question(prompt: str, meta: dict) -> str:
    """The part of a practice question worth showing on a small card. A document question starts
    with a long excerpt; the question itself is the last line."""
    if meta.get("origin") == "doc_extract" and '"""\n\n' in prompt:
        return prompt.rsplit('"""\n\n', 1)[1]
    return prompt


def _card_source(source: str) -> str:
    """Where a question came from, as a person would say it: the file, without internal markers."""
    if source == "generated from your tables":
        return "your tables"
    return re.sub(r"(?:#chunk\d+|:\d+)$", "", source)


def preview(root: Path, limit: int = 12) -> dict:
    """Sample practice questions for the UI. Test items are never revealed, only counted."""
    paths = ProjectPaths(root)
    out = []
    for split in ("dev", "train"):
        for it in read_jsonl(paths.split(split)):
            if len(out) >= limit:
                break
            user = next(m["content"] for m in it["messages"] if m["role"] == "user")
            out.append({"item_id": it["item_id"], "split": split, "prompt": user, "question": _card_question(user, it["meta"]), "reference": it["reference"], "source": _card_source(it["meta"].get("source", ""))})
    counts = {s: len(read_jsonl(paths.split(s))) for s in ("train", "dev", "test")}
    manifest = json.loads(paths.manifest.read_text()) if paths.manifest.is_file() else {}
    return {
        "items": out,
        "counts": counts,
        "test_locked": True,
        "verifiers": manifest.get("verifiers", []),
        "warnings": manifest.get("warnings", []),
        # What the exam was built from, so the screen can be restored when a project is reopened.
        "inputs": manifest.get("inputs") or [f.get("path", "") for f in manifest.get("sources", [])],
        "rules": manifest.get("rules", []),
        "duplicates_removed": manifest.get("split", {}).get("duplicates_removed", 0),
        "leakage_free": bool(manifest.get("split", {}).get("leakage_free", True)),
        "ready": manifest.get("ready", counts["test"] >= MIN_TEST_ITEMS and counts["train"] >= MIN_TRAIN_ITEMS),
        "blocker": manifest.get("blocker"),
        "template_info": manifest.get("template_info", {}),
    }


def rules_preview(root: Optional[Path], texts: list[str]) -> dict:
    """Check rules as they are typed. Before an exam exists only the wording can be checked; after,
    the rules are also tried on the practice examples (the hidden exam is never read here)."""
    items, info = [], {}
    if root:
        paths = ProjectPaths(root)
        items = read_jsonl(paths.split("train")) + read_jsonl(paths.split("dev"))
        if paths.manifest.is_file():
            try:
                info = json.loads(paths.manifest.read_text(encoding="utf-8")).get("template_info", {})
            except (OSError, ValueError):
                info = {}
    _, reports, _ = compile_rules(texts, items, info)
    return {"rules": reports}
