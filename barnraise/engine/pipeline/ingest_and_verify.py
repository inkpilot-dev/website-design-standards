"""Data ingestion and auto-verifier studio.

raw files (PDF, CSV, Markdown, TXT, JSON/JSONL, Python, DOCX)
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

from engine import events
from engine.contract import ProjectPaths, atomic_write_json, iter_files, load_project, read_jsonl, save_project, sha256_file, write_jsonl
from engine.templates import CODE_SYSTEM, CUSTOM_SYSTEM, DOCS_SYSTEM, SQL_SYSTEM, business_system
from engine.textnorm import near_duplicate_groups, normalize, prompt_hash, split_fraction, word_count
from engine.verify import rules as rules_mod
from engine.verify import sql as sql_mod

MAX_FILE_BYTES = 50 * 1024 * 1024
INPUT_KEYS = ["text", "message", "description", "input", "prompt", "question", "email", "memo", "details", "transaction", "note", "query"]
OUTPUT_KEYS = ["answer", "output", "response", "completion", "reply", "target", "label"]
ID_KEYS = {"id", "row_id", "index", "idx", "uuid", "#"}

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


def coerce(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return None
    t = s.replace(",", "")
    if t.startswith("$"):
        t = t[1:]
    if re.fullmatch(r"-?\d+", t) and not (len(t) > 1 and t.startswith("0") and not t.startswith("0.")):
        try:
            return int(t)
        except ValueError:
            return s
    if re.fullmatch(r"-?\d*\.\d+", t):
        return float(t)
    return s


def _table_name(path: Path, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_]", "_", path.stem.lower()).strip("_") or "table"
    if base[0].isdigit():
        base = "t_" + base
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


def _read_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader  # noqa: WPS433

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError("Word files need the optional 'python-docx' package") from exc
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def parse_sources(paths: list[str], template: str) -> Sources:
    src = Sources()
    files = list(iter_files(paths))
    prog = events.Progress("parse", max(len(files), 1), 2, 40)
    for i, path in enumerate(files):
        ext = path.suffix.lower()
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                src.warnings.append(f"{path.name}: skipped (larger than 50 MB)")
                continue
            digest = sha256_file(path)
            kind = "other"
            if ext in (".csv", ".tsv"):
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                dialect = csv.excel_tab if ext == ".tsv" else csv.excel
                rows = [{(k or "").strip(): coerce(v) for k, v in r.items() if k} for r in csv.DictReader(io.StringIO(text), dialect=dialect)]
                if template == "code_sql":
                    src.tables[_table_name(path, set(src.tables))] = rows
                    kind = "table"
                else:
                    src.records += [(f"{path.name}:{n + 2}", r) for n, r in enumerate(rows)]
                    kind = "records"
            elif ext in (".jsonl", ".ndjson"):
                for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            src.records.append((f"{path.name}:{n + 1}", obj))
                kind = "records"
            elif ext == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
                rows = obj if isinstance(obj, list) else obj.get("examples", obj.get("data", [])) if isinstance(obj, dict) else []
                src.records += [(f"{path.name}:{n + 1}", r) for n, r in enumerate(rows) if isinstance(r, dict)]
                kind = "records"
            elif ext == ".pdf":
                pages = _read_pdf(path)
                src.documents.append({"path": path.name, "text": "\n".join(pages), "pages": len(pages)})
                kind = "document"
                if sum(len(p) for p in pages) < 50 * max(len(pages), 1):
                    src.warnings.append(f"{path.name}: very little text found; it may be a scanned image (OCR is not bundled yet)")
            elif ext == ".docx":
                src.documents.append({"path": path.name, "text": _read_docx(path), "pages": None})
                kind = "document"
            elif ext in (".md", ".markdown", ".txt", ".rst"):
                src.documents.append({"path": path.name, "text": path.read_text(encoding="utf-8", errors="replace"), "pages": None})
                kind = "document"
            elif ext == ".py":
                src.code.append({"path": path.name, "source": path.read_text(encoding="utf-8", errors="replace")})
                kind = "code"
            else:
                src.warnings.append(f"{path.name}: unsupported file type '{ext}', skipped")
                continue
            src.files.append({"path": str(path), "name": path.name, "sha256": digest, "bytes": size, "kind": kind})
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingestion
            src.warnings.append(f"{path.name}: could not be read ({type(exc).__name__}: {exc})")
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
        raise ValueError("Add a spreadsheet (CSV) or JSONL file of past examples, e.g. columns: description, category, amount.")
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
                    add(f"{t}:sumwhere:{n}:{c}={v}", [f"What is the total {n} for {c} '{v}'?", f"Add up {n} where {c} is {v}."], f"SELECT SUM({N}) FROM {T} WHERE {C} = {_lit(v)}")
                    add(f"{t}:maxwhere:{n}:{c}={v}", [f"What is the largest {n} where {c} is '{v}'?"], f"SELECT MAX({N}) FROM {T} WHERE {C} = {_lit(v)}")
                    budget -= 3
            for n in nums[:3]:
                N = _q(n)
                add(f"{t}:groupsum:{n}:{c}", [f"List each {c} with its total {n}, highest first.", f"Show total {n} per {c}, sorted from highest to lowest."], f"SELECT {C}, SUM({N}) AS total FROM {T} GROUP BY {C} ORDER BY total DESC")
                add(f"{t}:top:{n}:{c}", [f"Which {c} has the highest total {n}?"], f"SELECT {C} FROM {T} GROUP BY {C} ORDER BY SUM({N}) DESC LIMIT 1")
                add(f"{t}:groupavg:{n}:{c}", [f"What is the average {n} for each {c}?"], f"SELECT {C}, AVG({N}) FROM {T} GROUP BY {C}")
            if budget <= 0:
                break
        for n in nums[:4]:
            N = _q(n)
            add(f"{t}:avg:{n}", [f"What is the average {n} in {t}?"], f"SELECT AVG({N}) FROM {T}")
            add(f"{t}:sum:{n}", [f"What is the total {n} across all {t}?"], f"SELECT SUM({N}) FROM {T}")
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
                    add(f"join:{a}:{b}:{n}:{c}", [f"What is the total {n} from {a} for each {c} in {b}?"], f"SELECT {_q(b)}.{_q(c)}, SUM({_q(a)}.{_q(n)}) FROM {_q(a)} JOIN {_q(b)} ON {_q(a)}.{_q(k)} = {_q(b)}.{_q(k)} GROUP BY {_q(b)}.{_q(c)}")
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
        raise ValueError("Add at least one CSV spreadsheet, or Python files whose docstrings contain >>> examples.")
    return items, verifiers, info


# --- documents --------------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_SPAN_PATTERNS = [
    re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b\d+\s+(?:business\s+)?(?:days?|weeks?|months?|years?|hours?)\b", re.I),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"(?<=[a-z,;] )(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"),
    re.compile(r"[\"“]([A-Z][^\"”]{2,40})[\"”]"),
]


def chunk_document(text: str, target: int = 700) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        if len(cur) + len(p) + 1 > target and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return chunks


def cloze_items(doc: dict, max_per_chunk: int = 6) -> list[dict]:
    out = []
    for ci, chunk in enumerate(chunk_document(doc["text"])):
        made = 0
        for sent in _SENT.split(chunk):
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
        raise ValueError("Add the documents you want your AI to know (PDF, Word, Markdown or text).")
    items = [it for d in src.documents for it in cloze_items(d)]
    if not items:
        raise ValueError("I couldn't find facts to quiz on (dates, amounts, durations, names). Try longer documents.")
    verifiers = [
        _vj("json_valid", "json_valid", "gate", description="Reply is a JSON object with answer and quote"),
        _vj("citation", "citation", "reward", 1.0, "Quotes the document word for word and gives the right answer"),
    ]
    return items, verifiers, {"documents": len(src.documents)}


# --- custom ----------------------------------------------------------------------------------


def build_custom(src: Sources, rule_texts: list[str]) -> tuple[list[dict], list[dict], dict]:
    pairs = []
    for source, r in src.records:
        qk = _pick_key(list(r), INPUT_KEYS)
        ak = _pick_key(list(r), OUTPUT_KEYS)
        if qk and ak and r.get(qk) is not None and r.get(ak) is not None:
            pairs.append((source, str(r[qk]), r[ak]))
    if not pairs:
        raise ValueError("Your examples need a question column (prompt/question/input) and an answer column (answer/output/response).")
    json_answers = all(isinstance(a, dict) or (isinstance(a, str) and a.strip().startswith("{")) for _, _, a in pairs)
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
    for source, q, a in pairs:
        ref_json = a if isinstance(a, dict) else (json.loads(a) if json_answers else None)
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


def compile_rules(rule_texts: list[str], items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (verifier specs, per-rule reports). Rules are gates (must-pass)."""
    from engine.verify.core import parse_json_output  # noqa: WPS433

    refs = [(it.get("reference_json") if it.get("reference_json") is not None else parse_json_output(it.get("reference") or ""), it.get("reference") or "") for it in items[:200]]
    specs, reports = [], []
    for i, text in enumerate(t for t in rule_texts if t.strip()):
        compiled = rules_mod.compile_rule(text)
        report = compiled.to_dict()
        if compiled.ok:
            st = rules_mod.self_test(compiled.expr, refs) if refs else None
            report["self_test"] = st
            if st and st["references"] and st["accepted"] < st["references"]:
                report["warning"] = f"{st['references'] - st['accepted']} of your own examples break this rule. Check the rule or those examples."
            specs.append(_vj(f"rule:{i + 1}", "rule", "gate", 1.0, compiled.explanation, expr=compiled.expr, text=text))
        reports.append(report)
    return specs, reports


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
    for it, c in zip(unique, clusters):
        it["group"] = "g:" + hashlib.sha256(find(it["item_id"]).encode()).hexdigest()[:16]
    n_groups = len({it["group"] for it in unique})
    # 3) split fractions: keep at least min(30, n/2) test items whenever possible
    n = len(unique)
    test_frac = 0.3 if n * 0.3 >= 30 else min(0.5, 30 / max(n, 1))
    dev_frac = 0.1
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for it in unique:
        u = split_fraction(salt, it["group"])
        split = "test" if u < test_frac else ("dev" if u < test_frac + dev_frac else "train")
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
    if template == "business_rules":
        items, verifiers, info = build_business(src, rule_texts)
    elif template == "code_sql":
        items, verifiers, info = build_code_sql(src, paths, rule_texts)
    elif template == "docs_legal":
        items, verifiers, info = build_docs(src, rule_texts)
    elif template == "custom":
        items, verifiers, info = build_custom(src, rule_texts)
    else:
        raise ValueError(f"unknown template {template}")
    events.Progress("build_items", 1, 40, 70).update(1, {"items": float(len(items))}, force=True)
    events.stage("build_items", "completed", f"{len(items)} questions drafted")

    events.stage("rules", "started", f"Checking {len(rule_texts)} rule(s)")
    rule_specs, rule_reports = compile_rules(rule_texts, items)
    for it in items:
        it["verifiers"] = it["verifiers"] + [s["id"] for s in rule_specs]
    verifiers += rule_specs
    for rep in rule_reports:
        events.log(("Rule ok: " + rep["explanation"]) if rep["ok"] else ("Rule not understood: " + rep["text"] + " - " + rep["error"]), "info" if rep["ok"] else "warn")
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
    warnings = list(src.warnings)
    if counts["test"] < 30:
        warnings.append(f"Only {counts['test']} hidden test questions. Add more examples so the before-and-after comparison is trustworthy (aim for 100+).")
    if counts["train"] < 20:
        warnings.append("Very few training examples; results may be weak.")
    manifest = {
        "version": 1,
        "template": template,
        "created_at": time.time(),
        "sources": src.files,
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
    summary = {"counts": counts, "leakage_free": True, "duplicates_removed": split_stats["duplicates_removed"], "warnings": warnings, "rules": rule_reports, "template_info": info, "seconds": round(time.monotonic() - t0, 2), "verifiers": manifest["verifiers"]}
    return summary


def preview(root: Path, limit: int = 12) -> dict:
    """Sample practice questions for the UI. Test items are never revealed, only counted."""
    paths = ProjectPaths(root)
    out = []
    for split in ("dev", "train"):
        for it in read_jsonl(paths.split(split)):
            if len(out) >= limit:
                break
            user = next(m["content"] for m in it["messages"] if m["role"] == "user")
            out.append({"item_id": it["item_id"], "split": split, "prompt": user, "reference": it["reference"], "source": it["meta"].get("source", "")})
    counts = {s: len(read_jsonl(paths.split(s))) for s in ("train", "dev", "test")}
    manifest = json.loads(paths.manifest.read_text()) if paths.manifest.is_file() else {}
    return {"items": out, "counts": counts, "test_locked": True, "verifiers": manifest.get("verifiers", []), "warnings": manifest.get("warnings", [])}


def rules_preview(root: Optional[Path], texts: list[str]) -> dict:
    items = read_jsonl(ProjectPaths(root).split("train")) if root else []
    _, reports = compile_rules(texts, items)
    return {"rules": reports}
