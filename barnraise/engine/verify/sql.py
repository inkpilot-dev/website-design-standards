"""SQL execution verifier over a read-only SQLite copy of the user's spreadsheets."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

_FENCE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.S | re.I)
_ORDER_BY = re.compile(r"\border\s+by\b", re.I)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_STEP_LIMIT = 2_000_000  # VM instructions (checked every 1000)


class SqlRejected(Exception):
    pass


def extract_sql(text: str, parsed: Any = None) -> str:
    if isinstance(parsed, dict) and isinstance(parsed.get("sql"), str):
        return parsed["sql"].strip()
    m = _FENCE.search(text or "")
    sql = (m.group(1) if m else (text or "")).strip()
    return sql


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def connect_readonly(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    conn.set_authorizer(_authorizer)
    counter = {"n": 0}

    def progress() -> int:
        counter["n"] += 1
        return 1 if counter["n"] * 1000 > _STEP_LIMIT else 0

    conn.set_progress_handler(progress, 1000)
    return conn


def run_query(db: Path, sql: str, max_rows: int = 10_000) -> list[tuple]:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SqlRejected("empty query")
    if ";" in sql:
        raise SqlRejected("only one statement is allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise SqlRejected("only SELECT queries are allowed")
    conn = connect_readonly(db)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise SqlRejected("query returned too many rows")
        return rows
    except sqlite3.DatabaseError as exc:
        raise SqlRejected(str(exc)) from exc
    finally:
        conn.close()


def _norm_row(row: tuple) -> tuple:
    out = []
    for v in row:
        if isinstance(v, float):
            out.append(round(v, 4))
        elif isinstance(v, str):
            out.append(v.strip())
        else:
            out.append(v)
    return tuple(out)


def results_match(gold_sql: str, gold: list[tuple], got: list[tuple]) -> bool:
    g = [_norm_row(r) for r in gold]
    c = [_norm_row(r) for r in got]
    if _ORDER_BY.search(gold_sql):
        return g == c
    return sorted(map(json.dumps, g)) == sorted(map(json.dumps, c))


def verify(db: Path, gold_sql: str, candidate_sql: str) -> tuple[bool, dict]:
    try:
        gold = run_query(db, gold_sql)
    except SqlRejected as exc:
        return False, {"error": f"gold query failed: {exc}"}
    try:
        got = run_query(db, candidate_sql)
    except SqlRejected as exc:
        return False, {"error": str(exc)}
    ok = results_match(gold_sql, gold, got)
    return ok, {"rows": len(got), "expected_rows": len(gold)}


def build_database(tables: dict[str, list[dict]], path: Path, perturb_seed: Optional[int] = None) -> dict[str, list[tuple[str, str]]]:
    """Create a SQLite DB from {table: rows}. With perturb_seed, shuffle each column
    independently (same schema, scrambled values) for the audit verifier that catches
    hard-coded answers. Returns {table: [(column, type)]}."""
    import random

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    schema: dict[str, list[tuple[str, str]]] = {}
    rng = random.Random(perturb_seed)
    for table, rows in tables.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        types = []
        for c in cols:
            vals = [r.get(c) for r in rows if r.get(c) not in (None, "")]
            if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
                types.append("INTEGER")
            elif vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                types.append("REAL")
            else:
                types.append("TEXT")
        schema[table] = list(zip(cols, types))
        col_defs = ", ".join(f'"{c}" {t}' for c, t in schema[table])
        conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
        data = [[r.get(c) for c in cols] for r in rows]
        if perturb_seed is not None:
            for j in range(len(cols)):
                column = [row[j] for row in data]
                rng.shuffle(column)
                for i, v in enumerate(column):
                    data[i][j] = v
        conn.executemany(f'INSERT INTO "{table}" VALUES ({", ".join("?" for _ in cols)})', data)
    conn.commit()
    conn.close()
    return schema
