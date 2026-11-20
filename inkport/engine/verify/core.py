"""Verifier registry and item scoring (contracts/schemas/verifier.schema.json)."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from engine.contract import ProjectPaths
from engine.textnorm import normalize, normalize_loose, token_f1
from engine.verify import judge as judge_mod
from engine.verify import rules, sandbox, sql

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.S | re.I)
SLOW_KINDS = {"python_tests", "sql_exec", "pairwise_judge"}


def parse_json_output(text: str) -> Optional[dict]:
    """Strictly a JSON object, optionally inside a ```json fence. No partial recovery."""
    if not text:
        return None
    m = _JSON_FENCE.search(text)
    candidate = (m.group(1) if m else text).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class Context:
    root: Path
    judge: Optional[Callable[[list[dict]], str]] = None
    baseline_output: Optional[str] = None
    cache: dict = field(default_factory=dict)

    @property
    def paths(self) -> ProjectPaths:
        return ProjectPaths(self.root)


def load_verifier_set(root: Path) -> dict:
    p = ProjectPaths(root).verifiers
    return json.loads(p.read_text(encoding="utf-8"))


def _verdict(spec: dict, passed: Optional[bool], score: Optional[float], **detail: Any) -> dict:
    return {"verifier_id": spec["id"], "kind": spec["kind"], "role": spec["role"], "pass": passed, "score": score, "detail": detail}


def _numbers_equal(a: Any, b: Any, tol: float) -> bool:
    fa, fb = rules._number(a), rules._number(b)  # noqa: SLF001 - shared helper
    return fa is not None and fb is not None and abs(fa - fb) <= tol


def run_verifier(spec: dict, item: dict, text: str, parsed: Optional[dict], ctx: Context) -> dict:
    kind, params = spec["kind"], spec.get("params", {})
    ref_json = item.get("reference_json") or {}
    meta = item.get("meta", {})
    try:
        if kind == "json_valid":
            ok = parsed is not None
            return _verdict(spec, ok, float(ok))
        if kind == "json_schema":
            if parsed is None:
                return _verdict(spec, False, 0.0, error="not JSON")
            import jsonschema  # noqa: WPS433

            errors = sorted(jsonschema.Draft202012Validator(params["schema"]).iter_errors(parsed), key=str)
            return _verdict(spec, not errors, float(not errors), errors=[e.message for e in errors[:3]])
        if kind == "field_match":
            f = params["field"]
            if f not in ref_json:
                return _verdict(spec, None, None, skipped=True)
            expected, got = ref_json[f], (parsed or {}).get(f)
            mode = params.get("mode", "exact")
            if mode == "numeric":
                ok = _numbers_equal(got, expected, float(params.get("tol_abs", 0.01)))
            elif mode == "set":
                ok = isinstance(got, list) and sorted(map(normalize, map(str, got))) == sorted(map(normalize, map(str, expected)))
            else:
                ok = got is not None and normalize_loose(str(got)) == normalize_loose(str(expected))
            return _verdict(spec, ok, float(ok), expected=expected, got=got)
        if kind == "exact_match":
            ref = item.get("reference")
            if ref is None:
                return _verdict(spec, None, None, skipped=True)
            if item.get("reference_json") is not None:
                ok = parsed == item["reference_json"]
            else:
                ok = normalize_loose(text) == normalize_loose(ref)
            return _verdict(spec, ok, float(ok))
        if kind == "token_f1":
            ref = item.get("reference")
            if ref is None:
                return _verdict(spec, None, None, skipped=True)
            f1 = token_f1(text, ref)
            return _verdict(spec, f1 >= float(params.get("threshold", 0.6)), f1)
        if kind == "contains":
            needles = params.get("any") or meta.get("must_contain") or []
            ok = any(normalize(n) in normalize(text) for n in needles) if needles else None
            return _verdict(spec, ok, None if ok is None else float(ok))
        if kind == "regex":
            ok = bool(re.search(params["pattern"], text or "")) == bool(params.get("must_match", True))
            return _verdict(spec, ok, float(ok))
        if kind == "rule":
            user_input = next((m["content"] for m in reversed(item.get("messages", [])) if m["role"] == "user"), "")
            ok = rules.evaluate(params["expr"], parsed, text or "", user_input)
            return _verdict(spec, ok, float(ok))
        if kind == "sql_exec":
            gold = meta.get("gold_sql")
            if not gold:
                return _verdict(spec, None, None, skipped=True)
            db = ctx.paths.tables_db_perturbed if params.get("db") == "perturbed" else ctx.paths.tables_db
            ok, detail = sql.verify(db, gold, sql.extract_sql(text, parsed))
            return _verdict(spec, ok, float(ok), **detail)
        if kind == "python_tests":
            tests = meta.get("tests")
            if not tests:
                return _verdict(spec, None, None, skipped=True)
            res = sandbox.run_tests(sandbox.extract_code(text), tests, timeout_s=float(params.get("timeout_s", 6.0)))
            frac = res["passed"] / max(res["total"], 1)
            return _verdict(spec, res["passed"] == res["total"] and res["error"] is None, frac, error=res["error"], tests_passed=res["passed"], tests_total=res["total"])
        if kind == "citation":
            source = meta.get("source_text")
            if not source or parsed is None:
                return _verdict(spec, False if source else None, 0.0 if source else None, error="answer must be JSON with answer and quote")
            quote = str(parsed.get("quote", "") or "")
            answer = str(parsed.get("answer", "") or "")
            quote_ok = len(quote.strip()) >= 3 and normalize(quote) in normalize(source)
            expected = meta.get("answer")
            answer_ok = True if not expected else (normalize_loose(expected) in normalize_loose(answer) or (normalize_loose(answer) != "" and normalize_loose(answer) in normalize_loose(expected) and len(normalize_loose(answer)) >= 0.6 * len(normalize_loose(expected))))
            score = 0.5 * quote_ok + 0.5 * answer_ok
            return _verdict(spec, quote_ok and answer_ok, score, quote_found=quote_ok, answer_correct=answer_ok)
        if kind == "pairwise_judge":
            if ctx.judge is None or ctx.baseline_output is None:
                return _verdict(spec, None, None, skipped=True)
            prompt = next((m["content"] for m in reversed(item.get("messages", [])) if m["role"] == "user"), "")
            res = judge_mod.compare(ctx.judge, prompt, text, ctx.baseline_output, context=item.get("reference") or "(none)")
            return _verdict(spec, res["winner"] == "candidate", res["score"], winner=res["winner"], votes=res["votes"])
    except rules.RuleError as exc:
        return _verdict(spec, False, 0.0, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a broken verifier must never crash a training run
        return _verdict(spec, False, 0.0, error=f"{type(exc).__name__}: {exc}")
    return _verdict(spec, None, None, error=f"unknown verifier kind {kind}", skipped=True)


def score_item(item: dict, text: str, vset: dict, ctx: Context, roles: tuple[str, ...] = ("gate", "reward")) -> dict:
    """Apply the item's verifiers. Returns {"pass", "score", "verdicts", "audit_pass"}."""
    by_id = {v["id"]: v for v in vset["verifiers"]}
    parsed = parse_json_output(text)
    specs = [by_id[vid] for vid in item.get("verifiers", []) if vid in by_id]
    verdicts = [run_verifier(s, item, text, parsed, ctx) for s in specs if s["role"] in roles or s["role"] == "audit"]
    gates = [v for v in verdicts if v["role"] == "gate" and v["pass"] is not None]
    rewards = [(v, by_id[v["verifier_id"]]["weight"]) for v in verdicts if v["role"] == "reward" and v["score"] is not None]
    audits = [v for v in verdicts if v["role"] == "audit" and v["pass"] is not None]
    gates_ok = all(v["pass"] for v in gates)
    if rewards:
        total_w = sum(w for _, w in rewards) or 1.0
        reward = sum(v["score"] * w for v, w in rewards) / total_w
    else:
        reward = 1.0 if gates else 0.0
    score = reward if gates_ok else 0.0
    passed = gates_ok and score >= float(vset.get("pass_threshold", 1.0)) - 1e-9 and (bool(gates) or bool(rewards))
    return {
        "pass": passed,
        "score": round(score, 4),
        "verdicts": verdicts,
        "audit_pass": all(v["pass"] for v in audits) if audits else None,
    }


def score_batch(items: list[dict], outputs: list[str], vset: dict, ctx: Context, roles: tuple[str, ...] = ("gate", "reward")) -> list[dict]:
    slow = any(v["kind"] in SLOW_KINDS for v in vset["verifiers"])
    if not slow or len(items) < 4:
        return [score_item(i, o, vset, ctx, roles) for i, o in zip(items, outputs)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda pair: score_item(pair[0], pair[1], vset, ctx, roles), zip(items, outputs)))
