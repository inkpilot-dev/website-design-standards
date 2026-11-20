"""Plain-English rules -> safe expressions, and a sandboxed expression interpreter.

Rules never run through eval(). They are parsed with ``ast`` and executed by the small
interpreter below, which allows only comparisons, boolean logic, arithmetic, literals,
field access on the model's JSON output, single-generator comprehensions and a fixed set of
pure helper functions. Every evaluation has a step budget and string/list size caps.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from engine.textnorm import normalize, word_count

MAX_EXPR_CHARS = 2000
MAX_STEPS = 20_000
MAX_SEQ_LEN = 10_000


class RuleError(Exception):
    pass


def _number(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("$", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _matches(pattern: str, text: str) -> bool:
    if len(pattern) > 200:
        raise RuleError("pattern too long")
    return re.search(pattern, str(text)[:MAX_SEQ_LEN]) is not None


FUNCS: dict[str, Callable[..., Any]] = {
    "len": lambda x: len(x) if x is not None else 0,
    "sum": lambda xs: sum(_number(x) or 0.0 for x in (xs or [])),
    "abs": abs,
    "round": lambda x, n=2: round(float(x), int(n)),
    "min": lambda *a: min(*a),
    "max": lambda *a: max(*a),
    "lower": lambda s: str(s).lower() if s is not None else "",
    "upper": lambda s: str(s).upper() if s is not None else "",
    "contains": lambda hay, needle: str(needle).lower() in str(hay or "").lower(),
    "startswith": lambda s, p: str(s or "").startswith(str(p)),
    "endswith": lambda s, p: str(s or "").endswith(str(p)),
    "matches": _matches,
    "word_count": lambda s: word_count(str(s or "")),
    "number": _number,
    "is_number": lambda x: _number(x) is not None,
    "keys": lambda d: list(d.keys()) if isinstance(d, dict) else [],
    "any": lambda xs: any(xs or []),
    "all": lambda xs: all(xs or []),
    "count": lambda xs, v: sum(1 for x in (xs or []) if x == v),
    "norm": lambda s: normalize(str(s or "")),
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class _Undefined:
    """Missing fields evaluate to Undefined (falsy, compares unequal to everything)."""

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return other is None or isinstance(other, _Undefined)

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "undefined"


UNDEFINED = _Undefined()


class Interpreter:
    def __init__(self, env: dict[str, Any]):
        self.env = env
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise RuleError("rule is too expensive to evaluate")

    def run(self, node: ast.AST) -> Any:
        self.tick()
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            raise RuleError(f"'{type(node).__name__}' is not allowed in rules")
        return method(node)

    def _Expression(self, n: ast.Expression) -> Any:
        return self.run(n.body)

    def _Constant(self, n: ast.Constant) -> Any:
        if not isinstance(n.value, (str, int, float, bool, type(None))):
            raise RuleError("unsupported literal")
        return n.value

    def _Name(self, n: ast.Name) -> Any:
        if n.id in self.env:
            return self.env[n.id]
        if n.id in ("true", "True"):
            return True
        if n.id in ("false", "False"):
            return False
        if n.id in ("none", "None", "null"):
            return None
        raise RuleError(f"unknown name '{n.id}'")

    def _List(self, n: ast.List) -> Any:
        return [self.run(e) for e in n.elts]

    _Tuple = _List

    def _Attribute(self, n: ast.Attribute) -> Any:
        if n.attr.startswith("_"):
            raise RuleError("private attributes are not allowed")
        base = self.run(n.value)
        if isinstance(base, dict):
            return base.get(n.attr, UNDEFINED)
        return UNDEFINED

    def _Subscript(self, n: ast.Subscript) -> Any:
        base = self.run(n.value)
        key = self.run(n.slice)
        try:
            if isinstance(base, dict):
                return base.get(key, UNDEFINED)
            if isinstance(base, (list, str)) and isinstance(key, int):
                return base[key]
        except (IndexError, TypeError):
            return UNDEFINED
        return UNDEFINED

    def _BoolOp(self, n: ast.BoolOp) -> Any:
        if isinstance(n.op, ast.And):
            result: Any = True
            for v in n.values:
                result = self.run(v)
                if not result:
                    return result
            return result
        result = False
        for v in n.values:
            result = self.run(v)
            if result:
                return result
        return result

    def _UnaryOp(self, n: ast.UnaryOp) -> Any:
        v = self.run(n.operand)
        if isinstance(n.op, ast.Not):
            return not v
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.UAdd):
            return +v
        raise RuleError("unsupported unary operator")

    def _BinOp(self, n: ast.BinOp) -> Any:
        op = _BIN.get(type(n.op))
        if op is None:
            raise RuleError("unsupported operator")
        a, b = self.run(n.left), self.run(n.right)
        if isinstance(n.op, ast.Mult) and (isinstance(a, (str, list)) or isinstance(b, (str, list))):
            raise RuleError("repeating text or lists is not allowed")
        if isinstance(a, _Undefined) or isinstance(b, _Undefined):
            return UNDEFINED
        try:
            return op(a, b)
        except (TypeError, ZeroDivisionError):
            return UNDEFINED

    def _Compare(self, n: ast.Compare) -> Any:
        left = self.run(n.left)
        for op_node, comp in zip(n.ops, n.comparators):
            right = self.run(comp)
            fn = _CMP[type(op_node)]
            if isinstance(left, _Undefined) or isinstance(right, _Undefined):
                if not isinstance(op_node, (ast.Eq, ast.NotEq)):
                    return False
            try:
                if not fn(left, right):
                    return False
            except TypeError:
                return False
            left = right
        return True

    def _IfExp(self, n: ast.IfExp) -> Any:
        return self.run(n.body) if self.run(n.test) else self.run(n.orelse)

    def _Call(self, n: ast.Call) -> Any:
        if not isinstance(n.func, ast.Name) or n.func.id not in FUNCS or n.keywords:
            raise RuleError("only the built-in rule functions can be called")
        args = [self.run(a) for a in n.args]
        for a in args:
            if isinstance(a, (str, list)) and len(a) > MAX_SEQ_LEN:
                raise RuleError("value too large")
        try:
            return FUNCS[n.func.id](*args)
        except RuleError:
            raise
        except Exception:  # noqa: BLE001 - user data of the wrong type just fails the rule
            return UNDEFINED

    def _comprehension(self, n: ast.ListComp | ast.GeneratorExp) -> list:
        if len(n.generators) != 1:
            raise RuleError("only one 'for' is allowed")
        gen = n.generators[0]
        if not isinstance(gen.target, ast.Name) or gen.is_async:
            raise RuleError("unsupported comprehension")
        seq = self.run(gen.iter)
        if not isinstance(seq, (list, tuple)):
            return []
        out = []
        saved = self.env.get(gen.target.id, None)
        for item in seq[:MAX_SEQ_LEN]:
            self.env[gen.target.id] = item
            if all(self.run(c) for c in gen.ifs):
                out.append(self.run(n.elt))
        if saved is None:
            self.env.pop(gen.target.id, None)
        else:
            self.env[gen.target.id] = saved
        return out

    _ListComp = _comprehension
    _GeneratorExp = _comprehension


def parse_expr(expr: str) -> ast.Expression:
    if len(expr) > MAX_EXPR_CHARS:
        raise RuleError("rule is too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"could not read the rule: {exc.msg}") from exc
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    reserved = {"output", "text", "input"} | set(FUNCS) | {"True", "False", "None", "true", "false", "none", "null"}
    comp_targets = {g.target.id for n in ast.walk(tree) if isinstance(n, (ast.ListComp, ast.GeneratorExp)) for g in n.generators if isinstance(g.target, ast.Name)}
    unknown = names - reserved - comp_targets
    if unknown:
        raise RuleError("unknown name(s): " + ", ".join(sorted(unknown)))
    return tree


def evaluate(expr: str, output: Any, text: str, user_input: str = "") -> bool:
    tree = parse_expr(expr)
    env = {"output": output if output is not None else UNDEFINED, "text": text, "input": user_input}
    return bool(Interpreter(env).run(tree))


# ------------------------------------------------------------------------------------------
# Plain-English rule compiler
# ------------------------------------------------------------------------------------------


@dataclass
class CompiledRule:
    ok: bool
    text: str
    expr: str = ""
    explanation: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "text": self.text, "expr": self.expr, "explanation": self.explanation, "error": self.error}


_FIELD = r"(?:the\s+)?[\"']?([a-z_][a-z0-9_ ]{0,40}?)[\"']?"
_NUM = r"(-?\$?[\d,]*\.?\d+)"


def _field(name: str) -> str:
    return "output." + re.sub(r"\s+", "_", name.strip().lower())


def _num(s: str) -> str:
    return str(float(s.replace("$", "").replace(",", ""))).rstrip("0").rstrip(".") if "." in s else s.replace("$", "").replace(",", "")


def _quoted_or_rest(s: str) -> str:
    m = re.search(r"[\"“']([^\"”']+)[\"”']", s)
    return (m.group(1) if m else s).strip().rstrip(".")


def compile_rule(text: str) -> CompiledRule:
    raw = text.strip()
    t = raw.lower().strip().rstrip(".")
    if not t:
        return CompiledRule(False, raw, error="empty rule")

    def done(expr: str, explanation: str) -> CompiledRule:
        try:
            parse_expr(expr)
        except RuleError as exc:
            return CompiledRule(False, raw, error=str(exc))
        return CompiledRule(True, raw, expr, explanation)

    if t.startswith(("expr:", "=")):
        expr = raw.split(":", 1)[1].strip() if t.startswith("expr:") else raw[1:].strip()
        return done(expr, "Custom expression")

    if re.fullmatch(r"(?:(?:the )?(?:answer|output|reply) )?must be valid json", t):
        return done("output != None", "The answer must be a valid JSON object")

    if re.fullmatch(r"(?:(?:the )?(?:answer|output|reply) )?must include a (?:citation|quote)", t):
        return done("len(output.quote) > 0","The answer must quote the source document")

    m = re.fullmatch(r"(?:the )?total must equal (?:the )?sum of (?:the )?(?:line )?items", t)
    if m:
        return done("abs(number(output.total) - sum(i.amount for i in output.line_items)) <= 0.01", "total = sum of line_items[].amount (to the cent)")

    m = re.fullmatch(_FIELD + r" must equal (?:the )?sum of " + _FIELD + r"\.([a-z_]+)", t)
    if m:
        f, lst, sub = m.group(1), m.group(2), m.group(3)
        return done(f"abs(number({_field(f)}) - sum(i.{sub} for i in {_field(lst)})) <= 0.01", f"{f} = sum of {lst}[].{sub}")

    m = re.fullmatch(_FIELD + r" must be one of[: ]\s*(.+)", t)
    if m:
        options = [o.strip().strip("\"'") for o in re.split(r",|\bor\b|/", raw.split("one of", 1)[1].lstrip(": ")) if o.strip().strip("\"'")]
        opts = "[" + ", ".join(json.dumps(o.lower()) for o in options) + "]"
        return done(f"lower({_field(m.group(1))}) in {opts}", f"{m.group(1)} must be one of: {', '.join(options)}")

    m = re.fullmatch(_FIELD + r" must be (?:positive|greater than zero|above zero)", t)
    if m:
        return done(f"number({_field(m.group(1))}) > 0", f"{m.group(1)} > 0")
    m = re.fullmatch(_FIELD + r" must (?:not be negative|be non-negative|be zero or more)", t)
    if m:
        return done(f"number({_field(m.group(1))}) >= 0", f"{m.group(1)} >= 0")
    m = re.fullmatch(_FIELD + r" must be (?:greater than|more than|above|over) " + _NUM, t)
    if m:
        return done(f"number({_field(m.group(1))}) > {_num(m.group(2))}", f"{m.group(1)} > {_num(m.group(2))}")
    m = re.fullmatch(_FIELD + r" must be (?:less than|under|below) " + _NUM, t)
    if m:
        return done(f"number({_field(m.group(1))}) < {_num(m.group(2))}", f"{m.group(1)} < {_num(m.group(2))}")
    m = re.fullmatch(_FIELD + r" must be (?:at least|no less than) " + _NUM, t)
    if m:
        return done(f"number({_field(m.group(1))}) >= {_num(m.group(2))}", f"{m.group(1)} >= {_num(m.group(2))}")
    m = re.fullmatch(_FIELD + r" must be (?:at most|no more than) " + _NUM, t)
    if m:
        return done(f"number({_field(m.group(1))}) <= {_num(m.group(2))}", f"{m.group(1)} <= {_num(m.group(2))}")
    m = re.fullmatch(_FIELD + r" must be between " + _NUM + r" and " + _NUM, t)
    if m:
        lo, hi = _num(m.group(2)), _num(m.group(3))
        return done(f"{lo} <= number({_field(m.group(1))}) <= {hi}", f"{lo} <= {m.group(1)} <= {hi}")

    m = re.fullmatch(r"(?:the )?(?:answer|output|reply )?\s*(?:must be |should be )?(?:no more than|at most|under|fewer than|less than) (\d+) words(?: long)?", t)
    if m:
        n = int(m.group(1)) - (1 if re.search(r"under|fewer than|less than", t) else 0)
        return done(f"word_count(text) <= {n}", f"At most {n} words")

    m = re.fullmatch(r"(?:the )?(?:answer|output|reply)?\s*must not (?:include|mention|contain|say|use) (.+)", t)
    if m:
        needle = _quoted_or_rest(raw.split(re.search(r"(include|mention|contain|say|use)", raw, re.I).group(0), 1)[1])
        return done(f"not contains(text, {json.dumps(needle)})", f"Must never mention “{needle}”")

    m = re.fullmatch(r"(?:the )?(?:answer|output|reply)?\s*must (?:include|mention|contain|say) (.+)", t)
    if m:
        needle = _quoted_or_rest(raw.split(re.search(r"(include|mention|contain|say)", raw, re.I).group(0), 1)[1])
        return done(f"contains(text, {json.dumps(needle)})", f"Must mention “{needle}”")

    m = re.fullmatch(_FIELD + r" (?:is required|must be present|must be filled in)", t)
    if m:
        return done(f"{_field(m.group(1))} != None and {_field(m.group(1))} != ''", f"{m.group(1)} must be filled in")

    return CompiledRule(
        False,
        raw,
        error=(
            "I couldn't turn that into a check. Try phrasing like: "
            "“category must be one of A, B, C”, “amount must be positive”, "
            "“no more than 50 words”, “must not mention X”, or start with expr: for a custom expression."
        ),
    )


# ------------------------------------------------------------------------------------------
# Rule self-test: a rule must accept good references and reject mutated ones
# ------------------------------------------------------------------------------------------


def _mutants(ref: dict, expr: str) -> list[dict]:
    fields = set(re.findall(r"output\.([a-z_][a-z0-9_]*)", expr))
    out = []
    for f in fields:
        if f not in ref:
            continue
        v = ref[f]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            for nv in (v * 1.37 + 1, -abs(v) - 1):
                m = dict(ref)
                m[f] = round(nv, 2)
                out.append(m)
        elif isinstance(v, str):
            m = dict(ref)
            m[f] = "zz-not-valid"
            out.append(m)
        elif isinstance(v, list):
            m = dict(ref)
            m[f] = v[:-1] if v else [{"amount": 1}]
            out.append(m)
        m = dict(ref)
        m.pop(f)
        out.append(m)
    return out


def self_test(expr: str, references: list[tuple[Optional[dict], str]]) -> dict:
    """references: (parsed reference JSON or None, reference text). Returns acceptance stats."""
    accepted = sum(1 for js, txt in references if evaluate(expr, js, txt))
    mutants = [m for js, _ in references if isinstance(js, dict) for m in _mutants(js, expr)]
    caught = sum(1 for m in mutants if not evaluate(expr, m, json.dumps(m)))
    return {
        "references": len(references),
        "accepted": accepted,
        "mutants": len(mutants),
        "caught": caught,
        "discrimination": (caught / len(mutants)) if mutants else None,
        "ok": accepted == len(references) and (not mutants or caught / len(mutants) >= 0.8),
    }
