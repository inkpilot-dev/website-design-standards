"""Plain-English rules -> safe expressions, and a sandboxed expression interpreter.

Rules never run through eval(). They are parsed with ``ast`` and executed by the small
interpreter below, which allows only comparisons, boolean logic, arithmetic, literals,
field access on the model's JSON output, single-generator comprehensions and a fixed set of
pure helper functions. Every evaluation has a step budget and string/list size caps.
"""

from __future__ import annotations

import ast
import json
import keyword
import operator
import re
import unicodedata
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Callable, Optional

from engine.textnorm import normalize, word_count

MAX_EXPR_CHARS = 2000
MAX_RULE_CHARS = 400
MAX_CLAUSES_TRIED = 3000
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
            v = float(s)
        except ValueError:
            return None
        return v if v == v and abs(v) != float("inf") else None  # "nan" and "inf" are words, not amounts
    return None


def _matches(pattern: str, text: str) -> bool:
    if len(pattern) > 200:
        raise RuleError("pattern too long")
    return re.search(pattern, str(text)[:MAX_SEQ_LEN]) is not None


def _word_forms(word: str) -> set[str]:
    """"discounts" also means "discount", and the other way round."""
    forms = {word}
    if not re.search(r"[a-z]$", word) or len(word) < 3:
        return forms
    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")
    elif word.endswith(("ses", "xes", "zes", "ches", "shes")):
        forms.add(word[:-2])
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        forms.add(word[:-1])
    else:
        forms.update({word + "s", word + "es"})
        if word.endswith("y"):
            forms.add(word[:-1] + "ies")
    return forms


def _mentions(hay: Any, needle: Any, exact: Any = False) -> bool:
    """Whether some words appear in a text, as whole words: "drop" is not found in "dropoff_time"."""
    text = normalize(str(hay)) if hay is not None and not isinstance(hay, _Undefined) else ""
    wanted = normalize(str(needle))
    if not wanted:
        return False
    if len(wanted) > 200:
        raise RuleError("text to look for is too long")
    forms = {wanted} if exact else _word_forms(wanted)
    pattern = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
    if re.match(r"[a-z0-9_]", wanted):
        pattern = r"(?<![a-z0-9_])" + pattern
    if re.search(r"[a-z0-9_]$", wanted):
        pattern += r"(?![a-z0-9_])"
    return re.search(pattern, text[:MAX_SEQ_LEN]) is not None


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
    "mentions": _mentions,
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
    # A value as plain text for comparing: nothing for a missing value, "0" for zero.
    "clean": lambda x: "" if x is None or isinstance(x, _Undefined) else normalize(str(x)),
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


def field_key(name: Any) -> str:
    """How a column name is compared: lower case, words joined by underscores, punctuation dropped
    ("Total Amount ($)" and "total_amount" are the same column)."""
    folded = unicodedata.normalize("NFKC", str(name)).casefold()
    return re.sub(r"[^\w]+|_+", "_", folded).strip("_")


def lookup(record: dict, name: str) -> Any:
    """A column by the name a rule uses for it. Spreadsheet headers are rarely typed the way a
    rule is ("Total Amount" vs "total amount"), so letter case, spaces and punctuation are ignored."""
    if name in record:
        return record[name]
    want = field_key(name)
    for key, value in record.items():
        if field_key(key) == want:
            return value
    return UNDEFINED


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
            return lookup(base, n.attr)
        return UNDEFINED

    def _Subscript(self, n: ast.Subscript) -> Any:
        base = self.run(n.value)
        key = self.run(n.slice)
        try:
            if isinstance(base, dict):
                return lookup(base, key) if isinstance(key, str) else base.get(key, UNDEFINED)
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
#
# A rule is one sentence about the answer ("no more than 50 words", "must not mention X") or
# about one of its columns ("amount must be positive", "category must be one of A, B"), which
# may depend on a condition ("if category is travel then amount must be under 500"). The
# sentence is tidied first (quotes, contractions, "should" / "has to" / "cannot" and the like
# all become "must" / "must not"), then read clause by clause. Anything that cannot be read
# is refused with a hint: a rule that silently checks the wrong thing is worse than no rule.


@dataclass
class CompiledRule:
    ok: bool
    text: str
    expr: str = ""
    explanation: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "text": self.text, "expr": self.expr, "explanation": self.explanation, "error": self.error}


HINT = (
    "I couldn't turn that into a check. Try phrasing like: "
    "“category must be one of A, B, C”, “amount must be positive”, "
    "“if category is travel then amount must be under 500”, "
    "“no more than 50 words”, “must not mention X”, or start with expr: for a custom expression."
)


class _NotUnderstood(Exception):
    pass


_QOPEN, _QCLOSE = "", ""
_QUOTED = re.compile(r'"([^"]*)"|(?<!\w)\'([^\']*)\'(?!\w)')
_PLACEHOLDER = re.compile(_QOPEN + r"(\d+)" + _QCLOSE)


class _Quotes:
    """Quoted words are taken out of the sentence while it is read, so that nothing inside the
    quotes is mistaken for grammar ("must not mention 'if only'")."""

    def __init__(self) -> None:
        self.values: list[str] = []

    def hide(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            self.values.append(m.group(1) if m.group(1) is not None else m.group(2))
            return f"{_QOPEN}{len(self.values) - 1}{_QCLOSE}"

        return _QUOTED.sub(sub, text)

    def show(self, text: str) -> str:
        return _PLACEHOLDER.sub(lambda m: self.values[int(m.group(1))], text)

    def only(self, text: str) -> Optional[str]:
        m = _PLACEHOLDER.fullmatch(text.strip())
        return self.values[int(m.group(1))] if m else None

    def within(self, text: str) -> list[str]:
        return [self.values[int(i)] for i in _PLACEHOLDER.findall(text)]


@dataclass
class _Ctx:
    """What is known about the examples while a rule is read (nothing, before an exam is built)."""

    quotes: _Quotes
    columns: Optional[set] = None  # keys of the columns the answers have
    input_field: Optional[str] = None  # key of the column the question is taken from
    quoted_names: set = dataclass_field(default_factory=set)  # column names the user put in quotes
    work: int = 0  # clauses tried so far; reading a rule is cut short rather than left to run away


_TYPOGRAPHY = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‚": "'", "‛": "'", "–": "-", "—": "-", "−": "-", " ": " ", " ": " ", "≥": ">=", "≤": "<=", "≠": "!=", "×": "*", "÷": "/"})
_INVISIBLE = "​‌‍⁠﻿︎️⃣"

_SYNONYMS = [
    (r"^(?:please|kindly)\s+", ""),
    (r"^(?:make sure|be sure|ensure|check|remember|verify)(?:\s+that)?\s+", ""),
    (r"^to\s+", ""),
    (r"^keep\s+(?:it|(?:the |all |every |each )?(?:answers?|repl(?:y|ies)|responses?|outputs?))\s+", "must be "),
    (r"\b(?:can|may|must|should|will)\s+only\s+(?:ever\s+)?be\b", "must be one of"),
    (r"\b(?:cannot|can not|can never|will not|will never|must never|should never|shall never|may never|shall not|should not|may not|must not ever|(?:is|are) not allowed to|(?:is|are) never allowed to|(?:is|are) not to)\b", "must not"),
    (r"\b(?:must always|should always|will always|always must|always should|always (?:has|have|needs?) to|has to|have to|needs to|need to|ought to|should|shall|(?:is|are) required to|(?:is|are) supposed to)\b", "must"),
    (r"\b(?:is|are) always\b", "must be"),
    (r"\b(?:is|are) never\b", "must not be"),
    (r"\bmust be always\b", "must be"),
    (r"\bmust not be ever\b", "must not be"),
    (r"^always\s+", "must "),
    (r"^(?:never|do not)\s+", "must not "),
]


def _decoration(ch: str) -> bool:
    return ch in _INVISIBLE or unicodedata.category(ch) in ("So", "Sk")


_CONTRACTIONS = [(r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"), (r"\b(is|are|was|were|does|do|did|should|must|has|have)n't\b", r"\1 not")]
_CLAUSE_BREAK = re.compile(r"\b(?:and|then|if|when|whenever|unless)\b|,", re.I)
_LOOKS_FOR = re.compile(r"\b(?:mentions?|says?|contains?|includes?|uses?|states?|cites?|references?|refers? to|talks? about|discuss(?:es)?|(?:starts?|begins?|ends?) with)\b", re.I)


def _own_words(t: str, pattern: str, repl: str) -> str:
    """Replace a phrase where it is the rule's own verb ("the reply shouldn't ..."), but not where
    it is part of the words to look for ("never say we can't help")."""

    def sub(m: re.Match) -> str:
        before = m.string[: m.start()]
        breaks = list(_CLAUSE_BREAK.finditer(before))
        clause = before[breaks[-1].end() :] if breaks else before
        return m.group(0) if _LOOKS_FOR.search(clause) else m.expand(repl)

    return re.sub(pattern, sub, t, flags=re.I)


def _plain(raw: str, quotes: _Quotes) -> str:
    t = unicodedata.normalize("NFKC", raw).translate(_TYPOGRAPHY)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:[-*•·>]+\s+|\(?\d{1,2}[.)]\s+)", "", t)  # a bullet or "1." pasted along with the rule
    while t and _decoration(t[0]):
        t = t[1:].lstrip()
    while t and (_decoration(t[-1]) or t[-1] in ".!?;:,"):
        t = t[:-1].rstrip()
    for pattern, repl in _CONTRACTIONS:
        t = _own_words(t, pattern, repl)
    t = quotes.hide(t)
    for pattern, repl in _SYNONYMS:
        t = _own_words(t, pattern, repl)
    return re.sub(r"\s+", " ", t).strip()


# --- what a clause is about ---------------------------------------------------------------

_ANSWER_KEYS = {"", "answer", "answers", "output", "outputs", "reply", "replies", "response", "responses", "it", "result", "results", "ai", "model", "assistant", "text", "everything"}
_INPUT_KEYS = {"question", "questions", "input", "inputs", "request", "requests", "prompt", "message", "user_message", "query", "user_input", "user_question"}
_NOT_A_NAME = {"must", "should", "if", "then", "when", "unless", "is", "are", "be", "not", "than", "and", "or", "that", "which"}
_NOT_A_VALUE = _NOT_A_NAME - {"and", "or", "not"}
_FILLER = {"the", "a", "an", "each", "every", "all", "any", "of", "to", "in", "on", "for", "with"}
_RESERVED_NAMES = {"output", "text", "input", "true", "false", "none", "null"}


def _subject(ctx: _Ctx, text: str) -> tuple[str, str]:
    """('answer' | 'input' | 'field', name as written)."""
    in_quotes = ctx.quotes.only(text) is not None
    s = re.sub(r"\s+", " ", ctx.quotes.show(text)).strip().strip("\"'")
    previous = None
    while previous != s:
        previous = s
        s = re.sub(r"^(?:the|a|an|each|every|all|any|its|their|your|my|our|all of the|(?:value|values|amount) (?:of|in|for))\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(?:field|column|value|values|cell|entry|entries)$", "", s, flags=re.I)
    key = field_key(s)
    if in_quotes:
        ctx.quoted_names.add(s)
    if ctx.columns is not None and key in ctx.columns:
        return "field", s
    if s and not key:
        raise _NotUnderstood  # only punctuation where a name should be
    if key in _ANSWER_KEYS:
        return "answer", ""
    if key in _INPUT_KEYS or (ctx.input_field is not None and key == ctx.input_field):
        return "input", s
    return "field", s


def _field_expr(ctx: _Ctx, name: str) -> str:
    key = field_key(name)
    words = name.split()
    if not key or len(words) > 5 or len(name) > 60:
        raise _NotUnderstood
    known = ctx.columns is not None and key in ctx.columns
    if not known and name not in ctx.quoted_names:
        # Typed without quotes, a column name is a few ordinary words: not a number, not grammar.
        if any(w.lower() in _NOT_A_NAME for w in words) or not re.fullmatch(r"\w[\w \-]*", name) or not re.search(r"[^\W\d_]", name) or key in _FILLER:
            raise _NotUnderstood
    if ctx.columns is not None and key not in ctx.columns:
        # "amounts must be positive" about a column called "amount", and the other way round.
        for other in (key[:-1] if key.endswith("s") else "", key[:-2] if key.endswith("es") else "", key[:-3] + "y" if key.endswith("ies") else "", key + "s", key + "es", key[:-1] + "ies" if key.endswith("y") else ""):
            if other and other in ctx.columns:
                key = other
                break
    if key.isascii() and key.isidentifier() and not keyword.iskeyword(key) and key not in _RESERVED_NAMES and key not in FUNCS:
        return f"output.{key}"
    return f"output[{json.dumps(key, ensure_ascii=False)}]"


def _target(ctx: _Ctx, subject: tuple[str, str]) -> tuple[str, str]:
    """(expression for the value, how to call it in the explanation)."""
    kind, name = subject
    if kind == "answer":
        return "text", "the answer"
    if kind == "input":
        return "input", "the question"
    return _field_expr(ctx, name), name


# --- values ---------------------------------------------------------------------------------

_WORD_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100", "a hundred": "100", "one hundred": "100", "a thousand": "1000", "one thousand": "1000"}
_UNITS = r"(?:\s*(?:%|percent|dollars?|usd|euros?|eur|pounds?|gbp|cents?))?"


def _number_literal(text: str) -> Optional[str]:
    s = text.strip().lower()
    if s in _WORD_NUMBERS:
        return _WORD_NUMBERS[s]
    m = re.fullmatch(r"([-+]?)\s*[$€£]?\s*(\d[\d,]*(?:\.\d+)?|\.\d+)" + _UNITS, s)
    if not m:
        return None
    digits = m.group(2).replace(",", "")
    value = float(digits)
    shown = str(int(value)) if value == int(value) and "." not in digits else repr(value)
    return ("-" if m.group(1) == "-" else "") + shown


@dataclass
class _Value:
    text: str
    quoted: bool

    @property
    def number(self) -> Optional[str]:
        return None if self.quoted else _number_literal(self.text)


def _values(ctx: _Ctx, text: str, split_and: bool) -> list[_Value]:
    separators = r"\s*(?:,|;|/|\||\bor\b" + (r"|\band\b" if split_and else "") + r")\s*"
    out = []
    for part in re.split(separators, text, flags=re.I):
        part = part.strip()
        if not part:
            continue
        quoted = ctx.quotes.only(part)
        value = quoted if quoted is not None else ctx.quotes.show(part).strip().strip("\"'").strip()
        if value:
            out.append(_Value(value, quoted is not None))
    if not out or any(len(v.text) > 200 for v in out):
        raise _NotUnderstood
    return out


def _free_value_ok(v: _Value) -> bool:
    """An unquoted value has to look like a value, not like the rest of a sentence."""
    words = v.text.split()
    return v.quoted or (len(words) <= 5 and not any(w.lower() in _NOT_A_VALUE for w in words))


def _names(values: list[_Value]) -> str:
    return ", ".join(f"“{v.text}”" for v in values)


# --- comparisons ----------------------------------------------------------------------------

_FLIP = {">": "<=", "<": ">=", ">=": "<", "<=": ">", "==": "!=", "!=": "=="}
_SIGNS = {">": ">", "<": "<", ">=": ">=", "=>": ">=", "<=": "<=", "=<": "<=", "=": "==", "==": "==", "!=": "!=", "<>": "!="}
_OP_WORDS = {">": "more than", "<": "fewer than", ">=": "at least", "<=": "at most", "==": "exactly", "!=": "not"}
_UP = r"(?:greater|more|higher|bigger|larger)"
_DOWN = r"(?:less|lower|smaller|fewer)"
_COMPARATORS = [
    (rf"(?:{_UP} than|above|over) or equal to\s+(.+)", ">="),
    (rf"(?:{_DOWN} than|below|under) or equal to\s+(.+)", "<="),
    (rf"(?:at least|no {_DOWN} than|not {_DOWN} than|not below|not under|a minimum of|minimum of|minimum|min|equal to or {_UP} than)\s+(.+)", ">="),
    (rf"(?:at most|no {_UP} than|not {_UP} than|not above|not over|a maximum of|maximum of|maximum|max|up to|equal to or {_DOWN} than)\s+(.+)", "<="),
    (rf"(?:{_UP} than|above|over|after|later than)\s+(.+)", ">"),
    (rf"(?:{_DOWN} than|below|under|before|earlier than)\s+(.+)", "<"),
    (r"(.+?)\s+or\s+(?:more|greater|higher|above|over|later|longer)", ">="),
    (r"(.+?)\s+or\s+(?:less|lower|fewer|below|under|smaller|earlier|shorter)", "<="),
    (r"(?:equal to|exactly|the same as|same as|equal)\s+(.+)", "=="),
]
_LENGTH = re.compile(r"(.+?)\s+(words?|characters?|chars?|letters?|digits?)(?:\s+long|\s+in length)?", re.I)
_ARITH = re.compile(r"\s+(\+|-|\*|/|x|plus|minus|times|multiplied by|divided by)\s+", re.I)
_ARITH_OPS = {"+": "+", "plus": "+", "-": "-", "minus": "-", "*": "*", "x": "*", "times": "*", "multiplied by": "*", "/": "/", "divided by": "/"}


def _length(ctx: _Ctx, subject: tuple[str, str], op: str, rhs: str) -> Optional[tuple[str, str]]:
    """"50 words", "280 characters", "5 digits" as the right-hand side of a comparison."""
    m = _LENGTH.fullmatch(rhs.strip())
    n = _number_literal(m.group(1)) if m else None
    if n is None or not n.isdigit():
        return None
    ref, label = _target(ctx, subject)
    unit = m.group(2).lower()
    if unit.startswith("word"):
        return f"word_count({ref}) {op} {n}", f"{label}: {_OP_WORDS[op]} {n} words"
    if unit.startswith("digit"):
        return f'(matches("^[0-9]*$", clean({ref})) and len(clean({ref})) {op} {n})', f"{label}: {_OP_WORDS[op]} {n} digits"
    return f"len(clean({ref})) {op} {n}", f"{label}: {_OP_WORDS[op]} {n} characters"


def _operand(ctx: _Ctx, text: str) -> tuple[str, str]:
    n = _number_literal(text)
    if n is not None:
        return n, n
    kind, name = _subject(ctx, text)
    if kind != "field":
        raise _NotUnderstood
    return f"number({_field_expr(ctx, name)})", name


def _arithmetic(ctx: _Ctx, text: str) -> Optional[tuple[str, str]]:
    """"subtotal plus tax", "quantity x price", "10% of subtotal", "the sum of a, b and c"."""
    t = text.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(.+)", t, re.I)
    if m:
        expr, label = _operand(ctx, m.group(2))
        return f"{expr} * {m.group(1)} / 100", f"{m.group(1)}% of {label}"
    m = re.fullmatch(r"(?:the )?(sum|total|product) of\s+(.+)", t, re.I)
    if m:
        parts = [p for p in re.split(r"\s*(?:,|\band\b|\bplus\b|\btimes\b)\s*", m.group(2), flags=re.I) if p.strip()]
        if len(parts) < 2:
            return None
        sign = " * " if m.group(1).lower() == "product" else " + "
        operands = [_operand(ctx, p) for p in parts]
        return sign.join(e for e, _ in operands), sign.join(n for _, n in operands)
    pieces = _ARITH.split(t)
    if len(pieces) < 3:
        return None
    exprs, labels = [], []
    for i, piece in enumerate(pieces):
        if i % 2:
            op = _ARITH_OPS[piece.lower()]
            exprs.append(op)
            labels.append(op)
        else:
            e, n = _operand(ctx, piece)
            exprs.append(e)
            labels.append(n)
    return " ".join(exprs), " ".join(labels)


def _equals(ctx: _Ctx, subject: tuple[str, str], negate: bool, values: list[_Value], plain: bool) -> tuple[str, str]:
    """The value is (one of) the given values."""
    ref, label = _target(ctx, subject)
    explicit = all(v.quoted or v.number is not None for v in values)
    if not explicit and not plain:
        raise _NotUnderstood
    if subject[0] != "field" and not explicit:
        raise _NotUnderstood  # "the answer must be polite" cannot be checked
    if not all(_free_value_ok(v) for v in values):
        raise _NotUnderstood
    if all(v.number is not None for v in values):
        numbers = [v.number for v in values]
        if len(numbers) == 1:
            return f"number({ref}) {'!=' if negate else '=='} {numbers[0]}", f"{label} {'!=' if negate else '='} {numbers[0]}"
        return f"number({ref}) {'not in' if negate else 'in'} [{', '.join(numbers)}]", f"{label} is {'not ' if negate else ''}one of: {', '.join(numbers)}"
    # "food and drink" may be one choice or two; accept both readings rather than fail good answers.
    options = list(dict.fromkeys(o for v in values for o in ([v.text] if v.quoted else [v.text, *re.split(r"\s+and\s+", v.text, flags=re.I)]) if o.strip()))
    literals = ", ".join(json.dumps(normalize(o), ensure_ascii=False) for o in options)
    if len(values) == 1 and len(options) == 1:
        return f"clean({ref}) {'!=' if negate else '=='} {literals}", f"{label} is {'not ' if negate else ''}{_names(values)}"
    return f"clean({ref}) {'not in' if negate else 'in'} [{literals}]", f"{label} is {'not ' if negate else ''}one of: {', '.join(v.text for v in values)}"


def _compare(ctx: _Ctx, subject: tuple[str, str], op: str, rhs: str, plain: bool) -> tuple[str, str]:
    ref, label = _target(ctx, subject)
    rhs = rhs.strip()
    measured = _length(ctx, subject, op, rhs)
    if measured:
        return measured
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", rhs):  # dates written year-month-day sort like text
        return f"clean({ref}) {op} {json.dumps(rhs)}", f"{label} {op} {rhs}"
    n = _number_literal(rhs)
    if n is not None and ctx.quotes.only(rhs) is None:
        return f"number({ref}) {op} {n}", f"{label} {op.replace('==', '=')} {n}"
    sums = _arithmetic(ctx, rhs)
    if sums:
        expr, shown = sums
        if op in ("==", "!="):
            return f"abs(number({ref}) - ({expr})) {'<=' if op == '==' else '>'} 0.01", f"{label} {'=' if op == '==' else '!='} {shown}"
        return f"number({ref}) {op} ({expr})", f"{label} {op} {shown}"
    if op in ("==", "!="):
        return _equals(ctx, subject, op == "!=", _values(ctx, rhs, split_and=False), plain)
    other_kind, other = _subject(ctx, rhs)  # "discount must be less than price"
    if other_kind != "field" or subject[0] != "field" or ctx.quotes.within(rhs):
        raise _NotUnderstood
    other_expr = _field_expr(ctx, other)
    if ctx.columns is not None and not set(fields_used(other_expr)) <= ctx.columns:
        raise _NotUnderstood
    return f"number({ref}) {op} number({other_expr})", f"{label} {op} {other}"


# --- clauses --------------------------------------------------------------------------------


def _be(ctx: _Ctx, subject: tuple[str, str], negate: bool, pred: str, plain: bool) -> tuple[str, str]:
    """What comes after "must be" / "must not be"."""
    ref, label = _target(ctx, subject)
    p = pred.strip()
    low = p.lower()
    no = "not " if negate else ""

    def numeric(op: str, n: str) -> tuple[str, str]:
        op = _FLIP[op] if negate else op
        return f"number({ref}) {op} {n}", f"{label} {op} {n}"

    def test(expr: str, words: str) -> tuple[str, str]:
        return (f"not ({expr})" if negate else expr), f"{label} is {no}{words}"

    if re.fullmatch(r"(?:in |all |all in )?lower ?case(?: letters)?", low):
        return test(f"lower({ref}) == {ref}", "in lower case")
    if re.fullmatch(r"(?:in |all |all in )?(?:upper ?case|caps|capitals|capital letters)(?: letters)?", low):
        return test(f"upper({ref}) == {ref}", "in capitals")
    m = re.fullmatch(r"(?:one of(?: the following)?|any of|any one of|either|among)\b[:\s]*(.+)", p, re.I)
    if m:
        return _equals(ctx, subject, negate, _values(ctx, m.group(1), split_and=False), plain=True)
    measured = _length(ctx, subject, "!=" if negate else "==", p)
    if measured:
        return measured
    if re.fullmatch(r"(?:a )?(?:positive|greater than zero|more than zero|above zero|over zero)(?: number| amount| value)?", low):
        return numeric(">", "0")
    if re.fullmatch(r"(?:a )?(?:negative|less than zero|below zero|under zero)(?: number| amount| value)?", low):
        return numeric("<", "0")
    if re.fullmatch(r"non[- ]?negative|(?:zero|0) or (?:more|greater|higher|above|positive)|positive or zero", low):
        return numeric(">=", "0")
    if re.fullmatch(r"non[- ]?positive|(?:zero|0) or (?:less|lower|below|negative)|negative or zero", low):
        return numeric("<=", "0")
    m = re.fullmatch(r"(?:between|from|in the range(?: of)?|in range|within)\s+(.+?)\s+(?:and|to|-)\s+(.+?)(?:\s+\(?(inclusive|exclusive)\)?)?", p, re.I) or re.fullmatch(r"(-?[$€£]?[\d.,]+%?)\s*(?:to|-)\s*(-?[$€£]?[\d.,]+%?)()", p, re.I)
    if m:
        lo, hi = _number_literal(m.group(1)), _number_literal(m.group(2))
        if lo is None or hi is None:
            raise _NotUnderstood
        strict = (m.group(3) or "").lower() == "exclusive"
        a, b = ("<", "<") if strict else ("<=", "<=")
        return test(f"{lo} {a} number({ref}) {b} {hi}", f"between {lo} and {hi}")
    for pattern, op in _COMPARATORS:
        m = re.fullmatch(pattern, p, re.I)
        if m:
            try:
                return _compare(ctx, subject, _FLIP[op] if negate else op, m.group(1), plain)
            except _NotUnderstood:
                continue
    if re.fullmatch(r"(?:a |an )?(?:valid )?(?:number|numeric|numeric value|numerical|amount)", low):
        return test(f"is_number({ref})", "a number")
    if re.fullmatch(r"(?:a |an )?(?:whole number|integer|whole|round number)", low):
        return test(f"(is_number({ref}) and number({ref}) % 1 == 0)", "a whole number")
    if re.fullmatch(r"(?:a |an )?(?:valid )?json(?: object)?", low) and subject[0] == "answer":
        return test("output != None", "a valid JSON object")
    filled, empty = (f"clean({ref}) != ''", f"{label} is filled in"), (f"clean({ref}) == ''", f"{label} is empty")
    if re.fullmatch(r"empty|blank|missing|null|none|not set|left (?:blank|empty|out)|absent|omitted", low):
        return filled if negate else empty
    if re.fullmatch(r"present|filled in|filled|filled out|provided|given|set|included|there|specified|required|mandatory|compulsory|completed|supplied|needed", low):
        return empty if negate else filled
    if low in ("true", "yes"):
        return test(f'clean({ref}) in ["true", "yes"]', "yes")
    if low in ("false", "no"):
        return test(f'clean({ref}) in ["false", "no"]', "no")
    p = re.sub(r"^in\s+(?=.*(?:,|\bor\b))", "", p, flags=re.I)  # "must be in A, B or C"
    return _equals(ctx, subject, negate, _values(ctx, p, split_and=False), plain)


def _says(ctx: _Ctx, subject: tuple[str, str], negate: bool, what: str) -> tuple[str, str]:
    """The answer (or a column, or the question) mentions some words, or never does."""
    ref, label = _target(ctx, subject)
    filler = r"^(?:the |any |any of the |a |an )?(?:words?|phrases?|terms?|names?|text|strings?)\s+(?:like\s+)?"
    quoted = ctx.quotes.within(what)
    if quoted:
        needles = quoted
    else:
        parts = [re.sub(filler, "", p.strip(), flags=re.I) for p in re.split(r"\s*(?:,|;|\bor\b|\band\b)\s*", what, flags=re.I)]
        if any(re.search(r"\bmust\b", part, re.I) for part in parts[1:]):
            raise _NotUnderstood  # "... and must not mention Y" is a second rule, not more words to look for
        needles = [n for n in (p.strip().strip("\"'").strip() for p in parts) if n]
    if not needles or any(len(n) > 200 for n in needles):
        raise _NotUnderstood

    def test(needle: str) -> str:
        # Words in quotes are looked for exactly as written; otherwise "discounts" also finds "discount".
        return f"mentions({ref}, {json.dumps(needle, ensure_ascii=False)}{', true' if quoted else ''})"

    tests = [test(n) for n in needles]
    shown = " or ".join(f"“{n}”" for n in needles)
    if negate:
        return " and ".join(f"not {t}" for t in tests), f"{label} never mentions {shown}"
    if len(needles) == 1:
        return tests[0], f"{label} mentions {shown}"
    if re.search(r"\bor\b", what, re.I):
        return "(" + " or ".join(tests) + ")", f"{label} mentions {shown}"
    together = "(" + " and ".join(tests) + ")"
    shown = " and ".join(f"“{n}”" for n in needles)
    if not quoted and "," not in what:
        # "terms and conditions" may be one phrase or two things to mention; accept either reading.
        return f"({test(re.sub(filler, '', what.strip(), flags=re.I))} or {together})", f"{label} mentions {shown}"
    return together, f"{label} mentions {shown}"


def _affix(ctx: _Ctx, subject: tuple[str, str], negate: bool, func: str, what: str) -> tuple[str, str]:
    ref, label = _target(ctx, subject)
    values = _values(ctx, re.sub(r"^(?:the |a |an )?(?:words?|phrases?|letters?|text)\s+", "", what.strip(), flags=re.I), split_and=False)
    tests = [f"{func}(clean({ref}), {json.dumps(normalize(v.text), ensure_ascii=False)})" for v in values]
    expr = tests[0] if len(tests) == 1 else "(" + " or ".join(tests) + ")"
    verb = "starts with" if func == "startswith" else "ends with"
    return (f"not {expr}" if negate else expr), f"{label} {'never ' if negate else ''}{verb} {' or '.join(f'“{v.text}”' for v in values)}"


_ANSWER = r"(?:(?:the |each |every |all )?(?:answers?|outputs?|repl(?:y|ies)|responses?|it) )?"


def _must(ctx: _Ctx, subject_text: str, negate: bool, rest: str, plain: bool = True) -> tuple[str, str]:
    """What comes after "must" / "must not". ``plain`` allows a bare value ("must be USD")."""
    subject = _subject(ctx, subject_text)
    rest = rest.strip()
    m = re.fullmatch(r"be\s+(.+)", rest, re.I)
    if m:
        return _be(ctx, subject, negate, m.group(1), plain)
    verbs = r"contain|include|mention|say|state|cite|reference|have" + (r"|use|talk about|discuss|refer to" if negate else "")
    m = re.fullmatch(rf"(?:{verbs})\s+(.+)", rest, re.I)
    if m:
        return _says(ctx, subject, negate, m.group(1))
    m = re.fullmatch(r"(?:start|begin)\s+with\s+(.+)", rest, re.I)
    if m:
        return _affix(ctx, subject, negate, "startswith", m.group(1))
    m = re.fullmatch(r"end\s+with\s+(.+)", rest, re.I)
    if m:
        return _affix(ctx, subject, negate, "endswith", m.group(1))
    m = re.fullmatch(r"(?:equal|match|total|come to|add up to|sum to)(?:\s+to)?\s+(.+)", rest, re.I)
    if m and subject[0] == "field":
        return _compare(ctx, subject, "!=" if negate else "==", m.group(1), plain)
    m = re.fullmatch(r"exceed\s+(.+)", rest, re.I)
    if m:
        return _compare(ctx, subject, "<=" if negate else ">", m.group(1), plain)
    if re.fullmatch(r"exist|have a value|appear", rest, re.I) and subject[0] == "field":
        return _be(ctx, subject, negate, "present", plain)
    raise _NotUnderstood


def _simple(ctx: _Ctx, t: str, bare_is: bool, bare_must: bool = True) -> tuple[str, str]:
    """One clause: no "if", no "and" joining two rules. A bare value ("must be USD") is taken
    as a value after "must be" unless ``bare_must`` is off, and after a plain "is" only when
    ``bare_is`` is on (in a condition: "if category is travel")."""
    t = t.strip()
    low = t.lower()
    ctx.work += 1
    if not t or ctx.work > MAX_CLAUSES_TRIED:
        raise _NotUnderstood
    if re.fullmatch(_ANSWER + r"must be (?:a )?(?:valid )?json(?: object)?", low):
        return "output != None", "The answer is a valid JSON object"
    if re.fullmatch(_ANSWER + r"must (?:include|have|contain|give|provide|use|cite) (?:a |an |the )?(?:citation|quote|quotation|source)s?(?: from the (?:document|source|text))?", low):
        return "len(output.quote) > 0", "The answer quotes the source document"
    if re.fullmatch(r"(?:the )?total must (?:be |be equal to |equal |match )(?:to )?(?:the )?sum of (?:the |all |all the )?(?:line )?items", low):
        return "abs(number(output.total) - sum(i.amount for i in output.line_items)) <= 0.01", "total = sum of line_items[].amount (to the cent)"
    m = re.fullmatch(r"(.+?) must (?:be |be equal to |equal |match )(?:to )?(?:the )?sum of (.+?)\.([a-z_][a-z0-9_]*)", t, re.I)
    if m and _subject(ctx, m.group(1))[0] == "field" and _subject(ctx, m.group(2))[0] == "field":
        total, items, part = _field_expr(ctx, _subject(ctx, m.group(1))[1]), _field_expr(ctx, _subject(ctx, m.group(2))[1]), m.group(3).lower()
        return f"abs(number({total}) - sum(i.{part} for i in {items})) <= 0.01", f"{_subject(ctx, m.group(1))[1]} = sum of {_subject(ctx, m.group(2))[1]}[].{part}"
    m = re.fullmatch(r"(?:(.*?)\s+)??must\s+(not\s+)?(.+)", t, re.I)  # the first "must" is the rule's verb
    if m:
        return _must(ctx, m.group(1) or "", bool(m.group(2)), m.group(3), bare_must)
    m = re.fullmatch(r"(.+?)\s*(>=|<=|=>|=<|!=|<>|==|=|>|<)\s*(.+)", t)
    if m:
        return _compare(ctx, _subject(ctx, m.group(1)), _SIGNS[m.group(2)], m.group(3), bare_must)
    for pattern, op in _COMPARATORS:  # "no more than 50 words", "50 words or less"
        m = re.fullmatch(pattern, t, re.I)
        measured = _length(ctx, ("answer", ""), op, m.group(1)) if m else None
        if measured:
            return measured
    m = re.fullmatch(r"(.+?)\s+(?:is|are)\s+(not\s+)?(.+)", t, re.I)
    if m:
        return _must(ctx, m.group(1), bool(m.group(2)), "be " + m.group(3), bare_is and bare_must)
    raise _NotUnderstood


_VERB_FORMS = {"contains": "contain", "includes": "include", "mentions": "mention", "says": "say", "has": "have", "starts with": "start with", "begins with": "begin with", "ends with": "end with", "equals": "equal", "matches": "match", "exceeds": "exceed"}
_VERBS = "|".join(sorted({*_VERB_FORMS, *_VERB_FORMS.values()}, key=len, reverse=True))


def _one_condition(ctx: _Ctx, c: str, strict: bool = False) -> tuple[str, str]:
    c = c.strip()
    bare = not strict
    m = re.fullmatch(r"(?:there is|there are) (no|not an?|an?|some|any)\s+(.+)", c, re.I)
    if m:
        return _simple(ctx, f"{m.group(2)} must {'not ' if m.group(1).lower() in ('no', 'not a', 'not an') else ''}be present", True, bare)
    m = re.fullmatch(rf"(.+?)\s+(?:does|do|did)\s+not\s+({_VERBS})\s+(.+)", c, re.I)
    if m:
        return _simple(ctx, f"{m.group(1)} must not {_VERB_FORMS.get(m.group(2).lower(), m.group(2))} {m.group(3)}", True, bare)
    m = re.fullmatch(rf"(.+?)\s+({_VERBS})\s+(.+)", c, re.I)
    if m:
        try:
            return _simple(ctx, f"{m.group(1)} must {_VERB_FORMS.get(m.group(2).lower(), m.group(2))} {m.group(3)}", True, bare)
        except _NotUnderstood:
            pass
    m = re.fullmatch(r"(.+?)\s+(?:is|are|was|were)\s+not\s+(.+)", c, re.I)
    if m:
        try:
            return _simple(ctx, f"{m.group(1)} must not be {m.group(2)}", True, bare)
        except _NotUnderstood:
            pass
    m = re.fullmatch(r"(.+?)\s+(?:is|are|was|were)\s+(.+)", c, re.I)
    if m:
        try:
            return _simple(ctx, f"{m.group(1)} must be {m.group(2)}", True, bare)
        except _NotUnderstood:
            pass
    if re.search(r"\bmust\b", c, re.I):
        raise _NotUnderstood
    return _simple(ctx, c, True, bare)


def _joined(ctx: _Ctx, t: str, one, glue: tuple[str, ...], loose: bool = True) -> tuple[str, str]:
    """A clause, or clauses joined by "and" (and "or", inside conditions).

    "and" is also a word inside values ("food and drink"), so the order matters: first the whole
    text as one clause without guessing at bare values, then as two clauses, and only then as
    one clause whose value happens to contain the word."""
    try:
        return one(ctx, t, True)
    except _NotUnderstood:
        pass
    for m in re.finditer(r",?\s+(" + "|".join(glue) + r")\s+(?:also\s+)?", t, re.I):
        left, right = t[: m.start()], t[m.end() :]
        try:
            a = one(ctx, left, False)
        except _NotUnderstood:
            continue
        # "amount must be positive and under 500": the second half borrows the start of the first.
        lead = re.match(r"(.+?\bmust(?:\s+not)?(?:\s+be)?)\s", left, re.I)
        who = re.match(r"(.+?)\s+must\b", left, re.I)
        candidates = [(right, True)]
        if who and re.match(r"must\b", right, re.I):
            candidates.append((f"{who.group(1)} {right}", True))
        elif lead:
            candidates.append((f"{lead.group(1)} {right}", False))
        b = None
        for candidate, may_guess in candidates:
            try:
                b = _joined(ctx, candidate, one, glue, loose=may_guess)
                break
            except _NotUnderstood:
                continue
        if b is None:
            continue
        word = m.group(1).lower()
        return f"({a[0]}) {word} ({b[0]})", f"{a[1]} {word} {b[1]}"
    if not loose:
        raise _NotUnderstood
    return one(ctx, t, False)


def _rule(ctx: _Ctx, t: str, plain: bool) -> tuple[str, str]:
    return _joined(ctx, t, lambda c, x, strict: _simple(c, x, plain and not strict, not strict), ("and",))


def _condition(ctx: _Ctx, t: str) -> tuple[str, str]:
    return _joined(ctx, t, _one_condition, ("and", "or"))


def _when(condition: tuple[str, str], then: tuple[str, str], unless: bool = False) -> tuple[str, str]:
    if unless:
        return f"({condition[0]}) or ({then[0]})", f"Unless {condition[1]}: {then[1]}"
    return f"(not ({condition[0]})) or ({then[0]})", f"When {condition[1]}: {then[1]}"


def _read(ctx: _Ctx, t: str) -> tuple[str, str]:
    m = re.fullmatch(r"(?:if|when|whenever|where|in case|for (?:rows|answers|items|records|examples) where)\s+(.+)", t, re.I)
    if m:
        body = m.group(1)
        cuts = sorted(re.finditer(r",?\s+then\s+|,\s+", body, re.I), key=lambda c: (0 if "then" in c.group(0).lower() else 1, c.start()))
        for cut in cuts:
            try:
                return _when(_condition(ctx, body[: cut.start()]), _rule(ctx, body[cut.end() :], plain=True))
            except _NotUnderstood:
                continue
        raise _NotUnderstood
    for cut in reversed(list(re.finditer(r",?\s+(if|when|whenever|unless|except if|except when)\s+", t, re.I))):
        try:
            return _when(_condition(ctx, t[cut.end() :]), _rule(ctx, t[: cut.start()], plain=True), unless=cut.group(1).lower() in ("unless", "except if", "except when"))
        except _NotUnderstood:
            continue
    return _rule(ctx, t, plain=False)


def compile_rule(text: str, columns: Optional[set] = None, input_field: Optional[str] = None) -> CompiledRule:
    """Turn one sentence into a check. ``columns`` (keys of the answer's columns) and
    ``input_field`` (key of the column the question comes from) are given once the examples
    are known, so that names in the rule can be matched to the user's own headers."""
    raw = text.strip()
    if not raw:
        return CompiledRule(False, raw, error="empty rule")

    def done(expr: str, explanation: str) -> CompiledRule:
        try:
            parse_expr(expr)
        except RuleError as exc:
            return CompiledRule(False, raw, error=str(exc))
        except (ValueError, RecursionError, MemoryError):
            return CompiledRule(False, raw, error="could not read the rule")
        return CompiledRule(True, raw, expr, explanation)

    if raw.lower().startswith("expr:") or raw.startswith("="):
        return done(raw.split(":", 1)[1].strip() if raw.lower().startswith("expr:") else raw[1:].strip(), "Custom expression")
    if len(raw) > MAX_RULE_CHARS:
        return CompiledRule(False, raw, error="This rule is too long to check. Split it into shorter rules, one check each.")
    quotes = _Quotes()
    ctx = _Ctx(quotes, set(columns) if columns is not None else None, input_field)
    try:
        expr, explanation = _read(ctx, _plain(raw, quotes))
    except (_NotUnderstood, RecursionError, ValueError, IndexError, KeyError, re.error):
        # Whatever was typed, the answer is a verdict on the rule, never a crash.
        return CompiledRule(False, raw, error=HINT)
    explanation = quotes.show(explanation)
    return done(expr, explanation[0].upper() + explanation[1:])


# ------------------------------------------------------------------------------------------
# Rules against the user's own examples
# ------------------------------------------------------------------------------------------


def fields_used(expr: str) -> list[str]:
    """Keys of the answer's columns that a rule reads (``output.amount``, ``output["net total"]``)."""
    found = []
    for node in ast.walk(parse_expr(expr)):
        key = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "output":
            key = node.attr
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "output" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
        if key is not None and field_key(key) not in found:
            found.append(field_key(key))
    return found


def breaking(expr: str, references: list[tuple]) -> list[int]:
    """Positions of the references that do not follow the rule. Each reference is
    (parsed answer or None, answer text, question)."""
    tree = parse_expr(expr)
    out = []
    for i, ref in enumerate(references):
        js, txt = ref[0], ref[1]
        env = {"output": js if js is not None else UNDEFINED, "text": txt, "input": ref[2] if len(ref) > 2 else ""}
        try:
            ok = bool(Interpreter(env).run(tree))
        except RuleError:
            ok = False
        if not ok:
            out.append(i)
    return out


def _mutants(ref: dict, expr: str) -> list[dict]:
    wanted = set(fields_used(expr))
    out = []
    for f in [k for k in ref if field_key(k) in wanted]:
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


def self_test(expr: str, references: list[tuple]) -> dict:
    """references: (parsed reference JSON or None, reference text[, question]). Acceptance stats:
    a good rule accepts the references and rejects versions of them with a value changed."""
    broken = set(breaking(expr, references))
    accepted = len(references) - len(broken)
    mutants = [(m, ref[2] if len(ref) > 2 else "") for ref in references if isinstance(ref[0], dict) for m in _mutants(ref[0], expr)]
    caught = len(breaking(expr, [(m, json.dumps(m), q) for m, q in mutants]))
    return {
        "references": len(references),
        "accepted": accepted,
        "mutants": len(mutants),
        "caught": caught,
        "discrimination": (caught / len(mutants)) if mutants else None,
        "ok": accepted == len(references) and (not mutants or caught / len(mutants) >= 0.8),
    }
