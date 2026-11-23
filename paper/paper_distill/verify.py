"""Pull the final answer out of a reply and compare it with a reference.

Checking is deliberately strict. A correct trace that is wrongly rejected costs one
sample; a wrong trace that is wrongly accepted teaches the student an error.
"""
from __future__ import annotations

import re
from fractions import Fraction

FINAL_RE = re.compile(r"^[\s>*_#`-]*final answer[\s*_`]*[:：][\s*_`]*(.*?)[\s*_`]*$", re.I)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_TOKEN = re.compile(
    rf"(?P<mixed>(?<![\w.])-?\d+\s+\d+\s*/\s*\d+)"
    rf"|(?P<frac>(?<![\w.])-?{_NUM}\s*/\s*\d+)"
    rf"|(?P<num>(?<![\w.])-?{_NUM}(?:\.\d+)?)")
_TIME = re.compile(r"(?<!\d)(\d{1,2})[:.](\d{2})\s*([ap])?\.?(?:m\b\.?)?", re.I)


def clean(s: str) -> str:
    s = s.strip().replace("−", "-").replace("–", "-").replace("\u2009", " ").replace("\u00a0", " ")
    for _ in range(3):
        s = _BOXED.sub(r"\1", s)
        s = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace("\\%", "%").replace("\\$", "$").replace("\\,", "").replace("\\!", "").replace("\\ ", " ")
    s = re.sub(r"\\[()\[\]]", "", s).replace("$", " ")
    return s.strip().strip("*_`").strip()


def extract_final_answer(text: str, strict: bool = True) -> str | None:
    """The text after 'Final answer:'.

    strict: that line must be the last non-empty line (used when building the dataset,
    so every kept record ends the same way). Otherwise take the last such line anywhere,
    falling back to the last \\boxed{...} (used when scoring arbitrary models).
    """
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in ([lines[-1]] if strict else reversed(lines)):
        m = FINAL_RE.match(ln)
        if m and m.group(1).strip():
            return clean(m.group(1))
    if not strict:
        boxed = _BOXED.findall(text)
        if boxed:
            return clean(boxed[-1])
    return None


def numeric_value(s: str) -> Fraction | None:
    """The single number the string states, or None if it states none or several different ones."""
    vals = set()
    for m in _TOKEN.finditer(clean(s)):
        if m.group("mixed"):
            whole, num, den = re.findall(r"-?\d+", m.group("mixed").replace(" /", "/").replace("/ ", "/"))
            if int(den) == 0:
                return None
            sign = -1 if whole.startswith("-") else 1
            vals.add(sign * (abs(int(whole)) + Fraction(int(num), int(den))))
        elif m.group("frac"):
            num, den = m.group("frac").replace(",", "").replace(" ", "").split("/")
            if int(den) == 0:
                return None
            vals.add(Fraction(int(num), int(den)))
        else:
            vals.add(Fraction(m.group("num").replace(",", "")))
    return vals.pop() if len(vals) == 1 else None


def _tokens(s: str) -> list[str]:
    s = re.sub(r"\([^()]*\)", " ", clean(s).lower())
    s = re.sub(r"[^\w\s'-]", " ", s)
    toks = [t.strip("'-_") for t in s.split()]
    return [t for t in toks if t and t not in {"the", "a", "an"}]


def _contains(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _items(s: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|→|->|>|\bthen\b|\band\b|\s-\s)\s*", clean(s))
    parts = [p for p in parts if p.strip()]
    if len(parts) == 1 and "-" in parts[0]:
        parts = parts[0].split("-")
    return [" ".join(_tokens(p)) for p in parts if _tokens(p)]


def _minutes(s: str) -> int | None:
    found = set()
    for h, m, ap in _TIME.findall(clean(s)):
        h, m = int(h), int(m)
        if ap:
            h = h % 12 + (12 if ap.lower() == "p" else 0)
        if h < 24 and m < 60:
            found.add(h * 60 + m)
    return found.pop() if len(found) == 1 else None


def _norm_exact(s: str) -> str:
    s = s.strip().strip("`").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s*([\[\],()])\s*", r"\1", s)


def _yesno(s: str) -> str | None:
    toks = _tokens(s)
    if not toks:
        return None
    return {"yes": "yes", "true": "yes", "no": "no", "false": "no"}.get(toks[0])


def check_answer(answer_type: str, truth: str, candidate: str | None, meta: dict | None = None) -> bool:
    if candidate is None or not candidate.strip():
        return False
    meta = meta or {}
    if answer_type in ("int", "fraction"):
        v = numeric_value(candidate)
        return v is not None and v == Fraction(truth)
    if answer_type == "number":
        v = numeric_value(candidate)
        return v is not None and abs(v - Fraction(truth)) < Fraction(1, 2 * 10 ** meta.get("dp", 2))
    if answer_type == "yesno":
        return _yesno(candidate) is not None and _yesno(candidate) == _yesno(truth)
    if answer_type == "text":
        t, c = _tokens(truth), _tokens(candidate)
        distractors = meta.get("distractors")
        if not distractors:
            return c == t
        return _contains(c, t) and not any(_contains(c, _tokens(x)) for x in distractors)
    if answer_type == "list":
        return _items(candidate) == _items(truth)
    if answer_type == "set":
        c = _items(candidate)
        return len(c) == len(set(c)) and sorted(c) == sorted(_items(truth))
    if answer_type == "time":
        return _minutes(candidate) is not None and _minutes(candidate) == _minutes(truth)
    if answer_type == "exact":
        return _norm_exact(clean(candidate)) == _norm_exact(truth)
    if answer_type == "word":
        toks = re.findall(r"[a-z]+", clean(candidate).lower())
        return len(toks) == 1 and toks[0] == truth.lower()
    if answer_type == "choice":
        pick = lambda s: re.match(r"^\W*(?:option\s*)?\(?([A-E])\)?(?![a-z])", clean(s), re.I)  # noqa: E731
        a, b = pick(candidate), pick(truth)
        return bool(a and b and a.group(1).upper() == b.group(1).upper())
    raise ValueError(f"unknown answer_type {answer_type!r}")


def canonical(answer: str | None) -> str | None:
    """A key under which equivalent answers collide, for majority voting."""
    if answer is None or not answer.strip():
        return None
    v = numeric_value(answer)
    if v is not None:
        return f"num:{v}"
    yn = _yesno(answer)
    if yn and len(_tokens(answer)) <= 2:
        return f"yn:{yn}"
    return "txt:" + " ".join(_tokens(answer))
