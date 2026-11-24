"""Filters applied to a teacher trace after its answer has been checked.

These mirror what DeepSeek describes for its own distillation set (drop mixed-language
reasoning and very long paragraphs) plus the usual guards against truncation and loops.
"""
from __future__ import annotations

import re
from collections import Counter

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def _repetition(text: str) -> float:
    """Share of 6-word windows that repeat an earlier window: near 0 for prose, high for loops."""
    words = text.split()
    if len(words) < 60:
        return 0.0
    grams = Counter(tuple(words[i:i + 6]) for i in range(len(words) - 5))
    return sum(c - 1 for c in grams.values()) / max(1, sum(grams.values()))


def trace_problems(reasoning: str, answer: str, finish_reason: str | None, q: dict, mode: str = "think") -> list[str]:
    """Reasons to reject this trace; empty means keep. `q` is the `quality` block of the config."""
    if finish_reason == "reasoning_limit":  # stopped on purpose once the reasoning ran past its cap
        return ["reasoning_too_long"]
    out: list[str] = []
    if finish_reason not in (None, "stop"):
        out.append("truncated")
    if "<think>" in reasoning + answer or "</think>" in reasoning + answer:
        out.append("stray_think_tag")
    rw, aw = len(reasoning.split()), len(answer.split())
    if mode == "think":
        if rw < q["min_reasoning_words"]:
            out.append("reasoning_too_short")
        if rw > q["max_reasoning_words"]:
            out.append("reasoning_too_long")
    if aw == 0:
        out.append("empty_answer")
    if aw > q["max_answer_words"]:
        out.append("answer_too_long")
    text = reasoning + "\n" + answer
    if q.get("english_only", True) and len(_CJK.findall(text)) > 0.02 * max(1, len(text)):
        out.append("mixed_language")
    if max((len(p.split()) for p in re.split(r"\n\s*\n", reasoning)), default=0) > q["max_paragraph_words"]:
        out.append("long_paragraph")
    if max(_repetition(reasoning), _repetition(answer)) > q["max_repetition"]:
        out.append("repetitive")
    names = q.get("forbidden_terms") or []
    if names and re.search(r"\b(" + "|".join(map(re.escape, names)) + r")\b", text, re.I):
        out.append("mentions_model_identity")
    return out
