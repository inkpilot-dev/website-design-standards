"""Blind, position-swapped pairwise judging with a local open-weight model.

The judge never sees which answer came from which model. It is run twice with the answers
swapped; a side wins only when both orders agree, otherwise the comparison is a tie. The
judge is used only for evaluation and for templates without deterministic checks, never as
a training reward on shared GPUs, and never with a frontier API (their terms forbid using
outputs to train competing models).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

JUDGE_SYSTEM = (
    "You are a strict evaluator. Compare two responses to the same task using ONLY the rubric. "
    "Ignore length, tone, and the order in which responses appear. If they are equally good or "
    "equally bad on a criterion, answer \"tie\" for that criterion. Output JSON only."
)

JUDGE_USER = """[TASK]
{prompt}
[REFERENCE MATERIAL]
{context}
[RUBRIC]
{rubric}
[RESPONSE A]
{a}
[RESPONSE B]
{b}
Return: {{"criteria":[{{"id":<int>,"winner":"A"|"B"|"tie"}}],"overall":"A"|"B"|"tie","confidence":1-5}}"""

DEFAULT_RUBRIC = [
    {"id": 1, "text": "Answers the user's actual question", "weight": 2},
    {"id": 2, "text": "Every factual claim is supported by the provided material or the reference answer", "weight": 2},
    {"id": 3, "text": "Follows the required format", "weight": 1},
]

GenerateFn = Callable[[list[dict]], str]


def _parse(text: str) -> Optional[str]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    winner = str(obj.get("overall", "")).strip().upper()
    return winner if winner in ("A", "B", "TIE") else None


def compare(generate: GenerateFn, prompt: str, candidate: str, baseline: str, context: str = "(none)", rubric: Optional[list[dict]] = None) -> dict:
    """Returns {"winner": "candidate"|"baseline"|"tie", "score": 1|0.5|0, "votes": [...]}."""
    rubric = rubric or DEFAULT_RUBRIC
    rubric_text = "\n".join(f"{r['id']}. {r['text']} (weight {r['weight']})" for r in rubric)

    def ask(a: str, b: str) -> Optional[str]:
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(prompt=prompt, context=context, rubric=rubric_text, a=a, b=b)},
        ]
        return _parse(generate(msgs))

    first = ask(candidate, baseline)  # candidate is A
    second = ask(baseline, candidate)  # candidate is B
    cand_first = {"A": "candidate", "B": "baseline"}.get(first or "", "tie")
    cand_second = {"B": "candidate", "A": "baseline"}.get(second or "", "tie")
    winner = cand_first if cand_first == cand_second else "tie"
    return {"winner": winner, "score": {"candidate": 1.0, "tie": 0.5, "baseline": 0.0}[winner], "votes": [first, second]}


def calibration(human_labels: list[str], judge_labels: list[str]) -> dict:
    """Agreement and Cohen's kappa between human and judge labels (candidate/baseline/tie)."""
    pairs = [(h, j) for h, j in zip(human_labels, judge_labels)]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "ok": False}
    agree = sum(1 for h, j in pairs if h == j) / n
    cats = {"candidate", "baseline", "tie"}
    pe = sum((sum(1 for h, _ in pairs if h == c) / n) * (sum(1 for _, j in pairs if j == c) / n) for c in cats)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": agree, "kappa": kappa, "ok": n >= 30 and agree >= 0.75 and kappa >= 0.6}
