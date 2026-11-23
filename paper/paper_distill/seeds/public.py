"""Optional seeds and benchmark from GSM8K (MIT licence, github.com/openai/grade-school-math).

Only the train split is ever used for training data; the test split is for evaluation.
The dataset's own worked solutions are not used: the teacher writes the reasoning.
"""
from __future__ import annotations

import random

from . import question_key


def _rows(split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("GSM8K needs the `datasets` package: pip install datasets") from e
    out = []
    for ex in load_dataset("openai/gsm8k", "main", split=split):
        solution, _, final = ex["answer"].rpartition("####")
        steps = solution.count("<<")
        q = ex["question"].strip() + " Give the answer as a number."
        out.append({"id": question_key(q), "source": "public:gsm8k", "generator": "gsm8k", "domain": "arithmetic",
                    "skill": "word_problems_gsm8k", "difficulty": 1 if steps <= 2 else min(5, steps - 1),
                    "question": q, "ground_truth": final.strip().replace(",", ""), "answer_type": "int",
                    "meta": {}, "mode": "think"})
    return out


def gsm8k_train(n: int, seed: int) -> list[dict]:
    rows = _rows("train")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def gsm8k_test(n: int | None = None) -> list[dict]:
    rows = _rows("test")
    return rows if n is None else rows[:n]
