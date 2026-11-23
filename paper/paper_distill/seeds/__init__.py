"""Registry of procedural generators and the seed sampler."""
from __future__ import annotations

import hashlib
import random
import re

from . import algo_gen, logic_gen, math_gen, relational_gen
from .base import Problem

GENERATORS = math_gen.GENERATORS + logic_gen.GENERATORS + relational_gen.GENERATORS + algo_gen.GENERATORS
DIFFICULTY_WEIGHTS = [0.15, 0.25, 0.25, 0.20, 0.15]  # difficulty 1..5


def question_key(question: str) -> str:
    """Stable id for a question: hash of its lower-cased, whitespace-collapsed text."""
    norm = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def make_problem(gen, rng: random.Random, difficulty: int) -> Problem:
    for _ in range(50):
        try:
            return gen(rng, difficulty)
        except (IndexError, ValueError):  # a rare unlucky draw (e.g. ran out of names); draw again
            continue
    raise RuntimeError(f"{gen.__name__} failed 50 times at difficulty {difficulty}")


def generate(n: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    """n distinct problems, spread evenly over the generators. Deterministic for a given seed."""
    rng = random.Random(seed)
    seen = set(exclude or ())
    out: list[dict] = []
    stale = 0
    while len(out) < n:
        gen = GENERATORS[len(out) % len(GENERATORS)] if stale == 0 else rng.choice(GENERATORS)
        d = rng.choices([1, 2, 3, 4, 5], DIFFICULTY_WEIGHTS)[0]
        p = make_problem(gen, rng, d)
        key = question_key(p.question)
        if key in seen:
            stale += 1
            if stale > 20000:
                raise RuntimeError("could not find enough distinct problems")
            continue
        stale = 0
        seen.add(key)
        out.append({"id": key, "source": "procedural", "generator": gen.__name__, "domain": p.domain,
                    "skill": p.skill, "difficulty": p.difficulty, "question": p.question,
                    "ground_truth": p.answer, "answer_type": p.answer_type, "meta": p.meta})
    return out
