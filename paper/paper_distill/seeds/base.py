"""Shared pieces for the procedural problem generators.

Every generator returns a Problem whose answer is computed by code, so a
teacher trace can be checked against exact ground truth before it is kept.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

MALE = ["Arjun", "Mateo", "Kofi", "Liam", "Hiro", "Omar", "Pavel", "Tunde", "Felix", "Rafael",
        "Idris", "Jonas", "Minh", "Samir", "Dmitri", "Callum", "Emeka", "Lucas", "Tariq", "Viktor"]
FEMALE = ["Amara", "Sofia", "Mei", "Zainab", "Ingrid", "Priya", "Leila", "Naomi", "Chloe", "Yuki",
          "Fatima", "Elena", "Aisha", "Greta", "Lucia", "Hana", "Nadia", "Ruth", "Tamar", "Wren"]
NAMES = MALE + FEMALE

FMT = {
    "int": "Give the answer as a whole number.",
    "fraction": "Give the answer as a simplified fraction, or as a whole number if it is one.",
    "money": "Give the answer in dollars to two decimal places.",
    "name": "Answer with the name only.",
    "yesno": "Answer Yes or No.",
    "time": "Give the time in 24-hour HH:MM format.",
    "list": "List the names in order, separated by commas.",
    "set": "List the names in alphabetical order, separated by commas.",
}

_ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "l", "m", "n", "p", "pr", "r", "s", "st",
           "t", "tr", "v", "z", "kl", "sh", "th"]
_NUCLEI = ["a", "e", "i", "o", "u", "ai", "or", "en", "an", "el", "ar", "is", "um"]


@dataclass
class Problem:
    question: str
    answer: str  # canonical ground truth, as text
    answer_type: str  # int | fraction | number | text | yesno | list | set | exact | word | time
    domain: str
    skill: str
    difficulty: int  # 1 (two or three steps) .. 5 (many steps)
    meta: dict = field(default_factory=dict)


def frac_str(x: Fraction) -> str:
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def coined(rng: random.Random, syllables: int = 2) -> str:
    """A pronounceable made-up word, so problems never lean on real-world facts."""
    return "".join(rng.choice(_ONSETS) + rng.choice(_NUCLEI) for _ in range(syllables))


def coined_set(rng: random.Random, n: int, syllables: int = 2, suffix: str = "") -> list[str]:
    out: list[str] = []
    while len(out) < n:
        w = coined(rng, syllables) + suffix
        if w not in out and all(w not in o and o not in w for o in out):
            out.append(w)
    return out


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"
