"""Built-in general-skills regression suite and self-replay prompts.

Generated deterministically (no downloads, no licensing questions). The gate compares the
base model and the candidate on this suite (each of its questions once: ``exam``) to catch
catastrophic forgetting. The replay
prompts are answered by the *base* model itself: part of them is mixed into SFT, which anchors
general behavior without importing anyone else's data, and part is held back and asked again
while training runs.

The replay prompts are the same kinds of question as the suite, with other words, and no prompt
occurs twice or is one of the suite's: what the final exam asks was never rehearsed. (They used to
be drawn from the same fifteen words with another seed: 35 of 132 were exam questions and 30
were repeats.)
"""

from __future__ import annotations

import json
import random

WORDS = ["apple", "river", "blue", "stone", "cloud", "maple", "north", "pixel", "garden", "lantern", "harbor", "violet", "copper", "meadow", "canyon"]
KEYS = ["color", "name", "city", "animal", "item", "status"]
# For the replay prompts only: none of these words or keys appears in a question of the suite.
REPLAY_WORDS = [
    "anchor", "basket", "candle", "desert", "ember", "falcon", "glacier", "hammer", "island", "jacket", "kettle", "ladder",
    "marble", "needle", "orchid", "pebble", "quartz", "ribbon", "saddle", "timber", "umbrella", "valley", "walnut", "yellow",
    "zipper", "bridge", "castle", "dragon", "engine", "forest", "guitar", "helmet", "jungle", "kitten", "lemon", "mirror",
    "napkin", "ocean", "pencil", "rocket", "silver", "tunnel", "velvet", "window", "bottle", "circle", "feather", "magnet",
]
REPLAY_KEYS = ["size", "shape", "place", "tool", "label", "mood"]


def _items(seed: int, n: int, words: list[str] = WORDS, keys: list[str] = KEYS, avoid: frozenset = frozenset(), distinct: bool = False) -> list[dict]:
    """``n`` questions, six kinds in turn. With ``distinct``, no question is repeated and none is in
    ``avoid``; a kind that has run out of new questions (there are only so many words) gives its
    turn to sums and comparisons, which never run out."""
    rng = random.Random(seed)
    out: list[dict] = []
    seen = set(avoid)
    kinds = ["add", "word", "upper", "json", "count", "compare"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        for attempt in range(60 if distinct else 1):
            if attempt >= 40:
                kind = "add" if attempt % 2 else "compare"
            item = _one(rng, kind, words, keys, f"general-{seed}-{i}")
            q = item["messages"][0]["content"]
            if not distinct or q not in seen:
                break
        seen.add(q)
        out.append(item)
    return out


def _one(rng: random.Random, kind: str, words: list[str], keys: list[str], item_id: str) -> dict:
    if kind == "add":
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        q, ref, check = f"What is {a} + {b}? Reply with only the number.", str(a + b), {"kind": "numeric", "value": a + b}
    elif kind == "word":
        w = rng.choice(words)
        q, ref, check = f"Reply with only the word: {w}", w, {"kind": "exact", "value": w}
    elif kind == "upper":
        w = rng.choice(words)
        q, ref, check = f"Write the word '{w}' in uppercase letters only.", w.upper(), {"kind": "exact_case", "value": w.upper()}
    elif kind == "json":
        k, v = rng.choice(keys), rng.choice(words)
        q, ref, check = f'Return a JSON object with the key "{k}" set to "{v}". Reply with JSON only.', json.dumps({k: v}), {"kind": "json", "value": {k: v}}
    elif kind == "count":
        w = rng.choice(words)
        q, ref, check = f"How many letters are in the word '{w}'? Reply with only the number.", str(len(w)), {"kind": "numeric", "value": len(w)}
    else:
        a, b = rng.sample(range(5, 99), 2)
        q, ref, check = f"Which is larger, {a} or {b}? Reply with only the number.", str(max(a, b)), {"kind": "numeric", "value": max(a, b)}
    return {"id": item_id, "messages": [{"role": "user", "content": q}], "reference": ref, "check": check}


def suite(n: int = 60) -> list[dict]:
    return _items(seed=7, n=n)


def exam(n: int = 60) -> list[dict]:
    """The ``n`` general questions of the final exam: the first ``n`` different questions of the
    suite's sequence.

    ``suite(60)`` asks 8 of its questions twice. Answers are decided by greedy decoding, so a
    repeated question is kept or lost twice, and the exam's test of "more than chance" counted that
    as two independent losses: three lost questions could weigh as much as five or six. Each
    question is now asked once; the 52 different ones of ``suite(60)`` are still there, in the same
    order, followed by the next 8 new ones of the sequence."""
    seen: set[str] = set()
    out: list[dict] = []
    for item in _items(seed=7, n=4 * n + 24):
        q = item["messages"][0]["content"]
        if q not in seen:
            seen.add(q)
            out.append(item)
            if len(out) == n:
                break
    return out


def replay_prompts(n: int = 64) -> list[dict]:
    extra = [
        "Give one short tip for staying organized at work.",
        "Write one friendly sentence welcoming a new customer.",
        "Explain in one sentence what a budget is.",
        "List three common office supplies, separated by commas.",
        "Summarize in one sentence why backups matter.",
        "Write a polite one-sentence reminder that an invoice is due.",
    ]
    exam = frozenset(it["messages"][0]["content"] for it in _items(seed=7, n=240))
    items = _items(seed=99, n=max(0, n - len(extra)), words=REPLAY_WORDS, keys=REPLAY_KEYS, avoid=exam, distinct=True)
    items += [{"id": f"replay-extra-{i}", "messages": [{"role": "user", "content": t}], "reference": None, "check": None} for i, t in enumerate(extra)]
    return items[:n]


def check(item: dict, output: str) -> bool:
    c = item["check"]
    text = (output or "").strip()
    if c["kind"] == "numeric":
        tok = text.replace(",", "").split()
        try:
            return bool(tok) and abs(float(tok[0].rstrip(".")) - float(c["value"])) < 1e-9
        except ValueError:
            return False
    if c["kind"] == "exact":
        return text.strip(" .!\"'").lower() == str(c["value"]).lower()
    if c["kind"] == "exact_case":
        return text.strip(" .!\"'") == str(c["value"])
    if c["kind"] == "json":
        from engine.verify.core import parse_json_output  # noqa: WPS433

        return parse_json_output(text) == c["value"]
    return False
