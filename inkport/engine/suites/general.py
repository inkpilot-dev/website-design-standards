"""Built-in general-skills regression suite and self-replay prompts.

Generated deterministically (no downloads, no licensing questions). The gate compares the
base model and the candidate on this suite to catch catastrophic forgetting. The replay
prompts (a disjoint seed) are answered by the *base* model itself and mixed into SFT, which
anchors general behavior without importing anyone else's data.
"""

from __future__ import annotations

import json
import random

WORDS = ["apple", "river", "blue", "stone", "cloud", "maple", "north", "pixel", "garden", "lantern", "harbor", "violet", "copper", "meadow", "canyon"]
KEYS = ["color", "name", "city", "animal", "item", "status"]


def _items(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    kinds = ["add", "word", "upper", "json", "count", "compare"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "add":
            a, b = rng.randint(2, 60), rng.randint(2, 60)
            q, ref, check = f"What is {a} + {b}? Reply with only the number.", str(a + b), {"kind": "numeric", "value": a + b}
        elif kind == "word":
            w = rng.choice(WORDS)
            q, ref, check = f"Reply with only the word: {w}", w, {"kind": "exact", "value": w}
        elif kind == "upper":
            w = rng.choice(WORDS)
            q, ref, check = f"Write the word '{w}' in uppercase letters only.", w.upper(), {"kind": "exact_case", "value": w.upper()}
        elif kind == "json":
            k, v = rng.choice(KEYS), rng.choice(WORDS)
            q, ref, check = f'Return a JSON object with the key "{k}" set to "{v}". Reply with JSON only.', json.dumps({k: v}), {"kind": "json", "value": {k: v}}
        elif kind == "count":
            w = rng.choice(WORDS)
            q, ref, check = f"How many letters are in the word '{w}'? Reply with only the number.", str(len(w)), {"kind": "numeric", "value": len(w)}
        else:
            a, b = rng.sample(range(5, 99), 2)
            q, ref, check = f"Which is larger, {a} or {b}? Reply with only the number.", str(max(a, b)), {"kind": "numeric", "value": max(a, b)}
        out.append({"id": f"general-{seed}-{i}", "messages": [{"role": "user", "content": q}], "reference": ref, "check": check})
    return out


def suite(n: int = 60) -> list[dict]:
    return _items(seed=7, n=n)


def replay_prompts(n: int = 64) -> list[dict]:
    extra = [
        "Give one short tip for staying organized at work.",
        "Write one friendly sentence welcoming a new customer.",
        "Explain in one sentence what a budget is.",
        "List three common office supplies, separated by commas.",
        "Summarize in one sentence why backups matter.",
        "Write a polite one-sentence reminder that an invoice is due.",
    ]
    items = _items(seed=99, n=max(0, n - len(extra)))
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
