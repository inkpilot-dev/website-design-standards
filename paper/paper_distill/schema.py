"""The one record format used everywhere, and its validator.

A dataset file is JSON Lines: one record per line, every record with exactly the
keys below, in this order. `assistant_text` is the single place that turns a record
into the text the student is trained to produce.
"""
from __future__ import annotations

import re

from .verify import FINAL_RE, extract_final_answer

FIELDS = ["id", "source", "domain", "skill", "difficulty", "mode", "system", "question", "reasoning",
          "answer", "final_answer", "verification", "teacher", "stats"]
MODES = ("think", "direct")
VERIFICATION_METHODS = ("ground_truth", "teacher_agreement")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def assistant_text(reasoning: str, answer: str) -> str:
    """Reasoning inside think tags, then the visible answer. An empty think block means a direct answer."""
    return f"{THINK_OPEN}\n{reasoning.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"


def make_record(*, seed: dict, system: str, reasoning: str, answer: str, final_answer: str, mode: str,
                method: str, reference: str, teacher: dict, completion_tokens: int | None) -> dict:
    return {
        "id": seed["id"],
        "source": seed["source"],
        "domain": seed["domain"],
        "skill": seed["skill"],
        "difficulty": int(seed["difficulty"]),
        "mode": mode,
        "system": system,
        "question": seed["question"].strip(),
        "reasoning": reasoning.strip(),
        "answer": answer.strip(),
        "final_answer": final_answer,
        "verification": {"method": method, "reference": reference, "answer_type": seed["answer_type"]},
        "teacher": {"model": teacher["model"], "license": teacher["license"],
                    "temperature": float(teacher["temperature"])},
        "stats": {"reasoning_words": len(reasoning.split()), "answer_words": len(answer.split()),
                  "completion_tokens": completion_tokens},
    }


def validate_record(rec: dict) -> list[str]:
    """Every way this record breaks the format. An empty list means it is valid."""
    errs: list[str] = []
    if list(rec.keys()) != FIELDS:
        return [f"keys {list(rec.keys())} != {FIELDS}"]
    for k in ("id", "source", "domain", "skill", "mode", "system", "question", "reasoning", "answer", "final_answer"):
        if not isinstance(rec[k], str):
            errs.append(f"{k} is not a string")
    if errs:
        return errs
    if not re.fullmatch(r"[0-9a-f]{16}", rec["id"]):
        errs.append("id is not 16 hex characters")
    if not (rec["source"] in ("procedural", "teacher_authored") or rec["source"].startswith("public:")):
        errs.append(f"unknown source {rec['source']!r}")
    if not (isinstance(rec["difficulty"], int) and 1 <= rec["difficulty"] <= 5):
        errs.append("difficulty is not an integer 1..5")
    if rec["mode"] not in MODES:
        errs.append(f"unknown mode {rec['mode']!r}")
    for k in ("domain", "skill", "system", "question", "answer", "final_answer"):
        if not rec[k].strip():
            errs.append(f"{k} is empty")
        if rec[k] != rec[k].strip():
            errs.append(f"{k} has leading or trailing whitespace")
    if rec["mode"] == "think" and not rec["reasoning"].strip():
        errs.append("think-mode record has no reasoning")
    if rec["mode"] == "direct" and rec["reasoning"]:
        errs.append("direct-mode record has reasoning")
    for k in ("reasoning", "answer"):
        if THINK_OPEN in rec[k] or THINK_CLOSE in rec[k]:
            errs.append(f"{k} contains a think tag")
    last = [ln for ln in rec["answer"].splitlines() if ln.strip()][-1:] or [""]
    if not FINAL_RE.match(last[0]):
        errs.append("answer does not end with a 'Final answer:' line")
    elif extract_final_answer(rec["answer"]) != rec["final_answer"]:
        errs.append("final_answer does not match the answer's last line")
    v = rec["verification"]
    if not (isinstance(v, dict) and list(v.keys()) == ["method", "reference", "answer_type"]
            and v["method"] in VERIFICATION_METHODS and isinstance(v["reference"], str) and v["reference"]):
        errs.append("verification block is malformed")
    t = rec["teacher"]
    if not (isinstance(t, dict) and list(t.keys()) == ["model", "license", "temperature"]
            and all(isinstance(t[k], str) and t[k] for k in ("model", "license"))):
        errs.append("teacher block is malformed")
    st = rec["stats"]
    if not (isinstance(st, dict) and list(st.keys()) == ["reasoning_words", "answer_words", "completion_tokens"]):
        errs.append("stats block is malformed")
    return errs
