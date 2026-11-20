"""Prompt wording for the four starter templates (metadata lives in contracts/templates.json)."""

from __future__ import annotations

BUSINESS_SYSTEM = (
    "You are a careful bookkeeping assistant for a small business. Read the record and reply with "
    "ONLY a JSON object with exactly these fields: {fields}.{choices}{rules}{policy}"
)

SQL_SYSTEM = (
    "You write SQLite queries for a small business owner. Use only the tables below. Reply with ONLY "
    'a JSON object of the form {{"sql": "<one SELECT statement>"}}.\n\n{schema}'
)

CODE_SYSTEM = "You are a precise Python programmer. Reply with only the complete function in a ```python code block. Use only the standard library."

DOCS_SYSTEM = (
    "You answer questions about the user's own documents. Use ONLY the excerpt provided. Reply with ONLY "
    'a JSON object: {"answer": "<short answer>", "quote": "<the exact sentence from the excerpt that supports it>"}.'
)

CUSTOM_SYSTEM = "You are a helpful assistant trained on the user's own examples. Answer in the same style and format as those examples."


def business_system(fields: list[str], enums: dict[str, list[str]], rules: list[str], policy: str) -> str:
    choices = ""
    if enums:
        choices = " " + " ".join(f"{f} must be one of: {', '.join(v)}." for f, v in enums.items())
    rule_txt = (" Business rules: " + "; ".join(rules) + ".") if rules else ""
    policy_txt = ("\n\nCompany policy:\n" + policy.strip()[:1500]) if policy.strip() else ""
    return BUSINESS_SYSTEM.format(fields=", ".join(fields), choices=choices, rules=rule_txt, policy=policy_txt)
