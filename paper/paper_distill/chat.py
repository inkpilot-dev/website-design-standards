"""The exact text the student sees and produces. Training and inference both go through here."""
from __future__ import annotations

from .schema import assistant_text

_SENTINEL = "PAPER_SENTINEL_7f3a"


def prompt_text(tok, system: str, question: str) -> str:
    """System + user turns, ending where the assistant's reply begins (reasoning switched on)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def end_of_turn(tok) -> str:
    """Whatever the model's chat template writes after an assistant message, e.g. '<|im_end|>'."""
    text = tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": _SENTINEL}],
                                   tokenize=False)
    return text.split(_SENTINEL, 1)[1].strip()


def encode_example(tok, rec: dict, max_len: int, eot: str) -> dict | None:
    """Token ids and labels for one record. The loss covers only the teacher's text.

    Returns None when the example does not fit: truncating would cut off the answer.
    """
    prompt = tok(prompt_text(tok, rec["system"], rec["question"]), add_special_tokens=False)["input_ids"]
    target = tok(assistant_text(rec["reasoning"], rec["answer"]) + eot, add_special_tokens=False)["input_ids"]
    if len(prompt) + len(target) > max_len:
        return None
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}
