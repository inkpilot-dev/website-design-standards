"""Client for the teacher model, over any OpenAI-compatible chat endpoint."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests


@dataclass
class Reply:
    reasoning: str
    content: str
    finish_reason: str | None
    completion_tokens: int | None
    seconds: float


def split_reasoning(message: dict) -> tuple[str, str]:
    """Separate reasoning from the visible reply, wherever the server put it.

    Ollama returns `reasoning`, llama.cpp returns `reasoning_content`, and some
    servers leave the think tags inline in `content`.
    """
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or message.get("thinking") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if not reasoning.strip() and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning, content = head.replace("<think>", "", 1), tail
    return reasoning.strip(), re.sub(r"^\s*<think>\s*</think>\s*", "", content).strip()


def solve_messages(instruction: str, question: str, system_in_user: bool) -> list[dict]:
    if system_in_user:
        return [{"role": "user", "content": f"{instruction}\n\n{question}"}]
    return [{"role": "system", "content": instruction}, {"role": "user", "content": question}]


class Teacher:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {cfg.get('api_key', 'local')}"

    def chat(self, messages: list[dict], *, max_tokens: int | None = None, direct: bool = False,
             temperature: float | None = None) -> Reply:
        body = {"model": self.cfg["model"], "messages": messages, "stream": False,
                "temperature": self.cfg["temperature"] if temperature is None else temperature,
                "top_p": self.cfg.get("top_p", 0.95), "max_tokens": max_tokens or self.cfg["max_tokens"]}
        body.update(self.cfg.get("extra_body") or {})
        if direct:
            body.update(self.cfg.get("direct_body") or {})
        last = None
        for attempt in range(5):
            t0 = time.time()
            try:
                r = self.session.post(self.url, json=body, timeout=self.cfg.get("timeout_s", 900))
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise RuntimeError(f"teacher rejected the request ({r.status_code}): {r.text[:500]}")
                data = r.json()
                choice = data["choices"][0]
                reasoning, content = split_reasoning(choice.get("message") or {})
                return Reply(reasoning, content, choice.get("finish_reason"),
                             (data.get("usage") or {}).get("completion_tokens"), time.time() - t0)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError, KeyError, ValueError) as e:
                last = e
                time.sleep(min(60, 2 ** attempt * 3))
        raise RuntimeError(f"teacher unreachable at {self.url} after 5 tries: {last}")
