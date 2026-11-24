"""Client for the teacher model, over any OpenAI-compatible chat endpoint."""
from __future__ import annotations

import json
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
             temperature: float | None = None, reasoning_limit: int | None = None) -> Reply:
        """One reply from the teacher.

        reasoning_limit: stop the reply as soon as its reasoning passes this many words.
        A trace that long would be rejected anyway, so there is no point paying for the rest.
        Such a reply comes back with finish_reason "reasoning_limit".
        """
        stream = bool(self.cfg.get("stream", True))
        body = {"model": self.cfg["model"], "messages": messages, "stream": stream,
                "temperature": self.cfg["temperature"] if temperature is None else temperature,
                "top_p": self.cfg.get("top_p", 0.95), "max_tokens": max_tokens or self.cfg["max_tokens"]}
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self.cfg.get("extra_body") or {})
        if direct:
            body.update(self.cfg.get("direct_body") or {})
        limit_s = self.cfg.get("timeout_s", 240)
        last = None
        for attempt in range(5):
            t0 = time.time()
            try:
                try:
                    r = self.session.post(self.url, json=body, timeout=(10, limit_s), stream=stream)
                except requests.ReadTimeout:
                    # The server accepted the request and then went quiet. Asking again rarely helps and
                    # would tie this worker up for minutes, so report it and let the caller move on.
                    return Reply("", "", "timeout", None, time.time() - t0)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise RuntimeError(f"teacher rejected the request ({r.status_code}): {r.text[:500]}")
                if stream:
                    message, finish, tokens = self._read_stream(r, reasoning_limit, t0 + limit_s)
                else:
                    data = r.json()
                    choice = data["choices"][0]
                    message, finish = choice.get("message") or {}, choice.get("finish_reason")
                    tokens = (data.get("usage") or {}).get("completion_tokens")
                reasoning, content = split_reasoning(message)
                return Reply(reasoning, content, finish, tokens, time.time() - t0)
            except (requests.RequestException, KeyError, ValueError) as e:
                last = e
                time.sleep(min(60, 2 ** attempt * 3))
        raise RuntimeError(f"teacher unreachable at {self.url} after 5 tries: {last}")

    @staticmethod
    def _read_stream(r, reasoning_limit: int | None, deadline: float) -> tuple[dict, str | None, int | None]:
        """Collect a streamed reply, cutting it short if the reasoning overruns its limit
        or the reply takes longer than the time allowed (finish_reason "timeout")."""
        reasoning, content, finish, usage, chunks = [], [], None, None, 0
        try:
            for raw in r.iter_lines():
                if time.time() > deadline:
                    finish = "timeout"
                    break
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                usage = (event.get("usage") or {}).get("completion_tokens") or usage
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
                    if isinstance(piece, str) and piece:
                        reasoning.append(piece)
                        chunks += 1
                    if delta.get("content"):
                        content.append(delta["content"])
                        chunks += 1
                    finish = choice.get("finish_reason") or finish
                if reasoning_limit and chunks % 16 == 0:
                    text = "".join(content)
                    # some servers leave the think block inline; it is reasoning until the tag closes
                    thinking = "".join(reasoning)
                    if not thinking.strip() and "</think>" not in text and text.lstrip().startswith("<think>"):
                        thinking = text
                    if len(thinking.split()) > reasoning_limit:
                        finish = "reasoning_limit"
                        break
                if finish is not None and usage is not None:
                    break  # complete; do not wait on a server that is slow to close the stream
        except requests.ConnectionError as e:
            if "timed out" not in str(e).lower():
                raise  # the server went away mid-reply: let the caller retry
            finish = "timeout"  # the stream stalled
        finally:
            r.close()  # closing the connection is what makes the server stop generating
        if finish is None:
            raise ValueError("stream ended without a finish reason")
        # servers that do not report usage: one streamed piece is close to one token
        return {"content": "".join(content), "reasoning": "".join(reasoning)}, finish, usage or chunks
