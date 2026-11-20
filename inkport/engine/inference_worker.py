"""Local inference worker: serves a project's active version over an OpenAI-style HTTP API.

Bound to 127.0.0.1 on an ephemeral port and protected by a shared secret
(X-Inkport-Worker-Token). It is never exposed directly: the Rust gateway on
127.0.0.1:11435 adds user authentication, Host/Origin checks and stable model names, then
proxies here (streaming included). Emits {"type":"ready","port":N} once the model is loaded.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from engine import events
from engine.backends.base import Backend, make_backend
from engine.contract import ProjectPaths, active_version, load_project, read_jsonl
from engine.hardware_profile import detect, make_plan

MAX_BODY = 4 * 1024 * 1024


def serving_plan(root: Path, version: Optional[str] = None) -> tuple[dict, Optional[Path], str]:
    project = load_project(root)
    paths = ProjectPaths(root)
    act = active_version(root)
    version = version or (act["version"] if act else None)
    adapter = None
    backend = project.get("backend")
    quant = project.get("quant")
    if version:
        adapter = paths.versions / version / "adapter"
        meta_p = adapter / "inkport_adapter.json"
        if meta_p.is_file():
            meta = json.loads(meta_p.read_text())
            backend = "torch" if meta.get("backend") in ("torch", "unsloth") else meta.get("backend", backend)
            quant = meta.get("quant", quant)
    report = detect()
    plan = make_plan({**project, "backend": backend, "quant": quant}, report, preset="standard")
    if backend:
        plan["backend"] = "torch" if backend == "unsloth" else backend
    if quant:
        plan["quant"] = quant
    return plan, adapter, version or "base"


def default_system_prompt(root: Path) -> Optional[str]:
    for split in ("train", "dev"):
        for it in read_jsonl(ProjectPaths(root).split(split))[:1]:
            for m in it["messages"]:
                if m["role"] == "system":
                    return m["content"]
    return None


class Worker:
    def __init__(self, root: Path, version: Optional[str] = None):
        self.root = root
        self.project = load_project(root)
        self.slug = self.project.get("slug") or self.project["id"]
        self.lock = threading.Lock()
        self.backend: Optional[Backend] = None
        self.version = "base"
        self.system = default_system_prompt(root)
        self.load(version)

    @property
    def model_ids(self) -> list[str]:
        return [f"inkport/{self.slug}", f"inkport/{self.slug}@base"]

    def load(self, version: Optional[str]) -> None:
        plan, adapter, ver = serving_plan(self.root, version)
        events.log(f"Loading {plan['base_model']['id']} ({plan['backend']}) version {ver}", "info")
        b = make_backend(plan, inference_only=True)
        b.load(adapter if adapter and adapter.is_dir() else None)
        with self.lock:
            old, self.backend, self.version, self.plan = self.backend, b, ver, plan
        if old is not None:
            old.unload()

    def prepare(self, body: dict) -> tuple[list[dict], bool, int, float, float]:
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError("messages must be a non-empty list")
        clean = []
        for m in msgs:
            role, content = m.get("role"), m.get("content")
            if isinstance(content, list):  # OpenAI content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("each message needs role system|user|assistant and text content")
            clean.append({"role": role, "content": content})
        if self.system and not any(m["role"] == "system" for m in clean) and not body.get("inkport_raw"):
            clean.insert(0, {"role": "system", "content": self.system})
        model = str(body.get("model") or self.model_ids[0])
        use_adapter = not model.endswith("@base")
        max_tokens = max(1, min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 512), 4096))
        temperature = float(body.get("temperature") if body.get("temperature") is not None else 0.0)
        top_p = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        return clean, use_adapter, max_tokens, max(0.0, min(temperature, 2.0)), max(0.01, min(top_p, 1.0))

    def count(self, text: str) -> int:
        tok = getattr(self.backend, "tok", None)
        try:
            return len(tok.encode(text, add_special_tokens=False)) if tok is not None else len(text.split())
        except Exception:  # noqa: BLE001
            return len(text.split())


def make_handler(worker: Worker, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "inkport-worker/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # route access logs to stderr only
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            got = self.headers.get("X-Inkport-Worker-Token", "")
            if not token or not hmac.compare_digest(got, token):
                self._json(401, {"error": {"message": "unauthorized", "type": "auth_error"}})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                return self._json(200, {"ok": True, "version": worker.version, "models": worker.model_ids})
            if not self._authorized():
                return None
            if self.path == "/v1/models":
                now = int(time.time())
                return self._json(200, {"object": "list", "data": [{"id": m, "object": "model", "created": now, "owned_by": "inkport"} for m in worker.model_ids]})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._json(413 if length > MAX_BODY else 400, {"error": {"message": "invalid body size"}})
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return self._json(400, {"error": {"message": "body must be JSON"}})
            if self.path == "/internal/reload":
                try:
                    worker.load(body.get("version"))
                    return self._json(200, {"ok": True, "version": worker.version})
                except Exception as exc:  # noqa: BLE001
                    return self._json(500, {"error": {"message": str(exc)}})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                msgs, use_adapter, max_tokens, temperature, top_p = worker.prepare(body)
            except (ValueError, TypeError) as exc:
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            model = str(body.get("model") or worker.model_ids[0])
            backend = worker.backend
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def send(delta: dict, finish: Optional[str] = None) -> None:
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                try:
                    send({"role": "assistant"})
                    for piece in backend.stream(msgs, max_tokens, temperature, top_p, use_adapter=use_adapter):
                        send({"content": piece})
                    send({}, "stop")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return None
            try:
                text = backend.generate([msgs], max_tokens, temperature, top_p, 1, 0, use_adapter=use_adapter)[0][0]
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"error": {"message": f"generation failed: {exc}"}})
            prompt_tokens = sum(worker.count(m["content"]) for m in msgs)
            completion_tokens = worker.count(text)
            return self._json(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                    "inkport": {"version": worker.version, "adapter": use_adapter},
                },
            )

    return Handler


def serve(root: Path, port: int = 0, version: Optional[str] = None) -> None:
    token = os.environ.get("INKPORT_WORKER_TOKEN", "")
    if not token:
        raise ValueError("INKPORT_WORKER_TOKEN must be set (the gateway passes it)")
    worker = Worker(root, version)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(worker, token))
    httpd.daemon_threads = True
    events.emit("ready", host="127.0.0.1", port=httpd.server_address[1], model=worker.model_ids[0])
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
