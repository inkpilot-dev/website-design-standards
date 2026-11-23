"""The teacher client against a local stub that speaks the OpenAI chat format.

This checks the request we send and how replies and failures are handled. It says
nothing about how any particular real server behaves.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from paper_distill import teacher as teacher_mod
from paper_distill.teacher import Teacher, solve_messages


class Stub(BaseHTTPRequestHandler):
    seen: list = []
    script: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, self.headers.get("Authorization"), body))
        status, payload = Stub.script.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(teacher_mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    Stub.seen, Stub.script = [], []
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def cfg(url, **over):
    return {"base_url": url, "api_key": "k", "model": "m", "temperature": 0.6, "top_p": 0.95, "max_tokens": 100,
            "timeout_s": 5, "extra_body": {"seed": 1}, "direct_body": {"reasoning_effort": "none"}, **over}


def ok(message, finish="stop"):
    return 200, {"choices": [{"message": message, "finish_reason": finish}], "usage": {"completion_tokens": 12}}


def test_request_shape_and_reply_parsing(server):
    Stub.script = [ok({"role": "assistant", "content": "Final answer: 4", "reasoning": "2+2"})]
    r = Teacher(cfg(server)).chat(solve_messages("Do it.", "What is 2+2?", True))
    path, auth, body = Stub.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["messages"] == [{"role": "user", "content": "Do it.\n\nWhat is 2+2?"}]
    assert body["model"] == "m" and body["stream"] is False and body["seed"] == 1 and "reasoning_effort" not in body
    assert (r.reasoning, r.content, r.finish_reason, r.completion_tokens) == ("2+2", "Final answer: 4", "stop", 12)


def test_direct_mode_and_system_role(server):
    Stub.script = [ok({"content": "Final answer: 4"}, "length")]
    r = Teacher(cfg(server)).chat(solve_messages("Do it.", "Q?", False), direct=True, max_tokens=7)
    body = Stub.seen[0][2]
    assert body["messages"][0] == {"role": "system", "content": "Do it."}
    assert body["reasoning_effort"] == "none" and body["max_tokens"] == 7 and r.finish_reason == "length"


def test_retries_server_errors_then_succeeds(server):
    Stub.script = [(503, {"error": "busy"}), (429, {"error": "slow down"}), ok({"content": "Final answer: 1"})]
    assert Teacher(cfg(server)).chat([{"role": "user", "content": "x"}]).content == "Final answer: 1"
    assert len(Stub.seen) == 3


def test_bad_request_fails_at_once(server):
    Stub.script = [(404, {"error": "model not found"})]
    with pytest.raises(RuntimeError, match="rejected"):
        Teacher(cfg(server)).chat([{"role": "user", "content": "x"}])
    assert len(Stub.seen) == 1


def test_gives_up_when_unreachable(server):
    Stub.script = [(500, {})] * 5
    with pytest.raises(RuntimeError, match="unreachable"):
        Teacher(cfg(server)).chat([{"role": "user", "content": "x"}])
