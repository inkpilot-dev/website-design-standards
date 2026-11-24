"""The teacher client against a local stub that speaks the OpenAI chat format.

This checks the request we send and how replies and failures are handled. It says
nothing about how any particular real server behaves.
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from paper_distill import teacher as teacher_mod
from paper_distill.teacher import Teacher, solve_messages

_sleep = time.sleep  # the real one; the fixture below silences time.sleep to skip retry back-off


class Stub(BaseHTTPRequestHandler):
    seen: list = []
    script: list = []
    sent_pieces = 0
    inline_tags = False
    send_usage = True
    stall_after = None   # stop sending after this many pieces, keeping the connection open
    stall_headers = False

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append((self.path, self.headers.get("Authorization"), body))
        status, payload = Stub.script.pop(0)
        if Stub.stall_headers:
            _sleep(3)
            return
        if status == 200 and body.get("stream"):
            return self.stream(payload)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream(self, payload):
        """Send the scripted message a word at a time, as server-sent events."""
        msg = payload["choices"][0]["message"]
        reasoning, content = msg.get("reasoning") or "", msg.get("content") or ""
        if Stub.inline_tags and reasoning:
            reasoning, content = "", f"<think>\n{reasoning}\n</think>\n\n{content}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for field, text in (("reasoning", reasoning), ("content", content)):
                for piece in re.findall(r"\S+\s*|\s+", text):  # pieces that join back to the exact text
                    self.event({"choices": [{"delta": {field: piece}, "finish_reason": None}]})
                    Stub.sent_pieces += 1
                    _sleep(0.0005)
                    if Stub.stall_after is not None and Stub.sent_pieces >= Stub.stall_after:
                        _sleep(3)
                        return
            self.event({"choices": [{"delta": {}, "finish_reason": payload["choices"][0]["finish_reason"]}]})
            if Stub.send_usage:
                self.event({"choices": [], "usage": payload["usage"]})
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionError):
            pass  # the client hung up, which is the point of the early-stop test

    def event(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(teacher_mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    Stub.seen, Stub.script, Stub.sent_pieces, Stub.inline_tags, Stub.send_usage = [], [], 0, False, True
    Stub.stall_after, Stub.stall_headers = None, False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def cfg(url, **over):
    return {"base_url": url, "api_key": "k", "model": "m", "temperature": 0.6, "top_p": 0.95, "max_tokens": 100,
            "timeout_s": 5, "extra_body": {"seed": 1}, "direct_body": {"reasoning_effort": "none"}, **over}


def ok(message, finish="stop"):
    return 200, {"choices": [{"message": message, "finish_reason": finish}], "usage": {"completion_tokens": 12}}


@pytest.mark.parametrize("stream", [True, False])
def test_request_shape_and_reply_parsing(server, stream):
    Stub.script = [ok({"role": "assistant", "content": "It is four.\n\nFinal answer: 4", "reasoning": "2 + 2"})]
    r = Teacher(cfg(server, stream=stream)).chat(solve_messages("Do it.", "What is 2+2?", True))
    path, auth, body = Stub.seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer k"
    assert body["messages"] == [{"role": "user", "content": "Do it.\n\nWhat is 2+2?"}]
    assert body["model"] == "m" and body["stream"] is stream and body["seed"] == 1 and "reasoning_effort" not in body
    assert ("stream_options" in body) is stream
    assert (r.reasoning, r.content, r.finish_reason, r.completion_tokens) == \
        ("2 + 2", "It is four.\n\nFinal answer: 4", "stop", 12)


def test_direct_mode_and_system_role(server):
    Stub.script = [ok({"content": "Final answer: 4"}, "length")]
    r = Teacher(cfg(server)).chat(solve_messages("Do it.", "Q?", False), direct=True, max_tokens=7)
    body = Stub.seen[0][2]
    assert body["messages"][0] == {"role": "system", "content": "Do it."}
    assert body["reasoning_effort"] == "none" and body["max_tokens"] == 7 and r.finish_reason == "length"


@pytest.mark.parametrize("inline", [False, True])
def test_long_reasoning_is_stopped_early(server, inline):
    Stub.inline_tags = inline
    Stub.script = [ok({"content": "Final answer: 4", "reasoning": "word " * 3000})]
    r = Teacher(cfg(server)).chat([{"role": "user", "content": "x"}], reasoning_limit=100)
    assert r.finish_reason == "reasoning_limit"
    assert 100 < len((r.reasoning or r.content).split()) < 200  # stopped just past the limit
    _sleep(0.2)
    assert Stub.sent_pieces < 1500  # the server did not get to send the rest


def test_reasoning_under_the_limit_is_untouched(server):
    Stub.script = [ok({"content": "Final answer: 4", "reasoning": "word " * 80})]
    r = Teacher(cfg(server)).chat([{"role": "user", "content": "x"}], reasoning_limit=100)
    assert r.finish_reason == "stop" and len(r.reasoning.split()) == 80 and r.content == "Final answer: 4"


def test_tokens_are_estimated_when_the_server_reports_none(server):
    Stub.send_usage = False
    Stub.script = [ok({"content": "Final answer: 4", "reasoning": "one two three"})]
    r = Teacher(cfg(server)).chat([{"role": "user", "content": "x"}])
    assert r.completion_tokens == 6  # three reasoning pieces and three content pieces


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


def test_a_stalled_stream_times_out_instead_of_hanging(server):
    Stub.stall_after = 5
    Stub.script = [ok({"content": "Final answer: 4", "reasoning": "word " * 50})]
    t0 = time.time()
    r = Teacher(cfg(server, timeout_s=0.4)).chat([{"role": "user", "content": "x"}])
    assert r.finish_reason == "timeout" and time.time() - t0 < 2
    assert len(Stub.seen) == 1  # it did not ask again


def test_a_server_that_never_answers_times_out_without_retrying(server):
    Stub.stall_headers = True
    Stub.script = [ok({"content": "x"})]
    t0 = time.time()
    r = Teacher(cfg(server, timeout_s=0.4)).chat([{"role": "user", "content": "x"}])
    assert r.finish_reason == "timeout" and time.time() - t0 < 2 and len(Stub.seen) == 1


def test_reply_does_not_wait_for_the_server_to_close_the_stream(server):
    Stub.script = [ok({"content": "Final answer: 4", "reasoning": "short"})]
    r = Teacher(cfg(server)).chat([{"role": "user", "content": "x"}])
    assert (r.finish_reason, r.completion_tokens, r.content) == ("stop", 12, "Final answer: 4")
