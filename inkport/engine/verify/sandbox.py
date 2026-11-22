"""Parent side of the restricted Python test runner (verifier kind `python_tests`)."""

from __future__ import annotations

import ast
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {
    "math", "re", "itertools", "functools", "collections", "string", "datetime", "decimal",
    "fractions", "statistics", "heapq", "bisect", "operator", "typing", "dataclasses", "json",
}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "input", "exit", "quit", "help", "memoryview",
    "__builtins__", "__loader__", "__spec__",
}
RUNNER = Path(__file__).with_name("sandbox_runner.py")
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


class SandboxRejected(Exception):
    pass


def extract_code(text: str) -> str:
    m = _FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def screen(code: str) -> None:
    """Static screen before anything runs. Raises SandboxRejected with a readable reason."""
    if len(code) > 20_000:
        raise SandboxRejected("code is too long")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxRejected(f"syntax error: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise SandboxRejected(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise SandboxRejected(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SandboxRejected(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxRejected(f"access to '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise SandboxRejected(f"'{type(node).__name__}' is not allowed")


def run_tests(code: str, tests: list[dict], timeout_s: float = 6.0, mem_mb: int = 512) -> dict:
    """Returns {"passed": int, "total": int, "results": [...], "error": str|None}."""
    try:
        screen(code)
    except SandboxRejected as exc:
        return {"passed": 0, "total": len(tests), "results": [], "error": f"rejected: {exc}"}
    nonce = secrets.token_hex(16)
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    kwargs: dict = {}
    if os.name == "posix":
        # Its own session, and limits the runner puts on itself before it touches the code (see
        # sandbox_runner.py). Not a function run between fork and exec: answers are checked from
        # several threads at once, where that can leave the child stuck before it has even started
        # (which would then be counted as a wrong answer after the time limit).
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    with tempfile.TemporaryDirectory(prefix="ik-sbx-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(RUNNER)],
                input=json.dumps({"code": code, "tests": tests, "nonce": nonce, "limits": {"cpu_s": int(timeout_s) + 1, "mem_mb": mem_mb}}),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                cwd=cwd,
                env=env,
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(tests), "results": [], "error": "timeout"}
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {}
    if payload.get("nonce") != nonce:
        return {"passed": 0, "total": len(tests), "results": [], "error": f"sandbox crashed (exit {proc.returncode})"}
    results = payload.get("results", [])
    return {"passed": sum(1 for r in results if r.get("pass")), "total": len(tests), "results": results, "error": None}
