"""Runs untrusted, model-written Python against tests. Executed as a separate interpreter:

    python -I -S -B sandbox_runner.py   (request JSON on stdin, result JSON on stdout)

Defense in depth (see engine/verify/sandbox.py for the parent-side controls):
  * the parent has already AST-screened the code (no imports outside an allowlist, no dunder
    access, no eval/exec/open);
  * this process installs a PEP 578 audit hook *before* touching user code. Audit hooks cannot
    be removed, and they veto file writes, reads outside the stdlib, sockets, subprocesses,
    os.system/exec/fork/kill, ctypes and similar escape hatches;
  * before anything else this process puts CPU, memory, file-size and open-file limits on itself
    (POSIX; they cannot be raised again), and the parent enforces a wall-clock time limit.
This file intentionally imports only the standard library.
"""

import ast
import io
import json
import os
import sys
import sysconfig


def _stdlib_roots():
    roots = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            roots.add(os.path.realpath(p))
    roots.add(os.path.realpath(os.path.dirname(os.__file__)))
    return tuple(sorted(roots))


_ALLOWED_READ_ROOTS = _stdlib_roots()
_BLOCKED_PREFIXES = (
    "socket.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.remove",
    "os.unlink",
    "os.rename",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.symlink",
    "os.link",
    "os.truncate",
    "os.putenv",
    "os.unsetenv",
    "os.startfile",
    "shutil.",
    "ctypes.",
    "winreg.",
    "urllib.",
    "http.",
    "ftplib.",
    "smtplib.",
    "webbrowser.",
    "sys.setrecursionlimit",
    "code.__new__",
    "pty.",
    "resource.setrlimit",
)
_BLOCKED_IMPORTS = {"ctypes", "_ctypes", "socket", "_socket", "subprocess", "_posixsubprocess", "multiprocessing", "_multiprocessing", "threading", "_thread", "asyncio", "signal", "pty", "mmap", "resource", "winreg", "msvcrt"}
_armed = False


def _audit(event, args):
    if not _armed:
        return
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(path, int):
            raise PermissionError("sandbox: opening file descriptors is not allowed")
        mode = mode or "r"
        if any(c in str(mode) for c in "wax+"):
            raise PermissionError("sandbox: writing files is not allowed")
        real = os.path.realpath(str(path))
        if not real.startswith(_ALLOWED_READ_ROOTS):
            raise PermissionError("sandbox: reading files is not allowed")
        return
    if event == "import" and args and str(args[0]).split(".")[0] in _BLOCKED_IMPORTS:
        raise PermissionError(f"sandbox: importing {args[0]} is not allowed")
    for prefix in _BLOCKED_PREFIXES:
        if event.startswith(prefix):
            raise PermissionError(f"sandbox: {event} is not allowed")


def _equal(result, expect_src):
    if repr(result) == expect_src.strip():
        return True
    try:
        return result == ast.literal_eval(expect_src)
    except Exception:  # noqa: BLE001
        return False


def _limit_myself(limits):
    """CPU seconds, address space, file size, open files. Hard limits: the code cannot lift them."""
    try:
        import resource
    except ImportError:  # Windows: the parent's wall-clock limit is what applies
        return
    cpu_s, mem_mb = int(limits.get("cpu_s", 7)), int(limits.get("mem_mb", 512))
    wanted = [(resource.RLIMIT_CPU, cpu_s, cpu_s + 1), (resource.RLIMIT_FSIZE, 1 << 20, 1 << 20), (resource.RLIMIT_NOFILE, 64, 64)]
    if sys.platform != "darwin":  # RLIMIT_AS is unreliable on macOS
        wanted.append((resource.RLIMIT_AS, mem_mb << 20, mem_mb << 20))
    for which, soft, hard in wanted:
        try:
            resource.setrlimit(which, (soft, hard))
        except (ValueError, OSError):  # already lower than this
            pass


def main():
    global _armed
    req = json.loads(sys.stdin.read())
    _limit_myself(req.get("limits") or {})
    real_stdout = sys.stdout
    nonce = req["nonce"]
    code, tests = req["code"], req["tests"]
    sys.addaudithook(_audit)
    # Pre-import what the harness itself needs, then arm the hook.
    import math, re, itertools, functools, collections, string, datetime, decimal, fractions, statistics, heapq, bisect, operator, typing, dataclasses  # noqa: E401,F401

    _armed = True
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    namespace = {"__name__": "solution", "__builtins__": __builtins__}
    results = []
    load_error = None
    try:
        exec(compile(code, "solution.py", "exec"), namespace)  # noqa: S102 - this IS the sandbox
    except BaseException as exc:  # noqa: BLE001
        load_error = f"{type(exc).__name__}: {exc}"
    for t in tests:
        if load_error:
            results.append({"call": t["call"], "pass": False, "error": load_error})
            continue
        try:
            value = eval(compile(t["call"], "test", "eval"), namespace)  # noqa: S307
            results.append({"call": t["call"], "pass": _equal(value, t["expect"]), "got": repr(value)[:200]})
        except BaseException as exc:  # noqa: BLE001
            results.append({"call": t["call"], "pass": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
    real_stdout.write(json.dumps({"nonce": nonce, "results": results}) + "\n")
    real_stdout.flush()


if __name__ == "__main__":
    main()
