"""NDJSON event protocol (contracts/schemas/event.schema.json).

Every line on the protocol stream is one JSON object. The protocol stream is a private
duplicate of the original stdout file descriptor: after ``install_protocol_stdout()`` runs,
``sys.stdout`` points at stderr, so stray prints from third-party libraries can never corrupt
the protocol that the Rust supervisor parses.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from typing import Any, Optional

SCHEMA_VERSION = 1

_lock = threading.Lock()
_stream = sys.stdout
_job_id = os.environ.get("INKPORT_JOB_ID", "local")
_last_emit = time.monotonic()
_heartbeat_started = False


def set_job_id(job_id: str) -> None:
    global _job_id
    _job_id = job_id


def install_protocol_stdout() -> None:
    """Move the protocol to a private fd and point sys.stdout at stderr."""
    global _stream
    try:
        fd = os.dup(sys.stdout.fileno())
        _stream = os.fdopen(fd, "w", buffering=1, encoding="utf-8", newline="\n")
        sys.stdout = sys.stderr
    except (OSError, ValueError, AttributeError):
        _stream = sys.stdout


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        try:
            return _clean(value.item())
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def emit(event_type: str, **fields: Any) -> dict:
    global _last_emit
    record = {"v": SCHEMA_VERSION, "type": event_type, "job_id": _job_id, "ts": round(time.time(), 3)}
    record.update(_clean(fields))
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    with _lock:
        _stream.write(line + "\n")
        _stream.flush()
        _last_emit = time.monotonic()
    return record


def log(message: str, level: str = "info") -> None:
    emit("log", level=level, message=str(message))


def stage(name: str, status: str, detail: str = "") -> None:
    fields: dict[str, Any] = {"stage": name, "status": status}
    if detail:
        fields["detail"] = detail
    emit("stage", **fields)


def error(code: str, message: str, retryable: bool = False) -> None:
    emit("error", code=code, message=message, retryable=retryable)


def done(ok: bool, **result: Any) -> None:
    emit("done", ok=ok, result=result)


class Progress:
    """Tracks a stage's progress and converts it into overall progress plus ETA."""

    def __init__(self, stage_name: str, total: int, overall_start: float, overall_end: float):
        self.stage = stage_name
        self.total = max(int(total), 1)
        self.o0 = overall_start
        self.o1 = overall_end
        self.t0 = time.monotonic()
        self.last_sent = 0.0

    def update(self, step: int, metrics: Optional[dict] = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_sent < 0.25 and step < self.total:
            return
        self.last_sent = now
        step = max(0, min(int(step), self.total))
        frac = step / self.total
        elapsed = now - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0.02 else None
        emit(
            "progress",
            stage=self.stage,
            step=step,
            total=self.total,
            pct=round(100.0 * frac, 2),
            overall_pct=round(self.o0 + (self.o1 - self.o0) * frac, 2),
            eta_s=None if eta is None else round(eta, 1),
            metrics=metrics or {},
        )


def start_heartbeat(interval_s: float = 15.0) -> None:
    """Emit heartbeats while long silent operations (model loading) run."""
    global _heartbeat_started
    if _heartbeat_started:
        return
    _heartbeat_started = True

    def _beat() -> None:
        while True:
            time.sleep(interval_s / 3)
            if time.monotonic() - _last_emit >= interval_s:
                emit("heartbeat")

    threading.Thread(target=_beat, name="inkport-heartbeat", daemon=True).start()


def watch_parent_stdin() -> None:
    """Exit when the supervisor dies: it holds our stdin open; EOF means it is gone."""
    if os.environ.get("INKPORT_WATCH_STDIN") != "1":
        return
    target = _watch_stdin_windows if os.name == "nt" else _watch_stdin_posix
    threading.Thread(target=target, name="inkport-parent-watch", daemon=True).start()


def _watch_stdin_posix() -> None:
    try:
        while sys.stdin.read(1024):
            pass
    except Exception:  # noqa: BLE001
        pass
    os._exit(3)


def _watch_stdin_windows() -> None:
    """Poll the pipe instead of blocking in a read.

    On Windows a blocking read on the stdin pipe serializes every other operation on that pipe,
    including the start-up of any child process that inherits it: the child (and the thread that
    waits for it) would hang forever. PeekNamedPipe returns immediately and reports a broken pipe
    once the supervisor is gone.
    """
    try:
        import ctypes  # noqa: WPS433
        import msvcrt  # noqa: WPS433
        from ctypes import wintypes  # noqa: WPS433

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetFileType.argtypes = [wintypes.HANDLE]
        k32.GetFileType.restype = wintypes.DWORD
        k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD]
        k32.PeekNamedPipe.restype = wintypes.BOOL
        fd = sys.stdin.fileno()
        handle = msvcrt.get_osfhandle(fd)
        if k32.GetFileType(handle) != 3:  # FILE_TYPE_PIPE; a console or NUL is never watched
            return
        while True:
            avail = wintypes.DWORD(0)
            if not k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
                if ctypes.get_last_error() in (109, 233):  # ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED
                    break
                return  # anything unexpected: stop watching rather than risk killing a healthy job
            if avail.value:
                os.read(fd, min(int(avail.value), 65536))  # data is waiting, so this cannot block
            time.sleep(0.5)
    except Exception:  # noqa: BLE001
        return
    os._exit(3)
