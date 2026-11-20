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
_job_id = os.environ.get("BARNRAISE_JOB_ID", "local")
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

    threading.Thread(target=_beat, name="barnraise-heartbeat", daemon=True).start()


def watch_parent_stdin() -> None:
    """Exit when the supervisor dies: it holds our stdin open; EOF means it is gone."""
    if os.environ.get("BARNRAISE_WATCH_STDIN") != "1":
        return

    def _watch() -> None:
        try:
            while sys.stdin.read(1024):
                pass
        except Exception:  # noqa: BLE001
            pass
        os._exit(3)

    threading.Thread(target=_watch, name="barnraise-parent-watch", daemon=True).start()
