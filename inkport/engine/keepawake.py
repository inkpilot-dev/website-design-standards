"""Keep the computer awake while a long job runs.

Training takes hours and needs no keyboard or mouse, so by default the operating system puts the
computer to sleep in the middle of it (30 minutes without input on a typical Windows PC), and a
run left overnight would barely have moved by morning. This asks the system not to sleep for as
long as this process lives; the screen may still turn off, and closing a laptop's lid still
sleeps. Everything here is best effort: if the request cannot be made, training simply runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_held = False


def keep_awake(reason: str = "Inkport is training") -> bool:
    """Ask the OS not to sleep until this process exits. Returns True if the request was made."""
    global _held
    if _held:
        return True
    try:
        if sys.platform == "win32":
            import ctypes  # noqa: WPS433

            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: stays in force for this thread until it exits
            # (call from the main thread, which lives as long as the job).
            _held = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
        elif sys.platform == "darwin":
            # Exits by itself when this process does (-w).
            if shutil.which("caffeinate"):
                subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603, S607
                _held = True
        else:
            # systemd desktops: hold a sleep inhibitor for as long as this process is alive.
            if shutil.which("systemd-inhibit") and shutil.which("tail"):
                helper = subprocess.Popen(  # noqa: S603, S607
                    ["systemd-inhibit", "--what=sleep:idle", "--who=Inkport", f"--why={reason}", "--mode=block", "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    helper.wait(timeout=0.5)  # exits at once where there is no login manager to ask
                except subprocess.TimeoutExpired:
                    _held = True
    except Exception:  # noqa: BLE001 - never let this stop a job
        _held = False
    return _held
