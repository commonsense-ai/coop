"""Worker -> CLI handoff: one small JSON file, replaced atomically on every update."""

import json
import threading
import time
from pathlib import Path

FILENAME = "status.json"
BEAT = 2.0  # seconds between ticks; the view redraws at 2 Hz, so faster buys nothing


class StatusFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state: dict = {}

    def update(self, **fields) -> None:
        self.state |= fields
        self.state["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


class Heartbeat:
    """Keeps a phase that is one long blocking call visibly alive.

    A 584MB checkpoint is minutes of silence on a home link. Without a tick, status.json
    freezes, the screen shows a motionless line, and past 300s `coop status` tells the
    volunteer their healthy worker has stopped and points them at a log that says nothing
    (the daemon runs with HF_HUB_DISABLE_PROGRESS_BARS). Volunteers reported that as stuck.

    `status=None` is a no-op, so callers with no status file need no branch."""

    def __init__(self, status, phase: str, every: float = BEAT):
        self.status, self.phase, self.every = status, phase, every
        # cleared per phase: last round's byte counts would otherwise show as this
        # download's progress before the first real report arrives
        self.extra: dict = {"bytes_done": None, "bytes_total": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def bytes(self, done: int, total: int) -> None:
        """Byte counts as the transfer reports them — jumpy, and never a bar."""
        self.extra = {"bytes_done": done, "bytes_total": total}

    def beat(self) -> None:
        if self.status:
            self.status.update(
                phase=self.phase, phase_secs=round(time.monotonic() - self._t0), **self.extra
            )

    def __enter__(self) -> "Heartbeat":
        self._t0 = time.monotonic()
        self.beat()
        if self.status:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.every):
            self.beat()

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def read_status(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
