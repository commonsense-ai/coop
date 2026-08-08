"""Machine-local load shaping for volunteers whose hardware can't hold full tilt.

A card that drops off the PCIe bus under load is a hardware fault, and no amount of
Python fixes a microsecond transient. What this module can do is lower the peak the
card is asked to reach (split each step into smaller kernels), hand the driver its own
hard power cap, and notice when the GPU has stopped answering instead of hanging on it
forever. Only the power cap clamps transients; the rest just asks for less.

Deliberately not part of `config/run.yaml`: that file is the run's shared truth for
every volunteer, and one contributor's tired PSU is nobody else's business.
"""

import dataclasses
import logging
import os
import subprocess
import threading
import time

log = logging.getLogger(__name__)

DEFAULT_STALL_SECS = 300

# EX_TEMPFAIL: the round is abandoned, not poisoned — a supervisor should retry.
STALL_EXIT = 75


@dataclasses.dataclass(frozen=True)
class Throttle:
    """How hard this machine is willing to be pushed. All-zero is stock behaviour."""

    micro_batch: int = 0  # sequences per kernel; 0 = whole batch in one go
    power_limit_w: int | None = None
    stall_secs: int = DEFAULT_STALL_SECS

    @classmethod
    def gentle(cls, **kw) -> "Throttle":
        """The preset for marginal power delivery: the smallest kernels a batch allows."""
        return dataclasses.replace(cls(micro_batch=1), **kw)

    @property
    def shaping(self) -> bool:
        return self.micro_batch > 0

    def splits(self, batch_size: int) -> int:
        """Micro-batches to split a step's batch into. 1 means untouched."""
        if self.micro_batch <= 0:
            return 1
        return max(1, min(batch_size, -(-batch_size // self.micro_batch)))

    def describe(self, batch_size: int) -> str:
        n = self.splits(batch_size)
        bits = [f"micro-batch {batch_size // n}, {n} kernels per step"] if n > 1 else []
        if self.power_limit_w:
            bits.append(f"{self.power_limit_w} W cap")
        return ", ".join(bits) or "nothing to shape at this batch size"

    def apply(self, device: str) -> None:
        """Enforce the caps that live outside the training loop."""
        if self.power_limit_w and device.startswith("cuda"):
            log.info("%s", set_power_limit(self.power_limit_w))


def set_power_limit(watts: int) -> str:
    """The only real defence against transient spikes: the card's own power governor.
    Needs root, so a refusal is normal — say exactly what to run instead."""
    manual = f"sudo nvidia-smi -pm 1 && sudo nvidia-smi --power-limit {watts}"
    try:
        p = subprocess.run(
            ["nvidia-smi", "--power-limit", str(watts)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not run nvidia-smi ({e}) — set the cap yourself:\n    {manual}"
    if p.returncode == 0:
        return f"GPU power cap set to {watts} W"
    return f"could not set the {watts} W GPU power cap (needs root) — run:\n    {manual}"


class StallWatchdog:
    """Fail loudly when the GPU stops answering.

    A card that has fallen off the bus leaves the training thread parked in an
    uninterruptible driver ioctl, so nothing in Python can unstick it — a plain
    timeout would just hang next to it. Detect the stall, name it in the log, and
    request exit. The process may still linger until the driver call returns; that
    part needs a device reset, not us.
    """

    def __init__(self, stall_secs: int = DEFAULT_STALL_SECS, on_stall=None):
        self.stall_secs = stall_secs
        self._on_stall = on_stall or _exit_stalled
        self._beat = time.monotonic()
        self._done = threading.Event()

    def beat(self) -> None:
        self._beat = time.monotonic()

    def __enter__(self) -> "StallWatchdog":
        self.beat()
        if self.stall_secs > 0:
            threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._done.set()
        return False

    def _watch(self) -> None:
        poll = max(1.0, min(self.stall_secs / 4, 15.0))
        while not self._done.wait(poll):
            idle = time.monotonic() - self._beat
            if idle >= self.stall_secs:
                self._on_stall(idle)
                return


def _exit_stalled(idle: float) -> None:
    log.error(
        "no training progress for %.0fs — the GPU has stopped responding. "
        "Check `nvidia-smi`; if it errors the card needs a reset or a power cycle.",
        idle,
    )
    os._exit(STALL_EXIT)
