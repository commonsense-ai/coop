"""Which accelerator this machine trains on, and how to say it to a volunteer.

Detection lives in one place so the worker, the trainer, and the CLI can never
disagree about what a donor is running on.
"""

import functools
import logging
import re
import shutil
import subprocess
from typing import NamedTuple

import torch

log = logging.getLogger(__name__)

PLAIN = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}
# leaderboard spelling; xla is here so a TPU round labels itself the day TPUs land
KIND = {"cuda": "nvidia-gpu", "mps": "apple-gpu", "xla": "google-tpu", "cpu": "cpu"}
CUDA_FIX = "install a CUDA torch: `uv pip install --torch-backend auto torch`"
NEWER_FIX = "update your NVIDIA driver, then: `uv pip install -U --torch-backend auto torch`"
OLD_CARD = "only an older torch still ships kernels for this card"
# cudaErrorNoKernelImageForDevice, as torch words it when a launch finds no cubin
NO_KERNEL = "no kernel image is available"


class Gap(NamedTuple):
    """Why an NVIDIA machine is going unused, and what fixes it — if anything does."""

    reason: str
    fix: str | None


def pick_device() -> str:
    if torch.cuda.is_available() and not arch_gap():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def kind(device: str) -> str:
    """What the leaderboard calls this machine. Finer than gpu/cpu — a board people read
    to decide what to donate should say whose silicon did the work. No spaces: coop
    clients on 0.3.0 and earlier match this cell with `\\S+`, and a space there blanks their
    `coop status` totals."""
    return KIND.get(device.partition(":")[0], "cpu")


def describe(device: str) -> str:
    """The name a donor recognises — their actual card when torch will tell us."""
    base, _, index = device.partition(":")
    if base != "cuda":
        return PLAIN.get(base, device)
    try:
        props = torch.cuda.get_device_properties(int(index) if index else 0)
        return f"{props.name} ({props.total_memory / 1024**3:.0f} GB)"
    except Exception:
        return PLAIN["cuda"]  # a name is a nicety; never fail a command over it


@functools.cache
def nvidia_present() -> bool:
    """Is there an NVIDIA driver here, whether or not torch can reach it?

    Broad except on purpose: this only decides whether to print a hint, so any
    surprise from shelling out must degrade to "can't tell", never break the
    command it was called from."""
    if not shutil.which("nvidia-smi"):
        return False  # the common case: no driver tooling, so spawn nothing at all
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return p.returncode == 0 and "GPU" in p.stdout
    except Exception:
        return False


def compiled_archs() -> tuple[list[int], list[int]]:
    """(cubin, PTX) capabilities this torch was built for, each as 10*major + minor."""
    cubin, ptx = [], []
    for entry in torch.cuda.get_arch_list():
        m = re.fullmatch(r"(sm|compute)_(\d+)[a-z]*", entry)
        if m:
            (cubin if m[1] == "sm" else ptx).append(int(m[2]))
    return sorted(cubin), sorted(ptx)


@functools.cache
def arch_gap(index: int = 0) -> Gap | None:
    """This torch holds no kernels for this card, so the first launch of every round
    dies with cudaErrorNoKernelImageForDevice. `torch.cuda.is_available()` is True
    throughout — driver and card are both fine — so nothing catches it earlier, and
    a retry loop reruns the identical failure forever."""
    try:
        major, minor = torch.cuda.get_device_capability(index)
        cubin, ptx = compiled_archs()
    except Exception:
        return None  # can't tell; a launch failure is still caught while training
    cc = major * 10 + minor
    if not cubin:
        return None
    # a cubin covers its own major version from its minor upward; PTX JITs onto
    # anything at or above the capability it was written for
    if any(b // 10 == cc // 10 and b % 10 <= cc % 10 for b in cubin) or any(p <= cc for p in ptx):
        return None
    span = f"sm_{cubin[0]}" if len(cubin) == 1 else f"sm_{cubin[0]}-sm_{cubin[-1]}"
    reason = f"{describe(f'cuda:{index}')} needs sm_{cc} kernels; this torch has {span}"
    return Gap(reason, NEWER_FIX if cc > cubin[-1] else OLD_CARD)


def cuda_gap() -> Gap | None:
    """Why an NVIDIA box is about to use its CPU instead. The costliest thing a
    donor can get silently wrong: PyPI's default torch is CPU-only on Windows, so
    plain resolution strands GPU owners without ever saying so."""
    if torch.cuda.is_available():
        return arch_gap()
    if not nvidia_present():
        return None
    if getattr(torch.version, "cuda", None) is None:
        return Gap("this torch was built without CUDA support", CUDA_FIX)
    return Gap("torch can't reach the driver", CUDA_FIX)


def kernel_missing(e: BaseException) -> bool:
    return NO_KERNEL in str(e)


def cpu_fallback(gap: Gap | None = None) -> str:
    """A card torch has no kernels for fails identically every round, so retrying is
    an infinite loop that donates nothing. Say why, then keep the machine useful."""
    gap = gap or arch_gap()
    log.error("your NVIDIA GPU can't run this torch build%s", f" — {gap.reason}" if gap else "")
    if gap and gap.fix:
        log.error("fix: %s", gap.fix)
    log.warning("training on your CPU instead — slower, but it still counts")
    return "cpu"
