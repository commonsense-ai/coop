"""Which accelerator this machine trains on, and how to say it to a volunteer.

Detection lives in one place so the worker, the trainer, and the CLI can never
disagree about what a donor is running on.
"""

import functools
import subprocess

import torch

PLAIN = {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}
CUDA_FIX = "install a CUDA torch: `uv pip install --torch-backend auto torch`"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    try:
        p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return p.returncode == 0 and "GPU" in p.stdout
    except Exception:
        return False


def cuda_gap() -> str | None:
    """Why an NVIDIA box is about to use its CPU instead. The costliest thing a
    donor can get silently wrong: PyPI's default torch is CPU-only on Windows, so
    plain resolution strands GPU owners without ever saying so."""
    if torch.cuda.is_available() or not nvidia_present():
        return None
    if getattr(torch.version, "cuda", None) is None:
        return "this torch was built without CUDA support"
    return "torch can't reach the driver"
