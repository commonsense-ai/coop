"""What this machine can actually do, with no network and nothing submitted.

    uv run python -m coop.bench --steps 20

Runs the live run config's real shapes against a synthetic shard, so the number it
prints is this machine's training speed and nothing else. The phase split is the
point: it is what turned "MPS is slow" into "the gradient norm is 36% of the step".
"""

import argparse
import contextlib
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from coop import load_config, setup_logging
from coop.data import iter_batches
from coop.device import describe, pick_device, resolve, step_sync, unusable
from coop.model import GPT, GPTConfig, count_params
from coop.trainer import autocast_ctx, clip_grads, make_adamw


def synthetic_shard(path: Path, vocab: int, tokens: int = 4_000_000) -> str:
    """Random ids are the wrong text but the right work: every kernel sees real shapes."""
    np.random.default_rng(0).integers(0, vocab, size=tokens, dtype=np.uint16).tofile(path)
    return str(path)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_gb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0  # host RSS says more about the allocator than about the model


def run(cfg: dict, device: str, steps: int, warmup: int, precision: str, compile_model: bool):
    inner = cfg["inner"]
    batch, block = inner["batch_size"], cfg["model"]["block_size"]
    torch.set_float32_matmul_precision("high")
    model = GPT.from_config(GPTConfig(**cfg["model"])).to(device)
    if compile_model:
        model = torch.compile(model)
    opt = make_adamw(model.parameters(), inner)
    params = list(model.parameters())
    amp = lambda: autocast_ctx(device, precision)  # noqa: E731
    tick = step_sync(device)

    with tempfile.TemporaryDirectory() as tmp:
        shard = synthetic_shard(Path(tmp) / "bench.bin", cfg["model"]["vocab_size"])
        batches = iter_batches(shard, batch, block, seed=0)
        model.train()
        phases = dict(data=0.0, forward=0.0, backward=0.0, clip=0.0, optimizer=0.0)
        for i in range(steps + warmup):
            timed = i >= warmup
            if i == warmup:
                sync(device)
                t_all = time.perf_counter()

            def mark(name: str, t0: float) -> float:
                if timed:
                    sync(device)
                    now = time.perf_counter()
                    phases[name] += now - t0
                    return now
                return t0

            t = time.perf_counter()
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            t = mark("data", t)
            with amp():
                _, loss = model(x, y)
            t = mark("forward", t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            t = mark("backward", t)
            clip_grads(params, inner["grad_clip"], device)
            t = mark("clip", t)
            opt.step()
            tick()
            mark("optimizer", t)
        sync(device)
        wall = time.perf_counter() - t_all

    per = wall / steps
    return per, batch * block / per, peak_gb(device), phases, count_params(model)


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="measure this machine's training speed")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument("--device", default=None, help="cuda | mps | tpu | cpu (default: auto)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3, help="steps to discard before timing")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--compile", action="store_true", dest="compile_model")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the config's batch size — find what fills a big card",
    )
    a = ap.parse_args()

    device = resolve(a.device) if a.device else pick_device()
    if why := unusable(device):
        raise SystemExit(why)
    cfg = load_config(a.config)
    if a.batch_size:
        cfg["inner"]["batch_size"] = a.batch_size

    per, tps, gb, phases, n = run(cfg, device, a.steps, a.warmup, a.precision, a.compile_model)
    fp32 = isinstance(autocast_ctx(device, a.precision), contextlib.nullcontext)
    print(f"{describe(device)} · {n / 1e6:.0f}M params · {'fp32' if fp32 else 'bf16'}")
    print(f"batch {cfg['inner']['batch_size']} x {cfg['model']['block_size']} tokens")
    print(
        f"\n{per * 1e3:.0f} ms/step · {tps:,.0f} tokens/sec"
        + (f" · peak {gb:.1f} GB" if gb else "")
    )
    total = sum(phases.values()) or 1.0
    print("\nwhere the step goes (syncing to measure costs a little of it):")
    for name, secs in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {secs / a.steps * 1e3:7.1f} ms  {secs / total * 100:4.0f}%")


if __name__ == "__main__":
    main()
