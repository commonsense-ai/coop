"""Worker: DiLoCo inner loop -> pseudo-gradient."""

import argparse
import contextlib
import dataclasses
import functools
import json
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from coop import hubio, load_config, setup_logging, submit
from coop.data import iter_batches
from coop.join import pick_device
from coop.model import GPT, GPTConfig, load_canonical_state
from coop.throttle import StallWatchdog, Throttle

log = logging.getLogger(__name__)


@functools.cache
def _bf16_ok(device: str) -> bool:
    if device.startswith("cuda"):
        return torch.cuda.is_bf16_supported()
    if device == "mps":
        # bf16-on-MPS depends on torch build + macOS version; one probe matmul
        # beats version sniffing and never poisons a real round
        try:
            with torch.autocast("mps", dtype=torch.bfloat16):
                a = torch.ones(2, 2, device="mps")
                (a @ a).sum().item()
            return True
        except (RuntimeError, TypeError):
            return False
    return False


def autocast_ctx(device: str, precision: str = "auto"):
    """bf16 autocast where supported: activations halve, params stay fp32, so the
    pseudo-gradient (theta_outer - theta_local) keeps its semantics. cpu is
    excluded — bf16 autocast is a slowdown on most consumer CPUs."""
    if precision == "fp32" or device == "cpu":
        return contextlib.nullcontext()
    if precision == "bf16" or _bf16_ok(device):
        return torch.autocast(device.split(":")[0], dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_adamw(params, inner: dict):
    kw = dict(lr=inner["lr"], betas=tuple(inner["betas"]), weight_decay=inner["weight_decay"])
    try:
        # single-kernel update; support varies by device/torch build, so probe
        return torch.optim.AdamW(params, fused=True, **kw)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(params, **kw)


def run_worker(
    cfg: dict,
    data_bin: str,
    out_dir: str = "out",
    device: str = "cpu",
    seed: int = 0,
    do_submit: bool = True,
    dry_run: bool = False,
    h_override: int | None = None,
    status=None,
    stop=None,
    username: str | None = None,
    accumulator=None,
    precision: str = "auto",
    compile_model: bool = False,
    throttle: Throttle | None = None,
) -> tuple[Path | None, Path | None]:
    inner = cfg["inner"]
    thr = throttle or Throttle()
    # TF32 for any matmul that stays fp32; a no-op off cuda
    torch.set_float32_matmul_precision("high")
    if status:
        status.update(phase="downloading checkpoint")
    state, ckpt_meta = hubio.download_checkpoint(cfg["repos"]["model"])
    start_step = ckpt_meta["step"]
    log.info("starting from outer step %d", start_step)

    model = GPT.from_config(GPTConfig(**cfg["model"]))
    load_canonical_state(model, state)
    theta_outer = {k: v.detach().clone() for k, v in model.named_parameters()}
    model = model.to(device)
    raw = model  # compile prefixes parameter names with _orig_mod.; delta keys must not change
    if compile_model:
        model = torch.compile(model)

    opt = make_adamw(model.parameters(), inner)
    amp = functools.partial(autocast_ctx, device, precision)
    log.info(
        "device %s, %s%s",
        device,
        "fp32" if isinstance(amp(), contextlib.nullcontext) else "bf16",
        ", compiled" if compile_model else "",
    )
    batch_size = inner["batch_size"]
    batches = iter_batches(data_bin, batch_size, cfg["model"]["block_size"], seed=seed)
    pin = device.startswith("cuda")  # non_blocking only overlaps from pinned memory
    h = h_override or inner["h_steps"]
    if thr.shaping:
        log.info("throttled: %s", thr.describe(batch_size))
    splits = thr.splits(batch_size)
    t0 = time.time()
    model.train()
    steps_done = 0
    with StallWatchdog(thr.stall_secs) as watchdog:
        for i in range(h):
            if stop is not None and stop.is_set():
                log.info("stop requested — packaging the %d steps finished so far", steps_done)
                break
            x, y = next(batches)
            opt.zero_grad(set_to_none=True)
            loss_sum = None
            for mx, my in zip(x.chunk(splits), y.chunk(splits)):
                if pin:
                    mx, my = mx.pin_memory(), my.pin_memory()
                mx = mx.to(device, non_blocking=True)
                my = my.to(device, non_blocking=True)
                with amp():
                    _, loss = model(mx, my)
                # weight by real chunk size: uneven splits must still average to the
                # full-batch loss, or the pseudo-gradient stops matching everyone else's
                share = mx.shape[0] / x.shape[0]
                (loss * share).backward()
                part = loss.detach() * share
                loss_sum = part if loss_sum is None else loss_sum + part
            torch.nn.utils.clip_grad_norm_(model.parameters(), inner["grad_clip"])
            opt.step()
            steps_done = i + 1
            watchdog.beat()
            if i % 10 == 0 or i == h - 1:
                step_loss = loss_sum.item()  # one sync per logged step, not per micro-batch
                log.info("inner step %d/%d loss %.4f", i + 1, h, step_loss)
                if status:
                    status.update(
                        phase="training",
                        start_step=start_step,
                        inner_step=steps_done,
                        h_steps=h,
                        loss=round(step_loss, 4),
                        steps_per_sec=round(steps_done / max(time.time() - t0, 1e-6), 2),
                    )
    wall = time.time() - t0
    if steps_done == 0:
        return None, None  # stopped before any training: nothing worth submitting

    delta = {k: (theta_outer[k] - v.detach().cpu()).float() for k, v in raw.named_parameters()}
    raw_delta = delta  # accumulation averages float deltas; quantize only per upload
    meta = {
        # a per-round whoami() once flaked into "anonymous" mid-run; resolve identity
        # once at startup (join does) and thread it through instead
        "username": username or hubio.whoami(),
        "start_step": start_step,
        "h_steps": steps_done,
        "wall_secs": round(wall, 2),
        "tokens": steps_done * batch_size * cfg["model"]["block_size"],
        "tier": "cpu" if device == "cpu" else "gpu",
        "quant": "none",
    }
    quant = inner.get("quantize")
    if quant == "int8":
        delta = submit.quantize_delta(delta)
        meta["quant"] = "int8"
    elif quant == "int4":
        delta = submit.quantize_delta_int4(delta)
        meta["quant"] = "int4"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    delta_path = out / f"delta_step{start_step}.safetensors"
    meta_path = out / f"delta_step{start_step}.json"
    save_file(delta, str(delta_path))
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(
        "pseudo-gradient written: %s (%.1f MB, %d tokens in %.0fs)",
        delta_path,
        delta_path.stat().st_size / 1e6,
        meta["tokens"],
        wall,
    )

    if do_submit:
        if status:
            status.update(phase="submitting")
        if accumulator is not None and not dry_run:
            info = submit.submit_accumulated(cfg, accumulator, raw_delta, meta, out_dir=out)
        else:
            info = submit.submit(cfg, str(delta_path), meta, dry_run=dry_run, out_dir=out)
        url = getattr(info, "pr_url", None)
        if status and url:
            status.update(last_pr=str(url))
    return delta_path, meta_path


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="run one DiLoCo worker round")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument("--data", required=True, help="token .bin (see python -m coop.data)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="torch.compile the model (first step is slow; pays off on long cuda rounds)",
    )
    ap.add_argument(
        "--gentle",
        action="store_true",
        help="smaller kernels per step: lowers peak GPU draw on marginal power delivery",
    )
    ap.add_argument(
        "--power-limit", type=int, default=None, metavar="W", help="cap the GPU via nvidia-smi"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="package but only log the submission")
    ap.add_argument("--loop", action="store_true", help="run rounds until interrupted")
    ap.add_argument("--pause", type=int, default=60, help="retry delay after a failed round")
    a = ap.parse_args()
    cfg = load_config(a.config)
    thr = Throttle.gentle() if a.gentle else Throttle()
    thr = dataclasses.replace(thr, power_limit_w=a.power_limit)
    thr.apply(a.device)
    acc = submit.StepAccumulator() if cfg["inner"].get("accumulate_rounds") else None
    rnd, h_next = 0, None
    while True:
        try:
            submit.drain(cfg, a.out, skip=acc.pending if acc else None)
            run_worker(
                cfg,
                a.data,
                out_dir=a.out,
                device=a.device,
                seed=a.seed + rnd,  # fresh batch order every round
                do_submit=not a.no_submit,
                dry_run=a.dry_run,
                h_override=h_next,
                accumulator=acc,
                precision=a.precision,
                compile_model=a.compile_model,
                throttle=thr,
            )
            # Rounds run back-to-back: each re-resolves the head checkpoint, and
            # same-user same-step submissions token-weight merge into one vote, so
            # only idle waiting wastes work. Full-depth follow-ups cut PR overhead.
            h_next = cfg["inner"].get("h_max", 500) if a.loop else None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # transient network/HF hiccups must not kill an unattended loop
            if not a.loop:
                raise
            log.warning("round failed (%s); retrying after pause", e)
            time.sleep(a.pause)
        rnd += 1
        if not a.loop:
            break


if __name__ == "__main__":
    main()
