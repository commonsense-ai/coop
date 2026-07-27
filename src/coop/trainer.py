"""Worker: DiLoCo inner loop -> pseudo-gradient."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from coop import hubio, load_config, submit
from coop.data import iter_batches
from coop.model import GPT, GPTConfig, load_canonical_state

log = logging.getLogger(__name__)


def run_worker(
    cfg: dict,
    data_bin: str,
    out_dir: str = "out",
    device: str = "cpu",
    seed: int = 0,
    do_submit: bool = True,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    inner = cfg["inner"]
    state, ckpt_meta = hubio.download_checkpoint(cfg["repos"]["model"])
    start_step = ckpt_meta["step"]
    log.info("starting from outer step %d", start_step)

    model = GPT.from_config(GPTConfig(**cfg["model"]))
    load_canonical_state(model, state)
    theta_outer = {k: v.detach().clone() for k, v in model.named_parameters()}
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=inner["lr"],
        betas=tuple(inner["betas"]),
        weight_decay=inner["weight_decay"],
    )
    batches = iter_batches(data_bin, inner["batch_size"], cfg["model"]["block_size"], seed=seed)
    h = inner["h_steps"]
    t0 = time.time()
    model.train()
    for i in range(h):
        x, y = next(batches)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), inner["grad_clip"])
        opt.step()
        if i % 10 == 0 or i == h - 1:
            log.info("inner step %d/%d loss %.4f", i + 1, h, loss.item())
    wall = time.time() - t0

    delta = {k: (theta_outer[k] - v.detach().cpu()).float() for k, v in model.named_parameters()}
    meta = {
        "username": hubio.whoami(),
        "start_step": start_step,
        "h_steps": h,
        "wall_secs": round(wall, 2),
        "tokens": h * inner["batch_size"] * cfg["model"]["block_size"],
        "tier": "cpu" if device == "cpu" else "gpu",
        "quant": "none",
    }
    if inner.get("quantize") == "int8":
        delta = submit.quantize_delta(delta)
        meta["quant"] = "int8"

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
        submit.submit(cfg, str(delta_path), meta, dry_run=dry_run)
    return delta_path, meta_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="run one DiLoCo worker round")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument("--data", required=True, help="token .bin (see python -m coop.data)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="package but only log the submission")
    ap.add_argument("--loop", action="store_true", help="run rounds until interrupted")
    ap.add_argument("--pause", type=int, default=60, help="seconds between rounds with --loop")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rnd = 0
    while True:
        try:
            run_worker(
                cfg,
                a.data,
                out_dir=a.out,
                device=a.device,
                seed=a.seed + rnd,  # fresh batch order every round
                do_submit=not a.no_submit,
                dry_run=a.dry_run,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # transient network/HF hiccups must not kill an unattended loop
            if not a.loop:
                raise
            log.warning("round failed (%s); retrying after pause", e)
        rnd += 1
        if not a.loop:
            break
        time.sleep(a.pause)


if __name__ == "__main__":
    main()
