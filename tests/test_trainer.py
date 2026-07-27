import json

import numpy as np
import torch
from safetensors.torch import load_file

from coop import hubio
from coop.model import GPT, GPTConfig, canonical_state
from coop.submit import dequantize_delta
from coop.trainer import run_worker

CFG = {
    "repos": {"model": "x/model", "dataset": "x/inbox"},
    "model": {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 16,
        "vocab_size": 64,
        "dropout": 0.0,
        "bias": True,
    },
    "inner": {
        "h_steps": 2,
        "batch_size": 2,
        "lr": 1.0e-3,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "quantize": "int8",
    },
}


def test_worker_round_produces_delta_and_meta(tmp_path, monkeypatch):
    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    monkeypatch.setattr(hubio, "download_checkpoint", lambda repo: (state, {"step": 3}))
    monkeypatch.setattr(hubio, "whoami", lambda: "tester")

    bin_path = tmp_path / "shard.bin"
    np.random.default_rng(0).integers(0, 64, size=2000).astype(np.uint16).tofile(bin_path)

    delta_path, meta_path = run_worker(
        CFG, str(bin_path), out_dir=str(tmp_path / "out"), do_submit=True, dry_run=True
    )

    meta = json.loads(meta_path.read_text())
    assert meta["username"] == "tester"
    assert meta["start_step"] == 3
    assert meta["h_steps"] == 2
    assert meta["tokens"] == 2 * 2 * 16
    assert meta["quant"] == "int8"
    assert meta["wall_secs"] >= 0

    delta = dequantize_delta(load_file(str(delta_path)))
    assert set(delta.keys()) == set(state.keys())
    # training moved the weights, so the pseudo-gradient is non-zero
    assert sum(v.abs().sum().item() for v in delta.values()) > 0


def test_adaptive_h():
    from coop.trainer import adaptive_h

    inner = {"h_min": 50, "h_max": 500}
    meta = {"h_steps": 100, "wall_secs": 50.0}  # 2 inner steps/sec
    assert adaptive_h(meta, period_secs=100.0, inner=inner) == 160  # 0.8 * 100 * 2
    assert adaptive_h(meta, period_secs=5.0, inner=inner) == 50  # clamped to floor
    assert adaptive_h(meta, period_secs=10_000.0, inner=inner) == 500  # DiLoCo ceiling
