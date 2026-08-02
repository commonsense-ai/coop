import contextlib
import json

import numpy as np
import torch
from safetensors.torch import load_file

from coop import hubio
from coop.model import GPT, GPTConfig, canonical_state
from coop.submit import dequantize_delta
from coop.trainer import autocast_ctx, make_adamw, pick_device, run_worker

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


class StopAfter:
    """is_set() flips true after n checks — simulates SIGTERM arriving mid-round."""

    def __init__(self, n):
        self.n = n

    def is_set(self):
        self.n -= 1
        return self.n < 0


def _shard(tmp_path):
    bin_path = tmp_path / "shard.bin"
    np.random.default_rng(0).integers(0, 64, size=2000).astype(np.uint16).tofile(bin_path)
    return str(bin_path)


def test_worker_flushes_partial_round_on_stop(tmp_path, monkeypatch):
    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    monkeypatch.setattr(hubio, "download_checkpoint", lambda repo: (state, {"step": 3}))
    monkeypatch.setattr(hubio, "whoami", lambda: "tester")

    delta_path, meta_path = run_worker(
        CFG,
        _shard(tmp_path),
        out_dir=str(tmp_path / "out"),
        do_submit=False,
        h_override=50,
        stop=StopAfter(2),
    )
    meta = json.loads(meta_path.read_text())
    assert meta["h_steps"] == 2  # only the steps actually finished
    assert meta["tokens"] == 2 * CFG["inner"]["batch_size"] * CFG["model"]["block_size"]
    assert delta_path.exists()


def test_worker_stopped_before_training_submits_nothing(tmp_path, monkeypatch):
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    monkeypatch.setattr(hubio, "download_checkpoint", lambda repo: (state, {"step": 3}))

    out = run_worker(
        CFG, _shard(tmp_path), out_dir=str(tmp_path / "out"), do_submit=False, stop=StopAfter(0)
    )
    assert out == (None, None)


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


def test_device_helpers_fall_back_safely():
    assert pick_device() in ("cuda", "mps", "cpu")
    # cpu always trains fp32 no matter what precision asks for
    assert isinstance(autocast_ctx("cpu", "auto"), contextlib.nullcontext)
    assert isinstance(autocast_ctx("cpu", "bf16"), contextlib.nullcontext)
    opt = make_adamw(GPT.from_config(GPTConfig(**CFG["model"])).parameters(), CFG["inner"])
    assert isinstance(opt, torch.optim.AdamW)


def test_delta_keys_survive_compile_wrapping(tmp_path, monkeypatch):
    """torch.compile prefixes named_parameters with _orig_mod. — the pseudo-gradient
    must keep canonical keys or the aggregator rejects the submission."""

    class Wrapped(torch.nn.Module):  # named_parameters mimics torch's OptimizedModule
        def __init__(self, mod):
            super().__init__()
            self._orig_mod = mod

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    monkeypatch.setattr(hubio, "download_checkpoint", lambda repo: (state, {"step": 3}))
    monkeypatch.setattr(hubio, "whoami", lambda: "tester")
    monkeypatch.setattr(torch, "compile", lambda m, **kw: Wrapped(m))

    delta_path, _ = run_worker(
        CFG, _shard(tmp_path), out_dir=str(tmp_path / "out"), do_submit=False, compile_model=True
    )
    delta = dequantize_delta(load_file(str(delta_path)))
    assert set(delta.keys()) == set(state.keys())
