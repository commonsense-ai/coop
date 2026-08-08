import contextlib
import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from coop import hubio
from coop.model import GPT, GPTConfig, canonical_state
from coop.submit import dequantize_delta
from coop.trainer import autocast_ctx, clip_grads, make_adamw, pick_device, run_worker

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


def grads_like(seed: int, scale: float) -> list[torch.Tensor]:
    """Stand-ins for a real round's gradients: mixed shapes, one much larger."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 32), (32,), (8, 4, 4), (1,), (256, 16)]
    ps = [torch.zeros(s) for s in shapes]
    for p in ps:
        p.grad = torch.randn(p.shape, generator=g) * scale
    return ps


@pytest.mark.parametrize("scale", [10.0, 1e-4])  # norm well over the limit, and well under
def test_clip_matches_torchs_own(scale):
    """The fast path has to be the same clip, not merely a similar one: it decides what
    every submitted pseudo-gradient contains."""
    mine, theirs = grads_like(0, scale), grads_like(0, scale)
    clip_grads(mine, 1.0, "cpu")
    torch.nn.utils.clip_grad_norm_(theirs, 1.0)
    for a, b in zip(mine, theirs):
        assert torch.allclose(a.grad, b.grad, rtol=1e-6, atol=1e-8)


def test_clip_leaves_small_gradients_alone():
    ps = grads_like(1, 1e-4)
    before = [p.grad.clone() for p in ps]
    clip_grads(ps, 1.0, "cpu")
    for p, b in zip(ps, before):
        assert torch.equal(p.grad, b)  # under the limit the scale is exactly 1.0


def test_clip_on_cuda_keeps_torchs_fused_path(monkeypatch):
    """No NVIDIA hardware here to prove a hand-rolled reduction faster or safe, and
    torch's own is one fused multi-tensor kernel there — so cuda must still reach it."""
    called = []
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda p, n, **kw: called.append(n) or torch.tensor(0.0)
    )
    clip_grads(grads_like(2, 10.0), 1.0, "cuda:0")
    assert called == [1.0]


def test_clip_survives_a_parameter_that_never_got_a_gradient():
    ps = grads_like(3, 10.0)
    ps.append(torch.zeros(4))  # frozen or unused: .grad is None
    clip_grads(ps, 1.0, "cpu")
    assert ps[-1].grad is None


def test_clip_with_nothing_to_clip_is_not_an_error():
    clip_grads([torch.zeros(2)], 1.0, "cpu")
