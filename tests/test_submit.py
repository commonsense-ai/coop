import logging

import torch

from coop.submit import (
    SCALE_SUFFIX,
    dequantize_delta,
    dequantize_delta_int4,
    quantize_delta,
    quantize_delta_int4,
    submission_paths,
    submit,
)

CFG = {"repos": {"dataset": "x/inbox"}}


def test_quantize_roundtrip():
    g = torch.Generator().manual_seed(0)
    delta = {"a": torch.randn(50, generator=g), "b": torch.zeros(3)}
    q = quantize_delta(delta)
    assert q["a"].dtype == torch.int8
    deq = dequantize_delta(q)
    assert set(deq) == set(delta)
    scale = q["a::scale"].item()
    assert (deq["a"] - delta["a"]).abs().max() <= scale  # error is at most one bucket
    assert torch.equal(deq["b"], delta["b"])


def test_submission_paths():
    st, js = submission_paths({"username": "alice", "start_step": 7})
    assert st.startswith("submissions/step_7/alice_") and st.endswith(".safetensors")
    assert js == st.replace(".safetensors", ".json")


def test_dry_run_logs_paths_without_network(caplog):
    meta = {"username": "alice", "start_step": 7}
    with caplog.at_level(logging.INFO, logger="coop.submit"):
        result = submit(CFG, "delta.safetensors", meta, dry_run=True)
    assert result is None
    assert "submissions/step_7/alice_" in caplog.text
    assert "x/inbox" in caplog.text


def test_int4_roundtrip_and_size():
    torch.manual_seed(0)
    delta = {"w": torch.randn(33, 7), "b": torch.randn(5)}  # odd numels exercise padding
    q = quantize_delta_int4(delta)
    assert q["w"].dtype == torch.uint8 and q["w"].numel() == (33 * 7 + 1) // 2
    out = dequantize_delta_int4(q)
    for k in delta:
        scale = q[k + SCALE_SUFFIX].item()
        assert out[k].numel() == delta[k].numel()
        # symmetric int4: max error is half a quantization step
        assert torch.allclose(out[k], delta[k].reshape(-1), atol=scale / 2 + 1e-6)
