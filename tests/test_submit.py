import logging

import torch

from coop.submit import dequantize_delta, quantize_delta, submission_paths, submit

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
