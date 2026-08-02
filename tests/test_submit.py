import logging

import torch

from coop.submit import (
    SCALE_SUFFIX,
    StepAccumulator,
    dequantize_delta,
    dequantize_delta_int4,
    quantize_delta,
    quantize_delta_int4,
    submission_paths,
    submit,
    submit_accumulated,
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


def test_step_accumulator_token_weighted_average_and_reset():
    d1 = {"w": torch.ones(4)}
    d2 = {"w": torch.full((4,), 4.0)}
    acc = StepAccumulator()

    merged, meta = acc.add(d1, {"start_step": 5, "tokens": 1000, "username": "a"})
    assert torch.equal(merged["w"], torch.ones(4))
    assert meta["tokens"] == 1000 and meta["rounds"] == 1

    merged, meta = acc.add(d2, {"start_step": 5, "tokens": 3000, "username": "a"})
    # (1000*1 + 3000*4) / 4000 = 3.25
    assert torch.allclose(merged["w"], torch.full((4,), 3.25))
    assert meta["tokens"] == 3000  # credit: largest single round
    assert meta["tokens_total"] == 4000 and meta["rounds"] == 2

    merged, meta = acc.add(d1, {"start_step": 6, "tokens": 500, "username": "a"})
    assert torch.equal(merged["w"], torch.ones(4))  # new step resets
    assert meta["tokens_total"] == 500 and meta["rounds"] == 1


class FakeInfo:
    pr_url = "https://huggingface.co/datasets/x/inbox/discussions/7"


def test_submit_accumulated_creates_then_refreshes_then_survives_close(monkeypatch):
    calls = []

    def fake_open(repo, ops, msg):
        calls.append(("open", msg))
        return FakeInfo()

    monkeypatch.setattr("coop.submit.hubio.open_pr", fake_open)

    def fake_update(repo, num, ops, msg):
        calls.append(("update", num))
        if len(calls) >= 3:  # third submission: the tick closed the PR meanwhile
            raise RuntimeError("pr closed")

    monkeypatch.setattr("coop.submit.hubio.update_pr", fake_update)
    cfg = {"repos": {"dataset": "x/inbox"}, "inner": {"quantize": "int8"}}
    acc = StepAccumulator()
    d = {"w": torch.ones(4)}
    meta = {"start_step": 5, "tokens": 1000, "username": "a"}

    submit_accumulated(cfg, acc, d, meta)  # round 1: opens PR #7
    assert acc.pr == 7 and calls[-1][0] == "open"
    first_paths = acc.paths

    submit_accumulated(cfg, acc, d, meta)  # round 2: refreshes PR #7, same paths
    assert calls[-1] == ("update", 7) and acc.paths == first_paths

    submit_accumulated(cfg, acc, d, meta)  # round 3: update fails -> fresh PR, fresh state
    assert calls[-2] == ("update", 7) and calls[-1][0] == "open"
    assert acc.rounds == 1 and acc.tokens_total == 1000 and acc.paths != first_paths
