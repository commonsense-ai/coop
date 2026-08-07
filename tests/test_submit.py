import json
import logging

import pytest
import torch

from coop.submit import (
    PENDING,
    SCALE_SUFFIX,
    StepAccumulator,
    dequantize_delta,
    dequantize_delta_int4,
    drain,
    quantize_delta,
    quantize_delta_int4,
    submission_paths,
    submit,
    submit_accumulated,
)

CFG = {"repos": {"dataset": "x/inbox"}}
QCFG = {"repos": {"dataset": "x/inbox"}, "inner": {"quantize": "int8"}}


def _parked(tmp_path):
    return sorted((tmp_path / PENDING).glob("*.json"))


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


def _offline(*a, **kw):
    raise OSError("connection reset")


def test_failed_upload_parks_the_round_and_drain_resends_it(tmp_path, monkeypatch):
    delta = tmp_path / "delta.safetensors"
    delta.write_bytes(b"already-quantized")
    meta = {"username": "a", "start_step": 5, "tokens": 100}

    monkeypatch.setattr("coop.submit.hubio.open_pr", _offline)
    with pytest.raises(OSError):
        submit(QCFG, str(delta), meta, out_dir=tmp_path)
    assert len(_parked(tmp_path)) == 1

    sent = []
    monkeypatch.setattr(
        "coop.submit.hubio.open_pr", lambda repo, ops, msg: sent.append(ops) or FakeInfo()
    )
    assert drain(QCFG, tmp_path) == 1
    assert _parked(tmp_path) == []  # sent and cleaned up
    # the parked bytes are re-sent verbatim, never re-quantized
    assert sent[0][0].path_or_fileobj == b"already-quantized"
    assert json.loads(sent[0][1].path_or_fileobj)["username"] == "a"


def test_drain_keeps_the_queue_when_still_offline(tmp_path, monkeypatch):
    delta = tmp_path / "delta.safetensors"
    delta.write_bytes(b"w")
    monkeypatch.setattr("coop.submit.hubio.open_pr", _offline)
    for step in (5, 6):
        with pytest.raises(OSError):
            submit(
                QCFG,
                str(delta),
                {"username": "a", "start_step": step, "tokens": 100},
                out_dir=tmp_path,
            )
    assert len(_parked(tmp_path)) == 2  # separate rounds, separate payloads
    assert drain(QCFG, tmp_path) == 0
    assert len(_parked(tmp_path)) == 2  # nothing dropped on the floor


def test_drain_drops_an_unreadable_payload_instead_of_wedging(tmp_path, monkeypatch):
    d = tmp_path / PENDING
    d.mkdir()
    (d / "step5_deadbeef.json").write_text("{ truncated")
    (d / "step5_deadbeef.safetensors").write_bytes(b"w")
    (d / "step6_cafe0000.json").write_text('{"username": "a", "start_step": 6}')
    (d / "step6_cafe0000.safetensors").write_bytes(b"w")

    monkeypatch.setattr("coop.submit.hubio.open_pr", lambda *a: FakeInfo())
    assert drain(QCFG, tmp_path) == 1  # the good one still goes out
    assert _parked(tmp_path) == []


def test_accumulator_parks_supersedes_then_clears_on_a_successful_upload(tmp_path, monkeypatch):
    acc = StepAccumulator()
    d = {"w": torch.ones(4)}
    meta = {"start_step": 5, "tokens": 1000, "username": "a"}
    monkeypatch.setattr("coop.submit.hubio.open_pr", _offline)

    with pytest.raises(OSError):
        submit_accumulated(QCFG, acc, d, meta, out_dir=tmp_path)
    first = acc.pending
    assert first is not None and len(_parked(tmp_path)) == 1

    with pytest.raises(OSError):
        submit_accumulated(QCFG, acc, d, meta, out_dir=tmp_path)
    # round 2 folds round 1 in, so its payload replaces the parked copy instead of adding
    assert acc.pending != first and len(_parked(tmp_path)) == 1

    monkeypatch.setattr("coop.submit.hubio.open_pr", lambda *a: FakeInfo())
    submit_accumulated(QCFG, acc, d, meta, out_dir=tmp_path)
    assert acc.pending is None and _parked(tmp_path) == []  # the PR carries both rounds


def test_accumulator_keeps_a_parked_payload_from_an_older_step(tmp_path, monkeypatch):
    """A new outer step resets the average, so the parked rounds it no longer carries
    must survive on disk for drain() instead of being cleared by the next success."""
    acc = StepAccumulator()
    d = {"w": torch.ones(4)}
    monkeypatch.setattr("coop.submit.hubio.open_pr", _offline)
    with pytest.raises(OSError):
        submit_accumulated(
            QCFG, acc, d, {"start_step": 5, "tokens": 1000, "username": "a"}, tmp_path
        )

    monkeypatch.setattr("coop.submit.hubio.open_pr", lambda *a: FakeInfo())
    submit_accumulated(QCFG, acc, d, {"start_step": 6, "tokens": 1000, "username": "a"}, tmp_path)
    assert acc.pending is None
    assert len(_parked(tmp_path)) == 1  # step 5's work still queued, not discarded
