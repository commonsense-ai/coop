"""No NVIDIA hardware here, so every cuda path is driven through injected torch
state. That covers the logic; it does not prove a real driver behaves this way."""

import argparse
import json
import os
from types import SimpleNamespace

import pytest
import torch

import coop.cli as cli
from coop import device as dev


@pytest.fixture(autouse=True)
def _no_cached_probe():
    dev.nvidia_present.cache_clear()
    yield
    dev.nvidia_present.cache_clear()


def fake_cuda(monkeypatch, *, available, built=True, name="NVIDIA GeForce RTX 4090", vram_gb=24):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "cuda", "12.4" if built else None, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name=name, total_memory=vram_gb * 1024**3),
        raising=False,
    )


def test_picks_cuda_over_everything_else(monkeypatch):
    fake_cuda(monkeypatch, available=True)
    assert dev.pick_device() == "cuda"


def test_falls_through_to_mps_then_cpu(monkeypatch):
    fake_cuda(monkeypatch, available=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert dev.pick_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert dev.pick_device() == "cpu"


def test_cuda_shows_the_actual_card(monkeypatch):
    fake_cuda(monkeypatch, available=True)
    assert dev.describe("cuda") == "NVIDIA GeForce RTX 4090 (24 GB)"


def test_cuda_index_is_honoured(monkeypatch):
    seen = []
    fake_cuda(monkeypatch, available=True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: seen.append(i) or SimpleNamespace(name="A100", total_memory=80 * 1024**3),
    )
    assert dev.describe("cuda:1") == "A100 (80 GB)"
    assert seen == [1]


def test_describe_survives_torch_refusing_to_name_the_card(monkeypatch):
    def boom(_):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert dev.describe("cuda") == "NVIDIA GPU"  # a missing name must not break a command


def test_plain_names_for_the_rest():
    assert dev.describe("mps") == "Apple GPU"
    assert dev.describe("cpu") == "CPU"


def test_no_gap_when_cuda_works(monkeypatch):
    fake_cuda(monkeypatch, available=True)
    monkeypatch.setattr(dev, "nvidia_present", lambda: True)
    assert dev.cuda_gap() is None


def stub_nvidia_smi(monkeypatch, present: bool, calls: list | None = None):
    def run(*a, **k):
        if calls is not None:
            calls.append(a)
        return SimpleNamespace(
            returncode=0 if present else 1,
            stdout="GPU 0: NVIDIA GeForce RTX 4090" if present else "",
        )

    monkeypatch.setattr(dev.subprocess, "run", run)


def test_no_gap_on_machines_without_nvidia(monkeypatch):
    fake_cuda(monkeypatch, available=False)
    stub_nvidia_smi(monkeypatch, present=False)
    assert dev.cuda_gap() is None


def test_gap_names_a_cpu_only_torch(monkeypatch):
    fake_cuda(monkeypatch, available=False, built=False)
    stub_nvidia_smi(monkeypatch, present=True)
    assert "without CUDA support" in dev.cuda_gap()


def test_gap_when_torch_has_cuda_but_cannot_reach_the_driver(monkeypatch):
    fake_cuda(monkeypatch, available=False, built=True)
    stub_nvidia_smi(monkeypatch, present=True)
    assert "driver" in dev.cuda_gap()


def test_detection_failing_can_never_break_the_command(monkeypatch):
    """A test that stubs subprocess.Popen for its own reasons must not blow up a
    coop command; the probe is a hint, not a dependency."""
    monkeypatch.setattr(dev.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=1))
    assert dev.nvidia_present() is False
    cli.warn_cuda_gap("cpu")  # must not raise


def test_apple_machines_never_shell_out(monkeypatch, capsys):
    calls: list = []
    stub_nvidia_smi(monkeypatch, present=True, calls=calls)
    cli.warn_cuda_gap("mps")
    assert calls == []  # no CUDA on Apple hardware: nothing to probe for
    assert capsys.readouterr().out == ""


def test_missing_nvidia_smi_is_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(dev.subprocess, "run", boom)
    assert dev.nvidia_present() is False


def offline_status(monkeypatch, tmp_path, status: dict, running: bool):
    """cmd_status with every network call stubbed out, so only the device line varies."""
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    monkeypatch.setattr(cli, "load_run_config", lambda *a, **k: {"repos": {"model": "x/y"}})
    monkeypatch.setattr(cli, "pending_rounds", lambda: 0)
    monkeypatch.setattr(cli.hubio, "whoami", lambda: "someone")

    def offline(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(cli.hubio, "download_file", offline)
    monkeypatch.setattr(cli.hubio, "list_open_prs", offline)
    monkeypatch.setattr(cli, "fetch_raw", offline)
    if running:
        (tmp_path / "worker.pid").write_text(str(os.getpid()))  # a pid that is alive
    (tmp_path / "status.json").write_text(json.dumps(status))
    cli.cmd_status(argparse.Namespace(repo="commonsense-ai/coop"))


def test_status_reports_the_device_the_worker_actually_got(monkeypatch, capsys, tmp_path):
    """The worker's own report wins: only it knows about an explicit --device."""
    monkeypatch.setattr(cli, "pick_device", lambda: "cpu")  # would disagree; must not win
    offline_status(
        monkeypatch,
        tmp_path,
        {"device": "cuda", "device_label": "NVIDIA A100 (80 GB)", "phase": "training"},
        running=True,
    )
    assert "device   NVIDIA A100 (80 GB)" in capsys.readouterr().out


def test_status_without_a_worker_says_what_would_be_used(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "pick_device", lambda: "cuda")
    monkeypatch.setattr(cli, "describe", lambda d: "NVIDIA GeForce RTX 4090 (24 GB)")
    offline_status(monkeypatch, tmp_path, {}, running=False)
    out = capsys.readouterr().out
    assert "device   NVIDIA GeForce RTX 4090 (24 GB) — what `coop start` would use" in out


def test_cpu_user_with_an_nvidia_card_is_told(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cuda_gap", lambda: "this torch was built without CUDA support")
    cli.warn_cuda_gap("cpu")
    out = capsys.readouterr().out
    assert "NVIDIA GPU but coop is on your CPU" in out
    assert "--torch-backend auto" in out


def test_no_warning_when_already_on_cuda(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cuda_gap", lambda: "should never be consulted")
    cli.warn_cuda_gap("cuda:0")
    assert capsys.readouterr().out == ""
