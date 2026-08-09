"""No NVIDIA hardware here, so every cuda path is driven through injected torch
state. That covers the logic; it does not prove a real driver behaves this way."""

import argparse
import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import coop.cli as cli
from coop import device as dev

PROBES = (dev.nvidia_present, dev.arch_gap, dev._xla)


@pytest.fixture(autouse=True)
def _no_cached_probe():
    for probe in PROBES:
        probe.cache_clear()
    yield
    for probe in PROBES:
        probe.cache_clear()


def fake_torch_xla(monkeypatch, *, kind="TPU", sync=True, broken=False):
    """Stand in for the plugin. No TPU on any machine that runs these tests, so what
    is under test is the import gate and the wiring — never the runtime itself."""
    calls: list[str] = []
    xla = types.ModuleType("torch_xla")
    runtime = types.ModuleType("torch_xla.runtime")
    core = types.ModuleType("torch_xla.core")
    xla_model = types.ModuleType("torch_xla.core.xla_model")

    def device_type():
        if broken:
            raise RuntimeError("libtpu is not where it should be")
        return kind

    runtime.device_type = device_type
    xla_model.mark_step = lambda: calls.append("mark_step")
    if sync:
        xla.sync = lambda: calls.append("sync")
    xla.runtime, xla.core, core.xla_model = runtime, core, xla_model
    for name, mod in [
        ("torch_xla", xla),
        ("torch_xla.runtime", runtime),
        ("torch_xla.core", core),
        ("torch_xla.core.xla_model", xla_model),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda n, *a, **k: object() if n == "torch_xla" else real(n, *a, **k),
    )
    return calls


ADA = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]  # a pre-Blackwell wheel


def fake_cuda(
    monkeypatch,
    *,
    available,
    built=True,
    name="NVIDIA GeForce RTX 4090",
    vram_gb=24,
    cc=(8, 9),
    archs=ADA,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "cuda", "12.4" if built else None, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(name=name, total_memory=vram_gb * 1024**3),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: cc, raising=False)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: archs, raising=False)


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
    monkeypatch.setattr(dev.shutil, "which", lambda n: "/usr/bin/nvidia-smi" if present else None)

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
    assert "without CUDA support" in dev.cuda_gap().reason


def test_gap_when_torch_has_cuda_but_cannot_reach_the_driver(monkeypatch):
    fake_cuda(monkeypatch, available=False, built=True)
    stub_nvidia_smi(monkeypatch, present=True)
    assert "driver" in dev.cuda_gap().reason


def blackwell(monkeypatch, **kw):
    """A card newer than the wheel: torch reports it, then no kernel will launch on it."""
    fake_cuda(
        monkeypatch,
        available=True,
        name="NVIDIA GeForce RTX 5090",
        vram_gb=32,
        cc=(12, 0),
        **kw,
    )


def test_a_card_the_wheel_has_no_kernels_for_is_a_gap(monkeypatch):
    blackwell(monkeypatch)
    gap = dev.arch_gap()
    assert "sm_120" in gap.reason and "sm_50-sm_90" in gap.reason
    assert "RTX 5090" in gap.reason  # the donor has to recognise which machine this is
    assert gap.fix == dev.NEWER_FIX


def test_that_gap_is_what_cpu_users_are_told(monkeypatch, capsys):
    """`torch.cuda.is_available()` is True here, so without this the volunteer sees a
    bare CUDA error every round and nothing that names the cause."""
    blackwell(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)  # an NVIDIA box
    assert dev.pick_device() == "cpu"
    cli.warn_cuda_gap("cpu")
    out = capsys.readouterr().out
    assert "sm_120" in out and "--torch-backend auto" in out


def test_a_cubin_covers_higher_minor_versions_of_its_own_major(monkeypatch):
    """An sm_86 card runs sm_80 kernels. Flagging that would strand working GPUs."""
    fake_cuda(monkeypatch, available=True, cc=(8, 6), archs=["sm_70", "sm_80"])
    assert dev.arch_gap() is None
    assert dev.pick_device() == "cuda"


def test_ptx_in_the_wheel_can_jit_onto_a_newer_card(monkeypatch):
    blackwell(monkeypatch, archs=[*ADA, "compute_90"])
    assert dev.arch_gap() is None


def test_a_card_older_than_the_wheel_gets_no_upgrade_advice(monkeypatch):
    """Kepler against a modern wheel: newer torch is the opposite of the fix."""
    fake_cuda(monkeypatch, available=True, name="Tesla K80", cc=(3, 7), archs=ADA)
    assert dev.arch_gap().fix == dev.OLD_CARD


def test_arch_gap_gives_up_quietly_when_torch_will_not_say(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom, raising=False)
    assert dev.arch_gap() is None  # unknown must not read as broken
    assert dev.pick_device() == "cuda"


def test_the_real_runtime_error_is_recognised():
    e = RuntimeError(
        "CUDA error: no kernel image is available for execution on the device\n"
        "Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/"
    )
    assert dev.kernel_missing(e)
    assert not dev.kernel_missing(RuntimeError("CUDA out of memory"))


def test_fallback_says_why_and_keeps_the_machine_working(monkeypatch, caplog):
    blackwell(monkeypatch)
    with caplog.at_level("WARNING"):
        assert dev.cpu_fallback() == "cpu"
    assert "sm_120" in caplog.text and dev.NEWER_FIX in caplog.text


def no_other_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_a_tpu_is_trained_on_when_one_is_attached(monkeypatch):
    no_other_accelerator(monkeypatch)
    fake_torch_xla(monkeypatch)
    assert dev.tpu_present()
    assert dev.pick_device() == "xla"


def test_xla_pointed_at_a_cpu_or_a_gpu_is_not_a_tpu(monkeypatch):
    """torch_xla also targets CPU and CUDA, and both are slower through XLA than the
    native paths coop already has. Only a TPU is a reason to route through it."""
    no_other_accelerator(monkeypatch)
    fake_torch_xla(monkeypatch, kind="CPU")
    assert not dev.tpu_present()
    assert dev.pick_device() == "cpu"


def test_a_machine_without_the_plugin_never_imports_it(monkeypatch):
    """The gate is a path lookup: importing torch_xla starts the XLA runtime, which
    no `coop status` on an ordinary laptop should ever pay for."""
    no_other_accelerator(monkeypatch)
    monkeypatch.delitem(sys.modules, "torch_xla", raising=False)
    assert not dev.tpu_present()
    assert "torch_xla" not in sys.modules


def test_a_half_installed_plugin_is_not_an_error(monkeypatch):
    no_other_accelerator(monkeypatch)
    fake_torch_xla(monkeypatch, broken=True)
    assert not dev.tpu_present()
    assert dev.pick_device() == "cpu"


def test_each_step_closes_the_xla_graph(monkeypatch):
    """Without this the inner loop piles every step into one graph, compiled only
    when something finally reads a tensor."""
    calls = fake_torch_xla(monkeypatch)
    sync = dev.step_sync("xla")
    sync()
    sync()
    assert calls == ["sync", "sync"]


def test_older_torch_xla_spells_it_mark_step(monkeypatch):
    calls = fake_torch_xla(monkeypatch, sync=False)
    dev.step_sync("xla")()
    assert calls == ["mark_step"]


def test_the_sync_is_free_on_every_other_device(monkeypatch):
    calls = fake_torch_xla(monkeypatch)
    dev.step_sync("cuda")()
    dev.step_sync("cpu")()
    assert calls == []


def test_a_tpu_round_labels_itself_on_the_board():
    """The board already had a word for this before TPUs could train; a round now
    reaching it must use that word rather than inventing a second spelling."""
    assert dev.kind("xla") == dev.kind("xla:0") == "google-tpu"


def test_volunteers_type_tpu_and_torch_hears_xla():
    assert dev.resolve("tpu") == "xla"
    assert dev.resolve("cuda") == "cuda" and dev.resolve("cpu") == "cpu"


def test_asking_for_a_tpu_without_one_says_so_before_training(monkeypatch):
    no_other_accelerator(monkeypatch)
    why = dev.unusable("xla")
    assert "torch_xla" in why
    assert dev.unusable("cpu") is None


def test_no_complaint_when_the_tpu_is_really_there(monkeypatch):
    fake_torch_xla(monkeypatch)
    assert dev.unusable("xla") is None


def test_generation_stays_off_the_tpu(monkeypatch):
    """XLA recompiles per sequence length, so playing on a TPU is the slow answer."""
    no_other_accelerator(monkeypatch)
    fake_torch_xla(monkeypatch)
    assert dev.pick_device() == "xla"  # training does want it
    assert dev.inference_device() == "cpu"
    assert dev.inference_device("tpu") == "cpu"  # even when asked for by name
    assert dev.inference_device("cuda") == "cuda"


def test_a_tpu_is_named_by_its_generation(monkeypatch):
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-1")
    assert dev.describe("xla") == "Google TPU (v5litepod-1)"
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE")
    assert dev.describe("xla:0") == "Google TPU"


def test_no_nvidia_smi_means_no_subprocess_at_all(monkeypatch):
    """Most machines have no driver tooling. Spawning there is pure cost, and it
    reaches into any subprocess mock the caller happens to have installed — which
    is exactly how this probe broke cmd_start's test on Linux."""
    monkeypatch.setattr(dev.shutil, "which", lambda name: None)
    spawned: list = []
    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: spawned.append(a))
    assert dev.nvidia_present() is False
    assert spawned == []


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
    gap = dev.Gap("this torch was built without CUDA support", dev.CUDA_FIX)
    monkeypatch.setattr(cli, "cuda_gap", lambda: gap)
    cli.warn_cuda_gap("cpu")
    out = capsys.readouterr().out
    assert "NVIDIA GPU but coop is on your CPU" in out
    assert "--torch-backend auto" in out


def test_no_warning_when_already_on_cuda(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cuda_gap", lambda: dev.Gap("never consulted", None))
    cli.warn_cuda_gap("cuda:0")
    assert capsys.readouterr().out == ""


def test_kind_names_the_silicon_not_just_gpu_or_cpu():
    assert dev.kind("cuda") == dev.kind("cuda:1") == "nvidia-gpu"
    assert dev.kind("mps") == "apple-gpu"
    assert dev.kind("xla") == dev.kind("xla:0") == "google-tpu"  # ready for TPU rounds
    assert dev.kind("cpu") == "cpu"
    assert dev.kind("something-new") == "cpu"  # a board cell is never blank


def test_kind_is_safe_to_put_in_a_board_cell():
    # 0.3.0 and earlier match that cell with `\S+`; a space there breaks their totals
    assert not any(" " in dev.kind(d) for d in ("cuda", "mps", "xla", "cpu"))
