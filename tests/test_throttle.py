import subprocess
import threading
import time

from coop.throttle import StallWatchdog, Throttle, set_power_limit


def test_default_throttle_is_stock_behaviour():
    thr = Throttle()
    assert not thr.shaping
    assert thr.splits(4) == 1


def test_gentle_runs_one_sequence_per_kernel():
    thr = Throttle.gentle()
    assert thr.shaping
    assert thr.splits(4) == 4  # batch 4 -> 4 kernels of 1
    assert thr.splits(32) == 32


def test_micro_batch_divides_the_batch():
    assert Throttle(micro_batch=2).splits(8) == 4
    assert Throttle(micro_batch=3).splits(8) == 3  # uneven splits are allowed


def test_micro_batch_at_or_above_batch_size_does_not_split():
    assert Throttle(micro_batch=4).splits(4) == 1
    assert Throttle(micro_batch=99).splits(4) == 1


def test_describe_reports_only_what_is_actually_happening():
    assert Throttle.gentle().describe(4) == "micro-batch 1, 4 kernels per step"
    assert "130 W cap" in Throttle.gentle(power_limit_w=130).describe(4)
    # micro_batch 4 on a batch of 4 changes nothing; say so rather than implying a win
    assert "nothing to shape" in Throttle(micro_batch=4).describe(4)


def test_power_limit_reports_success(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "")
    )
    assert "130 W" in set_power_limit(130)


def test_power_limit_refusal_names_the_command_to_run(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "denied")
    )
    assert "sudo nvidia-smi --power-limit 130" in set_power_limit(130)


def test_power_limit_without_nvidia_smi_still_advises(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert "sudo nvidia-smi --power-limit 130" in set_power_limit(130)


def test_watchdog_fires_when_progress_stops():
    fired = threading.Event()
    with StallWatchdog(stall_secs=1, on_stall=lambda idle: fired.set()):
        assert fired.wait(5), "watchdog never noticed the stall"


def test_watchdog_stays_quiet_while_steps_land():
    fired = threading.Event()
    with StallWatchdog(stall_secs=2, on_stall=lambda idle: fired.set()) as w:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            w.beat()
            time.sleep(0.1)
    assert not fired.is_set()


def test_watchdog_disabled_by_zero():
    fired = threading.Event()
    with StallWatchdog(stall_secs=0, on_stall=lambda idle: fired.set()):
        time.sleep(0.3)
    assert not fired.is_set()
