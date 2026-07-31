import argparse
import os

import pytest

import coop.cli as cli

CFG = {"repos": {"model": "commonsense-ai/tinystories-15m"}}


def test_model_from_words_strips_the_training_noun():
    assert cli.model_from_words([]) is None
    assert cli.model_from_words(["training"]) is None
    assert cli.model_from_words(["training", "tinystories"]) == "tinystories"
    assert cli.model_from_words(["tinystories-15m"]) == "tinystories-15m"
    with pytest.raises(SystemExit):
        cli.model_from_words(["tinystories", "gpt5"])


def test_resolve_model_accepts_aliases():
    for name in (None, "tinystories", "tinystories-15m", "commonsense-ai/tinystories-15m"):
        assert cli.resolve_model(name, CFG) == "commonsense-ai/tinystories-15m"


def test_resolve_model_rejects_unknown():
    with pytest.raises(SystemExit, match="unknown model"):
        cli.resolve_model("gpt5", CFG)


def test_alive():
    assert cli.alive(os.getpid())


def test_alive_false_for_dead_pid(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(cli.os, "kill", gone)
    assert not cli.alive(12345)


def test_read_pid_handles_missing_and_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    assert cli.read_pid() is None
    cli.PIDFILE.write_text("not-a-pid")
    assert cli.read_pid() is None
    cli.PIDFILE.write_text("4242")
    assert cli.read_pid() == 4242


def test_stop_cleans_stale_pidfile(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    cli.PIDFILE.write_text("99999999")
    monkeypatch.setattr(cli, "alive", lambda pid: False)
    cli.cmd_stop(None)
    assert not cli.PIDFILE.exists()
    assert "no worker running" in capsys.readouterr().out


def test_start_is_a_noop_when_already_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    cli.PIDFILE.write_text(str(os.getpid()))
    monkeypatch.setattr(cli, "load_run_config", lambda repo: CFG)
    monkeypatch.setattr(
        cli.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn a second worker")
    )
    a = argparse.Namespace(repo="x/y", model=None, hf_token=None, device=None)
    cli.cmd_start(a)
    assert "already contributing" in capsys.readouterr().out


def test_tail(tmp_path):
    p = tmp_path / "log"
    assert cli.tail(p, 5) == ""
    p.write_text("a\nb\nc\n")
    assert cli.tail(p, 2) == "b\nc"


def test_last_activity_skips_stderr_noise(tmp_path):
    p = tmp_path / "log"
    assert cli.last_activity(p) == ""
    p.write_text("07-30 21:05:28 inner step 21/200 loss 3.44\n  warnings.warn(\nTraceback:\n")
    assert cli.last_activity(p) == "07-30 21:05:28 inner step 21/200 loss 3.44"
