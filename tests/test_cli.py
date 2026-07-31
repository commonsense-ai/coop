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


def test_fmt_eta():
    assert cli.fmt_eta(45) == "45s"
    assert cli.fmt_eta(150) == "2m 30s"
    assert cli.fmt_eta(7300) == "2h 01m"
    assert cli.fmt_eta(-5) == "0s"


BOARD_MD = """# Leaderboard
| # | Contributor | Tier | Accepted | Tokens | Reputation | Score |
|---|-------------|------|----------|--------|------------|-------|
| 1 | naloxene | cpu | 16 | 8,806,400 | 1.000 | 8,806,400 |
| 2 | mia-riezebos | gpu | 3 | 2,457,600 | 0.900 | 2,211,840 |
"""


def test_parse_board():
    rows = cli.parse_board(BOARD_MD)
    assert [r["user"] for r in rows] == ["naloxene", "mia-riezebos"]
    assert rows[0]["tokens"] == 8806400
    assert rows[1]["rank"] == 2


def test_now_line_training_has_progress_and_eta():
    st = {
        "phase": "training",
        "inner_step": 100,
        "h_steps": 500,
        "loss": 3.21,
        "steps_per_sec": 2.0,
    }
    assert cli.now_line(st) == "training — inner step 100/500 · loss 3.21 · ~3m 20s left"


def test_now_line_waiting_names_the_next_step():
    assert "outer step 13" in cli.now_line({"phase": "waiting", "waiting_past_step": 12})


def test_now_line_passes_other_phases_through():
    assert cli.now_line({"phase": "downloading checkpoint"}) == "downloading checkpoint"


def test_bare_coop_prints_welcome(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["coop"])
    cli.main()
    out = capsys.readouterr().out
    assert "coop start" in out and "one-time setup" in out


class FakeTTY:
    def isatty(self):
        return True


def test_ensure_token_non_tty_exits(monkeypatch):
    monkeypatch.setattr(cli.hubio, "whoami", lambda: "anonymous")
    monkeypatch.setattr(cli.sys, "stdin", type("NoTTY", (), {"isatty": lambda self: False})())
    with pytest.raises(SystemExit, match="hf_"):
        cli.ensure_token(None)


def test_ensure_token_wizard_retries_then_succeeds(monkeypatch, capsys):
    import getpass

    import huggingface_hub

    state = {"user": "anonymous"}
    tokens = iter(["", "hf_bad", "hf_good"])

    def fake_login(token):
        if token == "hf_bad":
            raise ValueError("bad token")
        state["user"] = "alice"

    monkeypatch.setattr(cli.hubio, "whoami", lambda: state["user"])
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(getpass, "getpass", lambda prompt: next(tokens))
    monkeypatch.setattr(huggingface_hub, "login", fake_login)
    assert cli.ensure_token(None) == "alice"
    out = capsys.readouterr().out
    assert "one-time setup" in out
    assert "rejected that token" in out
    assert "you're in, alice!" in out


def test_ensure_token_skips_wizard_when_logged_in(monkeypatch, capsys):
    monkeypatch.setattr(cli.hubio, "whoami", lambda: "naloxene")
    assert cli.ensure_token(None) == "naloxene"
    assert "one-time setup" not in capsys.readouterr().out
