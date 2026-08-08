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


RUNS = [
    {"name": "fineweb-150m", "config": "config/run.yaml", "status": "live", "blurb": "145M"},
    {"name": "tinystories-15m", "config": "config/stage1.yaml", "status": "complete", "blurb": ""},
]


def test_choose_run_by_name_and_alias():
    assert cli.choose_run(RUNS, "fineweb", None, False)["name"] == "fineweb-150m"
    assert cli.choose_run(RUNS, "fineweb-150m", None, False)["name"] == "fineweb-150m"


def test_choose_run_refuses_completed_runs():
    with pytest.raises(SystemExit, match="complete"):
        cli.choose_run(RUNS, "tinystories", None, False)


def test_choose_run_rejects_unknown():
    with pytest.raises(SystemExit, match="unknown model"):
        cli.choose_run(RUNS, "gpt5", None, False)


def test_choose_run_remembers_setting_and_recovers_from_stale_one():
    assert cli.choose_run(RUNS, None, "config/run.yaml", True)["name"] == "fineweb-150m"
    assert cli.choose_run(RUNS, None, "config/gone.yaml", True) is None  # -> picker


def test_choose_run_non_interactive_defaults_to_the_live_run():
    assert cli.choose_run(RUNS, None, None, False)["name"] == "fineweb-150m"


def test_load_runs_falls_back_to_single_run(monkeypatch):
    def boom(repo, path, dest, ref="main"):
        raise OSError("no registry")

    monkeypatch.setattr(cli, "fetch_raw", boom)
    runs = cli.load_runs("o/r")
    assert runs[0]["config"] == "config/run.yaml" and runs[0]["status"] == "live"


def test_settings_roundtrip_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    assert cli.read_settings() == {}
    cli.write_settings(run_config="config/run.yaml")
    cli.write_settings(run_name="fineweb-150m")
    assert cli.read_settings() == {"run_config": "config/run.yaml", "run_name": "fineweb-150m"}


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


def test_pending_rounds_counts_parked_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    assert cli.pending_rounds() == 0  # no pending dir yet
    d = tmp_path / "out" / cli.submit.PENDING
    d.mkdir(parents=True)
    (d / "step5_abc.json").write_text("{}")
    (d / "step5_abc.safetensors").write_bytes(b"w")
    assert cli.pending_rounds() == 1  # the pair counts once


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


def test_start_latest_skips_the_picker_and_the_remembered_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    monkeypatch.setattr(cli, "LOGFILE", tmp_path / "worker.log")
    cli.write_settings(run_config="config/stage1.yaml")  # an older remembered run
    monkeypatch.setattr(cli, "load_runs", lambda repo: RUNS)
    monkeypatch.setattr(cli, "load_run_config", lambda repo, path=None: CFG)
    monkeypatch.setattr(cli, "ensure_token", lambda tok: "tester")
    monkeypatch.setattr(cli, "pick", lambda *a: pytest.fail("--latest must not open a menu"))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)  # a real terminal, still no prompt
    monkeypatch.setattr(cli, "alive", lambda pid: True)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    spawned = {}

    def fake_popen(cmd, **kw):
        spawned["cmd"] = cmd
        return argparse.Namespace(pid=42)

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    a = argparse.Namespace(
        repo="x/y",
        model=None,
        hf_token=None,
        device=None,
        choose=False,
        latest=True,
        rounds=0,
        no_progress=True,
    )
    cli.cmd_start(a)

    cmd = spawned["cmd"]
    assert cmd[cmd.index("--run-config") + 1] == "config/run.yaml"  # the live run, not the stored
    assert cli.read_settings()["run_name"] == "fineweb-150m"
    assert "training" in capsys.readouterr().out


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
| # | Contributor | Hardware | Accepted | Tokens | Reputation | Score |
|---|-------------|----------|----------|--------|------------|-------|
| 1 | naloxene | cpu | 16 | 8,806,400 | 1.000 | 8,806,400 |
| 2 | mia-riezebos | nvidia-gpu·cpu | 3 | 2,457,600 | 0.900 | 2,211,840 |
"""

OLD_BOARD_MD = """# Leaderboard
| # | Contributor | Tier | Accepted | Tokens | Reputation | Score |
|---|-------------|------|----------|--------|------------|-------|
| 1 | naloxene | cpu | 16 | 8,806,400 | 1.000 | 8,806,400 |
"""


def test_parse_board():
    rows = cli.parse_board(BOARD_MD)
    assert [r["user"] for r in rows] == ["naloxene", "mia-riezebos"]
    assert rows[0]["tokens"] == 8806400
    assert rows[1]["rank"] == 2


def test_parse_board_reads_archived_boards_too():
    """Past runs' boards keep the Tier column they were written with."""
    assert cli.parse_board(OLD_BOARD_MD)[0]["tokens"] == 8806400


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


def test_now_line_shows_shard_build_progress():
    st = {
        "phase": "building your data shard (one-time)",
        "shard_stage": "downloading docs",
        "shard_done": 5000,
        "shard_total": 20000,
        "shard_per_sec": 50.0,
    }
    line = cli.now_line(st)
    assert line.startswith("building your data shard — downloading docs ")
    assert "25%" in line and "~5m 00s left" in line


def test_now_line_shard_build_survives_a_missing_total():
    """Status written before the first progress report: say the phase, promise nothing."""
    st = {"phase": "building your data shard (one-time)"}
    assert cli.now_line(st) == "building your data shard (one-time)"


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


def test_bar_clamps_and_scales():
    assert cli.bar(0.0, 10) == "░" * 10
    assert cli.bar(1.0, 10) == "█" * 10
    assert cli.bar(2.0, 10) == "█" * 10
    assert cli.bar(0.5, 10) == "█" * 5 + "░" * 5
    assert cli.bar(0.029, 30).count("█") == 1  # small progress still visible


def test_farewell_shows_session_bars_and_resume(tmp_path, monkeypatch, capsys):
    def fake_fetch(repo, path, dest, ref="main"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(BOARD_MD)
        return dest

    def boom(repo):
        raise OSError("offline")

    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "fetch_raw", fake_fetch)
    monkeypatch.setattr(cli, "load_run_config", boom)
    st = {
        "tokens_session": 819200,
        "rounds_done": 1,
        "user": "naloxene",
        "last_pr": "https://huggingface.co/datasets/x/discussions/9",
    }
    cli.farewell(argparse.Namespace(repo="o/r"), st)
    out = capsys.readouterr().out
    assert "819,200 tokens" in out
    assert "█" in out and "░" in out
    assert "your share" in out and "rank 1" in out
    assert "discussions/9" in out
    assert "resume any time: coop start" in out


def test_update_says_so_when_there_is_nothing_new(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: None)
    cli.cmd_update(argparse.Namespace(repo="o/r", check=False, auto=None))
    assert "newest version" in capsys.readouterr().out


def test_update_check_reports_but_installs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(
        cli.update, "available", lambda repo, home, **k: {"version": "99.0.0", "notes": "faster"}
    )
    monkeypatch.setattr(
        cli.update, "apply", lambda *a, **k: pytest.fail("--check must not install")
    )
    cli.cmd_update(argparse.Namespace(repo="o/r", check=True, auto=None))
    out = capsys.readouterr().out
    assert "99.0.0 is out" in out and "faster" in out and "`coop update` installs it" in out


def test_update_installs_and_points_at_the_running_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    cli.PIDFILE.write_text(str(os.getpid()))
    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: {"version": "99.0.0"})
    monkeypatch.setattr(cli.update, "install_kind", lambda: cli.update.UVX)
    monkeypatch.setattr(cli.update, "apply", lambda repo, kind: (True, "ok"))
    cli.cmd_update(argparse.Namespace(repo="o/r", check=False, auto=None))
    out = capsys.readouterr().out
    assert "updated to coop 99.0.0" in out
    assert "coop stop" in out  # the worker is still on the old code


def test_update_refuses_to_move_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: {"version": "99.0.0"})
    monkeypatch.setattr(cli.update, "install_kind", lambda: cli.update.GIT)
    monkeypatch.setattr(cli.update, "apply", lambda *a, **k: pytest.fail("hands off the checkout"))
    with pytest.raises(SystemExit, match="git pull"):
        cli.cmd_update(argparse.Namespace(repo="o/r", check=False, auto=None))


def test_update_auto_on_persists_and_changes_the_notice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: None)
    cli.cmd_update(argparse.Namespace(repo="o/r", check=True, auto="on"))
    assert cli.read_settings()["auto_update"] is True
    assert "auto-update on" in capsys.readouterr().out

    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: {"version": "99.0.0"})
    assert "auto-update will take it" in cli.update_line("o/r")

    cli.cmd_update(argparse.Namespace(repo="o/r", check=True, auto="off"))
    assert cli.read_settings()["auto_update"] is False
    assert "run `coop update`" in cli.update_line("o/r")


def test_progress_auto_off_persists_and_start_stops_opening_the_view(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "watch", lambda *a, **k: pytest.fail("--auto must not open the view"))
    cli.cmd_progress(argparse.Namespace(repo="o/r", auto="off", once=False, advanced=False))
    assert cli.read_settings()["progress_auto"] is False
    assert "prints a summary and returns" in capsys.readouterr().out

    a = argparse.Namespace(repo="o/r", no_progress=False)
    with monkeypatch.context() as tty:
        tty.setattr(cli.sys, "stdin", FakeTTY())
        tty.setattr(cli.sys, "stdout", FakeTTY())
        assert cli.show_progress(a) is False

    cli.cmd_progress(argparse.Namespace(repo="o/r", auto="on", once=False, advanced=False))
    assert cli.read_settings()["progress_auto"] is True
    assert "opens the live view" in capsys.readouterr().out
    monkeypatch.setattr(cli.sys, "stdin", FakeTTY())
    monkeypatch.setattr(cli.sys, "stdout", FakeTTY())
    assert cli.show_progress(a) is True
    assert cli.show_progress(argparse.Namespace(repo="o/r", no_progress=True)) is False


def test_show_progress_never_opens_the_view_into_a_pipe(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", type("NoTTY", (), {"isatty": lambda self: False})())
    assert cli.show_progress(argparse.Namespace(repo="o/r", no_progress=False)) is False


def test_start_opens_the_view_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    monkeypatch.setattr(cli, "LOGFILE", tmp_path / "worker.log")
    monkeypatch.setattr(cli, "load_runs", lambda repo: RUNS)
    monkeypatch.setattr(cli, "load_run_config", lambda repo, path=None: CFG)
    monkeypatch.setattr(cli, "ensure_token", lambda tok: "tester")
    monkeypatch.setattr(cli, "alive", lambda pid: True)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: argparse.Namespace(pid=42))
    monkeypatch.setattr(cli, "show_progress", lambda a: True)
    watched = []
    monkeypatch.setattr(cli, "watch", lambda a: watched.append(a))
    a = argparse.Namespace(
        repo="x/y",
        model=None,
        hf_token=None,
        device=None,
        choose=False,
        latest=True,
        rounds=0,
        no_progress=False,
    )
    cli.cmd_start(a)
    assert watched == [a]
    assert "coop stop      stop contributing" not in capsys.readouterr().out


def test_probe_local_reads_the_worker_without_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    monkeypatch.setattr(cli, "LOGFILE", tmp_path / "worker.log")
    monkeypatch.setattr(cli, "idle_device_label", lambda: "CPU")
    monkeypatch.setattr(cli, "fetch_raw", lambda *a, **k: pytest.fail("the redraw must not fetch"))
    cli.PIDFILE.write_text(str(os.getpid()))
    (tmp_path / cli.STATUS_FILENAME).write_text(
        '{"phase": "training", "device_label": "Apple GPU"}'
    )

    ctx = cli.probe_local(argparse.Namespace(repo="o/r"))
    assert ctx["running"] is True and ctx["pid"] == os.getpid()
    assert ctx["device_label"] == "Apple GPU"  # what the worker got, not what we'd pick
    assert ctx["st"]["phase"] == "training"


def test_probe_remote_survives_an_unreachable_config(tmp_path, monkeypatch):
    """load_run_config exits on failure, and SystemExit would take the refresh thread
    with it — the view has to come back with an empty hand instead."""
    monkeypatch.setattr(cli, "HOME", tmp_path)

    def offline(repo, path=None, dest="run.yaml"):
        raise SystemExit("could not fetch the run config")

    monkeypatch.setattr(cli, "load_run_config", offline)
    assert cli.probe_remote(argparse.Namespace(repo="o/r")) == {}


def test_watch_stops_the_worker_when_the_view_says_so(monkeypatch):
    stopped = []
    monkeypatch.setattr(cli.progress, "view", lambda **kw: cli.progress.STOP)
    monkeypatch.setattr(cli, "cmd_stop", lambda a: stopped.append(a))
    a = argparse.Namespace(repo="o/r")
    cli.watch(a)
    assert stopped == [a]


def test_watch_restarts_then_returns_to_the_view(monkeypatch):
    choices = iter([cli.progress.START, cli.progress.LEAVE])
    started = []
    monkeypatch.setattr(cli.progress, "view", lambda **kw: next(choices))
    monkeypatch.setattr(cli, "cmd_start", lambda a: started.append(a))
    cli.watch(argparse.Namespace(repo="o/r"))
    assert len(started) == 1
    assert started[0].no_progress is True  # the loop reopens the view, cmd_start must not


def test_status_shows_a_waiting_update(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "PIDFILE", tmp_path / "worker.pid")
    monkeypatch.setattr(cli, "load_run_config", lambda repo: CFG)
    monkeypatch.setattr(cli.update, "available", lambda repo, home, **k: {"version": "99.0.0"})
    monkeypatch.setattr(cli.hubio, "whoami", lambda: "tester")
    monkeypatch.setattr(cli, "fetch_raw", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    cli.cmd_status(argparse.Namespace(repo="o/r"))
    assert "update   coop 99.0.0 is out" in capsys.readouterr().out


def test_now_line_says_a_worker_is_failing_instead_of_pretending():
    """`running` plus a frozen phase reads healthy; an AFK volunteer needs the truth."""
    line = cli.now_line({"phase": "round failed — retrying", "failing": 3, "last_error": "403"})
    assert "3 rounds in a row failed" in line
    assert "403" in line
    one = cli.now_line({"phase": "round failed — retrying", "failing": 1})
    assert "1 round in a row failed" in one
