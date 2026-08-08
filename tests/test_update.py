import sys

import pytest

import coop.join as join
import coop.update as update
from coop import __version__, settings


def manifest(version="99.0.0"):
    return {"version": version, "url": "https://example.invalid/tag", "notes": "faster rounds"}


def test_parse_version_orders_releases():
    assert update.parse_version("0.3.1") == (0, 3, 1)
    assert update.parse_version("1.0") == (1, 0, 0)
    assert update.parse_version("0.3.0rc1") == (0, 3, 0)


def test_unreadable_versions_never_look_newer():
    # a corrupt or hostile manifest must not talk a volunteer into an update
    assert not update.is_newer("nonsense", "0.2.2")
    assert not update.is_newer("", "0.2.2")
    assert not update.is_newer("0.2.2", "0.2.2")
    assert not update.is_newer("0.2.1", "0.2.2")
    assert update.is_newer("0.2.3", "0.2.2")
    assert update.is_newer("0.10.0", "0.9.9")


def test_is_newer_defaults_to_the_running_version():
    assert update.is_newer("999.0.0")
    assert not update.is_newer(__version__)


def test_available_only_reports_newer_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "fetch_manifest", lambda repo, timeout=10: manifest(__version__))
    assert update.available("o/r", tmp_path) is None
    monkeypatch.setattr(update, "fetch_manifest", lambda repo, timeout=10: manifest())
    assert update.available("o/r", tmp_path, force=True)["version"] == "99.0.0"


def test_the_check_is_cached_until_its_ttl_expires(tmp_path, monkeypatch):
    calls = []

    def fake(repo, timeout=10):
        calls.append(repo)
        return manifest()

    monkeypatch.setattr(update, "fetch_manifest", fake)
    assert update.available("o/r", tmp_path)
    assert update.available("o/r", tmp_path)  # served from the cache file
    assert len(calls) == 1
    assert update.available("o/r", tmp_path, ttl=0)
    assert len(calls) == 2


def test_a_failed_check_falls_back_to_what_was_last_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "fetch_manifest", lambda repo, timeout=10: manifest())
    update.available("o/r", tmp_path)

    def offline(repo, timeout=10):
        raise OSError("no network")

    monkeypatch.setattr(update, "fetch_manifest", offline)
    assert update.available("o/r", tmp_path, force=True)["version"] == "99.0.0"


def test_a_first_check_without_network_is_silent(tmp_path, monkeypatch):
    def offline(repo, timeout=10):
        raise OSError("no network")

    monkeypatch.setattr(update, "fetch_manifest", offline)
    assert update.available("o/r", tmp_path) is None


def test_install_kind_recognises_a_git_checkout():
    assert update.install_kind() == update.GIT  # the tests run from one


def as_install(monkeypatch, tmp_path, prefix):
    monkeypatch.setattr(update, "__file__", str(tmp_path / "site-packages" / "coop" / "update.py"))
    monkeypatch.setattr(update.sys, "prefix", prefix)


def test_install_kind_tells_uvx_from_a_uv_tool_install(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    as_install(monkeypatch, tmp_path, "/home/v/.cache/uv/archive-v0/abc123")
    assert update.install_kind() == update.UVX
    as_install(monkeypatch, tmp_path, "/home/v/.local/share/uv/tools/coop-ai")
    assert update.install_kind() == update.UV_TOOL
    as_install(monkeypatch, tmp_path, "/usr/local")
    assert update.install_kind() == update.PIP


def test_install_kind_honours_a_relocated_uv_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "elsewhere"))
    as_install(monkeypatch, tmp_path, str(tmp_path / "elsewhere" / "archive-v0" / "abc"))
    assert update.install_kind() == update.UVX


def test_update_argv_refreshes_the_uvx_cache_entry():
    argv = update.update_argv("o/r", update.UVX)
    assert argv[0].endswith("uvx")
    assert "--refresh-package" in argv and update.PACKAGE in argv
    assert "git+https://github.com/o/r" in argv


def test_update_argv_upgrades_a_pip_install_from_git():
    argv = update.update_argv("o/r", update.PIP)
    assert argv[:4] == [sys.executable, "-m", "pip", "install"]
    assert argv[-1] == "git+https://github.com/o/r"


def test_a_git_checkout_is_never_updated_for_the_volunteer():
    assert update.update_argv("o/r", update.GIT) is None
    ok, why = update.apply("o/r", update.GIT)
    assert not ok and "git pull" in why


def test_apply_never_raises_when_the_updater_is_missing(monkeypatch):
    def missing(*a, **k):
        raise OSError("uvx: not found")

    monkeypatch.setattr(update.subprocess, "run", missing)
    ok, out = update.apply("o/r", update.UVX)
    assert not ok and "uvx" in out


def test_worker_argv_re_execs_through_uvx_for_a_uvx_install():
    argv = update.worker_argv("o/r", ["--workdir", "/w"], update.UVX)
    assert argv[0].endswith("uvx")
    assert argv[-3:] == ["coop-join", "--workdir", "/w"]


def test_worker_argv_uses_this_interpreter_otherwise():
    argv = update.worker_argv("o/r", ["--workdir", "/w"], update.PIP)
    assert argv == [sys.executable, "-m", "coop.join", "--workdir", "/w"]


def test_restart_updates_first_then_execs(monkeypatch):
    applied, execed = [], []
    monkeypatch.setattr(
        update, "apply", lambda repo, kind=None: (applied.append(kind), (True, ""))[1]
    )
    monkeypatch.setattr(update.os, "execvp", lambda f, argv: execed.append(argv))
    update.restart_into_update("o/r", ["--workdir", "/w"], update.PIP)
    assert applied == [update.PIP]
    assert execed[0][:3] == [sys.executable, "-m", "coop.join"]


def test_a_uvx_worker_builds_the_new_env_before_the_point_of_no_return(monkeypatch):
    # execvp cannot be undone: the new version must already be on disk and known to
    # start, or a half-finished download leaves a volunteer with no worker at all
    order = []
    monkeypatch.setattr(
        update, "apply", lambda repo, kind=None: (order.append("built"), (True, ""))[1]
    )
    monkeypatch.setattr(update.os, "execvp", lambda f, argv: order.append(argv))
    update.restart_into_update("o/r", [], update.UVX)
    assert order[0] == "built"
    assert "--refresh-package" not in order[1]  # the entry is current; re-resolving is not
    assert order[1][-1] == "coop-join"


def test_restart_leaves_a_git_checkout_running(monkeypatch):
    monkeypatch.setattr(update.os, "execvp", lambda f, argv: pytest.fail("must not restart"))
    assert "git pull" in update.restart_into_update("o/r", [], update.GIT)


def test_restart_reports_a_failed_update_instead_of_execing(monkeypatch):
    monkeypatch.setattr(update, "apply", lambda repo, kind=None: (False, "boom\nno space left"))
    monkeypatch.setattr(update.os, "execvp", lambda f, argv: pytest.fail("must not restart"))
    assert update.restart_into_update("o/r", [], update.PIP) == "no space left"


def test_a_cache_write_that_fails_never_breaks_a_check(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "fetch_manifest", lambda repo, timeout=5: manifest())
    unwritable = tmp_path / "home"
    unwritable.write_text("not a directory")  # mkdir and write both fail here
    assert update.available("o/r", unwritable)["version"] == "99.0.0"
    update.mark_attempted(unwritable, "99.0.0")  # must not raise either


def test_one_update_attempt_per_version(tmp_path):
    assert not update.attempted(tmp_path, "0.3.0")
    update.mark_attempted(tmp_path, "0.3.0")
    assert update.attempted(tmp_path, "0.3.0")
    assert not update.attempted(tmp_path, "0.3.1")
    assert not update.attempted(tmp_path, "")


def test_auto_update_is_off_until_the_volunteer_asks(tmp_path):
    assert not update.auto_enabled(tmp_path)
    settings.write(tmp_path, auto_update=True)
    assert update.auto_enabled(tmp_path)
    settings.write(tmp_path, auto_update=False)
    assert not update.auto_enabled(tmp_path)


class FakeStatus:
    def __init__(self):
        self.state: dict = {}

    def update(self, **fields):
        self.state |= fields


def test_the_worker_only_flags_the_update_when_auto_is_off(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(update, "available", lambda repo, home, **k: manifest())
    monkeypatch.setattr(update, "restart_into_update", lambda *a, **k: pytest.fail("no restart"))
    monkeypatch.setattr(join, "NOTED", set())
    st = FakeStatus()
    with caplog.at_level("INFO", logger=join.log.name):
        join.check_for_update("o/r", tmp_path, st)
        join.check_for_update("o/r", tmp_path, st)  # every round, for hours
    assert st.state["update_available"] == "99.0.0"
    # the notice belongs in `coop status`, not repeated down the whole log
    assert len([r for r in caplog.records if "99.0.0" in r.getMessage()]) == 1


def test_the_worker_restarts_into_the_update_when_auto_is_on(tmp_path, monkeypatch):
    settings.write(tmp_path, auto_update=True)
    monkeypatch.setattr(update, "available", lambda repo, home, **k: manifest())
    seen = {}

    def fake_restart(repo, args, kind=None):
        seen["repo"], seen["args"] = repo, args
        return "exec failed"

    monkeypatch.setattr(update, "restart_into_update", fake_restart)
    st = FakeStatus()
    join.check_for_update("o/r", tmp_path, st)
    assert seen["repo"] == "o/r"
    assert update.attempted(tmp_path, "99.0.0")  # marked before the attempt, not after
    assert "still training" in st.state["phase"]


def test_a_version_that_failed_to_land_is_not_retried_forever(tmp_path, monkeypatch):
    settings.write(tmp_path, auto_update=True)
    update.mark_attempted(tmp_path, "99.0.0")
    monkeypatch.setattr(update, "available", lambda repo, home, **k: manifest())
    monkeypatch.setattr(update, "restart_into_update", lambda *a, **k: pytest.fail("would loop"))
    join.check_for_update("o/r", tmp_path, FakeStatus())


def test_no_new_version_is_a_no_op_for_the_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "available", lambda repo, home, **k: None)
    st = FakeStatus()
    join.check_for_update("o/r", tmp_path, st)
    assert st.state == {}
