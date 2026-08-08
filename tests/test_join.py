import os
import sys
import urllib.request

import pytest

import coop.join as join
from coop.status import StatusFile, read_status


def test_shard_progress_feeds_status_and_finishes_at_full(tmp_path):
    path = tmp_path / "status.json"
    report = join.shard_progress(StatusFile(path), "downloading docs", "docs")
    report(0, 20000)
    st = read_status(path)
    assert (st["shard_stage"], st["shard_done"], st["shard_total"]) == (
        "downloading docs",
        0,
        20000,
    )
    report(5000, 20000)  # throttled to ~1 Hz: this one is dropped, the last one never is
    assert read_status(path)["shard_done"] == 0
    report(20000, 20000)
    assert read_status(path)["shard_done"] == 20000


def test_derive_skip_deterministic_and_bounded():
    a = join.derive_skip("alice")
    assert a == join.derive_skip("alice")
    assert a % 20000 == 0
    assert join.derive_skip("alice", docs=1000) % 1000 == 0


def test_derive_skip_shard_fits_inside_dataset():
    # regression: naloxene hashed to slot 159 of 500 -> skip 3.18M -> empty shard
    for user in ["alice", "bob", "naloxene", "mia-riezebos"]:
        skip = join.derive_skip(user)
        assert skip + 20000 <= join.TRAIN_DOCS
    assert join.derive_skip("x", docs=30000, total_docs=60000) in (0, 30000)
    assert join.derive_skip("x", docs=30000, total_docs=29999) == 0  # degenerate: one slot


def test_fetch_raw(monkeypatch, tmp_path):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"data: 1"

    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = join.fetch_raw("o/r", "config/run.yaml", tmp_path / "sub" / "run.yaml")
    assert p.read_text() == "data: 1"
    assert captured["url"] == "https://raw.githubusercontent.com/o/r/main/config/run.yaml"


def test_machine_seed_stable_per_workdir_distinct_across(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    assert join.machine_seed(a) == join.machine_seed(a)  # persisted, stable
    assert join.machine_seed(a) != join.machine_seed(b)  # two machines, two seeds


def test_corpus_fingerprint_distinguishes_runs():
    a = join.corpus_fingerprint({"hf_dataset": "d1", "tokenizer": "t1"})
    b = join.corpus_fingerprint({"hf_dataset": "d2", "tokenizer": "t1"})
    c = join.corpus_fingerprint({"hf_dataset": "d1", "tokenizer": "t2"})
    assert len({a, b, c}) == 3  # any run-identity change means a different shard file
    assert a == join.corpus_fingerprint({"hf_dataset": "d1", "tokenizer": "t1"})


def test_config_changed_detects_new_run_but_not_blips(monkeypatch, tmp_path):
    def fake_fetch(repo, path, dest, ref="main"):
        dest.write_text("repos: {model: new/run}")
        return dest

    monkeypatch.setattr(join, "fetch_raw", fake_fetch)
    assert join.config_changed("o/r", tmp_path, "repos: {model: old/run}")
    assert not join.config_changed("o/r", tmp_path, "repos: {model: new/run}")

    def boom(repo, path, dest, ref="main"):
        raise OSError("offline")

    monkeypatch.setattr(join, "fetch_raw", boom)
    # a network blip must never restart the worker
    assert not join.config_changed("o/r", tmp_path, "repos: {model: old/run}")


def test_should_restart_needs_a_streak_and_stops_after_a_few(monkeypatch):
    monkeypatch.delenv(join.RESTARTS_ENV, raising=False)
    assert not join.should_restart(join.FAILS_BEFORE_RESTART - 1)  # one blip is not a wedge
    assert join.should_restart(join.FAILS_BEFORE_RESTART)
    # fresh processes failing identically means the outage is not ours to restart out of
    monkeypatch.setenv(join.RESTARTS_ENV, str(join.MAX_RESTARTS))
    assert not join.should_restart(join.FAILS_BEFORE_RESTART * 10)
    monkeypatch.setenv(join.RESTARTS_ENV, "junk")  # an unreadable counter must not wedge it
    assert join.should_restart(join.FAILS_BEFORE_RESTART)


def test_restart_worker_reexecs_this_worker_and_carries_the_streak(monkeypatch, tmp_path):
    # regression: a 403 on a closed PR poisoned the hub client, and every later round
    # failed on the cached error — including rounds that never reached an upload
    seen = {}

    def fake_execv(exe, argv):
        seen["argv"] = argv
        raise SystemExit  # execv never returns

    monkeypatch.setattr(os, "execv", fake_execv)
    monkeypatch.setattr(sys, "argv", ["coop-join", "--pause", "60"])
    monkeypatch.setenv(join.RESTARTS_ENV, "1")
    path = tmp_path / "status.json"
    with pytest.raises(SystemExit):
        join.restart_worker(StatusFile(path), join.FAILS_BEFORE_RESTART)
    assert seen["argv"][1:] == ["-m", "coop.join", "--pause", "60"]
    assert os.environ[join.RESTARTS_ENV] == "2"  # the child is told it is a restart
    assert "restarting" in read_status(path)["phase"]


def test_next_pause_backs_off_to_a_cap():
    pause = 60
    seen = []
    for _ in range(8):
        pause = join.next_pause(pause, 60)
        seen.append(pause)
    assert seen[:3] == [120, 240, 480]
    assert seen[-1] == join.PAUSE_MAX  # an outage lasts hours; stop polling every minute
    assert join.next_pause(join.PAUSE_MAX, 60) == join.PAUSE_MAX


def _recover_args(tmp_path, status):
    return ("o/r", tmp_path, "cfg", "config/run.yaml", status, join.FAILS_BEFORE_RESTART)


def test_recover_looks_for_a_fix_only_once_restarts_are_spent(monkeypatch, tmp_path):
    """The gap this closes: check_for_update lives on the path a broken worker never
    reaches, so the machines that need a release are the ones that never see it."""
    monkeypatch.setattr(join, "config_changed", lambda *a, **k: False)
    monkeypatch.setattr(join, "restart_worker", lambda *a: (_ for _ in ()).throw(SystemExit))
    looked = []
    monkeypatch.setattr(join, "check_for_update", lambda *a, **k: looked.append(k))

    monkeypatch.setenv(join.RESTARTS_ENV, "0")  # budget left: restart before repairing
    with pytest.raises(SystemExit):
        join.recover(*_recover_args(tmp_path, StatusFile(tmp_path / "s.json")))
    assert looked == []

    monkeypatch.setenv(join.RESTARTS_ENV, str(join.MAX_RESTARTS))  # spent, nothing landed
    join.recover(*_recover_args(tmp_path, StatusFile(tmp_path / "s.json")))
    assert looked == [{"repair": True}]


def test_recover_leaves_a_single_blip_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(
        join, "restart_worker", lambda *a: pytest.fail("one failure is not a wedge")
    )
    monkeypatch.setattr(join, "check_for_update", lambda *a, **k: pytest.fail("too eager"))
    monkeypatch.setattr(join, "config_changed", lambda *a, **k: pytest.fail("too eager"))
    join.recover("o/r", tmp_path, "cfg", "config/run.yaml", StatusFile(tmp_path / "s.json"), 1)
