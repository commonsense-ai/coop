import urllib.request

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
