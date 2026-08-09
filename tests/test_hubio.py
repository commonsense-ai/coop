import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from huggingface_hub.utils import EntryNotFoundError, _xet
from safetensors.torch import save_file

from coop import hubio


def fake_api(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(hubio, "HfApi", lambda token=None: mock)
    return mock


def xet_resets(monkeypatch) -> list:
    resets = []
    monkeypatch.setattr(_xet, "abort_xet_session", lambda: resets.append(1))
    return resets


def test_upload_checkpoint_is_one_commit(monkeypatch):
    mock = fake_api(monkeypatch)
    hubio.upload_checkpoint(
        "x/model", {"w": torch.ones(2)}, {"step": 3}, opt_state={"w": torch.zeros(2)}
    )
    assert mock.create_commit.call_count == 1  # batched: one write API call per tick
    kwargs = mock.create_commit.call_args.kwargs
    assert kwargs["repo_id"] == "x/model"
    assert kwargs["commit_message"] == "step 3"
    assert [op.path_in_repo for op in kwargs["operations"]] == [
        hubio.CKPT_FILE,
        hubio.OPT_FILE,
        hubio.META_FILE,
    ]


def test_download_checkpoint_pins_revision(monkeypatch, tmp_path):
    mock = fake_api(monkeypatch)
    mock.repo_info.return_value = SimpleNamespace(sha="abc123")
    ckpt = tmp_path / hubio.CKPT_FILE
    save_file({"w": torch.ones(2)}, str(ckpt))
    meta = tmp_path / hubio.META_FILE
    meta.write_text(json.dumps({"step": 5}))
    revisions = []

    def fake_download(repo, fn, revision=None, **kw):
        revisions.append(revision)
        return str(ckpt) if fn == hubio.CKPT_FILE else str(meta)

    monkeypatch.setattr(hubio, "hf_hub_download", fake_download)
    state, m = hubio.download_checkpoint("x/model")
    assert torch.equal(state["w"], torch.ones(2))
    assert m == {"step": 5}
    # both files read at the same pinned sha: no mixed-step state if a tick lands mid-read
    assert revisions == ["abc123", "abc123"]


def test_download_optimizer_missing(monkeypatch):
    def raise_(*a, **kw):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(hubio, "hf_hub_download", raise_)
    assert hubio.download_optimizer("x/model") is None


def test_list_open_prs_filters(monkeypatch):
    mock = fake_api(monkeypatch)
    mock.get_repo_discussions.return_value = iter(
        [
            SimpleNamespace(num=1, is_pull_request=True, status="open"),
            SimpleNamespace(num=2, is_pull_request=False, status="open"),
            SimpleNamespace(num=3, is_pull_request=True, status="closed"),
        ]
    )
    assert [p.num for p in hubio.list_open_prs("x/inbox")] == [1]


def test_download_pr_files_only_new_submissions(monkeypatch):
    mock = fake_api(monkeypatch)
    mock.list_repo_files.return_value = [
        "README.md",
        "submissions/step_1/old.safetensors",
        "submissions/step_2/alice_ab.safetensors",
        "submissions/step_2/alice_ab.json",
    ]
    monkeypatch.setattr(hubio, "hf_hub_download", lambda repo, fn, **kw: f"/local/{fn}")
    files = hubio.download_pr_files("x/inbox", 7, base_files={"submissions/step_1/old.safetensors"})
    assert set(files) == {
        "submissions/step_2/alice_ab.safetensors",
        "submissions/step_2/alice_ab.json",
    }
    assert mock.list_repo_files.call_args.kwargs["revision"] == "refs/pr/7"


def test_close_pr(monkeypatch):
    mock = fake_api(monkeypatch)
    hubio.merge_or_close_pr("x/inbox", 7, merge=False, comment="done")
    mock.comment_discussion.assert_called_once()
    mock.change_discussion_status.assert_called_once()
    assert mock.change_discussion_status.call_args.kwargs["new_status"] == "closed"
    mock.merge_pull_request.assert_not_called()


def test_merge_pr(monkeypatch):
    mock = fake_api(monkeypatch)
    hubio.merge_or_close_pr("x/inbox", 7, merge=True)
    mock.merge_pull_request.assert_called_once()
    mock.comment_discussion.assert_not_called()


def test_open_pr_creates_pr(monkeypatch):
    mock = fake_api(monkeypatch)
    hubio.open_pr("x/inbox", [], "msg")
    assert mock.create_commit.call_args.kwargs["create_pr"] is True


def test_a_failed_upload_resets_the_xet_session(monkeypatch):
    """The aggregator closing our PR mid-round 403s the write. Left alone, that error
    latches on the shared XetSession and every later transfer in the process replays it —
    so the retry that should open a fresh PR fails on the dead PR's error instead."""
    resets = xet_resets(monkeypatch)
    mock = fake_api(monkeypatch)
    mock.create_commit.side_effect = RuntimeError("403 Forbidden")
    with pytest.raises(RuntimeError):
        hubio.update_pr("x/inbox", 36, [], "msg")
    assert resets == [1]


def test_a_failed_download_resets_the_xet_session(monkeypatch):
    resets = xet_resets(monkeypatch)

    def raise_(*a, **kw):
        raise RuntimeError("Previous task error")

    monkeypatch.setattr(hubio, "hf_hub_download", raise_)
    with pytest.raises(RuntimeError):
        hubio.download_file("x/model", "val.bin")
    assert resets == [1]


def test_a_working_transfer_leaves_the_session_alone(monkeypatch):
    resets = xet_resets(monkeypatch)
    fake_api(monkeypatch)
    hubio.open_pr("x/inbox", [], "msg")
    monkeypatch.setattr(hubio, "hf_hub_download", lambda *a, **kw: "/local/f")
    hubio.download_file("x/model", "val.bin")
    assert resets == []


def test_a_missing_optimizer_is_not_a_failed_transfer(monkeypatch):
    """Step 0 has no optimizer file; that is a normal read, not a poisoned session."""
    resets = xet_resets(monkeypatch)

    def raise_(*a, **kw):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(hubio, "hf_hub_download", raise_)
    assert hubio.download_optimizer("x/model") is None
    assert resets == []


def fake_cache(monkeypatch, revisions, repo_id="x/model", repo_type="model"):
    """A scanned cache of (sha, last_modified) revisions; returns the deleted shas."""
    deleted = []

    def execute():
        deleted.extend(strategy.shas)

    strategy = SimpleNamespace(expected_freed_size=557_000_000, execute=execute, shas=())

    def delete_revisions(*shas):
        strategy.shas = shas
        return strategy

    repo = SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        revisions=[SimpleNamespace(commit_hash=s, last_modified=t) for s, t in revisions],
    )
    cache = SimpleNamespace(repos=[repo], delete_revisions=delete_revisions)
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)
    return deleted


def test_prune_cache_keeps_the_newest_and_evicts_the_rest(monkeypatch):
    deleted = fake_cache(monkeypatch, [("old", 1), ("mid", 2), ("new", 3)])
    assert hubio.prune_cache("x/model", keep=2) == 1
    assert deleted == ["old"]


def test_prune_cache_spares_the_revision_in_hand_even_when_it_looks_old(monkeypatch):
    """A cache hit doesn't restat the snapshot, so the sha we are reading can be the
    oldest by mtime. Evicting it would delete the checkpoint out from under the round."""
    deleted = fake_cache(monkeypatch, [("inhand", 1), ("mid", 2), ("new", 3)])
    assert hubio.prune_cache("x/model", current="inhand", keep=2) == 1
    assert deleted == ["mid"]  # "new" fills the spare slot; "inhand" is protected


def test_prune_cache_leaves_other_repos_alone(monkeypatch):
    deleted = fake_cache(monkeypatch, [("a", 1), ("b", 2)], repo_id="other/repo")
    assert hubio.prune_cache("x/model", keep=1) == 0
    assert deleted == []


def test_prune_cache_never_fails_a_round(monkeypatch):
    """Training already happened; an unscannable cache must not cost the round."""

    def boom():
        raise OSError("read-only home")

    monkeypatch.setattr("huggingface_hub.scan_cache_dir", boom)
    assert hubio.prune_cache("x/model") == 0


def test_download_checkpoint_prunes_the_revisions_it_supersedes(monkeypatch, tmp_path):
    mock = fake_api(monkeypatch)
    mock.repo_info.return_value = SimpleNamespace(sha="new")
    ckpt = tmp_path / hubio.CKPT_FILE
    save_file({"w": torch.ones(2)}, str(ckpt))
    meta = tmp_path / hubio.META_FILE
    meta.write_text(json.dumps({"step": 5}))
    monkeypatch.setattr(
        hubio,
        "hf_hub_download",
        lambda repo, fn, **kw: str(ckpt) if fn == hubio.CKPT_FILE else str(meta),
    )
    deleted = fake_cache(monkeypatch, [("old", 1), ("prev", 2), ("new", 3)])
    state, m = hubio.download_checkpoint("x/model")
    assert m == {"step": 5} and torch.equal(state["w"], torch.ones(2))
    assert deleted == ["old"]  # every outer step is a new sha; the cache must not just grow


def test_prune_cache_keep_zero_evicts_everything(monkeypatch):
    """What the aggregator asks for: once a tick has closed every PR, none of the inbox
    revisions it cached will ever be read again."""
    deleted = fake_cache(monkeypatch, [("pr1", 1), ("pr2", 2), ("pr3", 3)], repo_type="dataset")
    assert hubio.prune_cache("x/model", keep=0, repo_type="dataset") == 3
    assert sorted(deleted) == ["pr1", "pr2", "pr3"]


def test_byte_reporter_counts_without_drawing(capsys):
    """Borrowed from hf_hub_download's tqdm: it must report bytes and print nothing —
    the daemon's stdout is worker.log, and a progress bar there is unreadable noise."""
    seen = []
    reporter = hubio.byte_reporter(lambda done, total: seen.append((done, total)))
    with reporter(total=584_113_800, unit="B", desc="checkpoint.safetensors") as bar:
        bar.update(8_329)
        bar.update(469_515_579)
    assert seen[0] == (0, 584_113_800)  # the total is known before a byte arrives
    assert seen[-1] == (469_523_908, 584_113_800)
    assert capsys.readouterr().err == ""


def test_download_checkpoint_reports_bytes_only_for_the_checkpoint(monkeypatch, tmp_path):
    """meta.json is a few hundred bytes; its total would overwrite the one being waited on."""
    mock = fake_api(monkeypatch)
    mock.repo_info.return_value = SimpleNamespace(sha="new")
    ckpt = tmp_path / hubio.CKPT_FILE
    save_file({"w": torch.ones(2)}, str(ckpt))
    meta = tmp_path / hubio.META_FILE
    meta.write_text(json.dumps({"step": 5}))
    got = {}

    def fake_download(repo, fn, **kw):
        got[fn] = kw.get("tqdm_class")
        return str(ckpt) if fn == hubio.CKPT_FILE else str(meta)

    monkeypatch.setattr(hubio, "hf_hub_download", fake_download)
    fake_cache(monkeypatch, [])
    hubio.download_checkpoint("x/model", on_bytes=lambda done, total: None)
    assert got[hubio.CKPT_FILE] is not None
    assert got[hubio.META_FILE] is None
