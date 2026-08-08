"""Thin huggingface_hub wrappers. Every network call the project makes lives here."""

import functools
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file, save_file

CKPT_FILE = "checkpoint.safetensors"
OPT_FILE = "optimizer.safetensors"
META_FILE = "meta.json"


def _reset_xet_session() -> None:
    """A failed transfer latches its error onto huggingface_hub's process-global
    XetSession: every later transfer replays it ("Previous task error: ..."), uploads and
    downloads alike, and the latch never clears on its own. hub resets the session on only
    some failure paths, so one 403 — the aggregator closing our PR mid-round is enough —
    can strand an unattended worker for rounds, retrying a stale error it can't outlive."""
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception:
        pass  # nothing to reset (no xet session, or a hub without one)


def _transfer(fn):
    """Wraps the calls that move file bytes: a failure must hand the next one a live hub."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException:
            _reset_xet_session()
            raise

    return wrapper


def token() -> str | None:
    return os.environ.get("HF_TOKEN")


def api() -> HfApi:
    return HfApi(token=token())


def dataset_fs():
    """Filesystem view of the hub: lets the shard builder range-read inside a dataset's
    parquet files instead of streaming every row before the one it wants."""
    from huggingface_hub import HfFileSystem

    return HfFileSystem(token=token())


def whoami() -> str:
    try:
        return api().whoami()["name"]
    except Exception:
        return os.environ.get("HF_USERNAME", "anonymous")


def ensure_repos(model_repo: str, dataset_repo: str) -> None:
    a = api()
    a.create_repo(model_repo, repo_type="model", exist_ok=True)
    a.create_repo(dataset_repo, repo_type="dataset", exist_ok=True)


def resolve_revision(repo_id: str, revision: str = "main", repo_type: str = "model") -> str:
    """Pin a branch name to a commit sha so multi-file reads are atomic: a tick
    landing between two downloads would otherwise hand back mixed-step state."""
    return api().repo_info(repo_id, revision=revision, repo_type=repo_type).sha


@_transfer
def download_checkpoint(model_repo: str, revision: str = "main") -> tuple[dict, dict]:
    if revision == "main":
        revision = resolve_revision(model_repo)
    ckpt = hf_hub_download(model_repo, CKPT_FILE, revision=revision, token=token())
    meta = hf_hub_download(model_repo, META_FILE, revision=revision, token=token())
    return load_file(ckpt), json.loads(Path(meta).read_text())


@_transfer
def get_step(model_repo: str, revision: str = "main") -> int:
    """Current outer step without pulling the 59MB checkpoint — meta.json only."""
    path = hf_hub_download(model_repo, META_FILE, revision=revision, token=token())
    return json.loads(Path(path).read_text())["step"]


@_transfer
def download_file(
    repo_id: str, filename: str, repo_type: str = "model", revision: str = "main"
) -> str:
    return hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision, token=token())


@_transfer
def download_optimizer(model_repo: str, revision: str = "main") -> dict | None:
    try:
        path = hf_hub_download(model_repo, OPT_FILE, revision=revision, token=token())
    except EntryNotFoundError:
        return None
    return load_file(path)


@_transfer
def upload_checkpoint(model_repo: str, state_dict: dict, meta: dict, opt_state=None) -> None:
    # One commit for all files: a tick costs a single write call, not one per file.
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / CKPT_FILE
        save_file(state_dict, str(ckpt))
        ops = [CommitOperationAdd(CKPT_FILE, str(ckpt))]
        if opt_state is not None:
            opt = Path(td) / OPT_FILE
            save_file(opt_state, str(opt))
            ops.append(CommitOperationAdd(OPT_FILE, str(opt)))
        ops.append(CommitOperationAdd(META_FILE, json.dumps(meta, indent=2).encode()))
        api().create_commit(
            repo_id=model_repo,
            operations=ops,
            commit_message=f"step {meta.get('step')}",
        )


def list_open_prs(dataset_repo: str) -> list:
    discussions = api().get_repo_discussions(
        repo_id=dataset_repo,
        repo_type="dataset",
        discussion_type="pull_request",
        discussion_status="open",
    )
    return [d for d in discussions if d.is_pull_request and d.status == "open"]


def list_repo_files(repo_id: str, repo_type: str = "dataset", revision: str = "main") -> list[str]:
    return api().list_repo_files(repo_id, repo_type=repo_type, revision=revision)


@_transfer
def download_pr_files(
    dataset_repo: str, pr_num: int, base_files: set[str] | None = None
) -> dict[str, str]:
    """New submissions/* paths a PR adds -> local file paths."""
    rev = f"refs/pr/{pr_num}"
    files = api().list_repo_files(dataset_repo, repo_type="dataset", revision=rev)
    base = base_files or set()
    new = [f for f in files if f.startswith("submissions/") and f not in base]
    return {
        f: hf_hub_download(dataset_repo, f, repo_type="dataset", revision=rev, token=token())
        for f in new
    }


@_transfer
def open_pr(dataset_repo: str, operations: list[CommitOperationAdd], message: str):
    return api().create_commit(
        repo_id=dataset_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=message,
        create_pr=True,
    )


@_transfer
def update_pr(dataset_repo: str, pr_num: int, operations: list[CommitOperationAdd], message: str):
    """Replace files on an existing open PR (raises if the PR was closed meanwhile)."""
    return api().create_commit(
        repo_id=dataset_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=message,
        revision=f"refs/pr/{pr_num}",
    )


def merge_or_close_pr(
    dataset_repo: str, pr_num: int, merge: bool = False, comment: str | None = None
) -> None:
    # Default is close-without-merge: merged files would count against the dataset
    # repo's 100k-file cap forever, and the delta has already been downloaded.
    a = api()
    if comment:
        a.comment_discussion(dataset_repo, pr_num, comment=comment, repo_type="dataset")
    if merge:
        a.merge_pull_request(dataset_repo, pr_num, repo_type="dataset")
    else:
        a.change_discussion_status(dataset_repo, pr_num, new_status="closed", repo_type="dataset")
