"""One-command onboarding: fetch the live coordinator config, build a personal data
shard, then train-and-submit rounds until interrupted.

    uvx --from git+https://github.com/commonsense-ai/coop \
        coop-join --hf-token hf_xxx
"""

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

import torch
import yaml

from coop.status import FILENAME as STATUS_FILENAME
from coop.status import StatusFile

log = logging.getLogger(__name__)

DEFAULT_REPO = "commonsense-ai/coop"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def fetch_raw(repo: str, path: str, dest: Path, ref: str = "main") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RAW.format(repo=repo, ref=ref, path=path), timeout=30) as r:
        dest.write_bytes(r.read())
    return dest


# roneneldan/TinyStories train rows; config data.train_docs overrides at runtime
TRAIN_DOCS = 2_119_719


def derive_skip(username: str, docs: int = 20000, total_docs: int = TRAIN_DOCS) -> int:
    """Deterministic per-user shard offset so volunteers train on (mostly) disjoint slices.
    Slots wrap at the dataset end so every username maps to a full shard inside it."""
    slots = max(1, total_docs // docs)
    h = int(hashlib.sha256(username.encode()).hexdigest(), 16)
    return (h % slots) * docs


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    from coop import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser(description="join the run: train locally, submit pseudo-gradients")
    ap.add_argument("--hf-token", default=None, help="HF write token (or set HF_TOKEN)")
    ap.add_argument(
        "--repo", default=DEFAULT_REPO, help="GitHub repo holding the coordinator config"
    )
    ap.add_argument("--workdir", default="~/.coop")
    ap.add_argument("--docs", type=int, default=20000, help="TinyStories docs in your shard")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--once", action="store_true", help="run a single round instead of looping")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N rounds (0 = endless)")
    ap.add_argument("--pause", type=int, default=60, help="seconds between rounds")
    a = ap.parse_args()
    if a.once:
        a.rounds = 1

    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token
    # imported after the token is in the env so hubio picks it up
    from coop import hubio
    from coop.data import fetch_tinystories, load_tokenizer, tokenize_file
    from coop.trainer import adaptive_h, run_worker, wait_for_new_step

    user = hubio.whoami()
    if user == "anonymous":
        raise SystemExit("no HF credentials: pass --hf-token or set HF_TOKEN (write scope)")

    work = Path(a.workdir).expanduser()
    cfg = yaml.safe_load(fetch_raw(a.repo, "config/run.yaml", work / "run.yaml").read_text())
    tok_path = fetch_raw(a.repo, cfg["data"]["tokenizer"], work / "tokenizer.json")

    status = StatusFile(work / STATUS_FILENAME)
    status.update(user=user, phase="starting", rounds_target=a.rounds or None)

    skip = derive_skip(user, docs=a.docs, total_docs=cfg["data"].get("train_docs", TRAIN_DOCS))
    shard = work / f"shard_{skip}_{a.docs}.bin"
    if not shard.exists():
        log.info("building your data shard (docs %d..%d) ...", skip, skip + a.docs)
        status.update(phase="building your data shard (one-time)")
        txt = fetch_tinystories(str(work / "shard.txt"), a.docs, skip)
        tokenize_file(load_tokenizer(str(tok_path)), txt, str(shard))

    device = a.device or pick_device()
    log.info("joined as %s on %s; ctrl-c to stop", user, device)
    status.update(device=device)
    rnd, h_next, tokens_session = 0, None, 0
    while True:
        try:
            t0 = time.time()
            _, meta_path = run_worker(
                cfg,
                str(shard),
                out_dir=str(work / "out"),
                device=device,
                seed=rnd,
                h_override=h_next,
                status=status,
            )
            rnd += 1
            meta = json.loads(meta_path.read_text())
            tokens_session += meta["tokens"]
            status.update(rounds_done=rnd, tokens_session=tokens_session)
            if a.rounds and rnd >= a.rounds:
                status.update(phase="done")
                break
            status.update(phase="waiting", waiting_past_step=meta["start_step"])
            wait_for_new_step(cfg["repos"]["model"], meta["start_step"], poll=a.pause)
            h_next = adaptive_h(meta, time.time() - t0, cfg["inner"])
            log.info("next round: %d inner steps", h_next)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if a.rounds == 1:
                raise  # single-round runs must fail loudly, not exit 0
            # unattended volunteers: transient network/HF errors retry, never crash
            log.warning("round failed (%s); retrying after pause", e)
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
