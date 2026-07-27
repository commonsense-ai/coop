"""One-command onboarding: fetch the live coordinator config, build a personal data
shard, then train-and-submit rounds until interrupted.

    uvx --from git+https://github.com/commonsense-ai/decentralised-ai-training-poc \
        coop-join --hf-token hf_xxx
"""

import argparse
import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path

import torch
import yaml

log = logging.getLogger(__name__)

DEFAULT_REPO = "commonsense-ai/decentralised-ai-training-poc"
RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"


def fetch_raw(repo: str, path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RAW.format(repo=repo, path=path), timeout=30) as r:
        dest.write_bytes(r.read())
    return dest


def derive_skip(username: str, slots: int = 500, docs: int = 20000) -> int:
    """Deterministic per-user shard offset so volunteers train on (mostly) disjoint slices."""
    h = int(hashlib.sha256(username.encode()).hexdigest(), 16)
    return (h % slots) * docs


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="join the run: train locally, submit pseudo-gradients")
    ap.add_argument("--hf-token", default=None, help="HF write token (or set HF_TOKEN)")
    ap.add_argument(
        "--repo", default=DEFAULT_REPO, help="GitHub repo holding the coordinator config"
    )
    ap.add_argument("--workdir", default="~/.coop")
    ap.add_argument("--docs", type=int, default=20000, help="TinyStories docs in your shard")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--once", action="store_true", help="run a single round instead of looping")
    ap.add_argument("--pause", type=int, default=60, help="seconds between rounds")
    a = ap.parse_args()

    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token
    # imported after the token is in the env so hubio picks it up
    from coop import hubio
    from coop.data import fetch_tinystories, load_tokenizer, tokenize_file
    from coop.trainer import run_worker

    user = hubio.whoami()
    if user == "anonymous":
        raise SystemExit("no HF credentials: pass --hf-token or set HF_TOKEN (write scope)")

    work = Path(a.workdir).expanduser()
    cfg = yaml.safe_load(fetch_raw(a.repo, "config/run.yaml", work / "run.yaml").read_text())
    tok_path = fetch_raw(a.repo, cfg["data"]["tokenizer"], work / "tokenizer.json")

    skip = derive_skip(user, docs=a.docs)
    shard = work / f"shard_{skip}_{a.docs}.bin"
    if not shard.exists():
        log.info("building your data shard (docs %d..%d) ...", skip, skip + a.docs)
        txt = fetch_tinystories(str(work / "shard.txt"), a.docs, skip)
        tokenize_file(load_tokenizer(str(tok_path)), txt, str(shard))

    device = a.device or pick_device()
    log.info("joined as %s on %s; ctrl-c to stop", user, device)
    rnd = 0
    while True:
        try:
            run_worker(cfg, str(shard), out_dir=str(work / "out"), device=device, seed=rnd)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # unattended volunteers: transient network/HF errors retry, never crash
            log.warning("round failed (%s); retrying after pause", e)
        rnd += 1
        if a.once:
            break
        time.sleep(a.pause)


if __name__ == "__main__":
    main()
