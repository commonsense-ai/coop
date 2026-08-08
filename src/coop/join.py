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
import signal
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import yaml

from coop import update
from coop.device import arch_gap, cpu_fallback, describe, kernel_missing, pick_device
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


def machine_seed(work: Path) -> int:
    """Stable per-machine seed base: a user's machines must train different batches,
    or their merged submissions carry duplicated signal."""
    p = work / "machine-id"
    if not p.exists():
        p.write_text(uuid.uuid4().hex)
    return int(p.read_text()[:8], 16)


def corpus_fingerprint(dcfg: dict) -> str:
    """Shard files are keyed by dataset+tokenizer identity: a run change must never
    silently reuse a shard tokenized for a different run."""
    ident = f"{dcfg.get('hf_dataset')}|{dcfg.get('hf_config')}|{dcfg.get('tokenizer')}"
    return hashlib.sha256(ident.encode()).hexdigest()[:8]


def config_changed(repo: str, work: Path, current: str, path: str = "config/run.yaml") -> bool:
    """True when the coordinator's run config differs from what this worker started with.
    Fetch failures count as unchanged: never restart on a network blip."""
    try:
        latest = fetch_raw(repo, path, work / "run.latest.yaml").read_text()
    except OSError:
        return False
    return latest != current


NOTED: set[str] = set()  # announced by this process already; `coop logs` is not a nag


def check_for_update(repo: str, work: Path, status: StatusFile) -> None:
    """Between rounds only: a restart here can never cost a volunteer trained work,
    and the same is not true a single step earlier."""
    manifest = update.available(repo, work)
    if not manifest:
        return
    version = manifest.get("version", "")
    auto = update.auto_enabled(work)
    status.update(update_available=version)
    if not auto or update.attempted(work, version):
        if not auto and version not in NOTED:
            NOTED.add(version)
            log.info("%s", update.notice(manifest))
        return
    log.info("coop %s is out — updating and restarting into it", version)
    status.update(phase=f"updating to coop {version}")
    update.mark_attempted(work, version)  # before the attempt: a bad release can't loop
    why = update.restart_into_update(repo, sys.argv[1:])
    log.warning("auto-update didn't take (%s) — carrying on with the version you have", why)
    status.update(phase="auto-update failed — still training")


def shard_progress(status: StatusFile, stage: str, unit: str):
    """Report the one-time shard build to both `coop logs` and `coop status` — it is the
    longest a volunteer ever waits with nothing to look at, and it looks hung."""
    from coop.data import Progress

    drawn = Progress(stage, unit=unit)
    t0, last = time.time(), 0.0

    def report(done: int, total: int) -> None:
        nonlocal last
        drawn(done, total)
        now = time.time()
        if now - last < 1.0 and done < total:
            return  # status.json is rewritten on every update: throttle to ~1 Hz
        last = now
        status.update(
            shard_stage=stage,
            shard_done=done,
            shard_total=total,
            shard_per_sec=round(done / max(now - t0, 1e-6), 1),
        )

    return report


def derive_skip(username: str, docs: int = 20000, total_docs: int = TRAIN_DOCS) -> int:
    """Deterministic per-user shard offset so volunteers train on (mostly) disjoint slices.
    Slots wrap at the dataset end so every username maps to a full shard inside it."""
    slots = max(1, total_docs // docs)
    h = int(hashlib.sha256(username.encode()).hexdigest(), 16)
    return (h % slots) * docs


def main():
    from coop import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser(description="join the run: train locally, submit pseudo-gradients")
    ap.add_argument("--hf-token", default=None, help="HF write token (or set HF_TOKEN)")
    ap.add_argument(
        "--repo", default=DEFAULT_REPO, help="GitHub repo holding the coordinator config"
    )
    ap.add_argument("--run-config", default="config/run.yaml", help="which run to train")
    ap.add_argument("--workdir", default="~/.coop")
    ap.add_argument("--docs", type=int, default=20000, help="TinyStories docs in your shard")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--once", action="store_true", help="run a single round instead of looping")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N rounds (0 = endless)")
    ap.add_argument("--pause", type=int, default=60, help="retry delay after a failed round")
    a = ap.parse_args()
    if a.once:
        a.rounds = 1

    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token
    # imported after the token is in the env so hubio picks it up
    from coop import hubio, submit
    from coop.data import fetch_docs, load_tokenizer, tokenize_file
    from coop.trainer import run_worker

    user = hubio.whoami()
    if user == "anonymous":
        raise SystemExit("no HF credentials: pass --hf-token or set HF_TOKEN (write scope)")

    work = Path(a.workdir).expanduser()
    cfg_text = fetch_raw(a.repo, a.run_config, work / "run.yaml").read_text()
    cfg = yaml.safe_load(cfg_text)
    tok_path = fetch_raw(a.repo, cfg["data"]["tokenizer"], work / "tokenizer.json")

    status = StatusFile(work / STATUS_FILENAME)
    status.update(user=user, phase="starting", rounds_target=a.rounds or None)

    skip = derive_skip(user, docs=a.docs, total_docs=cfg["data"].get("train_docs", TRAIN_DOCS))
    shard = work / f"shard_{corpus_fingerprint(cfg['data'])}_{skip}_{a.docs}.bin"
    if not shard.exists():
        log.info("building your data shard (docs %d..%d) ...", skip, skip + a.docs)
        status.update(phase="building your data shard (one-time)")
        txt = fetch_docs(
            str(work / "shard.txt"),
            a.docs,
            skip,
            dataset=cfg["data"].get("hf_dataset", "roneneldan/TinyStories"),
            text_field=cfg["data"].get("text_field", "text"),
            config=cfg["data"].get("hf_config"),
            progress=shard_progress(status, "downloading docs", "docs"),
        )
        tokenize_file(
            load_tokenizer(str(tok_path)),
            txt,
            str(shard),
            progress=shard_progress(status, "tokenizing", "bytes"),
        )

    device = a.device or pick_device()
    # pick_device already skips a card this torch can't launch on; an explicit
    # --device cuda would otherwise crash-loop a whole session away
    if device.startswith("cuda") and (gap := arch_gap()):
        device = cpu_fallback(gap)
    label = describe(device)
    log.info("joined as %s on %s; ctrl-c to stop", user, label)
    # the label too: `coop status` reports what the worker actually got, and only
    # the worker knows whether an explicit --device overrode detection
    status.update(device=device, device_label=label)

    # coop stop sends SIGTERM: finish packaging + submitting the current round, then exit
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    seed_base = machine_seed(work)
    acc = submit.StepAccumulator() if cfg["inner"].get("accumulate_rounds") else None
    out = work / "out"
    rnd, h_next, tokens_session = 0, None, 0
    while not stop.is_set():
        try:
            # rounds an outage parked (this process or an earlier one) go out first; the
            # accumulator's own parked copy is skipped because its next upload resends it
            submit.drain(cfg, out, skip=acc.pending if acc else None)
            _, meta_path = run_worker(
                cfg,
                str(shard),
                out_dir=str(out),
                device=device,
                seed=seed_base + rnd,
                h_override=h_next,
                status=status,
                stop=stop,
                username=user,
                accumulator=acc,
            )
            if meta_path is None:  # stopped before the round trained anything
                break
            rnd += 1
            meta = json.loads(meta_path.read_text())
            tokens_session += meta["tokens"]
            status.update(rounds_done=rnd, tokens_session=tokens_session)
            if a.rounds and rnd >= a.rounds:
                break
            if config_changed(a.repo, work, cfg_text, path=a.run_config):
                # a new run (or retuned config) shipped: re-exec picks up everything —
                # new model, tokenizer, shard, credit rules — with the same pid
                log.info("coordinator config changed — restarting to adopt the new run")
                status.update(phase="new run detected — restarting to join it")
                os.execv(sys.executable, [sys.executable, "-m", "coop.join", *sys.argv[1:]])
            check_for_update(a.repo, work, status)
            # back-to-back rounds; see trainer.main for why waiting is the only waste
            h_next = cfg["inner"].get("h_max", 500)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if a.rounds == 1:
                raise  # single-round runs must fail loudly, not exit 0
            if device.startswith("cuda") and kernel_missing(e):
                device = cpu_fallback()  # permanent: pausing and retrying changes nothing
                status.update(device=device, device_label=describe(device))
                continue
            # unattended volunteers: transient network/HF errors retry, never crash
            log.warning("round failed (%s); retrying after pause", e)
            time.sleep(a.pause)
    status.update(phase="stopped")
    log.info("worker stopped")


if __name__ == "__main__":
    main()
