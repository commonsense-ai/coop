"""Volunteer-facing CLI: `coop start` / `stop` / `status` / `logs`.

Wraps the coop-join worker in a managed background process so contributing is
start-and-forget. Worker state (pid, log, shard, config) lives under ~/.coop.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from coop import hubio
from coop.join import DEFAULT_REPO, fetch_raw

HOME = Path(os.environ.get("COOP_HOME", "~/.coop")).expanduser()
PIDFILE = HOME / "worker.pid"
LOGFILE = HOME / "worker.log"
BOARD = "https://github.com/{repo}/blob/ledger/LEADERBOARD.md"


def load_run_config(repo: str) -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    try:
        return yaml.safe_load(fetch_raw(repo, "config/run.yaml", HOME / "run.yaml").read_text())
    except OSError as e:
        raise SystemExit(f"could not fetch the run config from github.com/{repo}: {e}") from e


def model_from_words(words: list[str]) -> str | None:
    """`coop start training tinystories` -> "tinystories"; the noun is optional filler."""
    rest = [w for w in words if w not in ("training", "train")]
    if len(rest) > 1:
        raise SystemExit(f"too many arguments: {' '.join(words)}")
    return rest[0] if rest else None


def resolve_model(name: str | None, cfg: dict) -> str:
    repo = cfg["repos"]["model"]
    short = repo.split("/")[-1]
    if name in (None, repo, short, short.split("-")[0]):
        return repo
    raise SystemExit(f"unknown model {name!r}: this run trains {repo} (just say `coop start`)")


def read_pid() -> int | None:
    try:
        return int(PIDFILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return ""


def ensure_token(explicit: str | None) -> str:
    from huggingface_hub import login

    if explicit:
        login(token=explicit)
    user = hubio.whoami()
    if user == "anonymous" and sys.stdin.isatty():
        import getpass

        print("coop needs a free Hugging Face account to credit your work.")
        print("create a token with WRITE access at https://huggingface.co/settings/tokens")
        tok = getpass.getpass("paste your token (input stays hidden): ").strip()
        if tok:
            login(token=tok)  # persists in the HF cache: next time is zero-setup
            user = hubio.whoami()
    if user == "anonymous":
        raise SystemExit("no valid token — try again with `coop start --hf-token hf_...`")
    return user


def cmd_start(a: argparse.Namespace) -> None:
    cfg = load_run_config(a.repo)
    model_repo = resolve_model(a.model, cfg)
    pid = read_pid()
    if pid and alive(pid):
        print(f"already contributing (pid {pid}) — `coop status` to check on it")
        return
    user = ensure_token(a.hf_token)
    env = os.environ | {
        # daemon log: coop's own lines with timestamps, no progress bars
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "COOP_LOG_TS": "1",
    }
    cmd = [sys.executable, "-m", "coop.join", "--workdir", str(HOME), "--repo", a.repo]
    if a.device:
        cmd += ["--device", a.device]
    with LOGFILE.open("ab") as log:
        p = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    PIDFILE.write_text(str(p.pid))
    time.sleep(2)
    if not alive(p.pid):
        print(tail(LOGFILE, 15))
        raise SystemExit(f"the worker exited immediately — log above, full log: {LOGFILE}")
    print(f"training {model_repo} as {user} (pid {p.pid})")
    print("the first round downloads the model and builds your data shard (a few minutes)")
    print("  coop status    how it's going")
    print("  coop logs -f   watch it work")
    print("  coop stop      stop contributing")


def cmd_stop(_a) -> None:
    pid = read_pid()
    if pid is None or not alive(pid):
        PIDFILE.unlink(missing_ok=True)
        print("no worker running")
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not alive(pid):
            break
        time.sleep(0.25)
    if alive(pid):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    PIDFILE.unlink(missing_ok=True)
    print("stopped — credited work stays on the leaderboard; `coop start` to resume")


def cmd_status(a: argparse.Namespace) -> None:
    cfg = load_run_config(a.repo)
    model_repo = cfg["repos"]["model"]
    pid = read_pid()
    if pid and alive(pid):
        mins = int((time.time() - PIDFILE.stat().st_mtime) / 60)
        print(f"worker   running (pid {pid}, up {mins // 60}h{mins % 60:02d}m)")
    else:
        print("worker   not running — `coop start` to contribute")
    try:
        meta = json.loads(Path(hubio.download_file(model_repo, "meta.json")).read_text())
        val = meta.get("eval", {}).get("val_loss")
        suffix = f" (val loss {val})" if val is not None else ""
        print(f"model    {model_repo} @ outer step {meta['step']}{suffix}")
    except Exception:
        print(f"model    {model_repo} (couldn't reach huggingface.co)")
    last = tail(LOGFILE, 1).strip()
    if last:
        print(f"last     {last}")
    print(f"board    {BOARD.format(repo=a.repo)}")
    print(f"log      {LOGFILE}")


def cmd_logs(a: argparse.Namespace) -> None:
    if not LOGFILE.exists():
        raise SystemExit(f"no log yet at {LOGFILE} — has the worker ever started?")
    print(tail(LOGFILE, a.lines))
    if not a.follow:
        return
    with LOGFILE.open("r", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = f.read()
                if chunk:
                    print(chunk, end="", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="coop", description="contribute your computer to a community-trained model"
    )
    ap.add_argument("--repo", default=DEFAULT_REPO, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")

    st = sub.add_parser("start", help="start contributing in the background")
    st.add_argument("words", nargs="*", metavar="[training] [model]")
    st.add_argument("--hf-token", default=None, help="Hugging Face write token (first run only)")
    st.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")

    sub.add_parser("stop", help="stop contributing")
    sub.add_parser("status", help="worker and model state")
    lg = sub.add_parser("logs", help="show what the worker is doing")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("-n", "--lines", type=int, default=40)

    a = ap.parse_args()
    if a.cmd == "start":
        a.model = model_from_words(a.words)
        cmd_start(a)
    elif a.cmd == "stop":
        cmd_stop(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "logs":
        cmd_logs(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
