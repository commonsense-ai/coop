"""Volunteer-facing CLI: `coop start` / `stop` / `status` / `logs`.

Wraps the coop-join worker in a managed background process so contributing is
start-and-forget. Worker state (pid, log, shard, config) lives under ~/.coop.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from coop import hubio
from coop.join import DEFAULT_REPO, fetch_raw, pick_device
from coop.status import FILENAME as STATUS_FILENAME
from coop.status import read_status

HOME = Path(os.environ.get("COOP_HOME", "~/.coop")).expanduser()
PIDFILE = HOME / "worker.pid"
LOGFILE = HOME / "worker.log"
BOARD = "https://github.com/{repo}/blob/ledger/LEADERBOARD.md"

WELCOME = """\
coop — help train a small language model with your computer

  coop start     begin contributing (runs in the background)
  coop status    live progress, your rank, who else is training
  coop logs -f   watch the worker do its thing
  coop stop      stop contributing — your credit stays

first time? just run `coop start` — it walks you through the one-time setup."""

ONBOARDING = """\
welcome! here's what happens when you contribute:

  * your computer downloads the current model (~60 MB) and trains it on a
    slice of short stories, a few minutes at a time
  * every finished round is submitted under your Hugging Face name and
    merged into the shared model — you earn credit on the public leaderboard
  * stop whenever you like with `coop stop`; earned credit is never lost

one-time setup — coop needs a free Hugging Face account:

  1. create an account   https://huggingface.co/join
  2. create a token      https://huggingface.co/settings/tokens  (type: Write)
  3. paste the token below (input stays hidden)
"""

DEVICE_NAMES = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}


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


TS = re.compile(r"^\d\d-\d\d \d\d:\d\d:\d\d ")
# leaderboard row: | rank | user | tier | accepted | tokens | reputation | score |
BOARD_ROW = re.compile(r"^\| (\d+) \| (\S+) \| \S+ \| \d+ \| ([\d,]+) \| ")
TOKEN_TARGET = 300_000_000  # Chinchilla-optimal for a ~15M-param model


def fmt_eta(secs: float) -> str:
    s = max(0, int(secs))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def parse_board(md: str) -> list[dict]:
    rows = []
    for ln in md.splitlines():
        m = BOARD_ROW.match(ln)
        if m:
            rows.append({"rank": int(m[1]), "user": m[2], "tokens": int(m[3].replace(",", ""))})
    return rows


def now_line(st: dict) -> str:
    phase = st.get("phase", "")
    if phase == "training" and st.get("h_steps"):
        i, h = st.get("inner_step", 0), st["h_steps"]
        rate = st.get("steps_per_sec") or 0
        eta = f" · ~{fmt_eta((h - i) / rate)} left" if rate else ""
        return f"training — inner step {i}/{h} · loss {st.get('loss', '?')}{eta}"
    if phase == "waiting":
        step = st.get("waiting_past_step")
        which = f"outer step {step + 1}" if isinstance(step, int) else "the next outer step"
        return f"submitted — waiting for {which} (your work merges at the next aggregator tick)"
    return phase


def last_activity(path: Path) -> str:
    """Newest coop log line — stderr noise (tracebacks, warnings) shares the file."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return next((ln for ln in reversed(lines) if TS.match(ln)), "")


def ensure_token(explicit: str | None) -> str:
    from huggingface_hub import login

    if explicit:
        login(token=explicit)
    user = hubio.whoami()
    if user != "anonymous":
        return user
    if not sys.stdin.isatty():
        raise SystemExit("no HF token found — run `coop start --hf-token hf_...`")

    import getpass

    print(ONBOARDING)
    for _ in range(3):
        tok = getpass.getpass("  token: ").strip()
        if not tok:
            continue
        try:
            login(token=tok)  # persists in the HF cache: next time is zero-setup
        except Exception:
            print("  Hugging Face rejected that token — check it's a Write token and try again")
            continue
        user = hubio.whoami()
        if user != "anonymous":
            print(f"  you're in, {user}!\n")
            return user
    raise SystemExit("couldn't validate a token — try `coop start --hf-token hf_...`")


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
    if a.rounds:
        cmd += ["--rounds", str(a.rounds)]
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
    device = a.device or pick_device()
    hw = DEVICE_NAMES.get(device, device)
    print(f"training {model_repo} as {user} on your {hw} (pid {p.pid})")
    if a.rounds:
        print(f"will stop by itself after {a.rounds} round{'s' if a.rounds > 1 else ''}")
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
    running = bool(pid and alive(pid))
    st = read_status(HOME / STATUS_FILENAME)

    if running:
        mins = int((time.time() - PIDFILE.stat().st_mtime) / 60)
        rnd = st.get("rounds_done", 0) + 1
        target = st.get("rounds_target")
        mode = f"round {rnd} of {target}" if target else f"endless · round {rnd}"
        print(f"worker   running (pid {pid}, up {mins // 60}h{mins % 60:02d}m) — {mode}")
        line = now_line(st)
        if line:
            age = time.time() - st.get("updated_at", time.time())
            stale = f" (no update for {fmt_eta(age)} — check `coop logs`)" if age > 300 else ""
            print(f"now      {line}{stale}")
    else:
        print("worker   not running — `coop start` to contribute")
    if st.get("rounds_done"):
        when = "this session" if running else "last session"
        toks = st.get("tokens_session", 0)
        print(f"session  {st['rounds_done']} rounds · {toks:,} tokens trained {when}")

    try:
        meta = json.loads(Path(hubio.download_file(model_repo, "meta.json")).read_text())
        val = meta.get("eval", {}).get("val_loss")
        suffix = f" (val loss {val})" if val is not None else ""
        print(f"model    {model_repo} @ outer step {meta['step']}{suffix}")
    except Exception:
        print(f"model    {model_repo} (couldn't reach huggingface.co)")

    user = st.get("user") or hubio.whoami()
    try:
        md = fetch_raw(a.repo, "LEADERBOARD.md", HOME / "board.md", ref="ledger").read_text()
        rows = parse_board(md)
        mine = next((r for r in rows if r["user"] == user), None)
        if mine:
            you = f"{user} — {mine['tokens']:,} tokens credited"
            print(f"you      {you} · rank {mine['rank']} of {len(rows)}")
        total = sum(r["tokens"] for r in rows)
        pct = 100 * total / TOKEN_TARGET
        print(f"goal     {total:,} of ~{TOKEN_TARGET:,} community tokens ({pct:.1f}%)")
    except OSError:
        pass
    try:
        by: dict[str, int] = {}
        for p in hubio.list_open_prs(cfg["repos"]["dataset"]):
            by[p.author] = by.get(p.author, 0) + 1
        if by:
            names = " · ".join(
                f"{u} ×{n}" + (" (you)" if u == user else "")
                for u, n in sorted(by.items(), key=lambda kv: -kv[1])
            )
            print(f"inbox    {sum(by.values())} submission(s) awaiting the next tick: {names}")
        else:
            print("inbox    empty — all submitted work has been aggregated")
    except Exception:
        pass
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
    st.add_argument(
        "--rounds", type=int, default=0, help="stop after N rounds (default: run until coop stop)"
    )

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
        print(WELCOME)


if __name__ == "__main__":
    main()
