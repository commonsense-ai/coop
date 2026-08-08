"""Volunteer-facing CLI: `coop start` / `progress` / `stop` / `status` / `logs`.

Wraps the coop-join worker in a managed background process so contributing is
start-and-forget. Worker state (pid, log, shard, config) lives under ~/.coop.
"""

import argparse
import functools
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from coop import __version__, hubio, progress, settings, submit, trend, update
from coop.device import cuda_gap, describe, pick_device
from coop.join import DEFAULT_REPO, fetch_raw
from coop.progress import bar, fmt_eta, now_line
from coop.status import FILENAME as STATUS_FILENAME
from coop.status import read_status

HOME = Path(os.environ.get("COOP_HOME", "~/.coop")).expanduser()
PIDFILE = HOME / "worker.pid"
LOGFILE = HOME / "worker.log"
BOARD = "https://github.com/{repo}/blob/ledger/LEADERBOARD.md"

WELCOME = """\
coop — help train a small language model with your computer

  coop start        begin contributing (runs in the background)
  coop progress     live progress bars — arrows switch the view, or stop the worker
  coop status       one printout: your rank, the model, who else is training
  coop logs -f      watch the worker do its thing
  coop stop         stop contributing — your credit stays
  coop run latest   talk to the model trained so far (no account needed)
  coop update       get the newest coop (`--auto on` to keep it current)

first time? just run `coop start` — it walks you through the one-time setup.
in a hurry? `coop start --latest` skips the menu and joins the current run."""

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


def warn_cuda_gap(device: str) -> None:
    """An NVIDIA owner training on CPU is losing most of their machine and has no
    way to know: torch reports no CUDA and nothing else says a word."""
    if device != "cpu":  # cuda already has it; Apple hardware never will
        return
    gap = cuda_gap()
    if gap:
        print(f"heads up: you have an NVIDIA GPU but coop is on your CPU — {gap.reason}")
        if gap.fix:
            print(f"          {gap.fix}")


def read_settings() -> dict:
    return settings.read(HOME)


def write_settings(**kw) -> None:
    settings.write(HOME, **kw)


def update_line(repo: str, force: bool = False) -> str:
    """One line about a newer coop, or nothing at all — a version check must never be
    the reason a volunteer can't read their own status."""
    manifest = update.available(repo, HOME, force=force)
    if not manifest:
        return ""
    return update.notice(manifest, auto=bool(read_settings().get("auto_update")))


def run_config_path() -> str:
    return read_settings().get("run_config", "config/run.yaml")


def load_run_config(repo: str, path: str | None = None, dest: str = "run.yaml") -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    path = path or run_config_path()
    try:
        return yaml.safe_load(fetch_raw(repo, path, HOME / dest).read_text())
    except OSError as e:
        raise SystemExit(f"could not fetch the run config from github.com/{repo}: {e}") from e


def load_runs(repo: str) -> list[dict]:
    """The run menu. Repos without a registry present their single run.yaml."""
    try:
        raw = fetch_raw(repo, "config/runs.yaml", HOME / "runs.yaml").read_text()
        return yaml.safe_load(raw)["runs"]
    except Exception:
        return [{"name": None, "config": "config/run.yaml", "status": "live", "blurb": ""}]


def choose_run(runs: list[dict], name: str | None, stored: str | None, interactive: bool):
    """Explicit name > remembered setting > picker (None) > the live run."""
    live = [r for r in runs if r.get("status") == "live"]
    if name:
        for r in runs:
            rn = r.get("name") or ""
            if name in (rn, rn.split("-")[0]):
                if r not in live:
                    raise SystemExit(f"{rn} is complete — it no longer accepts training")
                return r
        known = ", ".join(str(r.get("name")) for r in runs)
        raise SystemExit(f"unknown model {name!r} — available: {known}")
    if stored:
        for r in live:
            if r["config"] == stored:
                return r
    if interactive and len(runs) > 1:
        return None  # caller shows the arrow-key picker
    if not live:
        raise SystemExit("no run is accepting training right now — check the repo")
    return live[0]


def pick(rows: list[str], enabled: list[bool]) -> int:
    """Arrow-key picker (↑/↓ or j/k, enter). Falls back to a numbered prompt on
    terminals without raw mode."""
    idx = enabled.index(True)
    n = len(rows)

    def render(first: bool = False) -> None:
        if not first:
            sys.stdout.write(f"\x1b[{n}F")
        for i, r in enumerate(rows):
            line = f"  {'>' if i == idx else ' '} {r}"
            sys.stdout.write("\x1b[2K" + (f"\x1b[7m{line}\x1b[0m" if i == idx else line) + "\n")
        sys.stdout.flush()

    try:
        import termios
        import tty

        render(first=True)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x1b" and sys.stdin.read(1) == "[":
                    ch = {"A": "k", "B": "j"}.get(sys.stdin.read(1), "")
                if ch == "k":
                    idx = (idx - 1) % n
                elif ch == "j":
                    idx = (idx + 1) % n
                elif ch in ("\r", "\n") and enabled[idx]:
                    return idx
                elif ch == "\x03":
                    raise KeyboardInterrupt
                render()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, OSError):
        for i, r in enumerate(rows):
            print(f"  {i + 1}. {r}")
        while True:
            choice = input("pick a number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= n and enabled[int(choice) - 1]:
                return int(choice) - 1


def model_from_words(words: list[str]) -> str | None:
    """`coop start training tinystories` -> "tinystories"; the noun is optional filler."""
    rest = [w for w in words if w not in ("training", "train")]
    if len(rest) > 1:
        raise SystemExit(f"too many arguments: {' '.join(words)}")
    return rest[0] if rest else None


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
# leaderboard row: | rank | user | hardware | accepted | tokens | reputation | score |
# the hardware cell is anything-but-a-pipe: older clients matched `\S+` there, which is
# why the board still writes it without spaces
BOARD_ROW = re.compile(r"^\| (\d+) \| (\S+) \| [^|]+ \| \d+ \| ([\d,]+) \| ")
TOKEN_TARGET = 300_000_000  # fallback when the run config doesn't declare goal_tokens


def goal_tokens(cfg: dict) -> int:
    return int(cfg.get("goal_tokens", TOKEN_TARGET))


def goal_note(cfg: dict) -> str:
    """Why the goal is where it is. Empty when the run config doesn't say."""
    return str(cfg.get("goal_note") or "").strip()


def parse_board(md: str) -> list[dict]:
    rows = []
    for ln in md.splitlines():
        m = BOARD_ROW.match(ln)
        if m:
            rows.append({"rank": int(m[1]), "user": m[2], "tokens": int(m[3].replace(",", ""))})
    return rows


def fetch_trend(repo: str, cfg: dict) -> dict | None:
    """The val-loss series from the `ledger` branch, summarized. One number cannot say
    whether the loss is going down, so every loss line here is read off the series.
    Missing file = a run that has not evaluated yet; never fatal."""
    try:
        path = fetch_raw(repo, f"ledger/{trend.HISTORY}", HOME / trend.HISTORY, ref="ledger")
        return trend.summarize(trend.load(path), trend.spec(cfg))
    except (OSError, KeyError, ValueError):
        return None


def last_activity(path: Path) -> str:
    """Newest coop log line — stderr noise (tracebacks, warnings) shares the file."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return next((ln for ln in reversed(lines) if TS.match(ln)), "")


def pending_rounds() -> int:
    """Finished rounds an outage left parked on disk; the worker resends them itself."""
    return submit.pending_count(HOME / "out")


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
    pid = read_pid()
    if pid and alive(pid):
        print(f"already contributing (pid {pid}) — `coop status` to check on it")
        return
    runs = load_runs(a.repo)
    # --latest is the picker's opposite: no menu, no remembered setting, just the run
    # the co-op is training right now (the registry lists live runs newest first)
    stored = None if a.choose or a.latest else read_settings().get("run_config")
    sel = choose_run(runs, a.model, stored, interactive=sys.stdin.isatty() and not a.latest)
    if sel is None:
        print("which model do you want to train?  (arrows + enter)\n")
        live = [r.get("status") == "live" for r in runs]
        rows = [
            f"{r.get('name') or 'the current run':<18}"
            f"{r.get('blurb', '')}{'' if ok else '  [complete]'}"
            for r, ok in zip(runs, live)
        ]
        sel = runs[pick(rows, live)]
        print()
    write_settings(run_config=sel["config"], run_name=sel.get("name"))
    cfg = load_run_config(a.repo, sel["config"])
    model_repo = cfg["repos"]["model"]
    user = ensure_token(a.hf_token)
    env = os.environ | {
        # daemon log: coop's own lines with timestamps, no progress bars
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
        "COOP_LOG_TS": "1",
    }
    cmd = [sys.executable, "-m", "coop.join", "--workdir", str(HOME), "--repo", a.repo]
    cmd += ["--run-config", sel["config"]]
    if a.device:
        cmd += ["--device", a.device]
    if a.rounds:
        cmd += ["--rounds", str(a.rounds)]
    device = a.device or pick_device()
    warn_cuda_gap(device)  # before the worker starts: an idle GPU is worth fixing first
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
    print(f"training {model_repo} as {user} on your {describe(device)} (pid {p.pid})")
    if a.rounds:
        print(f"will stop by itself after {a.rounds} round{'s' if a.rounds > 1 else ''}")
    note = update_line(a.repo)
    if note:
        print(f"\n{note}")
    if show_progress(a):
        print("\nthe first round downloads the model and builds your data shard — both")
        print("have a bar below, so you can watch the one-time prep move.\n")
        watch(a)
        return
    print("(switch models any time: coop stop, then coop start --choose)")
    print("the first round downloads the model and builds your data shard — `coop progress`")
    print("shows a bar for both, so you can see the one-time prep move")
    print("  coop progress  live bars, and stopping without leaving them")
    print("  coop logs -f   watch it work")
    print("  coop stop      stop contributing")


def farewell(a: argparse.Namespace, st: dict) -> None:
    print("\nstopped — thanks for training with the co-op!")
    toks = st.get("tokens_session", 0)
    if toks:
        rounds = st.get("rounds_done", 0)
        print(f"\nyour session: {rounds} round{'s' if rounds != 1 else ''} · {toks:,} tokens,")
        print("all submitted to Hugging Face — they merge in at the next aggregation tick")
        if st.get("last_pr"):
            print(f"  {st['last_pr']}")
    n = pending_rounds()
    if n:
        s = "s" if n != 1 else ""
        print(f"\n{n} finished round{s} couldn't reach Hugging Face — saved on your machine.")
        print("the next `coop start` sends them; nothing is lost.")
    try:
        md = fetch_raw(a.repo, "LEADERBOARD.md", HOME / "board.md", ref="ledger").read_text()
        rows = parse_board(md)
        total = sum(r["tokens"] for r in rows)
        try:
            cfg = load_run_config(a.repo)
        except (SystemExit, Exception):  # goal display must never hide the bars
            cfg = {}
        target = goal_tokens(cfg)
        print(f"\ncommunity progress toward a fully trained model (~{target:,} tokens)")
        print(f"  {bar(total / target)} {100 * total / target:.1f}%")
        if why := goal_note(cfg):
            print(f"  {why}")
        user = st.get("user") or hubio.whoami()
        mine = next((r for r in rows if r["user"] == user), None)
        if mine and total:
            share = mine["tokens"] / total
            print("\nyour share of everything trained so far")
            detail = f"{100 * share:.0f}% · {mine['tokens']:,} tokens · rank {mine['rank']}"
            print(f"  {bar(share)} {detail}")
    except OSError:
        pass
    try:
        cfg = load_run_config(a.repo)
        meta = json.loads(Path(hubio.download_file(cfg["repos"]["model"], "meta.json")).read_text())
        val = meta.get("eval", {}).get("val_loss")
        if val is not None:
            start = trend.chance_loss(cfg["model"]["vocab_size"])
            print(f"\nmodel quality: val loss {val} — started at {start:.2f}, lower is better")
            tr = fetch_trend(a.repo, cfg)
            if tr:
                print(f"  {trend.headline(tr)}")
    except Exception:
        pass
    print("\nresume any time: coop start")


def cmd_stop(a: argparse.Namespace) -> None:
    pid = read_pid()
    if pid is None or not alive(pid):
        PIDFILE.unlink(missing_ok=True)
        print("no worker running")
        return
    st = read_status(HOME / STATUS_FILENAME)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGTERM)
    busy = st.get("phase") in ("training", "submitting")
    if busy:
        print("stopping — the worker is submitting your work in progress to Hugging Face first")
    grace = time.time() + (180 if busy else 15)
    shown = ""
    while alive(pid) and time.time() < grace:
        phase = read_status(HOME / STATUS_FILENAME).get("phase", "")
        if phase and phase != shown:
            print(f"  {phase} ...")
            shown = phase
        time.sleep(0.5)
    if alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        print("  the worker didn't finish in time — force-stopped, in-progress round lost")
    PIDFILE.unlink(missing_ok=True)
    farewell(a, read_status(HOME / STATUS_FILENAME))


@functools.cache
def idle_device_label() -> str:
    """What `coop start` would pick. Cached: the view redraws twice a second and
    detection shells out to nvidia-smi."""
    return describe(pick_device())


def probe_local(a: argparse.Namespace) -> dict:
    """Everything the view can know without a network call. Runs on every redraw,
    so it reads small local files and nothing else."""
    pid = read_pid()
    running = bool(pid and alive(pid))
    st = read_status(HOME / STATUS_FILENAME)
    pending = pending_rounds()
    return {
        "st": st,
        "running": running,
        "pid": pid,
        "uptime": time.time() - PIDFILE.stat().st_mtime if running else None,
        "stale_for": time.time() - st["updated_at"] if running and st.get("updated_at") else 0,
        "device_label": (running and st.get("device_label")) or idle_device_label(),
        "pending": pending,
        "pending_paused": running and pending >= submit.PENDING_MAX,
        "logfile": str(LOGFILE),
        "board_url": BOARD.format(repo=a.repo),
        "run_name": read_settings().get("run_name"),
    }


def probe_remote(a: argparse.Namespace) -> dict:
    """The board, the model and the inbox. Called on the view's refresh thread, so
    every piece degrades on its own: offline means missing rows, not a dead screen."""
    ctx: dict = {}
    try:
        # own cache file — ~/.coop/run.yaml belongs to the worker, and fetch_raw overwrites
        cfg = load_run_config(a.repo, dest="run.view.yaml")
    except SystemExit:
        # SystemExit is not an Exception, so it would escape the refresh thread and kill
        # it outright — offline must cost the remote rows, never the screen
        return ctx
    ctx["model_repo"] = cfg["repos"]["model"]
    ctx["goal"] = goal_tokens(cfg)
    ctx["goal_note"] = goal_note(cfg)
    user = read_status(HOME / STATUS_FILENAME).get("user") or hubio.whoami()
    ctx["user"] = user
    try:
        meta = json.loads(Path(hubio.download_file(ctx["model_repo"], "meta.json")).read_text())
        ctx["outer_step"] = meta["step"]
        ctx["val_loss"] = meta.get("eval", {}).get("val_loss")
    except Exception:
        pass
    ctx["trend"] = fetch_trend(a.repo, cfg)
    try:
        rows = parse_board(
            fetch_raw(a.repo, "LEADERBOARD.md", HOME / "board.md", ref="ledger").read_text()
        )
        ctx["total_tokens"] = sum(r["tokens"] for r in rows)
        mine = next((r for r in rows if r["user"] == user), None)
        if mine:
            ctx |= {"rank": mine["rank"], "of": len(rows), "my_tokens": mine["tokens"]}
    except OSError:
        pass
    try:
        by: dict[str, int] = {}
        for p in hubio.list_open_prs(cfg["repos"]["dataset"]):
            by[p.author] = by.get(p.author, 0) + 1
        names = " · ".join(
            f"{u} ×{n}" + (" (you)" if u == user else "")
            for u, n in sorted(by.items(), key=lambda kv: -kv[1])
        )
        ctx["inbox"] = (
            f"{sum(by.values())} submission(s) awaiting the next tick: {names}"
            if by
            else "empty — all submitted work has been aggregated"
        )
    except Exception:
        pass
    ctx["update_note"] = update_line(a.repo)
    return ctx


def show_progress(a: argparse.Namespace) -> bool:
    """The live view is the default face of a running worker. `--no-progress` skips it
    once, `coop progress --auto off` for good; a pipe never gets it at all."""
    if getattr(a, "no_progress", False):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return bool(read_settings().get("progress_auto", True))


def watch(a: argparse.Namespace, mode: str = progress.SIMPLE, once: bool = False) -> None:
    """The view decides, this acts. Looping rather than recursing so a volunteer can
    stop and start all evening without stacking frames."""
    while True:
        action = progress.view(
            probe=lambda: probe_local(a),
            remote=lambda: probe_remote(a),
            mode=mode,
            once=once,
        )
        if action == progress.STOP:
            cmd_stop(a)
            return
        if action != progress.START:
            return
        # the same start a volunteer would type: remembered run, picker if there is none
        cmd_start(
            argparse.Namespace(
                repo=a.repo,
                model=None,
                hf_token=None,
                choose=False,
                latest=False,
                device=None,
                rounds=0,
                no_progress=True,
            )
        )


def cmd_progress(a: argparse.Namespace) -> None:
    if a.auto:
        on = a.auto == "on"
        write_settings(progress_auto=on)
        print(
            "`coop start` opens the live view — `coop progress` reopens it any time"
            if on
            else "`coop start` prints a summary and returns — `coop progress` opens the view"
        )
        return
    watch(a, mode=progress.ADVANCED if a.advanced else progress.SIMPLE, once=a.once)


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
            # a worker that says it is failing has already explained the silence
            quiet = age > 300 and not st.get("failing")
            stale = f" (no update for {fmt_eta(age)} — check `coop logs`)" if quiet else ""
            print(f"now      {line}{stale}")
    else:
        print("worker   not running — `coop start` to contribute")
    # a background worker is invisible: without this line nobody can tell whether
    # the GPU they donated is the thing doing the work
    if running and st.get("device"):
        device = st["device"]
        print(f"device   {st.get('device_label') or describe(device)}")
    else:
        device = pick_device()
        print(f"device   {describe(device)} — what `coop start` would use")
    warn_cuda_gap(device)
    note = update_line(a.repo)
    if note:
        print(f"update   {note}")

    if st.get("rounds_done"):
        when = "this session" if running else "last session"
        toks = st.get("tokens_session", 0)
        print(f"session  {st['rounds_done']} rounds · {toks:,} tokens trained {when}")
    n = pending_rounds()
    if n:
        if not running:
            how = "they go out on the next `coop start`"
        elif n >= submit.PENDING_MAX:
            how = "training is paused until they send"
        else:
            how = "retrying every round"
        print(f"pending  {n} finished round(s) saved locally after a failed upload — {how}")

    try:
        meta = json.loads(Path(hubio.download_file(model_repo, "meta.json")).read_text())
        val = meta.get("eval", {}).get("val_loss")
        suffix = f" (val loss {val})" if val is not None else ""
        print(f"model    {model_repo} @ outer step {meta['step']}{suffix}")
    except Exception:
        print(f"model    {model_repo} (couldn't reach huggingface.co)")
    tr = fetch_trend(a.repo, cfg)
    if tr:
        print(f"loss     {trend.describe(tr)}")
        if tr.get("spark"):
            print(f"         {tr['spark']}  {trend.spark_caption(tr)}")

    user = st.get("user") or hubio.whoami()
    try:
        md = fetch_raw(a.repo, "LEADERBOARD.md", HOME / "board.md", ref="ledger").read_text()
        rows = parse_board(md)
        mine = next((r for r in rows if r["user"] == user), None)
        if mine:
            you = f"{user} — {mine['tokens']:,} tokens credited"
            print(f"you      {you} · rank {mine['rank']} of {len(rows)}")
        total = sum(r["tokens"] for r in rows)
        target = goal_tokens(cfg)
        pct = 100 * total / target
        print(f"goal     {total:,} of ~{target:,} community tokens ({pct:.1f}%)")
        if why := goal_note(cfg):
            print(f"{'':<9}{why}")
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


def run_model_from_words(words: list[str]) -> str | None:
    """`coop run the latest model` -> "latest"; the nouns are optional filler."""
    rest = [w for w in words if w not in ("model", "the")]
    if len(rest) > 1:
        raise SystemExit(f"too many arguments: {' '.join(words)}")
    return rest[0] if rest else None


def cmd_run(a: argparse.Namespace) -> None:
    from coop import play

    # settle the prompt before the download: a pipe with nothing in it should fail
    # now, not after several hundred MB of checkpoint
    prompt = a.prompt
    if prompt is None and not sys.stdin.isatty():  # echo "Once upon a" | coop run latest
        prompt = sys.stdin.read().strip()
        if not prompt:
            raise SystemExit('no prompt — try `coop run latest --prompt "Once upon a time"`')

    sel = play.pick_run(load_runs(a.repo), a.model)
    # own cache file: ~/.coop/run.yaml belongs to a worker that may be training a
    # different run right now, and fetch_raw overwrites in place
    cfg = load_run_config(a.repo, sel["config"], dest="run.play.yaml")
    name = sel.get("name") or cfg["repos"]["model"].split("/")[-1]
    print(f"loading {name} — the first run downloads the checkpoint, then it's cached")
    try:
        model, tok, meta, device = play.load_run(
            a.repo, cfg, HOME, name, revision=a.revision, device=a.device
        )
    except Exception as e:
        raise SystemExit(f"couldn't load {name}: {e}") from e

    val = meta.get("eval", {}).get("val_loss")
    line = f"{name} · outer step {meta['step']}"
    line += f" · val loss {val}" if val is not None else ""
    print(f"{line} · running on your {describe(device)}")
    warn_cuda_gap(device)

    def emit(prompt: str) -> None:
        for piece in play.stream(
            model,
            tok,
            prompt,
            n_tokens=a.tokens,
            temperature=a.temperature,
            top_k=a.top_k,
            device=device,
        ):
            print(piece, end="", flush=True)
        print()

    if prompt:
        emit(prompt)
        return

    print("\nit writes what comes next. type a prompt, or ctrl-c to leave.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt in ("exit", "quit"):
            return
        if not prompt:
            continue
        try:
            emit(prompt)
        except KeyboardInterrupt:  # cancels this generation, not the session
            print("\n(stopped)")


def cmd_update(a: argparse.Namespace) -> None:
    if a.auto:
        on = a.auto == "on"
        write_settings(auto_update=on)
        print(
            "auto-update on — a running worker takes new versions between rounds, never mid-round"
            if on
            else "auto-update off — new versions wait for `coop update`"
        )
    manifest = update.available(a.repo, HOME, force=True)
    if not manifest:
        print(f"coop {__version__} — you're on the newest version")
        return
    version = manifest.get("version", "?")
    print(f"coop {version} is out (you're on {__version__})")
    for line in (manifest.get("notes"), manifest.get("url")):
        if line:
            print(f"  {line}")
    if a.check:
        print("\n`coop update` installs it")
        return
    kind = update.install_kind()
    if kind == update.GIT:
        raise SystemExit("this is a git checkout of coop — `git pull` updates it")
    print("\nupdating — this can take a minute ...")
    ok, out = update.apply(a.repo, kind)
    if not ok:
        print(out)
        raise SystemExit("update failed — nothing was lost, you're still on a working coop")
    print(f"updated to coop {version}")
    pid = read_pid()
    if pid and alive(pid):
        if read_settings().get("auto_update"):
            print("the running worker moves over when its current round finishes")
        else:
            print("a worker is still running the old version — `coop stop`, then `coop start`")


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
    ap.add_argument("--version", action="version", version=f"coop {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    st = sub.add_parser("start", help="start contributing in the background")
    st.add_argument("words", nargs="*", metavar="[training] [model]")
    st.add_argument("--hf-token", default=None, help="Hugging Face write token (first run only)")
    st.add_argument("--choose", action="store_true", help="re-open the model picker")
    st.add_argument(
        "--latest", action="store_true", help="no picker, no questions: join the current run"
    )
    st.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    st.add_argument(
        "--rounds", type=int, default=0, help="stop after N rounds (default: run until coop stop)"
    )
    st.add_argument("--no-progress", action="store_true", help="don't open the live view this once")

    rn = sub.add_parser("run", help="talk to the model trained so far (no account needed)")
    rn.add_argument("words", nargs="*", metavar="[latest|model]")
    rn.add_argument("--prompt", default=None, help="generate once and exit; good for pipes")
    rn.add_argument("--tokens", type=int, default=200, help="how much to generate (default 200)")
    rn.add_argument("--temperature", type=float, default=0.8, help="higher = wilder")
    rn.add_argument("--top-k", type=int, default=50)
    rn.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    rn.add_argument("--revision", default="main", help="pin a checkpoint revision")

    sub.add_parser("stop", help="stop contributing")
    sub.add_parser("status", help="worker and model state")

    pg = sub.add_parser("progress", help="live progress bars you can watch (and stop from)")
    pg.add_argument(
        "--auto", choices=["on", "off"], help="open this view automatically from `coop start`"
    )
    pg.add_argument("--advanced", action="store_true", help="open on the detailed view")
    pg.add_argument("--once", action="store_true", help="print one snapshot and exit; for pipes")

    lg = sub.add_parser("logs", help="show what the worker is doing")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("-n", "--lines", type=int, default=40)

    up = sub.add_parser("update", help="get the newest coop")
    up.add_argument("--check", action="store_true", help="say what's new, install nothing")
    up.add_argument(
        "--auto", choices=["on", "off"], help="let the worker update itself between rounds"
    )

    a = ap.parse_args()
    if a.cmd == "start":
        a.model = model_from_words(a.words)
        cmd_start(a)
    elif a.cmd == "run":
        a.model = run_model_from_words(a.words)
        cmd_run(a)
    elif a.cmd == "stop":
        cmd_stop(a)
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "progress":
        cmd_progress(a)
    elif a.cmd == "logs":
        cmd_logs(a)
    elif a.cmd == "update":
        cmd_update(a)
    else:
        print(WELCOME)


if __name__ == "__main__":
    main()
