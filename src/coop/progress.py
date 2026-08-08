"""The live progress view: one screen a volunteer watches, and acts from.

`coop start` opens it by default. The honest answer to "is my computer actually
doing anything?" is a screen that shows it, not a command you have to know to
type; `coop progress --auto off` brings the print-and-exit start back.

Two views, because two people are watching: simple is three bars and a sentence,
advanced is everything `coop status` prints. Left/right moves between them,
up/down between the only two things anyone wants next — keep going, or stop.

Rendering is pure (a ctx dict in, lines out), and nothing here reads the process
table or the network. The caller passes a probe for each, so the network one can
run off the render thread and a hub outage never freezes the screen.
"""

import os
import select
import shutil
import sys
import threading

SIMPLE, ADVANCED = "simple", "advanced"
LEAVE, STOP, START = "leave", "stop", "start"

BAR_W = 24
TICK = 0.5  # idle redraw; a keypress wakes the loop immediately regardless
REFRESH = 120.0  # network refresh — the aggregator ticks far slower than this
LABEL = 12  # bar-label column, so the three bars line up

HIDE, SHOW = "\x1b[?25l", "\x1b[?25h"
KEYS = {
    "\r": "enter",
    "\n": "enter",
    "\t": "right",
    "k": "up",
    "j": "down",
    "h": "left",
    "l": "right",
    "q": "quit",
}
SEQ = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}


def bar(frac: float, width: int = 30) -> str:
    filled = round(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


def fmt_eta(secs: float) -> str:
    s = max(0, int(secs))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def fmt_tokens(n: int) -> str:
    """Bars are read at a glance; 37.2M lands and 37,224,448 does not."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n / scale:.1f}{suffix}"
    return str(int(n))


def phase_progress(st: dict) -> tuple[float | None, str, str]:
    """(fraction, what it's doing, time left) for the work in flight. The fraction
    is None whenever the worker is between measurable things — say so, promise no bar."""
    phase = st.get("phase", "")
    if st.get("failing"):
        # a bar over a worker that is getting nowhere is the one lie this screen could tell
        n = st["failing"]
        return None, f"{n} round{'s' if n > 1 else ''} in a row failed — retrying", ""
    if phase == "training" and st.get("h_steps"):
        i, h = st.get("inner_step", 0), st["h_steps"]
        rate = st.get("steps_per_sec") or 0
        eta = f" · ~{fmt_eta((h - i) / rate)} left" if rate else ""
        return i / h, f"inner step {i}/{h} · loss {st.get('loss', '?')}", eta
    if phase.startswith("building your data shard") and st.get("shard_total"):
        done, total = st.get("shard_done", 0), st["shard_total"]
        rate = st.get("shard_per_sec") or 0
        eta = f" · ~{fmt_eta((total - done) / rate)} left" if rate and done < total else ""
        return done / total, st.get("shard_stage", ""), eta
    return None, phase, ""


def now_line(st: dict) -> str:
    """The one-line form, shared by `coop status` and the advanced view."""
    phase = st.get("phase", "")
    if st.get("failing"):
        _, what, _ = phase_progress(st)
        err = st.get("last_error", "")
        return what + (f"\n         last error: {err}" if err else "")
    if phase == "waiting":
        step = st.get("waiting_past_step")
        which = f"outer step {step + 1}" if isinstance(step, int) else "the next outer step"
        return f"submitted — waiting for {which} (your work merges at the next aggregator tick)"
    frac, what, eta = phase_progress(st)
    if frac is None:
        return phase
    if phase == "training":
        return f"training — {what}{eta}"
    return f"building your data shard — {what} {bar(frac, 16)} {100 * frac:.0f}%{eta}"


def row(label: str, frac: float | None, detail: str) -> str:
    if frac is None:
        return f"{label:<{LABEL}} {detail}"
    return f"{label:<{LABEL}} {bar(frac, BAR_W)} {100 * frac:5.1f}%  {detail}"


def actions_for(running: bool) -> list[tuple[str, str]]:
    if running:
        return [(LEAVE, "keep training (leave this screen)"), (STOP, "stop contributing")]
    return [(START, "start contributing"), (LEAVE, "leave")]


def head(ctx: dict) -> str:
    st = ctx.get("st") or {}
    bits = ["coop"]
    for part in (ctx.get("run_name"), ctx.get("user") or st.get("user"), ctx.get("device_label")):
        if part and part != "anonymous":
            bits.append(str(part))
    return " · ".join(bits)


def simple_rows(ctx: dict) -> list[str]:
    st = ctx.get("st") or {}
    running = bool(ctx.get("running"))
    rows = []
    if running:
        frac, what, eta = phase_progress(st)
        rows.append(row("this round", frac, f"{what}{eta}" or "starting up"))
    else:
        rows.append(f"{'this round':<{LABEL}} not training right now — your credit is safe")

    total, goal = ctx.get("total_tokens"), ctx.get("goal") or 0
    if total is None or not goal:
        rows.append(row("the model", None, "reading the leaderboard ..."))
    else:
        rows.append(
            row("the model", total / goal, f"{fmt_tokens(total)} of ~{fmt_tokens(goal)} tokens")
        )
        mine = ctx.get("my_tokens")
        if mine:
            where = f"rank {ctx['rank']} of {ctx['of']}" if ctx.get("rank") else "credited to you"
            rows.append(row("your share", mine / total, f"{where} · {mine:,} tokens"))

    done, toks = st.get("rounds_done", 0), st.get("tokens_session", 0)
    if done or toks:
        when = "this session" if running else "last session"
        s = "" if done == 1 else "s"
        rows += ["", f"{done} round{s} · {toks:,} tokens {when} — each one submitted for you"]
    if ctx.get("pending"):
        how = (
            "training is paused until they send"
            if ctx.get("pending_paused")
            else "coop resends them by itself"
        )
        rows.append(f"{ctx['pending']} finished round(s) are parked offline — {how}")
    if ctx.get("update_note"):
        rows.append(ctx["update_note"])
    return rows


def advanced_rows(ctx: dict) -> list[str]:
    """The `coop status` fields, same labels and same order: a volunteer who has read
    one has read the other."""
    st = ctx.get("st") or {}
    rows = []
    if ctx.get("running"):
        rnd = st.get("rounds_done", 0) + 1
        target = st.get("rounds_target")
        mode = f"round {rnd} of {target}" if target else f"endless · round {rnd}"
        rows.append(
            f"worker   running (pid {ctx.get('pid')}, up {fmt_eta(ctx.get('uptime') or 0)})"
            f" — {mode}"
        )
        line = now_line(st)
        if line:
            # a worker that says it is failing has already explained the silence
            stale = 0 if st.get("failing") else (ctx.get("stale_for") or 0)
            warn = f" (no update for {fmt_eta(stale)} — check `coop logs`)" if stale > 300 else ""
            rows.append(f"now      {line}{warn}")
    else:
        rows.append("worker   not running — `coop start` to contribute")
    rows.append(f"device   {ctx.get('device_label') or '?'}")
    if ctx.get("update_note"):
        rows.append(f"update   {ctx['update_note']}")
    if st.get("rounds_done"):
        when = "this session" if ctx.get("running") else "last session"
        rows.append(
            f"session  {st['rounds_done']} rounds · {st.get('tokens_session', 0):,} "
            f"tokens trained {when}"
        )
    if ctx.get("pending"):
        how = "training is paused until they send" if ctx.get("pending_paused") else "retrying"
        rows.append(
            f"pending  {ctx['pending']} finished round(s) saved locally after a failed "
            f"upload — {how}"
        )
    if ctx.get("model_repo"):
        val = ctx.get("val_loss")
        step = ctx.get("outer_step")
        where = f" @ outer step {step}" if step is not None else " (couldn't reach huggingface.co)"
        rows.append(f"model    {ctx['model_repo']}{where}" + (f" (val loss {val})" if val else ""))
    if ctx.get("rank"):
        rows.append(
            f"you      {ctx.get('user')} — {ctx.get('my_tokens', 0):,} tokens credited "
            f"· rank {ctx['rank']} of {ctx['of']}"
        )
    if ctx.get("total_tokens") is not None and ctx.get("goal"):
        total, goal = ctx["total_tokens"], ctx["goal"]
        rows.append(f"goal     {total:,} of ~{goal:,} community tokens ({100 * total / goal:.1f}%)")
    if ctx.get("inbox") is not None:
        rows.append(f"inbox    {ctx['inbox']}")
    for label, key in (("board", "board_url"), ("log", "logfile")):
        if ctx.get(key):
            rows.append(f"{label:<8} {ctx[key]}")
    return rows


def footer(mode: str) -> str:
    other = ADVANCED if mode == SIMPLE else SIMPLE
    return f"↑↓ move · enter choose · ←→ {other} view · q leave (training keeps going)"


def clip(line: str, width: int) -> str:
    """Wrapped lines break the redraw's cursor arithmetic — a long repo name must
    lose its tail, not push the screen out of shape."""
    return line if width <= 0 or len(line) <= width else line[: max(1, width - 1)] + "…"


def render(ctx: dict, mode: str = SIMPLE, cursor: int | None = 0, width: int = 80) -> list[str]:
    """The whole screen. cursor=None renders the static snapshot: no menu, no footer."""
    lines = [head(ctx), ""]
    for r in advanced_rows(ctx) if mode == ADVANCED else simple_rows(ctx):
        lines += r.split("\n")  # a failure carries its last error on a second line
    hl = None
    if cursor is not None:
        acts = actions_for(bool(ctx.get("running")))
        cursor %= len(acts)
        lines.append("")
        for i, (_, label) in enumerate(acts):
            if i == cursor:
                hl = len(lines)
            lines.append(f"  {'>' if i == cursor else ' '} {label}")
        lines += ["", footer(mode)]
    out = [clip(ln, width) for ln in lines]
    if hl is not None:
        out[hl] = f"\x1b[7m{out[hl]}\x1b[0m"
    return out


def fit(lines: list[str], height: int, tail: int) -> list[str]:
    """A screen taller than the terminal scrolls, and scrolling breaks the redraw.
    Drop from the middle: the header says what this is, the tail is how to leave."""
    if height <= 0 or len(lines) <= height:
        return lines
    keep = height - tail
    return lines[-height:] if keep < 1 else lines[:keep] + lines[-tail:]


def draw(out, lines: list[str], prev: int) -> int:
    if prev:
        out.write(f"\x1b[{prev}F")
    out.write("\x1b[J")  # the two views differ in height; clear what the taller one left
    out.write("\n".join(lines) + "\n")
    out.flush()
    return len(lines)


def read_key(fd: int, timeout: float) -> str | None:
    """One keypress, or None when the timeout ran out and it's time to redraw."""
    if not select.select([fd], [], [], timeout)[0]:
        return None
    ch = os.read(fd, 1).decode(errors="ignore")
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch != "\x1b":
        return KEYS.get(ch, ch)
    # an arrow key arrives as one burst; a bare escape has nothing behind it
    if not select.select([fd], [], [], 0.05)[0]:
        return "quit"
    return SEQ.get(os.read(fd, 2).decode(errors="ignore"), "")


def interactive(out) -> bool:
    if not (sys.stdin.isatty() and out.isatty()):
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError:  # Windows: the snapshot is the whole story there
        return False
    return True


def _refresh(remote, box: dict, stop: threading.Event, every: float) -> None:
    while True:
        try:
            box["remote"] = remote()
        except Exception:
            pass  # a hub outage costs freshness, never the screen
        if stop.wait(every):
            return


def snapshot(probe, remote) -> dict:
    ctx = dict(probe())
    if remote:
        try:
            ctx |= remote()
        except Exception:
            pass
    return ctx


def view(
    probe,
    remote=None,
    mode: str = SIMPLE,
    once: bool = False,
    out=None,
    tick: float = TICK,
    refresh: float = REFRESH,
) -> str:
    """Run the view until the volunteer picks something. Returns that choice —
    acting on it belongs to the caller, which is what keeps this module free of
    the CLI's process handling."""
    out = out or sys.stdout
    if once or not interactive(out):
        for line in render(snapshot(probe, remote), mode, None, term()[0]):
            print(line, file=out)
        return LEAVE

    import termios
    import tty

    box: dict = {"remote": {}}
    stop = threading.Event()
    if remote:
        threading.Thread(target=_refresh, args=(remote, box, stop, refresh), daemon=True).start()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    cursor, drawn = 0, 0
    try:
        tty.setcbreak(fd)
        out.write(HIDE)
        while True:
            ctx = dict(probe()) | box["remote"]
            acts = actions_for(bool(ctx.get("running")))
            cursor = min(cursor, len(acts) - 1)
            width, height = term()
            lines = fit(render(ctx, mode, cursor, width), height - 1, len(acts) + 3)
            drawn = draw(out, lines, drawn)
            key = read_key(fd, tick)
            if key == "up":
                cursor = (cursor - 1) % len(acts)
            elif key == "down":
                cursor = (cursor + 1) % len(acts)
            elif key in ("left", "right"):
                mode = ADVANCED if mode == SIMPLE else SIMPLE
            elif key == "enter":
                return acts[cursor][0]
            elif key == "quit":
                return LEAVE
    except KeyboardInterrupt:  # the worker is its own session: ctrl-c leaves, never kills
        return LEAVE
    finally:
        stop.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        out.write(SHOW + "\n")
        out.flush()


def term() -> tuple[int, int]:
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines
