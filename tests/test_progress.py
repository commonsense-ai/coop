import io
import os

import pytest

import coop.progress as pg

TRAINING = {
    "phase": "training",
    "inner_step": 240,
    "h_steps": 500,
    "loss": 3.21,
    "steps_per_sec": 2.0,
    "rounds_done": 3,
    "tokens_session": 2457600,
    "user": "naloxene",
}

CTX = {
    "st": TRAINING,
    "running": True,
    "pid": 4242,
    "uptime": 3840,
    "device_label": "Apple GPU",
    "pending": 0,
    "logfile": "/home/x/.coop/worker.log",
    "board_url": "https://github.com/o/r/blob/ledger/LEADERBOARD.md",
    "run_name": "fineweb-150m",
    "user": "naloxene",
    "model_repo": "o/m",
    "outer_step": 37,
    "val_loss": 3.02,
    "goal": 300_000_000,
    "total_tokens": 37_200_000,
    "my_tokens": 2_457_600,
    "rank": 2,
    "of": 14,
    "inbox": "empty — all submitted work has been aggregated",
}


def test_phase_progress_splits_fraction_from_words():
    frac, what, eta = pg.phase_progress(TRAINING)
    assert frac == 240 / 500
    assert what == "inner step 240/500 · loss 3.21"
    assert eta == " · ~2m 10s left"


def test_phase_progress_reports_the_shard_build():
    frac, what, _ = pg.phase_progress(
        {
            "phase": "building your data shard (one-time)",
            "shard_stage": "tokenizing",
            "shard_done": 5000,
            "shard_total": 20000,
        }
    )
    assert frac == 0.25 and what == "tokenizing"


def test_phase_progress_promises_no_bar_without_numbers():
    """A phase with nothing to measure must not render an empty bar at 0%."""
    assert pg.phase_progress({"phase": "downloading checkpoint"}) == (
        None,
        "downloading checkpoint",
        "",
    )


def test_a_failing_worker_gets_words_instead_of_a_bar():
    """A bar over a worker that is getting nowhere is the one lie this screen could tell."""
    st = {"phase": "round failed — retrying", "failing": 3, "last_error": "403"}
    frac, what, _ = pg.phase_progress(st)
    assert frac is None and what == "3 rounds in a row failed — retrying"
    screen = "\n".join(pg.render(CTX | {"st": st, "stale_for": 9000}, pg.ADVANCED, width=160))
    assert "3 rounds in a row failed" in screen and "403" in screen
    assert "no update for" not in screen  # the failure already explains the silence


def test_a_second_line_stays_a_second_line():
    """now_line can carry a last-error line; an unsplit \\n would desync the redraw."""
    st = {"phase": "x", "failing": 2, "last_error": "boom"}
    lines = pg.render(CTX | {"st": st}, pg.ADVANCED, width=160)
    assert not any("\n" in ln for ln in lines)
    assert any(ln.strip().startswith("last error: boom") for ln in lines)


def test_parked_rounds_say_whether_training_stopped():
    paused = "\n".join(pg.render(CTX | {"pending": 8, "pending_paused": True}, pg.SIMPLE))
    assert "training is paused until they send" in paused
    retrying = "\n".join(pg.render(CTX | {"pending": 2, "pending_paused": False}, pg.SIMPLE))
    assert "coop resends them by itself" in retrying


def test_fmt_tokens_is_readable_at_a_glance():
    assert pg.fmt_tokens(37_200_000) == "37.2M"
    assert pg.fmt_tokens(2_500_000_000) == "2.5B"
    assert pg.fmt_tokens(4096) == "4.1K"
    assert pg.fmt_tokens(512) == "512"


def test_simple_view_has_a_bar_per_thing_a_volunteer_waits_on():
    screen = "\n".join(pg.render(CTX, pg.SIMPLE))
    assert "this round" in screen and "the model" in screen and "your share" in screen
    assert screen.count("█") and screen.count("░")
    assert "48.0%" in screen  # the round
    assert "37.2M of ~300.0M tokens" in screen
    assert "rank 2 of 14" in screen
    assert "3 rounds · 2,457,600 tokens this session" in screen


def test_simple_view_waits_for_the_board_without_faking_a_bar():
    screen = "\n".join(pg.render(CTX | {"total_tokens": None}, pg.SIMPLE))
    assert "reading the leaderboard" in screen
    assert "your share" not in screen


def test_advanced_view_carries_the_status_fields():
    screen = "\n".join(pg.render(CTX, pg.ADVANCED))
    for field in ("worker", "now", "device", "model", "you", "goal", "inbox", "board", "log"):
        assert f"{field:<8} " in screen or f"{field}   " in screen
    assert "pid 4242" in screen and "outer step 37" in screen and "val loss 3.02" in screen


def test_advanced_view_flags_a_worker_that_stopped_reporting():
    screen = "\n".join(pg.render(CTX | {"stale_for": 900}, pg.ADVANCED, width=160))
    assert "no update for 15m 00s" in screen


def test_actions_offer_stop_while_running_and_start_when_not():
    assert [k for k, _ in pg.actions_for(True)] == [pg.LEAVE, pg.STOP]
    assert [k for k, _ in pg.actions_for(False)] == [pg.START, pg.LEAVE]


def test_cursor_highlights_exactly_one_action():
    lines = pg.render(CTX, pg.SIMPLE, cursor=1)
    marked = [ln for ln in lines if "\x1b[7m" in ln]
    assert len(marked) == 1 and "stop contributing" in marked[0]
    assert any("> " in ln for ln in lines)


def test_snapshot_render_drops_the_menu():
    lines = pg.render(CTX, pg.SIMPLE, cursor=None)
    screen = "\n".join(lines)
    assert "stop contributing" not in screen and "↑↓" not in screen


def test_header_names_the_run_the_user_and_the_hardware():
    assert pg.head(CTX) == "coop · fineweb-150m · naloxene · Apple GPU"
    assert pg.head({"st": {"user": "anonymous"}}) == "coop"


def test_clip_keeps_lines_from_wrapping():
    """A wrapped line desynchronises the redraw's cursor arithmetic."""
    assert pg.clip("abcdefghij", 5) == "abcd…"
    assert pg.clip("short", 80) == "short"
    assert pg.clip("anything", 0) == "anything"


def test_fit_drops_the_middle_and_keeps_the_way_out():
    lines = [f"line{i}" for i in range(20)]
    out = pg.fit(lines, height=10, tail=4)
    assert len(out) == 10
    assert out[0] == "line0" and out[-1] == "line19"


def test_fit_leaves_a_screen_that_already_fits():
    lines = ["a", "b", "c"]
    assert pg.fit(lines, height=24, tail=4) == lines


def test_draw_rewinds_over_the_previous_frame():
    out = io.StringIO()
    n = pg.draw(out, ["a", "b"], prev=0)
    assert n == 2 and "\x1b[" not in out.getvalue().replace("\x1b[J", "")
    out = io.StringIO()
    pg.draw(out, ["a"], prev=2)
    assert out.getvalue().startswith("\x1b[2F")


def key(written: bytes, timeout: float = 1.0):
    r, w = os.pipe()
    os.write(w, written)
    try:
        return pg.read_key(r, timeout)
    finally:
        os.close(r), os.close(w)


def test_read_key_decodes_arrows_and_letters():
    assert key(b"\x1b[A") == "up"
    assert key(b"\x1b[B") == "down"
    assert key(b"\x1b[C") == "right"
    assert key(b"\x1b[D") == "left"
    assert key(b"j") == "down" and key(b"k") == "up"
    assert key(b"\r") == "enter"
    assert key(b"q") == "quit"


def test_read_key_treats_a_bare_escape_as_leaving():
    assert key(b"\x1b") == "quit"


def test_read_key_returns_none_when_nothing_is_pressed():
    r, w = os.pipe()
    try:
        assert pg.read_key(r, 0.01) is None
    finally:
        os.close(r), os.close(w)


def test_read_key_raises_on_ctrl_c():
    with pytest.raises(KeyboardInterrupt):
        key(b"\x03")


def test_view_prints_one_snapshot_when_there_is_no_terminal():
    out = io.StringIO()  # StringIO.isatty() is False: the pipe case
    assert pg.view(probe=lambda: CTX, remote=None, out=out) == pg.LEAVE
    screen = out.getvalue()
    assert "this round" in screen and "stop contributing" not in screen


def test_view_snapshot_survives_an_unreachable_hub():
    def boom():
        raise OSError("offline")

    out = io.StringIO()
    pg.view(probe=lambda: CTX, remote=boom, out=out, once=True)
    assert "this round" in out.getvalue()
