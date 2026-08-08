import re

from coop.ledger import (
    empty_ledger,
    hardware,
    load_ledger,
    render_leaderboard,
    save_ledger,
    score,
    update_ledger,
)


def meta(user, tokens=1000, tier="gpu", device=None):
    m = {"username": user, "tokens": tokens, "tier": tier, "start_step": 0}
    return m | {"device": device} if device else m


def test_accepted_updates_totals():
    led = empty_ledger()
    update_ledger(led, [meta("alice"), meta("alice"), meta("bob", 500, "cpu")], step=7)
    a, b = led["contributors"]["alice"], led["contributors"]["bob"]
    assert a == {
        "first_seen": 7,
        "submissions": 2,
        "tokens": 2000,
        "tier": "gpu",
        "devices": {"gpu": 2000},
        "reputation": 1.0,
    }
    assert b["tokens"] == 500 and b["tier"] == "cpu"
    assert led["step"] == 7


def test_hardware_lists_every_machine_biggest_first():
    """A donor with a GPU box and a laptop used to show whichever submitted last."""
    led = empty_ledger()
    update_ledger(
        led,
        [
            meta("mia", 500, device="cpu"),
            meta("mia", 3000, device="nvidia-gpu"),
            meta("mia", 900, device="cpu"),
            meta("mia", 2000, device="google-tpu"),
        ],
        step=3,
    )
    e = led["contributors"]["mia"]
    assert e["devices"] == {"nvidia-gpu": 3000, "google-tpu": 2000, "cpu": 1400}
    assert hardware(e) == "nvidia-gpu·google-tpu·cpu"
    assert "| 1 | mia | nvidia-gpu·google-tpu·cpu | 4 |" in render_leaderboard(led)


def test_hardware_falls_back_for_older_workers_and_ledgers():
    """Rounds from a 0.3.0 worker carry only gpu/cpu, and entries written before the
    split have no devices at all — neither may render an empty cell."""
    led = empty_ledger()
    update_ledger(led, [meta("bob", 500, "gpu")], step=1)  # no device field
    assert hardware(led["contributors"]["bob"]) == "gpu"
    assert hardware({"tier": "cpu"}) == "cpu"
    assert hardware({"tier": "cpu", "devices": {}}) == "cpu"


def test_a_worker_updating_mid_run_does_not_grow_a_second_machine():
    """Every active contributor spends the rollout with old rounds saying "gpu" and new
    ones naming the card. That is one machine, and the board must not imply two."""
    led = empty_ledger()
    update_ledger(led, [meta("mia", 9000, "gpu")], step=1)  # a 0.3.0 worker
    update_ledger(led, [meta("mia", 1000, device="apple-gpu")], step=2)  # after it updates
    assert hardware(led["contributors"]["mia"]) == "apple-gpu"
    assert led["contributors"]["mia"]["devices"] == {"gpu": 9000, "apple-gpu": 1000}  # kept raw

    update_ledger(led, [meta("mia", 50, device="cpu")], step=3)
    assert hardware(led["contributors"]["mia"]) == "apple-gpu·cpu"


def test_vague_rounds_stay_visible_until_a_card_is_known():
    led = empty_ledger()
    update_ledger(led, [meta("bo", 10, "gpu"), meta("bo", 5, "cpu", device="cpu")], step=1)
    assert hardware(led["contributors"]["bo"]) == "gpu·cpu"


def test_todays_board_still_parses_in_an_unupdated_client():
    """Those installs match this cell with `\\S+` and cannot be updated by us; a space in
    it blanks their "you" line and reports the community total as 0 of the goal."""
    old_client = re.compile(r"^\| (\d+) \| (\S+) \| \S+ \| \d+ \| ([\d,]+) \| ")
    led = empty_ledger()
    update_ledger(
        led,
        [meta("mia", 2_000_000, device="apple-gpu"), meta("mia", 5, device="cpu"), meta("bo", 7)],
        step=1,
    )
    rows = [old_client.match(ln) for ln in render_leaderboard(led).splitlines() if ln[:3] == "| 1"]
    assert rows and all(rows)
    assert rows[0].group(2) == "mia" and rows[0].group(3) == "2,000,005"


def test_reputation_monotone():
    led = empty_ledger()
    update_ledger(led, [meta("alice")], step=1)
    rep = led["contributors"]["alice"]["reputation"]
    assert rep == 1.0
    update_ledger(led, [], step=2, rejected=[meta("alice")])
    rep_after_reject = led["contributors"]["alice"]["reputation"]
    assert rep_after_reject < rep
    update_ledger(led, [meta("alice")], step=3)
    rep_after_accept = led["contributors"]["alice"]["reputation"]
    assert rep_after_reject < rep_after_accept <= 1.0


def test_leaderboard_sorted_by_score():
    led = empty_ledger()
    update_ledger(led, [meta("alice", 2000), meta("bob", 500)], step=1)
    update_ledger(led, [], step=1, rejected=[meta("alice")])
    md = render_leaderboard(led)
    assert score(led["contributors"]["alice"]) > score(led["contributors"]["bob"])
    assert md.index("| 1 | alice |") < md.index("| 2 | bob |")
    assert "Score = tokens contributed" in md  # the sort key is documented


def test_roundtrip(tmp_path):
    led = update_ledger(empty_ledger(), [meta("alice")], step=1)
    path = tmp_path / "ledger" / "ledger.json"
    save_ledger(led, path)
    assert load_ledger(path) == led
    assert load_ledger(tmp_path / "missing.json") == empty_ledger()


def test_render_leaderboard_lists_archived_runs():
    led = empty_ledger()
    board = render_leaderboard(led, archives=["LEADERBOARD-stage1.md"])
    assert "## Past runs" in board
    assert "[stage1 — final board](LEADERBOARD-stage1.md)" in board
    assert "## Past runs" not in render_leaderboard(led)


def test_board_tells_old_installs_how_to_get_the_update_command():
    # coop cannot announce itself to a version that shipped without `coop update`;
    # the board the aggregator rewrites every tick is the one channel that reaches them
    board = render_leaderboard(empty_ledger())
    assert "before 0.3.0" in board
    assert "uv tool install --force git+https://github.com/commonsense-ai/coop" in board
    assert board.index("Running coop from before") < board.index("Score = tokens contributed")
