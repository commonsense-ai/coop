from coop.ledger import (
    empty_ledger,
    load_ledger,
    render_leaderboard,
    save_ledger,
    score,
    update_ledger,
)


def meta(user, tokens=1000, tier="gpu"):
    return {"username": user, "tokens": tokens, "tier": tier, "start_step": 0}


def test_accepted_updates_totals():
    led = empty_ledger()
    update_ledger(led, [meta("alice"), meta("alice"), meta("bob", 500, "cpu")], step=7)
    a, b = led["contributors"]["alice"], led["contributors"]["bob"]
    assert a == {
        "first_seen": 7,
        "submissions": 2,
        "tokens": 2000,
        "tier": "gpu",
        "reputation": 1.0,
    }
    assert b["tokens"] == 500 and b["tier"] == "cpu"
    assert led["step"] == 7


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
