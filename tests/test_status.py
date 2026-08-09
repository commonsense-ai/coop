import time

from coop.status import Heartbeat, StatusFile, read_status


def test_statusfile_merges_and_roundtrips(tmp_path):
    p = tmp_path / "status.json"
    s = StatusFile(p)
    s.update(phase="training", inner_step=10)
    s.update(inner_step=20)
    st = read_status(p)
    assert st["phase"] == "training"
    assert st["inner_step"] == 20
    assert st["updated_at"] > 0
    assert not p.with_suffix(".tmp").exists()


def test_heartbeat_keeps_a_blocking_phase_from_looking_dead(tmp_path):
    """The 584MB checkpoint download is minutes of one call: without ticks, status.json
    freezes and `coop status` warns about a worker that is fine."""
    p = tmp_path / "status.json"
    s = StatusFile(p)
    with Heartbeat(s, "downloading checkpoint", every=0.02) as beat:
        first = read_status(p)["updated_at"]
        beat.bytes(470_000_000, 584_113_800)
        time.sleep(0.15)
        st = read_status(p)
    assert st["phase"] == "downloading checkpoint"
    assert st["updated_at"] > first  # it ticked on its own, with no work reporting in
    assert st["bytes_done"] == 470_000_000 and st["bytes_total"] == 584_113_800
    assert st["phase_secs"] >= 0

    frozen = read_status(p)["updated_at"]
    time.sleep(0.1)
    assert read_status(p)["updated_at"] == frozen  # and it stops when the phase does


def test_heartbeat_does_not_carry_the_last_transfer_into_this_one(tmp_path):
    """Stale byte counts would show a fresh download as already finished."""
    p = tmp_path / "status.json"
    s = StatusFile(p)
    with Heartbeat(s, "downloading checkpoint", every=10) as beat:
        beat.bytes(584_113_800, 584_113_800)
    with Heartbeat(s, "downloading checkpoint", every=10):
        st = read_status(p)
    assert st["bytes_done"] is None and st["bytes_total"] is None


def test_heartbeat_without_a_status_file_is_a_no_op():
    with Heartbeat(None, "submitting", every=0.01) as beat:
        beat.bytes(1, 2)  # a caller with no status file still needs no branch


def test_read_status_missing_and_garbage(tmp_path):
    assert read_status(tmp_path / "nope.json") == {}
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert read_status(p) == {}
