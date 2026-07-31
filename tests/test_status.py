from coop.status import StatusFile, read_status


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


def test_read_status_missing_and_garbage(tmp_path):
    assert read_status(tmp_path / "nope.json") == {}
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert read_status(p) == {}
