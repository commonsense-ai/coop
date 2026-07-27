import urllib.request

import coop.join as join


def test_derive_skip_deterministic_and_bounded():
    a = join.derive_skip("alice")
    assert a == join.derive_skip("alice")
    assert a % 20000 == 0
    assert 0 <= a < 500 * 20000
    assert join.derive_skip("alice", docs=1000) % 1000 == 0


def test_fetch_raw(monkeypatch, tmp_path):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"data: 1"

    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = join.fetch_raw("o/r", "config/run.yaml", tmp_path / "sub" / "run.yaml")
    assert p.read_text() == "data: 1"
    assert captured["url"] == "https://raw.githubusercontent.com/o/r/main/config/run.yaml"
