from pathlib import Path

import numpy as np
import pytest
import torch

from coop import data
from coop.data import EOT, iter_batches, load_tokenizer, tokenize_file, train_tokenizer

CORPUS = (
    "Once upon a time there was a tiny robot. "
    "The robot liked to read stories about dragons and stars. "
) * 50


class FakeStream:
    def __init__(self, docs):
        self.docs = docs

    def skip(self, n):
        return FakeStream(self.docs[n:])

    def take(self, n):
        return self.docs[:n]


def test_fetch_docs_uses_configured_dataset_and_field(tmp_path, monkeypatch):
    import datasets

    seen = {}

    def fake_load(name, config=None, split=None, streaming=True):
        seen["name"], seen["config"], seen["split"] = name, config, split
        return FakeStream([{"content": "hello"}, {"content": "world"}])

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    out = data.fetch_docs(
        str(tmp_path / "s.txt"), 2, dataset="org/corpus", text_field="content", config="sample"
    )
    assert seen == {"name": "org/corpus", "config": "sample", "split": "train"}
    assert "hello" in Path(out).read_text()


def test_fetch_docs_raises_past_dataset_end(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: FakeStream([{"text": "x"}])
    )  # config-agnostic
    with pytest.raises(ValueError, match="past the end"):
        data.fetch_docs(str(tmp_path / "s.txt"), 5, skip=10)


def test_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS)
    tok_path = tmp_path / "tok.json"
    tok = train_tokenizer([str(corpus)], vocab_size=512, out_path=str(tok_path))
    text = "Once upon a time there was a tiny robot."
    assert tok.decode(tok.encode(text).ids) == text
    assert load_tokenizer(tok_path).decode(tok.encode(text).ids) == text


def test_tokenize_and_batches(tmp_path):
    corpus = tmp_path / "corpus.txt"
    docs = [f"story number {i} about a robot and a dragon" for i in range(20)]
    corpus.write_text(f"\n{EOT}\n".join(docs))
    tok = train_tokenizer([str(corpus)], vocab_size=512, out_path=str(tmp_path / "tok.json"))

    bin_path = tmp_path / "tokens.bin"
    n = tokenize_file(tok, str(corpus), str(bin_path))
    raw = np.fromfile(bin_path, dtype=np.uint16)
    assert n > 0 and len(raw) == n
    assert (raw == tok.token_to_id(EOT)).sum() == 20  # one EOT per doc

    x, y = next(iter_batches(str(bin_path), batch=4, block=8, seed=0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert torch.equal(y[:, :-1], x[:, 1:])  # y is x shifted by one


def test_tokenize_file_leaves_no_truncated_shard(tmp_path, monkeypatch):
    """A shard killed mid-write must not land at the canonical path: join.py reuses any
    file that exists there, so a partial one would silently shrink every future round."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"\n{EOT}\n".join(f"story {i} about a robot" for i in range(20)))
    tok = train_tokenizer([str(corpus)], vocab_size=512, out_path=str(tmp_path / "tok.json"))
    bin_path = tmp_path / "tokens.bin"

    class Dying:
        def tofile(self, path):
            Path(path).write_bytes(b"\x00" * 8)  # partial write, then the machine dies
            raise OSError("no space left on device")

    monkeypatch.setattr(data.np, "asarray", lambda *a, **kw: Dying())
    with pytest.raises(OSError):
        tokenize_file(tok, str(corpus), str(bin_path))
    assert not bin_path.exists()

    monkeypatch.undo()
    assert tokenize_file(tok, str(corpus), str(bin_path)) > 0
    assert bin_path.exists() and not list(tmp_path.glob("*.tmp"))
