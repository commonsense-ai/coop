import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from coop import data
from coop.data import (
    EOT,
    NoRandomAccess,
    Progress,
    iter_batches,
    load_tokenizer,
    tokenize_file,
    train_tokenizer,
)

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


@pytest.fixture
def no_fast_path(monkeypatch):
    """Force the streaming fallback: a dataset with no seekable parquet behind it."""
    monkeypatch.setattr(
        data, "data_files", lambda *a: (_ for _ in ()).throw(NoRandomAccess("not parquet"))
    )


def write_parquet(path: Path, docs: list[str], rows_per_group: int, field: str = "text") -> None:
    table = pa.table({field: docs})
    pq.write_table(table, path, row_group_size=rows_per_group)


def test_fetch_docs_uses_configured_dataset_and_field(tmp_path, monkeypatch, no_fast_path):
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


def test_fetch_docs_raises_past_dataset_end(tmp_path, monkeypatch, no_fast_path):
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: FakeStream([{"text": "x"}])
    )  # config-agnostic
    with pytest.raises(ValueError, match="past the end"):
        data.fetch_docs(str(tmp_path / "s.txt"), 5, skip=10)


def test_fetch_docs_seeks_the_row_offset_across_files(tmp_path, monkeypatch):
    """The whole point of the parquet path: docs at `skip` without reading what's before
    them. Two files of 60 docs, 10 per row group; the slice straddles the boundary."""
    files = []
    for f in range(2):
        p = tmp_path / f"part{f}.parquet"
        write_parquet(p, [f"doc {f * 60 + i}" for i in range(60)], rows_per_group=10)
        files.append(str(p))
    monkeypatch.setattr(data, "data_files", lambda *a: files)
    monkeypatch.setattr(data, "CHUNK_ROWS", 20)

    seen = []
    out = data.fetch_docs(
        str(tmp_path / "s.txt"),
        25,
        skip=55,
        dataset="org/corpus",
        progress=lambda d, t: seen.append(d),
    )
    docs = [d for d in Path(out).read_text().split(EOT) if d.strip()]
    assert [d.strip() for d in docs] == [f"doc {i}" for i in range(55, 80)]
    assert seen[-1] == 25 and seen == sorted(seen)  # progress only ever moves forward


def test_fetch_docs_reads_no_footers_past_the_slice(tmp_path, monkeypatch):
    """A skip near the end of a 9M-row corpus must not pay for the files after it."""
    files = []
    for f in range(4):
        p = tmp_path / f"part{f}.parquet"
        write_parquet(p, [f"doc {f * 10 + i}" for i in range(10)], rows_per_group=5)
        files.append(str(p))
    monkeypatch.setattr(data, "data_files", lambda *a: files)
    opened = []
    real_open = data._open
    monkeypatch.setattr(data, "_open", lambda url: (opened.append(url), real_open(url))[1])

    out = data.fetch_docs(str(tmp_path / "s.txt"), 5, skip=12, dataset="org/corpus")
    assert [d.strip() for d in Path(out).read_text().split(EOT) if d.strip()] == [
        f"doc {i}" for i in range(12, 17)
    ]
    assert files[3] not in opened  # never touched: the slice ends in part1


def test_fetch_docs_raises_past_a_parquet_corpus_end(tmp_path, monkeypatch):
    """derive_skip wrapping wrong must fail loudly here, not as "cannot mmap an empty
    file" one step later."""
    p = tmp_path / "part.parquet"
    write_parquet(p, [f"doc {i}" for i in range(10)], rows_per_group=5)
    monkeypatch.setattr(data, "data_files", lambda *a: [str(p)])
    with pytest.raises(ValueError, match="past the end"):
        data.fetch_docs(str(tmp_path / "s.txt"), 5, skip=50, dataset="org/corpus")


def test_fetch_docs_never_falls_back_once_docs_are_written(tmp_path, monkeypatch):
    """The fallback restarts the slice from the top: taking it after a partial write
    would put the first docs in the shard twice. Fail the round instead — it retries."""
    p = tmp_path / "part.parquet"
    write_parquet(p, [f"doc {i}" for i in range(20)], rows_per_group=5)
    monkeypatch.setattr(data, "data_files", lambda *a: [str(p)])
    monkeypatch.setattr(data, "CHUNK_ROWS", 5)
    real = data._read

    def flaky(task, field):
        if flaky.calls:
            raise NoRandomAccess("lost the filesystem mid-slice")
        flaky.calls += 1
        return real(task, field)

    flaky.calls = 0
    monkeypatch.setattr(data, "_read", flaky)
    with pytest.raises(NoRandomAccess):
        data.fetch_docs(str(tmp_path / "s.txt"), 20, dataset="org/corpus")


class FakeBuilder:
    def __init__(self, files):
        self.config = type("C", (), {"data_files": {"train": files}})()


def test_fetch_docs_falls_back_when_the_corpus_is_not_parquet(tmp_path, monkeypatch):
    """json/csv corpora and loading scripts have no row offsets — streaming still works."""
    import datasets

    (tmp_path / "corpus.json").write_text("{}")
    monkeypatch.setattr(
        datasets,
        "load_dataset_builder",
        lambda *a, **k: FakeBuilder([str(tmp_path / "corpus.json")]),
    )
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: FakeStream([{"text": "hello"}, {"text": "world"}])
    )
    out = data.fetch_docs(str(tmp_path / "s.txt"), 2, dataset="org/corpus")
    assert "world" in Path(out).read_text()


def test_progress_bar_renders_a_terminal_line_and_logs_when_piped(caplog):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    tty = Tty()
    p = Progress("fetching", unit="docs", stream=tty)
    p(5, 10)
    assert "50%" in tty.getvalue() and "5/10 docs" in tty.getvalue()
    p(10, 10)
    assert tty.getvalue().endswith("\n")  # finished bars leave the cursor on a fresh line

    with caplog.at_level("INFO"):
        piped = Progress("tokenizing", unit="bytes", stream=io.StringIO(), every=0)
        piped(2_000_000, 4_000_000)
    assert "2/4 MB" in caplog.text and not piped.open


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
        def tofile(self, out):
            out.write(b"\x00" * 8)  # partial write, then the machine dies
            raise OSError("no space left on device")

    monkeypatch.setattr(data.np, "asarray", lambda *a, **kw: Dying())
    with pytest.raises(OSError):
        tokenize_file(tok, str(corpus), str(bin_path))
    assert not bin_path.exists()
    assert not list(tmp_path.glob("*.tmp"))  # nor a half-written one beside it

    monkeypatch.undo()
    assert tokenize_file(tok, str(corpus), str(bin_path)) > 0
    assert bin_path.exists() and not list(tmp_path.glob("*.tmp"))


def test_tokenize_file_batches_without_dropping_docs(tmp_path, monkeypatch):
    """The batched writer must produce exactly what one-doc-at-a-time did, including
    across the chunk boundary a doc can straddle."""
    docs = [f"story number {i} about a robot and a dragon " * (i % 7 + 1) for i in range(50)]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(f"\n{EOT}\n".join(docs))
    tok = train_tokenizer([str(corpus)], vocab_size=512, out_path=str(tmp_path / "tok.json"))
    eot_id = tok.token_to_id(EOT)

    expected = []
    for d in docs:
        expected += tok.encode(d.strip()).ids + [eot_id]

    seen = []
    monkeypatch.setattr(data, "ENCODE_BATCH", 7)  # several batches, ragged last one
    n = tokenize_file(
        tok, str(corpus), str(tmp_path / "t.bin"), progress=lambda d, t: seen.append((d, t))
    )
    assert n == len(expected)
    assert list(np.fromfile(tmp_path / "t.bin", dtype=np.uint16)) == expected
    assert seen[-1][0] == seen[-1][1] == corpus.stat().st_size  # ends at 100%


def test_doc_batches_spans_read_boundaries(tmp_path):
    docs = ["alpha beta", "gamma", "délta with a multibyte char", "epsilon"]
    p = tmp_path / "c.txt"
    p.write_text(f"\n{EOT}\n".join(docs) + f"\n{EOT}\n")
    out = [d for batch, _ in data.doc_batches(str(p), batch=2, chunk=7) for d in batch]
    assert out == docs
