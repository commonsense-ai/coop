"""Tokenizer train/load + streaming token batches."""

import argparse
import logging
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
EOT_BYTES = EOT.encode()
READERS = 8  # parallel row-group fetches; the hub, not the CPU, is the limit here
CHUNK_ROWS = 2000  # rows per fetch: enough to amortize a request, small enough to stream
ENCODE_BATCH = 512  # docs per encode_batch call — the Rust tokenizer threads across them

log = logging.getLogger(__name__)


def _eta(secs: float) -> str:
    s = max(0, int(secs))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


class Progress:
    """A long wait made visible. Draws a bar on a terminal; in a daemon log (`coop start`
    redirects to a file) it prints one line every `every` seconds instead of thousands of
    carriage returns."""

    def __init__(self, label: str, unit: str = "", stream=None, every: float = 20.0):
        self.label, self.unit, self.every = label, unit, every
        self.stream = sys.stderr if stream is None else stream
        self.tty = bool(getattr(self.stream, "isatty", bool)())
        self.t0 = time.time()
        self.last = 0.0
        self.done = self.total = 0
        self.open = False

    def __call__(self, done: int, total: int) -> None:
        self.done, self.total = done, total
        now = time.time()
        finished = total and done >= total
        if not finished and now - self.last < (0.2 if self.tty else self.every):
            return
        self.last = now
        if self.tty:
            self.stream.write("\r\x1b[2K" + self.line())
            self.stream.flush()
            self.open = True
        else:
            log.info("%s", self.line())
        if finished:
            self.close()

    def line(self) -> str:
        frac = min(max(self.done / self.total, 0.0), 1.0) if self.total else 0.0
        filled = round(frac * 24)
        rate = self.done / max(time.time() - self.t0, 1e-6)
        left = f" · ~{_eta((self.total - self.done) / rate)} left" if rate and frac < 1 else ""
        bar = "█" * filled + "░" * (24 - filled)
        return f"{self.label} {bar} {100 * frac:3.0f}%{self.count()}{left}"

    def count(self) -> str:
        if self.unit == "bytes":
            return f" · {self.done / 1e6:.0f}/{self.total / 1e6:.0f} MB"
        return f" · {self.done:,}/{self.total:,} {self.unit}" if self.unit else ""

    def close(self) -> None:
        if self.open:
            self.stream.write("\n")
            self.stream.flush()
            self.open = False


def train_tokenizer(files: list[str], vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files, trainer)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def doc_batches(text_path: str, batch: int = ENCODE_BATCH, chunk: int = 1 << 22):
    """(docs, bytes read so far) over an EOT-separated text file, a chunk at a time:
    a shard is hundreds of MB and never needs to be in memory whole."""
    docs: list[str] = []
    buf, seen = b"", 0
    with open(text_path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            seen += len(block)
            parts = (buf + block).split(EOT_BYTES)
            buf = parts.pop()
            for p in parts:
                doc = p.decode("utf-8", "replace").strip()
                if doc:
                    docs.append(doc)
                if len(docs) >= batch:
                    yield docs, seen
                    docs = []
    tail = buf.decode("utf-8", "replace").strip()
    if tail:
        docs.append(tail)
    if docs:
        yield docs, seen


def tokenize_file(tok: Tokenizer, text_path: str, out_bin: str, progress=None) -> int:
    """Docs are EOT-separated in the text file; vocab is 8192 so uint16 is safe.
    Batched and appended as it goes: a FineWeb shard is ~20M tokens, which is a 600 MB
    Python list if the ids are held to the end, and single-threaded if encoded one doc
    at a time."""
    eot_id = tok.token_to_id(EOT)
    size = Path(text_path).stat().st_size
    total = 0
    # write-then-rename: an interrupted write must not leave a truncated shard at the
    # canonical path, where the "does it already exist?" check would reuse it forever
    tmp = Path(str(out_bin) + ".tmp")
    try:
        with open(tmp, "wb") as out:
            for docs, seen in doc_batches(text_path, ENCODE_BATCH):
                ids: list[int] = []
                for enc in tok.encode_batch(docs):
                    ids.extend(enc.ids)
                    ids.append(eot_id)
                np.asarray(ids, dtype=np.uint16).tofile(out)
                total += len(ids)
                if progress:
                    progress(seen, size)
        tmp.replace(out_bin)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return total


class NoRandomAccess(Exception):
    """This dataset can't be opened at a row offset — the caller falls back to streaming."""


def _open(url: str):
    if url.startswith("hf://"):
        from coop import hubio

        return hubio.dataset_fs().open(url, "rb")
    return open(url, "rb")


def data_files(dataset: str, config: str | None, split: str) -> list[str]:
    """The split's parquet files in stream order. Anything else — a loading script,
    csv/json shards, a split this build doesn't have — has no row offsets to seek to."""
    try:
        from datasets import load_dataset_builder

        files = load_dataset_builder(dataset, config).config.data_files or {}
        urls = [str(f) for f in files.get(split) or []]
    except Exception as e:  # unresolvable here means unresolvable for streaming too
        raise NoRandomAccess(str(e)) from e
    if not urls or not all(u.endswith(".parquet") for u in urls):
        raise NoRandomAccess(f"{split} split of {dataset} is not parquet")
    if not all(u.startswith("hf://") or Path(u).exists() for u in urls):
        raise NoRandomAccess(f"{split} split of {dataset} is not on the hub")
    return urls


def plan(files: list[str], skip: int, n_docs: int) -> list[tuple]:
    """Row groups covering rows [skip, skip+n_docs), as (url, metadata, groups, drop, keep)
    fetches. Only the footers up to the file the slice ends in are ever read."""
    cur, end, pos, tasks = skip, skip + n_docs, 0, []
    for url in files:
        if cur >= end:
            break
        with _open(url) as f:
            md = pq.ParquetFile(f).metadata
        if pos + md.num_rows <= cur:  # whole file is before the slice: footer only
            pos += md.num_rows
            continue
        lo, hi = cur - pos, min(md.num_rows, end - pos)
        group: list[int] = []
        start = first = 0
        for i in range(md.num_row_groups):
            rows = md.row_group(i).num_rows
            s, e = start, start + rows
            start = e
            if s >= hi:
                break
            if e <= lo:
                continue
            if not group:
                first = s
            group.append(i)
            if e - first >= CHUNK_ROWS or e >= hi:
                tasks.append(
                    (url, md, tuple(group), max(lo, first) - first, min(hi, e) - max(lo, first))
                )
                group = []
        cur, pos = pos + hi, pos + md.num_rows
    return tasks


def _read(task: tuple, text_field: str) -> list[str]:
    url, md, groups, drop, keep = task
    with _open(url) as f:
        table = pq.ParquetFile(f, metadata=md).read_row_groups(list(groups), columns=[text_field])
    return table.column(text_field).slice(drop, keep).to_pylist()


def _ordered(fn, items: list, workers: int):
    """fn over items in parallel, results in order, at most `workers` fetches in flight:
    the shard is written as it arrives instead of being assembled in memory."""
    with ThreadPoolExecutor(max(1, min(workers, len(items) or 1))) as ex:
        futures: deque = deque()
        for item in items:
            futures.append(ex.submit(fn, item))
            if len(futures) > workers:
                yield futures.popleft().result()
        while futures:
            yield futures.popleft().result()


def _parquet_docs(dataset, config, split, skip, n_docs, text_field, emit) -> None:
    tasks = plan(data_files(dataset, config, split), skip, n_docs)
    for docs in _ordered(lambda t: _read(t, text_field), tasks, READERS):
        emit(docs)


def _stream_docs(dataset, config, split, skip, n_docs, text_field, emit) -> None:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    try:  # arrow batches skip a record batch at a time instead of a row at a time
        ds = ds.with_format("arrow")
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        pass
    for chunk in ds.skip(skip).take(n_docs):
        col = chunk[text_field]
        emit(col.to_pylist() if hasattr(col, "to_pylist") else [col])


def fetch_docs(
    out_txt: str,
    n_docs: int,
    skip: int = 0,
    split: str = "train",
    dataset: str = "roneneldan/TinyStories",
    text_field: str = "text",
    config: str | None = None,
    progress=None,
) -> str:
    """A worker-sized shard from any HF text dataset; `skip` picks disjoint slices.
    Parquet corpora are opened straight at the row offset: streaming has to read every
    row before `skip` first, which is 15 minutes into FineWeb-Edu's sample-10BT and
    hours near its end. `config` selects a subset (e.g. sample-10BT)."""
    written = 0
    with open(out_txt, "w") as f:

        def emit(docs: list[str]) -> None:
            nonlocal written
            f.write("".join(d.strip() + f"\n{EOT}\n" for d in docs))
            written += len(docs)
            if progress:
                progress(written, n_docs)

        if progress:
            progress(0, n_docs)  # a bar at 0% while the offset is located beats silence
        try:
            _parquet_docs(dataset, config, split, skip, n_docs, text_field, emit)
        except NoRandomAccess as why:
            log.info("%s — streaming to doc %d instead; this part is slow", why, skip)
            _stream_docs(dataset, config, split, skip, n_docs, text_field, emit)
    if written == 0:
        # fail here with a clear cause, not later as "cannot mmap an empty file"
        raise ValueError(f"no {split} docs in {dataset} at skip={skip}: past the end")
    return out_txt


def iter_batches(token_bin: str, batch: int, block: int, seed: int = 0):
    data = np.memmap(token_bin, dtype=np.uint16, mode="r")
    assert len(data) > block + 1, f"{token_bin} too small for block_size={block}"
    rng = np.random.default_rng(seed)
    while True:
        ix = rng.integers(0, len(data) - block - 1, size=batch)
        x = torch.stack([torch.from_numpy(data[i : i + block].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block].astype(np.int64)) for i in ix]
        )
        yield x, y


def main():
    from coop import setup_logging

    setup_logging()  # a piped run still gets progress lines; a terminal gets the bar
    ap = argparse.ArgumentParser(description="fetch and tokenize a text-dataset shard")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--config-name", default=None, help="dataset subset, e.g. sample-10BT")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tokenizer", default="tokenizer/tinystories-8k.json")
    ap.add_argument(
        "--train-tokenizer", action="store_true", help="maintainer only: retrain the shared vocab"
    )
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.skip}_{a.docs}" if a.split == "train" else f"{a.split}_{a.skip}_{a.docs}"
    fetching = Progress("fetching  ", unit="docs")
    txt = fetch_docs(
        str(out / f"shard_{tag}.txt"),
        a.docs,
        a.skip,
        a.split,
        dataset=a.dataset,
        text_field=a.text_field,
        config=a.config_name,
        progress=fetching,
    )
    fetching.close()
    if a.train_tokenizer:
        tok = train_tokenizer([txt], a.vocab, a.tokenizer)
    else:
        tok = load_tokenizer(a.tokenizer)
    bin_path = str(out / f"shard_{tag}.bin")
    tokenizing = Progress("tokenizing", unit="bytes")
    n = tokenize_file(tok, txt, bin_path, progress=tokenizing)
    tokenizing.close()
    print(f"{n} tokens -> {bin_path}")


if __name__ == "__main__":
    main()
