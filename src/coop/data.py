"""Tokenizer train/load + streaming token batches."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


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


def tokenize_file(tok: Tokenizer, text_path: str, out_bin: str) -> int:
    """Docs are EOT-separated in the text file; vocab is 8192 so uint16 is safe."""
    eot_id = tok.token_to_id(EOT)
    ids: list[int] = []
    for doc in Path(text_path).read_text().split(EOT):
        doc = doc.strip()
        if doc:
            ids.extend(tok.encode(doc).ids)
            ids.append(eot_id)
    np.asarray(ids, dtype=np.uint16).tofile(out_bin)
    return len(ids)


def fetch_docs(
    out_txt: str,
    n_docs: int,
    skip: int = 0,
    split: str = "train",
    dataset: str = "roneneldan/TinyStories",
    text_field: str = "text",
) -> str:
    """Stream a worker-sized shard from any HF text dataset; `skip` picks disjoint slices."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    n = 0
    with open(out_txt, "w") as f:
        for ex in ds.skip(skip).take(n_docs):
            f.write(ex[text_field].strip() + f"\n{EOT}\n")
            n += 1
    if n == 0:
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
    ap = argparse.ArgumentParser(description="fetch and tokenize a text-dataset shard")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
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
    txt = fetch_docs(
        str(out / f"shard_{tag}.txt"),
        a.docs,
        a.skip,
        a.split,
        dataset=a.dataset,
        text_field=a.text_field,
    )
    if a.train_tokenizer:
        tok = train_tokenizer([txt], a.vocab, a.tokenizer)
    else:
        tok = load_tokenizer(a.tokenizer)
    bin_path = str(out / f"shard_{tag}.bin")
    n = tokenize_file(tok, txt, bin_path)
    print(f"{n} tokens -> {bin_path}")


if __name__ == "__main__":
    main()
