import numpy as np
import torch

from coop.data import EOT, iter_batches, load_tokenizer, tokenize_file, train_tokenizer

CORPUS = (
    "Once upon a time there was a tiny robot. "
    "The robot liked to read stories about dragons and stars. "
) * 50


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
