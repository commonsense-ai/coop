"""Eval harness: val loss + sample generations for a checkpoint."""

import argparse
import statistics

import torch

from coop import hubio, load_config
from coop.data import iter_batches, load_tokenizer
from coop.model import GPT, GPTConfig, load_canonical_state


@torch.no_grad()
def eval_val_loss(model, val_bin: str, batch: int = 8, iters: int = 20, seed: int = 0) -> float:
    model.eval()
    it = iter_batches(val_bin, batch, model.cfg.block_size, seed=seed)
    losses = []
    for _ in range(iters):
        x, y = next(it)
        _, loss = model(x, y)
        losses.append(loss.item())
    return statistics.mean(losses)


@torch.no_grad()
def sample(model, tokenizer, prompt: str, n_tokens: int = 200, temperature=0.8, top_k=50) -> str:
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    out = model.generate(idx, n_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="evaluate the current checkpoint")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument("--val", help="validation token .bin")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    a = ap.parse_args()

    cfg = load_config(a.config)
    model = GPT.from_config(GPTConfig(**cfg["model"]))
    state, meta = hubio.download_checkpoint(cfg["repos"]["model"])
    load_canonical_state(model, state)
    print(f"checkpoint: step {meta['step']}")
    if a.val:
        print(f"val loss: {eval_val_loss(model, a.val):.4f}")
    tok = load_tokenizer(cfg["data"]["tokenizer"])
    print(sample(model, tok, a.prompt, a.tokens))


if __name__ == "__main__":
    main()
