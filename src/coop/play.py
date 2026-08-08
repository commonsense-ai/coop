"""Talk to a checkpoint: load what the volunteers have trained and generate from it.

The read-only half of the co-op. No token, no worker, no clone — someone who only
wants to hear what the model sounds like never has to reach the machine training it.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from coop import hubio
from coop.data import EOT, load_tokenizer
from coop.model import GPT, GPTConfig, load_canonical_state

LATEST = ("latest", "current", "newest", "live")


def pick_run(runs: list[dict], name: str | None) -> dict:
    """`latest` (or nothing) means the run being trained right now. Any run can be
    played, complete ones included — playing is read-only, unlike training."""
    if name is None or name.lower() in LATEST:
        live = [r for r in runs if r.get("status") == "live"]
        if live:
            return live[0]
        if not runs:
            raise SystemExit("no models to run — check the repo")
        return runs[0]  # every run finished: newest-first registry, so the top one
    for r in runs:
        rn = r.get("name") or ""
        if name in (rn, rn.split("-")[0]):
            return r
    known = ", ".join(str(r.get("name")) for r in runs)
    raise SystemExit(f"unknown model {name!r} — available: {known}, or `latest`")


def load_run(repo: str, cfg: dict, work: Path, name: str, revision: str = "main", device=None):
    """Weights from HF, tokenizer from the code repo. Both public: no token needed."""
    from coop.device import inference_device
    from coop.join import fetch_raw

    device = inference_device(device)
    model = GPT.from_config(GPTConfig(**cfg["model"]))
    state, meta = hubio.download_checkpoint(cfg["repos"]["model"], revision)
    load_canonical_state(model, state)
    model.eval().to(device)
    # per-run filename: the worker keeps its own tokenizer.json here, and runs differ
    tok = fetch_raw(repo, cfg["data"]["tokenizer"], work / f"tokenizer.{name}.json")
    return model, load_tokenizer(str(tok)), meta, device


def eot_id(tok) -> int | None:
    try:
        return tok.token_to_id(EOT)
    except AttributeError:
        return None


@torch.no_grad()
def stream(model, tok, prompt, n_tokens=200, temperature=0.8, top_k=50, device="cpu"):
    """Yield generated text as it arrives. Decodes the whole tail each step and emits
    the delta: byte-level BPE splits characters across tokens, so decoding one token at
    a time mangles anything non-ASCII."""
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    stop, shown, out = eot_id(tok), "", []
    for _ in range(n_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size :])
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if int(nxt) == stop:
            return
        out.append(int(nxt))
        idx = torch.cat((idx, nxt), dim=1)
        text = tok.decode(out)
        if len(text) > len(shown):  # else: a multi-byte char is still half-decoded
            yield text[len(shown) :]
            shown = text
