# Contributing

There are three ways to contribute. All of them earn credit on the same
[leaderboard](https://github.com/commonsense-ai/decentralised-ai-training-poc/blob/ledger/LEADERBOARD.md).

## 1. Donate compute (GPU or CPU)

You need a free Hugging Face account and a **write** token (Settings → Access Tokens).

```sh
uv sync
export HF_TOKEN=hf_...
uv run python -m coop.data --skip <N> --docs 20000   # pick an N others aren't using
uv run python -m coop.trainer --data data/shard_<N>_20000.bin
```

Each worker round downloads the latest checkpoint, trains `inner.h_steps` local steps,
and opens a PR on the inbox dataset repo. The aggregator picks it up on the next tick
(~15 min), credits your HF username in the ledger, and closes the PR with a comment.
A rejected submission (stale, malformed, or an outlier) costs reputation, so keep your
client unmodified and re-download the checkpoint before every round.

- `--device cuda` / `--device mps` if you have a GPU; plain CPU works for Stage 1.
- `--dry-run` shows what would be submitted without touching the network.

## 2. CPU-tier data work

Tokenization, dedup, filtering, and eval runs are credited by tokens processed, same
as training. Open a GitHub issue to claim a slice, submit results the same way
(`submissions/` PR with a `.json` sidecar; set `"tier": "cpu"`).

## 3. Code

```sh
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```

- Read [AGENTS.md](AGENTS.md) first; its architecture invariants are non-negotiable.
- Small focused PRs. Tests for any new behavior.
- Never commit weights, tokens, or anything over a few MB — large binaries live on HF.
