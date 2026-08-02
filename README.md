# coop

[![ci](https://github.com/commonsense-ai/coop/actions/workflows/ci.yml/badge.svg)](https://github.com/commonsense-ai/coop/actions/workflows/ci.yml)
[![leaderboard](https://img.shields.io/badge/%F0%9F%8F%86_leaderboard-live-brightgreen)](https://github.com/commonsense-ai/coop/blob/ledger/LEADERBOARD.md)

A small language model pretrained by volunteers. No server, no funding, no daemon —
the whole training loop runs on donated consumer hardware plus the free tiers of
Hugging Face and GitHub Actions.

## Status

Training is live. As of 2026-07-31: outer step 19, validation loss **2.88** (random
init starts at 9.01), **29M tokens** contributed — ~10% of the ~300M-token Stage 1
target. Current numbers and sample output:
[leaderboard](https://github.com/commonsense-ai/coop/blob/ledger/LEADERBOARD.md).

The full loop is production-proven, not just designed: multiple volunteers on
different machines (Apple Silicon and plain CPU) have trained the same outer step
and been averaged into one update — the actual data parallelism. A submission that
raced a tick was accepted one step later at reduced staleness weight; repeat rounds
from one user merged into a single vote (no token farming); `coop stop` flushed a
half-finished round instead of discarding it; and the inbox drains to zero every
tick.

## How it works

DiLoCo-style low-communication data parallelism:

1. **Workers** (you) download the current checkpoint from the
   [HF model repo](https://huggingface.co/commonsense-ai/tinystories-15m),
   run `H` local AdamW steps on a TinyStories shard, and compute a
   **pseudo-gradient**: `delta = theta_outer - theta_local`.
2. **Submission** is a pull request against a public
   [HF dataset repo](https://huggingface.co/datasets/commonsense-ai/tinystories-15m-inbox)
   (the "gradient inbox"), opened with `create_commit(create_pr=True)`. Any free HF
   account with a write token can submit; the maintainer grants no permissions.
3. **The aggregator** is a GitHub Actions cron job (scheduled every 15 min;
   GitHub's shared scheduler actually fires anywhere from minutes to a few hours
   apart — the protocol tolerates any cadence). Each tick is stateless: it reads the checkpoint and the open inbox PRs, drops over-stale
   submissions, clips and cosine-gates the rest, robust-aggregates them
   (trimmed mean / geometric median), takes one Nesterov outer step, uploads the
   new checkpoint, credits contributors in the
   [ledger](https://github.com/commonsense-ai/coop/tree/ledger/ledger),
   regenerates the
   [leaderboard](https://github.com/commonsense-ai/coop/blob/ledger/LEADERBOARD.md),
   and closes the processed PRs. Ledger state lives on the `ledger` branch;
   `main` only changes through approved pull requests.

Weights and optimizer state live **only** on Hugging Face (safetensors). Git holds
code, config, and the contributor ledger.

## Stage 1

~15M parameter decoder-only transformer (6 layers, 6 heads, d=396, 512 context,
8k byte-level BPE vocab, tied embeddings) on TinyStories. It trains on a laptop CPU.

## Donate compute

One command if you have Node (it uses [uv](https://docs.astral.sh/uv/) under
the hood and tells you how to get it):

```sh
npx coop-ai start
```

Or install with uv directly:

```sh
uv tool install git+https://github.com/commonsense-ai/coop
coop start
```

(`uv tool install coop-ai` / `pipx install coop-ai` once the PyPI package
clears review.)

The first run asks you to paste a Hugging Face
[write token](https://huggingface.co/settings/tokens) (free account) — after that
it's zero-setup. The worker runs in the background: it builds you a personal data
shard (a slice derived from your username so volunteers don't overlap), then trains
and submits rounds until you say otherwise.

```sh
coop status    # live progress + ETA, your rank, who else is training
coop logs -f   # watch it work
coop stop      # stop contributing; `coop start` resumes any time
```

`coop start --rounds 3` contributes a fixed number of rounds and stops by itself.

Prefer a foreground one-off? `uvx --from git+https://github.com/commonsense-ai/coop
coop-join --hf-token hf_...` runs rounds until ctrl-c (`--once` for a single round,
`--device cuda|mps|cpu` to override).

From a clone, the equivalent is:

```sh
uv sync
export HF_TOKEN=hf_...
uv run python -m coop.data --skip 0 --docs 20000
uv run python -m coop.trainer --data data/shard_0_20000.bin --loop
```

Run it as often as you like. Accepted submissions earn tokens on the leaderboard;
CPU-only machines can also contribute tokenization, dedup, filtering, and eval runs
(see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Maintainer bootstrap

```sh
export HF_TOKEN=hf_...   # write access to both HF repos
uv run python -m coop.data --train-tokenizer --skip 0 --docs 50000
uv run python -m coop.aggregate --init          # genesis checkpoint at step 0
```

Then set the `HF_TOKEN` repo secret on GitHub so `.github/workflows/aggregate.yml`
can run the tick.

## Development

```sh
uv run pytest -q
uv run ruff check .
```

Everything is configured in [config/run.yaml](config/run.yaml). Architecture rules
live in [AGENTS.md](AGENTS.md).

License: Apache-2.0.
