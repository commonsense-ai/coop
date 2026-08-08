# coop

[![ci](https://github.com/commonsense-ai/coop/actions/workflows/ci.yml/badge.svg)](https://github.com/commonsense-ai/coop/actions/workflows/ci.yml)
[![leaderboard](https://img.shields.io/badge/%F0%9F%8F%86_leaderboard-live-brightgreen)](https://github.com/commonsense-ai/coop/blob/ledger/LEADERBOARD.md)

A small language model pretrained by volunteers. No server, no funding, no daemon —
the whole training loop runs on donated consumer hardware plus the free tiers of
Hugging Face and GitHub Actions.

## Status

**Stage 2 is live**: a ~145M-param model pretraining from scratch on FineWeb-Edu.
Stage 1 (15M on TinyStories) completed past its Chinchilla-optimal budget — proof
that the whole mechanism works. Current numbers and sample output:
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
   [HF model repo](https://huggingface.co/commonsense-ai/fineweb-150m),
   run `H` local AdamW steps on a personal data shard, and compute a
   **pseudo-gradient**: `delta = theta_outer - theta_local`.
2. **Submission** is a pull request against a public
   [HF dataset repo](https://huggingface.co/datasets/commonsense-ai/fineweb-150m-inbox)
   (the "gradient inbox"), opened with `create_commit(create_pr=True)`. Any free HF
   account with a write token can submit; the maintainer grants no permissions.
3. **The aggregator** is a GitHub Actions cron job (scheduled every 5 min;
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

## Stage 2 (current run)

~145M parameter decoder-only transformer (12 layers, 14 heads, d=896, 1024 context,
32k byte-level BPE vocab, tied embeddings) on
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — real
educational web text. GPUs and Apple Silicon pull their weight here; plain CPUs are
better suited to CPU-tier work (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Stage 1 (complete)

The proof run: a 15M-param model pretrained past its Chinchilla-optimal budget on
TinyStories by volunteers in six days, val loss 9.01 → 2.8. It stays usable forever —
[commonsense-ai/tinystories-15m](https://huggingface.co/commonsense-ai/tinystories-15m)
has the weights, the model card, and a working load-and-generate snippet. The final
stage-1 leaderboard is archived as `LEADERBOARD-stage1.md` on the ledger branch.

## Try the model

Talk to whatever the volunteers have trained so far — no account, no clone, no
training:

```sh
npx coop-ai run latest     # bun: bunx coop-ai run latest
```

It downloads the current checkpoint (cached after the first time), then takes
prompts and writes what comes next. `run tinystories` plays the finished stage-1
model instead, which is far more coherent than a run still in progress. One-off
and pipeable:

```sh
coop run latest --prompt "The best way to learn mathematics is"
echo "Once upon a time" | coop run tinystories
```

`--tokens`, `--temperature`, `--top-k`, and `--device` are there when you want
them; `--revision` pins a specific checkpoint.

## Donate compute

One command if you have Node or [Bun](https://bun.sh) (it uses
[uv](https://docs.astral.sh/uv/) under the hood and tells you how to get it):

```sh
npx coop-ai start     # bun: bunx coop-ai start
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
coop status    # live progress + ETA, your rank, which GPU is doing the work
coop logs -f   # watch it work
coop stop      # stop contributing; `coop start` resumes any time
```

`coop start --rounds 3` contributes a fixed number of rounds and stops by itself.

### Staying current

Every release publishes `release.json` to the `ledger` branch, and that is the only
thing a volunteer's machine polls. `coop status` says when a newer version is out,
and:

```sh
coop update            # get it now (works out how you installed coop)
coop update --check    # what's new, install nothing
coop update --auto on  # keep it current by itself
```

With `--auto on`, a running worker adopts the new version **between rounds** — never
mid-round, so a restart can't cost you trained work — and picks its training back up
where it left off. Off by default: nothing on your machine changes unless you ask.
A clone is left alone either way; there, `git pull` is the update.

One exception, and it only fires when your worker is already broken. If it cannot
finish a single round — several failures in a row, a restart, still nothing — it will
take a newer version if one exists, even with auto-update off. You asked it to
contribute, it isn't contributing, and a fix on the channel is the only thing left
that can change that. `coop status` says so, and `coop logs` shows the attempt.

Self-updating only works from 0.3.0 on, so an install older than that can't reach it:

```
coop: error: argument cmd: invalid choice: 'update'
```

That means the `coop` on your PATH predates the command. Reinstall it once —
`uv tool install --force git+https://github.com/commonsense-ai/coop`, or just use
`npx coop-ai`, which always resolves the current code — and it keeps itself current
from then on. `uv tool list` is worth a look if you installed early: the package was
once named `coop` rather than `coop-ai`, and a leftover of the old name claims the
same `coop` executable.

Prefer a foreground one-off? `uvx --from git+https://github.com/commonsense-ai/coop
coop-join --hf-token hf_...` runs rounds until ctrl-c (`--once` for a single round,
`--device cuda|mps|cpu` to override).

From a clone, the equivalent is:

```sh
uv sync   # NVIDIA box? add `uv pip install --torch-backend auto torch`:
          # the lockfile ships CPU wheels so aggregator ticks stay fast
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
