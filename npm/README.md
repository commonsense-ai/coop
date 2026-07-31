# coop

Help train a real language model with your computer.

```sh
npx coop-ai start
```

That's the whole thing. Your machine downloads the current community model
(~60 MB), trains it on a slice of short stories a few minutes at a time, and
submits each finished round publicly under your Hugging Face name. An
aggregator merges everyone's rounds into the shared model, and you earn credit
on the [public leaderboard](https://github.com/commonsense-ai/coop/blob/ledger/LEADERBOARD.md).

The first run walks you through the one-time setup: a free
[Hugging Face account](https://huggingface.co/join) and a write token. After
that it's zero-setup.

## Commands

```sh
npx coop-ai start     # begin contributing (runs in the background)
npx coop-ai status    # live progress + ETA, your rank, who else is training
npx coop-ai logs -f   # watch the worker do its thing
npx coop-ai stop      # submits work-in-progress, shows your impact, stops
```

`start --rounds 3` contributes a fixed number of rounds and stops by itself.
Your GPU is used automatically when you have one (Apple Silicon and NVIDIA);
plain CPUs work too.

## Requirements

- Node 18+ (for this launcher)
- [uv](https://docs.astral.sh/uv/) — the launcher runs the Python CLI through
  it, and prints install instructions if it's missing

## What is this, really?

A ~15M-parameter language model pretrained from scratch entirely by
volunteers — no server, no funding: DiLoCo-style low-communication training
where workers submit pseudo-gradients as pull requests on a public Hugging
Face repo and a stateless GitHub Actions job aggregates them. Details, code,
and the architecture writeup live in the
[GitHub repository](https://github.com/commonsense-ai/coop). The model itself
is [commonsense-ai/tinystories-15m](https://huggingface.co/commonsense-ai/tinystories-15m).

License: Apache-2.0
