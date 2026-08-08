# coop

Help train a real language model with your computer.

```sh
npx coop-ai start     # bun: bunx coop-ai start
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
npx coop-ai start       # begin contributing (runs in the background)
npx coop-ai status      # live progress + ETA, your rank, which GPU is working
npx coop-ai logs -f     # watch the worker do its thing
npx coop-ai stop        # submits work-in-progress, shows your impact, stops
npx coop-ai run latest  # talk to the model trained so far — no account needed
npx coop-ai update      # get the newest coop (--auto on to keep it current)
```

`update` checks the release channel and installs a newer coop the same way you got
this one. `update --auto on` lets a running worker take new versions between rounds
(never mid-round); it's off until you turn it on.

`run latest` downloads the current checkpoint and takes prompts, so you can hear
what the co-op has built without training anything. `run tinystories` plays the
finished stage-1 model, which is far more coherent than a run still in progress.
Add `--prompt "..."` to generate once and exit.

Using [Bun](https://bun.sh)? Swap `npx` for `bunx` — same commands, same
launcher.

`start --rounds 3` contributes a fixed number of rounds and stops by itself.
Your GPU is used automatically when you have one (Apple Silicon and NVIDIA);
plain CPUs work too.

## Requirements

- Node 18+ or Bun 1.0+ (for this launcher)
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

License: undecided — all rights reserved for now, see LICENSE.md in the repository.
