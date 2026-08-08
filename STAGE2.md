# Stage 2 launch runbook

> **Status: launched.** Tokenizer trained (`tokenizer/fineweb-32k.json`),
> `train_docs` pinned (9,622,101 of sample-10BT; 50k-doc tail reserved for val),
> val.bin built from the reserved tail and uploaded, genesis at
> [commonsense-ai/fineweb-150m](https://huggingface.co/commonsense-ai/fineweb-150m),
> config cut over (stage 1 archived as `config/stage1.yaml`). Remaining mid-run
> work: verification via redundant shard assignment; multi-run ledger schema.
>
> **3B passed at outer step 113** (2026-08-08, val 4.6002, 3,090,272,256 tokens in
> 5.8 days from four contributors). Val loss was still descending — ~0.11 over the
> preceding 20 steps — so `goal_tokens` went to 6B rather than declaring the run done.

Stage 1 (tinystories-15m) is complete: ~363M tokens, val 2.83, four contributors.
Stage 2 scales to a ~145M-param model on FineWeb-Edu. `config/stage2.yaml` is the
draft run config; this file is everything left between here and launch day.
Work through it top to bottom — each step is mechanical once the one above it is done.

## Already landed (config-gated, dormant in stage 1)

- Val-loss circuit breaker (`outer.max_val_regression`) — discards steps that damage
  the model; no credit, no reputation harm.
- Probation weight (`outer.probation_weight`) — unproven identities vote at reduced
  influence until their first accepted submission; credit untouched.
- int4 pseudo-gradients (`inner.quantize: int4`) — ~75MB per 145M-param submission.
- Config-driven dataset (`data.hf_dataset` / `data.text_field`) — FineWeb-Edu is a
  config change, not a code change.
- Identity stamped from the PR's authenticated author; worker resolves whoami once.

## Remaining engineering (ordered; 1–2 are launch-blocking)

1. **One accumulating PR per user per step.** A fast GPU produces a ~75MB PR per
   round; at stage-2 scale the tick's download budget breaks. Worker keeps a
   token-weighted running average of its rounds locally and refreshes a single PR
   (`create_commit(revision="refs/pr/N")`), falling back to a fresh PR if the tick
   closed it mid-update. Aggregator unchanged.
2. **Per-machine seed.** `coop.join` seeds rounds with the round counter only —
   same user on two machines trains identical batches. Mix a stable machine
   fingerprint (hostname hash) into the seed base.
3. **Credit policy decision (Pierre):** largest-round-only vs sum-of-distinct-rounds
   (near-duplicate deltas collapsed by pairwise cosine, per-round plausibility cap
   at h_max × batch × block). Decides whether multi-machine users earn for machines
   two and three.
4. **Verification (can land mid-run):** redundant shard assignment — the same
   (shard, seed, checkpoint) occasionally given to two workers, deltas compared;
   disagreement flags both for audit. Changes the attack economics from "free" to
   "must run the real computation".
5. **Multi-run ledger schema:** freeze it so stage-1 credit is carried forward
   (founders keep their history when LEADERBOARD resets for stage 2).

## Launch-day mechanics (in order)

1. **Decide names**: model repo (`commonsense-ai/fineweb-150m`?) + inbox repo.
   Update `config/stage2.yaml` repos block.
2. **Train the 32k tokenizer** (maintainer, one-off, ~20 min):
   `uv run python -m coop.data --dataset HuggingFaceFW/fineweb-edu --train-tokenizer
   --vocab 32768 --tokenizer tokenizer/fineweb-32k.json --docs 200000`
3. **Set `data.train_docs`** from the chosen subset's row count:
   `curl -s 'https://datasets-server.huggingface.co/size?dataset=HuggingFaceFW/fineweb-edu'`
   (pick the subset — `sample-10BT` is the likely v1 — and use its `num_rows`).
4. **Build + upload val.bin**:
   `uv run python -m coop.data --dataset HuggingFaceFW/fineweb-edu --split train
   --skip <past-train-shards> --docs 2000 --tokenizer tokenizer/fineweb-32k.json`
   then upload as `val.bin` to the new model repo. (FineWeb-Edu has no val split;
   reserve a doc range no username can hash into.)
5. **Genesis checkpoint**:
   `HF_TOKEN=... uv run python -m coop.aggregate --init --config config/stage2.yaml`
6. **Cutover**: archive stage 1 — final leaderboard copied to `LEADERBOARD-stage1.md`
   on the ledger branch, model card gets a "training complete" banner — then
   `git mv config/stage2.yaml config/run.yaml` (PR). Every worker and tick reads
   run.yaml from main, so this single merge IS the launch: `coop start` worldwide
   begins training stage 2 on its next round.
7. **Announce**: follow-up post + comment in the stage-1 HN thread; recruit GPU
   contributors specifically (~5–10 GPUs sustained gets ~3B tokens in weeks, not
   months; publish the tokens/sec device table: ~1.6k slow CPU / ~8k fast CPU /
   ~21k Apple Silicon from stage 1).

## Numbers to sanity-check before launch

- Params: 12L·14H·896d = 146,024,704 tied. Chinchilla-optimal ≈ 3B tokens; SmolLM-style
  target 5–10B (state the goal in the announcement).
- Worker memory: f32 weights 580MB + AdamW 1.2GB + activations ≈ fits 16GB laptops
  at batch 4×1024; CPU-only machines fall below useful throughput — expect a
  GPU-tier run. Revisit bf16 if volunteers hit memory limits.
- Checkpoint download per round: ~580MB (weights only). At 5-minute ticks a worker
  re-downloads per step — bandwidth-heavy; consider longer tick spacing or delta
  checkpoints if volunteers complain.
- Tick budget: N users × ~75MB submissions must stay under the 10-minute invariant
  with the 100MB/s Actions runner — fine to ~30 active users with accumulating PRs,
  NOT fine without them (hence launch-blocking).
