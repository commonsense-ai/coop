# AGENTS.md

## Project

coop: a community-pretrained small LM. Workers run DiLoCo inner loops and submit
pseudo-gradients as PRs on a public HF dataset repo; a stateless GitHub Actions cron
job aggregates them into outer steps on a public HF model repo.

## Commands

- setup: `uv sync`
- test: `uv run pytest -q`
- launcher smoke: `npm/test/smoke.sh node` / `npm/test/smoke.sh bun` (needs that runtime)
- lint: `uv run ruff check .`
- format: `uv run ruff format .`
- worker round: `uv run python -m coop.trainer --data data/<shard>.bin`
- speed check: `uv run python -m coop.bench` (no network; `--device`, `--batch-size`,
  `--compile` to explore a machine, and the phase split says where a step goes)
- aggregator tick: `uv run python -m coop.aggregate`
- release: bump version in pyproject.toml + src/coop/__init__.py + npm/package.json,
  then `git tag vX.Y.Z && git push origin vX.Y.Z` — Actions publishes PyPI (coop-ai)
  and npm (coop-ai) via OIDC trusted publishing, then announces the release as
  `release.json` on the `ledger` branch

## Architecture invariants (do not violate)

- Weights and optimizer state live ONLY in the HF model repo as safetensors. Never in git.
- Pseudo-gradients arrive ONLY as PRs on the HF dataset repo (`create_commit(create_pr=True)`).
- The aggregator is stateless: every tick reads all state from the repos, does one outer
  step, writes state back, exits. A crashed tick resumes on the next cron fire.
- All network I/O goes through `src/coop/hubio.py`.
- Every outer step is a new commit, so every round pins a new sha and the hub cache would
  grow by a whole checkpoint per step. Anything that downloads in a loop prunes after
  itself (`hubio.prune_cache`) or it fills a volunteer's disk overnight.
- Batch API calls. One `create_commit` per checkpoint upload; never per-file calls in a loop.
  GITHUB_TOKEN is capped at 1,000 REST requests/hour/repo.
- Keep an aggregator tick under 10 minutes; Actions kills jobs at 6 hours.
- Close processed inbox PRs every tick (HF repos cap at 100k files).
- The ledger + leaderboard live on the unprotected `ledger` branch; `main` requires an
  approved PR (repo ruleset), so workflows must never push to `main`.
- GitHub blocks files > 100 MB; large binaries go to HF only.
- `config/run.yaml` is the single source of run configuration.
- The val-loss series (`ledger/history.jsonl`) is append-only, one point per outer step
  that actually landed, and every point carries the eval spec that produced it. Trend
  math compares only points sharing the current spec, so widening the eval starts a new
  segment instead of faking a drop no training caused. Eval params come from
  `trend.eval_params` at both the call site and the fingerprint — never inline defaults.
- `release.json` on `ledger` is the update channel volunteers poll; it is compared
  against `__version__`, so a tag that forgets the version bump tells every volunteer
  to update forever. Workers only ever restart between rounds, never mid-round.
- Auto-update is off by default and every path respects that, with one exception: a
  worker that cannot finish a single round (failures, restarts spent, nothing landed)
  checks the channel and takes a fix anyway. It is already not doing what the volunteer
  asked for. Do not widen this to workers that are merely slow or unlucky.
- A pseudo-gradient is only worth sending for `staleness.tau_max` outer steps. Anything
  parked on disk past that is discarded, not uploaded — the aggregator would reject it.
- Never import `torch_xla` at module scope. The import is what registers the `xla`
  backend, and it starts the XLA runtime as a side effect — it stays gated behind a
  `find_spec` check so only a machine with a TPU ever pays for it.
- Never pin torch to the CPU index in `[tool.uv.sources]` unqualified: uv applies a
  package's sources to `uvx`/`pip install git+...`, so the pin follows the package onto
  volunteers' machines and strands GPU donors on CPU. CI opts in with `--extra cpu`.

## Style

- Terse, direct. Comments explain WHY, not WHAT, and are rare.
- ruff is the linter and formatter; line length 100.
- No AI-attribution text anywhere: not in code, commits, or PRs.
