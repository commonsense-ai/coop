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
- aggregator tick: `uv run python -m coop.aggregate`
- release: bump version in pyproject.toml + src/coop/__init__.py + npm/package.json,
  then `git tag vX.Y.Z && git push origin vX.Y.Z` — Actions publishes PyPI (coop-ai)
  and npm (coop-ai) via OIDC trusted publishing

## Architecture invariants (do not violate)

- Weights and optimizer state live ONLY in the HF model repo as safetensors. Never in git.
- Pseudo-gradients arrive ONLY as PRs on the HF dataset repo (`create_commit(create_pr=True)`).
- The aggregator is stateless: every tick reads all state from the repos, does one outer
  step, writes state back, exits. A crashed tick resumes on the next cron fire.
- All network I/O goes through `src/coop/hubio.py`.
- Batch API calls. One `create_commit` per checkpoint upload; never per-file calls in a loop.
  GITHUB_TOKEN is capped at 1,000 REST requests/hour/repo.
- Keep an aggregator tick under 10 minutes; Actions kills jobs at 6 hours.
- Close processed inbox PRs every tick (HF repos cap at 100k files).
- The ledger + leaderboard live on the unprotected `ledger` branch; `main` requires an
  approved PR (repo ruleset), so workflows must never push to `main`.
- GitHub blocks files > 100 MB; large binaries go to HF only.
- `config/run.yaml` is the single source of run configuration.
- Never pin torch to the CPU index in `[tool.uv.sources]` unqualified: uv applies a
  package's sources to `uvx`/`pip install git+...`, so the pin follows the package onto
  volunteers' machines and strands GPU donors on CPU. CI opts in with `--extra cpu`.

## Style

- Terse, direct. Comments explain WHY, not WHAT, and are rare.
- ruff is the linter and formatter; line length 100.
- No AI-attribution text anywhere: not in code, commits, or PRs.
