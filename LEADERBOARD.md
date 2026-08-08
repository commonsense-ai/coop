# Leaderboard

Outer step **116** — updated 2026-08-08T23:22:29+00:00.

Val loss at step 116: **4.5311** — sample:

> The most important thing to understand about how we can survive our lives and our interactions. Because we don’t tend to leave our lives, we don’t fail to feel the same at all. So, if we don’t understand how we can deal with our lives or how we can survive our lives? You can tell us how we can survive our lives without seeing our identity. We can’t wait until we really understand how we can survive our lives without seeing our identity. Just imagine what we are going to do! Any good thing is wrong! Now we don’t fail to

**Running coop from before 0.3.0?** If `coop update` answers `invalid choice`,
your copy predates it. Reinstall once —
`uv tool install --force git+https://github.com/commonsense-ai/coop` — or use
`npx coop-ai`, which always runs the current code. After that coop keeps itself
current, and `coop update --auto on` lets it do so between rounds.

Score = tokens contributed × reputation. Reputation is an EMA of acceptance
(alpha=0.1): rejected submissions lower it, accepted ones restore it.
CPU work (tokenize / dedup / filter / eval) earns tokens on this same board.
Hardware lists every machine a contributor has trained on, biggest share first.

| # | Contributor | Hardware | Accepted | Tokens | Reputation | Score |
|---|-------------|----------|----------|--------|------------|-------|
| 1 | cjtsolutions | nvidia-gpu | 80 | 1,637,732,352 | 1.000 | 1,637,369,016 |
| 2 | naloxene | nvidia-gpu | 61 | 1,027,452,928 | 1.000 | 1,027,452,928 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
