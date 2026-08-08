# Leaderboard

Outer step **92** — updated 2026-08-08T17:14:49+00:00.

Val loss at step 92: **4.7227** — sample:

> The most important thing to understand about a variety of diseases and to develop their symptoms. People are at a higher risk than others. People who don’t live in a clinical phase or who don’t live in a family history or who don’t live in a life-changing environment — while those who live in a single family and who do live in a living-age household are more likely than those who live in one generation — but there are a few things that don’t live in these days. How do these symptoms happen? What are the signs of those symptoms? - What are the symptoms of a person

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
| 1 | cjtsolutions | nvidia-gpu | 58 | 1,190,449,152 | 0.998 | 1,187,767,261 |
| 2 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 3 | ezshroom | apple-gpu | 138 | 415,756,288 | 1.000 | 415,756,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
