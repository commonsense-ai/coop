# Leaderboard

Outer step **31** — updated 2026-08-08T04:58:03+00:00.

Val loss at step 31: **5.5596** — sample:

> The most important thing to understand about the same thing. For example, in the next decade, we know the things we get to learn. When, we know that, even say that the two-dimensional words, since the same place, not to make the only ones that is, that’s like the more common, so, that’s so many things that’s an average and more popular, and that’s still a lot of a bit is a good, for many parts of these things in the world, so that’s the next to get on, the time of us as we can’t understand

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
| 1 | naloxene | gpu | 39 | 501,526,528 | 1.000 | 501,526,528 |
| 2 | cjtsolutions | gpu | 20 | 355,491,840 | 1.000 | 355,491,840 |
| 3 | ezshroom | gpu | 28 | 105,676,800 | 1.000 | 105,676,800 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
