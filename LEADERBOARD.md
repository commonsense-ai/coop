# Leaderboard

Outer step **48** — updated 2026-08-08T08:23:18+00:00.

Val loss at step 48: **5.1547** — sample:

> The most important thing to understand about the effects of death may help you to keep the decisions you think. The best way that the science is to read is to make the readers work. And now, what are you to do when you work? And you can see you, or the best way to get there? And, they also agree that you are in the world by doing this. But if you think you are learning how things come up with and thinking about. They are so helpful in our lives. They are in all aspects. They can help keep others and feel them safe because of other things they aren't

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
| 1 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 2 | cjtsolutions | gpu | 20 | 355,491,840 | 1.000 | 355,491,840 |
| 3 | ezshroom | gpu | 60 | 190,464,000 | 1.000 | 190,464,000 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
