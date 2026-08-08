# Leaderboard

Outer step **94** — updated 2026-08-08T17:33:53+00:00.

Val loss at step 94: **4.7006** — sample:

> The most important thing to understand about the history of a few. How much does it tell us about our history? How much does that mean? How many people say about themselves? What role does it mean for our lives? How many parents understand and ask us about our actions? What happens when we ask our children about our lives? How much do we tell us about our lives? What about our lives? How many of us consider the history of our lives? What ways do we observe our lives? How much does this make us move? How much does they affect our lives? What are we telling us about our lives? How

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
| 1 | cjtsolutions | nvidia-gpu | 62 | 1,258,033,152 | 0.999 | 1,256,173,668 |
| 2 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 3 | ezshroom | apple-gpu | 141 | 425,996,288 | 1.000 | 425,996,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
