# Leaderboard

Outer step **111** — updated 2026-08-08T22:37:12+00:00.

Val loss at step 111: **4.6594** — sample:

> The most important thing to understand about and explain why what happened to everybody? Why not or what happens to them? How do we relate to the problem? How do we think this? How do we relate to the problem? How do we relate to the problem? How should we relate to these questions? What do we relate to the problem? Why do we relate to all that? How do we relate to this problem? How do we relate to all these questions? How do we describe this problem? How are we relate to our problem? How do we relate to this problem? How do we relate to them? How often do

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
| 1 | cjtsolutions | nvidia-gpu | 70 | 1,411,633,152 | 0.999 | 1,410,734,974 |
| 2 | naloxene | nvidia-gpu | 61 | 1,027,452,928 | 1.000 | 1,027,452,928 |
| 3 | ezshroom | apple-gpu | 172 | 571,404,288 | 1.000 | 571,404,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
