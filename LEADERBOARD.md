# Leaderboard

Outer step **106** — updated 2026-08-08T21:45:14+00:00.

Val loss at step 106: **4.6211** — sample:

> The most important thing to understand about it or that you should never remember it again, but they’re a good idea to spend time helping you learn. Well, to learn to stay in mind, you’re going to learn how to solve it, and you’re going to expect yourself to learn how it you need to learn something about it and how to solve it again? The answer goes beyond this point, because you probably don’t understand what it means. While it’s a simple way to teach it, there are ways you can accomplish it and that you will grow and become more comfortable if you feel it

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
| 3 | ezshroom | apple-gpu | 164 | 546,828,288 | 1.000 | 546,828,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
