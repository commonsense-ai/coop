# Leaderboard

Outer step **100** — updated 2026-08-08T18:54:33+00:00.

Val loss at step 100: **4.6453** — sample:

> The most important thing to understand about the benefits of various tools: - What is the difference between good and bad but good? Are everything that seems to fit? Are there any differences between good and bad and bad? Are there anything that seems to happen in the way that your mind does not feel like that? Can you agree on what do you want? Are you looking into the ways that I can understand them? Are there any differences between bad and bad? Are they the elements that seem to represent bad? Are there all these things? Are there any differences between bad and bad? The fact that they should be chosen is that

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
| 2 | naloxene | nvidia-gpu | 59 | 1,000,828,928 | 1.000 | 1,000,828,928 |
| 3 | ezshroom | apple-gpu | 152 | 460,812,288 | 1.000 | 460,812,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
