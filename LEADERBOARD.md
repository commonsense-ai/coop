# Leaderboard

Outer step **119** — updated 2026-08-08T23:52:33+00:00.

Val loss at step 119: **4.516** — sample:

> The most important thing to understand about the two subjects that you learn during the first two weeks of your life: (1) Have you ever heard of your life? You know you know you're studying your life? You know, you ask yourself how you're doing your life; and your family might know you just know you're listening to your friends? (2) Be sure you're getting the rest of your life! (3) Be aware of your life! (4) Make a list of your life before you've got the rest of your life. Don't buy those items with your favorite food and

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
| 1 | cjtsolutions | nvidia-gpu | 86 | 1,713,917,952 | 1.000 | 1,713,715,878 |
| 2 | naloxene | nvidia-gpu | 67 | 1,117,523,968 | 1.000 | 1,117,523,968 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
