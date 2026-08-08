# Leaderboard

Outer step **109** — updated 2026-08-08T22:11:50+00:00.

Val loss at step 109: **4.6566** — sample:

> The most important thing to understand about these questions. Some questions are asked if there are any questions or questions. Or the question of whether your answer is correct? The answers to the questions are presented in the questions. They are answered every time. How do you determine if there are any questions? Are there questions? If you suspect? Are there anyone answers? If you believe the answers are correct? What questions are correct? Are there questions? Are there questions? Will there answers. If there are answers or questions regarding questions, you may ask for the questions. Should the question be answered? Are there questions you believe you would

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
| 3 | ezshroom | apple-gpu | 168 | 559,116,288 | 1.000 | 559,116,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
