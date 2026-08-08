# Leaderboard

Outer step **71** — updated 2026-08-08T13:10:03+00:00.

Val loss at step 71: **5.1013** — sample:

> The most important thing to understand about this. The research and investigation “The research has developed very relevant knowledge that we can use to teach a better understanding of the concepts we perceive in our research,” said Dr. David M. Roberts, an expert on the University of Maryland at the University of California at Oregon. “You can’t think about the science, but you can’t know what you mean.” “They can communicate with science that you can write and understand the facts of the research.” In an interview with researchers at the University of Washington, researchers from the University of Minnesota and other Wisconsin-wide data

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
| 3 | ezshroom | gpu | 103 | 307,609,600 | 1.000 | 307,609,600 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
