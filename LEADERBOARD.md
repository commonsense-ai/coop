# Leaderboard

Outer step **41** — updated 2026-08-08T06:58:25+00:00.

Val loss at step 41: **5.3429** — sample:

> The most important thing to understand about the importance of things and feelings. Here is a lot of ways. The reason for the future of reading comes with it, then it comes to be true, and so it comes to us to make sense of and explore the topic of the people and the people make us feel new. But what is the same thing into your own. If you mean to start your question, try them, but keep them feel confident and we do not know what I’m thinking! So, do you say, the difference between the whole language and the language, your answer is not,

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
| 1 | naloxene | nvidia-gpu | 44 | 594,505,728 | 1.000 | 594,505,728 |
| 2 | cjtsolutions | gpu | 20 | 355,491,840 | 1.000 | 355,491,840 |
| 3 | ezshroom | gpu | 48 | 156,876,800 | 1.000 | 156,876,800 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
