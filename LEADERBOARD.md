# Leaderboard

Outer step **93** — updated 2026-08-08T17:30:00+00:00.

Val loss at step 93: **4.7082** — sample:

> The most important thing to understand about these questions, but how they relate to them? What about these issues? How can we ask you to ask them about the topic? How can we answer these questions? What will they relate to them? How will they relate to the concepts they tell us? What are their feelings about the topic? How can they explain what they understand what they want? How will they explain them? How should they understand their symptoms? How can they understand them? How should they solve their problems? How should they examine them? How will they listen to their symptoms? How will they respond to them when they ask

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
| 1 | cjtsolutions | nvidia-gpu | 60 | 1,247,793,152 | 0.998 | 1,245,516,179 |
| 2 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 3 | ezshroom | apple-gpu | 140 | 423,948,288 | 1.000 | 423,948,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
