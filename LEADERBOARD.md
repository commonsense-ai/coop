# Leaderboard

Outer step **112** — updated 2026-08-08T22:50:00+00:00.

Val loss at step 112: **4.6357** — sample:

> The most important thing to understand about. In fact, some parts of the story contain information about how you perceive it, such as the picture of someone else. Generally, it should be necessary to decide whether you were going to hear someone else. This will explain why we knew how they felt what we knew about it, whether or not they knew about something. If you should decide whether you wanted someone else to hear what we knew about this story, please contact us at firstname.lastname@example.org Take a note. If your child wants to hear their own stories or not understand what they knew about them, we can determine

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
| 1 | cjtsolutions | nvidia-gpu | 72 | 1,475,940,352 | 0.999 | 1,475,179,686 |
| 2 | naloxene | nvidia-gpu | 61 | 1,027,452,928 | 1.000 | 1,027,452,928 |
| 3 | ezshroom | apple-gpu | 174 | 575,500,288 | 1.000 | 575,500,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
