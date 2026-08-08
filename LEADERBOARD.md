# Leaderboard

Outer step **113** — updated 2026-08-08T22:52:23+00:00.

Val loss at step 113: **4.6002** — sample:

> The most important thing to understand about women’s impact on society and how they are at a young age. If you haven’t understood how to engage a diverse range of girls in a diverse context, including women’s who are also in the same room, you should know why women don’t understand how far we want to survive in the future if they are in this room. According to a study published in the June 2009 issue of the International Society for the Assessment of Child Development (NAPAOM), the UN’s main objective was to promote their participation in school improvement projects with a variety of organizations including those that

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
| 1 | cjtsolutions | nvidia-gpu | 74 | 1,484,132,352 | 1.000 | 1,483,512,792 |
| 2 | naloxene | nvidia-gpu | 61 | 1,027,452,928 | 1.000 | 1,027,452,928 |
| 3 | ezshroom | apple-gpu | 174 | 575,500,288 | 1.000 | 575,500,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
