# Leaderboard

Outer step **110** — updated 2026-08-08T22:22:19+00:00.

Val loss at step 110: **4.6634** — sample:

> The most important thing to understand about when you're looking into the subject, but it looks like this one of those things that are happening in the world where that becomes the subject. Here are several reasons why you might ask yourself about how you'll say something you're looking at? The idea goes all together to understand what happens when you're looking into something that goes through something the way you're looking into your mind that you're putting together the subject of something that you're looking into. Imagine all that you're dealing with? It's actually how you're looking into something that you're talking about before you're looking into the subject

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
| 3 | ezshroom | apple-gpu | 170 | 565,260,288 | 1.000 | 565,260,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
