# Leaderboard

Outer step **104** — updated 2026-08-08T20:40:01+00:00.

Val loss at step 104: **4.628** — sample:

> The most important thing to understand about the way we perceive their actions. It seems they might seem to have looked like they didn’t appear on the road. What they saw? That was true when they were they from a distance away—they didn’t exist and they weren’t interested in their actions as these couldn’t. When they knew what they saw, they talked about what they saw or didn’t, they knew—they meant they were afraid? That being true doesn’t exist on the road. Even if they didn’t exist without knowing what they saw or knew what they knew. Again they

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
| 3 | ezshroom | apple-gpu | 160 | 512,012,288 | 1.000 | 512,012,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
