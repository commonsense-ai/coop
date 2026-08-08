# Leaderboard

Outer step **98** — updated 2026-08-08T18:15:24+00:00.

Val loss at step 98: **4.6526** — sample:

> The most important thing to understand about it — when you hear them? If they hear someone’s thoughts, you're at something like they think, you're what you think, but if an event isn't relevant, you're just going to understand what each day in the video doesn't seem like, but it doesn't seem to tell them what to do -- but what they think they mean about it? When we think it's going to tell us what we're talking about, we're trying to understand what they've done before." - No one should ask questions or questions or questions about the events they've learned -

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
| 1 | cjtsolutions | nvidia-gpu | 69 | 1,409,585,152 | 0.999 | 1,408,588,625 |
| 2 | naloxene | nvidia-gpu | 55 | 841,084,928 | 1.000 | 841,084,928 |
| 3 | ezshroom | apple-gpu | 148 | 444,428,288 | 1.000 | 444,428,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
