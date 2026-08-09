# Leaderboard

Outer step **127** — updated 2026-08-09T02:52:43+00:00.

Val loss at step 127: **5.0743** — RISING, +0.00087 per 1M tokens (±0.000105) — training is not converging · 3 of 7 steps improved it · best 4.4831 at step 121 (6 steps ago)

`▁▁▁▁▁▁▃▃▃▃▃▂▂▂▂▂▂▅▅▅▅▅▅▅▅▅▅█████`  loss against tokens, 3.5B → 4.3B

Sample:

> The most important thing to understand about the cause of eczema? What causes endometritis? Should endometritis be contagious or itchy when coughingy. Numerous studies suggest that endometrium is associated with endometrial ovarian endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometrial endometungal endometrial endometrial endometrial endomet� endometrial endomet

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
| 1 | cjtsolutions | nvidia-gpu | 99 | 2,346,749,952 | 1.000 | 2,346,679,622 |
| 2 | naloxene | nvidia-gpu | 77 | 1,360,060,416 | 1.000 | 1,360,060,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
