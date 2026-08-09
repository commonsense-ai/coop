# Leaderboard

Outer step **124** — updated 2026-08-09T01:08:12+00:00.

Val loss at step 124: **4.8389** — RISING, +0.000787 per 1M tokens (±0.00027) — training is not converging · 2 of 4 steps improved it · best 4.4831 at step 121 (3 steps ago)

`▁▁▁▁▁▁▁▁▁▁▁▁▅▅▅▅▅▅▅▅▅▃▃▃▃▃▃▃▃▃▃█`  loss against tokens, 3.5B → 4.0B

Sample:

> The most important thing to understand about the various scenarios that may represent some of the primary causes of a chronic illness. Typically a chronic illness includes dehydration, scarring, numbness and vomiting, coughinginess, weakness and dizziness (usually accompanied by dizziness) and stiffness of the feet and knees and fingers on the floor. Likewise many serious conditions can trigger chronic illness. Certain illnesses can affect the onset and duration of symptoms due to marijuana or other chronic conditions. Often chronic illness affects many individuals, requiring treatment early in life. Lack of understanding of chronic illness allows individuals to diagnose conditions ranging from chronic illness to chronic illness. Lack of awareness extends beyond the ability

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
| 1 | cjtsolutions | nvidia-gpu | 94 | 2,030,948,352 | 1.000 | 2,030,845,276 |
| 2 | naloxene | nvidia-gpu | 75 | 1,335,484,416 | 1.000 | 1,335,484,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
