# Leaderboard

Outer step **125** — updated 2026-08-09T01:22:48+00:00.

Val loss at step 125: **4.8607** — RISING, +0.00079 per 1M tokens (±0.000178) — training is not converging · 2 of 5 steps improved it · best 4.4831 at step 121 (4 steps ago)

`▁▁▁▁▁▁▁▁▁▁▅▅▅▅▅▅▅▃▃▃▃▃▃▃▃▃██████`  loss against tokens, 3.5B → 4.0B

Sample:

> The most important thing to understand about them anyway isn’t always clear. Certainly your favorite car works great? Nothing else is affected except on our lives! Thousands of them are happy to win them anyway at school! Join Mr. Gregory Blake to submit a letter to your friend. Give us a call today. Click here.As we learn everything about food and nutrition, we want to inspire our children to participate in healthy lifestyle. Walking and eating habits are essential for our body and our bodies throughout life. Thousands of people suffer from insomnia, rheumatoid arthritis, autoimmune disease, obesity, autoimmune diseases, obesity and obesity. Eating healthy meals

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
| 1 | cjtsolutions | nvidia-gpu | 95 | 2,084,196,352 | 1.000 | 2,084,101,151 |
| 2 | naloxene | nvidia-gpu | 77 | 1,360,060,416 | 1.000 | 1,360,060,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
