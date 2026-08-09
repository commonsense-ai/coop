# Leaderboard

Outer step **126** — updated 2026-08-09T02:00:28+00:00.

Val loss at step 126: **5.1128** — RISING, +0.000927 per 1M tokens (±0.000137) — training is not converging · 2 of 6 steps improved it · best 4.4831 at step 121 (5 steps ago)

`▁▁▁▁▁▁▁▃▃▃▃▃▃▂▂▂▂▂▂▂▅▅▅▅▅▅▅▅▅▅▅█`  loss against tokens, 3.5B → 4.2B

Sample:

> The most important thing to understand about the topic! What’s your job? What’s the best way? What’s the best thing? What’s your job? What’s the best way to incorporate your knowledge? What’s the best way? Are we ready? Are you expecting? What’s your favorite? What’s your favorite? Show your mind is just waiting for you. Either one of you! Share your thoughts and feelings. Describe your thoughts around you! Share your thoughts and feelings. Identify your thoughts about yourself. Write a conversation, and inspire your thoughts to solve your thoughts. Identify your

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
| 1 | cjtsolutions | nvidia-gpu | 97 | 2,242,301,952 | 1.000 | 2,242,218,989 |
| 2 | naloxene | nvidia-gpu | 77 | 1,360,060,416 | 1.000 | 1,360,060,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
