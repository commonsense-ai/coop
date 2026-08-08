# Leaderboard

Outer step **118** — updated 2026-08-08T23:49:56+00:00.

Val loss at step 118: **4.5137** — sample:

> The most important thing to understand about your health is to ask your doctor before they know what you're on the table. If you feel a lot of your favorite symptoms of cancer, call your doctor before taking any medication. The primary thing you should remember about is to speak when you are trying to speak. If you don't understand what causes our cancer and what causes them, make sure you're taking a deep breath during a walk or two. The main reason lies in the first step usually involves talking about your health and wellness. You're trying to teach your doctor about what causes you. Your doctor should ask you questions about what

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
| 1 | cjtsolutions | nvidia-gpu | 84 | 1,705,725,952 | 1.000 | 1,705,477,670 |
| 2 | naloxene | nvidia-gpu | 65 | 1,111,379,968 | 1.000 | 1,111,379,968 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
