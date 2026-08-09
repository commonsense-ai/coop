# Leaderboard

Outer step **123** — updated 2026-08-09T00:52:38+00:00.

Val loss at step 123: **4.5838** — flat, +0.000502 per 1M tokens (±0.000462) — too noisy to call yet · 2 of 3 steps improved it · best 4.4831 at step 121 (2 steps ago)

Sample:

> The most important thing to understand about your thoughts, thoughts and thoughts. Always talk about your thoughts and feelings about your thoughts and emotions, and then recognize how your feelings contribute to our emotions. Aside from your thoughts, it might seem hard to ask your question about your feelings and feelings. Thanks to this list of the reasons why you hear your thoughts at the beginning of the year. Check out the main points of your talk about it during the season. Look at your thoughts and ideas, and they might come up to you soon. Look ahead and ask yourself to hear your feelings and feelings before you arrive at a meeting. If you prefer

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
| 1 | cjtsolutions | nvidia-gpu | 92 | 1,946,980,352 | 1.000 | 1,946,858,358 |
| 2 | naloxene | nvidia-gpu | 73 | 1,282,236,416 | 1.000 | 1,282,236,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
