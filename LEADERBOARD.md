# Leaderboard

Outer step **90** — updated 2026-08-08T16:56:25+00:00.

Val loss at step 90: **4.7483** — sample:

> The most important thing to understand about how to ensure people feel better about their behavior. Most people who suffer from these behaviors are some of the most important elements to help them understand their behavior without experiencing the most negative emotions being. This can also lead to some anxiety issues that lead to negative thoughts or a desire to understand their own behavior before and that they understand their emotions and beliefs so that they know how to express their emotions and beliefs about their emotions. This is what you need to consider when you think about something that exists to people when they think like they are feeling that they know about their emotions and beliefs. It’s almost

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
| 1 | cjtsolutions | nvidia-gpu | 54 | 1,124,913,152 | 0.997 | 1,121,050,556 |
| 2 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 3 | ezshroom | apple-gpu | 136 | 407,564,288 | 1.000 | 407,564,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
