# Leaderboard

Outer step **122** — updated 2026-08-09T00:38:49+00:00.

Val loss at step 122: **4.69** — only 3 measurements so far · 1 of 2 steps improved it · best 4.4831 at step 121 (1 step ago)

Sample:

> The most important thing to understand about your life, and then seek assistance in dealing with an issue such as anxiety, anxiety, or depression. Conversely, knowing about an issue is critical. Especially especially stressing events like anxiety and depression can leave room for us to participate in a diverse life. Additionally, knowing when to discuss feelings and feelings helps us remember that the environment plays a big role in life. Humans can experience feelings of anxiety or depression by practicing important life events with specific tasks. Plus, knowing when to talk about someone else doesn't mean they can cope with or accept a situation at all and can negatively affect what happens,

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
| 1 | cjtsolutions | nvidia-gpu | 91 | 1,879,396,352 | 1.000 | 1,879,265,509 |
| 2 | naloxene | nvidia-gpu | 72 | 1,233,084,416 | 1.000 | 1,233,084,416 |
| 3 | ezshroom | apple-gpu | 177 | 585,740,288 | 1.000 | 585,740,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
