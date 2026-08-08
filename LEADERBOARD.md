# Leaderboard

Outer step **108** — updated 2026-08-08T22:07:17+00:00.

Val loss at step 108: **4.641** — sample:

> The most important thing to understand about it - or simply it should recognize how an individual sees an individual's thoughts through them?The importance of human behaviour and how we engage nature. From natural processes to natural environments and communities – the moral values of individuals, the moral values, concepts and models of society – are discussed below. It is an excellent tool we can evaluate for each individual in every context of life: to achieve social goals and interests in society and society while promoting social justice and equality. The concept of human behavior: the concept of nature, how we relate to another and the concepts we perceive as social behaviors. How

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
| 3 | ezshroom | apple-gpu | 167 | 557,068,288 | 1.000 | 557,068,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
