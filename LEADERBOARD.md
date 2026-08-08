# Leaderboard

Outer step **101** — updated 2026-08-08T19:28:40+00:00.

Val loss at step 101: **4.6385** — sample:

> The most important thing to understand about something that happens once they start taking an action with a friend to identify and compare and explain what it is to teach. The main idea behind how this transition becomes clear-looking is that it helps to understand the value of the problem in the context of that matter within the context of the problem itself. It helps to explain this, though in reality, we know the importance of being in the context of the problem itself. Whereas it helps us explain what the response moves into the problem as well as what we expect in the context of the problem when it becomes an immediate experience into the problem. Then

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
| 3 | ezshroom | apple-gpu | 154 | 477,196,288 | 1.000 | 477,196,288 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
