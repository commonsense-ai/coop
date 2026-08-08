# Leaderboard

Outer step **75** — updated 2026-08-08T13:59:37+00:00.

Val loss at step 75: **4.978** — sample:

> The most important thing to understand about and try to read these ideas you want to answer. - Write a sentence carefully and listen you! - Read the sentence “I’ve been looking for some of the stories that tell you what you are thinking and what you are doing. Just make sure you are going to have a good story about yourself.” - Write your thoughts and ideas - Know what you like: - Get yourself involved - Try to teach your story in a sentence - Ask how you are speaking about. - Make a commentator - Make a note of what you are making -

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
| 1 | naloxene | nvidia-gpu | 54 | 825,929,728 | 1.000 | 825,929,728 |
| 2 | cjtsolutions | nvidia-gpu | 25 | 467,505,152 | 0.927 | 433,424,026 |
| 3 | ezshroom | gpu | 109 | 324,403,200 | 1.000 | 324,403,200 |
| 4 | miacx | cpu | 4 | 3,186,688 | 1.000 | 3,186,688 |
