"""Contributor ledger + leaderboard rendering. Plain JSON committed to the GitHub repo."""

import json
from datetime import datetime, timezone
from pathlib import Path

# One rejection drops a perfect reputation to 0.9; sustained acceptance pulls it back
# toward 1.0. Score scales with reputation so griefers cannot farm tokens.
REP_ALPHA = 0.1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_ledger() -> dict:
    return {"step": 0, "updated": None, "contributors": {}}


def load_ledger(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else empty_ledger()


def save_ledger(ledger: dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def _entry(ledger: dict, meta: dict, step: int) -> dict:
    return ledger["contributors"].setdefault(
        meta["username"],
        {
            "first_seen": step,
            "submissions": 0,
            "tokens": 0,
            "tier": meta.get("tier", "cpu"),
            "reputation": 1.0,
        },
    )


def update_ledger(ledger: dict, accepted: list[dict], step: int, rejected: list[dict] = ()) -> dict:
    for meta in accepted:
        e = _entry(ledger, meta, step)
        e["submissions"] += 1
        e["tokens"] += int(meta.get("tokens", 0))
        e["tier"] = meta.get("tier", e["tier"])
        e["reputation"] += REP_ALPHA * (1.0 - e["reputation"])
    for meta in rejected:
        e = _entry(ledger, meta, step)
        e["reputation"] += REP_ALPHA * (0.0 - e["reputation"])
    ledger["step"] = step
    ledger["updated"] = _now()
    return ledger


def score(entry: dict) -> float:
    return entry["tokens"] * entry["reputation"]


def render_leaderboard(ledger: dict) -> str:
    rows = sorted(ledger["contributors"].items(), key=lambda kv: (-score(kv[1]), kv[0]))
    lines = [
        "# Leaderboard",
        "",
        f"Outer step **{ledger['step']}** — updated {ledger['updated']}.",
        "",
        "Score = tokens contributed × reputation. Reputation is an EMA of acceptance",
        f"(alpha={REP_ALPHA}): rejected submissions lower it, accepted ones restore it.",
        "CPU-tier work (tokenize / dedup / filter / eval) earns tokens on this same board.",
        "",
        "| # | Contributor | Tier | Accepted | Tokens | Reputation | Score |",
        "|---|-------------|------|----------|--------|------------|-------|",
    ]
    for i, (user, e) in enumerate(rows, 1):
        lines.append(
            f"| {i} | {user} | {e['tier']} | {e['submissions']} | {e['tokens']:,} "
            f"| {e['reputation']:.3f} | {score(e):,.0f} |"
        )
    return "\n".join(lines) + "\n"
