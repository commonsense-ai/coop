"""Volunteer preferences: one small JSON file in the coop home.

The CLI owns writes; the worker only reads, so a `coop update --auto on` while a
round is running takes effect at the end of that round.
"""

import json
from pathlib import Path

FILENAME = "settings.json"


def path(home) -> Path:
    return Path(home) / FILENAME


def read(home) -> dict:
    try:
        return json.loads(path(home).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(home, **fields) -> dict:
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    merged = read(home) | fields
    path(home).write_text(json.dumps(merged, indent=2))
    return merged
