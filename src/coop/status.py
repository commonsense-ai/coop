"""Worker -> CLI handoff: one small JSON file, replaced atomically on every update."""

import json
import time
from pathlib import Path

FILENAME = "status.json"


class StatusFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state: dict = {}

    def update(self, **fields) -> None:
        self.state |= fields
        self.state["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


def read_status(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
