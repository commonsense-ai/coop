"""coop: a community-pretrained small language model."""

from pathlib import Path

import yaml

__version__ = "0.1.0"


def load_config(path: str = "config/run.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
