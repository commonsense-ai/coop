"""coop: a community-pretrained small language model."""

import logging
import os
from pathlib import Path

import yaml

__version__ = "0.1.0"


def load_config(path: str = "config/run.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def setup_logging() -> None:
    """Volunteer-readable output: coop's own lines only, not the HTTP client chatter.
    COOP_LOG_TS adds timestamps for daemon logs (`coop start` sets it)."""
    fmt = "%(asctime)s %(message)s" if os.environ.get("COOP_LOG_TS") else "%(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%m-%d %H:%M:%S")
    for name in ("httpx", "huggingface_hub", "urllib3", "filelock", "datasets", "fsspec"):
        logging.getLogger(name).setLevel(logging.WARNING)
