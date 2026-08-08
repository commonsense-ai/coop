import json
from pathlib import Path

import pytest

import coop

tomllib = pytest.importorskip("tomllib")  # 3.11+; CI resolves a newer interpreter

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_every_version_is_released_in_lockstep():
    # the update check compares release.json against __version__, so a tag that
    # bumped only pyproject.toml would tell every volunteer to update forever —
    # and a stale lockfile once published a 0.2.0 sdist under a v0.2.2 tag
    version = PYPROJECT["project"]["version"]
    npm = json.loads((ROOT / "npm" / "package.json").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = next(p["version"] for p in lock["package"] if p["name"] == "coop-ai")
    assert version == coop.__version__ == npm["version"] == locked


def test_torch_cpu_index_stays_behind_the_cpu_extra():
    # regression: a bare `torch = [{index = "pytorch-cpu"}]` source follows the package
    # into uvx/pip installs from git, so every NVIDIA donor trained on their CPU
    for src in PYPROJECT["tool"]["uv"]["sources"]["torch"]:
        assert src.get("extra") == "cpu", src
    assert any(d.startswith("torch") for d in PYPROJECT["project"]["dependencies"])
    assert PYPROJECT["project"]["optional-dependencies"]["cpu"] == ["torch>=2.4"]
