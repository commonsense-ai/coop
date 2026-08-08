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


def test_nothing_still_advertises_a_license_the_repo_no_longer_grants():
    # the terms are undecided (LICENSE.md); a manifest left saying Apache-2.0 publishes
    # a grant nobody has agreed to — pypi and npm read these fields, not the notice
    npm = json.loads((ROOT / "npm" / "package.json").read_text())
    assert "All rights reserved" in (ROOT / "LICENSE.md").read_text()
    assert not (ROOT / "LICENSE").exists()  # two license files, one of them wrong
    assert PYPROJECT["project"]["license"] == "LicenseRef-Proprietary"
    assert npm["license"] == "SEE LICENSE IN LICENSE.md"
    for readme in (ROOT / "README.md", ROOT / "npm" / "README.md"):
        assert "Apache-2.0" not in readme.read_text()


def test_the_npm_copy_of_the_notice_matches_the_real_one():
    # npm publishes from npm/, so the notice has to live there too — and drift means
    # the package ships terms the repository never agreed to
    assert (ROOT / "npm" / "LICENSE.md").read_text() == (ROOT / "LICENSE.md").read_text()
