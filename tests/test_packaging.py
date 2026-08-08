from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # 3.11+; CI resolves a newer interpreter

PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_torch_cpu_index_stays_behind_the_cpu_extra():
    # regression: a bare `torch = [{index = "pytorch-cpu"}]` source follows the package
    # into uvx/pip installs from git, so every NVIDIA donor trained on their CPU
    for src in PYPROJECT["tool"]["uv"]["sources"]["torch"]:
        assert src.get("extra") == "cpu", src
    assert any(d.startswith("torch") for d in PYPROJECT["project"]["dependencies"])
    assert PYPROJECT["project"]["optional-dependencies"]["cpu"] == ["torch>=2.4"]
