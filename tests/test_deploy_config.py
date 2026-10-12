"""Vercel deploy configuration stays consistent with the app and the Docker build."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirements() -> list[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_pyproject_dependencies_match_requirements():
    # Vercel installs from pyproject.toml, Docker and CI from requirements.txt.
    assert _pyproject()["project"]["dependencies"] == _requirements()


def test_vercel_entrypoint_is_the_fastapi_app():
    module_name, _, attr = _pyproject()["tool"]["vercel"]["entrypoint"].partition(":")
    app = getattr(importlib.import_module(module_name), attr)

    assert type(app).__name__ == "FastAPI"


def test_vercel_function_config_targets_the_entrypoint():
    entry = _pyproject()["tool"]["vercel"]["entrypoint"].partition(":")[0].replace(".", "/") + ".py"
    config = json.loads((ROOT / "vercel.json").read_text())

    assert entry in config["functions"]
    assert config["functions"][entry]["maxDuration"] >= 180
    assert "web" in config["buildCommand"]
