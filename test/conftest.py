#!/usr/bin/env python3

"""
Root conftest file for all pytest tests.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _installed_venv():
    """Return the virtual environment pytest is actually running in.

    Prefer the interpreter's own environment (``.venv`` on the NPU host,
    ``.venv-cuda`` on the documented CUDA path) and fall back to the project
    ``.venv``.  Returns ``None`` when neither is a real environment.

    ``sys.prefix`` is used rather than ``Path(sys.executable).resolve()``: the
    latter dereferences the ``<venv>/bin/python`` symlink to the base
    interpreter, which can escape the environment after a fresh install.
    """
    candidates = [
        Path(sys.prefix),               # the running interpreter's environment
        PROJECT_ROOT / ".venv",
    ]
    for venv in candidates:
        if (venv / "pyvenv.cfg").is_file() and (venv / "bin" / "python").is_file():
            return venv
    return None


def _bind_installed_environment():
    """Replicate ``source <venv>/bin/activate`` for this pytest process.

    The original test surface spawns ``uv run <console-script>`` from temporary
    working directories (``test/quiver``) and shell scripts that call a bare
    ``python`` (``test/util/scripts/*.sh``).  Running
    ``<venv>/bin/python -m pytest`` does not activate the environment, so those
    children could see neither the installed console scripts nor the interpreter
    holding the project dependencies.  Binding the real environment here keeps
    the original tests, their commands and their assertions unchanged; no
    wrapper executables are introduced.
    """
    venv = _installed_venv()
    if venv is None:
        # Nothing to bind: the original spawn/interpreter failures surface
        # loudly instead of being masked.
        return {"bound": False, "venv": None, "path_head": None}
    bin_dir = str(venv / "bin")
    os.environ["VIRTUAL_ENV"] = str(venv)
    os.environ["UV_PROJECT_ENVIRONMENT"] = str(venv)
    os.environ["UV_NO_SYNC"] = "1"
    path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if not path_parts or path_parts[0] != bin_dir:
        os.environ["PATH"] = os.pathsep.join(
            [bin_dir] + [p for p in path_parts if p != bin_dir]
        )
    return {"bound": True, "venv": str(venv), "path_head": bin_dir}


def pytest_configure(config):
    """Bind the installed project environment before collection."""
    _bind_installed_environment()


@pytest.fixture(scope="session", autouse=True)
def installed_project_environment():
    """Guarantee the binding even if hook ordering ever changes."""
    yield _bind_installed_environment()


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--keep-outputs",
        action="store_true",
        default=False,
        help="Save test outputs to a timestamped directory for later inspection"
    )