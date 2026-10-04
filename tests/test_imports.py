"""Smoke test: the package installed correctly and imports from src/.

Also keeps pytest from exiting 5 (no tests collected), which CI reads as failure.
"""

from pathlib import Path

import autods


def test_package_imports_from_src_layout():
    assert autods.__file__ is not None
    assert Path(autods.__file__).parent.name == "autods"


def test_submodules_import():
    from autods import config, state  # noqa: F401
