from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_core_package_imports_without_optional_extras() -> None:
    assert importlib.import_module("ragkit").__name__ == "ragkit"

    completed = subprocess.run(
        [sys.executable, "-c", "import ragkit"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
