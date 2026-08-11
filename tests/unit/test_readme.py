from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.unit
def test_readme_is_an_executable_layered_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_readme.py", "--root", str(ROOT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
