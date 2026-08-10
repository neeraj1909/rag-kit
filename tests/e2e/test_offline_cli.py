from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.e2e
def test_console_cli_returns_cited_fixture_answer() -> None:
    executable = Path(sys.executable).with_name("ragkit")
    assert executable.is_file()
    result = subprocess.run(
        [
            str(executable),
            "ask",
            "--config",
            "configs/offline.toml",
            "--source",
            "tests/fixtures/corpus",
            "What is the fixture answer?",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)

    assert "cobalt observatory" in payload["answer"].lower()
    evidence = payload["citations"][0]["evidence"][0]
    assert payload["citations"][0]["rank"] == 1
    assert evidence["source_uri"].endswith("answer.txt")
    assert evidence["locator"]["kind"] == "text_span"
