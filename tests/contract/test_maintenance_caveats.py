from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

REVIEWED_NODE24_ACTIONS = {
    "actions/checkout": ("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7.0.1"),
    "actions/upload-artifact": ("043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", "v7.0.1"),
    "actions/download-artifact": ("3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", "v8.0.1"),
    "astral-sh/setup-uv": ("20cfd1bf945f4377ade1205e4dbc17946fc9a30d", "v10.0.1"),
}


@pytest.mark.contract
def test_ci_uses_only_reviewed_sha_pinned_node24_actions() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    references = re.findall(r"uses:\s*([^@\s]+)@([0-9a-f]{40})(?:\s*#\s*([^\s]+))?", workflow)

    assert references
    assert {name for name, _, _ in references} == set(REVIEWED_NODE24_ACTIONS)
    for name, revision, comment in references:
        assert (revision, comment) == REVIEWED_NODE24_ACTIONS[name]


@pytest.mark.contract
def test_hosted_live_smoke_is_explicitly_double_opt_in() -> None:
    live_test = ROOT / "tests" / "live" / "test_openai_hosted_live.py"
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert live_test.is_file()
    source = live_test.read_text(encoding="utf-8")
    assert 'os.environ.get("RAGKIT_RUN_LIVE") != "1"' in source
    assert 'os.environ.get("OPENAI_API_KEY")' in source
    assert "@pytest.mark.live" in source
    assert "RAGKIT_RUN_LIVE=1" in contributing
    assert "pytest -m live" in contributing


@pytest.mark.contract
def test_supported_persistent_extra_contains_no_chromadb_surface() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bootstrap = (ROOT / "src" / "ragkit" / "infrastructure" / "bootstrap.py").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "chromadb" not in pyproject.casefold()
    assert "ChromaVectorStore" not in bootstrap
    assert '"sqlite"' in bootstrap
    assert "--extra persistent" not in dockerfile
