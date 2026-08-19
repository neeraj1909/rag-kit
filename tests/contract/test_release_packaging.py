from __future__ import annotations

import re
import runpy
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
REQUIRED_EXTRAS = {
    "text",
    "ocr",
    "layout",
    "vision",
    "media",
    "persistent",
    "hosted",
    "http",
    "reranking",
    "pgvector",
    "qdrant",
    "pinecone",
    "opensearch",
}


@pytest.mark.contract
def test_release_metadata_names_every_supported_install_profile() -> None:
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    assert metadata["requires-python"] == ">=3.11,<3.13"
    assert set(metadata["optional-dependencies"]) == REQUIRED_EXTRAS
    assert metadata["readme"] == "README.md"
    assert metadata["license"] == "MIT"
    assert tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"] == {
        "ragkit": "ragkit.cli.main:main",
        "ragkit-http": "ragkit.delivery.server:main",
    }
    assert (PROJECT_ROOT / "README.md").is_file()
    assert (PROJECT_ROOT / "LICENSE").is_file()
    assert (PROJECT_ROOT / "src" / "ragkit" / "py.typed").is_file()


@pytest.mark.contract
def test_ci_builds_archives_and_covers_each_python_extra_pair() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    python_versions = set(
        re.findall(r"python-version:\s*\[([^]]+)]", workflow)[0].replace('"', "").split(", ")
    )
    matrix_match = re.search(r"extra:\s*\[([^]]+)]", workflow)

    assert python_versions == {"3.11", "3.12"}
    assert matrix_match is not None
    assert set(matrix_match.group(1).split(", ")) == REQUIRED_EXTRAS
    assert "uv build --no-sources" in workflow
    assert "scripts/check_package.py" in workflow
    assert "name: install-${{ matrix.extra }}-python-${{ matrix.python-version }}" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert "TRANSFORMERS_OFFLINE" in workflow
    assert ".venv-release/bin/ragkit-http --help" in workflow
    assert "scripts/check_core_install.py" in workflow
    assert "*.tar.gz" in workflow
    assert "timeout 180 sudo apt-get" in workflow


@pytest.mark.contract
def test_optional_import_probe_matches_published_extras() -> None:
    script = runpy.run_path(str(PROJECT_ROOT / "scripts" / "check_extra.py"))

    assert set(script["EXTRA_IMPORTS"]) == REQUIRED_EXTRAS
    assert set(script["EXTRA_DISTRIBUTIONS"]) == REQUIRED_EXTRAS
    assert "find_spec" not in Path(script["__file__"]).read_text(encoding="utf-8")
