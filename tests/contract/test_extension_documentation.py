"""Executable checks for the adapter extension documentation."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_chunker_contract
from ragkit.domain import InvalidDomainValueError, LimitExceededError
from ragkit.ports import Chunker, ChunkingRequest
from ragkit.ports import models as port_models

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "extension-guide.md"
RECIPES = ROOT / "docs" / "recipes"

REQUIRED_RECIPE_SECTIONS = (
    "Business use case",
    "Contract",
    "Config schema",
    "Registry and bootstrap",
    "Tests",
    "Optional extra",
    "Limits",
    "Determinism",
    "Confidence and fallback",
    "Failure modes",
)

FAMILY_RECIPES = {
    "text": "knowledge-base-text.md",
    "ocr": "claims-ocr.md",
    "layout": "financial-layout.md",
    "vision": "equipment-vision.md",
    "media": "support-media.md",
}


def _markdown_headings(path: Path) -> set[str]:
    return {
        match.group(1).strip()
        for match in re.finditer(r"^## (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    }


def test_every_adapter_recipe_has_an_extension_checklist() -> None:
    recipes = sorted(RECIPES.glob("*.md"))
    assert recipes
    for recipe in recipes:
        missing = set(REQUIRED_RECIPE_SECTIONS) - _markdown_headings(recipe)
        assert not missing, f"{recipe.name} is missing recipe sections: {sorted(missing)}"


def test_one_business_recipe_exists_for_each_document_family() -> None:
    assert set(FAMILY_RECIPES) == {"text", "ocr", "layout", "vision", "media"}
    for family, name in FAMILY_RECIPES.items():
        recipe = RECIPES / name
        body = recipe.read_text(encoding="utf-8")
        assert f"Family: `{family}`" in body
        assert "## Business use case" in body


def test_guide_derived_adapter_passes_the_shared_chunker_contract(
    contract_corpus: ContractCorpus,
) -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    match = re.search(
        r"```python guide-derived-adapter\n(?P<code>.*?)\n```",
        guide,
        re.DOTALL,
    )
    assert match is not None, "extension guide must contain one executable adapter example"
    namespace: dict[str, object] = {}
    exec(compile(match.group("code"), str(GUIDE), "exec"), namespace)
    adapter_factory = cast(Callable[[str], Chunker], namespace["GuidePrefixChunker"])
    adapter = adapter_factory("indexed")

    chunks = assert_chunker_contract(
        adapter,
        ChunkingRequest((contract_corpus.document,), max_chunks=20),
    )

    assert chunks
    assert all(chunk.text.startswith("indexed: ") for chunk in chunks)

    with pytest.raises(InvalidDomainValueError, match="prefix"):
        adapter_factory(" ")
    with pytest.raises(LimitExceededError, match="truncate"):
        adapter.chunk(ChunkingRequest((contract_corpus.document,), max_chunks=1))


def test_guide_points_to_authoritative_boundaries_and_validation() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    required_references = (
        "src/ragkit/ports/interfaces.py",
        "src/ragkit/ports/models.py",
        "src/ragkit/infrastructure/config.py",
        "src/ragkit/infrastructure/bootstrap.py",
        "tests/contract_assertions.py",
        "uv run pytest -m contract --no-cov",
    )
    assert all(reference in guide for reference in required_references)


def test_public_boundary_values_have_direct_semantic_docstrings() -> None:
    public_values = (
        value
        for name, value in vars(port_models).items()
        if not name.startswith("_")
        and isinstance(value, type)
        and value.__module__ == port_models.__name__
    )
    for value in public_values:
        documentation = value.__doc__ or ""
        assert len(documentation.strip()) >= 20, f"{value.__name__} needs a semantic docstring"
        assert not documentation.startswith(f"{value.__name__}(")
