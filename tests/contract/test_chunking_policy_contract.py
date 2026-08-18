"""Public, dependency-light chunking policy contract."""

import re
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from ragkit.domain import InvalidDomainValueError
from ragkit.ports import (
    ChunkingPolicy,
    ChunkingRequest,
    ChunkingStrategy,
    DocumentFamily,
    default_chunking_strategy,
    is_chunking_strategy_supported,
    resolve_chunking_policy,
    supported_chunking_strategies,
    validate_chunking_strategy,
)

pytestmark = pytest.mark.contract

EXPECTED_STRATEGIES = (
    "auto",
    "fixed",
    "sliding_window",
    "recursive",
    "sentence",
    "paragraph",
    "section",
    "hierarchical",
    "semantic",
    "proposition",
    "book",
    "legal",
    "medical",
    "code",
    "conversation",
    "table",
    "layout_region",
    "image_region",
    "transcript_segment",
    "scene",
    "evidence",
)


EXPECTED_COMPATIBILITY = {
    DocumentFamily.TEXT: frozenset(
        {
            "auto",
            "fixed",
            "sliding_window",
            "recursive",
            "sentence",
            "paragraph",
            "section",
            "hierarchical",
            "semantic",
            "proposition",
            "book",
            "legal",
            "medical",
            "code",
            "conversation",
            "table",
            "evidence",
        }
    ),
    DocumentFamily.OCR: frozenset(
        {
            "auto",
            "fixed",
            "sliding_window",
            "recursive",
            "sentence",
            "paragraph",
            "section",
            "hierarchical",
            "semantic",
            "proposition",
            "book",
            "legal",
            "medical",
            "conversation",
            "table",
            "layout_region",
            "evidence",
        }
    ),
    DocumentFamily.LAYOUT: frozenset(
        {
            "auto",
            "fixed",
            "sliding_window",
            "recursive",
            "sentence",
            "paragraph",
            "section",
            "hierarchical",
            "semantic",
            "proposition",
            "book",
            "legal",
            "medical",
            "code",
            "conversation",
            "table",
            "layout_region",
            "evidence",
        }
    ),
    DocumentFamily.VISION: frozenset(
        {
            "auto",
            "fixed",
            "sliding_window",
            "recursive",
            "sentence",
            "paragraph",
            "semantic",
            "proposition",
            "image_region",
            "evidence",
        }
    ),
    DocumentFamily.MEDIA: frozenset(
        {
            "auto",
            "fixed",
            "sliding_window",
            "recursive",
            "sentence",
            "paragraph",
            "semantic",
            "proposition",
            "conversation",
            "transcript_segment",
            "scene",
            "evidence",
        }
    ),
}


def test_strategy_catalog_is_exact_and_stable() -> None:
    assert tuple(strategy.value for strategy in ChunkingStrategy) == EXPECTED_STRATEGIES


def test_policy_is_immutable_normalized_and_fingerprint_complete() -> None:
    policy = ChunkingPolicy(semantic_threshold=1)

    assert policy.semantic_threshold == 1.0
    assert policy.fingerprint_inputs() == {
        "schema": "chunking-policy-v1",
        "strategy": "auto",
        "max_chars": 1200,
        "overlap_chars": 120,
        "min_chunk_chars": 80,
        "semantic_threshold": 1.0,
        "include_parent_context": True,
    }
    with pytest.raises(FrozenInstanceError):
        policy.max_chars = 42  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy": "fixed"}, "strategy must be a ChunkingStrategy"),
        ({"max_chars": True}, "max_chars must be an integer"),
        ({"max_chars": 0}, "max_chars must be positive"),
        ({"overlap_chars": -1}, "overlap_chars must be non-negative"),
        (
            {"max_chars": 100, "overlap_chars": 100},
            "overlap_chars must be smaller than max_chars",
        ),
        ({"min_chunk_chars": 0}, "min_chunk_chars must be positive"),
        (
            {"max_chars": 100, "overlap_chars": 10, "min_chunk_chars": 101},
            "min_chunk_chars must not exceed max_chars",
        ),
        ({"semantic_threshold": True}, "semantic_threshold must be numeric"),
        ({"semantic_threshold": float("nan")}, "semantic_threshold must be finite"),
        ({"semantic_threshold": 1.01}, "semantic_threshold must be in [0, 1]"),
        ({"include_parent_context": 1}, "include_parent_context must be a boolean"),
    ],
)
def test_policy_rejects_invalid_runtime_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidDomainValueError, match=re.escape(message)):
        ChunkingPolicy(**kwargs)


def test_chunking_request_preserves_legacy_positional_call_with_default_policy() -> None:
    request = ChunkingRequest((), 10)

    assert request.policy == ChunkingPolicy()


def test_compatibility_matrix_is_exact_and_returns_immutable_values() -> None:
    actual = {
        family: frozenset(strategy.value for strategy in supported_chunking_strategies(family))
        for family in DocumentFamily
    }

    assert actual == EXPECTED_COMPATIBILITY


def test_compatibility_checks_reject_invalid_types_and_unsupported_pairs() -> None:
    assert is_chunking_strategy_supported(DocumentFamily.LAYOUT, ChunkingStrategy.TABLE)
    assert not is_chunking_strategy_supported(DocumentFamily.VISION, ChunkingStrategy.TABLE)

    with pytest.raises(InvalidDomainValueError, match=r"table.*vision"):
        validate_chunking_strategy(DocumentFamily.VISION, ChunkingStrategy.TABLE)
    with pytest.raises(InvalidDomainValueError, match="family must be a DocumentFamily"):
        supported_chunking_strategies("text")  # type: ignore[arg-type]
    with pytest.raises(InvalidDomainValueError, match="strategy must be a ChunkingStrategy"):
        is_chunking_strategy_supported(DocumentFamily.TEXT, "fixed")  # type: ignore[arg-type]


def test_auto_resolution_is_explicit_deterministic_and_validated() -> None:
    assert default_chunking_strategy(DocumentFamily.TEXT) is ChunkingStrategy.RECURSIVE
    for family in (
        DocumentFamily.OCR,
        DocumentFamily.LAYOUT,
        DocumentFamily.VISION,
        DocumentFamily.MEDIA,
    ):
        assert default_chunking_strategy(family) is ChunkingStrategy.EVIDENCE

    original = ChunkingPolicy(strategy=ChunkingStrategy.AUTO)
    resolved = resolve_chunking_policy(DocumentFamily.TEXT, original)
    explicit = ChunkingPolicy(strategy=ChunkingStrategy.PARAGRAPH)

    assert original.strategy is ChunkingStrategy.AUTO
    assert resolved.strategy is ChunkingStrategy.RECURSIVE
    assert resolved.fingerprint_inputs()["strategy"] == "recursive"
    assert resolve_chunking_policy(DocumentFamily.TEXT, explicit) is explicit
    with pytest.raises(InvalidDomainValueError, match=r"scene.*text"):
        resolve_chunking_policy(
            DocumentFamily.TEXT,
            ChunkingPolicy(strategy=ChunkingStrategy.SCENE),
        )
