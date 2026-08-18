from __future__ import annotations

from dataclasses import replace

import pytest

from ragkit.adapters.text_chunking import (
    TEXT_CHUNKING_STRATEGIES,
    TextStrategyChunker,
    chunk_text_documents,
    text_strategy_spans,
)
from ragkit.domain import (
    AssetRef,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionProvenance,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    SourceId,
    TextContent,
    TextSpanLocator,
    UnsupportedCapabilityError,
    derive_chunk_id,
)
from ragkit.ports import ChunkingPolicy, ChunkingRequest, ChunkingStrategy


def _document(text: str, **metadata: str) -> Document:
    asset = AssetRef("asset", "text/plain", "a" * 64, "memory://text")
    extractor = ComponentFingerprint.create("extractor", "test", {"version": 1})
    source_id = SourceId.from_locator("memory", {"uri": "memory://text"})
    document_id = DocumentId.from_assets(source_id, (asset.sha256,))
    return Document(
        document_id,
        source_id,
        (asset,),
        (
            TextContent(
                "body",
                text,
                ExtractionProvenance(asset, TextSpanLocator(100, 100 + len(text)), extractor, 1.0),
            ),
        ),
        metadata,
    )


def _policy(
    strategy: ChunkingStrategy,
    *,
    max_chars: int = 80,
    overlap_chars: int | None = None,
    min_chunk_chars: int = 1,
    semantic_threshold: float = 0.2,
    include_parent_context: bool = True,
) -> ChunkingPolicy:
    return ChunkingPolicy(
        strategy,
        max_chars,
        min(16, max_chars - 1) if overlap_chars is None else overlap_chars,
        min_chunk_chars,
        semantic_threshold,
        include_parent_context,
    )


@pytest.mark.parametrize("strategy", sorted(TEXT_CHUNKING_STRATEGIES, key=str))
def test_every_text_strategy_is_deterministic_bounded_and_exact(
    strategy: ChunkingStrategy,
) -> None:
    text = (
        "# Chapter 1\n\nHistory: Patient reports a persistent cough. "
        "The patient has no fever.\n\n"
        "Section 2.1 Treatment\nTake one tablet; review in seven days.\n\n"
        "Clinician: Do you feel better?\nPatient: Yes, breathing is easier."
    )
    policy = _policy(strategy)
    first = text_strategy_spans(text, policy, part_id="body")

    assert first == text_strategy_spans(text, policy, part_id="body")
    assert first
    assert all(0 <= item.start < item.end <= len(text) for item in first)
    assert all(len(text[item.start : item.end]) <= policy.max_chars for item in first)
    assert all(text[item.start : item.end].strip() for item in first)
    assert list(first) == sorted(first, key=lambda item: (item.start, item.end))
    assert all(
        character.isspace() or any(span.start <= offset < span.end for span in first)
        for offset, character in enumerate(text)
    )


def test_fixed_and_sliding_window_have_declared_boundary_behavior() -> None:
    text = "0123456789abcdefghijklmnopqrstuvwxyz"
    fixed = text_strategy_spans(text, _policy(ChunkingStrategy.FIXED, max_chars=10))
    sliding = text_strategy_spans(
        text,
        _policy(ChunkingStrategy.SLIDING_WINDOW, max_chars=10, overlap_chars=3),
    )

    assert [(item.start, item.end) for item in fixed] == [(0, 10), (10, 20), (20, 30), (30, 36)]
    assert [(item.start, item.end) for item in sliding] == [
        (0, 10),
        (7, 17),
        (14, 24),
        (21, 31),
        (28, 36),
    ]


def test_recursive_sentence_and_paragraph_strategies_prefer_natural_boundaries() -> None:
    text = "First sentence. Second sentence.\n\nA separate paragraph."

    recursive = text_strategy_spans(text, _policy(ChunkingStrategy.RECURSIVE, max_chars=34))
    sentence = text_strategy_spans(text, _policy(ChunkingStrategy.SENTENCE, max_chars=22))
    paragraph = text_strategy_spans(text, _policy(ChunkingStrategy.PARAGRAPH, max_chars=40))

    assert [text[item.start : item.end] for item in recursive] == [
        "First sentence. Second sentence.",
        "A separate paragraph.",
    ]
    assert [text[item.start : item.end] for item in sentence] == [
        "First sentence.",
        "Second sentence.",
        "A separate paragraph.",
    ]
    assert [text[item.start : item.end] for item in paragraph] == [
        "First sentence. Second sentence.",
        "A separate paragraph.",
    ]


def test_section_and_hierarchical_strategies_retain_heading_paths() -> None:
    text = "# Terms\n\nOpening.\n\n## Payment\n\nDue monthly.\n\nLate fees apply."

    section = text_strategy_spans(text, _policy(ChunkingStrategy.SECTION), part_id="body")
    hierarchical = text_strategy_spans(
        text, _policy(ChunkingStrategy.HIERARCHICAL, max_chars=28), part_id="body"
    )

    assert [item.structural_path for item in section] == ["Terms", "Terms / Payment"]
    assert [item.structural_path for item in hierarchical] == [
        "Terms",
        "Terms",
        "Terms / Payment",
        "Terms / Payment",
        "Terms / Payment",
    ]
    without_parent = text_strategy_spans(
        text,
        _policy(ChunkingStrategy.SECTION, include_parent_context=False),
        part_id="body",
    )
    assert {item.structural_path for item in without_parent} == {"body"}


def test_hierarchical_keeps_paragraph_children_that_section_can_pack() -> None:
    text = "# Findings\n\nFirst child paragraph.\n\nSecond child paragraph."
    section = text_strategy_spans(
        text,
        _policy(ChunkingStrategy.SECTION, max_chars=200),
        part_id="report",
    )
    hierarchical = text_strategy_spans(
        text,
        _policy(ChunkingStrategy.HIERARCHICAL, max_chars=200),
        part_id="report",
    )

    assert len(section) == 1
    assert [text[item.start : item.end] for item in hierarchical] == [
        "# Findings",
        "First child paragraph.",
        "Second child paragraph.",
    ]


def test_semantic_strategy_uses_deterministic_lexical_boundary_not_a_model() -> None:
    text = "Cats chase mice. Cats like mice. Rockets reach orbit."
    spans = text_strategy_spans(
        text,
        _policy(ChunkingStrategy.SEMANTIC, max_chars=80, semantic_threshold=0.2),
    )

    assert [text[item.start : item.end] for item in spans] == [
        "Cats chase mice. Cats like mice.",
        "Rockets reach orbit.",
    ]


def test_proposition_strategy_keeps_atomic_exact_clauses() -> None:
    text = "The service starts Monday; payment is due Friday, but cancellation remains allowed."
    spans = text_strategy_spans(text, _policy(ChunkingStrategy.PROPOSITION, max_chars=80))

    assert [text[item.start : item.end] for item in spans] == [
        "The service starts Monday;",
        "payment is due Friday,",
        "but cancellation remains allowed.",
    ]


@pytest.mark.parametrize(
    ("strategy", "text", "expected_paths", "domain"),
    [
        (
            ChunkingStrategy.BOOK,
            "Chapter 1: Arrival\n\nIt began.\n\nChapter 2: Return\n\nIt ended.",
            ["Chapter 1: Arrival", "Chapter 2: Return"],
            "book",
        ),
        (
            ChunkingStrategy.LEGAL,
            "Section 1 Scope\nThis applies.\nSection 2 Duties\nThe buyer pays.",
            ["Section 1 Scope", "Section 2 Duties"],
            "legal",
        ),
        (
            ChunkingStrategy.MEDICAL,
            "History: Cough for two days.\nDiagnosis: Viral infection.\nPlan: Rest and fluids.",
            ["History", "Diagnosis", "Plan"],
            "medical",
        ),
        (
            ChunkingStrategy.CODE,
            "import os\n\ndef read():\n    return os.getcwd()\n\nclass Writer:\n    pass\n",
            ["module", "def read", "class Writer"],
            "code",
        ),
        (
            ChunkingStrategy.CONVERSATION,
            "Context before call.\nAgent: Hello.\nCustomer: I need help.\nAgent: Certainly.",
            ["body", "Agent", "Customer", "Agent"],
            "conversation",
        ),
    ],
)
def test_domain_strategies_preserve_structural_units(
    strategy: ChunkingStrategy,
    text: str,
    expected_paths: list[str],
    domain: str,
) -> None:
    spans = text_strategy_spans(text, _policy(strategy), part_id="body")

    assert [item.structural_path for item in spans] == expected_paths
    assert {item.domain for item in spans} == {domain}


def test_adapter_chunks_rebase_exact_locators_and_identity_to_bound_fingerprint() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    document = _document(text, document_domain="book")
    policy = _policy(ChunkingStrategy.PARAGRAPH, max_chars=24)
    chunker = TextStrategyChunker(policy)

    chunks = chunker.chunk(ChunkingRequest((document,), 10, policy))

    assert chunks == chunker.chunk(ChunkingRequest((document,), 10, policy))
    assert [item.ordinal for item in chunks] == [0, 1]
    for chunk in chunks:
        locator = chunk.provenance[0].locator
        assert isinstance(locator, TextSpanLocator)
        assert chunk.text == text[locator.start - 100 : locator.end - 100]
        assert chunk.chunk_id == derive_chunk_id(chunk, chunker.fingerprint)
        assert chunk.metadata["chunking_strategy"] == "paragraph"
        assert chunk.metadata["document_domain"] == "book"

    assert (
        chunker.fingerprint
        != TextStrategyChunker(replace(policy, strategy=ChunkingStrategy.SENTENCE)).fingerprint
    )
    assert chunker.fingerprint != TextStrategyChunker(replace(policy, max_chars=23)).fingerprint


def test_recursive_strategy_handles_long_unbroken_text_without_recursion_or_loss() -> None:
    text = "x" * 25_000
    policy = _policy(ChunkingStrategy.RECURSIVE, max_chars=100)

    spans = text_strategy_spans(text, policy)

    assert len(spans) == 250
    assert "".join(text[item.start : item.end] for item in spans) == text


def test_dispatcher_hook_uses_caller_fingerprint_and_never_silently_truncates() -> None:
    document = _document("One. Two. Three.")
    policy = _policy(ChunkingStrategy.SENTENCE, max_chars=6)
    dispatcher_fingerprint = ComponentFingerprint.create(
        "chunker", "adaptive", policy.fingerprint_inputs()
    )

    chunks = chunk_text_documents(
        (document,), policy=policy, fingerprint=dispatcher_fingerprint, max_chunks=3
    )
    assert all(item.chunk_id == derive_chunk_id(item, dispatcher_fingerprint) for item in chunks)
    with pytest.raises(LimitExceededError, match="would truncate"):
        chunk_text_documents(
            (document,), policy=policy, fingerprint=dispatcher_fingerprint, max_chunks=2
        )


def test_adapter_rejects_auto_nontext_mismatch_and_non_text_parts() -> None:
    with pytest.raises(InvalidDomainValueError, match="resolved"):
        TextStrategyChunker(ChunkingPolicy())
    with pytest.raises(UnsupportedCapabilityError, match="not a textual"):
        TextStrategyChunker(_policy(ChunkingStrategy.TABLE))

    policy = _policy(ChunkingStrategy.FIXED)
    chunker = TextStrategyChunker(policy)
    with pytest.raises(InvalidDomainValueError, match="must equal"):
        chunker.chunk(ChunkingRequest((_document("text"),), 10, replace(policy, max_chars=79)))

    document = replace(_document("text"), parts=())
    assert chunker.chunk(ChunkingRequest((document,), 10, policy)) == ()

    exact = _document("text")
    malformed_part = replace(
        exact.parts[0],
        provenance=replace(exact.parts[0].provenance, locator=TextSpanLocator(100, 103)),
    )
    with pytest.raises(IntegrityError, match="exactly fill"):
        chunker.chunk(ChunkingRequest((replace(exact, parts=(malformed_part,)),), 10, policy))
