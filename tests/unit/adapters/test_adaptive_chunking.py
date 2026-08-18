from __future__ import annotations

import pytest

from ragkit.adapters import AdaptiveChunker
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    ComponentFingerprint,
    ContentPart,
    Document,
    DocumentId,
    ExtractionProvenance,
    ImageContent,
    KeyframeLocator,
    LayoutContent,
    MediaContent,
    OcrContent,
    PartRelation,
    RelationKind,
    SourceId,
    SourceLocator,
    TextContent,
    TextSpanLocator,
    TimeSpanLocator,
)
from ragkit.ports import (
    ChunkingPolicy,
    ChunkingRequest,
    ChunkingStrategy,
    DocumentFamily,
    supported_chunking_strategies,
)

pytestmark = pytest.mark.unit

_ASSET = AssetRef("asset", "application/octet-stream", "a" * 64)
_EXTRACTOR = ComponentFingerprint.create("extractor", "adaptive-test", {"version": 1})
_SOURCE = SourceId.from_locator("memory", {"name": "adaptive-test"})
_DOCUMENT = DocumentId.from_assets(_SOURCE, (_ASSET.sha256,))


def _provenance(locator: SourceLocator) -> ExtractionProvenance:
    return ExtractionProvenance(_ASSET, locator, _EXTRACTOR, confidence=0.9)


def _document(family: DocumentFamily) -> Document:
    parts: tuple[ContentPart, ...]
    if family is DocumentFamily.TEXT:
        text = "# Report\n\nPatient: stable.\n\nSpeaker A: follow up.\nvalue | result"
        parts = (TextContent("text", text, _provenance(TextSpanLocator(0, len(text)))),)
    elif family is DocumentFamily.OCR:
        parts = (
            OcrContent("ocr", "Patient stable. Follow up.", _provenance(BoxLocator(0, 0, 0, 1, 1))),
        )
    elif family is DocumentFamily.LAYOUT:
        parts = (LayoutContent("cell", "Total forty two.", _provenance(CellLocator("S", 0, 0))),)
    elif family is DocumentFamily.VISION:
        parts = (
            ImageContent(
                "image", "Gauge shows normal pressure.", _provenance(BoxLocator(0, 0, 0, 1, 1))
            ),
        )
    else:
        scene = MediaContent(
            "scene", "Technician closes valve.", _provenance(TimeSpanLocator(0, 1_000))
        )
        frame = MediaContent(
            "frame",
            "Valve close-up.",
            _provenance(KeyframeLocator(500, 12)),
            (PartRelation("frame", "scene", RelationKind.KEYFRAME_OF),),
        )
        parts = (scene, frame)
    return Document(_DOCUMENT, _SOURCE, (_ASSET,), parts)


@pytest.mark.parametrize("family", tuple(DocumentFamily))
def test_every_declared_family_strategy_executes_with_provenance(family: DocumentFamily) -> None:
    document = _document(family)
    for strategy in supported_chunking_strategies(family) - {ChunkingStrategy.AUTO}:
        policy = ChunkingPolicy(
            strategy=strategy,
            max_chars=20,
            overlap_chars=4,
            min_chunk_chars=1,
        )
        chunker = AdaptiveChunker(family, policy)

        chunks = chunker.chunk(ChunkingRequest((document,), 100, policy))

        assert chunks, (family, strategy)
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        assert all(chunk.provenance and chunk.source_part_ids for chunk in chunks)
        assert all(chunk.metadata["chunking_strategy"] == strategy.value for chunk in chunks)
        assert chunks == chunker.chunk(ChunkingRequest((document,), 100, policy))


def test_repeated_normalized_sentences_do_not_collide() -> None:
    document = Document(
        _DOCUMENT,
        _SOURCE,
        (_ASSET,),
        (OcrContent("ocr", "Repeat. Repeat.", _provenance(BoxLocator(0, 0, 0, 1, 1))),),
    )
    policy = ChunkingPolicy(
        strategy=ChunkingStrategy.SENTENCE,
        max_chars=8,
        overlap_chars=0,
        min_chunk_chars=1,
    )

    chunks = AdaptiveChunker(DocumentFamily.OCR, policy).chunk(
        ChunkingRequest((document,), 10, policy)
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Repeat. Repeat."

    evidence_policy = ChunkingPolicy(
        strategy=ChunkingStrategy.EVIDENCE,
        max_chars=6,
        overlap_chars=0,
        min_chunk_chars=1,
    )
    evidence = AdaptiveChunker(DocumentFamily.OCR, evidence_policy).chunk(
        ChunkingRequest((document,), 10, evidence_policy)
    )
    assert len(evidence) == 1
    assert evidence[0].text == "Repeat. Repeat."


def test_fixed_modality_chunks_assign_each_atomic_region_once() -> None:
    document = Document(
        _DOCUMENT,
        _SOURCE,
        (_ASSET,),
        (
            ImageContent("first", "first region", _provenance(BoxLocator(0, 0, 0, 0.4, 1))),
            ImageContent("second", "second area!", _provenance(BoxLocator(0, 0.6, 0, 1, 1))),
        ),
    )
    policy = ChunkingPolicy(
        strategy=ChunkingStrategy.FIXED,
        max_chars=10,
        overlap_chars=0,
        min_chunk_chars=1,
    )

    chunks = AdaptiveChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((document,), 10, policy)
    )

    assert [chunk.source_part_ids for chunk in chunks] == [("first",), ("second",)]
    assert [chunk.text for chunk in chunks] == ["first region", "second area!"]


def test_mixed_image_prose_chunks_report_actual_and_bound_families() -> None:
    document = Document(
        _DOCUMENT,
        _SOURCE,
        (_ASSET,),
        (
            OcrContent("ocr", "Printed label", _provenance(BoxLocator(0, 0, 0, 0.4, 1))),
            ImageContent("image", "Gauge face", _provenance(BoxLocator(0, 0.6, 0, 1, 1))),
        ),
        {"content_mode": "mixed_image_text"},
    )
    policy = ChunkingPolicy(
        strategy=ChunkingStrategy.FIXED,
        max_chars=12,
        overlap_chars=0,
        min_chunk_chars=1,
    )

    chunks = AdaptiveChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((document,), 10, policy)
    )

    assert [chunk.metadata["content_family"] for chunk in chunks] == ["ocr", "vision"]
    assert {chunk.metadata["document_family"] for chunk in chunks} == {"vision"}
