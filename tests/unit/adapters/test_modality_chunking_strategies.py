from __future__ import annotations

import json

import pytest

from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    ComponentFingerprint,
    ContentPart,
    Document,
    DocumentId,
    ExtractionNotice,
    ExtractionProvenance,
    ImageContent,
    InvalidDomainValueError,
    KeyframeLocator,
    LayoutContent,
    LimitExceededError,
    MediaContent,
    OcrContent,
    PageLocator,
    PartRelation,
    RelationKind,
    SourceId,
    SourceLocator,
    TimeSpanLocator,
)
from ragkit.ports import ChunkingPolicy, ChunkingRequest, ChunkingStrategy, DocumentFamily

pytestmark = pytest.mark.unit


def _provenance(locator: SourceLocator, *, confidence: float | None = None) -> ExtractionProvenance:
    return ExtractionProvenance(
        _ASSET,
        locator,
        _EXTRACTOR,
        confidence,
        (ExtractionNotice("fixture", "Evidence retained from fixture extraction."),),
    )


_ASSET = AssetRef("asset", "application/octet-stream", "a" * 64)
_EXTRACTOR = ComponentFingerprint.create("extractor", "fixture", {"version": 1})
_SOURCE_ID = SourceId.from_locator("memory", {"name": "modality-strategies"})
_DOCUMENT_ID = DocumentId.from_assets(_SOURCE_ID, (_ASSET.sha256,))


def _document(*parts: ContentPart) -> Document:
    return Document(
        _DOCUMENT_ID,
        _SOURCE_ID,
        (_ASSET,),
        parts,
        {"tenant": "test"},
    )


def _policy(strategy: ChunkingStrategy, *, max_chars: int = 120) -> ChunkingPolicy:
    return ChunkingPolicy(
        strategy=strategy,
        max_chars=max_chars,
        overlap_chars=0,
        min_chunk_chars=1,
    )


def test_table_strategy_groups_rows_with_headers_without_splitting_cells() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    name_header = LayoutContent("h-name", "Name", _provenance(CellLocator("People", 0, 0)))
    age_header = LayoutContent("h-age", "Age", _provenance(CellLocator("People", 0, 1)))
    alice = LayoutContent(
        "r1-name",
        "Alice",
        _provenance(CellLocator("People", 1, 0), confidence=0.98),
        (PartRelation("r1-name", "h-name", RelationKind.LABELED_BY),),
    )
    age = LayoutContent(
        "r1-age",
        "forty two years old",
        _provenance(CellLocator("People", 1, 1), confidence=0.97),
        (PartRelation("r1-age", "h-age", RelationKind.LABELED_BY),),
    )
    document = _document(name_header, age_header, alice, age)
    policy = _policy(ChunkingStrategy.TABLE, max_chars=8)
    chunker = ModalityChunker(DocumentFamily.LAYOUT, policy)

    chunks = chunker.chunk(ChunkingRequest((document,), max_chunks=10, policy=policy))

    assert len(chunks) == 1
    assert chunks[0].text == "Name: Alice | Age: forty two years old"
    assert chunks[0].source_part_ids == ("r1-name", "h-name", "r1-age", "h-age")
    assert chunks[0].provenance == tuple(
        part.provenance for part in (alice, name_header, age, age_header)
    )
    assert len(chunks[0].text) > policy.max_chars  # an atomic table row is never cut mid-cell
    relations = json.loads(str(chunks[0].metadata["source_relations_json"]))
    assert {item["target_part_id"] for item in relations} == {"h-name", "h-age"}
    assert chunks == chunker.chunk(ChunkingRequest((document,), max_chunks=10, policy=policy))


def test_layout_region_strategy_keeps_region_and_label_evidence_atomic() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    label = OcrContent(
        "label", "Total", _provenance(BoxLocator(2, 0.1, 0.1, 0.3, 0.2), confidence=0.8)
    )
    value = OcrContent(
        "value",
        "$42",
        _provenance(BoxLocator(2, 0.4, 0.1, 0.6, 0.2), confidence=0.76),
        (PartRelation("value", "label", RelationKind.LABELED_BY),),
    )
    policy = _policy(ChunkingStrategy.LAYOUT_REGION)
    chunks = ModalityChunker(DocumentFamily.OCR, policy).chunk(
        ChunkingRequest((_document(label, value),), 10, policy)
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Total $42"
    assert chunks[0].source_part_ids == ("value", "label")
    assert chunks[0].provenance == (value.provenance, label.provenance)
    assert all(item.notices == _provenance(PageLocator(0)).notices for item in chunks[0].provenance)


def test_image_region_strategy_emits_one_provenance_complete_chunk_per_region() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    first = ImageContent(
        "region-1",
        "fractured pipe",
        _provenance(BoxLocator(0, 0.0, 0.0, 0.4, 0.5), confidence=0.61),
    )
    second = ImageContent(
        "region-2",
        "pressure gauge",
        _provenance(BoxLocator(0, 0.5, 0.0, 1.0, 0.5), confidence=0.72),
    )
    policy = _policy(ChunkingStrategy.IMAGE_REGION)
    chunks = ModalityChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((_document(first, second),), 10, policy)
    )

    assert [chunk.text for chunk in chunks] == ["fractured pipe", "pressure gauge"]
    assert [chunk.source_part_ids for chunk in chunks] == [("region-1",), ("region-2",)]
    assert [chunk.provenance[0] for chunk in chunks] == [first.provenance, second.provenance]


@pytest.mark.parametrize("strategy", (ChunkingStrategy.EVIDENCE, ChunkingStrategy.IMAGE_REGION))
def test_mixed_image_vision_preserves_ocr_and_image_evidence(
    strategy: ChunkingStrategy,
) -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    ocr = OcrContent(
        "ocr-label",
        "pressure 42 psi",
        _provenance(BoxLocator(0, 0.1, 0.1, 0.4, 0.2), confidence=0.93),
    )
    image = ImageContent(
        "vision-region",
        "damaged pressure gauge",
        _provenance(BoxLocator(0, 0.5, 0.1, 0.9, 0.8), confidence=0.71),
    )
    document = Document(
        _DOCUMENT_ID,
        _SOURCE_ID,
        (_ASSET,),
        (ocr, image),
        {"content_mode": "mixed_image_text"},
    )
    policy = _policy(strategy)

    chunks = ModalityChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((document,), 10, policy)
    )

    assert [chunk.text for chunk in chunks] == ["pressure 42 psi", "damaged pressure gauge"]
    assert [chunk.source_part_ids for chunk in chunks] == [("ocr-label",), ("vision-region",)]
    assert [chunk.provenance for chunk in chunks] == [(ocr.provenance,), (image.provenance,)]
    assert [chunk.metadata["content_family"] for chunk in chunks] == ["ocr", "vision"]
    assert all(chunk.metadata["document_family"] == "vision" for chunk in chunks)


def test_unmarked_vision_document_still_rejects_foreign_ocr_parts() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    ocr = OcrContent(
        "ocr-label",
        "pressure 42 psi",
        _provenance(BoxLocator(0, 0.1, 0.1, 0.4, 0.2), confidence=0.93),
    )
    image = ImageContent(
        "vision-region",
        "damaged pressure gauge",
        _provenance(BoxLocator(0, 0.5, 0.1, 0.9, 0.8), confidence=0.71),
    )
    policy = _policy(ChunkingStrategy.EVIDENCE)

    with pytest.raises(InvalidDomainValueError, match="do not match"):
        ModalityChunker(DocumentFamily.VISION, policy).chunk(
            ChunkingRequest((_document(ocr, image),), 10, policy)
        )


def test_adaptive_dispatch_reaches_mixed_image_evidence_strategy() -> None:
    from ragkit.adapters.adaptive_chunking import AdaptiveChunker

    ocr = OcrContent(
        "ocr-label",
        "pressure 42 psi",
        _provenance(BoxLocator(0, 0.1, 0.1, 0.4, 0.2), confidence=0.93),
    )
    image = ImageContent(
        "vision-region",
        "damaged pressure gauge",
        _provenance(BoxLocator(0, 0.5, 0.1, 0.9, 0.8), confidence=0.71),
    )
    document = Document(
        _DOCUMENT_ID,
        _SOURCE_ID,
        (_ASSET,),
        (ocr, image),
        {"content_mode": "mixed_image_text"},
    )
    policy = _policy(ChunkingStrategy.EVIDENCE)

    chunks = AdaptiveChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((document,), 10, policy)
    )

    assert [chunk.source_part_ids for chunk in chunks] == [("ocr-label",), ("vision-region",)]


def test_transcript_segment_strategy_preserves_each_exact_time_interval() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    first = MediaContent("turn-1", "hello", _provenance(TimeSpanLocator(0, 500), confidence=0.9))
    second = MediaContent(
        "turn-2", "world", _provenance(TimeSpanLocator(500, 1_000), confidence=0.85)
    )
    policy = _policy(ChunkingStrategy.TRANSCRIPT_SEGMENT)
    chunks = ModalityChunker(DocumentFamily.MEDIA, policy).chunk(
        ChunkingRequest((_document(first, second),), 10, policy)
    )

    assert [chunk.text for chunk in chunks] == ["hello", "world"]
    assert [chunk.provenance[0].locator for chunk in chunks] == [
        TimeSpanLocator(0, 500),
        TimeSpanLocator(500, 1_000),
    ]


def test_scene_strategy_keeps_scene_and_keyframe_linked_in_one_chunk() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    scene = MediaContent("scene-0", "technician enters", _provenance(TimeSpanLocator(0, 2_000)))
    keyframe = MediaContent(
        "keyframe-0",
        "valve close-up",
        _provenance(KeyframeLocator(800, 24), confidence=0.7),
        (PartRelation("keyframe-0", "scene-0", RelationKind.KEYFRAME_OF),),
    )
    policy = _policy(ChunkingStrategy.SCENE)
    chunks = ModalityChunker(DocumentFamily.MEDIA, policy).chunk(
        ChunkingRequest((_document(scene, keyframe),), 10, policy)
    )

    assert len(chunks) == 1
    assert chunks[0].text == "technician enters valve close-up"
    assert chunks[0].source_part_ids == ("scene-0", "keyframe-0")
    assert chunks[0].provenance == (scene.provenance, keyframe.provenance)
    assert json.loads(str(chunks[0].metadata["source_relations_json"])) == [
        {
            "kind": "keyframe_of",
            "source_part_id": "keyframe-0",
            "target_part_id": "scene-0",
        }
    ]


def test_evidence_strategy_keeps_non_addressable_source_evidence_atomic() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    part = OcrContent(
        "ocr-1",
        "alpha beta gamma delta",
        _provenance(PageLocator(3), confidence=0.62),
    )
    policy = _policy(ChunkingStrategy.EVIDENCE, max_chars=11)
    chunker = ModalityChunker(DocumentFamily.OCR, policy)
    chunks = chunker.chunk(ChunkingRequest((_document(part),), 10, policy))

    assert [chunk.text for chunk in chunks] == ["alpha beta gamma delta"]
    assert all(chunk.provenance == (part.provenance,) for chunk in chunks)
    assert all(chunk.source_part_ids == (part.part_id,) for chunk in chunks)
    assert len(chunks[0].text) > policy.max_chars
    assert len({chunk.chunk_id for chunk in chunks}) == 1


def test_modality_fingerprint_covers_family_strategy_and_every_policy_input() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    baseline = _policy(ChunkingStrategy.EVIDENCE)
    changed_limit = ChunkingPolicy(
        strategy=ChunkingStrategy.EVIDENCE,
        max_chars=121,
        overlap_chars=0,
        min_chunk_chars=1,
    )

    assert (
        ModalityChunker(DocumentFamily.OCR, baseline).fingerprint
        == ModalityChunker(DocumentFamily.OCR, baseline).fingerprint
    )
    assert (
        ModalityChunker(DocumentFamily.OCR, baseline).fingerprint
        != ModalityChunker(DocumentFamily.LAYOUT, baseline).fingerprint
    )
    assert (
        ModalityChunker(DocumentFamily.OCR, baseline).fingerprint
        != ModalityChunker(DocumentFamily.OCR, changed_limit).fingerprint
    )


def test_modality_ordinals_restart_for_each_document() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    policy = _policy(ChunkingStrategy.IMAGE_REGION)
    first = _document(ImageContent("first", "one", _provenance(BoxLocator(0, 0.0, 0.0, 0.4, 1.0))))
    second = Document(
        DocumentId.from_assets(_SOURCE_ID, ("b" * 64,)),
        _SOURCE_ID,
        (AssetRef("asset-2", "image/png", "b" * 64),),
        (
            ImageContent(
                "second",
                "two",
                ExtractionProvenance(
                    AssetRef("asset-2", "image/png", "b" * 64),
                    BoxLocator(0, 0.6, 0.0, 1.0, 1.0),
                    _EXTRACTOR,
                ),
            ),
        ),
    )

    chunks = ModalityChunker(DocumentFamily.VISION, policy).chunk(
        ChunkingRequest((first, second), 10, policy)
    )

    assert [chunk.ordinal for chunk in chunks] == [0, 0]


def test_modality_chunker_rejects_auto_mismatched_policy_family_and_limits() -> None:
    from ragkit.adapters.modality_chunking import ModalityChunker

    with pytest.raises(InvalidDomainValueError, match="resolved"):
        ModalityChunker(DocumentFamily.OCR, ChunkingPolicy())
    with pytest.raises(InvalidDomainValueError, match="not supported"):
        ModalityChunker(DocumentFamily.VISION, _policy(ChunkingStrategy.TABLE))
    with pytest.raises(InvalidDomainValueError, match="modality family"):
        ModalityChunker(DocumentFamily.TEXT, _policy(ChunkingStrategy.EVIDENCE))

    policy = _policy(ChunkingStrategy.IMAGE_REGION)
    chunker = ModalityChunker(DocumentFamily.VISION, policy)
    first = ImageContent("region-1", "image one", _provenance(BoxLocator(0, 0.0, 0.0, 0.4, 1.0)))
    second = ImageContent("region-2", "image two", _provenance(BoxLocator(0, 0.6, 0.0, 1.0, 1.0)))
    document = _document(first, second)
    with pytest.raises(InvalidDomainValueError, match="bound chunking policy"):
        chunker.chunk(
            ChunkingRequest(
                (document,),
                10,
                _policy(ChunkingStrategy.IMAGE_REGION, max_chars=121),
            )
        )
    with pytest.raises(LimitExceededError, match="chunk limit"):
        chunker.chunk(ChunkingRequest((document,), 1, policy))
