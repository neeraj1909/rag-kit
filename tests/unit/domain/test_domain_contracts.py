from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from math import inf, nan

import pytest

from ragkit.domain import (
    And,
    AssetRef,
    BoxLocator,
    CellLocator,
    Chunk,
    ChunkId,
    Comparison,
    ComparisonOperator,
    ComponentFingerprint,
    ComponentManifest,
    Document,
    DocumentId,
    Embedding,
    ExtractionNotice,
    ExtractionProvenance,
    ImageContent,
    IndexCompatibilityError,
    IndexManifest,
    InvalidDomainValueError,
    KeyframeLocator,
    LayoutContent,
    MediaContent,
    MetadataFilter,
    NormalizationMode,
    Not,
    OcrContent,
    Or,
    PageLocator,
    PartRelation,
    RelationKind,
    RetrievalScore,
    ScoreKind,
    ScoreProvenance,
    SourceId,
    SourceLocator,
    TextContent,
    TextSpanLocator,
    TimeSpanLocator,
    UnsupportedCapabilityError,
    canonical_json,
    compare_manifests,
    content_part_from_dict,
    content_part_to_dict,
    locator_from_dict,
    locator_to_dict,
    provenance_from_dict,
    provenance_to_dict,
    sort_scored_chunks,
)

pytestmark = pytest.mark.unit


def asset() -> AssetRef:
    return AssetRef(asset_id="asset-1", media_type="application/pdf", sha256="a" * 64)


def provenance(locator: object, *, confidence: float | None = None) -> ExtractionProvenance:
    return ExtractionProvenance(
        asset=asset(),
        locator=locator,  # type: ignore[arg-type]
        extractor=ComponentFingerprint.create(
            kind="extractor", implementation="fixture", configuration={"mode": "exact"}
        ),
        confidence=confidence,
    )


def test_locators_accept_exact_non_empty_ranges_and_reject_invalid_states() -> None:
    locators: list[SourceLocator] = [
        TextSpanLocator(start=0, end=4),
        PageLocator(page=0),
        BoxLocator(page=0, x0=0.1, y0=0.2, x1=0.3, y1=0.4),
        CellLocator(sheet="Summary", start_row=0, start_column=0, end_row=1, end_column=2),
        TimeSpanLocator(start_ms=0, end_ms=1_500),
        KeyframeLocator(timestamp_ms=700, frame_number=21),
    ]
    assert [item.kind for item in locators] == [
        "text_span",
        "page",
        "box",
        "cell",
        "time",
        "keyframe",
    ]

    invalid: list[Callable[[], object]] = [
        lambda: TextSpanLocator(start=2, end=2),
        lambda: PageLocator(page=-1),
        lambda: BoxLocator(page=1, x0=0.8, y0=0.1, x1=0.3, y1=0.2),
        lambda: CellLocator(sheet="", start_row=0, start_column=0),
        lambda: CellLocator(sheet="Summary", start_row=-1, start_column=0),
        lambda: TimeSpanLocator(start_ms=2, end_ms=1),
        lambda: KeyframeLocator(timestamp_ms=-1, frame_number=0),
    ]
    for construct in invalid:
        with pytest.raises(InvalidDomainValueError):
            construct()


def test_five_content_families_require_applicable_exact_provenance() -> None:
    parts: list[TextContent | OcrContent | LayoutContent | ImageContent | MediaContent] = [
        TextContent(part_id="p-text", text="alpha", provenance=provenance(TextSpanLocator(0, 5))),
        OcrContent(
            part_id="p-ocr",
            text="invoice",
            provenance=provenance(BoxLocator(1, 0.1, 0.1, 0.4, 0.2), confidence=0.8),
        ),
        LayoutContent(
            part_id="p-layout",
            text="total",
            provenance=provenance(CellLocator("Sheet 1", 2, 1)),
        ),
        ImageContent(
            part_id="p-image",
            description="damaged valve",
            provenance=provenance(BoxLocator(2, 0.0, 0.0, 1.0, 1.0), confidence=0.7),
        ),
        MediaContent(
            part_id="p-media",
            transcript="restart the pump",
            provenance=provenance(TimeSpanLocator(100, 900), confidence=0.9),
        ),
    ]
    assert [part.family for part in parts] == ["text", "ocr", "layout", "vision", "media"]

    with pytest.raises(InvalidDomainValueError, match="TextContent"):
        TextContent(part_id="bad", text="x", provenance=provenance(PageLocator(1)))
    with pytest.raises(InvalidDomainValueError, match="confidence"):
        OcrContent(part_id="bad", text="x", provenance=provenance(BoxLocator(1, 0, 0, 1, 1)))
    with pytest.raises(InvalidDomainValueError, match="MediaContent"):
        MediaContent(part_id="bad", transcript="x", provenance=provenance(TextSpanLocator(0, 1)))


def test_provenance_confidence_notices_and_relations_are_validated_and_immutable() -> None:
    notice = ExtractionNotice(code="low_contrast", message="Scan has low contrast")
    prov = replace(provenance(PageLocator(1), confidence=0.25), notices=(notice,))
    relation = PartRelation(
        source_part_id="caption", target_part_id="figure", kind=RelationKind.CAPTION_OF
    )
    part = LayoutContent(
        part_id="caption",
        text="Figure 1",
        provenance=prov,
        relations=(relation,),
    )
    assert part.relations[0].target_part_id == "figure"
    with pytest.raises(FrozenInstanceError):
        part.text = "changed"  # type: ignore[misc]
    with pytest.raises(InvalidDomainValueError):
        replace(prov, confidence=1.01)
    with pytest.raises(InvalidDomainValueError):
        PartRelation("same", "same", RelationKind.DERIVED_FROM)


def test_stable_ids_are_versioned_deterministic_and_sensitive_to_identity_inputs() -> None:
    source_a = SourceId.from_locator("file", {"uri": "file:///tmp/caf\u00e9.txt"})
    source_b = SourceId.from_locator("file", {"uri": "file:///tmp/cafe\u0301.txt"})
    assert source_a == source_b
    assert str(source_a).startswith("src_v1_") and len(str(source_a)) == 71

    doc_a = DocumentId.from_assets(source_a, ["a" * 64], boundary="document-1")
    doc_b = DocumentId.from_assets(source_a, ["b" * 64], boundary="document-1")
    assert doc_a != doc_b
    chunker = ComponentFingerprint.create("chunker", "fixed", {"size": 100})
    chunk_a = ChunkId.from_content(doc_a, chunker, [("part", TextSpanLocator(0, 4))], "same")
    chunk_b = ChunkId.from_content(doc_a, chunker, [("part", TextSpanLocator(4, 8))], "same")
    assert chunk_a != chunk_b
    assert str(chunk_a).startswith("chk_v1_")


def test_canonical_serialization_is_typed_stable_and_rejects_ambiguous_numbers() -> None:
    first = canonical_json({"b": [2, 1], "a": "x\r\ny"})
    second = canonical_json({"a": "x\ny", "b": [2, 1]})
    assert first == second == '{"a":"x\\ny","b":[2,1]}'
    assert canonical_json(TextSpanLocator(0, 2)) == '{"end":2,"kind":"text_span","start":0}'
    for value in (nan, inf, -inf):
        with pytest.raises(InvalidDomainValueError):
            canonical_json({"score": value})
    with pytest.raises(InvalidDomainValueError):
        canonical_json({1: "integer key"})


def test_documents_and_chunks_preserve_ordered_provenance_and_round_trip() -> None:
    source_id = SourceId.from_locator("memory", {"name": "fixture"})
    doc_id = DocumentId.from_assets(source_id, ["a" * 64])
    part = TextContent(part_id="p1", text="hello", provenance=provenance(TextSpanLocator(0, 5)))
    document = Document(document_id=doc_id, source_id=source_id, assets=(asset(),), parts=(part,))
    restored = Document.from_dict(document.to_dict())
    assert restored == document

    chunk_id = ChunkId.from_content(
        doc_id,
        ComponentFingerprint.create("chunker", "fixture", {"size": 5}),
        [(part.part_id, part.provenance.locator)],
        part.text,
    )
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        ordinal=0,
        text="hello",
        provenance=(part.provenance,),
        source_part_ids=(part.part_id,),
        metadata={"department": "support", "revision": 2},
    )
    assert Chunk.from_dict(chunk.to_dict()) == chunk
    assert chunk.to_dict()["ordinal"] == 0
    with pytest.raises(InvalidDomainValueError, match="align"):
        replace(chunk, source_part_ids=("p1", "p2"))
    with pytest.raises(InvalidDomainValueError, match="ordinal"):
        replace(chunk, ordinal=-1)


def test_document_rejects_a_relation_attached_to_the_wrong_source_part() -> None:
    source_id = SourceId.from_locator("memory", {"name": "relations"})
    document_id = DocumentId.from_assets(source_id, [asset().sha256])
    wrong_relation = PartRelation("not-caption", "figure", RelationKind.CAPTION_OF)
    caption = LayoutContent(
        "caption",
        "Figure 1",
        provenance(BoxLocator(1, 0.0, 0.0, 0.5, 0.1)),
        (wrong_relation,),
    )
    figure = ImageContent(
        "figure",
        "Valve assembly",
        provenance(BoxLocator(1, 0.0, 0.1, 0.5, 0.5)),
    )

    with pytest.raises(InvalidDomainValueError, match="source"):
        Document(document_id, source_id, (asset(),), (caption, figure))


def test_cross_layer_provenance_records_round_trip_without_type_loss() -> None:
    locator = CellLocator("Summary", 2, 3, 4, 5)
    restored_locator = locator_from_dict(locator_to_dict(locator))
    assert restored_locator == locator and type(restored_locator) is CellLocator
    prov = provenance(locator, confidence=0.6)
    assert provenance_from_dict(provenance_to_dict(prov)) == prov
    part = LayoutContent("cell", "42", prov)
    restored_part = content_part_from_dict(content_part_to_dict(part))
    assert restored_part == part and type(restored_part) is LayoutContent


@pytest.mark.parametrize(
    "part",
    [
        TextContent("text", "policy", provenance(TextSpanLocator(2, 8))),
        OcrContent(
            "ocr",
            "invoice",
            replace(
                provenance(BoxLocator(1, 0.1, 0.2, 0.3, 0.4), confidence=0.73),
                notices=(ExtractionNotice("rotation_corrected", "Rotated clockwise"),),
            ),
        ),
        LayoutContent(
            "caption",
            "Figure 1",
            provenance(CellLocator("Summary", 2, 1, 2, 3), confidence=0.95),
            (PartRelation("caption", "figure", RelationKind.CAPTION_OF),),
        ),
        ImageContent(
            "figure",
            "Damaged valve",
            provenance(PageLocator(3), confidence=0.81),
        ),
        MediaContent(
            "speech",
            "Restart the pump",
            provenance(TimeSpanLocator(120, 980), confidence=0.9),
        ),
        MediaContent(
            "keyframe",
            "Pressure gauge at warning level",
            provenance(KeyframeLocator(1_200, 36), confidence=0.88),
            (PartRelation("keyframe", "speech", RelationKind.KEYFRAME_OF),),
        ),
    ],
    ids=("text-span", "ocr-box", "layout-cell", "image-page", "media-time", "media-keyframe"),
)
def test_every_typed_content_variant_round_trips_losslessly(
    part: TextContent | OcrContent | LayoutContent | ImageContent | MediaContent,
) -> None:
    encoded = content_part_to_dict(part)
    restored = content_part_from_dict(encoded)

    assert restored == part
    assert type(restored) is type(part)
    assert type(restored.provenance.locator) is type(part.provenance.locator)
    assert content_part_to_dict(restored) == encoded


def test_embeddings_and_scores_enforce_dimensions_finiteness_and_ordering() -> None:
    embedding = Embedding(values=(1.0, 0.0), dimension=2, normalized=True)
    assert embedding.dimension == len(embedding.values)
    with pytest.raises(InvalidDomainValueError):
        Embedding(values=(1.0,), dimension=2)
    with pytest.raises(InvalidDomainValueError):
        Embedding(values=(nan,), dimension=1)

    score_source = ScoreProvenance(
        component=ComponentFingerprint.create("retriever", "dense", {"metric": "cosine"}),
        stage="retrieval",
        kind=ScoreKind.DISTANCE,
        metric="cosine_distance",
        conversion="negate:v1",
    )
    score = RetrievalScore(relevance=-0.2, raw_score=0.2, provenance=score_source)
    assert score.raw_score is not None
    assert score.relevance == -score.raw_score
    with pytest.raises(InvalidDomainValueError):
        replace(score, relevance=inf)
    converted = RetrievalScore.from_raw(0.25, score_source)
    assert converted.relevance == -0.25


def test_scored_chunk_order_is_higher_first_with_stable_id_ties_and_unique_ids() -> None:
    def scored(label: str, relevance: float) -> tuple[ChunkId, RetrievalScore]:
        identifier = ChunkId.from_payload({"label": label})
        source = ScoreProvenance(
            ComponentFingerprint.create("retriever", "fake", {}),
            "retrieval",
            ScoreKind.SIMILARITY,
            "dot",
            "identity:v1",
        )
        return identifier, RetrievalScore(relevance, relevance, source)

    low = scored("low", 0.1)
    tie_b = scored("tie-b", 0.9)
    tie_a = scored("tie-a", 0.9)
    ordered = sort_scored_chunks((low, tie_b, tie_a), top_k=2)
    assert [str(item[0]) for item in ordered] == sorted((str(tie_a[0]), str(tie_b[0])))
    with pytest.raises(InvalidDomainValueError, match="duplicate"):
        sort_scored_chunks((low, low), top_k=2)
    with pytest.raises(InvalidDomainValueError):
        sort_scored_chunks((low,), top_k=0)


def test_metadata_filters_are_typed_composable_and_losslessly_serializable() -> None:
    expression: MetadataFilter = And(
        (
            Comparison("department", ComparisonOperator.EQ, "support"),
            Or(
                (
                    Comparison("year", ComparisonOperator.GTE, 2024),
                    Not(Comparison("draft", ComparisonOperator.EQ, True)),
                )
            ),
        )
    )
    assert MetadataFilter.from_dict(expression.to_dict()) == expression
    with pytest.raises(InvalidDomainValueError):
        And(())
    with pytest.raises(InvalidDomainValueError):
        Comparison("", ComparisonOperator.EQ, "x")
    membership = Comparison("team", ComparisonOperator.IN, ("support", "field"))
    assert MetadataFilter.from_dict(membership.to_dict()) == membership
    with pytest.raises(InvalidDomainValueError):
        Comparison("score", ComparisonOperator.GT, nan)


def test_component_and_index_manifests_fingerprint_behavior_affecting_values() -> None:
    chunker = ComponentManifest("chunker", "fixed", "1", {"size": 100})
    embedder = ComponentManifest("embedder", "hashing", "1", {"dimension": 8, "normalize": True})
    manifest = IndexManifest(
        schema_version=1,
        corpus_fingerprint=ComponentFingerprint.create("corpus", "support", {"policy": 1}),
        chunker_fingerprint=chunker.fingerprint,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=8,
        normalization=NormalizationMode.L2,
        domain_schema_fingerprint=ComponentFingerprint.create("schema", "domain", {"version": 1}),
    )
    assert IndexManifest.from_dict(manifest.to_dict()) == manifest
    changed = replace(manifest, embedding_dimension=16)
    assert changed.fingerprint != manifest.fingerprint
    differences = compare_manifests(manifest, changed)
    assert differences == {"embedding_dimension": (8, 16)}
    with pytest.raises(IndexCompatibilityError) as error:
        manifest.require_compatible(changed)
    assert error.value.differences == differences


def test_index_manifest_serializes_indexing_identity_and_reads_legacy_values() -> None:
    legacy = IndexManifest(
        schema_version=1,
        corpus_fingerprint=ComponentFingerprint.create("corpus", "legacy", {"version": 1}),
        chunker_fingerprint=ComponentFingerprint.create("chunker", "legacy", {"version": 1}),
        embedder_fingerprint=ComponentFingerprint.create("embedder", "legacy", {"version": 1}),
        embedding_dimension=8,
        normalization=NormalizationMode.L2,
        domain_schema_fingerprint=ComponentFingerprint.create("schema", "legacy", {"version": 1}),
    )
    legacy_payload = legacy.to_dict()
    legacy_payload.pop("indexing_fingerprint")
    restored = IndexManifest.from_dict(legacy_payload)
    assert restored == legacy
    assert restored.to_dict()["indexing_fingerprint"] == str(restored.indexing_fingerprint)

    changed = replace(
        restored,
        indexing_fingerprint=ComponentFingerprint.create(
            "indexing_policy", "dense", {"vector_database": "qdrant"}
        ),
    )
    assert compare_manifests(restored, changed) == {
        "indexing_fingerprint": (
            restored.indexing_fingerprint,
            changed.indexing_fingerprint,
        )
    }
    assert restored.fingerprint != changed.fingerprint


def test_typed_errors_keep_safe_context_and_original_cause() -> None:
    cause = OSError("offline")
    error = UnsupportedCapabilityError(
        "handwriting is unavailable", capability="handwriting", cause=cause
    )
    assert error.capability == "handwriting"
    assert error.__cause__ is cause
