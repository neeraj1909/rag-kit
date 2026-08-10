from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from typing import cast

import pytest

from ragkit.application import (
    AnsweringRequest,
    AnsweringService,
    IndexingRequest,
    IndexingService,
    RagPipeline,
)
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ContentPart,
    Document,
    DocumentId,
    Embedding,
    ExtractionProvenance,
    ImageContent,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    LayoutContent,
    MediaContent,
    NormalizationMode,
    OcrContent,
    PageLocator,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    SourceId,
    TextContent,
    TextSpanLocator,
    TimeSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    Chunker,
    ChunkingRequest,
    DeleteRequest,
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    Embedder,
    EmbeddingBatch,
    EmbeddingRequest,
    ExtractionRequest,
    FamilyClassifier,
    GenerationRequest,
    GenerationResult,
    Generator,
    ProjectionRequest,
    Prompt,
    PromptBuilder,
    PromptRequest,
    Reranker,
    RerankRequest,
    SourceConnector,
    SourceRequest,
    Telemetry,
    TelemetryEvent,
    TelemetryOutcome,
    TokenUsage,
    UpsertRequest,
    VectorSearchRequest,
    VectorStore,
)

pytestmark = pytest.mark.unit


def fingerprint(kind: str) -> ComponentFingerprint:
    return ComponentFingerprint.create(kind, "application_test", {"version": 1})


def manifest() -> IndexManifest:
    return IndexManifest(
        schema_version=1,
        corpus_fingerprint=fingerprint("corpus"),
        chunker_fingerprint=fingerprint("chunker"),
        embedder_fingerprint=fingerprint("embedder"),
        embedding_dimension=2,
        normalization=NormalizationMode.NONE,
        domain_schema_fingerprint=fingerprint("domain_schema"),
    )


def family_document(family: DocumentFamily) -> Document:
    content = b"evidence"
    asset = AssetRef(
        "asset-1", "application/octet-stream", sha256(content).hexdigest(), size_bytes=len(content)
    )
    source_id = SourceId.from_locator("memory", {"name": family.value})
    document_id = DocumentId.from_assets(source_id, (asset.sha256,))
    provenance = ExtractionProvenance(asset, TextSpanLocator(0, 8), fingerprint("extractor"))
    if family is DocumentFamily.TEXT:
        part: ContentPart = TextContent("part-1", "evidence", provenance)
    elif family is DocumentFamily.OCR:
        part = OcrContent(
            "part-1",
            "evidence",
            replace(provenance, locator=BoxLocator(0, 0.0, 0.0, 1.0, 1.0), confidence=0.9),
        )
    elif family is DocumentFamily.LAYOUT:
        part = LayoutContent(
            "part-1", "evidence", replace(provenance, locator=CellLocator("Sheet", 0, 0))
        )
    elif family is DocumentFamily.VISION:
        part = ImageContent(
            "part-1", "evidence", replace(provenance, locator=PageLocator(0), confidence=0.8)
        )
    else:
        part = MediaContent(
            "part-1",
            "evidence",
            replace(provenance, locator=TimeSpanLocator(0, 1_000), confidence=0.85),
        )
    return Document(document_id, source_id, (asset,), (part,))


def mixed_document() -> Document:
    base = family_document(DocumentFamily.TEXT)
    asset = base.assets[0]
    extractor = fingerprint("extractor")
    parts: tuple[ContentPart, ...] = (
        base.parts[0],
        OcrContent(
            "ocr",
            "scan",
            ExtractionProvenance(
                asset, BoxLocator(0, 0.0, 0.0, 0.5, 0.5), extractor, confidence=0.8
            ),
        ),
        LayoutContent(
            "layout",
            "cell",
            ExtractionProvenance(asset, CellLocator("Sheet", 0, 0), extractor),
        ),
        ImageContent(
            "image",
            "diagram",
            ExtractionProvenance(asset, PageLocator(0), extractor, confidence=0.7),
        ),
        MediaContent(
            "media",
            "speech",
            ExtractionProvenance(asset, TimeSpanLocator(0, 500), extractor, confidence=0.9),
        ),
    )
    return replace(base, parts=parts)


class StepClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        current = self.value
        self.value += 10
        return current


class RecordingTelemetry(Telemetry):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.events: list[TelemetryEvent] = []

    def record(self, event: TelemetryEvent) -> None:
        self.calls.append(f"telemetry:{event.operation}:{event.outcome.value}")
        self.events.append(event)


class RecordingConnector(SourceConnector):
    def __init__(self, calls: list[str], *, empty: bool = False) -> None:
        self.calls = calls
        self.empty = empty

    def fetch(self, request: SourceRequest) -> tuple[AcquiredAsset, ...]:
        self.calls.append("fetch")
        if self.empty:
            return ()
        content = b"evidence"
        return (
            AcquiredAsset(
                AssetRef(
                    "asset-1",
                    "application/octet-stream",
                    sha256(content).hexdigest(),
                    request.source_uri,
                    len(content),
                ),
                content,
            ),
        )


class RecordingClassifier(FamilyClassifier):
    def __init__(self, calls: list[str], family: DocumentFamily, *, omit: bool = False) -> None:
        self.calls = calls
        self.family = family
        self.omit = omit

    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        self.calls.append("classify")
        if self.omit:
            return ()
        return tuple(
            AssetClassification(
                item.reference.asset_id, self.family, 1.0, fingerprint("classifier")
            )
            for item in assets
        )


class RecordingExtractor(DocumentExtractor):
    def __init__(
        self,
        calls: list[str],
        family: DocumentFamily,
        *,
        empty: bool = False,
        error: Exception | None = None,
        document: Document | None = None,
        preserve_acquired_assets: bool = True,
    ) -> None:
        self.calls = calls
        self.family = family
        self.empty = empty
        self.error = error
        self.document = document
        self.preserve_acquired_assets = preserve_acquired_assets

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        self.calls.append("extract")
        if self.error is not None:
            raise self.error
        if self.empty:
            return ()
        document = self.document or family_document(self.family)
        if not self.preserve_acquired_assets:
            return (document,)
        acquired = {item.reference.asset_id: item.reference for item in request.assets}
        assets = tuple(acquired[item.asset_id] for item in document.assets)
        parts = tuple(
            replace(
                part,
                provenance=replace(
                    part.provenance,
                    asset=acquired[part.provenance.asset.asset_id],
                ),
            )
            for part in document.parts
        )
        return (replace(document, assets=assets, parts=parts),)


class RecordingProjector(DocumentProjector):
    def __init__(self, calls: list[str], *, provenance_tamper: str | None = None) -> None:
        self.calls = calls
        self.request: ProjectionRequest | None = None
        self.provenance_tamper = provenance_tamper

    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        self.calls.append("project")
        self.request = request
        if self.provenance_tamper is not None:
            document = request.documents[0]
            if self.provenance_tamper == "drop":
                return (replace(document, parts=document.parts[1:]),)
            part = document.parts[0]
            altered = replace(
                part,
                provenance=replace(part.provenance, confidence=0.123),
            )
            return (replace(document, parts=(altered, *document.parts[1:])),)
        return request.documents


class RecordingChunker(Chunker):
    def __init__(
        self,
        calls: list[str],
        *,
        empty: bool = False,
        provenance_tamper: str | None = None,
    ) -> None:
        self.calls = calls
        self.empty = empty
        self.provenance_tamper = provenance_tamper

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return fingerprint("chunker")

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        self.calls.append("chunk")
        if self.empty:
            return ()
        document = request.documents[0]
        part = document.parts[0]
        chunk = Chunk(
            ChunkId.from_content(
                document.document_id,
                self.fingerprint,
                ((part.part_id, part.provenance.locator),),
                "evidence",
            ),
            document.document_id,
            0,
            "evidence",
            (part.provenance,),
            (part.part_id,),
        )
        if self.provenance_tamper == "document":
            return (replace(chunk, document_id=DocumentId.from_payload({"foreign": True})),)
        if self.provenance_tamper == "part":
            return (replace(chunk, source_part_ids=("unknown-part",)),)
        if self.provenance_tamper == "provenance":
            altered = replace(chunk.provenance[0], confidence=0.321)
            return (replace(chunk, provenance=(altered,)),)
        if self.provenance_tamper == "narrowed":
            narrowed = replace(chunk.provenance[0], locator=TextSpanLocator(1, 7))
            return (replace(chunk, text="videnc", provenance=(narrowed,)),)
        if self.provenance_tamper == "page_box":
            narrowed = replace(
                chunk.provenance[0],
                locator=BoxLocator(0, 0.1, 0.1, 0.9, 0.9),
            )
            return (replace(chunk, provenance=(narrowed,)),)
        return (chunk,)


class RecordingEmbedder(Embedder):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @property
    def dimension(self) -> int:
        return 2

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return fingerprint("embedder")

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        self.calls.append("embed_documents")
        return EmbeddingBatch(
            tuple(Embedding((1.0, 0.0), 2) for _ in request.texts), self.fingerprint
        )

    def embed_query(self, text: str) -> Embedding:
        self.calls.append("embed_query")
        return Embedding((1.0, 0.0), 2)


class RecordingStore(VectorStore):
    def __init__(self, calls: list[str], candidates: tuple[ScoredChunk, ...] = ()) -> None:
        self.calls = calls
        self.candidates = candidates
        self.upsert_request: UpsertRequest | None = None
        self.search_request: VectorSearchRequest | None = None

    def upsert(self, request: UpsertRequest) -> None:
        self.calls.append("upsert")
        self.upsert_request = request

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self.calls.append("search")
        self.search_request = request
        return self.candidates[: request.top_k]

    def delete(self, request: DeleteRequest) -> None:
        self.calls.append("delete")


class RecordingReranker(Reranker):
    def __init__(
        self,
        calls: list[str],
        *,
        empty: bool = False,
        substituted_chunk: Chunk | None = None,
    ) -> None:
        self.calls = calls
        self.empty = empty
        self.substituted_chunk = substituted_chunk

    def rerank(self, request: RerankRequest) -> tuple[ScoredChunk, ...]:
        self.calls.append("rerank")
        if self.empty:
            return ()
        selected = request.candidates[: request.top_k]
        if self.substituted_chunk is None:
            return selected
        return (replace(selected[0], chunk=self.substituted_chunk), *selected[1:])


class RecordingPromptBuilder(PromptBuilder):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def build(self, request: PromptRequest) -> Prompt:
        self.calls.append("prompt")
        return Prompt("bounded prompt", tuple(item.chunk.chunk_id for item in request.context))


class RecordingGenerator(Generator):
    def __init__(
        self,
        calls: list[str],
        *,
        foreign_citation: ChunkId | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls = calls
        self.foreign_citation = foreign_citation
        self.error = error

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append("generate")
        if self.error is not None:
            raise self.error
        citations = (
            (self.foreign_citation,)
            if self.foreign_citation is not None
            else request.prompt.cited_chunk_ids
        )
        return GenerationResult(
            "grounded answer", citations, fingerprint("generator"), TokenUsage(2, 2)
        )


def scored_chunk() -> ScoredChunk:
    document = family_document(DocumentFamily.TEXT)
    part = document.parts[0]
    chunk = Chunk(
        ChunkId.from_content(
            document.document_id,
            fingerprint("chunker"),
            ((part.part_id, part.provenance.locator),),
            "evidence",
        ),
        document.document_id,
        0,
        "evidence",
        (part.provenance,),
        (part.part_id,),
    )
    score = RetrievalScore(
        1.0,
        1.0,
        ScoreProvenance(
            fingerprint("vector_store"), "retrieval", ScoreKind.SIMILARITY, "dot", "identity:v1"
        ),
    )
    return ScoredChunk(chunk, score, 1)


def indexing_service(
    calls: list[str],
    family: DocumentFamily,
    *,
    connector_empty: bool = False,
    classifier_omit: bool = False,
    extractor_empty: bool = False,
    chunker_empty: bool = False,
    extraction_error: Exception | None = None,
) -> tuple[IndexingService, RecordingStore, RecordingTelemetry]:
    store = RecordingStore(calls)
    telemetry = RecordingTelemetry(calls)
    return (
        IndexingService(
            RecordingConnector(calls, empty=connector_empty),
            RecordingClassifier(calls, family, omit=classifier_omit),
            RecordingExtractor(calls, family, empty=extractor_empty, error=extraction_error),
            RecordingProjector(calls),
            RecordingChunker(calls, empty=chunker_empty),
            RecordingEmbedder(calls),
            store,
            telemetry,
            clock=StepClock(),
        ),
        store,
        telemetry,
    )


@pytest.mark.parametrize("family", tuple(DocumentFamily))
def test_indexing_sequences_every_family_through_manifest_checked_upsert(
    family: DocumentFamily,
) -> None:
    calls: list[str] = []
    service, store, telemetry = indexing_service(calls, family)

    result = service.run(IndexingRequest("memory://fixture", manifest()))

    assert [item for item in calls if not item.startswith("telemetry:")] == [
        "fetch",
        "classify",
        "extract",
        "project",
        "chunk",
        "embed_documents",
        "upsert",
    ]
    assert result.manifest == manifest()
    assert (result.asset_count, result.document_count, result.chunk_count) == (1, 1, 1)
    assert result.diagnostics == ()
    assert store.upsert_request is not None and store.upsert_request.manifest == manifest()
    assert [event.operation for event in telemetry.events] == [
        "index.fetch",
        "index.classify",
        "index.extract",
        "index.project",
        "index.chunk",
        "index.embed",
        "index.upsert",
    ]
    assert all(event.outcome is TelemetryOutcome.SUCCESS for event in telemetry.events)
    assert all(event.finished_ns > event.started_ns for event in telemetry.events)
    assert all(event.attributes == () for event in telemetry.events)


def test_indexing_passes_mixed_content_to_capabilities_without_family_branching() -> None:
    calls: list[str] = []
    projector = RecordingProjector(calls)
    telemetry = RecordingTelemetry(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.TEXT),
        RecordingExtractor(calls, DocumentFamily.TEXT, document=mixed_document()),
        projector,
        RecordingChunker(calls),
        RecordingEmbedder(calls),
        RecordingStore(calls),
        telemetry,
        clock=StepClock(),
    )

    result = service.run(IndexingRequest("memory://mixed", manifest()))

    assert result.chunk_count == 1
    assert projector.request is not None
    assert tuple(part.family for part in projector.request.documents[0].parts) == (
        "text",
        "ocr",
        "layout",
        "vision",
        "media",
    )


@pytest.mark.parametrize(
    ("empty_at", "code", "expected_calls"),
    [
        ("connector", "no_assets", ["fetch"]),
        ("extractor", "no_documents", ["fetch", "classify", "extract"]),
        (
            "chunker",
            "no_chunks",
            ["fetch", "classify", "extract", "project", "chunk"],
        ),
    ],
)
def test_indexing_reports_safe_omissions_without_calling_later_capabilities(
    empty_at: str, code: str, expected_calls: list[str]
) -> None:
    calls: list[str] = []
    service, store, _ = indexing_service(
        calls,
        DocumentFamily.TEXT,
        connector_empty=empty_at == "connector",
        extractor_empty=empty_at == "extractor",
        chunker_empty=empty_at == "chunker",
    )

    result = service.run(IndexingRequest("memory://secret-location", manifest()))

    assert [item for item in calls if not item.startswith("telemetry:")] == expected_calls
    assert result.diagnostics[0].code == code
    assert "secret-location" not in result.diagnostics[0].message
    assert store.upsert_request is None


def test_indexing_rejects_silently_omitted_classifications() -> None:
    calls: list[str] = []
    service, _, telemetry = indexing_service(calls, DocumentFamily.TEXT, classifier_omit=True)

    with pytest.raises(IntegrityError, match="classification"):
        service.run(IndexingRequest("memory://fixture", manifest()))

    assert telemetry.events[-1].operation == "index.classify"
    assert telemetry.events[-1].outcome is TelemetryOutcome.SUCCESS
    assert not any("fixture" in str(event) for event in telemetry.events)


def test_indexing_propagates_capability_errors_and_records_sanitized_error_timing() -> None:
    calls: list[str] = []
    error = UnsupportedCapabilityError("OCR unavailable", capability="ocr")
    service, _, telemetry = indexing_service(calls, DocumentFamily.OCR, extraction_error=error)

    with pytest.raises(UnsupportedCapabilityError) as caught:
        service.run(IndexingRequest("memory://customer-secret", manifest()))

    assert caught.value is error
    event = telemetry.events[-1]
    assert event.operation == "index.extract"
    assert event.outcome is TelemetryOutcome.ERROR
    assert event.attributes == ()
    assert "customer-secret" not in str(event)


def test_indexing_rejects_manifest_components_before_acquisition() -> None:
    calls: list[str] = []
    service, _, _ = indexing_service(calls, DocumentFamily.TEXT)
    incompatible = replace(manifest(), chunker_fingerprint=fingerprint("other_chunker"))

    with pytest.raises(IndexCompatibilityError) as caught:
        service.run(IndexingRequest("memory://fixture", incompatible))

    assert "chunker_fingerprint" in caught.value.differences
    assert calls == []


def test_indexing_rejects_extracted_assets_not_equal_to_acquired_references() -> None:
    calls: list[str] = []
    telemetry = RecordingTelemetry(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.TEXT),
        RecordingExtractor(
            calls,
            DocumentFamily.TEXT,
            preserve_acquired_assets=False,
        ),
        RecordingProjector(calls),
        RecordingChunker(calls),
        RecordingEmbedder(calls),
        RecordingStore(calls),
        telemetry,
        clock=StepClock(),
    )

    with pytest.raises(IntegrityError, match="acquired asset"):
        service.run(IndexingRequest("memory://fixture", manifest()))


@pytest.mark.parametrize("tamper", ["drop", "alter"])
def test_indexing_rejects_projectors_that_drop_or_alter_original_part_provenance(
    tamper: str,
) -> None:
    calls: list[str] = []
    telemetry = RecordingTelemetry(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.TEXT),
        RecordingExtractor(calls, DocumentFamily.TEXT),
        RecordingProjector(calls, provenance_tamper=tamper),
        RecordingChunker(calls),
        RecordingEmbedder(calls),
        RecordingStore(calls),
        telemetry,
        clock=StepClock(),
    )

    with pytest.raises(IntegrityError, match="original part provenance"):
        service.run(IndexingRequest("memory://fixture", manifest()))


@pytest.mark.parametrize("tamper", ["document", "part", "provenance"])
def test_indexing_rejects_chunks_that_do_not_resolve_to_projected_part_provenance(
    tamper: str,
) -> None:
    calls: list[str] = []
    telemetry = RecordingTelemetry(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.TEXT),
        RecordingExtractor(calls, DocumentFamily.TEXT),
        RecordingProjector(calls),
        RecordingChunker(calls, provenance_tamper=tamper),
        RecordingEmbedder(calls),
        RecordingStore(calls),
        telemetry,
        clock=StepClock(),
    )

    with pytest.raises(IntegrityError, match=r"projected document|source part provenance"):
        service.run(IndexingRequest("memory://fixture", manifest()))


def test_indexing_accepts_chunk_provenance_narrowed_within_the_source_part() -> None:
    calls: list[str] = []
    telemetry = RecordingTelemetry(calls)
    store = RecordingStore(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.TEXT),
        RecordingExtractor(calls, DocumentFamily.TEXT),
        RecordingProjector(calls),
        RecordingChunker(calls, provenance_tamper="narrowed"),
        RecordingEmbedder(calls),
        store,
        telemetry,
        clock=StepClock(),
    )

    result = service.run(IndexingRequest("memory://fixture", manifest()))

    assert result.chunk_count == 1
    assert store.upsert_request is not None


def test_indexing_accepts_box_chunk_provenance_within_a_source_page() -> None:
    calls: list[str] = []
    telemetry = RecordingTelemetry(calls)
    service = IndexingService(
        RecordingConnector(calls),
        RecordingClassifier(calls, DocumentFamily.VISION),
        RecordingExtractor(calls, DocumentFamily.VISION),
        RecordingProjector(calls),
        RecordingChunker(calls, provenance_tamper="page_box"),
        RecordingEmbedder(calls),
        RecordingStore(calls),
        telemetry,
        clock=StepClock(),
    )

    result = service.run(IndexingRequest("memory://fixture", manifest()))

    assert result.chunk_count == 1


def answering_service(
    calls: list[str],
    *,
    candidates: tuple[ScoredChunk, ...],
    rerank_empty: bool = False,
    foreign_citation: ChunkId | None = None,
    generation_error: Exception | None = None,
    substituted_chunk: Chunk | None = None,
) -> tuple[AnsweringService, RecordingStore, RecordingTelemetry]:
    store = RecordingStore(calls, candidates)
    telemetry = RecordingTelemetry(calls)
    return (
        AnsweringService(
            RecordingEmbedder(calls),
            store,
            RecordingReranker(calls, empty=rerank_empty, substituted_chunk=substituted_chunk),
            RecordingPromptBuilder(calls),
            RecordingGenerator(calls, foreign_citation=foreign_citation, error=generation_error),
            telemetry,
            clock=StepClock(),
        ),
        store,
        telemetry,
    )


def test_answering_sequences_search_to_generation_and_resolves_exact_citations() -> None:
    calls: list[str] = []
    candidate = scored_chunk()
    service, store, telemetry = answering_service(calls, candidates=(candidate,))

    result = service.run(AnsweringRequest("What happened?", manifest()))

    assert [item for item in calls if not item.startswith("telemetry:")] == [
        "embed_query",
        "search",
        "rerank",
        "prompt",
        "generate",
    ]
    assert result.generation is not None and result.generation.answer == "grounded answer"
    assert result.context == (candidate,)
    assert result.citations[0].chunk_id == candidate.chunk.chunk_id
    assert result.citations[0].document_id == candidate.chunk.document_id
    assert result.citations[0].rank == candidate.rank
    assert result.citations[0].provenance == candidate.chunk.provenance
    assert store.search_request is not None
    assert store.search_request.expected_manifest == manifest()
    assert [event.operation for event in telemetry.events] == [
        "ask.embed_query",
        "ask.search",
        "ask.rerank",
        "ask.prompt",
        "ask.generate",
    ]


@pytest.mark.parametrize(
    ("rerank_empty", "code", "expected_calls"),
    [
        (False, "no_search_results", ["embed_query", "search"]),
        (True, "no_rerank_results", ["embed_query", "search", "rerank"]),
    ],
)
def test_answering_stops_cleanly_on_empty_results(
    rerank_empty: bool, code: str, expected_calls: list[str]
) -> None:
    calls: list[str] = []
    candidates = (scored_chunk(),) if rerank_empty else ()
    service, _, _ = answering_service(calls, candidates=candidates, rerank_empty=rerank_empty)

    result = service.run(AnsweringRequest("What happened?", manifest()))

    assert result.generation is None
    assert result.context == ()
    assert result.citations == ()
    assert result.diagnostics[0].code == code
    assert [item for item in calls if not item.startswith("telemetry:")] == expected_calls


def test_answering_rejects_generator_citations_not_present_in_prompt_context() -> None:
    calls: list[str] = []
    foreign = ChunkId.from_payload({"foreign": True})
    service, _, _ = answering_service(calls, candidates=(scored_chunk(),), foreign_citation=foreign)

    with pytest.raises(IntegrityError, match="citation"):
        service.run(AnsweringRequest("What happened?", manifest()))


def test_answering_records_generation_exceptions_without_content_attributes() -> None:
    calls: list[str] = []
    error = UnsupportedCapabilityError("generation unavailable", capability="generation")
    service, _, telemetry = answering_service(
        calls, candidates=(scored_chunk(),), generation_error=error
    )

    with pytest.raises(UnsupportedCapabilityError) as caught:
        service.run(AnsweringRequest("private question", manifest()))

    assert caught.value is error
    assert telemetry.events[-1].operation == "ask.generate"
    assert telemetry.events[-1].outcome is TelemetryOutcome.ERROR
    assert telemetry.events[-1].attributes == ()
    assert "private question" not in str(telemetry.events[-1])


def test_answering_rejects_manifest_components_before_embedding() -> None:
    calls: list[str] = []
    service, _, _ = answering_service(calls, candidates=(scored_chunk(),))
    incompatible = replace(manifest(), embedder_fingerprint=fingerprint("other_embedder"))

    with pytest.raises(IndexCompatibilityError) as caught:
        service.run(AnsweringRequest("question", incompatible))

    assert "embedder_fingerprint" in caught.value.differences
    assert calls == []


def test_answering_rejects_out_of_order_search_results() -> None:
    calls: list[str] = []
    first = scored_chunk()
    second_chunk = replace(
        first.chunk,
        chunk_id=ChunkId.from_payload({"second": True}),
        ordinal=1,
    )
    second = ScoredChunk(
        second_chunk,
        replace(first.score, relevance=2.0, raw_score=2.0),
        2,
    )
    service, _, _ = answering_service(calls, candidates=(first, second))

    with pytest.raises(IntegrityError, match="ordering"):
        service.run(AnsweringRequest("question", manifest()))


def test_answering_rejects_reranker_chunk_substitution_under_the_same_id() -> None:
    calls: list[str] = []
    candidate = scored_chunk()
    substituted = replace(candidate.chunk, text="tampered evidence")
    assert substituted.chunk_id == candidate.chunk.chunk_id
    service, _, _ = answering_service(
        calls,
        candidates=(candidate,),
        substituted_chunk=substituted,
    )

    with pytest.raises(IntegrityError, match="substituted"):
        service.run(AnsweringRequest("question", manifest()))


def test_public_pipeline_facade_delegates_to_both_use_cases() -> None:
    index_calls: list[str] = []
    ask_calls: list[str] = []
    indexing, _, _ = indexing_service(index_calls, DocumentFamily.TEXT)
    answering, _, _ = answering_service(ask_calls, candidates=(scored_chunk(),))
    pipeline = RagPipeline(indexing, answering)

    indexed = pipeline.index(IndexingRequest("memory://fixture", manifest()))
    answered = pipeline.ask(AnsweringRequest("What happened?", manifest()))

    assert indexed.chunk_count == 1
    assert answered.generation is not None


def test_application_public_values_are_immutable_and_validate_limits() -> None:
    with pytest.raises(InvalidDomainValueError, match="max_assets"):
        IndexingRequest("memory://fixture", manifest(), max_assets=0)
    with pytest.raises(InvalidDomainValueError, match="query"):
        AnsweringRequest(" ", manifest())

    request = AnsweringRequest("question", manifest())
    with pytest.raises(FrozenInstanceError):
        request.query = "changed"  # type: ignore[misc]
    assert cast(object, request.filters) is None
