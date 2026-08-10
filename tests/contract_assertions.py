"""Reusable behavioral assertions for fake and concrete port implementations."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

from ragkit.domain import Chunk, Document, ScoredChunk, UnsupportedCapabilityError
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    Chunker,
    ChunkingRequest,
    DocumentExtractor,
    DocumentProjector,
    Embedder,
    EmbeddingRequest,
    EvaluationReport,
    EvaluationRequest,
    Evaluator,
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
    UpsertRequest,
    VectorSearchRequest,
    VectorStore,
)


def assert_source_contract(
    connector: SourceConnector, request: SourceRequest
) -> tuple[AcquiredAsset, ...]:
    """Acquisition is deterministic, bounded, ordered, and returns complete bytes."""

    first = connector.fetch(request)
    second = connector.fetch(request)
    assert first == second
    assert len(first) <= request.max_assets
    assert all(len(item.content) <= request.max_bytes_per_asset for item in first)
    assert all(item.reference.size_bytes == len(item.content) for item in first)
    return first


def assert_classifier_contract(
    classifier: FamilyClassifier, assets: tuple[AcquiredAsset, ...]
) -> tuple[AssetClassification, ...]:
    """Classification is deterministic and aligned one-for-one in input order."""

    first = classifier.classify(assets)
    assert first == classifier.classify(assets)
    assert tuple(item.asset_id for item in first) == tuple(
        item.reference.asset_id for item in assets
    )
    return first


def assert_extractor_contract(
    extractor: DocumentExtractor, request: ExtractionRequest
) -> tuple[Document, ...]:
    """Extraction is deterministic, bounded, and every part resolves to original evidence."""

    first = extractor.extract(request)
    assert first == extractor.extract(request)
    assert len(first) <= request.max_documents
    input_assets = {item.reference.asset_id for item in request.assets}
    for document in first:
        document_assets = {item.asset_id for item in document.assets}
        assert document_assets <= input_assets
        assert all(part.provenance.asset.asset_id in document_assets for part in document.parts)
    return first


def assert_exact_provenance_preserved(before: Document, after: Document) -> None:
    """Projection maps every retained part identity to identical evidence."""

    before_parts = {item.part_id: item.provenance for item in before.parts}
    after_parts = {item.part_id: item.provenance for item in after.parts}
    assert before_parts.items() <= after_parts.items()


def assert_projector_contract(
    projector: DocumentProjector, request: ProjectionRequest
) -> tuple[Document, ...]:
    """Projection is deterministic, aligned, bounded, and evidence preserving."""

    first = projector.project(request)
    assert first == projector.project(request)
    assert tuple(item.document_id for item in first) == tuple(
        item.document_id for item in request.documents
    )
    assert all(len(item.parts) <= request.max_parts_per_document for item in first)
    for before, after in zip(request.documents, first, strict=True):
        assert_exact_provenance_preserved(before, after)
    return first


def assert_chunker_contract(chunker: Chunker, request: ChunkingRequest) -> tuple[Chunk, ...]:
    """Chunking is deterministic, bounded, ordered, and provenance complete."""

    first = chunker.chunk(request)
    assert first == chunker.chunk(request)
    assert len(first) <= request.max_chunks
    document_order = {item.document_id: index for index, item in enumerate(request.documents)}
    assert list(first) == sorted(
        first, key=lambda item: (document_order[item.document_id], item.ordinal)
    )
    assert all(item.provenance and item.source_part_ids for item in first)
    return first


def assert_embedding_alignment(embedder: Embedder, texts: tuple[str, ...]) -> None:
    """Embedding count, order, width, document/query parity, and determinism align."""

    first = embedder.embed_documents(EmbeddingRequest(texts))
    second = embedder.embed_documents(EmbeddingRequest(texts))
    assert first == second
    assert len(first.embeddings) == len(texts)
    assert all(item.dimension == embedder.dimension for item in first.embeddings)
    assert first.embeddings == tuple(embedder.embed_query(text) for text in texts)


def assert_ranked_results(results: tuple[ScoredChunk, ...], *, top_k: int) -> None:
    """Retrieval output is finite, unique, bounded, provenance complete, and canonical."""

    assert len(results) <= top_k
    chunks = [item.chunk for item in results]
    assert len({item.chunk_id for item in chunks}) == len(chunks)
    expected = sorted(results, key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id)))
    assert list(results) == expected
    assert [item.rank for item in results] == list(range(1, len(results) + 1))
    assert all(isfinite(item.score.relevance) for item in results)
    assert all(item.chunk.provenance for item in results)


def assert_vector_store_contract(
    store: VectorStore, upsert: UpsertRequest, search: VectorSearchRequest
) -> tuple[ScoredChunk, ...]:
    """Upsert is idempotent and dense search is deterministic and canonical."""

    store.upsert(upsert)
    first = store.search(search)
    store.upsert(upsert)
    second = store.search(search)
    assert first == second
    assert_ranked_results(first, top_k=search.top_k)
    return first


def assert_reranker_contract(reranker: Reranker, request: RerankRequest) -> tuple[ScoredChunk, ...]:
    """Reranking is deterministic, bounded, canonical, and never invents chunks."""

    first = reranker.rerank(request)
    assert first == reranker.rerank(request)
    assert_ranked_results(first, top_k=request.top_k)
    candidate_ids = {item.chunk.chunk_id for item in request.candidates}
    assert {item.chunk.chunk_id for item in first} <= candidate_ids
    return first


def assert_prompt_contract(builder: PromptBuilder, request: PromptRequest) -> Prompt:
    """Prompt construction is deterministic and cites only supplied context."""

    first = builder.build(request)
    assert first == builder.build(request)
    context_ids = {item.chunk.chunk_id for item in request.context}
    assert len(set(first.cited_chunk_ids)) == len(first.cited_chunk_ids)
    assert set(first.cited_chunk_ids) <= context_ids
    return first


def assert_generator_contract(generator: Generator, request: GenerationRequest) -> GenerationResult:
    """Generation is deterministic and cites only prompt-authorized chunks."""

    first = generator.generate(request)
    assert first == generator.generate(request)
    assert set(first.cited_chunk_ids) <= set(request.prompt.cited_chunk_ids)
    return first


def assert_evaluator_contract(evaluator: Evaluator, request: EvaluationRequest) -> EvaluationReport:
    """Evaluation is deterministic, finite, uniquely named, and case aligned."""

    first = evaluator.evaluate(request)
    assert first == evaluator.evaluate(request)
    assert first.evaluated_case_ids == tuple(case.example.example_id for case in request.cases)
    assert len({metric.name for metric in first.metrics}) == len(first.metrics)
    assert all(isfinite(metric.value) for metric in first.metrics)
    return first


def assert_telemetry_contract(
    telemetry: Telemetry,
    event: TelemetryEvent,
    snapshot: Callable[[], tuple[TelemetryEvent, ...]],
) -> None:
    """Telemetry preserves one sanitized event value and call order."""

    before = snapshot()
    telemetry.record(event)
    assert snapshot() == (*before, event)


def assert_unsupported_is_explicit(classifier: FamilyClassifier, asset: AcquiredAsset) -> None:
    """Unsupported classification never becomes a silent omission."""

    try:
        classifier.classify((asset,))
    except UnsupportedCapabilityError:
        return
    raise AssertionError("unsupported input was silently accepted or omitted")
