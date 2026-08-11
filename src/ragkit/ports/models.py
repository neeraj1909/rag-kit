"""Immutable provider-neutral values crossing port boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from ragkit.domain import (
    AssetRef,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    Document,
    Embedding,
    IndexManifest,
    InvalidDomainValueError,
    MetadataFilter,
    NormalizationMode,
    ScoredChunk,
)


def _positive(value: int, label: str) -> None:
    if value <= 0:
        raise InvalidDomainValueError(f"{label} must be positive")


def _non_empty(value: str, label: str) -> None:
    if not value.strip():
        raise InvalidDomainValueError(f"{label} must not be empty")


def _require_embedding_compatible(
    embedding: Embedding,
    embedder: ComponentFingerprint,
    manifest: IndexManifest,
) -> None:
    if embedder != manifest.embedder_fingerprint:
        raise InvalidDomainValueError("embedder fingerprint does not match the index manifest")
    if embedding.dimension != manifest.embedding_dimension:
        raise InvalidDomainValueError("embedding dimension does not match the index manifest")
    expected_normalized = manifest.normalization == NormalizationMode.L2
    if embedding.normalized != expected_normalized:
        raise InvalidDomainValueError("embedding normalization does not match the index manifest")


class DocumentFamily(StrEnum):
    """The five independently supported document implementation families."""

    TEXT = "text"
    OCR = "ocr"
    LAYOUT = "layout"
    VISION = "vision"
    MEDIA = "media"


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """Bound one acquisition address by asset count and bytes per complete asset."""

    source_uri: str
    max_assets: int
    max_bytes_per_asset: int

    def __post_init__(self) -> None:
        _non_empty(self.source_uri, "source_uri")
        _positive(self.max_assets, "max_assets")
        _positive(self.max_bytes_per_asset, "max_bytes_per_asset")


@dataclass(frozen=True, slots=True)
class AcquiredAsset:
    """Pair complete acquired bytes with the immutable reference to their source."""

    reference: AssetRef
    content: bytes

    def __post_init__(self) -> None:
        if len(self.content) == 0:
            raise InvalidDomainValueError("acquired asset content must not be empty")
        if self.reference.size_bytes is not None and len(self.content) != self.reference.size_bytes:
            raise InvalidDomainValueError("acquired bytes must match the declared asset size")


@dataclass(frozen=True, slots=True)
class AssetClassification:
    """Record one family decision, its optional confidence, and classifier identity."""

    asset_id: str
    family: DocumentFamily
    confidence: float | None
    classifier: ComponentFingerprint

    def __post_init__(self) -> None:
        _non_empty(self.asset_id, "asset_id")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise InvalidDomainValueError("classification confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """Align acquired assets and classifications under a document-count bound."""

    assets: tuple[AcquiredAsset, ...]
    classifications: tuple[AssetClassification, ...]
    max_documents: int

    def __post_init__(self) -> None:
        _positive(self.max_documents, "max_documents")
        asset_ids = tuple(item.reference.asset_id for item in self.assets)
        classified_ids = tuple(item.asset_id for item in self.classifications)
        if asset_ids != classified_ids:
            raise InvalidDomainValueError("classifications must align with assets in input order")


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    """Request evidence-preserving projections under a per-document part bound."""

    documents: tuple[Document, ...]
    max_parts_per_document: int

    def __post_init__(self) -> None:
        _positive(self.max_parts_per_document, "max_parts_per_document")


@dataclass(frozen=True, slots=True)
class ChunkingRequest:
    """Request ordered chunks from documents without allowing silent truncation."""

    documents: tuple[Document, ...]
    max_chunks: int

    def __post_init__(self) -> None:
        _positive(self.max_chunks, "max_chunks")


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """Carry an ordered, possibly empty batch of non-blank text representations."""

    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(not text.strip() for text in self.texts):
            raise InvalidDomainValueError("embedding texts must not contain blank values")


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Bind aligned, equal-width embeddings to the component that produced them."""

    embeddings: tuple[Embedding, ...]
    embedder: ComponentFingerprint

    def __post_init__(self) -> None:
        dimensions = {item.dimension for item in self.embeddings}
        if len(dimensions) > 1:
            raise InvalidDomainValueError("embedding batch dimensions must align")


@dataclass(frozen=True, slots=True)
class UpsertRequest:
    """Bind aligned chunks and embeddings to the manifest they must satisfy."""

    chunks: tuple[Chunk, ...]
    embeddings: EmbeddingBatch
    manifest: IndexManifest

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.embeddings.embeddings):
            raise InvalidDomainValueError("chunks and embeddings must align")
        if self.embeddings.embedder != self.manifest.embedder_fingerprint:
            raise InvalidDomainValueError("embedder fingerprint does not match the index manifest")
        for embedding in self.embeddings.embeddings:
            _require_embedding_compatible(embedding, self.embeddings.embedder, self.manifest)


@dataclass(frozen=True, slots=True)
class VectorSearchRequest:
    """Describe one bounded dense search under explicit index compatibility rules."""

    embedding: Embedding
    embedder: ComponentFingerprint
    top_k: int
    filters: MetadataFilter | None
    expected_manifest: IndexManifest

    def __post_init__(self) -> None:
        _positive(self.top_k, "top_k")
        _require_embedding_compatible(self.embedding, self.embedder, self.expected_manifest)


@dataclass(frozen=True, slots=True)
class DeleteRequest:
    """Name chunk identities to delete under an expected compatible manifest."""

    chunk_ids: tuple[ChunkId, ...]
    expected_manifest: IndexManifest


@dataclass(frozen=True, slots=True)
class SparseUpsertRequest:
    """Bind unique provenance-complete chunks to one sparse-index manifest."""

    chunks: tuple[Chunk, ...]
    manifest: IndexManifest

    def __post_init__(self) -> None:
        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise InvalidDomainValueError("sparse upsert contains duplicate chunk IDs")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Describe one non-blank bounded query with optional provider-neutral filters."""

    query: str
    top_k: int
    expected_manifest: IndexManifest
    filters: MetadataFilter | None = None

    def __post_init__(self) -> None:
        _non_empty(self.query, "query")
        _positive(self.top_k, "top_k")


@dataclass(frozen=True, slots=True)
class RerankRequest:
    """Request a bounded rescore of unique candidates without admitting new chunks."""

    query: str
    candidates: tuple[ScoredChunk, ...]
    top_k: int

    def __post_init__(self) -> None:
        _non_empty(self.query, "query")
        _positive(self.top_k, "top_k")
        chunk_ids = tuple(candidate.chunk.chunk_id for candidate in self.candidates)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise InvalidDomainValueError("rerank candidates contain duplicate chunk IDs")


@dataclass(frozen=True, slots=True)
class PromptRequest:
    """Bind a query and ranked evidence to a strict prompt-context character budget."""

    query: str
    context: tuple[ScoredChunk, ...]
    max_context_chars: int

    def __post_init__(self) -> None:
        _non_empty(self.query, "query")
        _positive(self.max_context_chars, "max_context_chars")


@dataclass(frozen=True, slots=True)
class Prompt:
    """Carry rendered prompt text and the exact chunk identities it authorizes."""

    text: str
    cited_chunk_ids: tuple[ChunkId, ...]

    def __post_init__(self) -> None:
        _non_empty(self.text, "prompt text")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Bind an authorized prompt and evidence to explicit generation controls."""

    query: str
    context: tuple[ScoredChunk, ...]
    prompt: Prompt
    temperature: float
    max_output_tokens: int

    def __post_init__(self) -> None:
        _non_empty(self.query, "query")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise InvalidDomainValueError("temperature must be finite and non-negative")
        _positive(self.max_output_tokens, "max_output_tokens")
        context_ids = {candidate.chunk.chunk_id for candidate in self.context}
        if not set(self.prompt.cited_chunk_ids) <= context_ids:
            raise InvalidDomainValueError("prompt citation is absent from generation context")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Report non-negative provider token counts without estimating missing usage."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise InvalidDomainValueError("token usage must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Return answer text, actual citations, model identity, and optional usage."""

    answer: str
    cited_chunk_ids: tuple[ChunkId, ...]
    model: ComponentFingerprint
    usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """Define one stable query and its relevant evidence labels for evaluation."""

    example_id: str
    query: str
    relevant_chunk_ids: tuple[ChunkId, ...]
    expected_answer: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.example_id, "example_id")
        _non_empty(self.query, "query")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Pair an evaluation example with already-observed retrieval and generation."""

    example: EvaluationExample
    retrieved: tuple[ScoredChunk, ...]
    generated: GenerationResult | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Bind non-empty observed cases to the evaluator fingerprint interpreting them."""

    cases: tuple[EvaluationCase, ...]
    evaluator: ComponentFingerprint

    def __post_init__(self) -> None:
        if not self.cases:
            raise InvalidDomainValueError("evaluation request requires at least one case")


@dataclass(frozen=True, slots=True)
class EvaluationMetric:
    """Store one finite value under a named, externally documented metric definition."""

    name: str
    value: float

    def __post_init__(self) -> None:
        _non_empty(self.name, "metric name")
        if not math.isfinite(self.value):
            raise InvalidDomainValueError("metric value must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Return named metrics and ordered case identities without provider payloads."""

    metrics: tuple[EvaluationMetric, ...]
    evaluated_case_ids: tuple[str, ...]


class TelemetryOutcome(StrEnum):
    """Classify a recorded operation as successful, failed, or explicitly partial."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class TelemetryAttribute:
    """Carry one bounded scalar operational attribute, never raw document content."""

    name: str
    value: str | int | float | bool | None

    def __post_init__(self) -> None:
        _non_empty(self.name, "telemetry attribute name")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise InvalidDomainValueError("telemetry values must be finite")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Describe one monotonic operation interval and its sanitized bounded metadata."""

    operation: str
    started_ns: int
    finished_ns: int
    outcome: TelemetryOutcome
    attributes: tuple[TelemetryAttribute, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.operation, "operation")
        if self.started_ns < 0 or self.finished_ns < self.started_ns:
            raise InvalidDomainValueError("telemetry timestamps must be monotonic")
