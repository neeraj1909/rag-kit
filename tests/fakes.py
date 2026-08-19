"""Deterministic provider-free adapters used by reusable contract tests."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    Document,
    Embedding,
    IndexCompatibilityError,
    IndexManifest,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    Chunker,
    ChunkingRequest,
    DeleteRequest,
    DocumentExtractor,
    DocumentProjector,
    Embedder,
    EmbeddingBatch,
    EmbeddingRequest,
    EvaluationMetric,
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
    RetrievalRequest,
    Retriever,
    SourceConnector,
    SourceRequest,
    Telemetry,
    TelemetryEvent,
    TokenUsage,
    UpsertRequest,
    VectorSearchRequest,
    VectorStore,
)
from ragkit.ports.models import DocumentFamily


def _fingerprint(kind: str) -> ComponentFingerprint:
    return ComponentFingerprint.create(kind, "contract_fake", {"version": 1})


def _representation(document: Document, part_id: str) -> str:
    part = next(item for item in document.parts if item.part_id == part_id)
    if hasattr(part, "text"):
        return part.text
    if hasattr(part, "description"):
        return part.description
    return part.transcript


class FakeSourceConnector(SourceConnector):
    """Acquire one stable in-memory asset."""

    def fetch(self, request: SourceRequest) -> tuple[AcquiredAsset, ...]:
        if not request.source_uri.startswith("memory://"):
            raise UnsupportedCapabilityError(
                "only memory sources are supported", capability="source_scheme"
            )
        content = b"contract fixture"
        from ragkit.domain import AssetRef

        reference = AssetRef(
            asset_id="asset-contract",
            media_type="text/plain",
            sha256=sha256(content).hexdigest(),
            uri=request.source_uri,
            size_bytes=len(content),
        )
        return (AcquiredAsset(reference, content),)


class FakeFamilyClassifier(FamilyClassifier):
    """Classify the one supported fixture media type explicitly."""

    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        results: list[AssetClassification] = []
        for item in assets:
            if item.reference.media_type != "text/plain":
                raise UnsupportedCapabilityError(
                    "media type is unsupported", capability=item.reference.media_type
                )
            results.append(
                AssetClassification(
                    item.reference.asset_id, DocumentFamily.TEXT, 1.0, _fingerprint("classifier")
                )
            )
        return tuple(results)


class FakeDocumentExtractor(DocumentExtractor):
    def __init__(self, documents: tuple[Document, ...]) -> None:
        self._documents = documents

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        unsupported = next(
            (
                item.family
                for item in request.classifications
                if item.family is not DocumentFamily.TEXT
            ),
            None,
        )
        if unsupported is not None:
            raise UnsupportedCapabilityError("family is unsupported", capability=unsupported.value)
        return self._documents[: request.max_documents]


class FakeDocumentProjector(DocumentProjector):
    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        return tuple(
            Document(
                item.document_id,
                item.source_id,
                item.assets,
                item.parts[: request.max_parts_per_document],
                item.metadata,
            )
            for item in request.documents
        )


class FakeChunker(Chunker):
    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("chunker")

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for document in request.documents:
            for ordinal, part in enumerate(document.parts):
                text = _representation(document, part.part_id)
                chunks.append(
                    Chunk(
                        ChunkId.from_content(
                            document.document_id,
                            self.fingerprint,
                            [(part.part_id, part.provenance.locator)],
                            text,
                        ),
                        document.document_id,
                        ordinal,
                        text,
                        (part.provenance,),
                        (part.part_id,),
                        document.metadata,
                    )
                )
        return tuple(chunks[: request.max_chunks])


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("embedder")

    def _embed(self, text: str) -> Embedding:
        encoded = text.encode()
        values = (
            float(len(encoded)),
            float(sum(encoded) % 101),
            float(sum(index * value for index, value in enumerate(encoded, start=1)) % 103),
        )
        return Embedding(values, self.dimension)

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        return EmbeddingBatch(tuple(self._embed(text) for text in request.texts), self.fingerprint)

    def embed_query(self, text: str) -> Embedding:
        if not text.strip():
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("query must not be blank")
        return self._embed(text)


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self._manifest: IndexManifest | None = None
        self._entries: dict[ChunkId, tuple[Chunk, Embedding]] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("vector_store")

    def _require_manifest(self, expected: IndexManifest) -> None:
        if self._manifest is None:
            return
        if expected != self._manifest:
            expected.require_compatible(self._manifest)

    def require_compatible(self, manifest: IndexManifest) -> None:
        self._require_manifest(manifest)

    def upsert(self, request: UpsertRequest) -> None:
        self._require_manifest(request.manifest)
        if self._manifest is None:
            self._manifest = request.manifest
        pending = dict(self._entries)
        for chunk, embedding in zip(request.chunks, request.embeddings.embeddings, strict=True):
            if embedding.dimension != request.manifest.embedding_dimension:
                raise IndexCompatibilityError(
                    {
                        "embedding_dimension": (
                            request.manifest.embedding_dimension,
                            embedding.dimension,
                        )
                    }
                )
            pending[chunk.chunk_id] = (chunk, embedding)
        self._entries = pending

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self._require_manifest(request.expected_manifest)
        if request.filters is not None:
            raise UnsupportedCapabilityError(
                "metadata filters are not implemented by the fake", capability="metadata_filter"
            )
        scores: list[tuple[Chunk, RetrievalScore]] = []
        score_source = ScoreProvenance(
            _fingerprint("vector_store"), "retrieval", ScoreKind.SIMILARITY, "dot", "identity:v1"
        )
        for chunk, embedding in self._entries.values():
            raw = sum(
                left * right
                for left, right in zip(request.embedding.values, embedding.values, strict=True)
            )
            scores.append((chunk, RetrievalScore(raw, raw, score_source)))
        ordered = sorted(scores, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))
        return tuple(
            ScoredChunk(chunk, score, rank)
            for rank, (chunk, score) in enumerate(ordered[: request.top_k], start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        self._require_manifest(request.expected_manifest)
        for chunk_id in request.chunk_ids:
            self._entries.pop(chunk_id, None)


class FakeRetriever(Retriever):
    def __init__(self, candidates: tuple[ScoredChunk, ...]) -> None:
        self._candidates = candidates

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("retriever")

    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        if request.filters is not None:
            raise UnsupportedCapabilityError(
                "metadata filters are unsupported", capability="metadata_filter"
            )
        return self._candidates[: request.top_k]


class FakeReranker(Reranker):
    def rerank(self, request: RerankRequest) -> tuple[ScoredChunk, ...]:
        ordered = sorted(
            request.candidates,
            key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id)),
        )[: request.top_k]
        return tuple(
            ScoredChunk(item.chunk, item.score, rank) for rank, item in enumerate(ordered, 1)
        )


class FakePromptBuilder(PromptBuilder):
    def build(self, request: PromptRequest) -> Prompt:
        context = "\n".join(item.chunk.text for item in request.context)
        bounded = context[: request.max_context_chars]
        citations = tuple(
            item.chunk.chunk_id for item in request.context if item.chunk.text in bounded
        )
        return Prompt(f"Question: {request.query}\nContext:\n{bounded}", citations)


class FakeGenerator(Generator):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        context_ids = {item.chunk.chunk_id for item in request.context}
        citations = tuple(item for item in request.prompt.cited_chunk_ids if item in context_ids)
        return GenerationResult(
            "Answer grounded in supplied context.",
            citations,
            _fingerprint("generator"),
            TokenUsage(input_tokens=len(request.prompt.text.split()), output_tokens=5),
        )


class FakeEvaluator(Evaluator):
    def evaluate(self, request: EvaluationRequest) -> EvaluationReport:
        hits = sum(
            bool(
                set(case.example.relevant_chunk_ids)
                & {item.chunk.chunk_id for item in case.retrieved}
            )
            for case in request.cases
        )
        value = hits / len(request.cases) if request.cases else 0.0
        return EvaluationReport(
            (EvaluationMetric("hit_rate", value),),
            tuple(case.example.example_id for case in request.cases),
        )


class FakeTelemetry(Telemetry):
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def record(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class BrokenProjector(FakeDocumentProjector):
    """Return a valid but wrong locator to prove exact provenance checks matter."""

    def __init__(self, replacement: tuple[Document, ...]) -> None:
        self._replacement = replacement

    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        return self._replacement


class BrokenEmbedder(FakeEmbedder):
    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        aligned = super().embed_documents(request)
        return EmbeddingBatch(aligned.embeddings[:-1], aligned.embedder)


class BrokenRetriever(FakeRetriever):
    """Preserve invalid duplicate/order behavior for negative contract probes."""


class SilentUnsupportedClassifier(FakeFamilyClassifier):
    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        return ()


def chunk_ids(values: Iterable[ScoredChunk]) -> tuple[ChunkId, ...]:
    return tuple(item.chunk.chunk_id for item in values)
