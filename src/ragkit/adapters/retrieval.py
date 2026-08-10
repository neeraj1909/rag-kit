"""Deterministic offline embedding, dense search, and no-op reranking."""

from __future__ import annotations

import math
import re
from hashlib import sha256
from threading import RLock

from ragkit.domain import (
    And,
    Chunk,
    ChunkId,
    Comparison,
    ComparisonOperator,
    ComponentFingerprint,
    Embedding,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    MetadataFilter,
    NormalizationMode,
    Not,
    Or,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
)
from ragkit.ports import (
    DeleteRequest,
    Embedder,
    EmbeddingBatch,
    EmbeddingRequest,
    Reranker,
    RerankRequest,
    UpsertRequest,
    VectorSearchRequest,
    VectorStore,
)


class HashingEmbedder(Embedder):
    """Map normalized word tokens to a fixed-width L2-normalized feature vector."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("embedding dimension must be positive")
        self._dimension = dimension
        self._fingerprint = ComponentFingerprint.create(
            "embedder",
            "feature_hashing",
            {"version": 1, "dimension": dimension, "normalization": "l2"},
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalization(self) -> NormalizationMode:
        return NormalizationMode.L2

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        return EmbeddingBatch(tuple(self._embed(text) for text in request.texts), self.fingerprint)

    def embed_query(self, text: str) -> Embedding:
        if not text.strip():
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("query must not be blank")
        return self._embed(text)

    def _embed(self, text: str) -> Embedding:
        values = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            values[0] = 1.0
            norm = 1.0
        return Embedding(tuple(value / norm for value in values), self.dimension, True)


class InMemoryVectorStore(VectorStore):
    """Manifest-bound process-local dense cosine store with atomic mutation."""

    def __init__(self) -> None:
        self._manifest: IndexManifest | None = None
        self._entries: dict[ChunkId, tuple[Chunk, Embedding]] = {}
        self._lock = RLock()
        self._fingerprint = ComponentFingerprint.create(
            "vector_store", "memory_cosine", {"version": 1}
        )

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def upsert(self, request: UpsertRequest) -> None:
        with self._lock:
            self._require_compatible(request.manifest)
            self._require_cosine_manifest(request.manifest)
            for embedding in request.embeddings.embeddings:
                _require_unit_length(embedding)
            pending = dict(self._entries)
            for chunk, embedding in zip(request.chunks, request.embeddings.embeddings, strict=True):
                pending[chunk.chunk_id] = (chunk, embedding)
            self._entries = pending
            if self._manifest is None:
                self._manifest = request.manifest

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        with self._lock:
            self._require_initialized(request.expected_manifest)
            self._require_compatible(request.expected_manifest)
            self._require_cosine_manifest(request.expected_manifest)
            _require_unit_length(request.embedding)
            entries = tuple(self._entries.values())
        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.SIMILARITY,
            "cosine",
            "identity:v1",
        )
        scored: list[tuple[Chunk, RetrievalScore]] = []
        for chunk, embedding in entries:
            if request.filters is not None and not _matches(request.filters, chunk.metadata):
                continue
            raw = sum(
                left * right
                for left, right in zip(request.embedding.values, embedding.values, strict=True)
            )
            scored.append((chunk, RetrievalScore.from_raw(raw, provenance)))
        ordered = sorted(scored, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        with self._lock:
            self._require_initialized(request.expected_manifest)
            self._require_compatible(request.expected_manifest)
            self._require_cosine_manifest(request.expected_manifest)
            pending = dict(self._entries)
            for chunk_id in request.chunk_ids:
                pending.pop(chunk_id, None)
            self._entries = pending

    def _require_compatible(self, expected: IndexManifest) -> None:
        if self._manifest is not None:
            expected.require_compatible(self._manifest)

    def _require_initialized(self, expected: IndexManifest) -> None:
        if self._manifest is None:
            raise IndexCompatibilityError({"manifest": (expected.fingerprint, None)})

    @staticmethod
    def _require_cosine_manifest(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


class NoOpReranker(Reranker):
    """Canonicalize and bound existing candidates without changing their scores."""

    def rerank(self, request: RerankRequest) -> tuple[ScoredChunk, ...]:
        ordered = sorted(
            request.candidates,
            key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id)),
        )[: request.top_k]
        return tuple(
            ScoredChunk(item.chunk, item.score, rank) for rank, item in enumerate(ordered, start=1)
        )


def _require_unit_length(embedding: Embedding) -> None:
    norm = math.sqrt(sum(value * value for value in embedding.values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("L2-normalized embeddings must have unit length")


def _matches(expression: MetadataFilter, metadata: object) -> bool:
    from collections.abc import Mapping

    if not isinstance(metadata, Mapping):
        return False
    if isinstance(expression, Comparison):
        actual = metadata.get(expression.field)
        expected = expression.value
        if expression.operator is ComparisonOperator.EQ:
            return actual == expected
        if expression.operator is ComparisonOperator.NE:
            return actual != expected
        if expression.operator is ComparisonOperator.IN:
            return isinstance(expected, tuple) and any(actual == item for item in expected)
        return _ordered_comparison(actual, expected, expression.operator)
    if isinstance(expression, And):
        return all(_matches(child, metadata) for child in expression.children)
    if isinstance(expression, Or):
        return any(_matches(child, metadata) for child in expression.children)
    if isinstance(expression, Not):
        return not _matches(expression.child, metadata)
    return False


def _ordered_comparison(actual: object, expected: object, operator: ComparisonOperator) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return _apply_str_order(actual, expected, operator)
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return _apply_float_order(float(actual), float(expected), operator)
    return False


def _apply_str_order(left: str, right: str, operator: ComparisonOperator) -> bool:
    if operator is ComparisonOperator.GT:
        return left > right
    if operator is ComparisonOperator.GTE:
        return left >= right
    if operator is ComparisonOperator.LT:
        return left < right
    return left <= right


def _apply_float_order(left: float, right: float, operator: ComparisonOperator) -> bool:
    if operator is ComparisonOperator.GT:
        return left > right
    if operator is ComparisonOperator.GTE:
        return left >= right
    if operator is ComparisonOperator.LT:
        return left < right
    return left <= right
