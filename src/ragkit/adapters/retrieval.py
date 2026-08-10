"""Deterministic offline embedding, dense search, and no-op reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    InvalidDomainValueError,
    LimitExceededError,
    MetadataFilter,
    NormalizationMode,
    Not,
    Or,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    derive_chunk_id,
)
from ragkit.ports import (
    DeleteRequest,
    Embedder,
    EmbeddingBatch,
    EmbeddingRequest,
    Reranker,
    RerankRequest,
    RetrievalRequest,
    Retriever,
    SparseIndex,
    SparseUpsertRequest,
    UpsertRequest,
    VectorSearchRequest,
    VectorStore,
)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Every behavior-affecting BM25/tokenization parameter."""

    k1: float = 1.2
    b: float = 0.75
    token_pattern: str = r"[^\W_]+"
    lowercase: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise InvalidDomainValueError("BM25 k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise InvalidDomainValueError("BM25 b must be finite and in [0, 1]")
        if not self.token_pattern:
            raise InvalidDomainValueError("BM25 token pattern must not be empty")
        try:
            re.compile(self.token_pattern)
        except re.error as error:
            raise InvalidDomainValueError("BM25 token pattern must be valid") from error


class BM25Retriever(Retriever, SparseIndex):
    """Deterministic in-memory Robertson BM25 index with exact metadata filters."""

    def __init__(self, *, config: BM25Config | None = None) -> None:
        self._config = config or BM25Config()
        self._config.validate()
        self._tokenizer = re.compile(self._config.token_pattern)
        self._manifest: IndexManifest | None = None
        self._chunks: dict[ChunkId, Chunk] = {}
        self._lock = RLock()
        self._fingerprint = ComponentFingerprint.create(
            "retriever",
            "bm25_robertson",
            {
                "version": 1,
                "k1": self._config.k1,
                "b": self._config.b,
                "tokenizer": {
                    "pattern": self._config.token_pattern,
                    "lowercase": self._config.lowercase,
                    "unicode": True,
                },
                "idf": "log(1+(N-df+0.5)/(df+0.5))",
                "filter_policy": "pre_filter_statistics_v1",
            },
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    def upsert(self, request: SparseUpsertRequest) -> None:
        with self._lock:
            self._require_compatible(request.manifest)
            if any(
                chunk.chunk_id != derive_chunk_id(chunk, request.manifest.chunker_fingerprint)
                for chunk in request.chunks
            ):
                raise IntegrityError("sparse upsert chunk ID does not match stable content")
            pending = dict(self._chunks)
            for chunk in request.chunks:
                existing = pending.get(chunk.chunk_id)
                if existing is not None and existing != chunk:
                    raise IntegrityError("sparse upsert changed chunk content under the same ID")
                pending[chunk.chunk_id] = chunk
            self._chunks = pending
            if self._manifest is None:
                self._manifest = request.manifest

    def delete(self, request: DeleteRequest) -> None:
        with self._lock:
            self._require_initialized(request.expected_manifest)
            self._require_compatible(request.expected_manifest)
            pending = dict(self._chunks)
            for chunk_id in request.chunk_ids:
                pending.pop(chunk_id, None)
            self._chunks = pending

    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        with self._lock:
            self._require_initialized(request.expected_manifest)
            self._require_compatible(request.expected_manifest)
            chunks = tuple(self._chunks.values())
        query_terms = self._tokens(request.query)
        if not query_terms:
            return ()
        eligible = tuple(
            chunk
            for chunk in chunks
            if request.filters is None or _matches(request.filters, chunk.metadata)
        )
        if not eligible:
            return ()
        tokenized = {chunk.chunk_id: self._tokens(chunk.text) for chunk in eligible}
        average_length = sum(len(tokens) for tokens in tokenized.values()) / len(tokenized)
        document_frequency = Counter(
            term for tokens in tokenized.values() for term in set(tokens) if term in query_terms
        )
        provenance = ScoreProvenance(
            self.fingerprint,
            "sparse_retrieval",
            ScoreKind.SIMILARITY,
            "bm25",
            "identity:v1",
        )
        scored: list[tuple[Chunk, RetrievalScore]] = []
        for chunk in eligible:
            tokens = tokenized[chunk.chunk_id]
            raw = self._score(
                query_terms,
                Counter(tokens),
                len(tokens),
                average_length,
                document_frequency,
                len(eligible),
            )
            if raw > 0:
                scored.append((chunk, RetrievalScore.from_raw(raw, provenance)))
        ordered = sorted(scored, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def _tokens(self, text: str) -> tuple[str, ...]:
        normalized = text.casefold() if self._config.lowercase else text
        return tuple(match.group(0) for match in self._tokenizer.finditer(normalized))

    def _score(
        self,
        query_terms: Sequence[str],
        frequencies: Counter[str],
        document_length: int,
        average_length: float,
        document_frequency: Counter[str],
        document_count: int,
    ) -> float:
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue
            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            length_ratio = document_length / average_length if average_length else 0.0
            denominator = term_frequency + self._config.k1 * (
                1.0 - self._config.b + self._config.b * length_ratio
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (self._config.k1 + 1.0)
                / denominator
            )
        return score

    def _require_compatible(self, expected: IndexManifest) -> None:
        if self._manifest is not None:
            expected.require_compatible(self._manifest)

    def _require_initialized(self, expected: IndexManifest) -> None:
        if self._manifest is None:
            raise IndexCompatibilityError({"manifest": (expected.fingerprint, None)})


class DenseRetriever(Retriever):
    """Adapt an embedder and vector store to the query-level retriever port."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._fingerprint = ComponentFingerprint.create(
            "retriever",
            "dense_vector_search",
            {
                "embedder": str(embedder.fingerprint),
                "vector_store": str(vector_store.fingerprint),
            },
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        return self._vector_store.search(
            VectorSearchRequest(
                self._embedder.embed_query(request.query),
                self._embedder.fingerprint,
                request.top_k,
                request.filters,
                request.expected_manifest,
            )
        )


class HybridRetriever(Retriever):
    """Fuse named ranks with RRF; raw child scores remain uncalibrated history."""

    def __init__(
        self,
        retrievers: tuple[tuple[str, Retriever], ...],
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        max_candidates: int = 10_000,
    ) -> None:
        names = tuple(name for name, _ in retrievers)
        if not retrievers or any(not name.strip() for name in names):
            raise InvalidDomainValueError("hybrid retrievers require non-empty names")
        if len(set(names)) != len(names):
            raise InvalidDomainValueError("hybrid retriever names must be unique")
        if rrf_k <= 0 or candidate_multiplier <= 0 or max_candidates <= 0:
            raise InvalidDomainValueError("hybrid RRF parameters must be positive")
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates
        self._fingerprint = ComponentFingerprint.create(
            "retriever",
            "reciprocal_rank_fusion",
            {
                "version": 1,
                "rrf_k": rrf_k,
                "candidate_multiplier": candidate_multiplier,
                "max_candidates": max_candidates,
                "sources": tuple(
                    {"name": name, "fingerprint": str(retriever.fingerprint)}
                    for name, retriever in retrievers
                ),
                "score_policy": "sum(1/(rrf_k+source_rank))",
            },
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        candidate_top_k = request.top_k * self._candidate_multiplier
        if candidate_top_k > self._max_candidates:
            raise LimitExceededError(
                f"hybrid candidate request {candidate_top_k} exceeds {self._max_candidates}"
            )
        chunks: dict[ChunkId, Chunk] = {}
        contributions: dict[ChunkId, float] = {}
        histories: dict[ChunkId, list[RetrievalScore]] = {}
        for _, retriever in self._retrievers:
            candidates = retriever.retrieve(replace(request, top_k=candidate_top_k))
            source_ids = [candidate.chunk.chunk_id for candidate in candidates]
            if len(set(source_ids)) != len(source_ids):
                raise IntegrityError("hybrid source returned duplicate chunk IDs")
            for candidate in candidates:
                chunk_id = candidate.chunk.chunk_id
                if chunk_id in chunks and chunks[chunk_id] != candidate.chunk:
                    raise IntegrityError("retrievers returned different chunk values for one ID")
                chunks[chunk_id] = candidate.chunk
                contributions[chunk_id] = contributions.get(chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + candidate.rank
                )
                histories.setdefault(chunk_id, []).extend(
                    (candidate.score, *candidate.prior_scores)
                )
        provenance = ScoreProvenance(
            self.fingerprint,
            "hybrid_retrieval",
            ScoreKind.SIMILARITY,
            "rrf",
            "identity:v1",
        )
        ordered = sorted(
            chunks,
            key=lambda chunk_id: (-contributions[chunk_id], str(chunk_id)),
        )[: request.top_k]
        return tuple(
            ScoredChunk(
                chunks[chunk_id],
                RetrievalScore.from_raw(contributions[chunk_id], provenance),
                rank,
                tuple(histories[chunk_id]),
            )
            for rank, chunk_id in enumerate(ordered, start=1)
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

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def upsert(self, request: UpsertRequest) -> None:
        with self._lock:
            self._require_compatible(request.manifest)
            self._require_cosine_manifest(request.manifest)
            if any(
                chunk.chunk_id != derive_chunk_id(chunk, request.manifest.chunker_fingerprint)
                for chunk in request.chunks
            ):
                raise IntegrityError("vector upsert chunk ID does not match stable content")
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
            ScoredChunk(item.chunk, item.score, rank, item.prior_scores)
            for rank, item in enumerate(ordered, start=1)
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
