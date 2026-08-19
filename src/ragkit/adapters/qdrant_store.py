"""Qdrant vector-store adapter with an explicit manifest sentinel."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ragkit.domain import (
    And,
    Chunk,
    Comparison,
    ComparisonOperator,
    ComponentFingerprint,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    MetadataFilter,
    MissingDependencyError,
    NormalizationMode,
    Or,
    ProviderError,
    RagkitError,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    UnsupportedCapabilityError,
    derive_chunk_id,
)
from ragkit.ports import DeleteRequest, UpsertRequest, VectorSearchRequest, VectorStore

_ID_NAMESPACE = uuid.UUID("44f1df2d-1d6b-4cbe-9d6b-5d2aa60e916d")
_VECTOR_NAME = "ragkit_dense"


@dataclass(frozen=True, slots=True)
class QdrantCollection:
    vector_name: str
    dimension: int
    distance: str


@dataclass(frozen=True, slots=True)
class QdrantPoint:
    point_id: str
    vector: Mapping[str, tuple[float, ...]]
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class QdrantMatch:
    point_id: str
    score: float
    payload: Mapping[str, object]


class QdrantBackend(Protocol):
    def describe(self, collection_name: str) -> QdrantCollection | None: ...

    def create(self, collection_name: str, vector_name: str, dimension: int) -> None: ...

    def retrieve(
        self, collection_name: str, point_ids: tuple[str, ...]
    ) -> tuple[QdrantPoint, ...]: ...

    def upsert(self, collection_name: str, points: tuple[QdrantPoint, ...]) -> None: ...

    def query(
        self,
        collection_name: str,
        vector_name: str,
        vector: tuple[float, ...],
        limit: int,
        query_filter: dict[str, object] | None,
    ) -> tuple[QdrantMatch, ...]: ...

    def delete(self, collection_name: str, point_ids: tuple[str, ...]) -> None: ...


class QdrantVectorStore(VectorStore):
    """Store named cosine vectors while retaining complete chunks in payloads.

    Initial collection creation and sentinel insertion are not transactional. A
    pre-existing collection without the sentinel fails closed. Qdrant does not promise
    stable-ID tie breaking at the top-k cutoff; returned candidates are canonicalized.
    """

    def __init__(
        self,
        backend: QdrantBackend,
        collection_name: str,
        *,
        max_batch_size: int = 10_000,
        max_top_k: int = 1_000,
    ) -> None:
        if not collection_name.strip() or len(collection_name) > 128:
            raise InvalidDomainValueError(
                "Qdrant collection name must be nonblank and at most 128 characters"
            )
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise InvalidDomainValueError("Qdrant max_batch_size must be positive")
        if type(max_top_k) is not int or max_top_k <= 0:
            raise InvalidDomainValueError("Qdrant max_top_k must be positive")
        self._backend = backend
        self._collection = collection_name
        self._max_batch_size = max_batch_size
        self._max_top_k = max_top_k
        self._sentinel_id = str(uuid.uuid5(_ID_NAMESPACE, f"manifest:{collection_name}"))
        self._fingerprint = ComponentFingerprint.create(
            "vector_store",
            "qdrant_cosine",
            {
                "version": 1,
                "payload_schema_version": 1,
                "metric": "cosine_similarity",
                "vector_name": _VECTOR_NAME,
                "physical_index_policy": "hnsw",
                "filter_policy": "qdrant_exact_subset_v1",
                "max_batch_size": max_batch_size,
                "max_top_k": max_top_k,
            },
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        collection_name: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        **kwargs: object,
    ) -> QdrantVectorStore:
        if not isinstance(url, str) or not url.strip():
            raise InvalidDomainValueError("Qdrant URL must not be blank")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise InvalidDomainValueError("Qdrant timeout must be positive")
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
        except ImportError as error:
            raise MissingDependencyError(
                "Qdrant adapter requires: install rag-kit[qdrant]", cause=error
            ) from error
        try:
            client = QdrantClient(url=url, api_key=api_key, timeout=float(timeout_seconds))
        except Exception:
            raise ProviderError("Qdrant initialization failed") from None
        return cls(_QdrantSdkBackend(client), collection_name, **kwargs)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def require_compatible(self, manifest: IndexManifest) -> None:
        self._validate_manifest(manifest)
        try:
            collection = self._backend.describe(self._collection)
            if collection is None:
                return
            self._require_collection(collection, manifest)
            self._require_sentinel(manifest)
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("Qdrant compatibility check failed") from None

    def upsert(self, request: UpsertRequest) -> None:
        self._validate_manifest(request.manifest)
        if len(request.chunks) > self._max_batch_size:
            raise LimitExceededError("Qdrant upsert exceeds max_batch_size")
        if any(
            chunk.chunk_id != derive_chunk_id(chunk, request.manifest.chunker_fingerprint)
            for chunk in request.chunks
        ):
            raise IntegrityError("Qdrant upsert chunk ID does not match stable content")
        for embedding in request.embeddings.embeddings:
            _require_unit_length(embedding.values)
        points = tuple(
            self._chunk_point(chunk, embedding.values)
            for chunk, embedding in zip(request.chunks, request.embeddings.embeddings, strict=True)
        )
        try:
            collection = self._backend.describe(self._collection)
            if collection is None:
                self._backend.create(
                    self._collection, _VECTOR_NAME, request.manifest.embedding_dimension
                )
                sentinel = self._manifest_point(request.manifest)
                self._backend.upsert(self._collection, (sentinel, *points))
                return
            self._require_collection(collection, request.manifest)
            self._require_sentinel(request.manifest)
            if points:
                self._backend.upsert(self._collection, points)
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("Qdrant upsert failed") from None

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self._validate_manifest(request.expected_manifest)
        _require_unit_length(request.embedding.values)
        if request.top_k > self._max_top_k:
            raise LimitExceededError("Qdrant search exceeds max_top_k")
        query_filter = _compile_filter(request.filters)
        try:
            self._require_existing(request.expected_manifest)
            matches = self._backend.query(
                self._collection,
                _VECTOR_NAME,
                request.embedding.values,
                request.top_k,
                query_filter,
            )
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("Qdrant search failed") from None
        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.SIMILARITY,
            "cosine",
            "identity:v1",
        )
        decoded: list[tuple[Chunk, RetrievalScore]] = []
        seen: set[str] = set()
        for match in matches:
            chunk = self._decode_match(match, request.expected_manifest)
            identifier = str(chunk.chunk_id)
            if identifier in seen:
                raise IntegrityError("Qdrant result has duplicate chunk identity")
            seen.add(identifier)
            score = _finite_float(match.score, "Qdrant similarity")
            decoded.append((chunk, RetrievalScore.from_raw(score, provenance)))
        ordered = sorted(decoded, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        self._validate_manifest(request.expected_manifest)
        point_ids = tuple(_point_id(str(item)) for item in request.chunk_ids)
        try:
            self._require_existing(request.expected_manifest)
            if point_ids:
                self._backend.delete(self._collection, point_ids)
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("Qdrant delete failed") from None

    def _require_existing(self, manifest: IndexManifest) -> None:
        collection = self._backend.describe(self._collection)
        if collection is None:
            raise IndexCompatibilityError({"manifest": (manifest.fingerprint, None)})
        self._require_collection(collection, manifest)
        self._require_sentinel(manifest)

    @staticmethod
    def _require_collection(collection: QdrantCollection, manifest: IndexManifest) -> None:
        differences: dict[str, tuple[object, object]] = {}
        if collection.vector_name != _VECTOR_NAME:
            differences["vector_name"] = (_VECTOR_NAME, collection.vector_name)
        if collection.dimension != manifest.embedding_dimension:
            differences["embedding_dimension"] = (
                manifest.embedding_dimension,
                collection.dimension,
            )
        if collection.distance.casefold() != "cosine":
            differences["metric"] = ("cosine", collection.distance)
        if differences:
            raise IndexCompatibilityError(differences)

    def _require_sentinel(self, manifest: IndexManifest) -> None:
        points = self._backend.retrieve(self._collection, (self._sentinel_id,))
        if len(points) != 1:
            raise IndexCompatibilityError({"manifest": (manifest.fingerprint, None)})
        point = points[0]
        if not isinstance(point.payload, Mapping):
            raise IntegrityError("Qdrant manifest sentinel is malformed")
        ragkit = point.payload.get("ragkit")
        if not isinstance(ragkit, Mapping) or ragkit.get("record_kind") != "manifest":
            raise IntegrityError("Qdrant manifest sentinel is malformed")
        actual_value = ragkit.get("manifest")
        if not isinstance(actual_value, Mapping):
            raise IntegrityError("Qdrant manifest sentinel is malformed")
        try:
            actual = IndexManifest.from_dict(actual_value)
        except (KeyError, TypeError, ValueError):
            raise IntegrityError("Qdrant manifest sentinel is malformed") from None
        manifest.require_compatible(actual)

    def _manifest_point(self, manifest: IndexManifest) -> QdrantPoint:
        return QdrantPoint(
            self._sentinel_id,
            {_VECTOR_NAME: tuple(0.0 for _ in range(manifest.embedding_dimension))},
            {"ragkit": {"record_kind": "manifest", "manifest": manifest.to_dict()}},
        )

    @staticmethod
    def _chunk_point(chunk: Chunk, vector: tuple[float, ...]) -> QdrantPoint:
        identifier = str(chunk.chunk_id)
        return QdrantPoint(
            _point_id(identifier),
            {_VECTOR_NAME: vector},
            {
                "ragkit": {
                    "record_kind": "chunk",
                    "chunk_id": identifier,
                    "chunk": chunk.to_dict(),
                    "metadata": dict(chunk.metadata),
                }
            },
        )

    @staticmethod
    def _decode_match(match: QdrantMatch, manifest: IndexManifest) -> Chunk:
        if not isinstance(match.payload, Mapping):
            raise IntegrityError("Qdrant result payload is malformed")
        ragkit = match.payload.get("ragkit")
        if not isinstance(ragkit, Mapping) or ragkit.get("record_kind") != "chunk":
            raise IntegrityError("Qdrant result payload is malformed")
        raw_chunk = ragkit.get("chunk")
        raw_identifier = ragkit.get("chunk_id")
        if not isinstance(raw_chunk, Mapping) or not isinstance(raw_identifier, str):
            raise IntegrityError("Qdrant result payload is malformed")
        try:
            chunk = Chunk.from_dict(raw_chunk)
        except (KeyError, TypeError, ValueError):
            raise IntegrityError("Qdrant stored chunk is malformed") from None
        if str(chunk.chunk_id) != raw_identifier or match.point_id != _point_id(raw_identifier):
            raise IntegrityError("Qdrant result has invalid point identity")
        if chunk.chunk_id != derive_chunk_id(chunk, manifest.chunker_fingerprint):
            raise IntegrityError("Qdrant stored chunk has invalid stable identity")
        return chunk

    @staticmethod
    def _validate_manifest(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


class _QdrantSdkBackend:
    """Small translation layer around qdrant-client; imported only by ``from_url``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _models() -> Any:
        from qdrant_client.http import models  # type: ignore[import-not-found]

        return models

    def describe(self, collection_name: str) -> QdrantCollection | None:
        try:
            exists = self._client.collection_exists(collection_name)
            if not exists:
                return None
            info = self._client.get_collection(collection_name)
            vectors = info.config.params.vectors
            params = vectors[_VECTOR_NAME] if isinstance(vectors, Mapping) else vectors
            raw_distance = getattr(params.distance, "value", params.distance)
            return QdrantCollection(_VECTOR_NAME, int(params.size), str(raw_distance).casefold())
        except Exception:
            raise ProviderError("Qdrant describe failed") from None

    def create(self, collection_name: str, vector_name: str, dimension: int) -> None:
        models = self._models()
        self._client.create_collection(
            collection_name,
            vectors_config={
                vector_name: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
        )

    def retrieve(self, collection_name: str, point_ids: tuple[str, ...]) -> tuple[QdrantPoint, ...]:
        records = self._client.retrieve(
            collection_name, ids=list(point_ids), with_payload=True, with_vectors=True
        )
        return tuple(
            QdrantPoint(
                str(item.id),
                cast(Mapping[str, tuple[float, ...]], item.vector),
                item.payload,
            )
            for item in records
        )

    def upsert(self, collection_name: str, points: tuple[QdrantPoint, ...]) -> None:
        models = self._models()
        self._client.upsert(
            collection_name,
            points=[
                models.PointStruct(id=item.point_id, vector=dict(item.vector), payload=item.payload)
                for item in points
            ],
            wait=True,
        )

    def query(
        self,
        collection_name: str,
        vector_name: str,
        vector: tuple[float, ...],
        limit: int,
        query_filter: dict[str, object] | None,
    ) -> tuple[QdrantMatch, ...]:
        models = self._models()
        converted = None if query_filter is None else models.Filter(**query_filter)
        response = self._client.query_points(
            collection_name,
            query=list(vector),
            using=vector_name,
            query_filter=converted,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return tuple(
            QdrantMatch(str(item.id), float(item.score), item.payload) for item in response.points
        )

    def delete(self, collection_name: str, point_ids: tuple[str, ...]) -> None:
        models = self._models()
        self._client.delete(
            collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )


def _compile_filter(expression: MetadataFilter | None) -> dict[str, object]:
    record_kind = {"key": "ragkit.record_kind", "match": {"value": "chunk"}}
    if expression is None:
        return {"must": [record_kind]}
    return {"must": [record_kind, _compile_filter_node(expression)]}


def _compile_filter_node(expression: MetadataFilter) -> dict[str, object]:
    if isinstance(expression, Comparison):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression.field) is None:
            raise UnsupportedCapabilityError(
                "Qdrant cannot safely address this metadata field",
                capability="metadata_filter",
            )
        key = f"ragkit.metadata.{expression.field}"
        value = expression.value
        if value is None:
            raise UnsupportedCapabilityError(
                "Qdrant null/missing filter semantics are not exact",
                capability="metadata_filter",
            )
        if expression.operator is ComparisonOperator.EQ:
            if isinstance(value, float):
                return {"key": key, "range": {"gte": value, "lte": value}}
            return {"key": key, "match": {"value": value}}
        if expression.operator is ComparisonOperator.IN:
            assert isinstance(value, tuple)
            if (
                not value
                or any(item is None or isinstance(item, float) for item in value)
                or len({type(item) for item in value}) != 1
            ):
                raise UnsupportedCapabilityError(
                    "Qdrant IN filters require homogeneous exact scalar values",
                    capability="metadata_filter",
                )
            return {"key": key, "match": {"any": list(value)}}
        if expression.operator is ComparisonOperator.NE:
            raise UnsupportedCapabilityError(
                "Qdrant inequality has ambiguous missing-field semantics",
                capability="metadata_filter",
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedCapabilityError(
                "Qdrant ordered filters require numeric metadata",
                capability="metadata_filter",
            )
        bound = {
            ComparisonOperator.GT: "gt",
            ComparisonOperator.GTE: "gte",
            ComparisonOperator.LT: "lt",
            ComparisonOperator.LTE: "lte",
        }[expression.operator]
        return {"key": key, "range": {bound: value}}
    if isinstance(expression, And):
        return {"must": [_compile_filter_node(item) for item in expression.children]}
    if isinstance(expression, Or):
        return {"should": [_compile_filter_node(item) for item in expression.children]}
    raise UnsupportedCapabilityError(
        "Qdrant cannot exactly represent this metadata filter",
        capability="metadata_filter",
    )


def _point_id(identifier: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"chunk:{identifier}"))


def _require_unit_length(values: Sequence[float]) -> None:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("L2-normalized embeddings must have unit length")


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError(f"{label} is malformed")
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError(f"{label} is malformed")
    return result
