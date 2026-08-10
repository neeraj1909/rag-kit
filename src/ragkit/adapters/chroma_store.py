"""Optional persistent Chroma vector-store adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from ragkit.domain import (
    And,
    Chunk,
    Comparison,
    ComparisonOperator,
    ComponentFingerprint,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    MetadataFilter,
    MissingDependencyError,
    NormalizationMode,
    Not,
    Or,
    ProviderError,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    UnsupportedCapabilityError,
)
from ragkit.ports import DeleteRequest, UpsertRequest, VectorSearchRequest, VectorStore

_MANIFEST_KEY = "ragkit_manifest_v1"
_METRIC_KEY = "ragkit_metric"
_CHUNK_KEY = "ragkit_chunk_v1"
_USER_PREFIX = "ragkit_user_"


class _Collection(Protocol):
    metadata: Mapping[str, object] | None
    configuration: Mapping[str, object]

    def upsert(self, **kwargs: object) -> None: ...
    def query(self, **kwargs: object) -> Mapping[str, object]: ...
    def delete(self, **kwargs: object) -> None: ...


class _Client(Protocol):
    def get_collection(self, name: str, **kwargs: object) -> _Collection: ...
    def create_collection(self, name: str, **kwargs: object) -> _Collection: ...


class ChromaVectorStore(VectorStore):
    """Manifest-bound cosine-distance collection persisted by Chroma."""

    def __init__(self, path: Path | str, collection_name: str, *, client: object | None = None):
        if not collection_name.strip():
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("Chroma collection name must not be blank")
        self._path = Path(path)
        self._name = collection_name
        self._client = cast(_Client | None, client)
        self._fingerprint = ComponentFingerprint.create(
            "vector_store", "chroma_persistent", {"version": 1, "metric": "cosine"}
        )

    def upsert(self, request: UpsertRequest) -> None:
        self._require_cosine_manifest(request.manifest)
        for embedding in request.embeddings.embeddings:
            _require_unit_length(embedding.values)
        collection = self._open(request.manifest, create=True)
        if not request.chunks:
            return
        metadatas = [_metadata(chunk) for chunk in request.chunks]
        try:
            collection.upsert(
                ids=[str(chunk.chunk_id) for chunk in request.chunks],
                embeddings=[list(item.values) for item in request.embeddings.embeddings],
                documents=[chunk.text for chunk in request.chunks],
                metadatas=metadatas,
            )
        except Exception as exc:
            raise ProviderError("Chroma upsert failed", cause=exc) from exc

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self._require_cosine_manifest(request.expected_manifest)
        _require_unit_length(request.embedding.values)
        where = _translate_filter(request.filters) if request.filters is not None else None
        collection = self._open(request.expected_manifest, create=False)
        kwargs: dict[str, object] = {
            "query_embeddings": [list(request.embedding.values)],
            "n_results": request.top_k,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        try:
            result = collection.query(**kwargs)
            rows = _decode_query(result)
        except (IntegrityError, ValueError, TypeError):
            raise
        except Exception as exc:
            raise ProviderError("Chroma query failed", cause=exc) from exc
        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.DISTANCE,
            "cosine",
            "negate:v1",
        )
        scored = [
            (chunk, RetrievalScore.from_raw(distance, provenance)) for chunk, distance in rows
        ]
        scored.sort(key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))
        return tuple(
            ScoredChunk(chunk, score, rank)
            for rank, (chunk, score) in enumerate(scored[: request.top_k], start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        collection = self._open(request.expected_manifest, create=False)
        if not request.chunk_ids:
            return
        try:
            collection.delete(ids=[str(identifier) for identifier in request.chunk_ids])
        except Exception as exc:
            raise ProviderError("Chroma delete failed", cause=exc) from exc

    def _client_instance(self) -> _Client:
        if self._client is not None:
            return self._client
        try:
            module = import_module("chromadb")
            client_type = module.PersistentClient
        except ImportError as exc:
            raise MissingDependencyError(
                "Chroma adapter requires: install rag-kit[persistent]", cause=exc
            ) from exc
        try:
            self._client = cast(_Client, client_type(path=str(self._path)))
        except Exception as exc:
            raise ProviderError("Chroma client initialization failed", cause=exc) from exc
        return self._client

    def _open(self, expected: IndexManifest, *, create: bool) -> _Collection:
        client = self._client_instance()
        try:
            collection = client.get_collection(self._name, embedding_function=None)
        except Exception as exc:
            if not _is_missing_collection(exc):
                raise ProviderError("Chroma collection open failed", cause=exc) from exc
            if not create:
                raise IndexCompatibilityError({"manifest": (expected.fingerprint, None)}) from exc
            try:
                collection = client.create_collection(
                    self._name,
                    metadata={
                        _MANIFEST_KEY: json.dumps(
                            expected.to_dict(), sort_keys=True, separators=(",", ":")
                        ),
                        _METRIC_KEY: "cosine",
                    },
                    configuration={"hnsw": {"space": "cosine"}},
                    embedding_function=None,
                )
            except Exception as create_exc:
                raise ProviderError(
                    "Chroma collection creation failed", cause=create_exc
                ) from create_exc
        actual = _manifest_from_metadata(collection.metadata)
        expected.require_compatible(actual)
        if not collection.metadata or collection.metadata.get(_METRIC_KEY) != "cosine":
            raise IndexCompatibilityError(
                {"metric": ("cosine", (collection.metadata or {}).get(_METRIC_KEY))}
            )
        hnsw = collection.configuration.get("hnsw")
        actual_space = hnsw.get("space") if isinstance(hnsw, Mapping) else None
        if actual_space != "cosine":
            raise IndexCompatibilityError({"metric": ("cosine", actual_space)})
        return collection

    @staticmethod
    def _require_cosine_manifest(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


def _manifest_from_metadata(metadata: Mapping[str, object] | None) -> IndexManifest:
    encoded = metadata.get(_MANIFEST_KEY) if metadata else None
    if not isinstance(encoded, str):
        raise IntegrityError("Chroma collection manifest is missing")
    try:
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise TypeError
        return IndexManifest.from_dict(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IntegrityError("Chroma collection manifest is malformed", cause=exc) from exc


def _metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {
        _CHUNK_KEY: json.dumps(chunk.to_dict(), sort_keys=True, separators=(",", ":"))
    }
    for key, value in chunk.metadata.items():
        if value is not None:
            result[f"{_USER_PREFIX}{key}"] = value
    return result


def _decode_query(result: Mapping[str, object]) -> list[tuple[Chunk, float]]:
    raw_ids = result.get("ids")
    raw_metadatas = result.get("metadatas")
    raw_distances = result.get("distances")
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) != 1
        or not isinstance(raw_metadatas, list)
        or len(raw_metadatas) != 1
        or not isinstance(raw_distances, list)
        or len(raw_distances) != 1
    ):
        raise IntegrityError("Chroma query returned malformed result batches")
    identifiers = raw_ids[0]
    metadatas = raw_metadatas[0]
    distances = raw_distances[0]
    if (
        not isinstance(identifiers, list)
        or not isinstance(metadatas, list)
        or not isinstance(distances, list)
    ):
        raise IntegrityError("Chroma query returned malformed result arrays")
    if len(identifiers) != len(metadatas) or len(metadatas) != len(distances):
        raise IntegrityError("Chroma query result arrays are misaligned")
    decoded: list[tuple[Chunk, float]] = []
    seen: set[str] = set()
    for identifier, metadata, distance in zip(identifiers, metadatas, distances, strict=True):
        if (
            not isinstance(identifier, str)
            or not isinstance(metadata, Mapping)
            or not isinstance(distance, (int, float))
        ):
            raise IntegrityError("Chroma query returned malformed row")
        encoded = metadata.get(_CHUNK_KEY)
        if not isinstance(encoded, str) or not math.isfinite(float(distance)):
            raise IntegrityError("Chroma query returned invalid chunk or distance")
        try:
            value = json.loads(encoded)
            if not isinstance(value, dict):
                raise TypeError
            chunk = Chunk.from_dict(value)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntegrityError("Chroma stored chunk is malformed", cause=exc) from exc
        if identifier != str(chunk.chunk_id) or identifier in seen:
            raise IntegrityError("Chroma query returned invalid chunk identity")
        seen.add(identifier)
        decoded.append((chunk, float(distance)))
    return decoded


def _translate_filter(expression: MetadataFilter) -> dict[str, object]:
    if isinstance(expression, Comparison):
        field = f"{_USER_PREFIX}{expression.field}"
        if expression.value is None:
            raise UnsupportedCapabilityError(
                "Chroma cannot represent null metadata filters exactly",
                capability="metadata_filter:null",
            )
        operator = {
            ComparisonOperator.EQ: "$eq",
            ComparisonOperator.NE: "$ne",
            ComparisonOperator.GT: "$gt",
            ComparisonOperator.GTE: "$gte",
            ComparisonOperator.LT: "$lt",
            ComparisonOperator.LTE: "$lte",
            ComparisonOperator.IN: "$in",
        }[expression.operator]
        value = list(expression.value) if isinstance(expression.value, tuple) else expression.value
        return {field: {operator: value}}
    if isinstance(expression, And):
        if len(expression.children) == 1:
            return _translate_filter(expression.children[0])
        return {"$and": [_translate_filter(child) for child in expression.children]}
    if isinstance(expression, Or):
        if len(expression.children) == 1:
            return _translate_filter(expression.children[0])
        return {"$or": [_translate_filter(child) for child in expression.children]}
    if isinstance(expression, Not):
        raise UnsupportedCapabilityError(
            "Chroma does not support an exact general NOT metadata filter",
            capability="metadata_filter:not",
        )
    raise UnsupportedCapabilityError(
        "Chroma metadata filter is unsupported", capability="metadata_filter"
    )


def _require_unit_length(values: tuple[float, ...]) -> None:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("L2-normalized embeddings must have unit length")


def _is_missing_collection(exc: Exception) -> bool:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return "notfound" in name or "not found" in message or "does not exist" in message
