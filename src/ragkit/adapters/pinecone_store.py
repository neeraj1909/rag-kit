"""Manifest-bound Pinecone vector-store adapter.

The data-plane adapter deliberately does not create indexes or manifest records.
Provisioning is an explicit administrative operation because Pinecone cannot
conditionally create a manifest record without a first-writer race.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
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
    InvalidDomainValueError,
    LimitExceededError,
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
    canonical_json,
    derive_chunk_id,
)
from ragkit.ports import DeleteRequest, UpsertRequest, VectorSearchRequest, VectorStore

_SCHEMA = "pinecone-vector-store-v1"
_KIND = "_rk_kind"
_CHUNK = "_rk_chunk"
_MANIFEST = "_rk_manifest"


class _PineconeIndex(Protocol):
    def fetch(self, **kwargs: object) -> object: ...

    def upsert(self, **kwargs: object) -> object: ...

    def query(self, **kwargs: object) -> object: ...

    def delete(self, **kwargs: object) -> object: ...


class PineconeVectorStore(VectorStore):
    """Use a pre-provisioned cosine index and namespace manifest sentinel."""

    MANIFEST_ID = "__ragkit_manifest_v1__"

    def __init__(
        self,
        *,
        index_host: str,
        namespace: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        batch_size: int = 100,
        max_metadata_bytes: int = 40_000,
        max_top_k: int = 1_000,
        index: object | None = None,
    ) -> None:
        if not index_host.strip() or not namespace.strip() or not api_key.strip():
            raise InvalidDomainValueError("Pinecone host, namespace, and API key must not be blank")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(max_retries) is not int
            or max_retries < 0
        ):
            raise InvalidDomainValueError(
                "Pinecone timeout must be positive and retries non-negative"
            )
        if type(batch_size) is not int or not 1 <= batch_size <= 1_000:
            raise InvalidDomainValueError("Pinecone batch size must be in [1, 1000]")
        if type(max_metadata_bytes) is not int or not 1 <= max_metadata_bytes <= 40_000:
            raise InvalidDomainValueError("Pinecone metadata limit must be in [1, 40000]")
        if type(max_top_k) is not int or max_top_k <= 0:
            raise InvalidDomainValueError("Pinecone max_top_k must be positive")
        if index is None and importlib.util.find_spec("pinecone") is None:
            raise MissingDependencyError("Pinecone adapter requires: install rag-kit[pinecone]")
        self._namespace = namespace
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._max_metadata_bytes = max_metadata_bytes
        self._max_top_k = max_top_k
        if index is None:
            try:
                module = import_module("pinecone")
                retry_config = module.RetryConfig(max_retries=max_retries)
                control = module.Pinecone(
                    api_key=api_key,
                    timeout=self._timeout_seconds,
                    retry_config=retry_config,
                )
                self._index = cast(_PineconeIndex, control.Index(host=index_host))
            except Exception as error:  # pragma: no cover - optional SDK path
                raise _provider_error("initialization", error) from None
        else:
            self._index = cast(_PineconeIndex, index)
        self._fingerprint = ComponentFingerprint.create(
            "vector_store",
            "pinecone_cosine",
            {
                "version": 1,
                "metric": "cosine",
                "manifest": _SCHEMA,
                "metadata_codec": "typed-hash-v1",
                "consistency": "eventual",
                "batch_size": batch_size,
                "max_metadata_bytes": max_metadata_bytes,
                "max_top_k": max_top_k,
                "timeout_seconds": self._timeout_seconds,
                "max_retries": max_retries,
            },
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def require_compatible(self, manifest: IndexManifest) -> None:
        """Fetch and validate the pre-provisioned sentinel before data-plane work."""

        self._require_cosine(manifest)
        try:
            response = self._client.fetch(ids=[self.MANIFEST_ID], namespace=self._namespace)
        except Exception as error:
            raise _provider_error("manifest fetch", error) from None
        vectors = _field(response, "vectors")
        if not isinstance(vectors, Mapping) or self.MANIFEST_ID not in vectors:
            raise IndexCompatibilityError({"manifest": (manifest.fingerprint, None)})
        record = vectors[self.MANIFEST_ID]
        metadata = _field(record, "metadata")
        if not isinstance(metadata, Mapping):
            raise IntegrityError("Pinecone manifest metadata is malformed")
        if metadata.get(_KIND) != "manifest" or metadata.get("_rk_schema") != _SCHEMA:
            raise IntegrityError("Pinecone manifest metadata is malformed")
        encoded = metadata.get(_MANIFEST)
        if not isinstance(encoded, str):
            raise IntegrityError("Pinecone manifest metadata is malformed")
        try:
            import json

            decoded = json.loads(encoded)
            if not isinstance(decoded, dict):
                raise TypeError
            actual = IndexManifest.from_dict(decoded)
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("Pinecone manifest metadata is malformed", cause=error) from error
        manifest.require_compatible(actual)

    def upsert(self, request: UpsertRequest) -> None:
        self._require_cosine(request.manifest)
        records = [
            self._encode_record(chunk, embedding.values, request.manifest)
            for chunk, embedding in zip(request.chunks, request.embeddings.embeddings, strict=True)
        ]
        self.require_compatible(request.manifest)
        try:
            for offset in range(0, len(records), self._batch_size):
                self._client.upsert(
                    vectors=records[offset : offset + self._batch_size],
                    namespace=self._namespace,
                )
        except Exception as error:
            raise _provider_error("upsert", error) from None

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        if request.top_k > self._max_top_k:
            raise LimitExceededError("Pinecone top_k exceeds the configured limit")
        self._require_cosine(request.expected_manifest)
        _require_unit(request.embedding.values)
        compiled = _compile_filter(request.filters)
        record_filter: dict[str, object] = {_KIND: {"$eq": "chunk"}}
        if compiled is not None:
            record_filter = {"$and": [record_filter, compiled]}
        self.require_compatible(request.expected_manifest)
        try:
            response = self._client.query(
                namespace=self._namespace,
                vector=list(request.embedding.values),
                top_k=request.top_k,
                filter=record_filter,
                include_metadata=True,
                include_values=False,
            )
        except Exception as error:
            raise _provider_error("query", error) from None
        matches = _field(response, "matches")
        if not isinstance(matches, (list, tuple)):
            raise IntegrityError("Pinecone query response is malformed")
        provenance = ScoreProvenance(
            self._fingerprint, "dense_retrieval", ScoreKind.SIMILARITY, "cosine", "identity:v1"
        )
        scored: list[tuple[Chunk, RetrievalScore]] = []
        seen: set[str] = set()
        for match in matches:
            identifier = _field(match, "id")
            raw_score = _field(match, "score")
            metadata = _field(match, "metadata")
            if (
                not isinstance(identifier, str)
                or isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)
            ):
                raise IntegrityError("Pinecone query response is malformed")
            if identifier in seen:
                raise IntegrityError("Pinecone query returned duplicate chunk identity")
            seen.add(identifier)
            chunk = _decode_chunk(metadata, identifier, request.expected_manifest)
            scored.append((chunk, RetrievalScore.from_raw(float(raw_score), provenance)))
        ordered = sorted(scored, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        self.require_compatible(request.expected_manifest)
        identifiers = [str(item) for item in request.chunk_ids]
        try:
            for offset in range(0, len(identifiers), self._batch_size):
                self._client.delete(
                    ids=identifiers[offset : offset + self._batch_size],
                    namespace=self._namespace,
                )
        except Exception as error:
            raise _provider_error("delete", error) from None

    @property
    def _client(self) -> _PineconeIndex:
        return self._index

    def _encode_record(
        self, chunk: Chunk, values: tuple[float, ...], manifest: IndexManifest
    ) -> dict[str, object]:
        if chunk.chunk_id != derive_chunk_id(chunk, manifest.chunker_fingerprint):
            raise IntegrityError("Pinecone upsert chunk ID does not match stable content")
        _require_unit(values)
        metadata: dict[str, object] = {_KIND: "chunk", _CHUNK: canonical_json(chunk.to_dict())}
        for key, value in chunk.metadata.items():
            stem = _metadata_stem(key)
            metadata[f"{stem}_present"] = True
            if value is None:
                metadata[f"{stem}_null"] = True
            elif isinstance(value, bool):
                metadata[f"{stem}_bool"] = value
            elif isinstance(value, str):
                metadata[f"{stem}_str"] = value
            else:
                metadata[f"{stem}_num"] = value
        if len(canonical_json(metadata).encode("utf-8")) > self._max_metadata_bytes:
            raise IntegrityError("Pinecone record metadata exceeds configured bound")
        return {"id": str(chunk.chunk_id), "values": list(values), "metadata": metadata}

    @staticmethod
    def _require_cosine(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _decode_chunk(metadata: object, identifier: str, manifest: IndexManifest) -> Chunk:
    if not isinstance(metadata, Mapping) or metadata.get(_KIND) != "chunk":
        raise IntegrityError("Pinecone stored chunk metadata is malformed")
    encoded = metadata.get(_CHUNK)
    try:
        import json

        value = json.loads(cast(str, encoded))
        if not isinstance(value, dict):
            raise TypeError
        chunk = Chunk.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("Pinecone stored chunk is malformed", cause=error) from error
    if identifier != str(chunk.chunk_id) or chunk.chunk_id != derive_chunk_id(
        chunk, manifest.chunker_fingerprint
    ):
        raise IntegrityError("Pinecone stored chunk has invalid chunk identity")
    return chunk


def _metadata_stem(field: str) -> str:
    return "_rk_m_" + sha256(field.encode("utf-8")).hexdigest()


def _compile_filter(expression: object, *, negated: bool = False) -> dict[str, object] | None:
    if expression is None:
        return None
    if isinstance(expression, Not):
        return _compile_filter(expression.child, negated=not negated)
    if isinstance(expression, And | Or):
        is_and = isinstance(expression, And)
        operator = "$and" if is_and != negated else "$or"
        children = [_compile_filter(child, negated=negated) for child in expression.children]
        return {operator: [child for child in children if child is not None]}
    if not isinstance(expression, Comparison):
        raise UnsupportedCapabilityError(
            "Pinecone cannot represent this metadata filter", capability="pinecone_filter"
        )
    operator = expression.operator
    complements = {
        ComparisonOperator.EQ: "$ne",
        ComparisonOperator.NE: "$eq",
        ComparisonOperator.GT: "$lte",
        ComparisonOperator.GTE: "$lt",
        ComparisonOperator.LT: "$gte",
        ComparisonOperator.LTE: "$gt",
        ComparisonOperator.IN: "$nin",
    }
    normal = {
        ComparisonOperator.EQ: "$eq",
        ComparisonOperator.NE: "$ne",
        ComparisonOperator.GT: "$gt",
        ComparisonOperator.GTE: "$gte",
        ComparisonOperator.LT: "$lt",
        ComparisonOperator.LTE: "$lte",
        ComparisonOperator.IN: "$in",
    }
    value = list(expression.value) if isinstance(expression.value, tuple) else expression.value
    stem = _metadata_stem(expression.field)
    if value is None:
        absent: dict[str, object] = {f"{stem}_present": {"$exists": False}}
        explicit_null: dict[str, object] = {f"{stem}_null": {"$eq": True}}
        equals_null: dict[str, object] = {"$or": [absent, explicit_null]}
        if expression.operator not in {ComparisonOperator.EQ, ComparisonOperator.NE}:
            raise UnsupportedCapabilityError(
                "Pinecone ordered null filters are unsupported", capability="pinecone_filter"
            )
        is_equal = expression.operator is ComparisonOperator.EQ
        if negated:
            is_equal = not is_equal
        if is_equal:
            return equals_null
        return {
            "$and": [
                {f"{stem}_present": {"$exists": True}},
                {f"{stem}_null": {"$exists": False}},
            ]
        }
    if isinstance(value, list):
        if (
            not value
            or any(item is None for item in value)
            or len({type(item) for item in value}) != 1
        ):
            raise UnsupportedCapabilityError(
                "Pinecone IN filters require one non-empty scalar type",
                capability="pinecone_filter",
            )
        sample = value[0]
    else:
        sample = value
    if operator in {
        ComparisonOperator.GT,
        ComparisonOperator.GTE,
        ComparisonOperator.LT,
        ComparisonOperator.LTE,
    } and (isinstance(sample, bool) or not isinstance(sample, (int, float))):
        raise UnsupportedCapabilityError(
            "Pinecone ordered filters require numeric metadata",
            capability="pinecone_filter",
        )
    suffix = "bool" if isinstance(sample, bool) else "str" if isinstance(sample, str) else "num"
    selected = complements[operator] if negated else normal[operator]
    field = f"{stem}_{suffix}"
    comparison: dict[str, object] = {field: {selected: value}}
    is_complement = negated or operator is ComparisonOperator.NE
    if is_complement and selected != "$eq":
        # A missing typed projection means the domain comparison is false (and its
        # complement true), including absent keys and values of a different type.
        return {"$or": [comparison, {field: {"$exists": False}}]}
    return comparison


def _require_unit(values: tuple[float, ...]) -> None:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("Pinecone cosine embeddings must have unit length")


def _provider_error(operation: str, error: Exception) -> ProviderError:
    name = type(error).__name__.casefold()
    text = str(error).casefold()
    if "timeout" in name or "timed out" in text:
        return ProviderError(f"Pinecone {operation} timed out")
    if "rate" in name or "429" in text:
        return ProviderError(f"Pinecone {operation} was rate limited")
    return ProviderError(f"Pinecone {operation} failed")
