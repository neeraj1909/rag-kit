"""Manifest-bound OpenSearch Lucene HNSW vector-store adapter."""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module
from typing import Protocol, cast
from urllib.parse import urlparse

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
    derive_chunk_id,
)
from ragkit.ports import DeleteRequest, UpsertRequest, VectorSearchRequest, VectorStore

_SCHEMA = "opensearch-vector-store-v1"


class _Indices(Protocol):
    def exists(self, **kwargs: object) -> object: ...

    def create(self, **kwargs: object) -> object: ...

    def get_mapping(self, **kwargs: object) -> object: ...


class _OpenSearchClient(Protocol):
    indices: _Indices

    def bulk(self, **kwargs: object) -> object: ...

    def search(self, **kwargs: object) -> object: ...


def _new_client(
    url: str,
    *,
    username: str | None,
    password: str | None,
    ca_certs: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> _OpenSearchClient:
    try:
        module = import_module("opensearchpy")
        auth = None if username is None else (username, cast(str, password))
        kwargs: dict[str, object] = {
            "hosts": [url],
            "http_compress": True,
            "use_ssl": url.startswith("https://"),
            "verify_certs": url.startswith("https://"),
            "timeout": timeout_seconds,
            "max_retries": max_retries,
            "retry_on_timeout": max_retries > 0,
        }
        if auth is not None:
            kwargs["http_auth"] = auth
        if ca_certs:
            kwargs["ca_certs"] = ca_certs
        return cast(_OpenSearchClient, module.OpenSearch(**kwargs))
    except Exception as error:  # pragma: no cover - optional SDK path
        raise _provider_error("initialization", error) from None


class OpenSearchVectorStore(VectorStore):
    """Persist exact chunks beside approximate Lucene cosine candidates."""

    def __init__(
        self,
        *,
        url: str,
        index_name: str,
        username: str | None = None,
        password: str | None = None,
        ca_certs: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        batch_size: int = 100,
        max_top_k: int = 1_000,
        client: object | None = None,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise InvalidDomainValueError("OpenSearch URL must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise InvalidDomainValueError("plain HTTP OpenSearch is allowed only on loopback")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,254}", index_name) is None or index_name in {
            ".",
            "..",
        }:
            raise InvalidDomainValueError("OpenSearch index name is invalid")
        if (username is None) != (password is None) or (
            username is not None and (not username or not password)
        ):
            raise InvalidDomainValueError(
                "OpenSearch username and password must be supplied together"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(max_retries) is not int
            or max_retries < 0
        ):
            raise InvalidDomainValueError(
                "OpenSearch timeout must be positive and retries non-negative"
            )
        if type(batch_size) is not int or not 1 <= batch_size <= 1_000:
            raise InvalidDomainValueError("OpenSearch batch size must be in [1, 1000]")
        if type(max_top_k) is not int or max_top_k <= 0:
            raise InvalidDomainValueError("OpenSearch max_top_k must be positive")
        if client is None and importlib.util.find_spec("opensearchpy") is None:
            raise MissingDependencyError("OpenSearch adapter requires: install rag-kit[opensearch]")
        self._url = url
        self._name = index_name
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._max_top_k = max_top_k
        self._injected = (
            cast(_OpenSearchClient, client)
            if client is not None
            else _new_client(
                url,
                username=username,
                password=password,
                ca_certs=ca_certs,
                timeout_seconds=self._timeout_seconds,
                max_retries=max_retries,
            )
        )
        self._fingerprint = ComponentFingerprint.create(
            "vector_store",
            "opensearch_lucene_hnsw_cosine",
            {
                "version": 1,
                "schema": _SCHEMA,
                "engine": "lucene",
                "method": "hnsw",
                "space_type": "cosinesimil",
                "native_score": "opensearch_lucene_cosinesimil_score",
                "refresh": "wait_for",
                "batch_size": batch_size,
                "max_top_k": max_top_k,
                "timeout_seconds": self._timeout_seconds,
                "max_retries": max_retries,
            },
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def require_compatible(self, manifest: IndexManifest) -> None:
        """Validate existing state; an absent index is compatible and remains absent."""

        self._require_compatible(manifest, create=False, allow_missing=True)

    def upsert(self, request: UpsertRequest) -> None:
        self._require_cosine(request.manifest)
        documents = [
            self._encode_document(chunk, embedding.values, request.manifest)
            for chunk, embedding in zip(request.chunks, request.embeddings.embeddings, strict=True)
        ]
        self._require_compatible(request.manifest, create=True)
        for offset in range(0, len(documents), self._batch_size):
            body: list[object] = []
            for identifier, source in documents[offset : offset + self._batch_size]:
                body.extend(({"index": {"_index": self._name, "_id": identifier}}, source))
            try:
                response = self._client.bulk(body=body, refresh="wait_for")
            except Exception as error:
                raise _provider_error("bulk upsert", error) from None
            _require_bulk_success(response, operation="upsert")

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        if request.top_k > self._max_top_k:
            raise LimitExceededError("OpenSearch top_k exceeds the configured limit")
        self._require_cosine(request.expected_manifest)
        _require_unit(request.embedding.values)
        compiled = _compile_filter(request.filters)
        knn: dict[str, object] = {
            "vector": list(request.embedding.values),
            "k": request.top_k,
        }
        if compiled is not None:
            knn["filter"] = compiled
        body = {
            "size": request.top_k,
            "_source": ["chunk"],
            "query": {"knn": {"embedding": knn}},
        }
        self._require_compatible(request.expected_manifest, create=False)
        try:
            response = self._client.search(index=self._name, body=body)
        except Exception as error:
            raise _provider_error("search", error) from None
        hits_root = _mapping(response, "OpenSearch search response")
        hits_value = hits_root.get("hits")
        hits_mapping = _mapping(hits_value, "OpenSearch search response")
        hits = hits_mapping.get("hits")
        if not isinstance(hits, list):
            raise IntegrityError("OpenSearch search response is malformed")
        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.SIMILARITY,
            "opensearch_lucene_cosinesimil_score",
            "identity:v1",
        )
        scored: list[tuple[Chunk, RetrievalScore]] = []
        seen: set[str] = set()
        for hit in hits:
            item = _mapping(hit, "OpenSearch search hit")
            identifier, raw, source = item.get("_id"), item.get("_score"), item.get("_source")
            if (
                not isinstance(identifier, str)
                or isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
            ):
                raise IntegrityError("OpenSearch search hit is malformed")
            if identifier in seen:
                raise IntegrityError("OpenSearch search returned duplicate chunk identity")
            seen.add(identifier)
            source_mapping = _mapping(source, "OpenSearch stored source")
            raw_chunk = source_mapping.get("chunk")
            try:
                chunk = Chunk.from_dict(_mapping(raw_chunk, "OpenSearch stored chunk"))
            except (KeyError, TypeError, ValueError) as error:
                raise IntegrityError("OpenSearch stored chunk is malformed", cause=error) from error
            if identifier != str(chunk.chunk_id) or chunk.chunk_id != derive_chunk_id(
                chunk, request.expected_manifest.chunker_fingerprint
            ):
                raise IntegrityError("OpenSearch stored chunk has invalid chunk identity")
            scored.append((chunk, RetrievalScore.from_raw(float(raw), provenance)))
        ordered = sorted(scored, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        self._require_compatible(request.expected_manifest, create=False)
        identifiers = [str(item) for item in request.chunk_ids]
        for offset in range(0, len(identifiers), self._batch_size):
            body: list[object] = [
                {"delete": {"_index": self._name, "_id": identifier}}
                for identifier in identifiers[offset : offset + self._batch_size]
            ]
            try:
                response = self._client.bulk(body=body, refresh="wait_for")
            except Exception as error:
                raise _provider_error("bulk delete", error) from None
            _require_bulk_success(response, operation="delete")

    @property
    def _client(self) -> _OpenSearchClient:
        return self._injected

    def _require_compatible(
        self, manifest: IndexManifest, *, create: bool, allow_missing: bool = False
    ) -> None:
        self._require_cosine(manifest)
        try:
            exists = bool(self._client.indices.exists(index=self._name))
            if not exists:
                if allow_missing:
                    return
                if not create:
                    raise IndexCompatibilityError({"manifest": (manifest.fingerprint, None)})
                try:
                    self._client.indices.create(index=self._name, body=_index_definition(manifest))
                except Exception:
                    # A concurrent creator may have won. Only continue if the index now exists;
                    # its manifest is still checked before any document operation.
                    if not bool(self._client.indices.exists(index=self._name)):
                        raise
            response = self._client.indices.get_mapping(index=self._name)
        except IndexCompatibilityError:
            raise
        except Exception as error:
            raise _provider_error("manifest check", error) from None
        root = _mapping(response, "OpenSearch mapping response")
        entry = _mapping(root.get(self._name), "OpenSearch index mapping")
        mappings = _mapping(entry.get("mappings"), "OpenSearch mappings")
        meta = _mapping(mappings.get("_meta"), "OpenSearch mapping metadata")
        ragkit = _mapping(meta.get("ragkit"), "OpenSearch ragkit manifest")
        if ragkit.get("schema") != _SCHEMA:
            raise IntegrityError("OpenSearch ragkit manifest is malformed")
        try:
            actual = IndexManifest.from_dict(
                _mapping(ragkit.get("manifest"), "OpenSearch ragkit manifest")
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("OpenSearch ragkit manifest is malformed", cause=error) from error
        manifest.require_compatible(actual)
        _validate_mapping(mappings, manifest)

    def _encode_document(
        self, chunk: Chunk, values: tuple[float, ...], manifest: IndexManifest
    ) -> tuple[str, dict[str, object]]:
        if chunk.chunk_id != derive_chunk_id(chunk, manifest.chunker_fingerprint):
            raise IntegrityError("OpenSearch upsert chunk ID does not match stable content")
        _require_unit(values)
        entries: list[dict[str, object]] = []
        for key, value in chunk.metadata.items():
            entry: dict[str, object] = {"key_hash": _key_hash(key), "present": True}
            if value is None:
                entry["is_null"] = True
            elif isinstance(value, bool):
                entry["boolean"] = value
            elif isinstance(value, str):
                entry["string"] = value
            else:
                entry["number"] = value
            entries.append(entry)
        return str(chunk.chunk_id), {
            "record_kind": "chunk",
            "embedding": list(values),
            "chunk": chunk.to_dict(),
            "metadata_entries": entries,
        }

    @staticmethod
    def _require_cosine(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


def _index_definition(manifest: IndexManifest) -> dict[str, object]:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "_meta": {"ragkit": {"schema": _SCHEMA, "manifest": manifest.to_dict()}},
            "properties": {
                "record_kind": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": manifest.embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
                "chunk": {"type": "object", "enabled": False},
                "metadata_entries": {
                    "type": "nested",
                    "properties": {
                        "key_hash": {"type": "keyword"},
                        "present": {"type": "boolean"},
                        "is_null": {"type": "boolean"},
                        "string": {"type": "keyword"},
                        "number": {"type": "double"},
                        "boolean": {"type": "boolean"},
                    },
                },
            },
        },
    }


def _validate_mapping(mappings: Mapping[str, object], manifest: IndexManifest) -> None:
    properties = _mapping(mappings.get("properties"), "OpenSearch mappings")
    embedding = _mapping(properties.get("embedding"), "OpenSearch embedding mapping")
    method = _mapping(embedding.get("method"), "OpenSearch embedding method")
    expected = {
        "type": "knn_vector",
        "dimension": manifest.embedding_dimension,
    }
    actual = {key: embedding.get(key) for key in expected}
    if actual != expected or any(
        method.get(key) != value
        for key, value in {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}.items()
    ):
        raise IndexCompatibilityError({"vector_mapping": (expected, actual)})
    record_kind = _mapping(properties.get("record_kind"), "OpenSearch record-kind mapping")
    chunk = _mapping(properties.get("chunk"), "OpenSearch chunk mapping")
    metadata = _mapping(properties.get("metadata_entries"), "OpenSearch metadata mapping")
    metadata_properties = _mapping(metadata.get("properties"), "OpenSearch metadata properties")
    field_types = {
        "key_hash": "keyword",
        "present": "boolean",
        "is_null": "boolean",
        "string": "keyword",
        "number": "double",
        "boolean": "boolean",
    }
    schema_valid = (
        record_kind.get("type") == "keyword"
        and chunk.get("type") == "object"
        and chunk.get("enabled") is False
        and metadata.get("type") == "nested"
        and all(
            _mapping(metadata_properties.get(field), "OpenSearch metadata field").get("type")
            == field_type
            for field, field_type in field_types.items()
        )
    )
    if not schema_valid:
        raise IndexCompatibilityError({"storage_mapping": (_SCHEMA, "incompatible")})


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} is malformed")
    return cast(Mapping[str, object], value)


def _key_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _compile_filter(expression: object, *, negated: bool = False) -> dict[str, object] | None:
    if expression is None:
        return None
    if isinstance(expression, Not):
        return _compile_filter(expression.child, negated=not negated)
    if isinstance(expression, And | Or):
        is_and = isinstance(expression, And)
        key = "must" if is_and != negated else "should"
        children = [_compile_filter(child, negated=negated) for child in expression.children]
        body: dict[str, object] = {key: [item for item in children if item is not None]}
        if key == "should":
            body["minimum_should_match"] = 1
        return {"bool": body}
    if not isinstance(expression, Comparison):
        raise UnsupportedCapabilityError(
            "OpenSearch cannot represent this metadata filter", capability="opensearch_filter"
        )
    base = _comparison_query(expression)
    return {"bool": {"must_not": [base]}} if negated else base


def _comparison_query(expression: Comparison) -> dict[str, object]:
    nested_key: dict[str, object] = {
        "term": {"metadata_entries.key_hash": _key_hash(expression.field)}
    }
    value = expression.value
    if value is None:
        present: dict[str, object] = {"nested": {"path": "metadata_entries", "query": nested_key}}
        explicit_null: dict[str, object] = {
            "nested": {
                "path": "metadata_entries",
                "query": {
                    "bool": {"must": [nested_key, {"term": {"metadata_entries.is_null": True}}]}
                },
            }
        }
        if expression.operator is ComparisonOperator.EQ:
            return {
                "bool": {
                    "should": [{"bool": {"must_not": [present]}}, explicit_null],
                    "minimum_should_match": 1,
                }
            }
        if expression.operator is ComparisonOperator.NE:
            return {"bool": {"must": [present], "must_not": [explicit_null]}}
        raise UnsupportedCapabilityError(
            "OpenSearch ordered null filters are unsupported", capability="opensearch_filter"
        )
    if expression.operator is ComparisonOperator.IN:
        assert isinstance(value, tuple)
        if not value or len({type(item) for item in value}) != 1:
            raise UnsupportedCapabilityError(
                "OpenSearch IN filters require one non-empty scalar type",
                capability="opensearch_filter",
            )
        leaf: dict[str, object] = {
            "terms": {f"metadata_entries.{_value_field(value[0])}": list(value)}
        }
    elif expression.operator in {ComparisonOperator.EQ, ComparisonOperator.NE}:
        leaf = {"term": {f"metadata_entries.{_value_field(value)}": value}}
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedCapabilityError(
                "OpenSearch ordered filters require numeric metadata",
                capability="opensearch_filter",
            )
        leaf = {
            "range": {f"metadata_entries.{_value_field(value)}": {expression.operator.value: value}}
        }
    nested: dict[str, object] = {
        "nested": {
            "path": "metadata_entries",
            "query": {"bool": {"must": [nested_key, leaf]}},
        }
    }
    if expression.operator is ComparisonOperator.NE:
        return {"bool": {"must_not": [nested]}}
    return nested


def _value_field(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    raise UnsupportedCapabilityError(
        "OpenSearch filter scalar is unsupported", capability="opensearch_filter"
    )


def _require_bulk_success(response: object, *, operation: str) -> None:
    root = _mapping(response, "OpenSearch bulk response")
    if root.get("errors") is False:
        return
    items = root.get("items")
    failed = 0
    inspected = False
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping):
                detail = next(iter(item.values()), None)
                if isinstance(detail, Mapping):
                    inspected = True
                    status = detail.get("status")
                    if isinstance(status, int) and status >= 300:
                        if operation == "delete" and status == 404:
                            continue
                        failed += 1
    if inspected and failed == 0:
        return
    raise ProviderError(f"OpenSearch bulk {operation} failed for {failed or 'unknown'} item(s)")


def _require_unit(values: tuple[float, ...]) -> None:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("OpenSearch cosine embeddings must have unit length")


def _provider_error(operation: str, error: Exception) -> ProviderError:
    name = type(error).__name__.casefold()
    text = str(error).casefold()
    if "timeout" in name or "timed out" in text:
        return ProviderError(f"OpenSearch {operation} timed out")
    if "429" in text or "rate" in name:
        return ProviderError(f"OpenSearch {operation} was rate limited")
    return ProviderError(f"OpenSearch {operation} failed")
