"""PostgreSQL/pgvector adapter with transactional manifest preflight."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, Protocol, cast

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
    Not,
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

PgVectorIndexPolicy = Literal["exact", "hnsw", "ivf_flat"]


class _Cursor(Protocol):
    def execute(self, query: str, params: tuple[object, ...] = ()) -> None: ...

    def executemany(self, query: str, params: Iterable[tuple[object, ...]]) -> None: ...

    def fetchone(self) -> Sequence[object] | None: ...

    def fetchall(self) -> Sequence[Sequence[object]]: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class PgVectorStore(VectorStore):
    """Store L2-normalized vectors in pre-provisioned pgvector tables.

    ``exact`` has deterministic cutoff semantics. ``hnsw`` and ``ivfflat`` identify
    an externally provisioned physical index, but PostgreSQL may return approximate
    candidates; the adapter only canonicalizes the candidates it receives.
    """

    def __init__(
        self,
        connection_factory: Callable[[], _Connection],
        collection_name: str,
        *,
        physical_index_policy: PgVectorIndexPolicy = "exact",
        max_batch_size: int = 10_000,
        max_top_k: int = 1_000,
    ) -> None:
        if not callable(connection_factory):
            raise InvalidDomainValueError("pgvector connection_factory must be callable")
        if not collection_name.strip() or len(collection_name) > 128:
            raise InvalidDomainValueError(
                "pgvector collection name must be nonblank and at most 128 characters"
            )
        if physical_index_policy not in {"exact", "hnsw", "ivf_flat"}:
            raise InvalidDomainValueError("unknown pgvector physical index policy")
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise InvalidDomainValueError("pgvector max_batch_size must be positive")
        if type(max_top_k) is not int or max_top_k <= 0:
            raise InvalidDomainValueError("pgvector max_top_k must be positive")
        self._connection_factory = connection_factory
        self._collection = collection_name
        self._physical_index_policy = physical_index_policy
        self._max_batch_size = max_batch_size
        self._max_top_k = max_top_k
        self._fingerprint = ComponentFingerprint.create(
            "vector_store",
            "pgvector_cosine",
            {
                "version": 1,
                "schema_version": 1,
                "metric": "cosine_distance",
                "filter_policy": "jsonb_exact_v1",
                "physical_index_policy": physical_index_policy,
                "max_batch_size": max_batch_size,
                "max_top_k": max_top_k,
            },
        )

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        collection_name: str,
        **kwargs: object,
    ) -> PgVectorStore:
        """Build a store while keeping the optional driver import at composition time."""

        if not isinstance(dsn, str) or not dsn.strip():
            raise InvalidDomainValueError("pgvector DSN must not be blank")
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as error:
            raise MissingDependencyError(
                "pgvector adapter requires: install rag-kit[pgvector]", cause=error
            ) from error

        def connect() -> _Connection:
            return cast(_Connection, psycopg.connect(dsn))

        return cls(connect, collection_name, **kwargs)  # type: ignore[arg-type]

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def require_compatible(self, manifest: IndexManifest) -> None:
        self._validate_manifest(manifest)
        connection = self._connect("compatibility check")
        cursor = connection.cursor()
        try:
            actual = self._read_manifest(cursor)
            if actual is None:
                return
            manifest.require_compatible(actual)
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("pgvector compatibility check failed") from None
        finally:
            cursor.close()
            connection.close()

    def upsert(self, request: UpsertRequest) -> None:
        self._validate_manifest(request.manifest)
        if len(request.chunks) > self._max_batch_size:
            raise LimitExceededError("pgvector upsert exceeds max_batch_size")
        if any(
            chunk.chunk_id != derive_chunk_id(chunk, request.manifest.chunker_fingerprint)
            for chunk in request.chunks
        ):
            raise IntegrityError("pgvector upsert chunk ID does not match stable content")
        for embedding in request.embeddings.embeddings:
            _require_unit_length(embedding.values)

        connection = self._connect("upsert")
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("LOCK TABLE ragkit_manifests IN SHARE ROW EXCLUSIVE MODE")
            actual = self._read_manifest(cursor)
            if actual is None:
                cursor.execute(
                    "INSERT INTO ragkit_manifests(collection_name, manifest_json) "
                    "VALUES (%s, %s::jsonb)",
                    (self._collection, _canonical_json(request.manifest.to_dict())),
                )
            else:
                request.manifest.require_compatible(actual)
            cursor.executemany(
                """
                INSERT INTO ragkit_entries(
                    collection_name, chunk_id, chunk_json, metadata, embedding
                ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::vector)
                ON CONFLICT(collection_name, chunk_id) DO UPDATE SET
                    chunk_json = excluded.chunk_json,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding
                """,
                (
                    (
                        self._collection,
                        str(chunk.chunk_id),
                        _canonical_json(chunk.to_dict()),
                        _canonical_json(dict(chunk.metadata)),
                        _vector_literal(embedding.values),
                    )
                    for chunk, embedding in zip(
                        request.chunks, request.embeddings.embeddings, strict=True
                    )
                ),
            )
            connection.commit()
        except RagkitError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise ProviderError("pgvector upsert failed") from None
        finally:
            cursor.close()
            connection.close()

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self._validate_manifest(request.expected_manifest)
        _require_unit_length(request.embedding.values)
        if request.top_k > self._max_top_k:
            raise LimitExceededError("pgvector search exceeds max_top_k")
        filter_sql, filter_params = _compile_filter(request.filters)
        connection = self._connect("search")
        cursor = connection.cursor()
        try:
            actual = self._read_manifest(cursor)
            if actual is None:
                raise IndexCompatibilityError(
                    {"manifest": (request.expected_manifest.fingerprint, None)}
                )
            request.expected_manifest.require_compatible(actual)
            vector = _vector_literal(request.embedding.values)
            dimension = request.expected_manifest.embedding_dimension
            distance_expression = "embedding <=> %s::vector"
            if self._physical_index_policy != "exact":
                # The cast matches the documented partial expression indexes used when
                # collections with different dimensions share the physical table.
                distance_expression = f"embedding::vector({dimension}) <=> %s::vector({dimension})"
            cursor.execute(
                f"SELECT chunk_json, {distance_expression} AS distance "
                "FROM ragkit_entries WHERE collection_name = %s"
                f"{filter_sql} ORDER BY distance ASC, chunk_id ASC LIMIT %s",
                (vector, self._collection, *filter_params, request.top_k),
            )
            rows = cursor.fetchall()
        except RagkitError:
            raise
        except Exception:
            raise ProviderError("pgvector search failed") from None
        finally:
            cursor.close()
            connection.close()

        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.DISTANCE,
            "cosine",
            "negate:v1",
        )
        decoded: list[tuple[Chunk, RetrievalScore]] = []
        seen: set[str] = set()
        for row in rows:
            if len(row) != 2:
                raise IntegrityError("pgvector stored row is malformed")
            chunk = _decode_chunk(row[0], request.expected_manifest)
            identifier = str(chunk.chunk_id)
            if identifier in seen:
                raise IntegrityError("pgvector result has duplicate chunk identity")
            seen.add(identifier)
            distance = _finite_float(row[1], "pgvector distance")
            decoded.append((chunk, RetrievalScore.from_raw(distance, provenance)))
        ordered = sorted(decoded, key=lambda item: (-item[1].relevance, str(item[0].chunk_id)))[
            : request.top_k
        ]
        return tuple(
            ScoredChunk(chunk, score, rank) for rank, (chunk, score) in enumerate(ordered, start=1)
        )

    def delete(self, request: DeleteRequest) -> None:
        self._validate_manifest(request.expected_manifest)
        connection = self._connect("delete")
        cursor = connection.cursor()
        try:
            cursor.execute("BEGIN")
            cursor.execute("LOCK TABLE ragkit_manifests IN SHARE ROW EXCLUSIVE MODE")
            actual = self._read_manifest(cursor)
            if actual is None:
                raise IndexCompatibilityError(
                    {"manifest": (request.expected_manifest.fingerprint, None)}
                )
            request.expected_manifest.require_compatible(actual)
            cursor.execute(
                "DELETE FROM ragkit_entries WHERE collection_name = %s AND chunk_id = ANY(%s)",
                (self._collection, [str(item) for item in request.chunk_ids]),
            )
            connection.commit()
        except RagkitError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise ProviderError("pgvector delete failed") from None
        finally:
            cursor.close()
            connection.close()

    def _connect(self, operation: str) -> _Connection:
        try:
            return self._connection_factory()
        except RagkitError:
            raise
        except Exception:
            raise ProviderError(f"pgvector {operation} failed") from None

    def _read_manifest(self, cursor: _Cursor) -> IndexManifest | None:
        cursor.execute(
            "SELECT manifest_json FROM ragkit_manifests WHERE collection_name = %s",
            (self._collection,),
        )
        row = cursor.fetchone()
        return None if row is None else _decode_manifest(row[0])

    @staticmethod
    def _validate_manifest(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


def _compile_filter(expression: MetadataFilter | None) -> tuple[str, tuple[object, ...]]:
    if expression is None:
        return "", ()
    sql, params = _compile_filter_node(expression)
    return f" AND ({sql})", params


def _compile_filter_node(expression: MetadataFilter) -> tuple[str, tuple[object, ...]]:
    if isinstance(expression, Comparison):
        field = expression.field
        value = expression.value
        operator = expression.operator
        if operator is ComparisonOperator.IN:
            assert isinstance(value, tuple)
            compiled = [
                _compile_filter_node(Comparison(field, ComparisonOperator.EQ, item))
                for item in value
            ]
            return _join_filters("OR", compiled)
        encoded = _canonical_json(value)
        if operator is ComparisonOperator.EQ:
            if value is None:
                return (
                    "(NOT (metadata ? %s) OR metadata -> %s = 'null'::jsonb)",
                    (field, field),
                )
            return "COALESCE(metadata -> %s = %s::jsonb, FALSE)", (field, encoded)
        if operator is ComparisonOperator.NE:
            equal_sql, equal_params = _compile_filter_node(
                Comparison(field, ComparisonOperator.EQ, value)
            )
            return f"NOT ({equal_sql})", equal_params
        sql_operator = {
            ComparisonOperator.GT: ">",
            ComparisonOperator.GTE: ">=",
            ComparisonOperator.LT: "<",
            ComparisonOperator.LTE: "<=",
        }[operator]
        if isinstance(value, bool) or value is None:
            raise UnsupportedCapabilityError(
                "pgvector cannot represent ordered boolean/null metadata comparisons",
                capability="metadata_filter",
            )
        if isinstance(value, str):
            return (
                f"(jsonb_typeof(metadata -> %s) = 'string' AND "
                f'(metadata ->> %s) COLLATE "C" {sql_operator} %s)',
                (field, field, value),
            )
        if isinstance(value, (int, float)):
            return (
                f"(jsonb_typeof(metadata -> %s) = 'number' AND "
                f"(metadata ->> %s)::double precision {sql_operator} %s)",
                (field, field, float(value)),
            )
    if isinstance(expression, And):
        return _join_filters("AND", [_compile_filter_node(item) for item in expression.children])
    if isinstance(expression, Or):
        return _join_filters("OR", [_compile_filter_node(item) for item in expression.children])
    if isinstance(expression, Not):
        sql, params = _compile_filter_node(expression.child)
        return f"NOT ({sql})", params
    raise UnsupportedCapabilityError(
        "pgvector cannot represent metadata filter", capability="metadata_filter"
    )


def _join_filters(
    operator: str, values: Sequence[tuple[str, tuple[object, ...]]]
) -> tuple[str, tuple[object, ...]]:
    return (
        f" {operator} ".join(f"({sql})" for sql, _ in values),
        tuple(item for _, params in values for item in params),
    )


def _decode_manifest(value: object) -> IndexManifest:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        if not isinstance(decoded, Mapping):
            raise TypeError
        return IndexManifest.from_dict(decoded)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise IntegrityError("pgvector stored manifest is malformed") from None


def _decode_chunk(value: object, manifest: IndexManifest) -> Chunk:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        if not isinstance(decoded, Mapping):
            raise TypeError
        chunk = Chunk.from_dict(decoded)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise IntegrityError("pgvector stored chunk is malformed") from None
    if chunk.chunk_id != derive_chunk_id(chunk, manifest.chunker_fingerprint):
        raise IntegrityError("pgvector stored chunk has invalid stable identity")
    return chunk


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


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
