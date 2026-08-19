"""Standard-library persistent SQLite vector-store adapter."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from ragkit.adapters.retrieval import _matches
from ragkit.domain import (
    Chunk,
    ComponentFingerprint,
    Embedding,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    NormalizationMode,
    ProviderError,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    derive_chunk_id,
)
from ragkit.ports import DeleteRequest, UpsertRequest, VectorSearchRequest, VectorStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ragkit_manifests (
    collection_name TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ragkit_entries (
    collection_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    chunk_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    PRIMARY KEY (collection_name, chunk_id),
    FOREIGN KEY (collection_name)
        REFERENCES ragkit_manifests(collection_name)
        ON DELETE CASCADE
);
"""


class SQLiteVectorStore(VectorStore):
    """Persist manifest-bound cosine vectors in one transactional SQLite file."""

    def __init__(self, path: Path | str, collection_name: str) -> None:
        if not collection_name.strip() or len(collection_name) > 128:
            raise InvalidDomainValueError(
                "SQLite collection name must be nonblank and at most 128 characters"
            )
        self._path = Path(path)
        self._name = collection_name
        self._fingerprint = ComponentFingerprint.create(
            "vector_store",
            "sqlite_cosine",
            {"version": 1, "schema_version": 1, "metric": "cosine"},
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def require_compatible(self, manifest: IndexManifest) -> None:
        self._validate_manifest(manifest)
        if not self._path.is_file():
            return
        connection = self._connect(create=False, expected=manifest)
        try:
            try:
                actual = self._read_manifest(connection)
            except sqlite3.OperationalError as error:
                if "no such table" not in str(error).casefold():
                    raise
                actual = None
            if actual is not None:
                manifest.require_compatible(actual)
        except IndexCompatibilityError:
            raise
        except sqlite3.Error as error:
            raise ProviderError("SQLite compatibility check failed", cause=error) from error
        finally:
            connection.close()

    def upsert(self, request: UpsertRequest) -> None:
        self._validate_manifest(request.manifest)
        if any(
            chunk.chunk_id != derive_chunk_id(chunk, request.manifest.chunker_fingerprint)
            for chunk in request.chunks
        ):
            raise IntegrityError("SQLite upsert chunk ID does not match stable content")
        for embedding in request.embeddings.embeddings:
            _require_unit_length(embedding)

        connection = self._connect(create=True)
        try:
            connection.executescript(_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            actual = self._read_manifest(connection)
            if actual is None:
                connection.execute(
                    "INSERT INTO ragkit_manifests(collection_name, manifest_json) VALUES (?, ?)",
                    (self._name, _canonical_json(request.manifest.to_dict())),
                )
            else:
                request.manifest.require_compatible(actual)
            connection.executemany(
                """
                INSERT INTO ragkit_entries(
                    collection_name, chunk_id, chunk_json, embedding_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(collection_name, chunk_id) DO UPDATE SET
                    chunk_json = excluded.chunk_json,
                    embedding_json = excluded.embedding_json
                """,
                (
                    (
                        self._name,
                        str(chunk.chunk_id),
                        _canonical_json(chunk.to_dict()),
                        _canonical_json(list(embedding.values)),
                    )
                    for chunk, embedding in zip(
                        request.chunks, request.embeddings.embeddings, strict=True
                    )
                ),
            )
            connection.commit()
        except (IndexCompatibilityError, IntegrityError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise ProviderError("SQLite upsert failed", cause=error) from error
        finally:
            connection.close()

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        self._validate_manifest(request.expected_manifest)
        _require_unit_length(request.embedding)
        connection = self._connect(create=False, expected=request.expected_manifest)
        try:
            actual = self._require_manifest(connection, request.expected_manifest)
            request.expected_manifest.require_compatible(actual)
            rows = connection.execute(
                """
                SELECT chunk_id, chunk_json, embedding_json
                FROM ragkit_entries
                WHERE collection_name = ?
                ORDER BY chunk_id
                """,
                (self._name,),
            ).fetchall()
        except (IndexCompatibilityError, IntegrityError):
            raise
        except sqlite3.Error as error:
            raise ProviderError("SQLite query failed", cause=error) from error
        finally:
            connection.close()

        provenance = ScoreProvenance(
            self._fingerprint,
            "dense_retrieval",
            ScoreKind.SIMILARITY,
            "cosine",
            "identity:v1",
        )
        scored: list[tuple[Chunk, RetrievalScore]] = []
        seen: set[str] = set()
        for row in rows:
            chunk, embedding = _decode_row(row, request.expected_manifest.embedding_dimension)
            identifier = str(chunk.chunk_id)
            if identifier in seen:
                raise IntegrityError("SQLite stored row has duplicate chunk identity")
            seen.add(identifier)
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
        self._validate_manifest(request.expected_manifest)
        connection = self._connect(create=False, expected=request.expected_manifest)
        try:
            connection.execute("BEGIN IMMEDIATE")
            actual = self._require_manifest(connection, request.expected_manifest)
            request.expected_manifest.require_compatible(actual)
            connection.executemany(
                "DELETE FROM ragkit_entries WHERE collection_name = ? AND chunk_id = ?",
                ((self._name, str(identifier)) for identifier in request.chunk_ids),
            )
            connection.commit()
        except (IndexCompatibilityError, IntegrityError):
            connection.rollback()
            raise
        except sqlite3.Error as error:
            connection.rollback()
            raise ProviderError("SQLite delete failed", cause=error) from error
        finally:
            connection.close()

    def _connect(
        self, *, create: bool, expected: IndexManifest | None = None
    ) -> sqlite3.Connection:
        if not create and not self._path.is_file():
            if expected is None:
                raise IntegrityError("SQLite expected manifest is unavailable")
            raise IndexCompatibilityError({"manifest": (expected.fingerprint, None)})
        if create:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self._path, timeout=30.0, isolation_level=None)
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise ProviderError("SQLite initialization failed", cause=error) from error

    def _read_manifest(self, connection: sqlite3.Connection) -> IndexManifest | None:
        row = connection.execute(
            "SELECT manifest_json FROM ragkit_manifests WHERE collection_name = ?",
            (self._name,),
        ).fetchone()
        return None if row is None else _decode_manifest(row[0])

    def _require_manifest(
        self, connection: sqlite3.Connection, expected: IndexManifest
    ) -> IndexManifest:
        try:
            actual = self._read_manifest(connection)
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).casefold():
                raise
            actual = None
        if actual is None:
            raise IndexCompatibilityError({"manifest": (expected.fingerprint, None)})
        return actual

    @staticmethod
    def _validate_manifest(manifest: IndexManifest) -> None:
        if manifest.normalization is not NormalizationMode.L2:
            raise IndexCompatibilityError(
                {"normalization": (NormalizationMode.L2, manifest.normalization)}
            )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_manifest(encoded: object) -> IndexManifest:
    if not isinstance(encoded, str):
        raise IntegrityError("SQLite stored manifest is malformed")
    try:
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise TypeError
        return IndexManifest.from_dict(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("SQLite stored manifest is malformed", cause=error) from error


def _decode_row(row: object, dimension: int) -> tuple[Chunk, Embedding]:
    if not isinstance(row, tuple) or len(row) != 3:
        raise IntegrityError("SQLite stored row is malformed")
    identifier, chunk_encoded, embedding_encoded = row
    if not all(isinstance(value, str) for value in row):
        raise IntegrityError("SQLite stored row is malformed")
    try:
        chunk_value = json.loads(chunk_encoded)
        embedding_value = json.loads(embedding_encoded)
        if not isinstance(chunk_value, dict) or not isinstance(embedding_value, list):
            raise TypeError
        chunk = Chunk.from_dict(chunk_value)
        values = tuple(float(item) for item in embedding_value)
        embedding = Embedding(values, dimension, True)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("SQLite stored row is malformed", cause=error) from error
    if identifier != str(chunk.chunk_id):
        raise IntegrityError("SQLite stored row has invalid chunk identity")
    _require_unit_length(embedding)
    return chunk, embedding


def _require_unit_length(embedding: Embedding) -> None:
    norm = math.sqrt(sum(value * value for value in embedding.values))
    if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise IntegrityError("L2-normalized embeddings must have unit length")
