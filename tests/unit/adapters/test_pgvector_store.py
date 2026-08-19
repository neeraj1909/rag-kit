from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.pgvector_store import PgVectorStore
from ragkit.adapters.retrieval import HashingEmbedder
from ragkit.domain import (
    Comparison,
    ComparisonOperator,
    IndexCompatibilityError,
    NormalizationMode,
    ProviderError,
    UnsupportedCapabilityError,
)
from ragkit.ports import DeleteRequest, EmbeddingRequest, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection
        self.rows: list[tuple[object, ...]] = []

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.connection.calls.append(("execute", sql, params))
        if "SELECT manifest_json" in sql:
            self.rows = [] if self.connection.manifest is None else [(self.connection.manifest,)]
        elif "INSERT INTO ragkit_manifests" in sql:
            self.connection.manifest = params[1]
        elif "SELECT chunk_json" in sql:
            self.rows = [(payload, distance) for payload, distance in self.connection.search_rows]

    def executemany(self, sql: str, values: Iterable[tuple[object, ...]]) -> None:
        materialized = list(values)
        self.connection.calls.append(("executemany", sql, tuple(materialized)))

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self.rows else self.rows[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def close(self) -> None:
        pass


class _Connection:
    def __init__(self) -> None:
        self.manifest: object | None = None
        self.search_rows: list[tuple[object, float]] = []
        self.calls: list[tuple[str, str, object]] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.calls.append(("commit", "", ()))

    def rollback(self) -> None:
        self.calls.append(("rollback", "", ()))

    def close(self) -> None:
        pass


def _requests(contract_corpus: ContractCorpus) -> tuple[UpsertRequest, VectorSearchRequest]:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunks = tuple(
        replace(chunk, metadata={"team": "search", "priority": index})
        for index, chunk in enumerate(contract_corpus.chunks)
    )
    embeddings = embedder.embed_documents(EmbeddingRequest(tuple(chunk.text for chunk in chunks)))
    return (
        UpsertRequest(chunks, embeddings, manifest),
        VectorSearchRequest(
            embedder.embed_query("alpha"),
            embedder.fingerprint,
            3,
            Comparison("team", ComparisonOperator.EQ, "search"),
            manifest,
        ),
    )


def test_pgvector_upsert_preflights_manifest_and_uses_stable_identity(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome", physical_index_policy="exact")
    upsert, _ = _requests(contract_corpus)

    store.upsert(upsert)

    statements = [item[1] for item in connection.calls]
    assert statements.index(next(item for item in statements if "SELECT manifest_json" in item)) < (
        statements.index(next(item for item in statements if "INSERT INTO ragkit_entries" in item))
    )
    entry_call = next(item for item in connection.calls if item[0] == "executemany")
    rows = entry_call[2]
    assert isinstance(rows, Sequence)
    identifiers = [row[1] for row in rows if isinstance(row, Sequence)]
    assert identifiers == [str(chunk.chunk_id) for chunk in upsert.chunks]


def test_pgvector_fresh_compatibility_preflight_is_read_only(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, _ = _requests(contract_corpus)

    store.require_compatible(upsert.manifest)

    assert [item[0] for item in connection.calls] == ["execute"]


def test_pgvector_runs_the_reusable_vector_store_contract(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, search = _requests(contract_corpus)
    connection.search_rows = [
        (json.dumps(chunk.to_dict()), float(index) / 10.0)
        for index, chunk in enumerate(upsert.chunks)
    ]

    results = assert_vector_store_contract(store, upsert, search)

    assert len(results) == min(len(upsert.chunks), search.top_k)


def test_pgvector_search_round_trips_chunks_and_negates_cosine_distance(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, search = _requests(contract_corpus)
    connection.manifest = json.dumps(upsert.manifest.to_dict())
    connection.search_rows = [
        (json.dumps(upsert.chunks[1].to_dict()), 0.25),
        (json.dumps(upsert.chunks[0].to_dict()), 0.25),
    ]

    results = store.search(search)

    assert [item.chunk.chunk_id for item in results] == sorted(
        (upsert.chunks[0].chunk_id, upsert.chunks[1].chunk_id), key=str
    )
    assert all(item.score.raw_score == 0.25 and item.score.relevance == -0.25 for item in results)
    assert all(item.score.provenance.metric == "cosine" for item in results)
    query = next(item for item in connection.calls if "SELECT chunk_json" in item[1])
    assert "metadata ->" in query[1]
    assert "<=>" in query[1]


def test_pgvector_manifest_mismatch_stops_before_entry_mutation(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, _ = _requests(contract_corpus)
    connection.manifest = json.dumps(replace(upsert.manifest, schema_version=2).to_dict())

    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.upsert(upsert)

    assert not any(item[0] == "executemany" for item in connection.calls)


def test_pgvector_ann_policy_uses_dimensioned_expression_index_query(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome", physical_index_policy="hnsw")
    upsert, search = _requests(contract_corpus)
    connection.manifest = json.dumps(upsert.manifest.to_dict())

    store.search(search)

    query = next(item[1] for item in connection.calls if "SELECT chunk_json" in item[1])
    assert "embedding::vector(32) <=> %s::vector(32)" in query


def test_pgvector_rejects_filter_it_cannot_represent_before_connection(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, search = _requests(contract_corpus)

    with pytest.raises(UnsupportedCapabilityError, match="ordered boolean"):
        store.search(
            replace(
                search,
                filters=Comparison("flag", ComparisonOperator.GT, True),
                expected_manifest=upsert.manifest,
            )
        )

    assert connection.calls == []


def test_pgvector_delete_is_idempotent_and_manifest_guarded(
    contract_corpus: ContractCorpus,
) -> None:
    connection = _Connection()
    store = PgVectorStore(lambda: connection, "takehome")
    upsert, _ = _requests(contract_corpus)
    connection.manifest = json.dumps(upsert.manifest.to_dict())
    request = DeleteRequest((upsert.chunks[0].chunk_id,), upsert.manifest)

    store.delete(request)
    store.delete(request)

    deletes = [item for item in connection.calls if "DELETE FROM ragkit_entries" in item[1]]
    assert len(deletes) == 2


def test_pgvector_translates_connection_failure_without_disclosing_details(
    contract_corpus: ContractCorpus,
) -> None:
    def fail() -> _Connection:
        raise RuntimeError("postgresql://user:secret@example.invalid")

    store = PgVectorStore(fail, "takehome")
    upsert, _ = _requests(contract_corpus)

    with pytest.raises(ProviderError, match="compatibility check") as captured:
        store.require_compatible(upsert.manifest)

    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None
