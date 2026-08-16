from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.retrieval import HashingEmbedder
from ragkit.adapters.sqlite_store import SQLiteVectorStore
from ragkit.domain import Comparison, ComparisonOperator, IndexCompatibilityError, NormalizationMode
from ragkit.ports import DeleteRequest, EmbeddingRequest, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.integration


def _requests(contract_corpus: ContractCorpus) -> tuple[UpsertRequest, VectorSearchRequest]:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunks = tuple(
        replace(chunk, metadata={"category": "keep" if index % 2 == 0 else "drop"})
        for index, chunk in enumerate(contract_corpus.chunks)
    )
    embeddings = embedder.embed_documents(EmbeddingRequest(tuple(chunk.text for chunk in chunks)))
    return (
        UpsertRequest(chunks, embeddings, manifest),
        VectorSearchRequest(
            embedder.embed_query("alpha"),
            embedder.fingerprint,
            10,
            Comparison("category", ComparisonOperator.EQ, "keep"),
            manifest,
        ),
    )


def test_sqlite_persists_reopens_filters_and_is_idempotent(
    tmp_path: Path, contract_corpus: ContractCorpus
) -> None:
    upsert, search = _requests(contract_corpus)
    database = tmp_path / "index.sqlite3"

    initial = SQLiteVectorStore(database, "ragkit-maintenance")
    results = assert_vector_store_contract(initial, upsert, search)
    reopened = SQLiteVectorStore(database, "ragkit-maintenance")
    reopened.upsert(upsert)
    reopened_results = reopened.search(search)

    assert reopened_results == results
    expected_by_id = {chunk.chunk_id: chunk for chunk in upsert.chunks}
    assert all(item.chunk == expected_by_id[item.chunk.chunk_id] for item in results)
    assert all(item.chunk.metadata["category"] == "keep" for item in results)
    assert all(item.score.raw_score == item.score.relevance for item in results)

    reopened.delete(DeleteRequest((results[0].chunk.chunk_id,), upsert.manifest))
    reopened.delete(DeleteRequest((results[0].chunk.chunk_id,), upsert.manifest))
    assert results[0].chunk.chunk_id not in {
        item.chunk.chunk_id for item in reopened.search(search)
    }

    payload = tmp_path / "reopen.json"
    payload.write_text(
        json.dumps(
            {
                "path": str(database),
                "manifest": upsert.manifest.to_dict(),
                "query": list(search.embedding.values),
            }
        ),
        encoding="utf-8",
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from ragkit.adapters import SQLiteVectorStore
from ragkit.domain import Embedding, IndexManifest
from ragkit.ports import VectorSearchRequest
p=json.load(open(sys.argv[1], encoding='utf-8'))
m=IndexManifest.from_dict(p['manifest'])
q=Embedding(tuple(p['query']), m.embedding_dimension, True)
r=SQLiteVectorStore(p['path'], 'ragkit-maintenance').search(
    VectorSearchRequest(q, m.embedder_fingerprint, 10, None, m))
print(len(r))
""",
            str(payload),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert int(probe.stdout.strip()) == len(upsert.chunks) - 1


def test_sqlite_rejects_manifest_mismatch_before_query_or_mutation(
    tmp_path: Path, contract_corpus: ContractCorpus
) -> None:
    upsert, search = _requests(contract_corpus)
    store = SQLiteVectorStore(tmp_path / "index.sqlite3", "ragkit-mismatch")
    store.upsert(upsert)
    incompatible = replace(upsert.manifest, schema_version=2)

    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.search(replace(search, expected_manifest=incompatible))
    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.delete(DeleteRequest((upsert.chunks[0].chunk_id,), incompatible))

    assert store.search(search)


def test_sqlite_rejects_malformed_stored_rows_instead_of_failing_open(
    tmp_path: Path, contract_corpus: ContractCorpus
) -> None:
    import sqlite3

    from ragkit.domain import IntegrityError

    upsert, search = _requests(contract_corpus)
    database = tmp_path / "index.sqlite3"
    store = SQLiteVectorStore(database, "ragkit-integrity")
    store.upsert(upsert)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ragkit_entries SET chunk_json = ? WHERE collection_name = ?",
            ("not-json", "ragkit-integrity"),
        )

    with pytest.raises(IntegrityError, match="stored row"):
        store.search(search)
