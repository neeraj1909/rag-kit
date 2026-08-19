"""Opt-in proof against a real, caller-provisioned Qdrant service."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.qdrant_store import QdrantVectorStore, _QdrantSdkBackend
from ragkit.adapters.retrieval import HashingEmbedder
from ragkit.domain import Comparison, ComparisonOperator, NormalizationMode
from ragkit.ports import DeleteRequest, EmbeddingRequest, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.integration


def test_qdrant_sdk_translation_with_local_in_memory_client(
    contract_corpus: ContractCorpus,
) -> None:
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("install rag-kit[qdrant] for the Qdrant SDK integration")
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunks = tuple(replace(item, metadata={"category": "keep"}) for item in contract_corpus.chunks)
    batch = embedder.embed_documents(EmbeddingRequest(tuple(item.text for item in chunks)))
    upsert = UpsertRequest(chunks, batch, manifest)
    search = VectorSearchRequest(
        embedder.embed_query("alpha"),
        embedder.fingerprint,
        10,
        Comparison("category", ComparisonOperator.EQ, "keep"),
        manifest,
    )
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(_QdrantSdkBackend(client), "ragkit-sdk-contract")

    results = assert_vector_store_contract(store, upsert, search)

    assert {item.chunk.chunk_id for item in results} == {item.chunk_id for item in chunks}
    store.require_compatible(manifest)


def test_qdrant_service_persists_reopens_filters_and_deletes(
    contract_corpus: ContractCorpus, socket_enabled: None
) -> None:
    if os.environ.get("RAGKIT_RUN_VECTOR_SERVICES") != "1":
        pytest.skip("set RAGKIT_RUN_VECTOR_SERVICES=1 for the Qdrant service test")
    url = os.environ.get("RAGKIT_QDRANT_URL")
    if not url:
        pytest.skip("set RAGKIT_QDRANT_URL for the Qdrant service test")
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pytest.skip("install rag-kit[qdrant] for the Qdrant service test")

    collection = f"ragkit_test_{uuid.uuid4().hex}"
    client = QdrantClient(url=url, timeout=10.0)
    try:
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
        batch = embedder.embed_documents(EmbeddingRequest(tuple(item.text for item in chunks)))
        upsert = UpsertRequest(chunks, batch, manifest)
        search = VectorSearchRequest(
            embedder.embed_query("alpha"),
            embedder.fingerprint,
            10,
            Comparison("category", ComparisonOperator.EQ, "keep"),
            manifest,
        )

        initial = QdrantVectorStore.from_url(url, collection)
        results = assert_vector_store_contract(initial, upsert, search)
        reopened = QdrantVectorStore.from_url(url, collection)
        reopened.require_compatible(manifest)
        assert reopened.search(search) == results
        assert all(item.chunk.metadata["category"] == "keep" for item in results)

        target = results[0].chunk.chunk_id
        reopened.delete(DeleteRequest((target,), manifest))
        reopened.delete(DeleteRequest((target,), manifest))
        assert target not in {item.chunk.chunk_id for item in reopened.search(search)}
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
