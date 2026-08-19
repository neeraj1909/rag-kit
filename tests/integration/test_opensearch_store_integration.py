"""Opt-in disposable-index proof against an actual OpenSearch service."""

from __future__ import annotations

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from conftest import ContractCorpus
from ragkit.adapters.opensearch_store import OpenSearchVectorStore
from ragkit.domain import Embedding, NormalizationMode
from ragkit.ports import DeleteRequest, EmbeddingBatch, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.integration


def test_opensearch_service_round_trip_and_idempotent_delete(
    contract_corpus: ContractCorpus,
) -> None:
    if os.environ.get("RAGKIT_RUN_OPENSEARCH_SERVICE") != "1":
        pytest.skip("set RAGKIT_RUN_OPENSEARCH_SERVICE=1 for the disposable-index service proof")
    module = pytest.importorskip("opensearchpy")
    url = os.environ.get("RAGKIT_OPENSEARCH_URL", "http://localhost:9200")
    index_name = f"ragkit-live-{uuid4().hex}"
    client = module.OpenSearch(hosts=[url], timeout=30, max_retries=0)
    store = OpenSearchVectorStore(url=url, index_name=index_name, client=client, max_retries=0)
    manifest = replace(contract_corpus.manifest, normalization=NormalizationMode.L2)
    vectors = tuple(Embedding((1.0, 0.0, 0.0), 3, True) for _ in contract_corpus.chunks)
    upsert = UpsertRequest(
        contract_corpus.chunks,
        EmbeddingBatch(vectors, manifest.embedder_fingerprint),
        manifest,
    )
    search = VectorSearchRequest(
        Embedding((1.0, 0.0, 0.0), 3, True),
        manifest.embedder_fingerprint,
        len(contract_corpus.chunks),
        None,
        manifest,
    )
    try:
        store.upsert(upsert)
        reopened = OpenSearchVectorStore(url=url, index_name=index_name, client=client)
        results = reopened.search(search)
        assert {item.chunk.chunk_id for item in results} == {
            item.chunk_id for item in contract_corpus.chunks
        }
        request = DeleteRequest(tuple(item.chunk_id for item in contract_corpus.chunks), manifest)
        reopened.delete(request)
        reopened.delete(request)
        assert reopened.search(search) == ()
    finally:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
