from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_retriever_contract
from ragkit.adapters import (
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
)
from ragkit.domain import NormalizationMode
from ragkit.ports import EmbeddingRequest, RetrievalRequest, SparseUpsertRequest, UpsertRequest

pytestmark = pytest.mark.contract


def test_dense_sparse_and_hybrid_share_the_retriever_contract(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(dimension=16)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
        normalization=NormalizationMode.L2,
    )
    sparse = BM25Retriever()
    sparse.upsert(SparseUpsertRequest(contract_corpus.chunks, manifest))
    vector_store = InMemoryVectorStore()
    vector_store.upsert(
        UpsertRequest(
            contract_corpus.chunks,
            embedder.embed_documents(
                EmbeddingRequest(tuple(chunk.text for chunk in contract_corpus.chunks))
            ),
            manifest,
        )
    )
    dense = DenseRetriever(embedder, vector_store)
    hybrid = HybridRetriever((("sparse", sparse), ("dense", dense)))
    request = RetrievalRequest("invoice", 4, manifest)

    sparse_results = assert_retriever_contract(sparse, request)
    dense_results = assert_retriever_contract(dense, request)
    hybrid_results = assert_retriever_contract(hybrid, request)

    assert sparse_results
    assert dense_results
    assert hybrid_results
    assert all(result.prior_scores for result in hybrid_results)
