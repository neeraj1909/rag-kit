from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.chroma_store import ChromaVectorStore
from ragkit.adapters.retrieval import HashingEmbedder
from ragkit.domain import Comparison, ComparisonOperator, IndexCompatibilityError, NormalizationMode
from ragkit.ports import DeleteRequest, EmbeddingRequest, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.integration


def test_chroma_persists_reopens_filters_and_is_idempotent(
    tmp_path: Path, contract_corpus: ContractCorpus
) -> None:
    pytest.importorskip("chromadb")
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
    request = UpsertRequest(chunks, embeddings, manifest)

    initial = ChromaVectorStore(tmp_path, "ragkit-phase3")
    search = VectorSearchRequest(
        embedder.embed_query("alpha"),
        embedder.fingerprint,
        10,
        Comparison("category", ComparisonOperator.EQ, "keep"),
        manifest,
    )
    assert_vector_store_contract(initial, request, search)
    reopened = ChromaVectorStore(tmp_path, "ragkit-phase3")
    reopened.upsert(request)
    results = reopened.search(search)

    assert results
    expected_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    assert all(result.chunk == expected_by_id[result.chunk.chunk_id] for result in results)
    assert all(result.chunk.metadata["category"] == "keep" for result in results)
    for result in results:
        assert result.score.raw_score is not None
        assert result.score.relevance == -result.score.raw_score
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    reopened.delete(DeleteRequest((results[0].chunk.chunk_id,), manifest))
    reopened.delete(DeleteRequest((results[0].chunk.chunk_id,), manifest))
    remaining = reopened.search(search)
    assert results[0].chunk.chunk_id not in {item.chunk.chunk_id for item in remaining}

    payload = tmp_path / "reopen.json"
    payload.write_text(
        json.dumps(
            {
                "path": str(tmp_path),
                "manifest": manifest.to_dict(),
                "query": list(embedder.embed_query("alpha").values),
            }
        )
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from ragkit.adapters import ChromaVectorStore
from ragkit.domain import Embedding, IndexManifest
from ragkit.ports import VectorSearchRequest
p=json.load(open(sys.argv[1], encoding='utf-8'))
m=IndexManifest.from_dict(p['manifest'])
q=Embedding(tuple(p['query']), m.embedding_dimension, True)
r=ChromaVectorStore(p['path'], 'ragkit-phase3').search(
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
    assert int(probe.stdout.strip()) == len(chunks) - 1


def test_chroma_rejects_manifest_mismatch_before_query_or_mutation(
    tmp_path: Path, contract_corpus: ContractCorpus
) -> None:
    pytest.importorskip("chromadb")
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunk = contract_corpus.chunks[0]
    batch = embedder.embed_documents(EmbeddingRequest((chunk.text,)))
    store = ChromaVectorStore(tmp_path, "ragkit-mismatch")
    store.upsert(UpsertRequest((chunk,), batch, manifest))
    incompatible = replace(manifest, schema_version=2)

    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 1, None, incompatible
            )
        )
    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.delete(DeleteRequest((chunk.chunk_id,), incompatible))
