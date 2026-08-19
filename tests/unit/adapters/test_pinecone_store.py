from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.pinecone_store import PineconeVectorStore
from ragkit.domain import (
    And,
    Comparison,
    ComparisonOperator,
    Embedding,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    NormalizationMode,
    Or,
    UnsupportedCapabilityError,
    canonical_json,
)
from ragkit.ports import DeleteRequest, EmbeddingBatch, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.unit


class _Index:
    def __init__(self, manifest: IndexManifest | None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.records: dict[str, dict[str, object]] = {}
        if manifest is not None:
            self.records[PineconeVectorStore.MANIFEST_ID] = {
                "id": PineconeVectorStore.MANIFEST_ID,
                "metadata": {
                    "_rk_kind": "manifest",
                    "_rk_schema": "pinecone-vector-store-v1",
                    "_rk_manifest": canonical_json(manifest.to_dict()),
                },
            }

    def fetch(self, **kwargs: object) -> object:
        self.calls.append(("fetch", kwargs))
        ids = kwargs["ids"]
        assert isinstance(ids, list)
        return SimpleNamespace(
            vectors={key: self.records[key] for key in ids if key in self.records}
        )

    def upsert(self, **kwargs: object) -> object:
        self.calls.append(("upsert", kwargs))
        vectors = kwargs["vectors"]
        assert isinstance(vectors, list)
        for vector in vectors:
            assert isinstance(vector, dict)
            self.records[str(vector["id"])] = vector
        return SimpleNamespace(upserted_count=len(vectors))

    def query(self, **kwargs: object) -> object:
        self.calls.append(("query", kwargs))
        matches = [
            SimpleNamespace(id=identifier, score=0.75, metadata=record["metadata"])
            for identifier, record in reversed(tuple(self.records.items()))
            if identifier != PineconeVectorStore.MANIFEST_ID
        ]
        return SimpleNamespace(matches=matches)

    def delete(self, **kwargs: object) -> object:
        self.calls.append(("delete", kwargs))
        ids = kwargs["ids"]
        assert isinstance(ids, list)
        for identifier in ids:
            self.records.pop(str(identifier), None)
        return SimpleNamespace()


def _requests(corpus: ContractCorpus) -> tuple[UpsertRequest, VectorSearchRequest]:
    manifest = replace(corpus.manifest, normalization=NormalizationMode.L2)
    embeddings = tuple(Embedding((1.0, 0.0, 0.0), 3, True) for _ in corpus.chunks)
    upsert = UpsertRequest(
        corpus.chunks, EmbeddingBatch(embeddings, manifest.embedder_fingerprint), manifest
    )
    expression = And(
        (
            Comparison("category", ComparisonOperator.EQ, "keep"),
            Or(
                (
                    Comparison("priority", ComparisonOperator.GTE, 2),
                    Comparison("optional", ComparisonOperator.EQ, None),
                )
            ),
        )
    )
    search = VectorSearchRequest(
        Embedding((1.0, 0.0, 0.0), 3, True),
        manifest.embedder_fingerprint,
        10,
        expression,
        manifest,
    )
    return upsert, search


def test_pinecone_is_manifest_first_idempotent_filterable_and_deletable(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    index = _Index(upsert.manifest)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=index,
    )

    store.upsert(upsert)
    store.upsert(upsert)
    results = store.search(search)
    store.delete(DeleteRequest((upsert.chunks[0].chunk_id,), upsert.manifest))
    store.delete(DeleteRequest((upsert.chunks[0].chunk_id,), upsert.manifest))

    assert [name for name, _ in index.calls[:2]] == ["fetch", "upsert"]
    assert len(results) == len(upsert.chunks)
    assert [str(item.chunk.chunk_id) for item in results] == sorted(
        str(chunk.chunk_id) for chunk in upsert.chunks
    )
    assert all(item.score.raw_score == 0.75 for item in results)
    assert results[0].score.provenance.metric == "cosine"
    query = next(payload for name, payload in index.calls if name == "query")
    assert isinstance(query, dict)
    assert query["namespace"] == "assignment"
    assert query["include_metadata"] is True
    assert query["filter"] and "$and" in query["filter"]


def test_pinecone_runs_the_reusable_vector_store_contract(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=_Index(upsert.manifest),
    )
    results = assert_vector_store_contract(store, upsert, search)
    assert len(results) == len(upsert.chunks)


def test_pinecone_rejects_missing_or_mismatched_manifest_before_data_work(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, _ = _requests(contract_corpus)
    missing = _Index(None)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=missing,
    )
    with pytest.raises(IndexCompatibilityError):
        store.upsert(upsert)
    assert [name for name, _ in missing.calls] == ["fetch"]

    incompatible = _Index(replace(upsert.manifest, schema_version=2))
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=incompatible,
    )
    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.upsert(upsert)
    assert [name for name, _ in incompatible.calls] == ["fetch"]


def test_pinecone_rejects_malformed_hit_and_metadata_excess(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, _search = _requests(contract_corpus)
    index = _Index(upsert.manifest)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        max_metadata_bytes=50,
        index=index,
    )
    with pytest.raises(IntegrityError, match="metadata"):
        store.upsert(upsert)
    assert index.calls == []


@pytest.mark.parametrize(
    "comparison",
    (
        Comparison("nullable", ComparisonOperator.IN, (None,)),
        Comparison("label", ComparisonOperator.GT, "alpha"),
        Comparison("flag", ComparisonOperator.LTE, True),
    ),
)
def test_pinecone_rejects_inexact_ordered_or_null_filters(
    contract_corpus: ContractCorpus, comparison: Comparison
) -> None:
    upsert, search = _requests(contract_corpus)
    index = _Index(upsert.manifest)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=index,
    )
    with pytest.raises(UnsupportedCapabilityError):
        store.search(replace(search, filters=comparison))
    assert index.calls == []


def test_pinecone_rejects_boolean_timeout_and_excess_top_k(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    index = _Index(upsert.manifest)
    with pytest.raises(InvalidDomainValueError, match="timeout"):
        PineconeVectorStore(
            index_host="example.svc.pinecone.io",
            namespace="assignment",
            api_key="secret",
            timeout_seconds=True,
            index=index,
        )
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        max_top_k=1,
        index=index,
    )
    with pytest.raises(LimitExceededError, match="top_k"):
        store.search(search)

    sound = PineconeVectorStore(
        index_host="example.svc.pinecone.io", namespace="assignment", api_key="secret", index=index
    )
    sound.upsert(upsert)
    record = index.records[str(upsert.chunks[0].chunk_id)]
    assert isinstance(record["metadata"], dict)
    record["metadata"]["_rk_chunk"] = "{}"
    with pytest.raises(IntegrityError, match="chunk"):
        sound.search(search)


def test_pinecone_rejects_unrepresentable_filter_before_manifest_fetch(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    index = _Index(upsert.manifest)
    store = PineconeVectorStore(
        index_host="example.svc.pinecone.io",
        namespace="assignment",
        api_key="secret",
        index=index,
    )
    unsupported = replace(
        search,
        filters=Comparison("mixed", ComparisonOperator.IN, (1, "one")),
    )

    with pytest.raises(UnsupportedCapabilityError, match="one non-empty scalar type"):
        store.search(unsupported)

    assert index.calls == []
    assert not hasattr(store, "_api_key")
