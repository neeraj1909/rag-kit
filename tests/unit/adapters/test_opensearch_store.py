from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import ContractCorpus
from contract_assertions import assert_vector_store_contract
from ragkit.adapters.opensearch_store import OpenSearchVectorStore
from ragkit.domain import (
    Comparison,
    ComparisonOperator,
    Embedding,
    IndexCompatibilityError,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    NormalizationMode,
    UnsupportedCapabilityError,
)
from ragkit.ports import DeleteRequest, EmbeddingBatch, UpsertRequest, VectorSearchRequest

pytestmark = pytest.mark.unit


class _Indices:
    def __init__(self, owner: _Client) -> None:
        self.owner = owner

    def exists(self, **kwargs: object) -> bool:
        self.owner.calls.append(("exists", kwargs))
        return self.owner.mapping is not None

    def create(self, **kwargs: object) -> object:
        self.owner.calls.append(("create", kwargs))
        body = kwargs["body"]
        assert isinstance(body, dict)
        self.owner.mapping = body["mappings"]
        return {"acknowledged": True}

    def get_mapping(self, **kwargs: object) -> object:
        self.owner.calls.append(("get_mapping", kwargs))
        return {self.owner.index_name: {"mappings": self.owner.mapping}}


class _Client:
    def __init__(self, index_name: str = "assignment") -> None:
        self.index_name = index_name
        self.mapping: object | None = None
        self.calls: list[tuple[str, object]] = []
        self.documents: dict[str, dict[str, object]] = {}
        self.indices = _Indices(self)

    def bulk(self, **kwargs: object) -> object:
        self.calls.append(("bulk", kwargs))
        body = kwargs["body"]
        assert isinstance(body, list)
        items: list[dict[str, object]] = []
        for offset in range(0, len(body), 2):
            action = body[offset]
            assert isinstance(action, dict)
            if "index" in action:
                meta = action["index"]
                assert isinstance(meta, dict)
                identifier = str(meta["_id"])
                source = body[offset + 1]
                assert isinstance(source, dict)
                self.documents[identifier] = source
                items.append({"index": {"status": 200}})
            else:
                meta = action["delete"]
                assert isinstance(meta, dict)
                self.documents.pop(str(meta["_id"]), None)
                items.append({"delete": {"status": 200}})
        return {"errors": False, "items": items}

    def search(self, **kwargs: object) -> object:
        self.calls.append(("search", kwargs))
        hits = [
            {"_id": identifier, "_score": 0.875, "_source": source}
            for identifier, source in reversed(tuple(self.documents.items()))
        ]
        return {"hits": {"hits": hits}}


def _requests(corpus: ContractCorpus) -> tuple[UpsertRequest, VectorSearchRequest]:
    manifest = replace(corpus.manifest, normalization=NormalizationMode.L2)
    embeddings = tuple(Embedding((1.0, 0.0, 0.0), 3, True) for _ in corpus.chunks)
    upsert = UpsertRequest(
        corpus.chunks, EmbeddingBatch(embeddings, manifest.embedder_fingerprint), manifest
    )
    search = VectorSearchRequest(
        Embedding((1.0, 0.0, 0.0), 3, True),
        manifest.embedder_fingerprint,
        10,
        Comparison("category", ComparisonOperator.NE, None),
        manifest,
    )
    return upsert, search


def test_opensearch_creates_manifest_mapping_then_round_trips_and_deletes(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )

    store.upsert(upsert)
    results = store.search(search)
    store.delete(DeleteRequest((upsert.chunks[0].chunk_id,), upsert.manifest))

    assert [name for name, _ in client.calls[:3]] == ["exists", "create", "get_mapping"]
    assert len(results) == len(upsert.chunks)
    assert [str(item.chunk.chunk_id) for item in results] == sorted(
        str(chunk.chunk_id) for chunk in upsert.chunks
    )
    assert results[0].score.provenance.metric == "opensearch_lucene_cosinesimil_score"
    search_body = next(payload for name, payload in client.calls if name == "search")
    assert isinstance(search_body, dict)
    assert search_body["index"] == "assignment"
    body = search_body["body"]
    assert isinstance(body, dict)
    assert "filter" in body["query"]["knn"]["embedding"]


def test_opensearch_runs_the_reusable_vector_store_contract(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, search = _requests(contract_corpus)
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=_Client()
    )
    results = assert_vector_store_contract(store, upsert, search)
    assert len(results) == len(upsert.chunks)


def test_opensearch_rejects_incompatible_mapping_before_bulk(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, _ = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )
    store.upsert(upsert)
    assert isinstance(client.mapping, dict)
    client.mapping["_meta"]["ragkit"]["manifest"]["schema_version"] = 2
    client.calls.clear()

    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.upsert(upsert)
    assert [name for name, _ in client.calls] == ["exists", "get_mapping"]


def test_opensearch_absent_index_is_compatible_without_being_created(
    contract_corpus: ContractCorpus,
) -> None:
    upsert, _ = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )

    store.require_compatible(upsert.manifest)

    assert [name for name, _ in client.calls] == ["exists"]
    assert client.mapping is None


def test_opensearch_rejects_malformed_hit_identity(contract_corpus: ContractCorpus) -> None:
    upsert, search = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )
    store.upsert(upsert)
    first = next(iter(client.documents.values()))
    first["chunk"] = {}
    with pytest.raises(IntegrityError, match="chunk"):
        store.search(search)


def test_opensearch_rejects_unrepresentable_filter_before_manifest_check(
    contract_corpus: ContractCorpus,
) -> None:
    _, search = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )
    unsupported = replace(
        search,
        filters=Comparison("mixed", ComparisonOperator.IN, (1, "one")),
    )

    with pytest.raises(UnsupportedCapabilityError, match="one non-empty scalar type"):
        store.search(unsupported)

    assert client.calls == []
    assert not hasattr(store, "_username")
    assert not hasattr(store, "_password")


def test_opensearch_rejects_ordered_boolean_before_manifest_check(
    contract_corpus: ContractCorpus,
) -> None:
    _, search = _requests(contract_corpus)
    client = _Client()
    store = OpenSearchVectorStore(
        url="http://localhost:9200", index_name="assignment", client=client
    )
    with pytest.raises(UnsupportedCapabilityError):
        store.search(replace(search, filters=Comparison("flag", ComparisonOperator.GT, True)))
    assert client.calls == []


def test_opensearch_rejects_boolean_timeout_and_excess_top_k(
    contract_corpus: ContractCorpus,
) -> None:
    _, search = _requests(contract_corpus)
    client = _Client()
    with pytest.raises(InvalidDomainValueError, match="timeout"):
        OpenSearchVectorStore(
            url="http://localhost:9200",
            index_name="assignment",
            timeout_seconds=True,
            client=client,
        )
    store = OpenSearchVectorStore(
        url="http://localhost:9200",
        index_name="assignment",
        max_top_k=1,
        client=client,
    )
    with pytest.raises(LimitExceededError, match="top_k"):
        store.search(search)
