from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from conftest import ContractCorpus
from ragkit.adapters.qdrant_store import (
    QdrantCollection,
    QdrantMatch,
    QdrantPoint,
    QdrantVectorStore,
)
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


class _Backend:
    def __init__(self) -> None:
        self.collection: QdrantCollection | None = None
        self.points: dict[str, QdrantPoint] = {}
        self.matches: tuple[QdrantMatch, ...] = ()
        self.calls: list[str] = []

    def describe(self, collection_name: str) -> QdrantCollection | None:
        self.calls.append("describe")
        return self.collection

    def create(self, collection_name: str, vector_name: str, dimension: int) -> None:
        self.calls.append("create")
        self.collection = QdrantCollection(vector_name, dimension, "cosine")

    def retrieve(self, collection_name: str, point_ids: tuple[str, ...]) -> tuple[QdrantPoint, ...]:
        self.calls.append("retrieve")
        return tuple(self.points[item] for item in point_ids if item in self.points)

    def upsert(self, collection_name: str, points: tuple[QdrantPoint, ...]) -> None:
        self.calls.append("upsert")
        self.points.update((point.point_id, point) for point in points)

    def query(
        self,
        collection_name: str,
        vector_name: str,
        vector: tuple[float, ...],
        limit: int,
        query_filter: dict[str, object] | None,
    ) -> tuple[QdrantMatch, ...]:
        self.calls.append("query")
        return self.matches[:limit]

    def delete(self, collection_name: str, point_ids: tuple[str, ...]) -> None:
        self.calls.append("delete")
        for item in point_ids:
            self.points.pop(item, None)


def _record_kind(point: QdrantPoint) -> object:
    payload = point.payload.get("ragkit")
    return payload.get("record_kind") if isinstance(payload, dict) else None


def _requests(contract_corpus: ContractCorpus) -> tuple[UpsertRequest, VectorSearchRequest]:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunks = tuple(replace(chunk, metadata={"team": "search"}) for chunk in contract_corpus.chunks)
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


def test_qdrant_initialization_failure_is_typed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenClient:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("provider included secret-token")

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=BrokenClient))
    with pytest.raises(ProviderError, match="Qdrant initialization failed") as error:
        QdrantVectorStore.from_url("http://localhost:6333", "takehome", api_key="secret-token")

    assert "secret-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_qdrant_first_upsert_creates_manifest_sentinel_before_chunks(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    upsert, _ = _requests(contract_corpus)

    store.upsert(upsert)

    assert backend.calls[:3] == ["describe", "create", "upsert"]
    assert len(backend.points) == len(upsert.chunks) + 1
    assert any(_record_kind(point) == "manifest" for point in backend.points.values())


def test_qdrant_fresh_compatibility_preflight_does_not_create(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    upsert, _ = _requests(contract_corpus)

    store.require_compatible(upsert.manifest)

    assert backend.calls == ["describe"]


def test_qdrant_existing_collection_without_manifest_fails_closed(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    backend.collection = QdrantCollection("ragkit_dense", 32, "cosine")
    store = QdrantVectorStore(backend, "takehome")
    upsert, _ = _requests(contract_corpus)

    with pytest.raises(IndexCompatibilityError, match="manifest"):
        store.upsert(upsert)

    assert "upsert" not in backend.calls


def test_qdrant_search_round_trips_identity_and_keeps_native_similarity(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    upsert, search = _requests(contract_corpus)
    store.upsert(upsert)
    chunk_points = [item for item in backend.points.values() if _record_kind(item) == "chunk"]
    backend.matches = tuple(
        QdrantMatch(point.point_id, 0.75, point.payload) for point in reversed(chunk_points[:2])
    )

    results = store.search(search)

    assert [item.chunk.chunk_id for item in results] == sorted(
        (upsert.chunks[0].chunk_id, upsert.chunks[1].chunk_id), key=str
    )
    assert all(item.score.raw_score == item.score.relevance == 0.75 for item in results)
    assert backend.calls[-2:] == ["retrieve", "query"]


def test_qdrant_rejects_inexact_filter_before_provider_work(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    _, search = _requests(contract_corpus)

    with pytest.raises(UnsupportedCapabilityError, match="null"):
        store.search(replace(search, filters=Comparison("team", ComparisonOperator.EQ, None)))

    assert backend.calls == []


def test_qdrant_rejects_mixed_type_in_before_provider_work(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    _, search = _requests(contract_corpus)
    with pytest.raises(UnsupportedCapabilityError, match="homogeneous"):
        store.search(
            replace(
                search,
                filters=Comparison("team", ComparisonOperator.IN, (1, "one")),
            )
        )
    assert backend.calls == []


def test_qdrant_delete_maps_chunk_ids_deterministically_and_is_idempotent(
    contract_corpus: ContractCorpus,
) -> None:
    backend = _Backend()
    store = QdrantVectorStore(backend, "takehome")
    upsert, _ = _requests(contract_corpus)
    store.upsert(upsert)
    target = upsert.chunks[0].chunk_id

    store.delete(DeleteRequest((target,), upsert.manifest))
    store.delete(DeleteRequest((target,), upsert.manifest))

    assert [item for item in backend.calls if item == "delete"] == ["delete", "delete"]


def test_qdrant_translates_backend_failure_without_disclosing_details(
    contract_corpus: ContractCorpus,
) -> None:
    class FailingBackend(_Backend):
        def describe(self, collection_name: str) -> QdrantCollection | None:
            raise RuntimeError("api-key=secret")

    store = QdrantVectorStore(FailingBackend(), "takehome")
    upsert, _ = _requests(contract_corpus)

    with pytest.raises(ProviderError, match="compatibility check") as captured:
        store.require_compatible(upsert.manifest)

    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None
