from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import ContractCorpus
from ragkit.adapters.retrieval import (
    BM25Config,
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
)
from ragkit.domain import (
    Chunk,
    Comparison,
    ComparisonOperator,
    ComponentFingerprint,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    NormalizationMode,
    ScoredChunk,
    derive_chunk_id,
)
from ragkit.ports import (
    DeleteRequest,
    EmbeddingRequest,
    RetrievalRequest,
    Retriever,
    SparseUpsertRequest,
    UpsertRequest,
    VectorSearchRequest,
)

pytestmark = pytest.mark.unit


class StaticRetriever(Retriever):
    def __init__(self, name: str, values: tuple[ScoredChunk, ...]) -> None:
        self._fingerprint = ComponentFingerprint.create("retriever", name, {})
        self._values = values

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        return self._values[: request.top_k]


def _manifest_and_chunks(
    contract_corpus: ContractCorpus,
) -> tuple[HashingEmbedder, IndexManifest, tuple[Chunk, ...]]:
    embedder = HashingEmbedder(dimension=16)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
        normalization=NormalizationMode.L2,
    )
    texts = (
        "alpha alpha deployment guide",
        "alpha billing handbook",
        "gamma operations manual",
        "alpha alpha deployment guide",
        "delta support policy",
        "epsilon incident timeline",
        "zeta maintenance schedule",
    )
    changed = tuple(
        replace(
            chunk,
            text=texts[index],
            metadata={"tenant": "north" if index != 1 else "south", "priority": index},
        )
        for index, chunk in enumerate(contract_corpus.chunks)
    )
    chunks = tuple(
        replace(chunk, chunk_id=derive_chunk_id(chunk, manifest.chunker_fingerprint))
        for chunk in changed
    )
    return embedder, manifest, chunks


def test_bm25_has_explicit_fingerprint_and_deterministic_filtered_ranking(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    config = BM25Config(k1=1.2, b=0.75, token_pattern=r"[^\W_]+", lowercase=True)
    retriever = BM25Retriever(config=config)
    retriever.upsert(SparseUpsertRequest(chunks, manifest))

    request = RetrievalRequest(
        "ALPHA deployment",
        10,
        manifest,
        Comparison("tenant", ComparisonOperator.EQ, "north"),
    )
    first = retriever.retrieve(request)

    assert first == retriever.retrieve(request)
    assert {item.chunk.chunk_id for item in first} == {
        chunks[0].chunk_id,
        chunks[3].chunk_id,
    }
    assert [item.rank for item in first] == [1, 2]
    assert list(first) == sorted(
        first, key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id))
    )
    assert all(item.score.raw_score == item.score.relevance for item in first)
    assert all(item.score.provenance.metric == "bm25" for item in first)
    assert retriever.fingerprint == BM25Retriever(config=config).fingerprint
    assert retriever.fingerprint != BM25Retriever(config=replace(config, k1=1.8)).fingerprint


def test_bm25_upsert_delete_are_idempotent_and_manifest_safe(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    with pytest.raises(InvalidDomainValueError, match="duplicate"):
        SparseUpsertRequest((chunks[0], chunks[0]), manifest)
    retriever = BM25Retriever()
    with pytest.raises(IndexCompatibilityError):
        retriever.retrieve(RetrievalRequest("alpha", 10, manifest))
    upsert = SparseUpsertRequest(chunks, manifest)
    retriever.upsert(upsert)
    retriever.upsert(upsert)
    assert retriever.size == len(chunks)

    before = retriever.retrieve(RetrievalRequest("alpha", 10, manifest))
    incompatible = replace(manifest, schema_version=manifest.schema_version + 1)
    with pytest.raises(IndexCompatibilityError):
        retriever.upsert(SparseUpsertRequest((chunks[0],), incompatible))
    with pytest.raises(IndexCompatibilityError):
        retriever.delete(DeleteRequest((chunks[0].chunk_id,), incompatible))
    with pytest.raises(IndexCompatibilityError):
        retriever.retrieve(RetrievalRequest("alpha", 10, incompatible))
    assert retriever.retrieve(RetrievalRequest("alpha", 10, manifest)) == before

    deletion = DeleteRequest((chunks[0].chunk_id,), manifest)
    retriever.delete(deletion)
    retriever.delete(deletion)
    assert retriever.size == len(chunks) - 1
    assert chunks[0].chunk_id not in {
        item.chunk.chunk_id for item in retriever.retrieve(RetrievalRequest("alpha", 10, manifest))
    }


def test_bm25_rejects_same_id_with_changed_chunk_without_mutation(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    retriever = BM25Retriever()
    retriever.upsert(SparseUpsertRequest((chunks[0],), manifest))

    with pytest.raises(IntegrityError, match="stable content"):
        retriever.upsert(
            SparseUpsertRequest((replace(chunks[0], text="changed evidence"),), manifest)
        )

    assert retriever.retrieve(RetrievalRequest("alpha", 1, manifest))[0].chunk == chunks[0]


def test_dense_retriever_embeds_query_and_delegates_exact_filters(
    contract_corpus: ContractCorpus,
) -> None:
    embedder, manifest, chunks = _manifest_and_chunks(contract_corpus)
    store = InMemoryVectorStore()
    store.upsert(
        UpsertRequest(
            chunks,
            embedder.embed_documents(EmbeddingRequest(tuple(item.text for item in chunks))),
            manifest,
        )
    )
    retriever = DenseRetriever(embedder, store)
    request = RetrievalRequest(
        "alpha",
        2,
        manifest,
        Comparison("tenant", ComparisonOperator.EQ, "south"),
    )

    results = retriever.retrieve(request)

    assert len(results) == 1
    assert results[0].chunk == chunks[1]
    assert results[0].score.provenance.metric == "cosine"


def test_dense_store_rejects_same_id_with_changed_chunk_before_mutation(
    contract_corpus: ContractCorpus,
) -> None:
    embedder, manifest, chunks = _manifest_and_chunks(contract_corpus)
    store = InMemoryVectorStore()
    original = chunks[0]
    store.upsert(
        UpsertRequest(
            (original,),
            embedder.embed_documents(EmbeddingRequest((original.text,))),
            manifest,
        )
    )
    tampered = replace(original, text="changed evidence")

    with pytest.raises(IntegrityError, match="stable content"):
        store.upsert(
            UpsertRequest(
                (tampered,),
                embedder.embed_documents(EmbeddingRequest((tampered.text,))),
                manifest,
            )
        )

    assert (
        store.search(
            # The original remains searchable after the rejected atomic mutation.
            VectorSearchRequest(
                embedder.embed_query(original.text), embedder.fingerprint, 1, None, manifest
            )
        )[0].chunk
        == original
    )


def test_hybrid_rrf_deduplicates_preserves_candidate_scores_and_breaks_ties_by_id(
    contract_corpus: ContractCorpus,
) -> None:
    embedder, manifest, chunks = _manifest_and_chunks(contract_corpus)
    sparse = BM25Retriever()
    sparse.upsert(SparseUpsertRequest(chunks, manifest))
    store = InMemoryVectorStore()
    store.upsert(
        UpsertRequest(
            chunks,
            embedder.embed_documents(EmbeddingRequest(tuple(item.text for item in chunks))),
            manifest,
        )
    )
    dense = DenseRetriever(embedder, store)
    hybrid = HybridRetriever((("sparse", sparse), ("dense", dense)), rrf_k=60)

    results = hybrid.retrieve(RetrievalRequest("alpha deployment", 3, manifest))

    assert len(results) == 3
    assert len({item.chunk.chunk_id for item in results}) == len(results)
    assert [item.rank for item in results] == [1, 2, 3]
    assert list(results) == sorted(
        results, key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id))
    )
    assert all(item.score.provenance.metric == "rrf" for item in results)
    assert all(item.score.raw_score == item.score.relevance for item in results)
    assert all(item.prior_scores for item in results)
    assert all(
        item.prior_scores[0].provenance.stage in {"sparse_retrieval", "dense_retrieval"}
        for item in results
    )
    assert all(
        {score.provenance.stage for score in item.prior_scores}
        <= {"sparse_retrieval", "dense_retrieval"}
        for item in results
    )


def test_hybrid_rejects_same_id_with_different_full_chunk_values(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    left = BM25Retriever()
    left.upsert(SparseUpsertRequest((chunks[0],), manifest))
    valid = left.retrieve(RetrievalRequest("alpha", 1, manifest))
    altered = (replace(valid[0], chunk=replace(valid[0].chunk, text="different alpha")),)

    hybrid = HybridRetriever(
        (("left", StaticRetriever("left", valid)), ("right", StaticRetriever("right", altered)))
    )
    with pytest.raises(IntegrityError, match="different chunk values"):
        hybrid.retrieve(RetrievalRequest("alpha", 1, manifest))


def test_hybrid_equal_rrf_scores_use_full_stable_chunk_id_ties(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    left = BM25Retriever()
    right = BM25Retriever()
    left.upsert(SparseUpsertRequest((chunks[0],), manifest))
    right.upsert(SparseUpsertRequest((chunks[1],), manifest))

    results = HybridRetriever((("left", left), ("right", right))).retrieve(
        RetrievalRequest("alpha", 2, manifest)
    )

    assert results[0].score.relevance == results[1].score.relevance
    assert [str(item.chunk.chunk_id) for item in results] == sorted(
        str(item.chunk.chunk_id) for item in results
    )


def test_hybrid_fingerprint_includes_each_named_child_behavior() -> None:
    default = HybridRetriever((("sparse", BM25Retriever()),))
    changed = HybridRetriever((("sparse", BM25Retriever(config=BM25Config(k1=1.8))),))

    assert default.fingerprint != changed.fingerprint


def test_hybrid_rejects_unbounded_candidate_expansion(
    contract_corpus: ContractCorpus,
) -> None:
    _, manifest, chunks = _manifest_and_chunks(contract_corpus)
    sparse = BM25Retriever()
    sparse.upsert(SparseUpsertRequest(chunks, manifest))
    hybrid = HybridRetriever((("sparse", sparse),), candidate_multiplier=4, max_candidates=5)

    with pytest.raises(LimitExceededError, match="candidate request"):
        hybrid.retrieve(RetrievalRequest("alpha", 2, manifest))


@pytest.mark.parametrize(
    "config",
    [
        BM25Config(k1=0.0),
        BM25Config(b=-0.1),
        BM25Config(b=1.1),
        BM25Config(token_pattern="("),
    ],
)
def test_bm25_rejects_invalid_behavior_parameters(config: BM25Config) -> None:
    with pytest.raises((ValueError, TypeError)):
        config.validate()
