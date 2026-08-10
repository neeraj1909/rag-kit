from __future__ import annotations

from dataclasses import replace

import pytest

from ragkit.domain import (
    AssetRef,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    DocumentId,
    Embedding,
    ExtractionProvenance,
    IndexManifest,
    InvalidDomainValueError,
    NormalizationMode,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    TextSpanLocator,
)
from ragkit.ports import (
    EmbeddingBatch,
    EvaluationRequest,
    GenerationRequest,
    Prompt,
    RerankRequest,
    UpsertRequest,
    VectorSearchRequest,
)

pytestmark = pytest.mark.unit


def fingerprint(kind: str, implementation: str = "fixture") -> ComponentFingerprint:
    return ComponentFingerprint.create(kind, implementation, {"version": 1})


def manifest(
    embedder: ComponentFingerprint,
    *,
    dimension: int = 2,
    normalization: NormalizationMode = NormalizationMode.L2,
) -> IndexManifest:
    return IndexManifest(
        schema_version=1,
        corpus_fingerprint=fingerprint("corpus"),
        chunker_fingerprint=fingerprint("chunker"),
        embedder_fingerprint=embedder,
        embedding_dimension=dimension,
        normalization=normalization,
        domain_schema_fingerprint=fingerprint("schema"),
    )


def scored_chunk(label: str = "one") -> ScoredChunk:
    asset = AssetRef(f"asset-{label}", "text/plain", "a" * 64)
    provenance = ExtractionProvenance(
        asset=asset,
        locator=TextSpanLocator(0, len(label)),
        extractor=fingerprint("extractor"),
    )
    chunk = Chunk(
        chunk_id=ChunkId.from_payload({"label": label}),
        document_id=DocumentId.from_payload({"label": label}),
        ordinal=0,
        text=label,
        provenance=(provenance,),
        source_part_ids=(f"part-{label}",),
    )
    score_provenance = ScoreProvenance(
        component=fingerprint("retriever"),
        stage="retrieval",
        kind=ScoreKind.SIMILARITY,
        metric="dot",
        conversion="identity:v1",
    )
    return ScoredChunk(chunk, RetrievalScore(1.0, 1.0, score_provenance), 1)


@pytest.mark.parametrize("mismatch", ["fingerprint", "dimension", "normalization"])
def test_upsert_rejects_embedding_manifest_mismatch(mismatch: str) -> None:
    expected_embedder = fingerprint("embedder", "expected")
    actual_embedder = (
        fingerprint("embedder", "other") if mismatch == "fingerprint" else expected_embedder
    )
    vector = Embedding(
        values=(1.0, 0.0, 0.0) if mismatch == "dimension" else (1.0, 0.0),
        dimension=3 if mismatch == "dimension" else 2,
        normalized=mismatch != "normalization",
    )
    with pytest.raises(InvalidDomainValueError, match=mismatch):
        UpsertRequest(
            chunks=(scored_chunk().chunk,),
            embeddings=EmbeddingBatch((vector,), actual_embedder),
            manifest=manifest(expected_embedder),
        )


@pytest.mark.parametrize("mismatch", ["fingerprint", "dimension", "normalization"])
def test_vector_search_rejects_query_manifest_mismatch(mismatch: str) -> None:
    expected_embedder = fingerprint("embedder", "expected")
    actual_embedder = (
        fingerprint("embedder", "other") if mismatch == "fingerprint" else expected_embedder
    )
    vector = Embedding(
        values=(1.0, 0.0, 0.0) if mismatch == "dimension" else (1.0, 0.0),
        dimension=3 if mismatch == "dimension" else 2,
        normalized=mismatch != "normalization",
    )
    with pytest.raises(InvalidDomainValueError, match=mismatch):
        VectorSearchRequest(
            embedding=vector,
            embedder=actual_embedder,
            top_k=1,
            filters=None,
            expected_manifest=manifest(expected_embedder),
        )


@pytest.mark.parametrize(
    ("normalization", "normalized"),
    [(NormalizationMode.NONE, False), (NormalizationMode.L2, True)],
)
def test_vector_requests_accept_matching_manifest_semantics(
    normalization: NormalizationMode, normalized: bool
) -> None:
    embedder = fingerprint("embedder")
    vector = Embedding((1.0, 0.0), 2, normalized)
    expected_manifest = manifest(embedder, normalization=normalization)

    UpsertRequest((scored_chunk().chunk,), EmbeddingBatch((vector,), embedder), expected_manifest)
    VectorSearchRequest(vector, embedder, 1, None, expected_manifest)


def test_rerank_rejects_duplicate_chunk_candidates() -> None:
    candidate = scored_chunk()
    with pytest.raises(InvalidDomainValueError, match="duplicate"):
        RerankRequest("query", (candidate, replace(candidate, rank=2)), 2)


def test_generation_prompt_citations_must_be_present_in_context() -> None:
    context = (scored_chunk("context"),)
    absent = ChunkId.from_payload({"label": "absent"})
    with pytest.raises(InvalidDomainValueError, match="citation"):
        GenerationRequest("query", context, Prompt("prompt", (absent,)), 0.0, 10)


def test_evaluation_request_rejects_empty_cases() -> None:
    with pytest.raises(InvalidDomainValueError, match="case"):
        EvaluationRequest((), fingerprint("evaluator"))
