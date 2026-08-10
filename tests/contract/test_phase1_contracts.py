"""Reusable behavioral checks for provider-neutral adapters."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

from conftest import ContractCorpus
from fakes import (
    BrokenEmbedder,
    BrokenProjector,
    BrokenRetriever,
    FakeDocumentExtractor,
    FakeDocumentProjector,
    FakeEmbedder,
    FakeEvaluator,
    FakeFamilyClassifier,
    FakeGenerator,
    FakePromptBuilder,
    FakeReranker,
    FakeRetriever,
    FakeSourceConnector,
    FakeTelemetry,
    FakeVectorStore,
    SilentUnsupportedClassifier,
)
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    Comparison,
    ComparisonOperator,
    Document,
    ExtractionProvenance,
    IndexCompatibilityError,
    PageLocator,
    ScoredChunk,
    TextContent,
    TextSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    AcquiredAsset,
    DeleteRequest,
    Embedder,
    EmbeddingRequest,
    EvaluationCase,
    EvaluationExample,
    EvaluationRequest,
    ExtractionRequest,
    FamilyClassifier,
    GenerationRequest,
    ProjectionRequest,
    PromptRequest,
    RerankRequest,
    RetrievalRequest,
    SourceRequest,
    TelemetryEvent,
    TelemetryOutcome,
    UpsertRequest,
    VectorSearchRequest,
)

pytestmark = pytest.mark.contract


def assert_exact_provenance_preserved(before: Document, after: Document) -> None:
    """Reusable projector contract: part identity maps to identical evidence."""
    before_parts = {item.part_id: item.provenance for item in before.parts}
    after_parts = {item.part_id: item.provenance for item in after.parts}
    assert before_parts.items() <= after_parts.items()


def assert_embedding_alignment(embedder: Embedder, texts: tuple[str, ...]) -> None:
    """Reusable embedder contract: count, order, width and determinism align."""
    first = embedder.embed_documents(EmbeddingRequest(texts))
    second = embedder.embed_documents(EmbeddingRequest(texts))
    assert first == second
    assert len(first.embeddings) == len(texts)
    assert all(item.dimension == embedder.dimension for item in first.embeddings)
    assert first.embeddings == tuple(embedder.embed_query(text) for text in texts)


def assert_ranked_results(results: tuple[ScoredChunk, ...], *, top_k: int) -> None:
    """Reusable retrieval contract: finite, unique, bounded, canonical ordering."""
    assert len(results) <= top_k
    chunks = [item.chunk for item in results]
    assert len({item.chunk_id for item in chunks}) == len(chunks)
    expected = sorted(
        results,
        key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id)),
    )
    assert list(results) == expected
    assert [item.rank for item in results] == list(range(1, len(results) + 1))
    assert all(isfinite(item.score.relevance) for item in results)
    assert all(item.chunk.provenance for item in results)


def assert_unsupported_is_explicit(classifier: FamilyClassifier, asset: AcquiredAsset) -> None:
    """Reusable classifier contract: never silently omit an unsupported asset."""
    try:
        classifier.classify((asset,))
    except UnsupportedCapabilityError:
        return
    raise AssertionError("unsupported input was silently accepted or omitted")


def test_connector_classifier_and_extractor_are_deterministic_and_aligned(
    contract_corpus: ContractCorpus,
) -> None:
    connector = FakeSourceConnector()
    request = SourceRequest("memory://contract", max_assets=1, max_bytes_per_asset=1_000)
    assert connector.fetch(request) == connector.fetch(request)

    corpus_asset = AcquiredAsset(contract_corpus.document.assets[0], b"source")
    classifications = FakeFamilyClassifier().classify(
        (
            AcquiredAsset(
                replace(contract_corpus.document.assets[0], media_type="text/plain"),
                b"source",
            ),
        )
    )
    extractor = FakeDocumentExtractor((contract_corpus.document,))
    extracted = extractor.extract(ExtractionRequest((corpus_asset,), classifications, 1))
    assert extracted == (contract_corpus.document,)


def test_projector_and_chunker_retain_all_six_locator_shapes(
    contract_corpus: ContractCorpus,
) -> None:
    projected = FakeDocumentProjector().project(ProjectionRequest((contract_corpus.document,), 20))
    assert_exact_provenance_preserved(contract_corpus.document, projected[0])
    assert {item.provenance[0].locator.kind for item in contract_corpus.chunks} == {
        "text_span",
        "page",
        "box",
        "cell",
        "time",
        "keyframe",
    }
    boxes = [
        item.provenance.locator
        for item in contract_corpus.document.parts
        if item.provenance.locator.kind == "box"
    ]
    assert len(boxes) == 2  # OCR page region plus an image region are distinct evidence.
    assert contract_corpus.document.parts[1].provenance.locator == PageLocator(0)
    assert contract_corpus.document.parts[2].provenance.locator == BoxLocator(0, 0.1, 0.1, 0.4, 0.2)
    assert contract_corpus.document.parts[3].provenance.locator == CellLocator("Summary", 0, 0)
    assert [chunk.ordinal for chunk in contract_corpus.chunks] == list(
        range(len(contract_corpus.chunks))
    )
    assert all(chunk.provenance and chunk.source_part_ids for chunk in contract_corpus.chunks)


@pytest.mark.parametrize("texts", [(), ("one",), ("one", "two", "three")])
def test_embedder_contract(texts: tuple[str, ...]) -> None:
    assert_embedding_alignment(FakeEmbedder(), texts)


def test_vector_store_is_idempotent_manifest_aware_and_canonical(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = FakeEmbedder()
    batch = embedder.embed_documents(
        EmbeddingRequest(tuple(item.text for item in contract_corpus.chunks))
    )
    store = FakeVectorStore()
    upsert = UpsertRequest(contract_corpus.chunks, batch, contract_corpus.manifest)
    store.upsert(upsert)
    store.upsert(upsert)
    assert store.size == len(contract_corpus.chunks)

    results = store.search(
        VectorSearchRequest(
            embedder.embed_query("alpha"),
            embedder.fingerprint,
            3,
            None,
            contract_corpus.manifest,
        )
    )
    assert_ranked_results(results, top_k=3)

    incompatible = replace(
        contract_corpus.manifest,
        corpus_fingerprint=contract_corpus.manifest.domain_schema_fingerprint,
    )
    size_before = store.size
    with pytest.raises(IndexCompatibilityError):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 3, None, incompatible
            )
        )
    assert store.size == size_before

    store.delete(DeleteRequest((contract_corpus.chunks[0].chunk_id,), contract_corpus.manifest))
    store.delete(DeleteRequest((contract_corpus.chunks[0].chunk_id,), contract_corpus.manifest))
    assert store.size == size_before - 1


def test_retrieval_reranking_prompt_generation_evaluation_and_telemetry(
    contract_corpus: ContractCorpus,
) -> None:
    retrieved = FakeRetriever(contract_corpus.scored).retrieve(RetrievalRequest("alpha", 4))
    assert_ranked_results(retrieved, top_k=4)
    reranked = FakeReranker().rerank(RerankRequest("alpha", tuple(reversed(retrieved)), 3))
    assert_ranked_results(reranked, top_k=3)
    assert {item.chunk.chunk_id for item in reranked} <= {item.chunk.chunk_id for item in retrieved}

    prompt = FakePromptBuilder().build(PromptRequest("alpha", reranked, 10_000))
    generated = FakeGenerator().generate(GenerationRequest("alpha", reranked, prompt, 0.0, 32))
    assert generated.cited_chunk_ids == prompt.cited_chunk_ids
    assert set(generated.cited_chunk_ids) <= {item.chunk.chunk_id for item in reranked}

    example = EvaluationExample("case-1", "alpha", (reranked[0].chunk.chunk_id,))
    report = FakeEvaluator().evaluate(
        EvaluationRequest(
            (EvaluationCase(example, reranked, generated),),
            contract_corpus.manifest.embedder_fingerprint,
        )
    )
    assert report.evaluated_case_ids == ("case-1",)
    assert report.metrics[0].name == "hit_rate" and report.metrics[0].value == 1.0

    telemetry = FakeTelemetry()
    event = TelemetryEvent("answer", 10, 20, TelemetryOutcome.SUCCESS)
    telemetry.record(event)
    assert telemetry.events == [event]


def test_unsupported_capabilities_are_typed_and_never_silently_omitted() -> None:
    unsupported = AcquiredAsset(
        AssetRef("asset-image", "image/png", "b" * 64),
        b"image",
    )
    assert_unsupported_is_explicit(FakeFamilyClassifier(), unsupported)
    with pytest.raises(UnsupportedCapabilityError):
        FakeRetriever(()).retrieve(
            RetrievalRequest(
                "query",
                1,
                Comparison("department", ComparisonOperator.EQ, "support"),
            )
        )


def test_deliberately_broken_fakes_prove_contract_checks_have_teeth(
    contract_corpus: ContractCorpus,
) -> None:
    original = contract_corpus.document
    first = original.parts[0]
    assert isinstance(first, TextContent)
    shifted = replace(
        first,
        provenance=ExtractionProvenance(
            first.provenance.asset,
            TextSpanLocator(1, 5),
            first.provenance.extractor,
            first.provenance.confidence,
        ),
    )
    wrong_document = replace(original, parts=(shifted, *original.parts[1:]))
    broken_projection = BrokenProjector((wrong_document,)).project(
        ProjectionRequest((original,), 20)
    )
    with pytest.raises(AssertionError):
        assert_exact_provenance_preserved(original, broken_projection[0])

    with pytest.raises(AssertionError):
        assert_embedding_alignment(BrokenEmbedder(), ("one", "two"))

    bad_results = (contract_corpus.scored[-1], contract_corpus.scored[0])
    with pytest.raises(AssertionError):
        assert_ranked_results(
            BrokenRetriever(bad_results).retrieve(RetrievalRequest("query", 2)),
            top_k=2,
        )

    unsupported = AcquiredAsset(AssetRef("asset-image", "image/png", "b" * 64), b"image")
    with pytest.raises(AssertionError):
        assert_unsupported_is_explicit(SilentUnsupportedClassifier(), unsupported)
