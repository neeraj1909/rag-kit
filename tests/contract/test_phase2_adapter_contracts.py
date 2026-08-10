from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import ContractCorpus
from contract_assertions import (
    assert_chunker_contract,
    assert_classifier_contract,
    assert_embedding_alignment,
    assert_evaluator_contract,
    assert_extractor_contract,
    assert_generator_contract,
    assert_projector_contract,
    assert_prompt_contract,
    assert_reranker_contract,
    assert_source_contract,
    assert_telemetry_contract,
    assert_vector_store_contract,
)
from ragkit.adapters import (
    DeterministicEvaluator,
    ExtractiveGenerator,
    FilesystemSourceConnector,
    HashingEmbedder,
    InMemoryTelemetry,
    InMemoryVectorStore,
    NoOpDocumentProjector,
    NoOpReranker,
    StructureAwareChunker,
    TemplatePromptBuilder,
    TextDocumentExtractor,
    TextFamilyClassifier,
)
from ragkit.domain import IndexCompatibilityError, NormalizationMode
from ragkit.ports import (
    ChunkingRequest,
    EmbeddingRequest,
    EvaluationCase,
    EvaluationExample,
    EvaluationRequest,
    ExtractionRequest,
    GenerationRequest,
    ProjectionRequest,
    PromptRequest,
    RerankRequest,
    SourceRequest,
    TelemetryEvent,
    TelemetryOutcome,
    UpsertRequest,
    VectorSearchRequest,
)

pytestmark = pytest.mark.contract


def test_all_real_phase2_adapters_satisfy_shared_behavioral_contracts(
    tmp_path: Path,
    contract_corpus: ContractCorpus,
) -> None:
    (tmp_path / "guide.md").write_text("# Guide\n\nAlpha is grounded here.")
    (tmp_path / "notes.txt").write_text("Beta is also grounded.")
    generated = tmp_path / "__pycache__"
    generated.mkdir()
    (generated / "ignored.py").write_text("not selected")

    assets = assert_source_contract(
        FilesystemSourceConnector(), SourceRequest(tmp_path.as_uri(), 2, 10_000)
    )
    assert all("__pycache__" not in (item.reference.uri or "") for item in assets)
    classifications = assert_classifier_contract(TextFamilyClassifier(), assets)
    documents = assert_extractor_contract(
        TextDocumentExtractor(), ExtractionRequest(assets, classifications, 2)
    )
    projected = assert_projector_contract(NoOpDocumentProjector(), ProjectionRequest(documents, 10))
    chunker = StructureAwareChunker(32)
    chunks = assert_chunker_contract(chunker, ChunkingRequest(projected, 20))

    embedder = HashingEmbedder(32)
    texts = tuple(item.text for item in chunks)
    assert_embedding_alignment(embedder, texts)
    embeddings = embedder.embed_documents(EmbeddingRequest(texts))
    manifest = replace(
        contract_corpus.manifest,
        chunker_fingerprint=chunker.fingerprint,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
        normalization=NormalizationMode.L2,
    )
    store = InMemoryVectorStore()
    query_embedding = embedder.embed_query("alpha")
    candidates = assert_vector_store_contract(
        store,
        UpsertRequest(chunks, embeddings, manifest),
        VectorSearchRequest(query_embedding, embedder.fingerprint, 5, None, manifest),
    )
    context = assert_reranker_contract(
        NoOpReranker(), RerankRequest("alpha", tuple(reversed(candidates)), 3)
    )
    prompt = assert_prompt_contract(TemplatePromptBuilder(), PromptRequest("alpha", context, 1_000))
    generated_answer = assert_generator_contract(
        ExtractiveGenerator(), GenerationRequest("alpha", context, prompt, 0.0, 20)
    )
    example = EvaluationExample(
        "phase2-real",
        "alpha",
        (context[0].chunk.chunk_id,),
        generated_answer.answer,
    )
    assert_evaluator_contract(
        DeterministicEvaluator(),
        EvaluationRequest(
            (EvaluationCase(example, context, generated_answer),), manifest.fingerprint
        ),
    )
    telemetry = InMemoryTelemetry()
    assert_telemetry_contract(
        telemetry,
        TelemetryEvent("phase2.contract", 1, 2, TelemetryOutcome.SUCCESS),
        lambda: telemetry.events,
    )


def test_real_offline_embedder_obeys_alignment_contract() -> None:
    embedder = HashingEmbedder(32)
    texts = ("alpha", "beta", "alpha")
    batch = embedder.embed_documents(EmbeddingRequest(texts))
    assert len(batch.embeddings) == len(texts)
    assert batch.embeddings == tuple(embedder.embed_query(text) for text in texts)
    assert batch.embeddings[0] == batch.embeddings[2]
    assert all(item.dimension == embedder.dimension for item in batch.embeddings)


def test_real_memory_store_rejects_manifest_mismatch_before_mutation(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
        normalization=NormalizationMode.L2,
    )
    batch = embedder.embed_documents(
        EmbeddingRequest(tuple(item.text for item in contract_corpus.chunks))
    )
    store = InMemoryVectorStore()
    store.upsert(UpsertRequest(contract_corpus.chunks, batch, manifest))
    before = store.size
    incompatible = replace(manifest, schema_version=2)
    with pytest.raises(IndexCompatibilityError):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"),
                embedder.fingerprint,
                2,
                None,
                incompatible,
            )
        )
    assert store.size == before


def test_real_memory_store_requires_manifest_initialization(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
        normalization=NormalizationMode.L2,
    )
    store = InMemoryVectorStore()
    with pytest.raises(IndexCompatibilityError, match="manifest"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 2, None, manifest
            )
        )
