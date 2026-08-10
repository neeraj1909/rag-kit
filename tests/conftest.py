"""Shared provenance-complete values for behavioral contract suites."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fakes import FakeChunker
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    Chunk,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionProvenance,
    ImageContent,
    IndexManifest,
    KeyframeLocator,
    LayoutContent,
    MediaContent,
    NormalizationMode,
    OcrContent,
    PageLocator,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
    SourceId,
    SourceLocator,
    TextContent,
    TextSpanLocator,
    TimeSpanLocator,
)
from ragkit.ports import ChunkingRequest


@dataclass(frozen=True)
class ContractCorpus:
    document: Document
    chunks: tuple[Chunk, ...]
    scored: tuple[ScoredChunk, ...]
    manifest: IndexManifest


@pytest.fixture
def contract_corpus() -> ContractCorpus:
    asset = AssetRef("asset-contract", "application/octet-stream", "a" * 64)
    extractor = ComponentFingerprint.create("extractor", "contract_fake", {"version": 1})

    def provenance(locator: SourceLocator, confidence: float | None = None) -> ExtractionProvenance:
        return ExtractionProvenance(asset, locator, extractor, confidence)

    parts = (
        TextContent("text", "alpha", provenance(TextSpanLocator(0, 5))),
        OcrContent("ocr-page", "invoice page", provenance(PageLocator(0), 0.75)),
        OcrContent("ocr", "invoice", provenance(BoxLocator(0, 0.1, 0.1, 0.4, 0.2), 0.9)),
        LayoutContent("cell", "total 42", provenance(CellLocator("Summary", 0, 0))),
        ImageContent("image", "damaged valve", provenance(BoxLocator(1, 0.2, 0.2, 0.3, 0.3), 0.8)),
        MediaContent("time", "restart pump", provenance(TimeSpanLocator(100, 900), 0.95)),
        MediaContent("keyframe", "gauge reading", provenance(KeyframeLocator(700, 21), 0.85)),
    )
    source_id = SourceId.from_locator("memory", {"name": "contract"})
    document_id = DocumentId.from_assets(source_id, (asset.sha256,))
    document = Document(document_id, source_id, (asset,), parts)
    chunks = FakeChunker().chunk(ChunkingRequest((document,), 20))
    score_source = ScoreProvenance(
        ComponentFingerprint.create("retriever", "contract_fake", {}),
        "retrieval",
        ScoreKind.SIMILARITY,
        "fixture",
        "identity:v1",
    )
    scored = tuple(
        ScoredChunk(chunk, RetrievalScore(float(len(chunks) - rank), None, score_source), rank + 1)
        for rank, chunk in enumerate(chunks)
    )
    manifest = IndexManifest(
        1,
        ComponentFingerprint.create("corpus", "contract", {}),
        FakeChunker().fingerprint,
        ComponentFingerprint.create("embedder", "contract_fake", {"version": 1}),
        3,
        NormalizationMode.NONE,
        ComponentFingerprint.create("schema", "domain", {"version": 1}),
    )
    return ContractCorpus(document, chunks, scored, manifest)
