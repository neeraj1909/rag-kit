from __future__ import annotations

from hashlib import sha256

import pytest

from ragkit.application import IndexingRequest, IndexingService
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ContentPart,
    Document,
    DocumentId,
    Embedding,
    ExtractionProvenance,
    ImageContent,
    IndexCompatibilityError,
    IndexManifest,
    InvalidDomainValueError,
    LayoutContent,
    MediaContent,
    NormalizationMode,
    OcrContent,
    PageLocator,
    ScoredChunk,
    SourceId,
    SourceLocator,
    TextContent,
    TextSpanLocator,
    TimeSpanLocator,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    Chunker,
    ChunkingRequest,
    DeleteRequest,
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    Embedder,
    EmbeddingBatch,
    EmbeddingRequest,
    ExtractionRequest,
    FamilyClassifier,
    IndexingPolicy,
    IndexingStrategy,
    PhysicalIndexStrategy,
    ProjectionRequest,
    SourceConnector,
    SourceRequest,
    SparseIndex,
    SparseUpsertRequest,
    Telemetry,
    TelemetryEvent,
    UpsertRequest,
    VectorDatabase,
    VectorSearchRequest,
    VectorStore,
    derive_indexing_fingerprint,
    resolve_indexing_policy,
)

pytestmark = pytest.mark.unit


def _fingerprint(kind: str) -> ComponentFingerprint:
    return ComponentFingerprint.create(kind, "indexing_strategy_test", {"version": 1})


class _Connector(SourceConnector):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def fetch(self, request: SourceRequest) -> tuple[AcquiredAsset, ...]:
        self.calls.append("fetch")
        content = b"evidence"
        return (
            AcquiredAsset(
                AssetRef(
                    "asset-1",
                    "text/plain",
                    sha256(content).hexdigest(),
                    request.source_uri,
                    len(content),
                ),
                content,
            ),
        )


class _Classifier(FamilyClassifier):
    def __init__(self, family: DocumentFamily = DocumentFamily.TEXT) -> None:
        self._family = family

    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        return tuple(
            AssetClassification(
                asset.reference.asset_id,
                self._family,
                1.0,
                _fingerprint("classifier"),
            )
            for asset in assets
        )


class _Extractor(DocumentExtractor):
    def __init__(self, family: DocumentFamily = DocumentFamily.TEXT) -> None:
        self._family = family

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        asset = request.assets[0].reference
        source_id = SourceId.from_locator("memory", {"name": "indexing"})
        document_id = DocumentId.from_assets(source_id, (asset.sha256,))
        locator: SourceLocator
        if self._family is DocumentFamily.TEXT:
            locator = TextSpanLocator(0, 8)
        elif self._family is DocumentFamily.OCR:
            locator = PageLocator(0)
        elif self._family is DocumentFamily.LAYOUT:
            locator = CellLocator("Sheet1", 0, 0)
        elif self._family is DocumentFamily.VISION:
            locator = BoxLocator(0, 0.1, 0.1, 0.9, 0.9)
        else:
            locator = TimeSpanLocator(0, 1_000)
        provenance = ExtractionProvenance(
            asset,
            locator,
            _fingerprint("extractor"),
            confidence=0.9 if self._family is DocumentFamily.OCR else None,
        )
        part: ContentPart
        if self._family is DocumentFamily.TEXT:
            part = TextContent("part-1", "evidence", provenance)
        elif self._family is DocumentFamily.OCR:
            part = OcrContent("part-1", "evidence", provenance)
        elif self._family is DocumentFamily.LAYOUT:
            part = LayoutContent("part-1", "evidence", provenance)
        elif self._family is DocumentFamily.VISION:
            part = ImageContent("part-1", "evidence", provenance)
        else:
            part = MediaContent("part-1", "evidence", provenance)
        return (
            Document(
                document_id,
                source_id,
                (asset,),
                (part,),
            ),
        )


class _Projector(DocumentProjector):
    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        return request.documents


class _Chunker(Chunker):
    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("chunker")

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        document = request.documents[0]
        part = document.parts[0]
        return (
            Chunk(
                ChunkId.from_content(
                    document.document_id,
                    self.fingerprint,
                    ((part.part_id, part.provenance.locator),),
                    "evidence",
                ),
                document.document_id,
                0,
                "evidence",
                (part.provenance,),
                (part.part_id,),
            ),
        )


class _Embedder(Embedder):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @property
    def dimension(self) -> int:
        return 2

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("embedder")

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        self.calls.append("embed")
        return EmbeddingBatch(
            tuple(Embedding((1.0, 0.0), 2) for _ in request.texts),
            self.fingerprint,
        )

    def embed_query(self, text: str) -> Embedding:
        return Embedding((1.0, 0.0), 2)


class _VectorStore(VectorStore):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("vector_store")

    def require_compatible(self, manifest: IndexManifest) -> None:
        self.calls.append("vector_preflight")

    def upsert(self, request: UpsertRequest) -> None:
        self.calls.append("vector_upsert")

    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        raise AssertionError("search is not part of indexing")

    def delete(self, request: DeleteRequest) -> None:
        raise AssertionError("delete is not part of indexing")


class _SparseIndex(SparseIndex):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return _fingerprint("sparse_index")

    def require_compatible(self, manifest: IndexManifest) -> None:
        self.calls.append("sparse_preflight")

    def upsert(self, request: SparseUpsertRequest) -> None:
        self.calls.append("sparse_upsert")

    def delete(self, request: DeleteRequest) -> None:
        raise AssertionError("delete is not part of indexing")


class _Telemetry(Telemetry):
    def record(self, event: TelemetryEvent) -> None:
        pass


def _policy(
    strategy: IndexingStrategy, family: DocumentFamily = DocumentFamily.TEXT
) -> IndexingPolicy:
    if strategy is IndexingStrategy.SPARSE:
        return resolve_indexing_policy(
            family,
            IndexingPolicy(strategy, VectorDatabase.NONE, PhysicalIndexStrategy.NONE),
        )
    return resolve_indexing_policy(
        family,
        IndexingPolicy(strategy, VectorDatabase.MEMORY, PhysicalIndexStrategy.EXACT),
    )


def _manifest(policy: IndexingPolicy) -> IndexManifest:
    vector_fingerprint = (
        None if policy.strategy is IndexingStrategy.SPARSE else _fingerprint("vector_store")
    )
    sparse_fingerprint = (
        None if policy.strategy is IndexingStrategy.DENSE else _fingerprint("sparse_index")
    )
    return IndexManifest(
        2,
        _fingerprint("corpus"),
        _fingerprint("chunker"),
        _fingerprint("embedder"),
        2,
        NormalizationMode.NONE,
        _fingerprint("domain_schema"),
        derive_indexing_fingerprint(policy, vector_fingerprint, sparse_fingerprint),
    )


def _service(
    calls: list[str], strategy: IndexingStrategy, family: DocumentFamily = DocumentFamily.TEXT
) -> tuple[IndexingService, IndexingPolicy]:
    policy = _policy(strategy, family)
    vector_store = None if strategy is IndexingStrategy.SPARSE else _VectorStore(calls)
    sparse_index = None if strategy is IndexingStrategy.DENSE else _SparseIndex(calls)
    return (
        IndexingService(
            _Connector(calls),
            _Classifier(family),
            _Extractor(family),
            _Projector(),
            _Chunker(),
            _Embedder(calls),
            vector_store,
            _Telemetry(),
            sparse_index=sparse_index,
            indexing_policy=policy,
        ),
        policy,
    )


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            IndexingStrategy.DENSE,
            ["fetch", "vector_preflight", "embed", "vector_upsert"],
        ),
        (
            IndexingStrategy.SPARSE,
            ["fetch", "sparse_preflight", "sparse_upsert"],
        ),
        (
            IndexingStrategy.HYBRID,
            [
                "fetch",
                "vector_preflight",
                "sparse_preflight",
                "embed",
                "vector_upsert",
                "sparse_upsert",
            ],
        ),
    ],
)
def test_indexing_materializes_only_the_selected_logical_indexes(
    strategy: IndexingStrategy, expected: list[str]
) -> None:
    calls: list[str] = []
    service, policy = _service(calls, strategy)
    result = service.run(
        IndexingRequest(
            "memory://fixture",
            _manifest(policy),
            indexing_policy=policy,
        )
    )
    assert calls == expected
    assert result.chunk_count == 1


def test_every_document_family_indexes_through_dense_sparse_and_hybrid_with_same_evidence() -> None:
    for family in DocumentFamily:
        identities: list[tuple[ChunkId, tuple[ExtractionProvenance, ...]]] = []
        for strategy in (
            IndexingStrategy.DENSE,
            IndexingStrategy.SPARSE,
            IndexingStrategy.HYBRID,
        ):
            calls: list[str] = []
            service, policy = _service(calls, strategy, family)
            result = service.run(
                IndexingRequest(
                    "memory://fixture",
                    _manifest(policy),
                    indexing_policy=policy,
                )
            )
            assert result.chunk_count == 1
            identities.append((result.indexed_chunk_ids[0], result.indexed_evidence[0].provenance))
        assert identities[0] == identities[1] == identities[2]


@pytest.mark.parametrize(
    ("strategy", "vector", "sparse"),
    [
        (IndexingStrategy.DENSE, False, False),
        (IndexingStrategy.DENSE, True, True),
        (IndexingStrategy.SPARSE, True, True),
        (IndexingStrategy.HYBRID, True, False),
    ],
)
def test_composition_rejects_missing_or_decorative_index_components(
    strategy: IndexingStrategy, vector: bool, sparse: bool
) -> None:
    calls: list[str] = []
    with pytest.raises(InvalidDomainValueError, match="requires"):
        IndexingService(
            _Connector(calls),
            _Classifier(),
            _Extractor(),
            _Projector(),
            _Chunker(),
            _Embedder(calls),
            _VectorStore(calls) if vector else None,
            _Telemetry(),
            sparse_index=_SparseIndex(calls) if sparse else None,
            indexing_policy=_policy(strategy),
        )


def test_policy_and_manifest_mismatch_fail_before_connector_or_store_preflight() -> None:
    calls: list[str] = []
    service, dense = _service(calls, IndexingStrategy.DENSE)
    sparse = _policy(IndexingStrategy.SPARSE)
    with pytest.raises(InvalidDomainValueError, match="request indexing policy"):
        service.run(
            IndexingRequest(
                "memory://fixture",
                _manifest(dense),
                indexing_policy=sparse,
            )
        )
    assert calls == []

    with pytest.raises(IndexCompatibilityError) as error:
        service.run(
            IndexingRequest(
                "memory://fixture",
                _manifest(sparse),
                indexing_policy=dense,
            )
        )
    assert "indexing_fingerprint" in error.value.differences
    assert calls == []
