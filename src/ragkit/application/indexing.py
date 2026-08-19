"""Source-to-index application orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter_ns

from ragkit.domain import (
    BoxLocator,
    CellLocator,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    Document,
    ExtractionProvenance,
    IndexCompatibilityError,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    KeyframeLocator,
    PageLocator,
    SourceLocator,
    TextSpanLocator,
    TimeSpanLocator,
    derive_chunk_id,
)
from ragkit.ports import (
    AcquiredAsset,
    Chunker,
    ChunkingPolicy,
    ChunkingRequest,
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    Embedder,
    EmbeddingRequest,
    ExtractionRequest,
    FamilyClassifier,
    IndexingPolicy,
    IndexingStrategy,
    ProjectionRequest,
    SourceConnector,
    SourceRequest,
    SparseIndex,
    SparseUpsertRequest,
    Telemetry,
    UpsertRequest,
    VectorStore,
    derive_indexing_fingerprint,
    resolve_indexing_policy,
)

from ._telemetry import PipelineDiagnostic, StageTiming, invoke_timed


def _positive(value: int, label: str) -> None:
    if value <= 0:
        raise InvalidDomainValueError(f"{label} must be positive")


@dataclass(frozen=True, slots=True)
class IndexingRequest:
    """One bounded indexing invocation under an explicit immutable manifest."""

    source_uri: str
    manifest: IndexManifest
    max_assets: int = 32
    max_bytes_per_asset: int = 50_000_000
    max_documents: int = 32
    max_parts_per_document: int = 10_000
    max_chunks: int = 100_000
    chunking_policy: ChunkingPolicy = field(default_factory=ChunkingPolicy)
    indexing_policy: IndexingPolicy = field(default_factory=IndexingPolicy)

    def __post_init__(self) -> None:
        if not self.source_uri.strip():
            raise InvalidDomainValueError("source_uri must not be empty")
        _positive(self.max_assets, "max_assets")
        _positive(self.max_bytes_per_asset, "max_bytes_per_asset")
        _positive(self.max_documents, "max_documents")
        _positive(self.max_parts_per_document, "max_parts_per_document")
        _positive(self.max_chunks, "max_chunks")
        if not isinstance(self.indexing_policy, IndexingPolicy):
            raise InvalidDomainValueError("indexing_policy must be an IndexingPolicy")


@dataclass(frozen=True, slots=True)
class IndexedEvidence:
    """Content-free evidence summary produced by the indexing boundary."""

    chunk_id: ChunkId
    source_part_ids: tuple[str, ...]
    provenance: tuple[ExtractionProvenance, ...]

    def __post_init__(self) -> None:
        if not self.source_part_ids or len(self.source_part_ids) != len(self.provenance):
            raise IntegrityError("indexed evidence parts must align with provenance")


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Observable indexing outcome without source content or provider details."""

    manifest: IndexManifest
    asset_count: int
    document_count: int
    chunk_count: int
    indexed_chunk_ids: tuple[ChunkId, ...]
    indexed_evidence: tuple[IndexedEvidence, ...]
    diagnostics: tuple[PipelineDiagnostic, ...]
    timings: tuple[StageTiming, ...]

    def __post_init__(self) -> None:
        if self.chunk_count != len(self.indexed_chunk_ids):
            raise IntegrityError("indexed chunk count must align with returned chunk IDs")
        if len(set(self.indexed_chunk_ids)) != len(self.indexed_chunk_ids):
            raise IntegrityError("indexed chunk IDs must be unique")
        if tuple(item.chunk_id for item in self.indexed_evidence) != self.indexed_chunk_ids:
            raise IntegrityError("indexed evidence must align with returned chunk IDs")


class IndexingService:
    """Coordinate injected indexing capabilities in one stable order."""

    def __init__(
        self,
        connector: SourceConnector,
        classifier: FamilyClassifier,
        extractor: DocumentExtractor,
        projector: DocumentProjector,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore | None,
        telemetry: Telemetry,
        *,
        sparse_index: SparseIndex | None = None,
        indexing_policy: IndexingPolicy | None = None,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._connector = connector
        self._classifier = classifier
        self._extractor = extractor
        self._projector = projector
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._telemetry = telemetry
        self._sparse_index = sparse_index
        self._indexing_policy = resolve_indexing_policy(
            DocumentFamily.TEXT, indexing_policy or IndexingPolicy()
        )
        self._indexing_fingerprint = derive_indexing_fingerprint(
            self._indexing_policy,
            None if vector_store is None else vector_store.fingerprint,
            None if sparse_index is None else sparse_index.fingerprint,
        )
        self._clock = clock
        self._require_indexing_components()

    @property
    def indexing_fingerprint(self) -> ComponentFingerprint:
        """Identify the resolved policy and concrete dense/sparse index codecs."""

        return self._indexing_fingerprint

    def run(self, request: IndexingRequest) -> IndexingResult:
        """Acquire, classify, extract, project, chunk, embed, then upsert."""

        self._require_manifest_components(request.manifest)
        requested_policy = resolve_indexing_policy(DocumentFamily.TEXT, request.indexing_policy)
        if requested_policy != self._indexing_policy:
            raise InvalidDomainValueError(
                "request indexing policy does not match the composed indexing policy"
            )
        timings: list[StageTiming] = []
        assets = invoke_timed(
            "index.fetch",
            lambda: self._connector.fetch(
                SourceRequest(request.source_uri, request.max_assets, request.max_bytes_per_asset)
            ),
            self._telemetry,
            self._clock,
            timings,
            component=self._connector,
        )
        if not assets:
            return self._omitted(request.manifest, 0, 0, "acquisition", "no_assets", timings)

        classifications = invoke_timed(
            "index.classify",
            lambda: self._classifier.classify(assets),
            self._telemetry,
            self._clock,
            timings,
            component=self._classifier,
        )
        asset_ids = tuple(asset.reference.asset_id for asset in assets)
        classified_ids = tuple(item.asset_id for item in classifications)
        if classified_ids != asset_ids:
            raise IntegrityError("classification output must align exactly with acquired assets")

        documents = invoke_timed(
            "index.extract",
            lambda: self._extractor.extract(
                ExtractionRequest(assets, classifications, request.max_documents)
            ),
            self._telemetry,
            self._clock,
            timings,
            component=self._extractor,
        )
        if not documents:
            return self._omitted(
                request.manifest,
                len(assets),
                0,
                "extraction",
                "no_documents",
                timings,
            )
        self._require_acquired_assets(assets, documents)

        projected = invoke_timed(
            "index.project",
            lambda: self._projector.project(
                ProjectionRequest(documents, request.max_parts_per_document)
            ),
            self._telemetry,
            self._clock,
            timings,
            component=self._projector,
        )
        self._require_preserved_projection(documents, projected)

        chunks = invoke_timed(
            "index.chunk",
            lambda: self._chunker.chunk(
                ChunkingRequest(projected, request.max_chunks, request.chunking_policy)
            ),
            self._telemetry,
            self._clock,
            timings,
            component=self._chunker,
        )
        if not chunks:
            return self._omitted(
                request.manifest,
                len(assets),
                len(projected),
                "chunking",
                "no_chunks",
                timings,
            )
        self._require_resolved_chunks(projected, chunks)
        self._require_store_compatibility(request.manifest)

        if self._indexing_policy.strategy in {
            IndexingStrategy.DENSE,
            IndexingStrategy.HYBRID,
        }:
            embeddings = invoke_timed(
                "index.embed",
                lambda: self._embedder.embed_documents(
                    EmbeddingRequest(tuple(chunk.text for chunk in chunks))
                ),
                self._telemetry,
                self._clock,
                timings,
                component=self._embedder,
                count=lambda batch: len(batch.embeddings),
            )
            upsert = UpsertRequest(chunks, embeddings, request.manifest)
            vector_store = self._vector_store
            if vector_store is None:  # Constructor validation makes this unreachable.
                raise IntegrityError("dense indexing requires a vector store")
            invoke_timed(
                "index.upsert",
                lambda: vector_store.upsert(upsert),
                self._telemetry,
                self._clock,
                timings,
                component=vector_store,
                count=lambda _result: len(chunks),
            )
        if self._indexing_policy.strategy in {
            IndexingStrategy.SPARSE,
            IndexingStrategy.HYBRID,
        }:
            sparse_index = self._sparse_index
            if sparse_index is None:  # Constructor validation makes this unreachable.
                raise IntegrityError("sparse indexing requires a sparse index")
            invoke_timed(
                "index.sparse_upsert",
                lambda: sparse_index.upsert(SparseUpsertRequest(chunks, request.manifest)),
                self._telemetry,
                self._clock,
                timings,
                component=sparse_index,
                count=lambda _result: len(chunks),
            )
        return IndexingResult(
            request.manifest,
            len(assets),
            len(projected),
            len(chunks),
            tuple(chunk.chunk_id for chunk in chunks),
            tuple(
                IndexedEvidence(chunk.chunk_id, chunk.source_part_ids, chunk.provenance)
                for chunk in chunks
            ),
            (),
            tuple(timings),
        )

    def _require_manifest_components(self, manifest: IndexManifest) -> None:
        differences: dict[str, tuple[object, object]] = {}
        if manifest.chunker_fingerprint != self._chunker.fingerprint:
            differences["chunker_fingerprint"] = (
                manifest.chunker_fingerprint,
                self._chunker.fingerprint,
            )
        if manifest.embedder_fingerprint != self._embedder.fingerprint:
            differences["embedder_fingerprint"] = (
                manifest.embedder_fingerprint,
                self._embedder.fingerprint,
            )
        if manifest.embedding_dimension != self._embedder.dimension:
            differences["embedding_dimension"] = (
                manifest.embedding_dimension,
                self._embedder.dimension,
            )
        if manifest.indexing_fingerprint != self._indexing_fingerprint:
            differences["indexing_fingerprint"] = (
                manifest.indexing_fingerprint,
                self._indexing_fingerprint,
            )
        if differences:
            raise IndexCompatibilityError(differences)

    def _require_indexing_components(self) -> None:
        strategy = self._indexing_policy.strategy
        has_vector = self._vector_store is not None
        has_sparse = self._sparse_index is not None
        expected = {
            IndexingStrategy.DENSE: (True, False),
            IndexingStrategy.SPARSE: (False, True),
            IndexingStrategy.HYBRID: (True, True),
        }[strategy]
        if (has_vector, has_sparse) != expected:
            raise InvalidDomainValueError(
                f"indexing policy {strategy.value!r} requires "
                f"vector_store={expected[0]} and sparse_index={expected[1]}"
            )

    def _require_store_compatibility(self, manifest: IndexManifest) -> None:
        """Preflight every selected index before the first pipeline effect."""

        if self._vector_store is not None:
            self._vector_store.require_compatible(manifest)
        if self._sparse_index is not None:
            self._sparse_index.require_compatible(manifest)

    @staticmethod
    def _require_acquired_assets(
        assets: tuple[AcquiredAsset, ...], documents: tuple[Document, ...]
    ) -> None:
        acquired_references = {item.reference for item in assets}
        if any(
            asset not in acquired_references for document in documents for asset in document.assets
        ):
            raise IntegrityError("extracted documents must retain exact acquired asset references")

    @staticmethod
    def _require_preserved_projection(
        documents: tuple[Document, ...], projected: tuple[Document, ...]
    ) -> None:
        if tuple(item.document_id for item in projected) != tuple(
            item.document_id for item in documents
        ):
            raise IntegrityError("projection output must preserve document identity and order")
        for original, result in zip(documents, projected, strict=True):
            projected_parts = {part.part_id: part for part in result.parts}
            if any(
                part.part_id not in projected_parts
                or projected_parts[part.part_id].provenance != part.provenance
                for part in original.parts
            ):
                raise IntegrityError("projection must preserve every original part provenance")

    def _require_resolved_chunks(
        self, documents: tuple[Document, ...], chunks: tuple[Chunk, ...]
    ) -> None:
        documents_by_id = {document.document_id: document for document in documents}
        for chunk in chunks:
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                raise IntegrityError("chunk must resolve to a projected document")
            parts_by_id = {part.part_id: part for part in document.parts}
            if any(
                part_id not in parts_by_id
                or not IndexingService._provenance_resolves(
                    parts_by_id[part_id].provenance, provenance
                )
                for part_id, provenance in zip(chunk.source_part_ids, chunk.provenance, strict=True)
            ):
                raise IntegrityError("chunk source part provenance must resolve exactly")
            expected_chunk_id = derive_chunk_id(chunk, self._chunker.fingerprint)
            if chunk.chunk_id != expected_chunk_id:
                raise IntegrityError("chunk ID must match exact chunk content and provenance")

    @staticmethod
    def _provenance_resolves(source: ExtractionProvenance, derived: ExtractionProvenance) -> bool:
        if replace(derived, locator=source.locator) != source:
            return False
        return IndexingService._locator_is_within(source.locator, derived.locator)

    @staticmethod
    def _locator_is_within(source: SourceLocator, derived: SourceLocator) -> bool:
        if derived == source:
            return True
        if isinstance(source, TextSpanLocator) and isinstance(derived, TextSpanLocator):
            return source.start <= derived.start and derived.end <= source.end
        if isinstance(source, PageLocator):
            if isinstance(derived, PageLocator):
                return derived.page == source.page
            return isinstance(derived, BoxLocator) and derived.page == source.page
        if isinstance(source, BoxLocator) and isinstance(derived, BoxLocator):
            return (
                derived.page == source.page
                and source.x0 <= derived.x0 < derived.x1 <= source.x1
                and source.y0 <= derived.y0 < derived.y1 <= source.y1
            )
        if isinstance(source, CellLocator) and isinstance(derived, CellLocator):
            source_end_row = source.end_row
            source_end_column = source.end_column
            derived_end_row = derived.end_row
            derived_end_column = derived.end_column
            if (
                source_end_row is None
                or source_end_column is None
                or derived_end_row is None
                or derived_end_column is None
            ):
                return False
            return (
                derived.sheet == source.sheet
                and source.start_row <= derived.start_row
                and derived_end_row <= source_end_row
                and source.start_column <= derived.start_column
                and derived_end_column <= source_end_column
            )
        if isinstance(source, TimeSpanLocator) and isinstance(derived, TimeSpanLocator):
            return source.start_ms <= derived.start_ms and derived.end_ms <= source.end_ms
        return isinstance(source, KeyframeLocator) and derived == source

    @staticmethod
    def _omitted(
        manifest: IndexManifest,
        asset_count: int,
        document_count: int,
        stage: str,
        code: str,
        timings: list[StageTiming],
    ) -> IndexingResult:
        return IndexingResult(
            manifest,
            asset_count,
            document_count,
            0,
            (),
            (),
            (PipelineDiagnostic(stage, code, f"{stage} produced no indexable output"),),
            tuple(timings),
        )
