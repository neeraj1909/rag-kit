"""Explicit composition root for dependency-free executable profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from urllib.parse import unquote, urlparse

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
from ragkit.application import AnsweringService, IndexingService, RagPipeline
from ragkit.domain import (
    ComponentFingerprint,
    IndexManifest,
    InvalidDomainValueError,
    NormalizationMode,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    Embedder,
    Evaluator,
    FamilyClassifier,
    Generator,
    PromptBuilder,
    Reranker,
    SourceConnector,
    Telemetry,
    VectorStore,
)

from .config import OfflineProfile

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class OfflineRuntime:
    """One process-local pipeline and its explicitly selected evaluator."""

    pipeline: RagPipeline
    evaluator: Evaluator
    evaluator_fingerprint: ComponentFingerprint
    chunker_fingerprint: ComponentFingerprint
    embedder_fingerprint: ComponentFingerprint
    embedding_dimension: int

    def manifest_for(self, source_uri: str) -> IndexManifest:
        """Describe index semantics for one logical corpus acquisition address."""

        if not isinstance(source_uri, str) or not source_uri.strip():
            raise InvalidDomainValueError("source URI must not be blank")
        canonical_source_uri = _canonical_filesystem_uri(source_uri)
        return IndexManifest(
            schema_version=1,
            corpus_fingerprint=ComponentFingerprint.create(
                "corpus",
                "filesystem-source",
                {
                    "source_uri": canonical_source_uri,
                    "selection_policy": "recursive_regular_files_except_pycache_v1",
                },
            ),
            chunker_fingerprint=self.chunker_fingerprint,
            embedder_fingerprint=self.embedder_fingerprint,
            embedding_dimension=self.embedding_dimension,
            normalization=NormalizationMode.L2,
            domain_schema_fingerprint=ComponentFingerprint.create(
                "domain_schema", "ragkit", {"version": 1}
            ),
        )


def _canonical_filesystem_uri(source_uri: str) -> str:
    parsed = urlparse(source_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise InvalidDomainValueError("remote file authorities are unsupported")
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(source_uri)
    else:
        raise InvalidDomainValueError(f"unsupported source scheme: {parsed.scheme}")
    return path.expanduser().resolve().as_uri()


def _unsupported(role: str, selection: str) -> UnsupportedCapabilityError:
    return UnsupportedCapabilityError(
        f"Phase 2 does not implement {role} selection {selection!r}",
        capability=f"{role}:{selection}",
    )


def _select(registry: dict[str, Callable[[], T]], role: str, selection: str) -> T:
    try:
        factory = registry[selection]
    except KeyError as error:
        raise _unsupported(role, selection) from error
    return factory()


def bootstrap(profile: OfflineProfile) -> OfflineRuntime:
    """Validate one profile and wire its adapters without hidden fallbacks."""

    if profile.family is not DocumentFamily.TEXT:
        raise _unsupported("family", profile.family.value)

    components = profile.components
    connector: SourceConnector = _select(
        {"filesystem": FilesystemSourceConnector}, "connector", components.connector
    )
    classifier: FamilyClassifier = _select(
        {"text": TextFamilyClassifier}, "classifier", components.classifier
    )
    extractor: DocumentExtractor = _select(
        {"text": TextDocumentExtractor}, "extractor", components.extractor
    )
    projector: DocumentProjector = _select(
        {"noop": NoOpDocumentProjector}, "projector", components.projector
    )
    chunker = _select(
        {"structure_aware": lambda: StructureAwareChunker(profile.limits.chunk_chars)},
        "chunker",
        components.chunker,
    )
    embedder: Embedder = _select(
        {"hashing": lambda: HashingEmbedder(profile.limits.embedding_dimension)},
        "embedder",
        components.embedder,
    )
    vector_store: VectorStore = _select(
        {"memory": InMemoryVectorStore}, "vector_store", components.vector_store
    )
    reranker: Reranker = _select({"noop": NoOpReranker}, "reranker", components.reranker)
    prompt_builder: PromptBuilder = _select(
        {"template": TemplatePromptBuilder}, "prompt_builder", components.prompt_builder
    )
    generator: Generator = _select(
        {"extractive": ExtractiveGenerator}, "generator", components.generator
    )
    evaluator: Evaluator = _select(
        {"deterministic": DeterministicEvaluator}, "evaluator", components.evaluator
    )
    telemetry: Telemetry = _select({"memory": InMemoryTelemetry}, "telemetry", components.telemetry)

    indexing = IndexingService(
        connector,
        classifier,
        extractor,
        projector,
        chunker,
        embedder,
        vector_store,
        telemetry,
    )
    answering = AnsweringService(
        embedder,
        vector_store,
        reranker,
        prompt_builder,
        generator,
        telemetry,
    )
    return OfflineRuntime(
        pipeline=RagPipeline(indexing, answering),
        evaluator=evaluator,
        evaluator_fingerprint=ComponentFingerprint.create(
            "evaluator", "deterministic", {"version": 1}
        ),
        chunker_fingerprint=chunker.fingerprint,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=embedder.dimension,
    )
