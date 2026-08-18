"""Explicit composition root for dependency-free executable profiles."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar
from urllib.parse import unquote, urlparse

from ragkit.adapters import (
    AdaptiveChunker,
    BM25Config,
    BM25Retriever,
    DeclaredFamilyClassifier,
    DenseRetriever,
    DeterministicEvaluator,
    ExtractiveGenerator,
    FilesystemSourceConnector,
    HashingEmbedder,
    HybridRetriever,
    InMemoryTelemetry,
    InMemoryVectorStore,
    LayoutDocumentExtractor,
    LocalCrossEncoderReranker,
    LocalFasterWhisperTranscriber,
    LocalSmolVLMBackend,
    MediaDocumentExtractor,
    MixedImageDocumentExtractor,
    NoOpDocumentProjector,
    NoOpReranker,
    OcrDocumentExtractor,
    OpenAIHostedGenerator,
    PySceneDetectBackend,
    SQLiteVectorStore,
    TemplatePromptBuilder,
    TextDocumentExtractor,
    TextFamilyClassifier,
    TorchTextEmbedder,
    VisionDocumentExtractor,
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
    ChunkingPolicy,
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    Embedder,
    Evaluator,
    FamilyClassifier,
    Generator,
    PromptBuilder,
    Reranker,
    Retriever,
    SourceConnector,
    SparseIndex,
    Telemetry,
    VectorStore,
)

from .config import OfflineProfile
from .optional import OptionalCapability, inspect_optional_capability

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
    chunking_policy: ChunkingPolicy

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
        f"ragkit does not implement {role} selection {selection!r}",
        capability=f"{role}:{selection}",
    )


def _select(registry: dict[str, Callable[[], T]], role: str, selection: str) -> T:
    try:
        factory = registry[selection]
    except KeyError as error:
        raise _unsupported(role, selection) from error
    return factory()


def _validate_family_selections(profile: OfflineProfile) -> None:
    extractor_matrix: dict[DocumentFamily, str | tuple[str, ...]] = {
        DocumentFamily.TEXT: "text",
        DocumentFamily.OCR: "ocr",
        DocumentFamily.LAYOUT: "layout",
        DocumentFamily.VISION: ("vision", "mixed_image"),
        DocumentFamily.MEDIA: "media",
    }
    expected_extractor = extractor_matrix[profile.family]
    expected_classifier = "text" if profile.family is DocumentFamily.TEXT else "declared"
    expected_chunker = "adaptive"
    actual = {
        "classifier": profile.components.classifier,
        "extractor": profile.components.extractor,
        "chunker": profile.components.chunker,
    }
    expected = {
        "classifier": expected_classifier,
        "extractor": expected_extractor,
        "chunker": expected_chunker,
    }
    mismatch: dict[str, tuple[object, object]] = {}
    for role, expected_value in expected.items():
        accepted = expected_value if isinstance(expected_value, tuple) else (expected_value,)
        if actual[role] not in accepted:
            mismatch[role] = (expected_value, actual[role])
    if mismatch:
        raise UnsupportedCapabilityError(
            f"component selections do not match {profile.family.value} family: {mismatch}",
            capability="family_component_matrix",
        )


def inspect_profile(profile: OfflineProfile) -> dict[str, object]:
    """Describe selected capability requirements without importing optional SDKs."""

    requirements: list[OptionalCapability] = []
    if profile.family is DocumentFamily.OCR:
        requirements.extend(
            (
                OptionalCapability("ocr", "PIL", distribution="pillow"),
                OptionalCapability(
                    "ocr",
                    "pytesseract",
                    binary="tesseract",
                    model=profile.settings.ocr_language,
                ),
            )
        )
        if profile.source.casefold().endswith(".pdf"):
            requirements.append(OptionalCapability("ocr", "pypdfium2"))
    if profile.family is DocumentFamily.LAYOUT:
        suffix = Path(profile.source).suffix.casefold()
        layout_module = {
            ".pdf": "pdfplumber",
            ".pptx": "pptx",
            ".xlsx": "openpyxl",
            ".xlsm": "openpyxl",
        }.get(suffix, "pdfplumber")
        distribution = {"pptx": "python-pptx"}.get(layout_module, layout_module)
        requirements.append(OptionalCapability("layout", layout_module, distribution=distribution))
    if profile.family is DocumentFamily.VISION:
        requirements.extend(
            (
                OptionalCapability("vision", "PIL", distribution="pillow"),
                OptionalCapability("vision", "torch"),
                OptionalCapability(
                    "vision",
                    "transformers",
                    model=(
                        f"{profile.settings.vision_model_id}@{profile.settings.vision_revision}"
                    ),
                ),
            )
        )
        if profile.components.extractor == "mixed_image":
            requirements.append(
                OptionalCapability(
                    "ocr",
                    "pytesseract",
                    binary="tesseract",
                    model=profile.settings.ocr_language,
                )
            )
    if profile.family is DocumentFamily.MEDIA:
        requirements.extend(
            (
                OptionalCapability(
                    "media",
                    "faster_whisper",
                    model=f"{profile.settings.media_model_id}@{profile.settings.media_revision}",
                ),
                OptionalCapability("media", "scenedetect"),
                OptionalCapability("media", "cv2", distribution="opencv-python"),
            )
        )
    if profile.components.embedder == "torch" and not any(
        item.extra == "vision" for item in requirements
    ):
        requirements.extend(
            (
                OptionalCapability("vision", "torch"),
                OptionalCapability(
                    "vision",
                    "transformers",
                    model=(
                        f"{profile.settings.embedder_model_id}@{profile.settings.embedder_revision}"
                    ),
                ),
            )
        )
    if profile.components.reranker == "cross-encoder":
        requirements.extend(
            (
                OptionalCapability("reranking", "torch"),
                OptionalCapability(
                    "reranking",
                    "transformers",
                    model=(
                        f"{profile.settings.reranker_model_id}@{profile.settings.reranker_revision}"
                    ),
                ),
            )
        )
    if profile.components.generator == "openai":
        requirements.append(
            OptionalCapability("hosted", "openai", credential_env=profile.settings.credential_env)
        )

    component_values = asdict(profile.components)
    selection_fingerprints = {
        role: str(
            ComponentFingerprint.create(
                "component_selection",
                role,
                {"selection": selection, "profile": str(profile.fingerprint)},
            )
        )
        for role, selection in component_values.items()
    }
    degraded = {
        DocumentFamily.TEXT: (),
        DocumentFamily.OCR: ("handwriting_best_effort", "low_confidence_explicit"),
        DocumentFamily.LAYOUT: ("embedded_images_require_ocr_or_vision",),
        DocumentFamily.VISION: ("model_descriptions_uncalibrated",),
        DocumentFamily.MEDIA: ("speaker_identity_unknown", "asr_confidence_unavailable"),
    }[profile.family]
    return {
        "supported_families": [family.value for family in DocumentFamily],
        "selected_family": profile.family.value,
        "selection_fingerprints": selection_fingerprints,
        "device": profile.settings.device,
        "limits": {
            **asdict(profile.limits),
            "adapter": {
                key: value
                for key, value in asdict(profile.settings).items()
                if key.startswith(
                    (
                        "ocr_max_",
                        "layout_max_",
                        "vision_max_",
                        "media_max_",
                        "bm25_",
                        "hybrid_",
                        "reranker_",
                    )
                )
                or key.endswith("_timeout_seconds")
                or key == "timeout_seconds"
            },
        },
        "degraded_modes": list(degraded),
        "requirements": [asdict(inspect_optional_capability(item)) for item in requirements],
    }


def bootstrap(profile: OfflineProfile, *, telemetry: Telemetry | None = None) -> OfflineRuntime:
    """Validate one profile and wire its adapters without hidden fallbacks."""

    _validate_family_selections(profile)
    chunking_policy = profile.chunking_policy
    components = profile.components
    connector: SourceConnector = _select(
        {"filesystem": FilesystemSourceConnector}, "connector", components.connector
    )
    classifier_factories: dict[str, Callable[[], FamilyClassifier]] = {
        "text": TextFamilyClassifier,
        "declared": lambda: DeclaredFamilyClassifier(profile.family),
    }
    classifier = _select(classifier_factories, "classifier", components.classifier)
    extractor_factories: dict[str, Callable[[], DocumentExtractor]] = {
        "text": TextDocumentExtractor,
        "ocr": lambda: OcrDocumentExtractor(
            language=profile.settings.ocr_language,
            content_mode=profile.settings.ocr_content_mode,
            max_pages=profile.settings.ocr_max_pages,
            max_pixels=profile.settings.ocr_max_pixels,
            timeout_seconds=profile.settings.ocr_timeout_seconds,
        ),
        "layout": lambda: LayoutDocumentExtractor(
            max_pages=profile.settings.layout_max_pages,
            max_slides=profile.settings.layout_max_slides,
            max_sheets=profile.settings.layout_max_sheets,
            max_cells=profile.settings.layout_max_cells,
            max_archive_uncompressed_bytes=profile.settings.layout_max_archive_bytes,
            max_compression_ratio=profile.settings.layout_max_compression_ratio,
        ),
        "vision": lambda: VisionDocumentExtractor(
            LocalSmolVLMBackend(
                model_id=profile.settings.vision_model_id,
                revision=profile.settings.vision_revision,
                image_longest_edge=profile.settings.vision_image_longest_edge,
            ),
            max_new_tokens=profile.settings.vision_max_new_tokens,
            max_pixels=profile.settings.vision_max_pixels,
            max_dimension=profile.settings.vision_max_dimension,
            max_regions=profile.settings.vision_max_regions,
            timeout_seconds=profile.settings.vision_timeout_seconds,
        ),
        "mixed_image": lambda: MixedImageDocumentExtractor(
            OcrDocumentExtractor(
                language=profile.settings.ocr_language,
                content_mode=profile.settings.ocr_content_mode,
                max_pages=profile.settings.ocr_max_pages,
                max_pixels=profile.settings.ocr_max_pixels,
                timeout_seconds=profile.settings.ocr_timeout_seconds,
            ),
            VisionDocumentExtractor(
                LocalSmolVLMBackend(
                    model_id=profile.settings.vision_model_id,
                    revision=profile.settings.vision_revision,
                    image_longest_edge=profile.settings.vision_image_longest_edge,
                ),
                max_new_tokens=profile.settings.vision_max_new_tokens,
                max_pixels=profile.settings.vision_max_pixels,
                max_dimension=profile.settings.vision_max_dimension,
                max_regions=profile.settings.vision_max_regions,
                timeout_seconds=profile.settings.vision_timeout_seconds,
            ),
        ),
        "media": lambda: MediaDocumentExtractor(
            transcriber=LocalFasterWhisperTranscriber(
                model_id=profile.settings.media_model_id,
                revision=profile.settings.media_revision,
            ),
            scene_detector=PySceneDetectBackend(),
            max_duration_ms=profile.settings.media_max_duration_ms,
            max_segments=profile.settings.media_max_segments,
            max_scenes=profile.settings.media_max_scenes,
            timeout_seconds=profile.settings.media_timeout_seconds,
        ),
    }
    extractor = _select(extractor_factories, "extractor", components.extractor)
    projector: DocumentProjector = _select(
        {"noop": NoOpDocumentProjector}, "projector", components.projector
    )
    chunker = _select(
        {
            "adaptive": lambda: AdaptiveChunker(profile.family, chunking_policy),
        },
        "chunker",
        components.chunker,
    )
    embedder: Embedder = _select(
        {
            "hashing": lambda: HashingEmbedder(profile.limits.embedding_dimension),
            "torch": lambda: TorchTextEmbedder(
                model_id=profile.settings.embedder_model_id,
                revision=profile.settings.embedder_revision,
                device=profile.settings.device,
                batch_size=profile.settings.batch_size,
                max_length=profile.settings.max_length,
                pooling=profile.settings.pooling,
            ),
        },
        "embedder",
        components.embedder,
    )
    vector_store: VectorStore = _select(
        {
            "memory": InMemoryVectorStore,
            "sqlite": lambda: SQLiteVectorStore(
                profile.settings.persistence_path, profile.settings.collection_name
            ),
        },
        "vector_store",
        components.vector_store,
    )
    dense_retriever = DenseRetriever(embedder, vector_store)
    bm25_retriever = BM25Retriever(
        config=BM25Config(
            k1=profile.settings.bm25_k1,
            b=profile.settings.bm25_b,
            token_pattern=profile.settings.bm25_token_pattern,
            lowercase=profile.settings.bm25_lowercase,
        )
    )
    retriever: Retriever = _select(
        {
            "dense": lambda: dense_retriever,
            "sparse": lambda: bm25_retriever,
            "hybrid": lambda: HybridRetriever(
                (("dense", dense_retriever), ("sparse", bm25_retriever)),
                rrf_k=profile.settings.hybrid_rrf_k,
                candidate_multiplier=profile.settings.hybrid_candidate_multiplier,
                max_candidates=profile.limits.max_chunks,
            ),
        },
        "retriever",
        components.retriever,
    )
    sparse_index: SparseIndex | None = (
        bm25_retriever if components.retriever in {"sparse", "hybrid"} else None
    )
    reranker: Reranker = _select(
        {
            "noop": NoOpReranker,
            "cross-encoder": lambda: LocalCrossEncoderReranker(
                model_id=profile.settings.reranker_model_id,
                revision=profile.settings.reranker_revision,
                device=profile.settings.device,
                batch_size=profile.settings.reranker_batch_size,
                max_length=profile.settings.reranker_max_length,
                max_top_k=profile.settings.reranker_max_top_k,
                max_candidates=profile.settings.reranker_max_candidates,
            ),
        },
        "reranker",
        components.reranker,
    )
    prompt_builder: PromptBuilder = _select(
        {"template": TemplatePromptBuilder}, "prompt_builder", components.prompt_builder
    )

    def hosted_generator() -> Generator:
        credential = os.environ.get(profile.settings.credential_env)
        if not credential:
            raise UnsupportedCapabilityError(
                f"hosted generator requires environment variable {profile.settings.credential_env}",
                capability="hosted_credential",
            )
        return OpenAIHostedGenerator(
            model=profile.settings.hosted_model,
            api_key=credential,
            timeout_seconds=profile.settings.timeout_seconds,
            max_retries=profile.settings.max_retries,
        )

    generator: Generator = _select(
        {"extractive": ExtractiveGenerator, "openai": hosted_generator},
        "generator",
        components.generator,
    )
    evaluator: Evaluator = _select(
        {"deterministic": DeterministicEvaluator}, "evaluator", components.evaluator
    )
    selected_telemetry = (
        telemetry
        if telemetry is not None
        else _select({"memory": InMemoryTelemetry}, "telemetry", components.telemetry)
    )

    indexing = IndexingService(
        connector,
        classifier,
        extractor,
        projector,
        chunker,
        embedder,
        vector_store,
        selected_telemetry,
        sparse_index=sparse_index,
    )
    answering = AnsweringService(
        retriever,
        reranker,
        prompt_builder,
        generator,
        selected_telemetry,
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
        chunking_policy=chunking_policy,
    )
