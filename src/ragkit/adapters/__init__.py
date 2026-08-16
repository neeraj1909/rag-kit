"""Dependency-light concrete adapters for the deterministic offline profile."""

from .classification import DeclaredFamilyClassifier
from .cross_encoder_reranker import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_MODEL_REVISION,
    LocalCrossEncoderReranker,
)
from .filesystem import FilesystemSourceConnector
from .generation import DeterministicEvaluator, ExtractiveGenerator, TemplatePromptBuilder
from .hosted import OpenAIHostedGenerator
from .layout import LayoutDocumentExtractor
from .media import LocalFasterWhisperTranscriber, MediaDocumentExtractor, PySceneDetectBackend
from .mixed import MixedImageDocumentExtractor
from .multimodal import EvidenceChunker
from .observability import InMemoryTelemetry, JsonLinesTelemetry, RequestCorrelatedTelemetry
from .ocr import OcrDocumentExtractor
from .retrieval import (
    BM25Config,
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
    NoOpReranker,
)
from .sqlite_store import SQLiteVectorStore
from .textual import (
    NoOpDocumentProjector,
    StructureAwareChunker,
    TextDocumentExtractor,
    TextFamilyClassifier,
)
from .torch_embedder import TorchTextEmbedder
from .vision import LocalSmolVLMBackend, VisionDocumentExtractor

__all__ = [
    "DEFAULT_RERANKER_MODEL_ID",
    "DEFAULT_RERANKER_MODEL_REVISION",
    "BM25Config",
    "BM25Retriever",
    "DeclaredFamilyClassifier",
    "DenseRetriever",
    "DeterministicEvaluator",
    "EvidenceChunker",
    "ExtractiveGenerator",
    "FilesystemSourceConnector",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryTelemetry",
    "InMemoryVectorStore",
    "JsonLinesTelemetry",
    "LayoutDocumentExtractor",
    "LocalCrossEncoderReranker",
    "LocalFasterWhisperTranscriber",
    "LocalSmolVLMBackend",
    "MediaDocumentExtractor",
    "MixedImageDocumentExtractor",
    "NoOpDocumentProjector",
    "NoOpReranker",
    "OcrDocumentExtractor",
    "OpenAIHostedGenerator",
    "PySceneDetectBackend",
    "RequestCorrelatedTelemetry",
    "SQLiteVectorStore",
    "StructureAwareChunker",
    "TemplatePromptBuilder",
    "TextDocumentExtractor",
    "TextFamilyClassifier",
    "TorchTextEmbedder",
    "VisionDocumentExtractor",
]
