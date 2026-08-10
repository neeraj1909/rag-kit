"""Dependency-light concrete adapters for the deterministic offline profile."""

from .chroma_store import ChromaVectorStore
from .classification import DeclaredFamilyClassifier
from .filesystem import FilesystemSourceConnector
from .generation import DeterministicEvaluator, ExtractiveGenerator, TemplatePromptBuilder
from .hosted import OpenAIHostedGenerator
from .layout import LayoutDocumentExtractor
from .media import LocalFasterWhisperTranscriber, MediaDocumentExtractor, PySceneDetectBackend
from .mixed import MixedImageDocumentExtractor
from .multimodal import EvidenceChunker
from .observability import InMemoryTelemetry
from .ocr import OcrDocumentExtractor
from .retrieval import HashingEmbedder, InMemoryVectorStore, NoOpReranker
from .textual import (
    NoOpDocumentProjector,
    StructureAwareChunker,
    TextDocumentExtractor,
    TextFamilyClassifier,
)
from .torch_embedder import TorchTextEmbedder
from .vision import LocalSmolVLMBackend, VisionDocumentExtractor

__all__ = [
    "ChromaVectorStore",
    "DeclaredFamilyClassifier",
    "DeterministicEvaluator",
    "EvidenceChunker",
    "ExtractiveGenerator",
    "FilesystemSourceConnector",
    "HashingEmbedder",
    "InMemoryTelemetry",
    "InMemoryVectorStore",
    "LayoutDocumentExtractor",
    "LocalFasterWhisperTranscriber",
    "LocalSmolVLMBackend",
    "MediaDocumentExtractor",
    "MixedImageDocumentExtractor",
    "NoOpDocumentProjector",
    "NoOpReranker",
    "OcrDocumentExtractor",
    "OpenAIHostedGenerator",
    "PySceneDetectBackend",
    "StructureAwareChunker",
    "TemplatePromptBuilder",
    "TextDocumentExtractor",
    "TextFamilyClassifier",
    "TorchTextEmbedder",
    "VisionDocumentExtractor",
]
