"""Dependency-light concrete adapters for the deterministic offline profile."""

from .filesystem import FilesystemSourceConnector
from .generation import DeterministicEvaluator, ExtractiveGenerator, TemplatePromptBuilder
from .observability import InMemoryTelemetry
from .retrieval import HashingEmbedder, InMemoryVectorStore, NoOpReranker
from .textual import (
    NoOpDocumentProjector,
    StructureAwareChunker,
    TextDocumentExtractor,
    TextFamilyClassifier,
)

__all__ = [
    "DeterministicEvaluator",
    "ExtractiveGenerator",
    "FilesystemSourceConnector",
    "HashingEmbedder",
    "InMemoryTelemetry",
    "InMemoryVectorStore",
    "NoOpDocumentProjector",
    "NoOpReranker",
    "StructureAwareChunker",
    "TemplatePromptBuilder",
    "TextDocumentExtractor",
    "TextFamilyClassifier",
]
