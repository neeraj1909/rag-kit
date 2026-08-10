"""Public application façade for indexing and answering."""

from __future__ import annotations

from dataclasses import dataclass

from .answering import AnsweringRequest, AnsweringResult, AnsweringService
from .indexing import IndexingRequest, IndexingResult, IndexingService


@dataclass(frozen=True, slots=True)
class RagPipeline:
    """Expose the two use cases without hiding their explicit typed requests."""

    indexing: IndexingService
    answering: AnsweringService

    def index(self, request: IndexingRequest) -> IndexingResult:
        return self.indexing.run(request)

    def ask(self, request: AnsweringRequest) -> AnsweringResult:
        return self.answering.run(request)
