"""Query-to-cited-answer application orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns

from ragkit.domain import (
    ChunkId,
    DocumentId,
    ExtractionProvenance,
    IndexManifest,
    IntegrityError,
    InvalidDomainValueError,
    MetadataFilter,
    ScoredChunk,
)
from ragkit.ports import (
    GenerationRequest,
    GenerationResult,
    Generator,
    PromptBuilder,
    PromptRequest,
    Reranker,
    RerankRequest,
    RetrievalRequest,
    Retriever,
    Telemetry,
)

from ._telemetry import PipelineDiagnostic, StageTiming, invoke_timed


def _positive(value: int, label: str) -> None:
    if value <= 0:
        raise InvalidDomainValueError(f"{label} must be positive")


@dataclass(frozen=True, slots=True)
class AnsweringRequest:
    """One bounded query under explicit index semantics."""

    query: str
    expected_manifest: IndexManifest
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    max_context_chars: int = 20_000
    temperature: float = 0.0
    max_output_tokens: int = 512
    filters: MetadataFilter | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise InvalidDomainValueError("query must not be empty")
        _positive(self.retrieval_top_k, "retrieval_top_k")
        _positive(self.rerank_top_k, "rerank_top_k")
        _positive(self.max_context_chars, "max_context_chars")
        _positive(self.max_output_tokens, "max_output_tokens")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise InvalidDomainValueError("temperature must be finite and non-negative")
        if self.rerank_top_k > self.retrieval_top_k:
            raise InvalidDomainValueError("rerank_top_k must not exceed retrieval_top_k")


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    """A generated citation resolved back to exact source evidence."""

    chunk_id: ChunkId
    document_id: DocumentId
    rank: int
    provenance: tuple[ExtractionProvenance, ...]

    def __post_init__(self) -> None:
        _positive(self.rank, "citation rank")


@dataclass(frozen=True, slots=True)
class AnsweringResult:
    """Generated output, exact evidence, diagnostics, and sanitized durations."""

    generation: GenerationResult | None
    context: tuple[ScoredChunk, ...]
    citations: tuple[AnswerCitation, ...]
    diagnostics: tuple[PipelineDiagnostic, ...]
    timings: tuple[StageTiming, ...]


class AnsweringService:
    """Coordinate injected retrieval, reranking, prompting, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        generator: Generator,
        telemetry: Telemetry,
        *,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._generator = generator
        self._telemetry = telemetry
        self._clock = clock

    def run(self, request: AnsweringRequest) -> AnsweringResult:
        """Retrieve, rerank, build a prompt, generate, and resolve citations."""

        timings: list[StageTiming] = []
        candidates = invoke_timed(
            "ask.retrieve",
            lambda: self._retriever.retrieve(
                RetrievalRequest(
                    request.query,
                    request.retrieval_top_k,
                    request.expected_manifest,
                    request.filters,
                )
            ),
            self._telemetry,
            self._clock,
            timings,
        )
        self._validate_ranked("retrieval", candidates, request.retrieval_top_k)
        if not candidates:
            return self._omitted("retrieval", "no_retrieval_results", timings)

        context = invoke_timed(
            "ask.rerank",
            lambda: self._reranker.rerank(
                RerankRequest(request.query, candidates, request.rerank_top_k)
            ),
            self._telemetry,
            self._clock,
            timings,
        )
        self._validate_ranked("reranking", context, request.rerank_top_k)
        candidate_chunks = {item.chunk.chunk_id: item.chunk for item in candidates}
        for item in context:
            searched_chunk = candidate_chunks.get(item.chunk.chunk_id)
            if searched_chunk is None:
                raise IntegrityError("reranking introduced a chunk absent from search results")
            if item.chunk != searched_chunk:
                raise IntegrityError("reranking substituted chunk content under an existing ID")
        if not context:
            return self._omitted("reranking", "no_rerank_results", timings)

        prompt = invoke_timed(
            "ask.prompt",
            lambda: self._prompt_builder.build(
                PromptRequest(request.query, context, request.max_context_chars)
            ),
            self._telemetry,
            self._clock,
            timings,
        )
        context_ids = {item.chunk.chunk_id for item in context}
        if len(set(prompt.cited_chunk_ids)) != len(prompt.cited_chunk_ids) or any(
            chunk_id not in context_ids for chunk_id in prompt.cited_chunk_ids
        ):
            raise IntegrityError("prompt citations must be unique members of the final context")

        generation = invoke_timed(
            "ask.generate",
            lambda: self._generator.generate(
                GenerationRequest(
                    request.query,
                    context,
                    prompt,
                    request.temperature,
                    request.max_output_tokens,
                )
            ),
            self._telemetry,
            self._clock,
            timings,
        )
        if len(set(generation.cited_chunk_ids)) != len(generation.cited_chunk_ids) or any(
            chunk_id not in prompt.cited_chunk_ids for chunk_id in generation.cited_chunk_ids
        ):
            raise IntegrityError("generated citation is absent from the supplied prompt context")
        context_by_id = {item.chunk.chunk_id: item for item in context}
        citations = tuple(
            AnswerCitation(
                chunk_id,
                context_by_id[chunk_id].chunk.document_id,
                context_by_id[chunk_id].rank,
                context_by_id[chunk_id].chunk.provenance,
            )
            for chunk_id in generation.cited_chunk_ids
        )
        return AnsweringResult(generation, context, citations, (), tuple(timings))

    @staticmethod
    def _validate_ranked(stage: str, values: tuple[ScoredChunk, ...], top_k: int) -> None:
        identifiers = tuple(item.chunk.chunk_id for item in values)
        if len(values) > top_k or len(set(identifiers)) != len(identifiers):
            raise IntegrityError(f"{stage} results must be bounded and unique")
        if tuple(item.rank for item in values) != tuple(range(1, len(values) + 1)):
            raise IntegrityError(f"{stage} results must have contiguous one-based ranks")
        expected = tuple(
            sorted(values, key=lambda item: (-item.score.relevance, str(item.chunk.chunk_id)))
        )
        if values != expected:
            raise IntegrityError(f"{stage} results must use deterministic relevance ordering")

    @staticmethod
    def _omitted(stage: str, code: str, timings: list[StageTiming]) -> AnsweringResult:
        return AnsweringResult(
            None,
            (),
            (),
            (PipelineDiagnostic(stage, code, f"{stage} produced no answerable evidence"),),
            tuple(timings),
        )
