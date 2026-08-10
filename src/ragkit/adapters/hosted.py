"""Optional hosted generation adapter with bounded, redacted failures."""

from __future__ import annotations

import importlib.util
import re
from importlib import import_module
from typing import Protocol, cast

from ragkit.domain import (
    ChunkId,
    ComponentFingerprint,
    InvalidDomainValueError,
    MissingDependencyError,
    ProviderError,
)
from ragkit.ports import GenerationRequest, GenerationResult, Generator, TokenUsage


class _Usage(Protocol):
    input_tokens: int
    output_tokens: int


class _Response(Protocol):
    output_text: str
    usage: _Usage | None


class _Responses(Protocol):
    def create(self, **kwargs: object) -> _Response: ...


class _Client(Protocol):
    responses: _Responses


class OpenAIHostedGenerator(Generator):
    """Call OpenAI Responses without exposing SDK types across the adapter boundary."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: object | None = None,
    ) -> None:
        if not model.strip() or not api_key.strip():
            raise InvalidDomainValueError("hosted generator model and API key must not be blank")
        if timeout_seconds <= 0 or max_retries < 0:
            raise InvalidDomainValueError(
                "hosted timeout must be positive and retries non-negative"
            )
        available = importlib.util.find_spec("openai") is not None
        if client is None and not available:
            raise MissingDependencyError("OpenAI adapter requires: install rag-kit[hosted]")
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client = cast(_Client | None, client)
        self._fingerprint = ComponentFingerprint.create(
            "generator",
            "openai_responses",
            {"version": 1, "model": model},
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            client = self._client or self._build_client()
            response = client.responses.create(
                model=self._model,
                input=request.prompt.text,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
            )
            raw_answer = response.output_text
            if not isinstance(raw_answer, str) or not raw_answer.strip():
                raise ValueError("empty output_text")
            usage = _token_usage(response.usage)
        except Exception as exc:
            raise _translate_provider_failure(exc) from None
        answer, citations = _sanitize_citations(raw_answer, request.prompt.cited_chunk_ids)
        return GenerationResult(answer, citations, self._fingerprint, usage)

    def _build_client(self) -> _Client:
        try:
            module = import_module("openai")
            client_type = module.OpenAI
        except ImportError as exc:  # pragma: no cover - guarded at construction
            raise MissingDependencyError(
                "OpenAI adapter requires: install rag-kit[hosted]", cause=exc
            ) from exc
        return cast(
            _Client,
            client_type(
                api_key=self._api_key,
                timeout=self._timeout_seconds,
                max_retries=self._max_retries,
            ),
        )


def _sanitize_citations(
    answer: str, allowed: tuple[ChunkId, ...]
) -> tuple[str, tuple[ChunkId, ...]]:
    allowed_by_text = {str(identifier): identifier for identifier in allowed}
    seen: set[ChunkId] = set()
    citations: list[ChunkId] = []
    pattern = re.compile(r"\[chunk_id:([^\]\s]+)\]")
    for raw in pattern.findall(answer):
        identifier = allowed_by_text.get(raw)
        if identifier is not None and identifier not in seen:
            seen.add(identifier)
            citations.append(identifier)
    sanitized = pattern.sub(
        lambda match: match.group(0) if match.group(1) in allowed_by_text else "[citation-removed]",
        answer,
    )
    return sanitized, tuple(citations)


def _token_usage(value: _Usage | None) -> TokenUsage | None:
    if value is None:
        return None
    return TokenUsage(value.input_tokens, value.output_tokens)


def _translate_provider_failure(exc: Exception) -> ProviderError:
    name = type(exc).__name__.casefold()
    rendered = str(exc).casefold()
    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in rendered:
        return ProviderError("hosted generation timed out")
    if "ratelimit" in name or "rate limit" in rendered or "429" in rendered:
        return ProviderError("hosted generation was rate limited")
    return ProviderError("hosted generation failed")
