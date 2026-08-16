from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from conftest import ContractCorpus
from ragkit.adapters.hosted import OpenAIHostedGenerator
from ragkit.domain import (
    MissingDependencyError,
    ProviderError,
)
from ragkit.ports import (
    GenerationRequest,
    Prompt,
    TokenUsage,
)

pytestmark = pytest.mark.unit


@dataclass
class _Usage:
    input_tokens: int = 12
    output_tokens: int = 4


@dataclass
class _Response:
    output_text: str
    usage: _Usage | None = None


class _Responses:
    def __init__(self, response: _Response | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _Client:
    def __init__(self, response: _Response | Exception) -> None:
        self.responses = _Responses(response)


def _request(contract_corpus: ContractCorpus) -> GenerationRequest:
    allowed = contract_corpus.scored[0].chunk.chunk_id
    return GenerationRequest(
        "alpha",
        contract_corpus.scored[:2],
        Prompt(f"Evidence [chunk_id:{allowed}]", (allowed,)),
        0.0,
        32,
    )


def test_hosted_generator_bounds_request_and_keeps_only_authorized_citations(
    contract_corpus: ContractCorpus,
) -> None:
    allowed = contract_corpus.scored[0].chunk.chunk_id
    unauthorized = contract_corpus.scored[2].chunk.chunk_id
    client = _Client(_Response(f"Answer [chunk_id:{allowed}] [chunk_id:{unauthorized}]", _Usage()))
    generator = OpenAIHostedGenerator(
        model="gpt-test-2026-01-01",
        api_key="sk-private",
        timeout_seconds=9.0,
        max_retries=1,
        client=client,
    )

    result = generator.generate(_request(contract_corpus))

    assert result.cited_chunk_ids == (allowed,)
    assert str(unauthorized) not in result.answer
    assert result.usage == TokenUsage(12, 4)
    call = client.responses.calls[0]
    assert call["model"] == "gpt-test-2026-01-01"
    assert call["max_output_tokens"] == 32
    assert "sk-private" not in repr(call)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (TimeoutError("request contains sk-private and raw evidence"), "timed out"),
        (RuntimeError("429 rate limit: sk-private"), "rate limited"),
        (RuntimeError("malformed provider response: sk-private"), "failed"),
    ],
)
def test_hosted_generator_classifies_failures_without_leaking_content_or_key(
    contract_corpus: ContractCorpus, failure: Exception, message: str
) -> None:
    generator = OpenAIHostedGenerator(
        model="gpt-test",
        api_key="sk-private",
        client=_Client(failure),
    )
    with pytest.raises(ProviderError, match=message) as caught:
        generator.generate(_request(contract_corpus))
    rendered = str(caught.value)
    assert "sk-private" not in rendered
    assert "raw evidence" not in rendered
    assert caught.value.__cause__ is None


def test_hosted_generator_rejects_blank_or_missing_sdk_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ragkit.adapters.hosted.importlib.util.find_spec", lambda name: None)
    with pytest.raises(MissingDependencyError, match=r"rag-kit\[hosted\]"):
        OpenAIHostedGenerator(model="gpt-test", api_key="sk-key")


def test_hosted_generator_passes_timeout_and_retry_policy_to_sdk(
    monkeypatch: pytest.MonkeyPatch, contract_corpus: ContractCorpus
) -> None:
    client = _Client(_Response("Grounded answer"))
    constructor_calls: list[dict[str, object]] = []

    def build_client(**kwargs: object) -> _Client:
        constructor_calls.append(kwargs)
        return client

    monkeypatch.setattr(
        "ragkit.adapters.hosted.import_module",
        lambda name: SimpleNamespace(OpenAI=build_client),
    )
    monkeypatch.setattr("ragkit.adapters.hosted.importlib.util.find_spec", lambda name: object())
    generator = OpenAIHostedGenerator(
        model="gpt-test",
        api_key="sk-key",
        timeout_seconds=7.5,
        max_retries=1,
    )

    generator.generate(_request(contract_corpus))

    assert constructor_calls == [{"api_key": "sk-key", "timeout": 7.5, "max_retries": 1}]
