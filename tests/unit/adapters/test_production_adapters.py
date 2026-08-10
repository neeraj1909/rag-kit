from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import ContractCorpus
from ragkit.adapters.chroma_store import ChromaVectorStore
from ragkit.adapters.hosted import OpenAIHostedGenerator
from ragkit.adapters.retrieval import HashingEmbedder
from ragkit.domain import (
    Comparison,
    ComparisonOperator,
    IndexCompatibilityError,
    IntegrityError,
    MissingDependencyError,
    NormalizationMode,
    Not,
    ProviderError,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    EmbeddingRequest,
    GenerationRequest,
    Prompt,
    TokenUsage,
    UpsertRequest,
    VectorSearchRequest,
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


class _Collection:
    def __init__(
        self,
        metadata: dict[str, object],
        query_result: dict[str, object] | None = None,
    ) -> None:
        self.metadata = metadata
        self.configuration = {"hnsw": {"space": "cosine"}}
        self.query_result = query_result or {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        self.query_calls = 0
        self.upsert_calls = 0

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls += 1

    def query(self, **kwargs: object) -> dict[str, object]:
        self.query_calls += 1
        return self.query_result

    def delete(self, **kwargs: object) -> None:
        raise AssertionError("delete was not expected")


class _ChromaClient:
    def __init__(self, collection: _Collection) -> None:
        self.collection = collection

    def get_collection(self, name: str, **kwargs: object) -> _Collection:
        return self.collection

    def create_collection(self, name: str, **kwargs: object) -> _Collection:
        raise AssertionError("create was not expected")


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


def test_chroma_preflights_manifest_before_provider_query_or_upsert(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    incompatible = replace(manifest, schema_version=2)
    collection = _Collection(
        {
            "ragkit_manifest_v1": json.dumps(manifest.to_dict()),
            "ragkit_metric": "cosine",
        }
    )
    store = ChromaVectorStore(Path("unused"), "test-collection", client=_ChromaClient(collection))
    chunk = contract_corpus.chunks[0]
    batch = embedder.embed_documents(EmbeddingRequest((chunk.text,)))

    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.upsert(UpsertRequest((chunk,), batch, incompatible))
    with pytest.raises(IndexCompatibilityError, match="schema_version"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 1, None, incompatible
            )
        )
    assert collection.upsert_calls == 0
    assert collection.query_calls == 0


def test_chroma_rejects_filter_it_cannot_translate_before_query(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    collection = _Collection(
        {
            "ragkit_manifest_v1": json.dumps(manifest.to_dict()),
            "ragkit_metric": "cosine",
        }
    )
    store = ChromaVectorStore("unused", "test-collection", client=_ChromaClient(collection))
    with pytest.raises(UnsupportedCapabilityError, match="general NOT"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"),
                embedder.fingerprint,
                1,
                Not(Comparison("category", ComparisonOperator.EQ, "keep")),
                manifest,
            )
        )
    assert collection.query_calls == 0


@pytest.mark.parametrize(
    "query_result",
    [
        {"ids": [[]], "metadatas": [[]]},
        {"ids": [["unexpected"]], "metadatas": [[]], "distances": [[]]},
    ],
)
def test_chroma_rejects_malformed_or_misaligned_query_batches(
    contract_corpus: ContractCorpus, query_result: dict[str, object]
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    collection = _Collection(
        {
            "ragkit_manifest_v1": json.dumps(manifest.to_dict()),
            "ragkit_metric": "cosine",
        },
        query_result,
    )
    store = ChromaVectorStore("unused", "test-collection", client=_ChromaClient(collection))

    with pytest.raises(IntegrityError, match="query"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 1, None, manifest
            )
        )


def test_chroma_rejects_provider_ids_that_do_not_match_stored_chunks(
    contract_corpus: ContractCorpus,
) -> None:
    embedder = HashingEmbedder(32)
    manifest = replace(
        contract_corpus.manifest,
        embedder_fingerprint=embedder.fingerprint,
        embedding_dimension=32,
        normalization=NormalizationMode.L2,
    )
    chunk = contract_corpus.chunks[0]
    encoded = json.dumps(chunk.to_dict())
    collection = _Collection(
        {
            "ragkit_manifest_v1": json.dumps(manifest.to_dict()),
            "ragkit_metric": "cosine",
        },
        {
            "ids": [["wrong-id", str(chunk.chunk_id)]],
            "metadatas": [[{"ragkit_chunk_v1": encoded}, {"ragkit_chunk_v1": encoded}]],
            "distances": [[0.1, 0.2]],
        },
    )
    store = ChromaVectorStore("unused", "test-collection", client=_ChromaClient(collection))

    with pytest.raises(IntegrityError, match="identity"):
        store.search(
            VectorSearchRequest(
                embedder.embed_query("alpha"), embedder.fingerprint, 2, None, manifest
            )
        )
