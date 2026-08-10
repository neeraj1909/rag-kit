from __future__ import annotations

import inspect
from abc import ABC
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from ragkit.ports import (
    Chunker,
    DocumentExtractor,
    DocumentProjector,
    Embedder,
    Evaluator,
    FamilyClassifier,
    Generator,
    PromptBuilder,
    Reranker,
    Retriever,
    SourceConnector,
    Telemetry,
    VectorStore,
)

pytestmark = pytest.mark.unit

PORTS = (
    SourceConnector,
    FamilyClassifier,
    DocumentExtractor,
    DocumentProjector,
    Chunker,
    Embedder,
    VectorStore,
    Retriever,
    Reranker,
    PromptBuilder,
    Generator,
    Evaluator,
    Telemetry,
)


def _contains_any(annotation: object) -> bool:
    return annotation is Any or any(_contains_any(item) for item in get_args(annotation))


def test_ports_are_abstract_and_cannot_be_instantiated() -> None:
    for port in PORTS:
        assert issubclass(port, ABC)
        assert inspect.isabstract(port)
        with pytest.raises(TypeError):
            port()  # type: ignore[abstract]


def test_port_methods_are_synchronous_explicit_and_fully_typed() -> None:
    for port in PORTS:
        for name in port.__abstractmethods__:
            member = inspect.getattr_static(port, name)
            function = member.fget if isinstance(member, property) else member
            assert function is not None
            assert not inspect.iscoroutinefunction(function)
            signature = inspect.signature(function)
            assert all(
                parameter.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for parameter in signature.parameters.values()
            )
            hints = get_type_hints(function)
            expected = {key for key in signature.parameters if key != "self"} | {"return"}
            assert expected <= hints.keys(), f"missing annotations on {port.__name__}.{name}"
            assert all(not _contains_any(hints[key]) for key in expected)


def test_each_port_documents_the_complete_behavioral_contract() -> None:
    required_concepts = (
        "order",
        "limit",
        "invalid",
        "error",
        "effect",
        "determin",
        "confidence",
        "score",
        "thread",
        "unsupported",
    )
    for port in PORTS:
        documentation = inspect.getdoc(port) or ""
        missing = [concept for concept in required_concepts if concept not in documentation.lower()]
        assert not missing, f"{port.__name__} missing semantic docs: {missing}"
        for name in port.__abstractmethods__:
            member = inspect.getattr_static(port, name)
            function = member.fget if isinstance(member, property) else member
            assert inspect.getdoc(function), f"{port.__name__}.{name} has no operation docs"


def test_no_port_annotation_exposes_provider_modules() -> None:
    forbidden = ("chromadb", "torch", "transformers", "openai")
    for port in PORTS:
        for name in port.__abstractmethods__:
            member = inspect.getattr_static(port, name)
            function = member.fget if isinstance(member, property) else member
            for annotation in get_type_hints(function).values():
                rendered = repr(annotation).lower()
                assert get_origin(annotation) is not dict
                assert not any(provider in rendered for provider in forbidden)
