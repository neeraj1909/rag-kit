from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

import pytest

from ragkit.application import (
    AnsweringRequest,
    AnsweringResult,
    IndexingRequest,
    IndexingResult,
)
from ragkit.delivery import server as server_module
from ragkit.delivery.http import HttpApp
from ragkit.domain import IndexManifest, InvalidDomainValueError
from ragkit.infrastructure.bootstrap import OfflineRuntime
from ragkit.infrastructure.config import OfflineProfile
from ragkit.ports import Telemetry, TelemetryEvent, TelemetryOutcome

pytestmark = pytest.mark.integration


@dataclass
class StubPipeline:
    manifest: IndexManifest

    def __post_init__(self) -> None:
        self.index_requests: list[IndexingRequest] = []
        self.ask_requests: list[AnsweringRequest] = []
        self.index_error: Exception | None = None

    def index(self, request: IndexingRequest) -> IndexingResult:
        self.index_requests.append(request)
        if self.index_error is not None:
            raise self.index_error
        return IndexingResult(self.manifest, 1, 1, 0, (), (), (), ())

    def ask(self, request: AnsweringRequest) -> AnsweringResult:
        self.ask_requests.append(request)
        return AnsweringResult(None, (), (), (), ())


@dataclass
class StubRuntime:
    pipeline: StubPipeline

    def manifest_for(self, source_uri: str) -> IndexManifest:
        assert source_uri
        return self.pipeline.manifest


class RecordingTelemetry(Telemetry):
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def record(self, event: TelemetryEvent) -> None:
        self.events.append(event)


class ContractCorpusView(Protocol):
    manifest: IndexManifest


Send = Callable[[dict[str, object]], Awaitable[None]]
T = TypeVar("T")


def run_synchronous(coroutine: Awaitable[T]) -> T:
    """Drive an immediately-completing ASGI exchange without opening a socket loop."""

    iterator = coroutine.__await__()
    try:
        iterator.send(None)
    except StopIteration as completed:
        return cast(T, completed.value)
    raise AssertionError("the dependency-free ASGI seam unexpectedly suspended")


async def request(
    app: HttpApp,
    method: str,
    path: str,
    *,
    body: object | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], dict[str, object]]:
    encoded = b"" if body is None else json.dumps(body).encode()
    incoming = [{"type": "http.request", "body": encoded, "more_body": False}]
    outgoing: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return incoming.pop(0)

    async def send(message: dict[str, object]) -> None:
        outgoing.append(message)

    raw_headers = [
        (key.lower().encode("ascii"), value.encode("ascii"))
        for key, value in (headers or {}).items()
    ]
    await app(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": raw_headers,
        },
        receive,
        send,
    )
    start, response = outgoing
    response_header_values = cast(list[tuple[bytes, bytes]], start["headers"])
    response_headers = {
        key.decode("ascii"): value.decode("ascii") for key, value in response_header_values
    }
    status = start["status"]
    response_body = response["body"]
    assert isinstance(status, int)
    assert isinstance(response_body, bytes)
    return (
        status,
        response_headers,
        cast(dict[str, object], json.loads(response_body)),
    )


@pytest.fixture
def profile() -> OfflineProfile:
    from ragkit.infrastructure.config import (
        AdapterSettings,
        ComponentSelections,
        RuntimeLimits,
    )
    from ragkit.ports import DocumentFamily

    return OfflineProfile(
        "http-test",
        DocumentFamily.TEXT,
        "tests/fixtures/corpus",
        ComponentSelections(
            "filesystem",
            "text",
            "text",
            "noop",
            "structure_aware",
            "hashing",
            "memory",
            "dense",
            "noop",
            "template",
            "extractive",
            "deterministic",
            "memory",
        ),
        RuntimeLimits(2, 1000, 2, 10, 20, 100, 8, 3, 1000, 100),
        AdapterSettings(),
    )


@pytest.fixture
def app(
    contract_corpus: ContractCorpusView, profile: OfflineProfile
) -> tuple[HttpApp, StubPipeline, RecordingTelemetry]:
    manifest = contract_corpus.manifest
    pipeline = StubPipeline(manifest)
    telemetry = RecordingTelemetry()
    values = iter((10, 20, 30, 40, 50, 60, 70, 80))
    return (
        HttpApp(
            pipeline,
            StubRuntime(pipeline).manifest_for,
            profile,
            telemetry,
            request_id_factory=lambda: "generated-request",
            clock=lambda: next(values),
        ),
        pipeline,
        telemetry,
    )


def test_health_readiness_and_unknown_routes_have_stable_schemas(
    app: tuple[HttpApp, StubPipeline, RecordingTelemetry],
) -> None:
    http, _, _ = app
    health = run_synchronous(request(http, "GET", "/healthz"))
    ready = run_synchronous(request(http, "GET", "/readyz"))
    missing = run_synchronous(request(http, "GET", "/missing"))

    assert health[0] == 200
    assert health[2] == {
        "request_id": "generated-request",
        "schema_version": "v1",
        "status": "ok",
    }
    assert ready[0] == 200 and ready[2]["status"] == "ready"
    assert missing[0] == 404 and missing[2]["error"] == {
        "code": "not_found",
        "message": "route not found",
    }


def test_index_and_ask_delegate_exactly_once_to_application_use_cases(
    app: tuple[HttpApp, StubPipeline, RecordingTelemetry],
) -> None:
    http, pipeline, _ = app
    indexed = run_synchronous(
        request(
            http,
            "POST",
            "/v1/index",
            body={"source_uri": "tests/fixtures/corpus"},
            headers={"X-Request-ID": "client-123"},
        )
    )
    answered = run_synchronous(
        request(
            http,
            "POST",
            "/v1/ask",
            body={"query": "private question", "source_uri": "tests/fixtures/corpus"},
        )
    )

    assert indexed[0] == 200 and indexed[1]["x-request-id"] == "client-123"
    assert indexed[2] == {
        "chunks": 0,
        "diagnostics": [],
        "documents": 1,
        "index_manifest_fingerprint": str(pipeline.manifest.fingerprint),
        "request_id": "client-123",
        "schema_version": "v1",
    }
    assert len(pipeline.index_requests) == 1
    assert pipeline.index_requests[0].source_uri == "tests/fixtures/corpus"
    assert pipeline.index_requests[0].max_chunks == 20
    assert answered[0] == 200
    assert answered[2] == {
        "answer": "",
        "citations": [],
        "diagnostics": [],
        "model_fingerprint": None,
        "request_id": "generated-request",
        "schema_version": "v1",
    }
    assert len(pipeline.ask_requests) == 1
    assert pipeline.ask_requests[0].query == "private question"
    assert pipeline.ask_requests[0].retrieval_top_k == 3


@pytest.mark.parametrize("source_uri", ["/etc/passwd", "tests/fixtures/corpus/../../.."])
def test_http_rejects_sources_outside_the_composed_profile(
    app: tuple[HttpApp, StubPipeline, RecordingTelemetry], source_uri: str
) -> None:
    http, pipeline, _ = app

    status, _, payload = run_synchronous(
        request(http, "POST", "/v1/index", body={"source_uri": source_uri})
    )

    assert status == 400
    assert payload["error"] == {
        "code": "source_not_allowed",
        "message": "source is not allowed by this profile",
    }
    assert pipeline.index_requests == []


def test_request_correlation_and_logs_exclude_raw_content_and_secrets(
    app: tuple[HttpApp, StubPipeline, RecordingTelemetry],
) -> None:
    http, pipeline, telemetry = app
    secret = "sk-secret-value"
    pipeline.index_error = InvalidDomainValueError(f"bad content {secret}")
    status, headers, payload = run_synchronous(
        request(
            http,
            "POST",
            "/v1/index",
            body={"source_uri": "tests/fixtures/corpus"},
            headers={"X-Request-ID": "invalid id with spaces"},
        )
    )

    assert status == 400
    assert headers["x-request-id"] == "generated-request"
    assert payload["error"] == {"code": "invalid_request", "message": "request is invalid"}
    assert secret not in json.dumps(payload)
    assert telemetry.events[-1].outcome is TelemetryOutcome.ERROR
    attributes = {item.name: item.value for item in telemetry.events[-1].attributes}
    assert attributes == {
        "method": "POST",
        "request_id": "generated-request",
        "route": "/v1/index",
        "status_code": 400,
    }
    assert secret not in repr(telemetry.events)
    assert "private" not in repr(telemetry.events)

    pipeline.index_error = RuntimeError(f"provider leaked {secret}")
    unexpected = run_synchronous(
        request(
            http,
            "POST",
            "/v1/index",
            body={"source_uri": "tests/fixtures/corpus"},
        )
    )
    assert unexpected[0] == 500
    assert unexpected[2]["error"] == {
        "code": "internal_error",
        "message": "internal server error",
    }
    assert secret not in repr(telemetry.events)

    missing = run_synchronous(request(http, "GET", f"/{secret}"))
    assert missing[0] == 404
    assert dict((item.name, item.value) for item in telemetry.events[-1].attributes)["route"] == (
        "unmatched"
    )
    assert secret not in repr(telemetry.events)


def test_schema_rejects_unknown_fields_and_oversized_body_before_delegation(
    app: tuple[HttpApp, StubPipeline, RecordingTelemetry],
) -> None:
    http, pipeline, _ = app
    invalid = run_synchronous(
        request(
            http,
            "POST",
            "/v1/ask",
            body={"query": "hello", "source_uri": "fixture", "temperature": 1},
        )
    )
    oversized = run_synchronous(
        request(
            http,
            "POST",
            "/v1/ask",
            body={"query": "x" * 17_000, "source_uri": "fixture"},
        )
    )

    assert invalid[0] == 400
    assert oversized[0] == 413
    assert pipeline.ask_requests == []


def test_server_injects_one_correlated_sink_into_pipeline_and_http(
    monkeypatch: pytest.MonkeyPatch, profile: OfflineProfile
) -> None:
    observed: dict[str, object] = {}
    runtime = cast(OfflineRuntime, object())
    app = cast(HttpApp, object())

    def fake_bootstrap(
        selected: OfflineProfile, *, telemetry: Telemetry | None = None
    ) -> OfflineRuntime:
        observed["bootstrap_profile"] = selected
        observed["bootstrap_telemetry"] = telemetry
        return runtime

    def fake_create_app(
        selected_runtime: OfflineRuntime,
        selected_profile: OfflineProfile,
        telemetry: Telemetry,
    ) -> HttpApp:
        observed["runtime"] = selected_runtime
        observed["http_profile"] = selected_profile
        observed["http_telemetry"] = telemetry
        return app

    class FakeUvicorn:
        @staticmethod
        def run(
            selected_app: object,
            *,
            host: str,
            port: int,
            access_log: bool,
            lifespan: str,
        ) -> None:
            observed["app"] = selected_app
            observed["server"] = (host, port, access_log, lifespan)

    monkeypatch.setattr(server_module, "load_config", lambda _path: profile)
    monkeypatch.setattr(server_module, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(server_module, "create_app", fake_create_app)
    monkeypatch.setattr(server_module, "import_module", lambda _name: FakeUvicorn)

    assert server_module.main(("--config", str(Path("profile.toml")))) == 0
    assert observed["bootstrap_profile"] is profile
    assert observed["http_profile"] is profile
    assert observed["runtime"] is runtime
    assert observed["bootstrap_telemetry"] is observed["http_telemetry"]
    assert observed["app"] is app
    assert observed["server"] == ("127.0.0.1", 8000, False, "off")
