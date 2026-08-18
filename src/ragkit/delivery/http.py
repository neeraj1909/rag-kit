"""A small ASGI delivery adapter over the existing application pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol, cast
from uuid import uuid4

from ragkit.application import (
    AnsweringRequest,
    AnsweringResult,
    IndexingRequest,
    IndexingResult,
)
from ragkit.domain import (
    IndexCompatibilityError,
    IndexManifest,
    InvalidDomainValueError,
    RagkitError,
    UnsupportedCapabilityError,
    locator_to_dict,
)
from ragkit.infrastructure.bootstrap import OfflineRuntime
from ragkit.infrastructure.config import OfflineProfile
from ragkit.ports import Telemetry, TelemetryAttribute, TelemetryEvent, TelemetryOutcome

Scope = Mapping[str, object]
Message = dict[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
_MAX_BODY_BYTES = 16_384


class _Pipeline(Protocol):
    def index(self, request: IndexingRequest) -> IndexingResult: ...

    def ask(self, request: AnsweringRequest) -> AnsweringResult: ...


class _HttpFailure(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.safe_message = message


class HttpApp:
    """Serve health, readiness, indexing, and answering through ASGI 3."""

    def __init__(
        self,
        pipeline: _Pipeline,
        manifest_for: Callable[[str], IndexManifest],
        profile: OfflineProfile,
        telemetry: Telemetry,
        *,
        ready: Callable[[], bool] = lambda: True,
        request_id_factory: Callable[[], str] = lambda: uuid4().hex,
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._pipeline = pipeline
        self._manifest_for = manifest_for
        self._profile = profile
        self._telemetry = telemetry
        self._ready = ready
        self._request_id_factory = request_id_factory
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            raise RuntimeError("HttpApp accepts HTTP ASGI scopes only")
        method = self._required_scope_text(scope, "method").upper()
        path = self._required_scope_text(scope, "path")
        request_id = self._correlation_id(scope)
        started_ns = self._clock()
        outcome = TelemetryOutcome.SUCCESS
        try:
            correlate = getattr(self._telemetry, "correlate", None)
            context = correlate(request_id) if callable(correlate) else nullcontext()
            with context:
                status, payload = await self._dispatch(method, path, receive, request_id)
            if status >= 500:
                outcome = TelemetryOutcome.PARTIAL
        except _HttpFailure as error:
            status = error.status
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(request_id, error.code, error.safe_message)
        except InvalidDomainValueError:
            status = 400
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(request_id, "invalid_request", "request is invalid")
        except IndexCompatibilityError:
            status = 409
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(
                request_id, "index_incompatible", "index is incompatible with this request"
            )
        except UnsupportedCapabilityError:
            status = 503
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(
                request_id, "capability_unavailable", "required capability is unavailable"
            )
        except RagkitError:
            status = 500
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(request_id, "operation_failed", "operation failed")
        except Exception:
            status = 500
            outcome = TelemetryOutcome.ERROR
            payload = self._error_payload(request_id, "internal_error", "internal server error")
        finished_ns = self._clock()
        self._telemetry.record(
            TelemetryEvent(
                "http.request",
                started_ns,
                finished_ns,
                outcome,
                (
                    TelemetryAttribute("method", self._telemetry_method(method)),
                    TelemetryAttribute("request_id", request_id),
                    TelemetryAttribute("route", self._telemetry_route(path)),
                    TelemetryAttribute("status_code", status),
                ),
            )
        )
        await self._send(send, status, request_id, payload)

    async def _dispatch(
        self, method: str, path: str, receive: Receive, request_id: str
    ) -> tuple[int, dict[str, object]]:
        if path == "/healthz":
            self._require_method(method, "GET")
            return 200, self._base_payload(request_id, {"status": "ok"})
        if path == "/readyz":
            self._require_method(method, "GET")
            try:
                ready = self._ready()
            except Exception:
                ready = False
            return (
                (200 if ready else 503),
                self._base_payload(request_id, {"status": "ready" if ready else "not_ready"}),
            )
        if path == "/v1/index":
            self._require_method(method, "POST")
            body = await self._json_body(receive)
            if set(body) not in ({"source_uri"}, {"source_uri", "chunking_strategy"}):
                raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
            source_uri = self._authorized_source(self._body_string(body, "source_uri"))
            self._require_chunking_strategy(body)
            return 200, self._index(source_uri, request_id)
        if path == "/v1/ask":
            self._require_method(method, "POST")
            body = await self._json_body(receive)
            if set(body) not in (
                {"query", "source_uri"},
                {"query", "source_uri", "chunking_strategy"},
            ):
                raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
            query = self._body_string(body, "query")
            source_uri = self._authorized_source(self._body_string(body, "source_uri"))
            self._require_chunking_strategy(body)
            return 200, self._ask(query, source_uri, request_id)
        raise _HttpFailure(404, "not_found", "route not found")

    def _authorized_source(self, requested: str) -> str:
        configured = self._profile.source
        if Path(requested).resolve(strict=False) != Path(configured).resolve(strict=False):
            raise _HttpFailure(400, "source_not_allowed", "source is not allowed by this profile")
        return configured

    def _require_chunking_strategy(self, body: dict[str, object]) -> None:
        requested = body.get("chunking_strategy")
        if requested is None:
            return
        if not isinstance(requested, str) or not requested.strip():
            raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
        accepted = {
            self._profile.settings.chunking_strategy.value,
            self._profile.chunking_policy.strategy.value,
        }
        if requested not in accepted:
            raise _HttpFailure(
                400,
                "chunking_strategy_not_configured",
                "chunking strategy is not configured by this profile",
            )

    def _index(self, source_uri: str, request_id: str) -> dict[str, object]:
        limits = self._profile.limits
        result = self._pipeline.index(
            IndexingRequest(
                source_uri=source_uri,
                manifest=self._manifest_for(source_uri),
                max_assets=limits.max_assets,
                max_bytes_per_asset=limits.max_bytes_per_asset,
                max_documents=limits.max_documents,
                max_parts_per_document=limits.max_parts_per_document,
                max_chunks=limits.max_chunks,
                chunking_policy=self._profile.chunking_policy,
            )
        )
        return self._base_payload(
            request_id,
            {
                "documents": result.document_count,
                "chunks": result.chunk_count,
                "index_manifest_fingerprint": str(result.manifest.fingerprint),
                "diagnostics": [item.code for item in result.diagnostics],
            },
        )

    def _ask(self, query: str, source_uri: str, request_id: str) -> dict[str, object]:
        limits = self._profile.limits
        result = self._pipeline.ask(
            AnsweringRequest(
                query=query,
                expected_manifest=self._manifest_for(source_uri),
                retrieval_top_k=limits.top_k,
                rerank_top_k=limits.top_k,
                max_context_chars=limits.max_context_chars,
                temperature=0.0,
                max_output_tokens=limits.max_output_tokens,
            )
        )
        return self._answer_payload(result, request_id)

    @staticmethod
    def _answer_payload(result: AnsweringResult, request_id: str) -> dict[str, object]:
        generation = result.generation
        return HttpApp._base_payload(
            request_id,
            {
                "answer": "" if generation is None else generation.answer,
                "citations": [
                    {
                        "chunk_id": str(citation.chunk_id),
                        "document_id": str(citation.document_id),
                        "rank": citation.rank,
                        "evidence": [
                            {
                                "asset_id": evidence.asset.asset_id,
                                "source_uri": evidence.asset.uri,
                                "locator": locator_to_dict(evidence.locator),
                                "extractor_fingerprint": str(evidence.extractor),
                                "confidence": evidence.confidence,
                                "notices": [notice.code for notice in evidence.notices],
                            }
                            for evidence in citation.provenance
                        ],
                    }
                    for citation in result.citations
                ],
                "model_fingerprint": None if generation is None else str(generation.model),
                "diagnostics": [item.code for item in result.diagnostics],
            },
        )

    @staticmethod
    async def _json_body(receive: Receive) -> dict[str, object]:
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                raise _HttpFailure(400, "invalid_request", "request body is invalid")
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                raise _HttpFailure(400, "invalid_request", "request body is invalid")
            body.extend(chunk)
            if len(body) > _MAX_BODY_BYTES:
                raise _HttpFailure(413, "request_too_large", "request body is too large")
            if not message.get("more_body", False):
                break
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _HttpFailure(400, "invalid_json", "request body must be valid JSON") from None
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
        return cast(dict[str, object], value)

    @staticmethod
    def _exact_string_body(body: dict[str, object], fields: set[str], field: str) -> str:
        if set(body) != fields:
            raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
        return HttpApp._body_string(body, field)

    @staticmethod
    def _body_string(body: dict[str, object], field: str) -> str:
        value = body.get(field)
        if not isinstance(value, str) or not value.strip():
            raise _HttpFailure(400, "invalid_schema", "request schema is invalid")
        return value

    @staticmethod
    def _require_method(actual: str, expected: str) -> None:
        if actual != expected:
            raise _HttpFailure(405, "method_not_allowed", "method not allowed")

    def _correlation_id(self, scope: Scope) -> str:
        headers = scope.get("headers", ())
        if isinstance(headers, (list, tuple)):
            for item in headers:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and item[0] == b"x-request-id"
                    and isinstance(item[1], bytes)
                ):
                    try:
                        candidate = item[1].decode("ascii")
                    except UnicodeDecodeError:
                        break
                    if _REQUEST_ID.fullmatch(candidate):
                        return candidate
                    break
        generated = self._request_id_factory()
        if not _REQUEST_ID.fullmatch(generated):
            raise RuntimeError("request_id_factory returned an invalid identifier")
        return generated

    @staticmethod
    def _required_scope_text(scope: Scope, name: str) -> str:
        value = scope.get(name)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"HTTP ASGI scope requires {name}")
        return value

    @staticmethod
    def _telemetry_method(method: str) -> str:
        return method if re.fullmatch(r"[A-Z]{1,16}", method) else "INVALID"

    @staticmethod
    def _telemetry_route(path: str) -> str:
        return path if path in {"/healthz", "/readyz", "/v1/index", "/v1/ask"} else "unmatched"

    @staticmethod
    def _base_payload(
        request_id: str, values: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {"schema_version": "v1", "request_id": request_id}
        if values is not None:
            payload.update(values)
        return payload

    @staticmethod
    def _error_payload(request_id: str, code: str, message: str) -> dict[str, object]:
        return HttpApp._base_payload(request_id, {"error": {"code": code, "message": message}})

    @staticmethod
    async def _send(send: Send, status: int, request_id: str, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    runtime: OfflineRuntime,
    profile: OfflineProfile,
    telemetry: Telemetry,
    *,
    ready: Callable[[], bool] = lambda: True,
) -> HttpApp:
    """Construct the ASGI adapter from an already-composed runtime."""

    return HttpApp(runtime.pipeline, runtime.manifest_for, profile, telemetry, ready=ready)
