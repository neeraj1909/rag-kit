"""Bounded content-free telemetry sinks."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from threading import RLock
from typing import TextIO

from ragkit.domain import (
    InvalidDomainValueError,
    LimitExceededError,
    UnsupportedCapabilityError,
)
from ragkit.ports import Telemetry, TelemetryAttribute, TelemetryEvent

_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")


class InMemoryTelemetry(Telemetry):
    """Retain a bounded immutable event snapshot and reject content-bearing fields."""

    _SENSITIVE_FRAGMENTS = (
        "answer",
        "api_key",
        "apikey",
        "authorization",
        "content",
        "credential",
        "password",
        "prompt",
        "query",
        "secret",
        "token",
    )

    def __init__(
        self,
        max_events: int = 1_000,
        max_attributes: int = 16,
        max_value_chars: int = 256,
    ) -> None:
        if min(max_events, max_attributes, max_value_chars) <= 0:
            raise InvalidDomainValueError("telemetry limits must be positive")
        self._max_events = max_events
        self._max_attributes = max_attributes
        self._max_value_chars = max_value_chars
        self._events: list[TelemetryEvent] = []
        self._lock = RLock()

    @property
    def events(self) -> tuple[TelemetryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def record(self, event: TelemetryEvent) -> None:
        _validate_event(event, self._max_attributes, self._max_value_chars)
        with self._lock:
            if len(self._events) >= self._max_events:
                raise LimitExceededError("telemetry event limit exceeded")
            self._events.append(event)


class JsonLinesTelemetry(Telemetry):
    """Write one deterministic sanitized JSON object per event."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        max_attributes: int = 16,
        max_value_chars: int = 256,
    ) -> None:
        if min(max_attributes, max_value_chars) <= 0:
            raise InvalidDomainValueError("telemetry limits must be positive")
        self._stream = sys.stderr if stream is None else stream
        self._max_attributes = max_attributes
        self._max_value_chars = max_value_chars
        self._lock = RLock()

    def record(self, event: TelemetryEvent) -> None:
        _validate_event(event, self._max_attributes, self._max_value_chars)
        payload = {
            "attributes": {item.name: item.value for item in event.attributes},
            "duration_ns": event.finished_ns - event.started_ns,
            "operation": event.operation,
            "outcome": event.outcome.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._stream.write(encoded + "\n")
            self._stream.flush()


class RequestCorrelatedTelemetry(Telemetry):
    """Add one request ID to nested events without sharing concurrent state."""

    def __init__(self, delegate: Telemetry) -> None:
        self._delegate = delegate
        self._request_id: ContextVar[str | None] = ContextVar("ragkit_request_id", default=None)

    @contextmanager
    def correlate(self, request_id: str) -> Iterator[None]:
        if _REQUEST_ID.fullmatch(request_id) is None:
            raise InvalidDomainValueError("request ID must be a bounded stable identifier")
        token: Token[str | None] = self._request_id.set(request_id)
        try:
            yield
        finally:
            self._request_id.reset(token)

    def record(self, event: TelemetryEvent) -> None:
        request_id = self._request_id.get()
        if request_id is None or any(item.name == "request_id" for item in event.attributes):
            self._delegate.record(event)
            return
        self._delegate.record(
            TelemetryEvent(
                event.operation,
                event.started_ns,
                event.finished_ns,
                event.outcome,
                (*event.attributes, TelemetryAttribute("request_id", request_id)),
            )
        )


def _validate_event(event: TelemetryEvent, max_attributes: int, max_value_chars: int) -> None:
    if len(event.operation) > max_value_chars:
        raise LimitExceededError("telemetry operation length limit exceeded")
    if _FIELD_NAME.fullmatch(event.operation) is None:
        raise InvalidDomainValueError("telemetry operation must be a stable field name")
    if len(event.attributes) > max_attributes:
        raise LimitExceededError("telemetry attribute limit exceeded")
    names: set[str] = set()
    for attribute in event.attributes:
        lowered = attribute.name.casefold()
        if attribute.name in names:
            raise InvalidDomainValueError("telemetry attribute names must be unique")
        names.add(attribute.name)
        if len(attribute.name) > max_value_chars:
            raise LimitExceededError("telemetry attribute name length limit exceeded")
        if _FIELD_NAME.fullmatch(attribute.name) is None:
            raise InvalidDomainValueError("telemetry attribute must be a stable field name")
        if any(fragment in lowered for fragment in InMemoryTelemetry._SENSITIVE_FRAGMENTS):
            raise UnsupportedCapabilityError(
                "sensitive telemetry attributes are prohibited",
                capability="sensitive_telemetry",
            )
        if isinstance(attribute.value, str) and len(attribute.value) > max_value_chars:
            raise LimitExceededError("telemetry value length limit exceeded")
        if attribute.value is not None and type(attribute.value) not in {str, int, float, bool}:
            raise InvalidDomainValueError("telemetry attribute values must be bounded scalars")
