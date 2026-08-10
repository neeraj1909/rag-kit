"""Bounded content-free in-memory telemetry for the offline profile."""

from __future__ import annotations

from threading import RLock

from ragkit.domain import (
    InvalidDomainValueError,
    LimitExceededError,
    UnsupportedCapabilityError,
)
from ragkit.ports import Telemetry, TelemetryEvent


class InMemoryTelemetry(Telemetry):
    """Retain a bounded immutable event snapshot and reject content-bearing fields."""

    _SENSITIVE_FRAGMENTS = ("prompt", "content", "text", "query", "answer", "secret", "token")

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
        if len(event.attributes) > self._max_attributes:
            raise LimitExceededError("telemetry attribute limit exceeded")
        for attribute in event.attributes:
            lowered = attribute.name.casefold()
            if any(fragment in lowered for fragment in self._SENSITIVE_FRAGMENTS):
                raise UnsupportedCapabilityError(
                    "sensitive telemetry attributes are prohibited",
                    capability="sensitive_telemetry",
                )
            if isinstance(attribute.value, str) and len(attribute.value) > self._max_value_chars:
                raise LimitExceededError("telemetry value length limit exceeded")
        with self._lock:
            if len(self._events) >= self._max_events:
                raise LimitExceededError("telemetry event limit exceeded")
            self._events.append(event)
