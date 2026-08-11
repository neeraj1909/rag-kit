from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

from ragkit.adapters import InMemoryTelemetry, JsonLinesTelemetry, RequestCorrelatedTelemetry
from ragkit.domain import InvalidDomainValueError, LimitExceededError, UnsupportedCapabilityError
from ragkit.ports import TelemetryAttribute, TelemetryEvent, TelemetryOutcome

pytestmark = pytest.mark.unit


def event(*attributes: TelemetryAttribute) -> TelemetryEvent:
    return TelemetryEvent("http.request", 10, 25, TelemetryOutcome.SUCCESS, attributes)


def test_json_lines_telemetry_emits_bounded_structured_metadata() -> None:
    stream = io.StringIO()
    sink = JsonLinesTelemetry(stream)

    sink.record(
        event(
            TelemetryAttribute("request_id", "request-1"),
            TelemetryAttribute("status_code", 200),
        )
    )

    assert json.loads(stream.getvalue()) == {
        "attributes": {"request_id": "request-1", "status_code": 200},
        "duration_ns": 15,
        "operation": "http.request",
        "outcome": "success",
    }


@pytest.mark.parametrize("operation", ["ask.prompt", "ask.generate", "index.extract"])
def test_telemetry_accepts_stable_application_stage_names(operation: str) -> None:
    sink = InMemoryTelemetry()

    sink.record(TelemetryEvent(operation, 10, 25, TelemetryOutcome.SUCCESS))

    assert sink.events[0].operation == operation


def test_request_correlation_is_scoped_and_concurrency_safe() -> None:
    delegate = InMemoryTelemetry()
    sink = RequestCorrelatedTelemetry(delegate)

    def record(request_id: str) -> None:
        with sink.correlate(request_id):
            sink.record(TelemetryEvent("ask.retrieve", 10, 25, TelemetryOutcome.SUCCESS))

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(record, ("request-one", "request-two")))
    sink.record(TelemetryEvent("background.task", 30, 40, TelemetryOutcome.SUCCESS))

    correlated = {
        next(item.value for item in event.attributes if item.name == "request_id")
        for event in delegate.events[:2]
    }
    assert correlated == {"request-one", "request-two"}
    assert delegate.events[2].attributes == ()


@pytest.mark.parametrize(
    "name",
    ["prompt", "raw_content", "query", "answer", "authorization", "api_key", "password"],
)
def test_all_telemetry_sinks_reject_sensitive_attribute_names(name: str) -> None:
    value = event(TelemetryAttribute(name, "must-not-appear"))

    with pytest.raises(UnsupportedCapabilityError, match="sensitive"):
        InMemoryTelemetry().record(value)
    with pytest.raises(UnsupportedCapabilityError, match="sensitive"):
        JsonLinesTelemetry(io.StringIO()).record(value)


def test_telemetry_rejects_duplicates_and_bounds_before_writing() -> None:
    stream = io.StringIO()
    sink = JsonLinesTelemetry(stream, max_attributes=1, max_value_chars=32)

    with pytest.raises(LimitExceededError, match="attribute limit"):
        sink.record(event(TelemetryAttribute("one", 1), TelemetryAttribute("two", 2)))
    with pytest.raises(LimitExceededError, match="operation length"):
        JsonLinesTelemetry(stream, max_value_chars=8).record(
            TelemetryEvent("operation-too-long", 1, 2, TelemetryOutcome.SUCCESS)
        )
    with pytest.raises(InvalidDomainValueError, match="unique"):
        JsonLinesTelemetry(stream).record(
            event(TelemetryAttribute("route", "a"), TelemetryAttribute("route", "b"))
        )
    with pytest.raises(InvalidDomainValueError, match="bounded scalars"):
        JsonLinesTelemetry(stream).record(
            event(TelemetryAttribute("route", cast(str, {"raw": "content"})))
        )

    assert stream.getvalue() == ""
