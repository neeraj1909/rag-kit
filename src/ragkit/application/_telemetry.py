"""Shared, content-free application timing records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ragkit.domain import ComponentFingerprint, InvalidDomainValueError
from ragkit.ports import Telemetry, TelemetryAttribute, TelemetryEvent, TelemetryOutcome

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PipelineDiagnostic:
    """A stable machine-readable explanation for an intentional early stop."""

    stage: str
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.stage or not self.code or not self.message:
            raise InvalidDomainValueError("diagnostic fields must not be empty")


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Sanitized duration evidence returned by one application invocation."""

    operation: str
    duration_ns: int
    outcome: TelemetryOutcome

    def __post_init__(self) -> None:
        if not self.operation or self.duration_ns < 0:
            raise InvalidDomainValueError("stage timing must have an operation and duration")


def invoke_timed(
    operation: str,
    call: Callable[[], T],
    telemetry: Telemetry,
    clock: Callable[[], int],
    timings: list[StageTiming],
    *,
    component: object,
    count: Callable[[T], int] | None = None,
) -> T:
    """Run one injected capability and record bounded operational metadata."""

    started_ns = clock()
    fingerprint = _component_fingerprint(component)
    try:
        result = call()
    except Exception as error:
        finished_ns = clock()
        event = TelemetryEvent(
            operation,
            started_ns,
            finished_ns,
            TelemetryOutcome.ERROR,
            _attributes(fingerprint, 0, _error_category(error)),
        )
        telemetry.record(event)
        timings.append(StageTiming(operation, finished_ns - started_ns, event.outcome))
        raise
    finished_ns = clock()
    event = TelemetryEvent(
        operation,
        started_ns,
        finished_ns,
        TelemetryOutcome.SUCCESS,
        _attributes(
            fingerprint,
            _result_count(result) if count is None else count(result),
            "none",
        ),
    )
    telemetry.record(event)
    timings.append(StageTiming(operation, finished_ns - started_ns, event.outcome))
    return result


def _component_fingerprint(component: object) -> ComponentFingerprint:
    declared = getattr(component, "fingerprint", None)
    if isinstance(declared, ComponentFingerprint):
        return declared
    implementation = f"{type(component).__module__}.{type(component).__qualname__}"
    return ComponentFingerprint.create("runtime_component", implementation, {"identity_version": 1})


def _result_count(result: object) -> int:
    if isinstance(result, tuple):
        return len(result)
    return 0 if result is None else 1


def _error_category(error: Exception) -> str:
    module = type(error).__module__
    if module.startswith("ragkit."):
        return type(error).__name__
    return "unexpected_error"


def _attributes(
    fingerprint: ComponentFingerprint, result_count: int, error_category: str
) -> tuple[TelemetryAttribute, ...]:
    return (
        TelemetryAttribute("component_fingerprint", str(fingerprint)),
        TelemetryAttribute("result_count", result_count),
        TelemetryAttribute("error_category", error_category),
    )
