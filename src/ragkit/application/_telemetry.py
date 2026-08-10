"""Shared, content-free application timing records."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from ragkit.domain import InvalidDomainValueError
from ragkit.ports import Telemetry, TelemetryEvent, TelemetryOutcome

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
) -> T:
    """Run one injected capability and record content-free timing metadata."""

    started_ns = clock()
    try:
        result = call()
    except Exception:
        finished_ns = clock()
        event = TelemetryEvent(operation, started_ns, finished_ns, TelemetryOutcome.ERROR)
        telemetry.record(event)
        timings.append(StageTiming(operation, finished_ns - started_ns, event.outcome))
        raise
    finished_ns = clock()
    event = TelemetryEvent(operation, started_ns, finished_ns, TelemetryOutcome.SUCCESS)
    telemetry.record(event)
    timings.append(StageTiming(operation, finished_ns - started_ns, event.outcome))
    return result
