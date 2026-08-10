"""Small deterministic benchmark harness with injectable measurement seams."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def content_fingerprint(path: Path) -> str:
    """Fingerprint the regular-file inventory selected by the filesystem connector."""

    if path.is_symlink():
        raise ValueError("benchmark input must not be a symbolic link")
    if path.is_dir():
        inventory = tuple(
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and "__pycache__" not in item.relative_to(path).parts
        )
    else:
        inventory = (path,)
    digest = hashlib.sha256()
    for item in inventory:
        if item.is_symlink():
            raise ValueError("benchmark inventory must not contain symbolic links")
        name = str(item.relative_to(path) if path.is_dir() else item.name).encode()
        content = item.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    hardware: dict[str, str]
    software: dict[str, str]
    config_fingerprint: str
    corpus_fingerprint: str
    workload_fingerprint: str = "unspecified"
    components: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.config_fingerprint
            or not self.corpus_fingerprint
            or not self.workload_fingerprint
        ):
            raise ValueError("benchmark fingerprints must not be empty")
        if not self.software:
            raise ValueError("benchmark software provenance must not be empty")
        if not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for values in (self.hardware, self.software, self.components)
            for key, value in values.items()
        ):
            raise ValueError("benchmark metadata maps require non-empty strings")

    @classmethod
    def current(
        cls,
        config_fingerprint: str,
        corpus_fingerprint: str,
        workload_fingerprint: str = "unspecified",
        build_identifier: str = "unavailable",
        component_fingerprints: dict[str, str] | None = None,
    ) -> BenchmarkMetadata:
        try:
            ragkit_version = version("rag-kit")
        except PackageNotFoundError:
            ragkit_version = "uninstalled"
        return cls(
            {
                "machine": platform.machine() or "unknown",
                "processor": platform.processor() or "unknown",
            },
            {
                "python": platform.python_version(),
                "implementation": sys.implementation.name,
                "rag-kit": ragkit_version,
                "build": build_identifier,
            },
            config_fingerprint,
            corpus_fingerprint,
            workload_fingerprint,
            dict(component_fingerprints or {}),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    schema_version: str
    warmups: int
    repetitions: int
    latency_ns: tuple[int, ...]
    p50_ns: int
    p95_ns: int
    throughput_per_second: float
    peak_memory_bytes: int
    metadata: BenchmarkMetadata
    memory_scope: str = "python_tracemalloc"
    latency_gate: int | None = None
    latency_gate_passed: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "ragkit-benchmark-report/v1":
            raise ValueError("unsupported benchmark report schema")
        if type(self.warmups) is not int or self.warmups < 0:
            raise ValueError("benchmark warmups must be a non-negative integer")
        if type(self.repetitions) is not int or self.repetitions <= 0:
            raise ValueError("benchmark repetitions must be a positive integer")
        if len(self.latency_ns) != self.repetitions or any(
            type(value) is not int or value <= 0 for value in self.latency_ns
        ):
            raise ValueError("benchmark latency samples must be positive and aligned")
        if self.p50_ns != _percentile(self.latency_ns, 0.5) or self.p95_ns != _percentile(
            self.latency_ns, 0.95
        ):
            raise ValueError("benchmark percentiles do not match latency samples")
        expected_throughput = self.repetitions / (sum(self.latency_ns) / 1_000_000_000)
        if not math.isfinite(self.throughput_per_second) or not math.isclose(
            self.throughput_per_second, expected_throughput, rel_tol=1e-12
        ):
            raise ValueError("benchmark throughput does not match latency samples")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise ValueError("benchmark peak memory must be a non-negative integer")
        if not self.memory_scope:
            raise ValueError("benchmark memory scope must not be empty")
        if self.latency_gate is None:
            if self.latency_gate_passed is not None:
                raise ValueError("ungated benchmarks cannot report a gate result")
        elif (
            type(self.latency_gate) is not int
            or self.latency_gate <= 0
            or self.latency_gate_passed != (self.p95_ns <= self.latency_gate)
        ):
            raise ValueError("benchmark latency gate result is inconsistent")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkResult:
        value: dict[str, Any] = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "warmups",
            "repetitions",
            "latency_ns",
            "p50_ns",
            "p95_ns",
            "throughput_per_second",
            "peak_memory_bytes",
            "metadata",
            "memory_scope",
            "latency_gate",
            "latency_gate_passed",
        }:
            raise ValueError("benchmark report has unexpected fields")
        metadata = value["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {
            "hardware",
            "software",
            "config_fingerprint",
            "corpus_fingerprint",
            "workload_fingerprint",
            "components",
        }:
            raise ValueError("benchmark metadata has unexpected fields")
        value["latency_ns"] = tuple(value["latency_ns"])
        value["metadata"] = BenchmarkMetadata(**metadata)
        return cls(**value)


def _percentile(values: tuple[int, ...], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def benchmark(
    operation: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
    metadata: BenchmarkMetadata,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    memory_bytes: Callable[[], int] | None = None,
    memory_scope: str | None = None,
    latency_gate: int | None = None,
) -> BenchmarkResult:
    """Run a fixed operation; latency is informational unless a gate is explicitly supplied."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions positive")
    if latency_gate is not None and latency_gate <= 0:
        raise ValueError("latency gate must be positive when supplied")
    for _ in range(warmups):
        operation()
    started_tracing = memory_bytes is None
    if started_tracing:
        tracemalloc.start()
    durations: list[int] = []
    for _ in range(repetitions):
        started = clock_ns()
        operation()
        duration = clock_ns() - started
        if duration < 0:
            raise ValueError("benchmark clock must be monotonic")
        durations.append(duration)
    if memory_bytes is None:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak_memory = memory_bytes()
    latency = tuple(durations)
    total_seconds = sum(latency) / 1_000_000_000
    throughput = repetitions / total_seconds if total_seconds else math.inf
    if not math.isfinite(throughput):
        raise ValueError("benchmark duration is too small to report finite throughput")
    p95 = _percentile(latency, 0.95)
    return BenchmarkResult(
        schema_version="ragkit-benchmark-report/v1",
        warmups=warmups,
        repetitions=repetitions,
        latency_ns=latency,
        p50_ns=_percentile(latency, 0.5),
        p95_ns=p95,
        throughput_per_second=throughput,
        peak_memory_bytes=peak_memory,
        metadata=metadata,
        memory_scope=(
            memory_scope or ("python_tracemalloc" if memory_bytes is None else "caller_supplied")
        ),
        latency_gate=latency_gate,
        latency_gate_passed=None if latency_gate is None else p95 <= latency_gate,
    )
