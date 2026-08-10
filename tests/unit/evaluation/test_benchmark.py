from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.evaluation.benchmark import BenchmarkMetadata, benchmark, content_fingerprint


class SequenceClock:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = iter(values)

    def __call__(self) -> int:
        return next(self.values)


@pytest.mark.unit
def test_benchmark_uses_warmups_repetitions_and_reports_honest_metadata() -> None:
    calls: list[int] = []
    result = benchmark(
        lambda: calls.append(len(calls)),
        warmups=1,
        repetitions=3,
        clock_ns=SequenceClock((0, 20, 20, 50, 50, 90)),
        memory_bytes=lambda: 4096,
        metadata=BenchmarkMetadata(
            hardware={"cpu": "test-cpu"},
            software={"python": "test-python"},
            config_fingerprint="cfg",
            corpus_fingerprint="corpus",
        ),
    )

    assert len(calls) == 4
    assert result.latency_ns == (20, 30, 40)
    assert result.p50_ns == 30
    assert result.p95_ns == 40
    assert result.throughput_per_second == pytest.approx(33_333_333.333333332)
    assert result.peak_memory_bytes == 4096
    assert result.latency_gate is None
    assert result.latency_gate_passed is None
    assert json.loads(result.to_json())["warmups"] == 1
    assert type(result).from_json(result.to_json()) == result


@pytest.mark.unit
def test_latency_gate_is_only_applied_when_explicit() -> None:
    result = benchmark(
        lambda: None,
        warmups=0,
        repetitions=1,
        clock_ns=SequenceClock((0, 20)),
        memory_bytes=lambda: 0,
        metadata=BenchmarkMetadata({}, {"runner": "test"}, "cfg", "corpus"),
        latency_gate=19,
    )

    assert result.latency_gate_passed is False


@pytest.mark.unit
def test_corpus_fingerprint_ignores_connector_excluded_pycache(tmp_path: Path) -> None:
    (tmp_path / "evidence.txt").write_text("stable evidence", encoding="utf-8")
    before = content_fingerprint(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "generated.pyc").write_bytes(b"interpreter-specific")

    assert content_fingerprint(tmp_path) == before


@pytest.mark.unit
def test_current_metadata_names_package_build_and_component_identity() -> None:
    metadata = BenchmarkMetadata.current(
        "cfg", "corpus", "workload", "abc123+dirty", {"retriever": "cmp_v1_test"}
    )

    assert metadata.software["rag-kit"]
    assert metadata.software["build"] == "abc123+dirty"
    assert metadata.components == {"retriever": "cmp_v1_test"}


@pytest.mark.unit
def test_content_fingerprint_frames_file_names_and_bytes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"bc")
    (right / "ab").write_bytes(b"c")

    assert content_fingerprint(left) != content_fingerprint(right)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "future/v9"),
        ("repetitions", 4),
        ("p95_ns", 999),
        ("throughput_per_second", 1.0),
        ("latency_gate_passed", True),
    ],
)
def test_benchmark_loader_rejects_inconsistent_evidence(field: str, value: object) -> None:
    valid = benchmark(
        lambda: None,
        warmups=0,
        repetitions=1,
        clock_ns=SequenceClock((0, 20)),
        memory_bytes=lambda: 0,
        metadata=BenchmarkMetadata({}, {"rag-kit": "test"}, "cfg", "corpus"),
    )
    payload = json.loads(valid.to_json())
    payload[field] = value

    with pytest.raises(ValueError):
        type(valid).from_json(json.dumps(payload))
