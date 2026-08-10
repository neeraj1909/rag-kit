#!/usr/bin/env python3
"""Benchmark one fixed ragkit CLI query without implying a universal latency SLO."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
from pathlib import Path
from typing import cast

from ragkit.evaluation.benchmark import BenchmarkMetadata, benchmark, content_fingerprint
from ragkit.infrastructure import inspect_profile, load_config


def _build_identifier() -> str:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        return "unavailable"
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        check=False,
        capture_output=True,
        text=True,
    )
    suffix = "+dirty" if status.returncode == 0 and status.stdout.strip() else "+clean"
    return revision.stdout.strip() + suffix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = cast(Path, args.config).resolve()
    source = cast(Path, args.source).resolve()
    query = cast(str, args.query)
    command = (
        "uv",
        "run",
        "ragkit",
        "ask",
        "--config",
        str(config),
        "--source",
        str(source),
        query,
    )

    def operation() -> object:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    workload_fingerprint = "sha256:" + hashlib.sha256(query.encode()).hexdigest()
    profile = load_config(config)
    capabilities = inspect_profile(profile)
    component_fingerprints = cast(dict[str, str], capabilities["selection_fingerprints"])
    metadata = BenchmarkMetadata.current(
        str(profile.fingerprint),
        content_fingerprint(source),
        workload_fingerprint,
        _build_identifier(),
        component_fingerprints,
    )

    def peak_child_bytes() -> int:
        # Linux reports ru_maxrss in KiB. ragkit's supported CI and unattended
        # development environment are Linux.
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024

    result = benchmark(
        operation,
        warmups=cast(int, args.warmups),
        repetitions=cast(int, args.repetitions),
        metadata=metadata,
        memory_bytes=peak_child_bytes,
        memory_scope="cumulative_child_high_water_rss_including_uv_launcher_and_setup",
    )
    output = cast(Path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.to_json() + "\n", encoding="utf-8")
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
