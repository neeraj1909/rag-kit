from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_runner_executes_sparse_and_hybrid_against_fixed_gold(tmp_path: Path) -> None:
    assert "chk_v1_" not in Path("tests/fixtures/evaluation/phase4-text-v1.json").read_text()
    output = tmp_path / "reports"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"

    completed = subprocess.run(
        (
            sys.executable,
            "scripts/evaluate_phase4.py",
            "--dataset",
            "tests/fixtures/evaluation/phase4-text-v1.json",
            "--gold-selectors",
            "tests/fixtures/evaluation/phase4-text-gold-selectors-v1.json",
            "--profile",
            "configs/sparse.toml",
            "--profile",
            "configs/hybrid.toml",
            "--output-dir",
            str(output),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["schema_version"] == "ragkit-phase4-evaluation-run/v1"
    assert summary["dataset_fingerprint"].startswith("sha256:")
    assert [item["profile"] for item in summary["profiles"]] == ["sparse", "hybrid"]
    assert all(item["status"] == "pass" for item in summary["profiles"])

    for profile in ("sparse", "hybrid"):
        report = json.loads((output / f"{profile}-text-evaluation-v1.json").read_text())
        assert report["overall_status"] == "pass"
        assert report["provenance"]["config_fingerprint"].startswith("cmp_v1_")
        assert report["provenance"]["corpus_fingerprint"].startswith("sha256:")
        assert all(
            value.startswith("cmp_v1_") for value in report["provenance"]["components"].values()
        )
        assert report["provenance"]["software"]["rag-kit"]
        thresholds = {item["metric"]: item for item in report["threshold_results"]}
        assert thresholds["recall_at_k"]["passed"] is True
        assert thresholds["citation_precision"]["passed"] is True
        assert thresholds["citation_coverage"]["passed"] is True
        assert thresholds["locator_validity"]["passed"] is True
        assert thresholds["extraction_coverage"]["passed"] is True
        assert all(
            case["metrics"]["recall_at_k"]["value"] == 1.0 for case in report["case_results"]
        )
        assert all(
            case["metrics"]["extraction_coverage"]["value"] == 1.0
            for case in report["case_results"]
        )
        observations = json.loads((output / f"{profile}-text-observations-v1.json").read_text())
        assert observations["component_selections"]["retriever"] == profile
        latency = observations["query_latency_ns"]
        assert latency["sample_count"] == 3
        assert latency["p50"] > 0
        assert latency["p95"] >= latency["p50"]
        assert latency["gated"] is False
        assert all(case["retrieved_evidence_ids"] for case in observations["cases"])
        assert all(case["gold_chunk_locators"] for case in observations["cases"])


def test_runner_records_unavailable_families_without_fabricating_passes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"

    completed = subprocess.run(
        (
            sys.executable,
            "scripts/evaluate_phase4.py",
            "--dataset",
            "tests/fixtures/evaluation/phase4-text-v1.json",
            "--gold-selectors",
            "tests/fixtures/evaluation/phase4-text-gold-selectors-v1.json",
            "--profile",
            "configs/sparse.toml",
            "--output-dir",
            str(output),
            "--family-matrix",
            "tests/fixtures/evaluation/phase4-family-matrix-v1.json",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Traceback" not in completed.stderr, completed.stderr
    assert (output / "five-family-execution-v1.json").is_file(), completed.stderr
    matrix = json.loads((output / "five-family-execution-v1.json").read_text())
    has_failure = any(item["status"] == "fail" for item in matrix["families"].values())
    assert completed.returncode == int(has_failure), completed.stderr
    assert set(matrix["families"]) == {"text", "ocr", "layout", "vision", "media"}
    assert matrix["families"]["text"]["status"] == "pass"
    assert matrix["families"]["ocr"]["status"] in {"pass", "ineligible"}
    assert matrix["families"]["layout"]["status"] in {"pass", "ineligible"}
    assert matrix["families"]["vision"]["status"] == "ineligible"
    assert matrix["families"]["media"]["status"] == "ineligible"
    assert all(
        matrix["families"][name]["evidence"]
        for name in ("text", "ocr", "layout", "vision", "media")
    )
    allowed_requirement_fields = {
        "extra",
        "module",
        "distribution",
        "version",
        "binary",
        "model",
        "model_cached",
        "credential_present",
    }
    assert all(
        set(requirement) == allowed_requirement_fields
        for family in matrix["families"].values()
        for requirement in family["requirements"]
    )
    assert matrix["families"]["text"]["metrics"]["retrieval_recall"]["value"] == 1.0
    assert matrix["families"]["text"]["metrics"]["extraction_coverage"]["value"] == 1.0
    for family in ("ocr", "layout"):
        if matrix["families"][family]["status"] == "pass":
            assert all(
                metric["value"] == 1.0 for metric in matrix["families"][family]["metrics"].values()
            )
    for family in ("vision", "media"):
        evidence = matrix["families"][family]["evidence"]
        assert any(
            marker in evidence
            for marker in (
                "RAGKIT_RUN_MODEL_INTEGRATION=1",
                "optional modules unavailable",
                "revision-pinned local model unavailable",
            )
        )
        assert all(
            metric["value"] is None for metric in matrix["families"][family]["metrics"].values()
        )
