from __future__ import annotations

import json
import os

import pytest

from ragkit.cli.main import main

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    ("config", "query", "locator_kind", "dependency"),
    [
        ("configs/ocr.toml", "What is the claim ID?", "box", "pytesseract"),
        ("configs/layout.toml", "What is the standard price?", "cell", "openpyxl"),
    ],
)
def test_local_document_family_answers_retain_exact_citations(
    config: str,
    query: str,
    locator_kind: str,
    dependency: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip(dependency)

    assert main(["ask", "--config", config, query]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["answer"]
    assert result["citations"]
    assert result["citations"][0]["evidence"][0]["locator"]["kind"] == locator_kind


@pytest.mark.modality_integration
@pytest.mark.parametrize(
    ("config", "query", "locator_kinds"),
    [
        ("configs/vision.toml", "What equipment is visible?", {"box"}),
        ("configs/mixed-image.toml", "What labels and equipment are visible?", {"box"}),
        ("configs/media.toml", "What action was reported?", {"time"}),
    ],
)
def test_provisioned_model_families_run_end_to_end(
    config: str,
    query: str,
    locator_kinds: set[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("explicit reviewed model provisioning is required")

    assert main(["ask", "--config", config, query]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["answer"]
    evidence = [item for citation in result["citations"] for item in citation["evidence"]]
    assert evidence
    assert {item["locator"]["kind"] for item in evidence} <= locator_kinds
