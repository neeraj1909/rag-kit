from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from ragkit.adapters.layout import LayoutDocumentExtractor
from ragkit.adapters.ocr import OcrDocumentExtractor
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    LayoutContent,
    MissingDependencyError,
    OcrContent,
    RelationKind,
)
from ragkit.ports import AcquiredAsset, AssetClassification, DocumentFamily, ExtractionRequest

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _request(path: Path, media_type: str, family: DocumentFamily) -> ExtractionRequest:
    content = path.read_bytes()
    reference = AssetRef(
        path.name, media_type, sha256(content).hexdigest(), path.as_uri(), len(content)
    )
    from ragkit.domain import ComponentFingerprint

    classification = AssetClassification(
        reference.asset_id,
        family,
        1.0,
        ComponentFingerprint.create("classifier", "fixture", {"version": 1}),
    )
    return ExtractionRequest((AcquiredAsset(reference, content),), (classification,), 1)


@pytest.mark.integration
@pytest.mark.modality_integration
@pytest.mark.parametrize(
    ("name", "media_type", "expected"),
    [
        ("annual-report.pdf", "application/pdf", ("Revenue", "2026")),
        (
            "pricing.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ("Supplier pricing", "Standard"),
        ),
        (
            "pricing.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ("Summary", "Standard"),
        ),
    ],
)
def test_layout_real_fixtures(name: str, media_type: str, expected: tuple[str, str]) -> None:
    dependency = {
        "annual-report.pdf": "pdfplumber",
        "pricing.pptx": "pptx",
        "pricing.xlsx": "openpyxl",
    }[name]
    pytest.importorskip(dependency)
    document = LayoutDocumentExtractor().extract(
        _request(FIXTURES / "layout" / name, media_type, DocumentFamily.LAYOUT)
    )[0]

    layout_parts = tuple(part for part in document.parts if isinstance(part, LayoutContent))
    searchable = " ".join(part.text for part in layout_parts)
    assert all(item in searchable for item in expected)
    assert all(
        isinstance(part.provenance.locator, (BoxLocator, CellLocator)) for part in document.parts
    )
    if name == "annual-report.pdf":
        assert {
            part.provenance.locator.page
            for part in layout_parts
            if isinstance(part.provenance.locator, BoxLocator)
        } == {0, 1}
    if name == "pricing.pptx":
        merged_header = next(part for part in layout_parts if part.text == "Supplier plans")
        assert isinstance(merged_header.provenance.locator, CellLocator)
        assert merged_header.provenance.locator.sheet.startswith("slide:0:table:")
        assert merged_header.provenance.locator.start_row == 0
        assert merged_header.provenance.locator.start_column == 0
        assert merged_header.provenance.locator.end_row == 0
        assert merged_header.provenance.locator.end_column == 1
    if name == "pricing.xlsx":
        assert {
            part.provenance.locator.sheet
            for part in document.parts
            if isinstance(part.provenance.locator, CellLocator)
        } == {"Summary", "Details"}


@pytest.mark.integration
@pytest.mark.modality_integration
def test_ocr_real_printed_fixture_or_actionable_tesseract_error() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("pytesseract")
    request = _request(FIXTURES / "ocr" / "printed-claim.png", "image/png", DocumentFamily.OCR)
    try:
        document = OcrDocumentExtractor().extract(request)[0]
    except MissingDependencyError as error:
        assert "Tesseract executable" in str(error)
        return
    ocr_parts = tuple(part for part in document.parts if isinstance(part, OcrContent))
    assert "CLAIM" in " ".join(part.text.upper() for part in ocr_parts)
    assert all(isinstance(part.provenance.locator, BoxLocator) for part in document.parts)


@pytest.mark.integration
@pytest.mark.modality_integration
def test_ocr_real_form_marks_degradation_and_retains_field_links() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("pytesseract")
    request = _request(
        FIXTURES / "ocr" / "handwritten-form.png",
        "image/png",
        DocumentFamily.OCR,
    )
    document = OcrDocumentExtractor(content_mode="form").extract(request)[0]

    assert document.metadata["degraded_mode"] == "form"
    assert all(part.provenance.confidence is not None for part in document.parts)
    assert any(
        relation.kind is RelationKind.LABELED_BY
        for part in document.parts
        for relation in part.relations
    )
