from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from ragkit.adapters.layout import LayoutDocumentExtractor
from ragkit.adapters.ocr import OcrDocumentExtractor, OcrToken
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    CellLocator,
    ComponentFingerprint,
    IntegrityError,
    LayoutContent,
    LimitExceededError,
    MissingDependencyError,
    OcrContent,
    PartialExtractionError,
    PartRelation,
    RelationKind,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    DocumentFamily,
    ExtractionRequest,
)


def _request(content: bytes, media_type: str, family: DocumentFamily) -> ExtractionRequest:
    reference = AssetRef(
        "fixture",
        media_type,
        sha256(content).hexdigest(),
        "memory://fixture",
        len(content),
    )
    asset = AcquiredAsset(reference, content)
    classification = AssetClassification(
        reference.asset_id,
        family,
        1.0,
        _fingerprint(),
    )
    return ExtractionRequest((asset,), (classification,), 1)


def _fingerprint() -> ComponentFingerprint:
    return ComponentFingerprint.create("classifier", "fixture", {"version": 1})


@dataclass
class _FakeOcrEngine:
    tokens: tuple[OcrToken, ...]

    def recognize(
        self, image: object, *, language: str, timeout_seconds: float
    ) -> tuple[OcrToken, ...]:
        del image, language, timeout_seconds
        return self.tokens


def test_ocr_preserves_normalized_boxes_confidence_and_low_confidence_notice() -> None:
    pytest.importorskip("PIL")
    output = BytesIO()
    image = pytest.importorskip("PIL.Image")
    image.new("RGB", (200, 100), "white").save(output, "PNG")
    engine = _FakeOcrEngine(
        (
            OcrToken("Claim", 10, 20, 50, 20, 96.0),
            OcrToken("7B", 70, 20, 30, 20, 42.0),
        )
    )

    document = OcrDocumentExtractor(engine=engine, low_confidence_threshold=0.6).extract(
        _request(output.getvalue(), "image/png", DocumentFamily.OCR)
    )[0]

    assert all(isinstance(part, OcrContent) for part in document.parts)
    ocr_parts = tuple(part for part in document.parts if isinstance(part, OcrContent))
    assert [part.text for part in ocr_parts] == ["Claim", "7B"]
    first = document.parts[0]
    assert first.provenance.confidence == pytest.approx(0.96)
    assert first.provenance.locator == BoxLocator(0, 0.05, 0.2, 0.3, 0.4)
    assert [notice.code for notice in document.parts[1].provenance.notices] == [
        "low_ocr_confidence"
    ]


def test_ocr_handwriting_and_forms_are_explicitly_degraded() -> None:
    pytest.importorskip("PIL")
    output = BytesIO()
    image = pytest.importorskip("PIL.Image")
    image.new("L", (20, 20), 255).save(output, "PNG")
    extractor = OcrDocumentExtractor(
        engine=_FakeOcrEngine(
            (
                OcrToken("Name", 0, 0, 8, 10, 85.0),
                OcrToken("Alice", 10, 0, 8, 10, 72.0),
            )
        ),
        content_mode="form",
    )

    document = extractor.extract(_request(output.getvalue(), "image/png", DocumentFamily.OCR))[0]

    codes = {notice.code for notice in document.parts[0].provenance.notices}
    assert codes == {"form_structure_unverified"}
    assert document.metadata["degraded_mode"] == "form"
    assert document.parts[1].relations == (
        PartRelation(
            document.parts[1].part_id,
            document.parts[0].part_id,
            RelationKind.LABELED_BY,
        ),
    )


def test_ocr_rejects_empty_corrupt_and_oversized_rasters() -> None:
    pytest.importorskip("PIL")
    blank = BytesIO()
    image = pytest.importorskip("PIL.Image")
    image.new("RGB", (10, 10), "white").save(blank, "PNG")
    with pytest.raises(PartialExtractionError, match="no OCR text"):
        OcrDocumentExtractor(engine=_FakeOcrEngine(())).extract(
            _request(blank.getvalue(), "image/png", DocumentFamily.OCR)
        )
    with pytest.raises(IntegrityError, match="decode"):
        OcrDocumentExtractor(engine=_FakeOcrEngine(())).extract(
            _request(b"not-an-image", "image/png", DocumentFamily.OCR)
        )
    with pytest.raises(LimitExceededError, match="pixel"):
        OcrDocumentExtractor(engine=_FakeOcrEngine(()), max_pixels=50).extract(
            _request(blank.getvalue(), "image/png", DocumentFamily.OCR)
        )


def test_default_ocr_engine_reports_missing_tesseract_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PIL")
    pytesseract = pytest.importorskip("pytesseract")

    output = BytesIO()
    image = pytest.importorskip("PIL.Image")
    image.new("RGB", (10, 10), "white").save(output, "PNG")

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_data", fail)
    with pytest.raises(MissingDependencyError, match="Tesseract executable"):
        OcrDocumentExtractor().extract(_request(output.getvalue(), "image/png", DocumentFamily.OCR))


def test_ocr_rasterizes_each_pdf_page_with_bounded_page_provenance() -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("pypdfium2")
    content = (FIXTURES / "layout" / "annual-report.pdf").read_bytes()
    extractor = OcrDocumentExtractor(
        engine=_FakeOcrEngine((OcrToken("page", 10, 10, 20, 10, 90.0),)),
        target_dpi=72,
    )

    document = extractor.extract(_request(content, "application/pdf", DocumentFamily.OCR))[0]

    assert [
        part.provenance.locator.page
        for part in document.parts
        if isinstance(part, OcrContent) and isinstance(part.provenance.locator, BoxLocator)
    ] == [0, 1]
    with pytest.raises(LimitExceededError, match="page count"):
        OcrDocumentExtractor(engine=_FakeOcrEngine(()), max_pages=1).extract(
            _request(content, "application/pdf", DocumentFamily.OCR)
        )


def test_layout_xlsx_retains_sheet_cells_merges_formulas_and_relationships() -> None:
    pytest.importorskip("openpyxl")
    content = (FIXTURES / "layout" / "pricing.xlsx").read_bytes()

    document = LayoutDocumentExtractor().extract(
        _request(
            content,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            DocumentFamily.LAYOUT,
        )
    )[0]

    cells = {
        (
            part.provenance.locator.sheet,
            part.provenance.locator.start_row,
            part.provenance.locator.start_column,
        ): part
        for part in document.parts
        if isinstance(part, LayoutContent) and isinstance(part.provenance.locator, CellLocator)
    }
    assert cells[("Summary", 0, 0)].provenance.locator == CellLocator("Summary", 0, 0, 0, 1)
    assert cells[("Summary", 2, 1)].text.startswith("=SUM(")
    assert any(
        notice.code == "formula_cached_value_unavailable"
        for notice in cells[("Summary", 2, 1)].provenance.notices
    )
    assert {locator[0] for locator in cells} == {"Summary", "Details"}
    assert any(part.relations for part in document.parts[1:])
    detail_data = next(part for part in document.parts if part.part_id == "xlsx-Details-r1-c0")
    assert any(
        relation.kind is RelationKind.LABELED_BY and relation.target_part_id == "xlsx-Details-r0-c0"
        for relation in detail_data.relations
    )
    first_detail = next(part for part in document.parts if part.part_id == "xlsx-Details-r0-c0")
    assert not any(relation.kind is RelationKind.CONTINUES for relation in first_detail.relations)


def test_layout_rejects_corrupt_input_and_limits() -> None:
    pytest.importorskip("openpyxl")
    with pytest.raises(IntegrityError, match=r"archive|workbook"):
        LayoutDocumentExtractor().extract(
            _request(
                b"not-xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                DocumentFamily.LAYOUT,
            )
        )
    content = (FIXTURES / "layout" / "pricing.xlsx").read_bytes()
    with pytest.raises(LimitExceededError, match="sheet"):
        LayoutDocumentExtractor(max_sheets=1).extract(
            _request(
                content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                DocumentFamily.LAYOUT,
            )
        )


def test_layout_fails_closed_for_scans_and_unprocessed_slide_images() -> None:
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pptx")
    scanned = (FIXTURES / "layout" / "scanned-page.pdf").read_bytes()
    with pytest.raises(PartialExtractionError, match="vision/OCR"):
        LayoutDocumentExtractor().extract(
            _request(scanned, "application/pdf", DocumentFamily.LAYOUT)
        )

    image_slide = (FIXTURES / "layout" / "image-slide.pptx").read_bytes()
    with pytest.raises(PartialExtractionError, match="route it to vision"):
        LayoutDocumentExtractor().extract(
            _request(
                image_slide,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                DocumentFamily.LAYOUT,
            )
        )


def test_layout_enforces_pdf_slide_and_populated_cell_limits() -> None:
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pptx")
    pytest.importorskip("openpyxl")
    pdf = (FIXTURES / "layout" / "annual-report.pdf").read_bytes()
    with pytest.raises(LimitExceededError, match="PDF page count"):
        LayoutDocumentExtractor(max_pages=1).extract(
            _request(pdf, "application/pdf", DocumentFamily.LAYOUT)
        )

    presentation = (FIXTURES / "layout" / "pricing.pptx").read_bytes()
    with pytest.raises(LimitExceededError, match="slide count"):
        LayoutDocumentExtractor(max_slides=1).extract(
            _request(
                presentation,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                DocumentFamily.LAYOUT,
            )
        )

    workbook = (FIXTURES / "layout" / "pricing.xlsx").read_bytes()
    with pytest.raises(LimitExceededError, match="populated cell count"):
        LayoutDocumentExtractor(max_cells=1).extract(
            _request(
                workbook,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                DocumentFamily.LAYOUT,
            )
        )


FIXTURES = Path(__file__).parents[2] / "fixtures"
