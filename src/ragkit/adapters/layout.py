"""Bounded PDF, PowerPoint, and workbook layout extraction."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from ragkit.domain import (
    BoxLocator,
    CellLocator,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionNotice,
    ExtractionProvenance,
    IntegrityError,
    InvalidDomainValueError,
    LayoutContent,
    LimitExceededError,
    MissingDependencyError,
    PartialExtractionError,
    PartRelation,
    RelationKind,
    SourceId,
    UnsupportedCapabilityError,
)
from ragkit.ports import AcquiredAsset, DocumentExtractor, DocumentFamily, ExtractionRequest

_PDF = "application/pdf"
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_XLSX = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
}


class LayoutDocumentExtractor(DocumentExtractor):
    """Preserve container boundaries, reading order, cells, and ambiguity notices."""

    def __init__(
        self,
        *,
        max_pages: int = 100,
        max_slides: int = 100,
        max_sheets: int = 20,
        max_cells: int = 100_000,
        max_archive_uncompressed_bytes: int = 256 * 1024 * 1024,
        max_compression_ratio: float = 100.0,
    ) -> None:
        if (
            min(max_pages, max_slides, max_sheets, max_cells, max_archive_uncompressed_bytes) <= 0
            or max_compression_ratio <= 0
        ):
            raise InvalidDomainValueError("layout limits must be positive")
        self._max_pages = max_pages
        self._max_slides = max_slides
        self._max_sheets = max_sheets
        self._max_cells = max_cells
        self._max_archive_uncompressed_bytes = max_archive_uncompressed_bytes
        self._max_compression_ratio = max_compression_ratio
        self._fingerprint = ComponentFingerprint.create(
            "extractor",
            "local_layout",
            {
                "version": 1,
                "max_pages": max_pages,
                "max_slides": max_slides,
                "max_sheets": max_sheets,
                "max_cells": max_cells,
                "max_archive_uncompressed_bytes": max_archive_uncompressed_bytes,
                "max_compression_ratio": max_compression_ratio,
            },
        )

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if len(request.assets) > request.max_documents:
            raise LimitExceededError("layout document count exceeds max_documents")
        documents: list[Document] = []
        for asset, classification in zip(request.assets, request.classifications, strict=True):
            if classification.family is not DocumentFamily.LAYOUT:
                raise UnsupportedCapabilityError(
                    f"layout extractor cannot handle {classification.family.value}",
                    capability=classification.family.value,
                )
            media_type = asset.reference.media_type
            if media_type == _PDF:
                parts, subtype, container_count = self._pdf_parts(asset)
                metadata_name = "pages"
            elif media_type == _PPTX:
                self._preflight_archive(asset.content)
                parts, subtype, container_count = self._pptx_parts(asset)
                metadata_name = "slides"
            elif media_type in _XLSX:
                self._preflight_archive(asset.content)
                parts, subtype, container_count = self._xlsx_parts(asset)
                metadata_name = "sheets"
            else:
                raise UnsupportedCapabilityError(
                    f"unsupported layout media type: {media_type}",
                    capability="layout_media_type",
                )
            source_id = SourceId.from_locator(
                "layout", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    parts,
                    {
                        "source_uri": asset.reference.uri,
                        "file_name": Path(asset.reference.uri or "").name,
                        "layout_subtype": subtype,
                        metadata_name: container_count,
                    },
                )
            )
        return tuple(documents)

    def _preflight_archive(self, content: bytes) -> None:
        try:
            with ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
        except BadZipFile as error:
            raise IntegrityError("layout archive is corrupt", cause=error) from error
        expanded = sum(member.file_size for member in members)
        compressed = sum(member.compress_size for member in members)
        if expanded > self._max_archive_uncompressed_bytes:
            raise LimitExceededError("layout archive expanded size exceeds configured limit")
        ratio = expanded / max(compressed, 1)
        if ratio > self._max_compression_ratio:
            raise LimitExceededError("layout archive compression ratio exceeds configured limit")

    def _pdf_parts(self, asset: AcquiredAsset) -> tuple[tuple[LayoutContent, ...], str, int]:
        try:
            pdfplumber = import_module("pdfplumber")
        except ImportError as error:
            raise MissingDependencyError(
                "PDF layout requires pdfplumber; install rag-kit[layout]"
            ) from error
        try:
            with pdfplumber.open(BytesIO(asset.content)) as pdf:
                if len(pdf.pages) > self._max_pages:
                    raise LimitExceededError(f"PDF page count exceeds {self._max_pages}")
                parts: list[LayoutContent] = []
                for page_index, page in enumerate(pdf.pages):
                    if page.images:
                        raise PartialExtractionError(
                            f"PDF page {page_index} contains images; route them to vision/OCR"
                        )
                    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                    if not words:
                        raise PartialExtractionError(
                            f"PDF page {page_index} has no machine-readable text; route it to OCR"
                        )
                    for word in words:
                        text = str(word.get("text", "")).strip()
                        if not text:
                            continue
                        locator = _box(
                            page_index,
                            float(word["x0"]),
                            float(word["top"]),
                            float(word["x1"]),
                            float(word["bottom"]),
                            float(page.width),
                            float(page.height),
                        )
                        parts.append(
                            self._part(
                                f"pdf-p{page_index}-w{len(parts)}",
                                text,
                                asset,
                                locator,
                                (
                                    ExtractionNotice(
                                        "heuristic_reading_order",
                                        "PDF word order follows pdfplumber text-flow heuristics",
                                    ),
                                ),
                                parts,
                            )
                        )
                if not parts:
                    raise IntegrityError("PDF contains no searchable layout text")
                return tuple(parts), "pdf", len(pdf.pages)
        except (LimitExceededError, PartialExtractionError, IntegrityError):
            raise
        except Exception as error:
            message = "unable to open PDF layout"
            if (
                "password" in str(error).casefold()
                or type(error).__name__ == "PDFPasswordIncorrect"
            ):
                message = "encrypted PDF requires an explicit password-capable adapter"
            raise IntegrityError(message, cause=error) from error

    def _pptx_parts(self, asset: AcquiredAsset) -> tuple[tuple[LayoutContent, ...], str, int]:
        try:
            presentation_module = import_module("pptx")
            shape_types = import_module("pptx.enum.shapes")
        except ImportError as error:
            raise MissingDependencyError(
                "PowerPoint layout requires python-pptx; install rag-kit[layout]"
            ) from error
        try:
            presentation = presentation_module.Presentation(BytesIO(asset.content))
            if len(presentation.slides) > self._max_slides:
                raise LimitExceededError(f"slide count exceeds {self._max_slides}")
            parts: list[LayoutContent] = []
            table_cell_count = 0
            for slide_index, slide in enumerate(presentation.slides):
                ordered = sorted(slide.shapes, key=lambda shape: (shape.top, shape.left))
                for shape_index, shape in enumerate(ordered):
                    if shape.shape_type == shape_types.MSO_SHAPE_TYPE.PICTURE:
                        raise PartialExtractionError(
                            f"slide {slide_index} contains an image; route it to vision"
                        )
                    if getattr(shape, "has_table", False):
                        table = shape.table
                        header_ids: dict[int, str] = {}
                        for row_index, row in enumerate(table.rows):
                            for column_index, cell in enumerate(row.cells):
                                text = cell.text.strip()
                                if text:
                                    table_cell_count += 1
                                    part_id = (
                                        f"pptx-s{slide_index}-t{shape_index}-"
                                        f"r{row_index}-c{column_index}"
                                    )
                                    parts.append(
                                        self._part(
                                            part_id,
                                            text,
                                            asset,
                                            CellLocator(
                                                f"slide:{slide_index}:table:{shape_index}",
                                                row_index,
                                                column_index,
                                                (
                                                    row_index + cell.span_height - 1
                                                    if cell.is_merge_origin
                                                    else None
                                                ),
                                                (
                                                    column_index + cell.span_width - 1
                                                    if cell.is_merge_origin
                                                    else None
                                                ),
                                            ),
                                            (),
                                            parts,
                                            header_ids.get(column_index),
                                        )
                                    )
                                    if row_index == 0:
                                        header_ids[column_index] = part_id
                                    if table_cell_count > self._max_cells:
                                        raise LimitExceededError(
                                            f"presentation cell count exceeds {self._max_cells}"
                                        )
                    elif getattr(shape, "has_text_frame", False):
                        text = shape.text.strip()
                        if text:
                            parts.append(
                                self._part(
                                    f"pptx-s{slide_index}-shape{shape_index}",
                                    text,
                                    asset,
                                    _box(
                                        slide_index,
                                        float(shape.left),
                                        float(shape.top),
                                        float(shape.left + shape.width),
                                        float(shape.top + shape.height),
                                        float(presentation.slide_width),
                                        float(presentation.slide_height),
                                    ),
                                    (),
                                    parts,
                                )
                            )
            if not parts:
                raise IntegrityError("presentation contains no searchable layout text")
            return tuple(parts), "pptx", len(presentation.slides)
        except (LimitExceededError, PartialExtractionError, IntegrityError):
            raise
        except Exception as error:
            raise IntegrityError("unable to open PowerPoint layout", cause=error) from error

    def _xlsx_parts(self, asset: AcquiredAsset) -> tuple[tuple[LayoutContent, ...], str, int]:
        try:
            openpyxl = import_module("openpyxl")
        except ImportError as error:
            raise MissingDependencyError(
                "workbook layout requires openpyxl; install rag-kit[layout]"
            ) from error
        try:
            formula_book = openpyxl.load_workbook(
                BytesIO(asset.content), data_only=False, read_only=False, keep_links=False
            )
        except Exception as error:
            raise IntegrityError("unable to open workbook layout", cause=error) from error
        try:
            value_book = openpyxl.load_workbook(
                BytesIO(asset.content), data_only=True, read_only=False, keep_links=False
            )
        except Exception as error:
            formula_book.close()
            raise IntegrityError("unable to open workbook layout", cause=error) from error
        try:
            if len(formula_book.worksheets) > self._max_sheets:
                raise LimitExceededError(f"workbook sheet count exceeds {self._max_sheets}")
            populated = sum(
                1
                for sheet in formula_book.worksheets
                for row in sheet.iter_rows()
                for cell in row
                if cell.value is not None
            )
            if populated > self._max_cells:
                raise LimitExceededError(f"populated cell count exceeds {self._max_cells}")
            parts: list[LayoutContent] = []
            for sheet in formula_book.worksheets:
                value_sheet = value_book[sheet.title]
                header_ids: dict[int, str] = {}
                merged = {(item.min_row, item.min_col): item for item in sheet.merged_cells.ranges}
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value is None:
                            continue
                        range_value = merged.get((cell.row, cell.column))
                        locator = CellLocator(
                            sheet.title,
                            cell.row - 1,
                            cell.column - 1,
                            None if range_value is None else range_value.max_row - 1,
                            None if range_value is None else range_value.max_col - 1,
                        )
                        notices: list[ExtractionNotice] = []
                        text = str(cell.value)
                        part_id = f"xlsx-{sheet.title}-r{cell.row - 1}-c{cell.column - 1}"
                        if cell.data_type == "f":
                            displayed = value_sheet.cell(cell.row, cell.column).value
                            if displayed is None:
                                notices.append(
                                    ExtractionNotice(
                                        "formula_cached_value_unavailable",
                                        "formula is retained but the file has no cached "
                                        "displayed value",
                                    )
                                )
                            else:
                                text = f"{text} [displayed: {displayed}]"
                        parts.append(
                            self._part(
                                part_id,
                                text,
                                asset,
                                locator,
                                tuple(notices),
                                parts,
                                header_ids.get(cell.column - 1),
                            )
                        )
                        if cell.row == 1:
                            header_ids[cell.column - 1] = part_id
            if not parts:
                raise IntegrityError("workbook contains no populated cells")
            return tuple(parts), "xlsx", len(formula_book.worksheets)
        finally:
            formula_book.close()
            value_book.close()

    def _part(
        self,
        part_id: str,
        text: str,
        asset: AcquiredAsset,
        locator: BoxLocator | CellLocator,
        notices: tuple[ExtractionNotice, ...],
        prior: Sequence[LayoutContent],
        labeled_by: str | None = None,
    ) -> LayoutContent:
        relations: list[PartRelation] = []
        if prior and _same_container(locator, prior[-1].provenance.locator):
            relations.append(PartRelation(part_id, prior[-1].part_id, RelationKind.CONTINUES))
        if labeled_by is not None:
            relations.append(PartRelation(part_id, labeled_by, RelationKind.LABELED_BY))
        return LayoutContent(
            part_id,
            text,
            ExtractionProvenance(asset.reference, locator, self._fingerprint, None, notices),
            tuple(relations),
        )


def _same_container(left: BoxLocator | CellLocator, right: object) -> bool:
    if isinstance(left, BoxLocator) and isinstance(right, BoxLocator):
        return left.page == right.page
    if isinstance(left, CellLocator) and isinstance(right, CellLocator):
        return left.sheet == right.sheet
    return False


def _box(
    page: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    height: float,
) -> BoxLocator:
    if width <= 0 or height <= 0:
        raise IntegrityError("layout container dimensions must be positive")
    return BoxLocator(page, x0 / width, y0 / height, x1 / width, y1 / height)
