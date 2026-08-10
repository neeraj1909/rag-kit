"""Bounded printed-text OCR with optional local dependencies and exact boxes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from math import ceil
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

from ragkit.domain import (
    BoxLocator,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionNotice,
    ExtractionProvenance,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    MissingDependencyError,
    OcrContent,
    PartialExtractionError,
    PartRelation,
    ProviderError,
    RelationKind,
    SourceId,
    UnsupportedCapabilityError,
)
from ragkit.ports import DocumentExtractor, DocumentFamily, ExtractionRequest

_IMAGE_TYPES = {"image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp"}
_PDF_TYPE = "application/pdf"
_PDFIUM_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class OcrToken:
    """One engine word in pixel coordinates with raw Tesseract confidence."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise InvalidDomainValueError("OCR token text must not be blank")
        if min(self.left, self.top) < 0 or min(self.width, self.height) <= 0:
            raise InvalidDomainValueError("OCR token box must be positive and non-negative")
        if not 0 <= self.confidence <= 100:
            raise InvalidDomainValueError("raw OCR confidence must be in [0, 100]")


class OcrEngine(Protocol):
    """Injectable recognition seam; ordinary use selects local pytesseract."""

    def recognize(
        self, image: object, *, language: str, timeout_seconds: float
    ) -> tuple[OcrToken, ...]: ...


class _Raster(Protocol):
    size: tuple[int, int]


class _PytesseractEngine:
    def recognize(
        self, image: object, *, language: str, timeout_seconds: float
    ) -> tuple[OcrToken, ...]:
        try:
            pytesseract = import_module("pytesseract")
        except ImportError as error:
            raise MissingDependencyError(
                "OCR requires the optional Python packages; install rag-kit[ocr]"
            ) from error
        try:
            values = cast(
                Mapping[str, Sequence[object]],
                pytesseract.image_to_data(
                    image,
                    lang=language,
                    output_type=pytesseract.Output.DICT,
                    timeout=timeout_seconds,
                ),
            )
        except pytesseract.TesseractNotFoundError as error:
            raise MissingDependencyError(
                "Tesseract executable is required for OCR; install it and configure tessdata"
            ) from error
        except RuntimeError as error:
            raise ProviderError(
                f"Tesseract exceeded the {timeout_seconds:g}s page timeout", cause=error
            ) from error
        except pytesseract.TesseractError as error:
            raise ProviderError(
                f"Tesseract failed; verify language data for {language!r}", cause=error
            ) from error
        required = ("text", "left", "top", "width", "height", "conf")
        if any(key not in values for key in required):
            raise ProviderError("Tesseract returned malformed word data")
        lengths = {len(values[key]) for key in required}
        if len(lengths) != 1:
            raise ProviderError("Tesseract returned misaligned word data")
        tokens: list[OcrToken] = []
        for index in range(next(iter(lengths), 0)):
            text = str(values["text"][index]).strip()
            try:
                confidence = float(str(values["conf"][index]))
            except (TypeError, ValueError) as error:
                raise ProviderError(
                    "Tesseract returned invalid confidence data", cause=error
                ) from error
            if not text or confidence < 0:
                continue
            try:
                tokens.append(
                    OcrToken(
                        text,
                        int(float(str(values["left"][index]))),
                        int(float(str(values["top"][index]))),
                        int(float(str(values["width"][index]))),
                        int(float(str(values["height"][index]))),
                        confidence,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ProviderError("Tesseract returned invalid box data", cause=error) from error
        return tuple(tokens)


class OcrDocumentExtractor(DocumentExtractor):
    """Extract printed raster/PDF words; handwriting and forms remain degraded."""

    def __init__(
        self,
        *,
        engine: OcrEngine | None = None,
        language: str = "eng",
        content_mode: str = "printed",
        low_confidence_threshold: float = 0.6,
        max_pages: int = 25,
        max_pixels: int = 20_000_000,
        target_dpi: int = 300,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not language.strip():
            raise InvalidDomainValueError("OCR language must not be blank")
        if content_mode not in {"printed", "handwriting", "form"}:
            raise InvalidDomainValueError("content_mode must be printed, handwriting, or form")
        if not 0 <= low_confidence_threshold <= 1:
            raise InvalidDomainValueError("low confidence threshold must be in [0, 1]")
        if min(max_pages, max_pixels, target_dpi) <= 0 or timeout_seconds <= 0:
            raise InvalidDomainValueError("OCR resource limits must be positive")
        self._engine = engine or _PytesseractEngine()
        self._language = language
        self._content_mode = content_mode
        self._low_confidence_threshold = low_confidence_threshold
        self._max_pages = max_pages
        self._max_pixels = max_pixels
        self._target_dpi = target_dpi
        self._timeout_seconds = timeout_seconds
        self._fingerprint = ComponentFingerprint.create(
            "extractor",
            "pytesseract_printed",
            {
                "version": 1,
                "language": language,
                "content_mode": content_mode,
                "low_confidence_threshold": low_confidence_threshold,
                "max_pages": max_pages,
                "max_pixels": max_pixels,
                "target_dpi": target_dpi,
                "timeout_seconds": timeout_seconds,
            },
        )

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if len(request.assets) > request.max_documents:
            raise LimitExceededError("OCR document count exceeds max_documents")
        documents: list[Document] = []
        for asset, classification in zip(request.assets, request.classifications, strict=True):
            if classification.family is not DocumentFamily.OCR:
                raise UnsupportedCapabilityError(
                    f"OCR extractor cannot handle {classification.family.value}",
                    capability=classification.family.value,
                )
            pages = self._decode_pages(asset.content, asset.reference.media_type)
            parts: list[OcrContent] = []
            for page_index, image in enumerate(pages):
                width, height = image.size
                tokens = self._engine.recognize(
                    image,
                    language=self._language,
                    timeout_seconds=self._timeout_seconds,
                )
                if not tokens:
                    raise PartialExtractionError(
                        f"page {page_index} produced no OCR text; "
                        "complete extraction is unavailable"
                    )
                for word_index, token in enumerate(tokens):
                    confidence = token.confidence / 100.0
                    notices = self._notices(confidence)
                    part_id = f"ocr-p{page_index}-w{word_index}"
                    locator = _normalized_box(page_index, token, width, height)
                    relations: tuple[PartRelation, ...] = ()
                    if self._content_mode == "form" and parts:
                        prior = parts[-1]
                        prior_locator = prior.provenance.locator
                        if (
                            isinstance(prior_locator, BoxLocator)
                            and prior_locator.page == locator.page
                            and prior_locator.x1 <= locator.x0
                            and prior_locator.y0 < locator.y1
                            and locator.y0 < prior_locator.y1
                        ):
                            relations = (
                                PartRelation(
                                    part_id,
                                    prior.part_id,
                                    RelationKind.LABELED_BY,
                                ),
                            )
                    parts.append(
                        OcrContent(
                            part_id,
                            token.text,
                            ExtractionProvenance(
                                asset.reference,
                                locator,
                                self._fingerprint,
                                confidence,
                                notices,
                            ),
                            relations,
                        )
                    )
            source_id = SourceId.from_locator(
                "ocr", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    tuple(parts),
                    {
                        "source_uri": asset.reference.uri,
                        "file_name": Path(asset.reference.uri or "").name,
                        "page_count": len(pages),
                        "ocr_language": self._language,
                        "degraded_mode": (
                            None if self._content_mode == "printed" else self._content_mode
                        ),
                    },
                )
            )
        return tuple(documents)

    def _notices(self, confidence: float) -> tuple[ExtractionNotice, ...]:
        notices: list[ExtractionNotice] = []
        if confidence < self._low_confidence_threshold:
            notices.append(
                ExtractionNotice(
                    "low_ocr_confidence", "raw Tesseract confidence is below review threshold"
                )
            )
        if self._content_mode == "handwriting":
            notices.append(
                ExtractionNotice(
                    "handwriting_best_effort",
                    "Tesseract printed-text OCR does not verify handwriting recognition",
                )
            )
        elif self._content_mode == "form":
            notices.append(
                ExtractionNotice(
                    "form_structure_unverified",
                    "Field/value links are heuristic and unverified; "
                    "checkbox state is not inferred",
                )
            )
        return tuple(notices)

    def _decode_pages(self, content: bytes, media_type: str) -> tuple[_Raster, ...]:
        if media_type == _PDF_TYPE:
            return self._decode_pdf(content)
        if media_type not in _IMAGE_TYPES:
            raise UnsupportedCapabilityError(
                f"unsupported OCR media type: {media_type}", capability="ocr_media_type"
            )
        try:
            image_module = import_module("PIL.Image")
            image_ops = import_module("PIL.ImageOps")
            image_sequence = import_module("PIL.ImageSequence")
        except ImportError as error:
            raise MissingDependencyError(
                "image OCR requires Pillow; install rag-kit[ocr]"
            ) from error
        try:
            opened = image_module.open(BytesIO(content))
            try:
                frames_list: list[_Raster] = []
                for frame in image_sequence.Iterator(opened):
                    width, height = cast(tuple[int, int], frame.size)
                    if width * height > self._max_pixels:
                        raise LimitExceededError(
                            f"OCR raster exceeds {self._max_pixels} pixel limit"
                        )
                    frames_list.append(
                        cast(_Raster, image_ops.exif_transpose(frame.copy()).convert("RGB"))
                    )
                frames = tuple(frames_list)
            finally:
                opened.close()
        except LimitExceededError:
            raise
        except (OSError, ValueError) as error:
            raise IntegrityError("unable to decode OCR image", cause=error) from error
        self._check_pages(frames)
        return frames

    def _decode_pdf(self, content: bytes) -> tuple[_Raster, ...]:
        try:
            pdfium = import_module("pypdfium2")
        except ImportError as error:
            raise MissingDependencyError(
                "PDF OCR requires pypdfium2; install rag-kit[ocr]"
            ) from error
        try:
            with _PDFIUM_LOCK:
                pdf = pdfium.PdfDocument(content)
                try:
                    if len(pdf) > self._max_pages:
                        raise LimitExceededError(f"PDF page count exceeds {self._max_pages}")
                    pages: list[_Raster] = []
                    scale = self._target_dpi / 72
                    for page in pdf:
                        try:
                            width, height = page.get_size()
                            if ceil(width * scale) * ceil(height * scale) > self._max_pixels:
                                raise LimitExceededError(
                                    f"rendered PDF page exceeds {self._max_pixels} pixel limit"
                                )
                            bitmap = page.render(scale=scale)
                            try:
                                pages.append(cast(_Raster, bitmap.to_pil().convert("RGB")))
                            finally:
                                bitmap.close()
                        finally:
                            page.close()
                finally:
                    pdf.close()
        except LimitExceededError:
            raise
        except Exception as error:
            raise IntegrityError("unable to decode OCR PDF", cause=error) from error
        self._check_pages(tuple(pages))
        return tuple(pages)

    def _check_pages(self, pages: tuple[_Raster, ...]) -> None:
        if not pages:
            raise IntegrityError("OCR asset contains no pages")
        if len(pages) > self._max_pages:
            raise LimitExceededError(f"OCR page count exceeds {self._max_pages}")
        for image in pages:
            width, height = image.size
            if width * height > self._max_pixels:
                raise LimitExceededError(f"OCR raster exceeds {self._max_pixels} pixel limit")


def _normalized_box(page: int, token: OcrToken, width: int, height: int) -> BoxLocator:
    if token.left + token.width > width or token.top + token.height > height:
        raise ProviderError("OCR engine returned a box outside the source raster")
    return BoxLocator(
        page,
        token.left / width,
        token.top / height,
        (token.left + token.width) / width,
        (token.top + token.height) / height,
    )
