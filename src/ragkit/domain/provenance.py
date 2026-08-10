"""Exact source location and extraction provenance values."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from .errors import InvalidDomainValueError
from .identity import ComponentFingerprint


def _non_empty(value: str, label: str) -> None:
    if not value:
        raise InvalidDomainValueError(f"{label} must not be empty")


@dataclass(frozen=True, slots=True)
class TextSpanLocator:
    start: int
    end: int
    kind: ClassVar[str] = "text_span"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InvalidDomainValueError("text span must be a non-empty non-negative range")


@dataclass(frozen=True, slots=True)
class PageLocator:
    page: int
    kind: ClassVar[str] = "page"

    def __post_init__(self) -> None:
        if self.page < 0:
            raise InvalidDomainValueError("page must be non-negative")


@dataclass(frozen=True, slots=True)
class BoxLocator:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    kind: ClassVar[str] = "box"

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if self.page < 0 or not all(math.isfinite(value) for value in values):
            raise InvalidDomainValueError("box page and coordinates must be valid")
        if not 0 <= self.x0 < self.x1 <= 1 or not 0 <= self.y0 < self.y1 <= 1:
            raise InvalidDomainValueError("box must satisfy normalized x0/x1 and y0/y1 ordering")


@dataclass(frozen=True, slots=True)
class CellLocator:
    sheet: str
    start_row: int
    start_column: int
    end_row: int | None = None
    end_column: int | None = None
    kind: ClassVar[str] = "cell"

    def __post_init__(self) -> None:
        _non_empty(self.sheet, "sheet")
        end_row = self.start_row if self.end_row is None else self.end_row
        end_column = self.start_column if self.end_column is None else self.end_column
        if self.start_row < 0 or self.start_column < 0:
            raise InvalidDomainValueError("cell coordinates must be non-negative")
        if end_row < self.start_row or end_column < self.start_column:
            raise InvalidDomainValueError("cell range end must not precede start")
        object.__setattr__(self, "end_row", end_row)
        object.__setattr__(self, "end_column", end_column)


@dataclass(frozen=True, slots=True)
class TimeSpanLocator:
    start_ms: int
    end_ms: int
    kind: ClassVar[str] = "time"

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise InvalidDomainValueError("time span must be a non-empty non-negative range")


@dataclass(frozen=True, slots=True)
class KeyframeLocator:
    timestamp_ms: int
    frame_number: int
    kind: ClassVar[str] = "keyframe"

    def __post_init__(self) -> None:
        if self.timestamp_ms < 0 or self.frame_number < 0:
            raise InvalidDomainValueError("keyframe position must be non-negative")


SourceLocator: TypeAlias = (
    TextSpanLocator | PageLocator | BoxLocator | CellLocator | TimeSpanLocator | KeyframeLocator
)


@dataclass(frozen=True, slots=True)
class AssetRef:
    asset_id: str
    media_type: str
    sha256: str
    uri: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        _non_empty(self.asset_id, "asset_id")
        _non_empty(self.media_type, "media_type")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise InvalidDomainValueError("asset sha256 must be a full lowercase digest")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise InvalidDomainValueError("asset size must be non-negative")


@dataclass(frozen=True, slots=True)
class ExtractionNotice:
    code: str
    message: str

    def __post_init__(self) -> None:
        _non_empty(self.code, "notice code")
        _non_empty(self.message, "notice message")


@dataclass(frozen=True, slots=True)
class ExtractionProvenance:
    asset: AssetRef
    locator: SourceLocator
    extractor: ComponentFingerprint
    confidence: float | None = None
    notices: tuple[ExtractionNotice, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise InvalidDomainValueError("confidence must be finite and between zero and one")
