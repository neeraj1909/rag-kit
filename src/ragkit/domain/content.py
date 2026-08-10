"""Typed multimodal content parts and their relationships."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TypeAlias

from .errors import InvalidDomainValueError
from .provenance import (
    BoxLocator,
    CellLocator,
    ExtractionProvenance,
    KeyframeLocator,
    PageLocator,
    TextSpanLocator,
    TimeSpanLocator,
)


class RelationKind(StrEnum):
    """Evidence-preserving structural relationship between content parts."""

    CAPTION_OF = "caption_of"
    DERIVED_FROM = "derived_from"
    CONTINUES = "continues"
    CONTAINS = "contains"
    KEYFRAME_OF = "keyframe_of"
    LABELED_BY = "labeled_by"


@dataclass(frozen=True, slots=True)
class PartRelation:
    source_part_id: str
    target_part_id: str
    kind: RelationKind

    def __post_init__(self) -> None:
        if not self.source_part_id or not self.target_part_id:
            raise InvalidDomainValueError("relation part IDs must not be empty")
        if self.source_part_id == self.target_part_id:
            raise InvalidDomainValueError("a part cannot relate to itself")


@dataclass(frozen=True, slots=True)
class TextContent:
    part_id: str
    text: str
    provenance: ExtractionProvenance
    relations: tuple[PartRelation, ...] = ()
    family: ClassVar[str] = "text"

    def __post_init__(self) -> None:
        _validate_part(self.part_id, self.text)
        if not isinstance(self.provenance.locator, TextSpanLocator):
            raise InvalidDomainValueError("TextContent requires an exact text-span locator")


@dataclass(frozen=True, slots=True)
class OcrContent:
    part_id: str
    text: str
    provenance: ExtractionProvenance
    relations: tuple[PartRelation, ...] = ()
    family: ClassVar[str] = "ocr"

    def __post_init__(self) -> None:
        _validate_part(self.part_id, self.text)
        if not isinstance(self.provenance.locator, (PageLocator, BoxLocator)):
            raise InvalidDomainValueError("OcrContent requires a page or box locator")
        if self.provenance.confidence is None:
            raise InvalidDomainValueError("OCR confidence must be explicit")


@dataclass(frozen=True, slots=True)
class LayoutContent:
    part_id: str
    text: str
    provenance: ExtractionProvenance
    relations: tuple[PartRelation, ...] = ()
    family: ClassVar[str] = "layout"

    def __post_init__(self) -> None:
        _validate_part(self.part_id, self.text)
        if not isinstance(self.provenance.locator, (PageLocator, BoxLocator, CellLocator)):
            raise InvalidDomainValueError("LayoutContent requires a page, box, or cell locator")


@dataclass(frozen=True, slots=True)
class ImageContent:
    part_id: str
    description: str
    provenance: ExtractionProvenance
    relations: tuple[PartRelation, ...] = ()
    family: ClassVar[str] = "vision"

    def __post_init__(self) -> None:
        _validate_part(self.part_id, self.description)
        if not isinstance(self.provenance.locator, (PageLocator, BoxLocator)):
            raise InvalidDomainValueError("ImageContent requires a page or box locator")


@dataclass(frozen=True, slots=True)
class MediaContent:
    part_id: str
    transcript: str
    provenance: ExtractionProvenance
    relations: tuple[PartRelation, ...] = ()
    family: ClassVar[str] = "media"

    def __post_init__(self) -> None:
        _validate_part(self.part_id, self.transcript)
        if not isinstance(self.provenance.locator, (TimeSpanLocator, KeyframeLocator)):
            raise InvalidDomainValueError("MediaContent requires a time-span or keyframe locator")


ContentPart: TypeAlias = TextContent | OcrContent | LayoutContent | ImageContent | MediaContent


def _validate_part(part_id: str, value: str) -> None:
    if not part_id or not value:
        raise InvalidDomainValueError("content part ID and representation must not be empty")
