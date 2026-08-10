"""Documents, chunks, embeddings, and retrieval score records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

from .content import (
    ContentPart,
    ImageContent,
    LayoutContent,
    MediaContent,
    OcrContent,
    PartRelation,
    RelationKind,
    TextContent,
)
from .errors import InvalidDomainValueError
from .identity import ChunkId, ComponentFingerprint, DocumentId, SourceId
from .provenance import (
    AssetRef,
    BoxLocator,
    CellLocator,
    ExtractionNotice,
    ExtractionProvenance,
    KeyframeLocator,
    PageLocator,
    SourceLocator,
    TextSpanLocator,
    TimeSpanLocator,
)

MetadataValue: TypeAlias = str | int | float | bool | None


def _immutable_metadata(value: Mapping[str, MetadataValue]) -> Mapping[str, MetadataValue]:
    if not all(isinstance(key, str) and key for key in value):
        raise InvalidDomainValueError("metadata keys must be non-empty strings")
    for item in value.values():
        if isinstance(item, float) and not math.isfinite(item):
            raise InvalidDomainValueError("metadata numbers must be finite")
    return MappingProxyType(dict(value))


def _empty_metadata() -> Mapping[str, MetadataValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Document:
    document_id: DocumentId
    source_id: SourceId
    assets: tuple[AssetRef, ...]
    parts: tuple[ContentPart, ...]
    metadata: Mapping[str, MetadataValue] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.assets:
            raise InvalidDomainValueError("document requires original assets")
        asset_ids = {item.asset_id for item in self.assets}
        if len(asset_ids) != len(self.assets):
            raise InvalidDomainValueError("document asset IDs must be unique")
        part_ids = {item.part_id for item in self.parts}
        if len(part_ids) != len(self.parts):
            raise InvalidDomainValueError("document part IDs must be unique")
        if any(part.provenance.asset.asset_id not in asset_ids for part in self.parts):
            raise InvalidDomainValueError("every part must resolve to a document asset")
        if any(
            relation.source_part_id != part.part_id
            for part in self.parts
            for relation in part.relations
        ):
            raise InvalidDomainValueError("relation source must match its owning content part")
        if any(
            relation.target_part_id not in part_ids
            for part in self.parts
            for relation in part.relations
        ):
            raise InvalidDomainValueError("relations must resolve within the document")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": str(self.document_id),
            "source_id": str(self.source_id),
            "assets": [_asset_to_dict(item) for item in self.assets],
            "parts": [_part_to_dict(item) for item in self.parts],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Document:
        assets = cast(list[dict[str, object]], value["assets"])
        parts = cast(list[dict[str, object]], value["parts"])
        return cls(
            DocumentId(cast(str, value["document_id"])),
            SourceId(cast(str, value["source_id"])),
            tuple(_asset_from_dict(item) for item in assets),
            tuple(_part_from_dict(item) for item in parts),
            cast(dict[str, MetadataValue], value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: ChunkId
    document_id: DocumentId
    ordinal: int
    text: str
    provenance: tuple[ExtractionProvenance, ...]
    source_part_ids: tuple[str, ...]
    metadata: Mapping[str, MetadataValue] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise InvalidDomainValueError("chunk ordinal must be non-negative")
        if not self.text or not self.provenance:
            raise InvalidDomainValueError("chunk requires text and exact provenance")
        if len(self.provenance) != len(self.source_part_ids):
            raise InvalidDomainValueError("chunk provenance and source part IDs must align")
        if not all(self.source_part_ids) or len(set(self.source_part_ids)) != len(
            self.source_part_ids
        ):
            raise InvalidDomainValueError("chunk source part IDs must be non-empty and unique")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "ordinal": self.ordinal,
            "text": self.text,
            "provenance": [_provenance_to_dict(item) for item in self.provenance],
            "source_part_ids": list(self.source_part_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Chunk:
        provenance = cast(list[dict[str, object]], value["provenance"])
        return cls(
            ChunkId(cast(str, value["chunk_id"])),
            DocumentId(cast(str, value["document_id"])),
            cast(int, value["ordinal"]),
            cast(str, value["text"]),
            tuple(_provenance_from_dict(item) for item in provenance),
            tuple(cast(list[str], value["source_part_ids"])),
            cast(dict[str, MetadataValue], value.get("metadata", {})),
        )


def derive_chunk_id(chunk: Chunk, chunker: ComponentFingerprint) -> ChunkId:
    """Recompute stable chunk identity from exact representation and provenance."""

    return ChunkId.from_content(
        chunk.document_id,
        chunker,
        tuple(
            (part_id, provenance.locator)
            for part_id, provenance in zip(chunk.source_part_ids, chunk.provenance, strict=True)
        ),
        chunk.text,
    )


@dataclass(frozen=True, slots=True)
class Embedding:
    values: tuple[float, ...]
    dimension: int
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.dimension <= 0 or len(self.values) != self.dimension:
            raise InvalidDomainValueError("embedding dimension must match its non-empty values")
        if not all(math.isfinite(value) for value in self.values):
            raise InvalidDomainValueError("embedding values must be finite")


class ScoreKind(StrEnum):
    SIMILARITY = "similarity"
    DISTANCE = "distance"
    LOGIT = "logit"


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    component: ComponentFingerprint
    stage: str
    kind: ScoreKind
    metric: str
    conversion: str

    def __post_init__(self) -> None:
        if not self.stage or not self.metric or not self.conversion:
            raise InvalidDomainValueError("score provenance fields must not be empty")


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    relevance: float
    raw_score: float | None
    provenance: ScoreProvenance

    def __post_init__(self) -> None:
        if not math.isfinite(self.relevance):
            raise InvalidDomainValueError("canonical relevance must be finite")
        if self.raw_score is not None and not math.isfinite(self.raw_score):
            raise InvalidDomainValueError("raw score must be finite")

    @classmethod
    def from_raw(cls, raw_score: float, provenance: ScoreProvenance) -> RetrievalScore:
        """Apply the standard versioned monotonic conversion for a native score."""

        if provenance.kind is ScoreKind.DISTANCE:
            relevance = -raw_score
            expected = "negate:v1"
        else:
            relevance = raw_score
            expected = "identity:v1"
        if provenance.conversion != expected:
            raise InvalidDomainValueError(
                f"{provenance.kind.value} requires the standard {expected} conversion"
            )
        return cls(relevance, raw_score, provenance)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """One ranked chunk plus current score and uncalibrated prior-stage history."""

    chunk: Chunk
    score: RetrievalScore
    rank: int
    prior_scores: tuple[RetrievalScore, ...] = ()

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise InvalidDomainValueError("rank is one-based")


def sort_scored_chunks(
    values: Sequence[tuple[ChunkId, RetrievalScore]], *, top_k: int
) -> tuple[tuple[ChunkId, RetrievalScore], ...]:
    """Validate and sort results by descending relevance, then stable chunk ID."""

    if top_k <= 0:
        raise InvalidDomainValueError("top_k must be positive")
    identifiers = [identifier for identifier, _ in values]
    if len(set(identifiers)) != len(identifiers):
        raise InvalidDomainValueError("duplicate chunk IDs are not allowed")
    return tuple(sorted(values, key=lambda item: (-item[1].relevance, str(item[0])))[:top_k])


def _asset_to_dict(value: AssetRef) -> dict[str, object]:
    return {
        "asset_id": value.asset_id,
        "media_type": value.media_type,
        "sha256": value.sha256,
        "uri": value.uri,
        "size_bytes": value.size_bytes,
    }


def _asset_from_dict(value: Mapping[str, object]) -> AssetRef:
    return AssetRef(
        cast(str, value["asset_id"]),
        cast(str, value["media_type"]),
        cast(str, value["sha256"]),
        cast(str | None, value.get("uri")),
        cast(int | None, value.get("size_bytes")),
    )


def _locator_to_dict(locator: SourceLocator) -> dict[str, object]:
    if isinstance(locator, TextSpanLocator):
        return {"kind": locator.kind, "start": locator.start, "end": locator.end}
    if isinstance(locator, PageLocator):
        return {"kind": locator.kind, "page": locator.page}
    if isinstance(locator, BoxLocator):
        return {
            "kind": locator.kind,
            "page": locator.page,
            "x0": locator.x0,
            "y0": locator.y0,
            "x1": locator.x1,
            "y1": locator.y1,
        }
    if isinstance(locator, CellLocator):
        return {
            "kind": locator.kind,
            "sheet": locator.sheet,
            "start_row": locator.start_row,
            "start_column": locator.start_column,
            "end_row": locator.end_row,
            "end_column": locator.end_column,
        }
    if isinstance(locator, TimeSpanLocator):
        return {"kind": locator.kind, "start_ms": locator.start_ms, "end_ms": locator.end_ms}
    return {
        "kind": locator.kind,
        "timestamp_ms": locator.timestamp_ms,
        "frame_number": locator.frame_number,
    }


def _locator_from_dict(value: Mapping[str, object]) -> SourceLocator:
    kind = value["kind"]
    if kind == "text_span":
        return TextSpanLocator(cast(int, value["start"]), cast(int, value["end"]))
    if kind == "page":
        return PageLocator(cast(int, value["page"]))
    if kind == "box":
        return BoxLocator(
            cast(int, value["page"]),
            cast(float, value["x0"]),
            cast(float, value["y0"]),
            cast(float, value["x1"]),
            cast(float, value["y1"]),
        )
    if kind == "cell":
        return CellLocator(
            cast(str, value["sheet"]),
            cast(int, value["start_row"]),
            cast(int, value["start_column"]),
            cast(int, value["end_row"]),
            cast(int, value["end_column"]),
        )
    if kind == "time":
        return TimeSpanLocator(cast(int, value["start_ms"]), cast(int, value["end_ms"]))
    if kind == "keyframe":
        return KeyframeLocator(cast(int, value["timestamp_ms"]), cast(int, value["frame_number"]))
    raise InvalidDomainValueError("unknown source locator kind")


def _provenance_to_dict(value: ExtractionProvenance) -> dict[str, object]:
    return {
        "asset": _asset_to_dict(value.asset),
        "locator": _locator_to_dict(value.locator),
        "extractor": str(value.extractor),
        "confidence": value.confidence,
        "notices": [{"code": item.code, "message": item.message} for item in value.notices],
    }


def _provenance_from_dict(value: Mapping[str, object]) -> ExtractionProvenance:
    notices = cast(list[dict[str, str]], value.get("notices", []))
    return ExtractionProvenance(
        _asset_from_dict(cast(dict[str, object], value["asset"])),
        _locator_from_dict(cast(dict[str, object], value["locator"])),
        ComponentFingerprint(cast(str, value["extractor"])),
        cast(float | None, value.get("confidence")),
        tuple(ExtractionNotice(**item) for item in notices),
    )


def _part_to_dict(value: ContentPart) -> dict[str, object]:
    representation = (
        value.description
        if isinstance(value, ImageContent)
        else value.transcript
        if isinstance(value, MediaContent)
        else value.text
    )
    return {
        "family": value.family,
        "part_id": value.part_id,
        "representation": representation,
        "provenance": _provenance_to_dict(value.provenance),
        "relations": [
            {
                "source_part_id": item.source_part_id,
                "target_part_id": item.target_part_id,
                "kind": item.kind.value,
            }
            for item in value.relations
        ],
    }


def _part_from_dict(value: Mapping[str, object]) -> ContentPart:
    family = value["family"]
    part_id = cast(str, value["part_id"])
    representation = cast(str, value["representation"])
    provenance = _provenance_from_dict(cast(dict[str, object], value["provenance"]))
    relations = tuple(
        PartRelation(item["source_part_id"], item["target_part_id"], RelationKind(item["kind"]))
        for item in cast(list[dict[str, str]], value.get("relations", []))
    )
    if family == "text":
        return TextContent(part_id, representation, provenance, relations)
    if family == "ocr":
        return OcrContent(part_id, representation, provenance, relations)
    if family == "layout":
        return LayoutContent(part_id, representation, provenance, relations)
    if family == "vision":
        return ImageContent(part_id, representation, provenance, relations)
    if family == "media":
        return MediaContent(part_id, representation, provenance, relations)
    raise InvalidDomainValueError("unknown content family")


def locator_to_dict(locator: SourceLocator) -> dict[str, object]:
    """Serialize one exact locator without erasing its variant."""

    return _locator_to_dict(locator)


def locator_from_dict(value: Mapping[str, object]) -> SourceLocator:
    """Restore an exact locator and validate its invariants."""

    return _locator_from_dict(value)


def provenance_to_dict(value: ExtractionProvenance) -> dict[str, object]:
    """Serialize provenance including original asset, confidence, and notices."""

    return _provenance_to_dict(value)


def provenance_from_dict(value: Mapping[str, object]) -> ExtractionProvenance:
    """Restore provenance without dropping source evidence."""

    return _provenance_from_dict(value)


def content_part_to_dict(value: ContentPart) -> dict[str, object]:
    """Serialize a content part while retaining its family discriminator."""

    return _part_to_dict(value)


def content_part_from_dict(value: Mapping[str, object]) -> ContentPart:
    """Restore the discriminated content-part variant."""

    return _part_from_dict(value)
