"""Shared deterministic chunking for normalized multimodal content parts."""

from __future__ import annotations

import json

from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ImageContent,
    InvalidDomainValueError,
    LayoutContent,
    LimitExceededError,
    MediaContent,
    OcrContent,
    TextContent,
)
from ragkit.ports import Chunker, ChunkingRequest


class EvidenceChunker(Chunker):
    """Split normalized representations while retaining their exact evidence locator.

    The pure adapter is deterministic and thread-safe. Splits from one content part
    deliberately share that part's locator because the normalized representation is
    derived evidence rather than a character-addressable original source.
    """

    def __init__(self, max_chars: int = 800) -> None:
        if max_chars <= 0:
            raise InvalidDomainValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._fingerprint = ComponentFingerprint.create(
            "chunker", "normalized_multimodal_evidence", {"version": 1, "max_chars": max_chars}
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for document in request.documents:
            ordinal = 0
            for part in document.parts:
                representation = _representation(part)
                for text in _segments(representation, self._max_chars):
                    metadata = dict(document.metadata)
                    metadata["content_family"] = part.family
                    if part.relations:
                        metadata["source_relations_json"] = json.dumps(
                            [
                                {
                                    "source_part_id": relation.source_part_id,
                                    "target_part_id": relation.target_part_id,
                                    "kind": relation.kind.value,
                                }
                                for relation in part.relations
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    chunks.append(
                        Chunk(
                            ChunkId.from_content(
                                document.document_id,
                                self.fingerprint,
                                ((part.part_id, part.provenance.locator),),
                                text,
                            ),
                            document.document_id,
                            ordinal,
                            text,
                            (part.provenance,),
                            (part.part_id,),
                            metadata,
                        )
                    )
                    ordinal += 1
        if len(chunks) > request.max_chunks:
            raise LimitExceededError(
                f"chunk limit {request.max_chunks} would truncate {len(chunks)} chunks"
            )
        return tuple(chunks)


def _representation(
    part: TextContent | OcrContent | LayoutContent | ImageContent | MediaContent,
) -> str:
    if isinstance(part, (TextContent, OcrContent, LayoutContent)):
        return part.text
    if isinstance(part, ImageContent):
        return part.description
    return part.transcript


def _segments(text: str, limit: int) -> tuple[str, ...]:
    segments: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", cursor, end + 1), text.rfind("\n", cursor, end + 1))
            if boundary > cursor:
                end = boundary
        value = text[cursor:end].strip()
        if value:
            segments.append(value)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    return tuple(segments)
