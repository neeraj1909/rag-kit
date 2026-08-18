"""Manifest-bound dispatch across the complete chunking policy catalog."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ContentPart,
    Document,
    ExtractionProvenance,
    ImageContent,
    InvalidDomainValueError,
    LayoutContent,
    LimitExceededError,
    MediaContent,
    OcrContent,
    TextContent,
    TextSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    Chunker,
    ChunkingPolicy,
    ChunkingRequest,
    ChunkingStrategy,
    DocumentFamily,
    resolve_chunking_policy,
)

from .modality_chunking import ModalityChunker
from .text_chunking import TEXT_CHUNKING_STRATEGIES, chunk_text_documents, text_strategy_spans

_MODALITY_NATIVE = frozenset(
    {
        ChunkingStrategy.TABLE,
        ChunkingStrategy.LAYOUT_REGION,
        ChunkingStrategy.IMAGE_REGION,
        ChunkingStrategy.TRANSCRIPT_SEGMENT,
        ChunkingStrategy.SCENE,
        ChunkingStrategy.EVIDENCE,
    }
)


class AdaptiveChunker(Chunker):
    """Bind one family-compatible policy and dispatch without silent fallback."""

    def __init__(self, family: DocumentFamily, policy: ChunkingPolicy) -> None:
        resolved = resolve_chunking_policy(family, policy)
        if resolved.strategy is ChunkingStrategy.AUTO:  # defensive: resolver is exhaustive
            raise InvalidDomainValueError("adaptive chunking policy must resolve before binding")
        self._family = family
        self._policy = resolved
        self._native: ModalityChunker | None = None
        if family is not DocumentFamily.TEXT and resolved.strategy in _MODALITY_NATIVE:
            self._native = ModalityChunker(family, resolved)
            self._fingerprint = self._native.fingerprint
        else:
            self._fingerprint = ComponentFingerprint.create(
                "chunker",
                "adaptive_strategy_suite",
                {"version": 1, "family": family.value, **resolved.fingerprint_inputs()},
            )

    @property
    def policy(self) -> ChunkingPolicy:
        return self._policy

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        requested_policy = (
            self.policy
            if request.policy == ChunkingPolicy()
            else resolve_chunking_policy(self._family, request.policy)
        )
        if requested_policy != self.policy:
            raise InvalidDomainValueError(
                "chunking request policy must equal the manifest-bound chunking policy"
            )
        if request.policy != self.policy:
            request = replace(request, policy=self.policy)
        self._validate_families(request.documents)
        if self._native is not None:
            return self._native.chunk(request)
        if self._family is DocumentFamily.TEXT:
            return self._chunk_text(request)
        if self.policy.strategy not in TEXT_CHUNKING_STRATEGIES:
            raise UnsupportedCapabilityError(
                f"strategy {self.policy.strategy.value!r} has no "
                f"{self._family.value!r} implementation",
                capability="adaptive_chunking_strategy",
            )
        return self._chunk_normalized_modality(request)

    def _validate_families(self, documents: tuple[Document, ...]) -> None:
        foreign = tuple(
            part.part_id
            for document in documents
            for part in document.parts
            if part.family != self._family.value and not self._is_mixed_image_part(document, part)
        )
        if foreign:
            raise InvalidDomainValueError(
                f"document parts do not match bound {self._family.value!r} family: "
                + ", ".join(foreign)
            )

    def _is_mixed_image_part(self, document: Document, part: ContentPart) -> bool:
        return (
            self._family is DocumentFamily.VISION
            and document.metadata.get("content_mode") == "mixed_image_text"
            and isinstance(part, (OcrContent, ImageContent))
        )

    def _chunk_text(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        if self.policy.strategy in TEXT_CHUNKING_STRATEGIES:
            return chunk_text_documents(
                request.documents,
                policy=self.policy,
                fingerprint=self.fingerprint,
                max_chunks=request.max_chunks,
            )
        if self.policy.strategy not in {ChunkingStrategy.TABLE, ChunkingStrategy.EVIDENCE}:
            raise UnsupportedCapabilityError(
                f"strategy {self.policy.strategy.value!r} is not implemented for text",
                capability="adaptive_text_chunking",
            )
        chunks: list[Chunk] = []
        for document in request.documents:
            ordinal = 0
            for part in document.parts:
                if not isinstance(part, TextContent) or not isinstance(
                    part.provenance.locator, TextSpanLocator
                ):
                    raise UnsupportedCapabilityError(
                        "text table/evidence chunking requires exact textual parts",
                        capability="adaptive_text_chunking",
                    )
                spans = (
                    _table_row_spans(part.text)
                    if self.policy.strategy is ChunkingStrategy.TABLE
                    else _fixed_spans(part.text, self.policy.max_chars)
                )
                for row, (start, end) in enumerate(spans):
                    locator = TextSpanLocator(
                        part.provenance.locator.start + start,
                        part.provenance.locator.start + end,
                    )
                    chunks.append(
                        self._make_chunk(
                            document,
                            ordinal,
                            part.text[start:end],
                            (part,),
                            (replace(part.provenance, locator=locator),),
                            f"{part.part_id}/row-{row}",
                        )
                    )
                    ordinal += 1
        return _bounded(chunks, request.max_chunks)

    def _chunk_normalized_modality(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for document in request.documents:
            rendered = tuple(_representation(part) for part in document.parts)
            if self.policy.strategy is ChunkingStrategy.FIXED:
                groups = _fixed_atomic_groups(rendered, self.policy.max_chars)
                paths = [(str(document.document_id), self._family.value)] * len(groups)
            elif self.policy.strategy is ChunkingStrategy.SLIDING_WINDOW:
                groups = _sliding_atomic_groups(
                    rendered, self.policy.max_chars, self.policy.overlap_chars
                )
                paths = [(str(document.document_id), self._family.value)] * len(groups)
            else:
                joined, ranges = _joined_atomic_parts(rendered)
                spans = text_strategy_spans(
                    joined,
                    self.policy,
                    part_id=str(document.document_id),
                    document_metadata=document.metadata,
                )
                groups = []
                paths = []
                for span in spans:
                    group = tuple(
                        index
                        for index, (start, end) in enumerate(ranges)
                        if start < span.end and end > span.start
                    )
                    if group and (not groups or group != groups[-1]):
                        groups.append(group)
                        paths.append((span.structural_path, span.domain))
            for ordinal, (group, path) in enumerate(zip(groups, paths, strict=True)):
                parts = tuple(document.parts[index] for index in group)
                text = "\n\n".join(rendered[index] for index in group)
                chunks.append(
                    self._make_chunk(
                        document,
                        ordinal,
                        text,
                        parts,
                        tuple(part.provenance for part in parts),
                        path[0],
                        document_domain=path[1],
                    )
                )
        return _bounded(chunks, request.max_chunks)

    def _make_chunk(
        self,
        document: Document,
        ordinal: int,
        text: str,
        parts: tuple[ContentPart, ...],
        provenance: tuple[ExtractionProvenance, ...],
        structural_path: str,
        *,
        document_domain: str | None = None,
    ) -> Chunk:
        metadata = dict(document.metadata)
        content_families = tuple(dict.fromkeys(part.family for part in parts))
        metadata.update(
            {
                "content_family": (content_families[0] if len(content_families) == 1 else "mixed"),
                "document_family": self._family.value,
                "chunking_strategy": self.policy.strategy.value,
                "structural_path": structural_path,
            }
        )
        if document_domain is not None:
            metadata["document_domain"] = document_domain
        relations = [
            {
                "source_part_id": relation.source_part_id,
                "target_part_id": relation.target_part_id,
                "kind": relation.kind.value,
            }
            for part in parts
            for relation in part.relations
        ]
        if relations:
            metadata["source_relations_json"] = json.dumps(
                relations, sort_keys=True, separators=(",", ":")
            )
        evidence = tuple(
            (part.part_id, item.locator) for part, item in zip(parts, provenance, strict=True)
        )
        return Chunk(
            ChunkId.from_content(document.document_id, self.fingerprint, evidence, text),
            document.document_id,
            ordinal,
            text,
            provenance,
            tuple(part.part_id for part in parts),
            metadata,
        )


def _representation(part: ContentPart) -> str:
    if isinstance(part, (OcrContent, LayoutContent, TextContent)):
        return part.text
    if isinstance(part, ImageContent):
        return part.description
    if isinstance(part, MediaContent):
        return part.transcript
    raise UnsupportedCapabilityError(
        f"no normalized text representation for {type(part).__name__}",
        capability="normalized_modality_chunking",
    )


def _fixed_spans(text: str, limit: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for start in range(0, len(text), limit):
        end = min(start + limit, len(text))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return tuple(spans)


def _table_row_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(r"(?m)^\s*\S.*?(?=\n|$)", text)
        if match.group(0).strip()
    )


def _joined_atomic_parts(
    rendered: tuple[str, ...],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    pieces: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for value in rendered:
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(value)
        cursor += len(value)
        ranges.append((start, cursor))
    return "".join(pieces), tuple(ranges)


def _fixed_atomic_groups(rendered: tuple[str, ...], limit: int) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_size = 0
    for index, value in enumerate(rendered):
        added = len(value) + (2 if current else 0)
        if current and current_size + added > limit:
            groups.append(tuple(current))
            current = []
            current_size = 0
            added = len(value)
        current.append(index)
        current_size += added
    if current:
        groups.append(tuple(current))
    return groups


def _sliding_atomic_groups(
    rendered: tuple[str, ...], limit: int, overlap: int
) -> list[tuple[int, ...]]:
    fixed = _fixed_atomic_groups(rendered, limit)
    if overlap == 0 or len(fixed) < 2:
        return fixed
    groups: list[tuple[int, ...]] = [fixed[0]]
    for group in fixed[1:]:
        previous = groups[-1]
        group_size = sum(len(rendered[index]) for index in group) + 2 * (len(group) - 1)
        overlap_budget = min(overlap, max(0, limit - group_size))
        if overlap_budget == 0:
            groups.append(group)
            continue
        carried: list[int] = []
        size = 0
        for index in reversed(previous):
            value_size = len(rendered[index]) + (2 if carried else 0)
            if size + value_size > overlap_budget:
                break
            carried.insert(0, index)
            size += value_size
            if size >= overlap_budget:
                break
        candidate = tuple((*carried, *group))
        groups.append(candidate if candidate != previous else group)
    return groups


def _bounded(chunks: list[Chunk], maximum: int) -> tuple[Chunk, ...]:
    if len(chunks) > maximum:
        raise LimitExceededError(f"chunk limit {maximum} would truncate {len(chunks)} chunks")
    return tuple(chunks)
