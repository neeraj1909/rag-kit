"""Deterministic chunking policies for normalized non-text evidence."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable

from ragkit.domain import (
    BoxLocator,
    CellLocator,
    Chunk,
    ChunkId,
    ComponentFingerprint,
    ContentPart,
    Document,
    ImageContent,
    InvalidDomainValueError,
    KeyframeLocator,
    LayoutContent,
    LimitExceededError,
    MediaContent,
    OcrContent,
    PageLocator,
    RelationKind,
    TimeSpanLocator,
)
from ragkit.ports import (
    Chunker,
    ChunkingPolicy,
    ChunkingRequest,
    ChunkingStrategy,
    DocumentFamily,
    validate_chunking_strategy,
)

_MODALITY_FAMILIES = frozenset(
    {DocumentFamily.OCR, DocumentFamily.LAYOUT, DocumentFamily.VISION, DocumentFamily.MEDIA}
)
_MODALITY_STRATEGIES = frozenset(
    {
        ChunkingStrategy.TABLE,
        ChunkingStrategy.LAYOUT_REGION,
        ChunkingStrategy.IMAGE_REGION,
        ChunkingStrategy.TRANSCRIPT_SEGMENT,
        ChunkingStrategy.SCENE,
        ChunkingStrategy.EVIDENCE,
    }
)


class ModalityChunker(Chunker):
    """Apply one resolved modality strategy without weakening source evidence.

    Table rows, layout/image regions, transcript segments, and scenes are atomic:
    they may exceed ``max_chars`` instead of being cut at a meaningless character
    position. The generic evidence strategy is the bounded fallback and repeats the
    exact source provenance for every derived segment. A document explicitly marked
    ``content_mode=mixed_image_text`` may contain both OCR and image parts under the
    vision family. Evidence chunking retains both; image-region chunking emits OCR
    regions and image regions as separate source-ordered chunks.
    """

    def __init__(self, family: DocumentFamily, policy: ChunkingPolicy) -> None:
        if not isinstance(family, DocumentFamily):
            raise InvalidDomainValueError("family must be a DocumentFamily")
        if family not in _MODALITY_FAMILIES:
            raise InvalidDomainValueError("ModalityChunker requires a non-text modality family")
        if not isinstance(policy, ChunkingPolicy):
            raise InvalidDomainValueError("policy must be a ChunkingPolicy")
        if policy.strategy is ChunkingStrategy.AUTO:
            raise InvalidDomainValueError("ModalityChunker requires a resolved chunking policy")
        validate_chunking_strategy(family, policy.strategy)
        if policy.strategy not in _MODALITY_STRATEGIES:
            raise InvalidDomainValueError(
                f"strategy {policy.strategy.value!r} is not a modality-specific strategy"
            )
        self._family = family
        self._policy = policy
        self._fingerprint = ComponentFingerprint.create(
            "chunker",
            "modality_policy",
            {"version": 1, "family": family.value, **policy.fingerprint_inputs()},
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        if request.policy != self._policy:
            raise InvalidDomainValueError("request must use the chunker's bound chunking policy")
        chunks: list[Chunk] = []
        for document in request.documents:
            self._validate_document_family(document)
            groups = self._groups(document)
            for ordinal, (parts, text) in enumerate(groups):
                chunks.append(self._chunk(document, ordinal, parts, text))
        if len(chunks) > request.max_chunks:
            raise LimitExceededError(
                f"chunk limit {request.max_chunks} would truncate {len(chunks)} chunks"
            )
        return tuple(chunks)

    def _validate_document_family(self, document: Document) -> None:
        foreign = tuple(
            part.part_id for part in document.parts if part.family != self._family.value
        )
        is_supported_mixed_image = (
            self._family is DocumentFamily.VISION
            and document.metadata.get("content_mode") == "mixed_image_text"
            and self._policy.strategy in {ChunkingStrategy.EVIDENCE, ChunkingStrategy.IMAGE_REGION}
            and all(isinstance(part, (OcrContent, ImageContent)) for part in document.parts)
        )
        if is_supported_mixed_image:
            return
        if foreign:
            raise InvalidDomainValueError(
                f"document parts do not match bound {self._family.value!r} modality: "
                + ", ".join(foreign)
            )

    def _groups(self, document: Document) -> tuple[tuple[tuple[ContentPart, ...], str], ...]:
        strategy = self._policy.strategy
        if strategy is ChunkingStrategy.TABLE:
            return _table_groups(document)
        if strategy is ChunkingStrategy.LAYOUT_REGION:
            return _region_groups(document, (OcrContent, LayoutContent))
        if strategy is ChunkingStrategy.IMAGE_REGION:
            expected: tuple[type[ContentPart], ...] = (ImageContent,)
            if document.metadata.get("content_mode") == "mixed_image_text":
                expected = (OcrContent, ImageContent)
            return _region_groups(document, expected)
        if strategy is ChunkingStrategy.TRANSCRIPT_SEGMENT:
            return _region_groups(document, (MediaContent,))
        if strategy is ChunkingStrategy.SCENE:
            return _scene_groups(document)
        if strategy is ChunkingStrategy.EVIDENCE:
            return _evidence_groups(document, self._policy.max_chars)
        raise InvalidDomainValueError(f"unsupported modality strategy: {strategy.value}")

    def _chunk(
        self,
        document: Document,
        ordinal: int,
        parts: tuple[ContentPart, ...],
        text: str,
    ) -> Chunk:
        metadata = dict(document.metadata)
        content_families = tuple(dict.fromkeys(part.family for part in parts))
        metadata["content_family"] = (
            content_families[0] if len(content_families) == 1 else self._family.value
        )
        metadata["document_family"] = self._family.value
        metadata["chunking_strategy"] = self._policy.strategy.value
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
        evidence = tuple((part.part_id, part.provenance.locator) for part in parts)
        return Chunk(
            ChunkId.from_content(document.document_id, self.fingerprint, evidence, text),
            document.document_id,
            ordinal,
            text,
            tuple(part.provenance for part in parts),
            tuple(part.part_id for part in parts),
            metadata,
        )


def _table_groups(document: Document) -> tuple[tuple[tuple[ContentPart, ...], str], ...]:
    if not all(isinstance(part, (OcrContent, LayoutContent)) for part in document.parts):
        raise InvalidDomainValueError("table strategy requires OCR or layout content parts")
    parts_by_id = {part.part_id: part for part in document.parts}
    label_ids = _label_ids(document.parts)
    rows: OrderedDict[tuple[object, ...], list[ContentPart]] = OrderedDict()
    for index, part in enumerate(document.parts):
        if part.part_id in label_ids:
            continue
        locator = part.provenance.locator
        if isinstance(locator, CellLocator):
            key: tuple[object, ...] = ("cell", locator.sheet, locator.start_row)
        elif isinstance(locator, BoxLocator):
            # Exact y coordinates are stable extraction evidence and avoid guessing a
            # tolerance that would silently merge distinct visual rows.
            key = ("box", locator.page, locator.y0, locator.y1)
        elif isinstance(locator, PageLocator):
            key = ("page", locator.page, index)
        else:
            raise InvalidDomainValueError("table parts require cell, box, or page locators")
        rows.setdefault(key, []).append(part)

    groups: list[tuple[tuple[ContentPart, ...], str]] = []
    for row in rows.values():
        evidence: list[ContentPart] = []
        rendered: list[str] = []
        for part in row:
            labels = _related(part, parts_by_id, RelationKind.LABELED_BY)
            _append_unique(evidence, (part, *labels))
            value = _representation(part)
            if labels:
                rendered.append(
                    f"{' / '.join(_representation(label) for label in labels)}: {value}"
                )
            else:
                rendered.append(value)
        groups.append((tuple(evidence), " | ".join(rendered)))
    return tuple(groups)


def _region_groups(
    document: Document,
    expected_types: tuple[type[ContentPart], ...],
) -> tuple[tuple[tuple[ContentPart, ...], str], ...]:
    if not all(isinstance(part, expected_types) for part in document.parts):
        raise InvalidDomainValueError(
            "selected region strategy received an incompatible content part"
        )
    parts_by_id = {part.part_id: part for part in document.parts}
    label_ids = _label_ids(document.parts)
    groups: list[tuple[tuple[ContentPart, ...], str]] = []
    for part in document.parts:
        if part.part_id in label_ids:
            continue
        related = _related(part, parts_by_id)
        evidence = (part, *related)
        text = " ".join((*(_representation(item) for item in related), _representation(part)))
        groups.append((evidence, text))
    return tuple(groups)


def _scene_groups(document: Document) -> tuple[tuple[tuple[ContentPart, ...], str], ...]:
    if not all(isinstance(part, MediaContent) for part in document.parts):
        raise InvalidDomainValueError("scene strategy requires media content parts")
    parts_by_id = {part.part_id: part for part in document.parts}
    keyframes_by_scene: dict[str, list[ContentPart]] = {}
    for part in document.parts:
        if not isinstance(part.provenance.locator, KeyframeLocator):
            continue
        scene_ids = [
            relation.target_part_id
            for relation in part.relations
            if relation.kind is RelationKind.KEYFRAME_OF
        ]
        if len(scene_ids) != 1:
            raise InvalidDomainValueError(
                "each scene keyframe requires exactly one KEYFRAME_OF link"
            )
        scene = parts_by_id[scene_ids[0]]
        if not isinstance(scene.provenance.locator, TimeSpanLocator):
            raise InvalidDomainValueError("a keyframe must link to a time-span scene")
        keyframes_by_scene.setdefault(scene.part_id, []).append(part)

    grouped_keyframes = {
        keyframe.part_id for keyframes in keyframes_by_scene.values() for keyframe in keyframes
    }
    groups: list[tuple[tuple[ContentPart, ...], str]] = []
    for part in document.parts:
        if part.part_id in grouped_keyframes:
            continue
        keyframes = tuple(keyframes_by_scene.get(part.part_id, ()))
        evidence = (part, *keyframes)
        groups.append((evidence, " ".join(_representation(item) for item in evidence)))
    return tuple(groups)


def _evidence_groups(
    document: Document, max_chars: int
) -> tuple[tuple[tuple[ContentPart, ...], str], ...]:
    del max_chars
    parts_by_id = {part.part_id: part for part in document.parts}
    label_ids = _label_ids(document.parts)
    groups: list[tuple[tuple[ContentPart, ...], str]] = []
    for part in document.parts:
        if part.part_id in label_ids:
            continue
        related = _related(part, parts_by_id)
        evidence = (part, *related)
        representation = " ".join(
            (*(_representation(item) for item in related), _representation(part))
        )
        groups.append((evidence, representation))
    return tuple(groups)


def _label_ids(parts: Iterable[ContentPart]) -> frozenset[str]:
    return frozenset(
        relation.target_part_id
        for part in parts
        for relation in part.relations
        if relation.kind is RelationKind.LABELED_BY
    )


def _related(
    part: ContentPart,
    parts_by_id: dict[str, ContentPart],
    kind: RelationKind | None = None,
) -> tuple[ContentPart, ...]:
    identifiers = tuple(
        dict.fromkeys(
            relation.target_part_id
            for relation in part.relations
            if (kind is None or relation.kind is kind) and relation.target_part_id != part.part_id
        )
    )
    return tuple(parts_by_id[identifier] for identifier in identifiers)


def _append_unique(target: list[ContentPart], values: Iterable[ContentPart]) -> None:
    known = {part.part_id for part in target}
    for part in values:
        if part.part_id not in known:
            target.append(part)
            known.add(part.part_id)


def _representation(part: ContentPart) -> str:
    if isinstance(part, (OcrContent, LayoutContent)):
        return part.text
    if isinstance(part, ImageContent):
        return part.description
    if isinstance(part, MediaContent):
        return part.transcript
    raise InvalidDomainValueError(f"unsupported modality content part: {type(part).__name__}")
