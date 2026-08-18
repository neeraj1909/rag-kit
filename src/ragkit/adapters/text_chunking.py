"""Deterministic, exact-span chunking strategies for textual documents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    Document,
    IntegrityError,
    InvalidDomainValueError,
    LimitExceededError,
    TextContent,
    TextSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import Chunker, ChunkingPolicy, ChunkingRequest, ChunkingStrategy


@dataclass(frozen=True, slots=True)
class TextChunkSpan:
    """One non-blank exact source range and its inferred structural context."""

    start: int
    end: int
    structural_path: str
    domain: str


class TextStrategyChunker(Chunker):
    """Bind one resolved text policy to a manifest-safe component fingerprint."""

    def __init__(self, policy: ChunkingPolicy) -> None:
        if policy.strategy is ChunkingStrategy.AUTO:
            raise InvalidDomainValueError("text chunking policy must be resolved before binding")
        if policy.strategy not in TEXT_CHUNKING_STRATEGIES:
            raise UnsupportedCapabilityError(
                f"strategy {policy.strategy.value!r} is not a textual chunking strategy",
                capability="text_chunking_strategy",
            )
        self._policy = policy
        self._fingerprint = ComponentFingerprint.create(
            "chunker", "text_strategy_suite", policy.fingerprint_inputs()
        )

    @property
    def policy(self) -> ChunkingPolicy:
        return self._policy

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        if request.policy != self.policy:
            raise InvalidDomainValueError(
                "chunking request policy must equal the policy bound to the text chunker"
            )
        return chunk_text_documents(
            request.documents,
            policy=self.policy,
            fingerprint=self.fingerprint,
            max_chunks=request.max_chunks,
        )


TEXT_CHUNKING_STRATEGIES = frozenset(
    {
        ChunkingStrategy.FIXED,
        ChunkingStrategy.SLIDING_WINDOW,
        ChunkingStrategy.RECURSIVE,
        ChunkingStrategy.SENTENCE,
        ChunkingStrategy.PARAGRAPH,
        ChunkingStrategy.SECTION,
        ChunkingStrategy.HIERARCHICAL,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.PROPOSITION,
        ChunkingStrategy.BOOK,
        ChunkingStrategy.LEGAL,
        ChunkingStrategy.MEDICAL,
        ChunkingStrategy.CODE,
        ChunkingStrategy.CONVERSATION,
    }
)


def chunk_text_documents(
    documents: Sequence[Document],
    *,
    policy: ChunkingPolicy,
    fingerprint: ComponentFingerprint,
    max_chunks: int,
) -> tuple[Chunk, ...]:
    """Chunk exact textual parts using an explicit caller-owned identity fingerprint.

    The standalone adapter passes its own fingerprint. A family dispatcher can pass
    its root fingerprint so chunk IDs and the index manifest describe the same
    behavior-affecting policy.
    """

    if max_chunks <= 0:
        raise InvalidDomainValueError("max_chunks must be positive")
    if policy.strategy is ChunkingStrategy.AUTO or policy.strategy not in TEXT_CHUNKING_STRATEGIES:
        raise UnsupportedCapabilityError(
            f"strategy {policy.strategy.value!r} is not a resolved textual strategy",
            capability="text_chunking_strategy",
        )
    chunks: list[Chunk] = []
    for document in documents:
        ordinal = 0
        for part in document.parts:
            if not isinstance(part, TextContent) or not isinstance(
                part.provenance.locator, TextSpanLocator
            ):
                raise UnsupportedCapabilityError(
                    "text strategy chunker supports exact textual parts only",
                    capability="text_chunking",
                )
            if part.provenance.locator.end - part.provenance.locator.start != len(part.text):
                raise IntegrityError("text part does not exactly fill its declared source span")
            spans = text_strategy_spans(
                part.text,
                policy,
                part_id=part.part_id,
                document_metadata=document.metadata,
            )
            for span in spans:
                locator = TextSpanLocator(
                    part.provenance.locator.start + span.start,
                    part.provenance.locator.start + span.end,
                )
                text = part.text[span.start : span.end]
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "chunking_strategy": policy.strategy.value,
                        "structural_path": span.structural_path,
                        "document_domain": span.domain,
                    }
                )
                provenance = replace(part.provenance, locator=locator)
                chunks.append(
                    Chunk(
                        ChunkId.from_content(
                            document.document_id,
                            fingerprint,
                            ((part.part_id, locator),),
                            text,
                        ),
                        document.document_id,
                        ordinal,
                        text,
                        (provenance,),
                        (part.part_id,),
                        metadata,
                    )
                )
                ordinal += 1
    if len(chunks) > max_chunks:
        raise LimitExceededError(f"chunk limit {max_chunks} would truncate {len(chunks)} chunks")
    return tuple(chunks)


def text_strategy_spans(
    text: str,
    policy: ChunkingPolicy,
    *,
    part_id: str = "text",
    document_metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> tuple[TextChunkSpan, ...]:
    """Return deterministic exact source spans for one explicit textual strategy."""

    if not text.strip():
        return ()
    strategy = policy.strategy
    if strategy is ChunkingStrategy.AUTO:
        raise InvalidDomainValueError("text strategy must be resolved before chunking")
    if strategy not in TEXT_CHUNKING_STRATEGIES:
        raise UnsupportedCapabilityError(
            f"strategy {strategy.value!r} is not valid for exact text",
            capability="text_chunking_strategy",
        )
    metadata = document_metadata or {}
    domain = _domain_name(strategy, metadata)
    if strategy is ChunkingStrategy.FIXED:
        spans = _fixed_spans(text, policy.max_chars)
    elif strategy is ChunkingStrategy.SLIDING_WINDOW:
        spans = _sliding_spans(text, policy.max_chars, policy.overlap_chars)
    elif strategy is ChunkingStrategy.RECURSIVE:
        paragraphs = _paragraph_spans(text)
        spans = tuple(
            span
            for start, end in paragraphs
            for span in _recursive_spans(text, start, end, policy.max_chars)
        )
    elif strategy is ChunkingStrategy.SENTENCE:
        spans = _pack_units(text, _sentence_spans(text), policy.max_chars)
    elif strategy is ChunkingStrategy.PARAGRAPH:
        spans = _pack_units(text, _paragraph_spans(text), policy.max_chars)
    elif strategy is ChunkingStrategy.SECTION:
        return _apply_parent_context(
            _section_strategy(text, policy, part_id, domain, hierarchical=False), policy, part_id
        )
    elif strategy is ChunkingStrategy.HIERARCHICAL:
        return _apply_parent_context(
            _section_strategy(text, policy, part_id, domain, hierarchical=True), policy, part_id
        )
    elif strategy is ChunkingStrategy.SEMANTIC:
        spans = _semantic_spans(text, policy.max_chars, policy.semantic_threshold)
    elif strategy is ChunkingStrategy.PROPOSITION:
        spans = _bounded_units(text, _proposition_spans(text), policy.max_chars)
    elif strategy is ChunkingStrategy.BOOK:
        return _apply_parent_context(_book_spans(text, policy, part_id), policy, part_id)
    elif strategy is ChunkingStrategy.LEGAL:
        return _apply_parent_context(_legal_spans(text, policy, part_id), policy, part_id)
    elif strategy is ChunkingStrategy.MEDICAL:
        return _apply_parent_context(_medical_spans(text, policy, part_id), policy, part_id)
    elif strategy is ChunkingStrategy.CODE:
        return _apply_parent_context(_code_spans(text, policy, part_id), policy, part_id)
    else:
        return _apply_parent_context(_conversation_spans(text, policy, part_id), policy, part_id)

    spans = _merge_short(text, spans, policy.min_chunk_chars, policy.max_chars)
    return _tagged(spans, part_id, domain)


def _domain_name(
    strategy: ChunkingStrategy,
    metadata: Mapping[str, str | int | float | bool | None],
) -> str:
    explicit = metadata.get("document_domain")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().casefold()
    if strategy in {
        ChunkingStrategy.BOOK,
        ChunkingStrategy.LEGAL,
        ChunkingStrategy.MEDICAL,
        ChunkingStrategy.CODE,
        ChunkingStrategy.CONVERSATION,
    }:
        return strategy.value
    return "text"


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _fixed_spans(text: str, limit: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + limit, len(text))
        span = _trim(text, cursor, end)
        if span is not None:
            spans.append(span)
        cursor = end
    return tuple(spans)


def _sliding_spans(text: str, limit: int, overlap: int) -> tuple[tuple[int, int], ...]:
    step = limit - overlap
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + limit, len(text))
        span = _trim(text, cursor, end)
        if span is not None and (not spans or span != spans[-1]):
            spans.append(span)
        if end == len(text):
            break
        cursor += step
    return tuple(spans)


def _recursive_spans(text: str, start: int, end: int, limit: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while True:
        remaining = _trim(text, cursor, end)
        if remaining is None:
            break
        cursor, remaining_end = remaining
        if remaining_end - cursor <= limit:
            spans.append((cursor, remaining_end))
            break
        window_end = cursor + limit
        boundary = -1
        for pattern in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"):
            matches = tuple(re.finditer(pattern, text[cursor : window_end + 1]))
            if matches:
                boundary = cursor + matches[-1].start()
                if boundary > cursor:
                    break
        if boundary <= cursor:
            boundary = window_end
        span = _trim(text, cursor, boundary)
        if span is not None:
            spans.append(span)
        cursor = boundary
    return tuple(spans)


def _sentence_spans(
    text: str, start: int = 0, end: int | None = None
) -> tuple[tuple[int, int], ...]:
    end = len(text) if end is None else end
    region = text[start:end]
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?s).*?(?:[.!?]+(?=\s|$)|\Z)", region):
        if not match.group(0):
            continue
        span = _trim(text, start + match.start(), start + match.end())
        if span is not None:
            spans.append(span)
    return tuple(spans)


def _paragraph_spans(
    text: str, start: int = 0, end: int | None = None
) -> tuple[tuple[int, int], ...]:
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    cursor = start
    for separator in re.finditer(r"\n\s*\n", text[start:end]):
        boundary = start + separator.start()
        span = _trim(text, cursor, boundary)
        if span is not None:
            spans.append(span)
        cursor = start + separator.end()
    span = _trim(text, cursor, end)
    if span is not None:
        spans.append(span)
    return tuple(spans)


def _pack_units(
    text: str, units: Sequence[tuple[int, int]], limit: int
) -> tuple[tuple[int, int], ...]:
    expanded: list[tuple[int, int]] = []
    for start, end in units:
        expanded.extend(_recursive_spans(text, start, end, limit))
    if not expanded:
        return ()
    packed: list[tuple[int, int]] = []
    current_start, current_end = expanded[0]
    for start, end in expanded[1:]:
        if end - current_start <= limit:
            current_end = end
        else:
            packed.append((current_start, current_end))
            current_start, current_end = start, end
    packed.append((current_start, current_end))
    return tuple(packed)


def _bounded_units(
    text: str, units: Sequence[tuple[int, int]], limit: int
) -> tuple[tuple[int, int], ...]:
    return tuple(
        bounded for start, end in units for bounded in _recursive_spans(text, start, end, limit)
    )


def _merge_short(
    text: str,
    spans: Sequence[tuple[int, int]],
    minimum: int,
    maximum: int,
) -> tuple[tuple[int, int], ...]:
    del text
    merged: list[tuple[int, int]] = []
    for span in spans:
        if (
            merged
            and span[1] - merged[-1][0] <= maximum
            and (span[1] - span[0] < minimum or merged[-1][1] - merged[-1][0] < minimum)
        ):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return tuple(merged)


def _tagged(spans: Sequence[tuple[int, int]], path: str, domain: str) -> tuple[TextChunkSpan, ...]:
    return tuple(TextChunkSpan(start, end, path, domain) for start, end in spans)


def _apply_parent_context(
    spans: tuple[TextChunkSpan, ...],
    policy: ChunkingPolicy,
    fallback: str,
) -> tuple[TextChunkSpan, ...]:
    if policy.include_parent_context:
        return spans
    return tuple(replace(span, structural_path=fallback) for span in spans)


def _heading_sections(text: str, pattern: re.Pattern[str]) -> tuple[tuple[int, int, str], ...]:
    headings = tuple(pattern.finditer(text))
    if not headings:
        trimmed = _trim(text, 0, len(text))
        return () if trimmed is None else ((trimmed[0], trimmed[1], ""),)
    sections: list[tuple[int, int, str]] = []
    prefix = _trim(text, 0, headings[0].start())
    if prefix is not None:
        sections.append((prefix[0], prefix[1], ""))
    hierarchy: dict[int, str] = {}
    for index, heading in enumerate(headings):
        level_text = heading.groupdict().get("level") or "#"
        level = len(level_text) if level_text.startswith("#") else 1
        title = (heading.groupdict().get("title") or heading.group(0)).strip(" #:\t")
        hierarchy[level] = title
        hierarchy = {key: value for key, value in hierarchy.items() if key <= level}
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        trimmed = _trim(text, heading.start(), end)
        if trimmed is not None:
            path = " / ".join(hierarchy[key] for key in sorted(hierarchy))
            sections.append((trimmed[0], trimmed[1], path))
    return tuple(sections)


_MARKDOWN_HEADING = re.compile(r"(?m)^(?P<level>#{1,6})[ \t]+(?P<title>[^\n]+?)[ \t]*$")


def _section_strategy(
    text: str,
    policy: ChunkingPolicy,
    fallback: str,
    domain: str,
    *,
    hierarchical: bool,
) -> tuple[TextChunkSpan, ...]:
    result: list[TextChunkSpan] = []
    for start, end, path in _heading_sections(text, _MARKDOWN_HEADING):
        units = _paragraph_spans(text, start, end) if hierarchical else ((start, end),)
        spans = (
            _bounded_units(text, units, policy.max_chars)
            if hierarchical
            else _pack_units(text, units, policy.max_chars)
        )
        spans = _merge_short(text, spans, policy.min_chunk_chars, policy.max_chars)
        result.extend(_tagged(spans, path or fallback, domain))
    return tuple(result)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _semantic_spans(text: str, limit: int, threshold: float) -> tuple[tuple[int, int], ...]:
    sentences: list[tuple[int, int]] = []
    for span in _sentence_spans(text):
        sentences.extend(_recursive_spans(text, span[0], span[1], limit))
    if not sentences:
        return ()
    groups: list[tuple[int, int]] = []
    start, end = sentences[0]
    group_tokens = _tokens(text[start:end])
    for next_start, next_end in sentences[1:]:
        next_tokens = _tokens(text[next_start:next_end])
        if next_end - start <= limit and _jaccard(group_tokens, next_tokens) >= threshold:
            end = next_end
            group_tokens |= next_tokens
        else:
            groups.append((start, end))
            start, end = next_start, next_end
            group_tokens = next_tokens
    groups.append((start, end))
    return tuple(groups)


def _proposition_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for start, end in _sentence_spans(text):
        cursor = start
        region = text[start:end]
        for boundary in re.finditer(r";\s+|,\s+(?=(?:and|but|or)\b)", region, re.IGNORECASE):
            split = start + boundary.end()
            span = _trim(text, cursor, split)
            if span is not None:
                spans.append(span)
            cursor = split
        span = _trim(text, cursor, end)
        if span is not None:
            spans.append(span)
    return tuple(spans)


_BOOK_HEADING = re.compile(
    r"(?im)^(?P<level>#+)[ \t]+(?P<title>[^\n]+)$|"
    r"^(?P<title_plain>(?:chapter|part|book)\s+"
    r"(?:[ivxlcdm]+|\d+|[a-z]+)(?:\s*[:.-]\s*[^\n]+)?)[ \t]*$"
)


def _book_spans(text: str, policy: ChunkingPolicy, fallback: str) -> tuple[TextChunkSpan, ...]:
    result: list[TextChunkSpan] = []
    for start, end, path in _book_sections(text):
        spans = _pack_units(text, _paragraph_spans(text, start, end), policy.max_chars)
        result.extend(
            _tagged(
                _merge_short(text, spans, policy.min_chunk_chars, policy.max_chars),
                path or fallback,
                "book",
            )
        )
    return tuple(result)


def _book_sections(text: str) -> tuple[tuple[int, int, str], ...]:
    matches = tuple(_BOOK_HEADING.finditer(text))
    if not matches:
        trimmed = _trim(text, 0, len(text))
        return () if trimmed is None else ((trimmed[0], trimmed[1], ""),)
    sections: list[tuple[int, int, str]] = []
    prefix = _trim(text, 0, matches[0].start())
    if prefix is not None:
        sections.append((prefix[0], prefix[1], ""))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        span = _trim(text, match.start(), end)
        if span is not None:
            title = (
                match.group("title") or match.group("title_plain") or _fallback_title(match.group())
            )
            sections.append((span[0], span[1], title.strip(" #\t")))
    return tuple(sections)


def _fallback_title(value: str) -> str:
    return value.splitlines()[0].strip()


_LEGAL_HEADING = re.compile(
    r"(?im)^(?:(?P<level>#+)[ \t]+(?P<title>[^\n]+)|"
    r"(?P<title_plain>(?:(?:article|section|clause|schedule)\s+[\w.-]+|"
    r"\d+(?:\.\d+)*[.)])(?:[ \t]+[^\n]+)?))[ \t]*$"
)


def _legal_spans(text: str, policy: ChunkingPolicy, fallback: str) -> tuple[TextChunkSpan, ...]:
    sections = _flat_named_sections(text, _LEGAL_HEADING)
    return _domain_sections(text, sections, policy, fallback, "legal", propositions=True)


_MEDICAL_HEADING = re.compile(
    r"(?im)^(?P<title_plain>(?:chief complaint|history(?: of present illness)?|"
    r"findings|assessment|diagnosis|medications?|allergies|plan|impression|"
    r"laboratory results?|vital signs))\s*:\s*(?:[^\n]*)$"
)


def _medical_spans(text: str, policy: ChunkingPolicy, fallback: str) -> tuple[TextChunkSpan, ...]:
    sections = _flat_named_sections(text, _MEDICAL_HEADING)
    return _domain_sections(text, sections, policy, fallback, "medical", propositions=False)


def _flat_named_sections(text: str, pattern: re.Pattern[str]) -> tuple[tuple[int, int, str], ...]:
    matches = tuple(pattern.finditer(text))
    if not matches:
        trimmed = _trim(text, 0, len(text))
        return () if trimmed is None else ((trimmed[0], trimmed[1], ""),)
    sections: list[tuple[int, int, str]] = []
    prefix = _trim(text, 0, matches[0].start())
    if prefix is not None:
        sections.append((prefix[0], prefix[1], ""))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        span = _trim(text, match.start(), end)
        if span is not None:
            title = match.groupdict().get("title") or match.groupdict().get("title_plain")
            sections.append((span[0], span[1], (title or match.group()).strip(" #:\t")))
    return tuple(sections)


def _domain_sections(
    text: str,
    sections: Sequence[tuple[int, int, str]],
    policy: ChunkingPolicy,
    fallback: str,
    domain: str,
    *,
    propositions: bool,
) -> tuple[TextChunkSpan, ...]:
    result: list[TextChunkSpan] = []
    for start, end, path in sections:
        units = (
            _proposition_spans_in_range(text, start, end)
            if propositions
            else _paragraph_spans(text, start, end)
        )
        spans = (
            _bounded_units(text, units, policy.max_chars)
            if propositions
            else _pack_units(text, units, policy.max_chars)
        )
        spans = _merge_short(text, spans, policy.min_chunk_chars, policy.max_chars)
        result.extend(_tagged(spans, path or fallback, domain))
    return tuple(result)


def _proposition_spans_in_range(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start + left, start + right) for left, right in _proposition_spans(text[start:end])
    )


_CODE_BOUNDARY = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?P<decl>"
    r"(?:async[ \t]+)?def[ \t]+[A-Za-z_]\w*|class[ \t]+[A-Za-z_]\w*|"
    r"(?:export[ \t]+)?(?:function|interface|enum|struct|trait|impl|fn)"
    r"[ \t]+[A-Za-z_]\w*|(?:public[ \t]+)?(?:class|interface|enum)"
    r"[ \t]+[A-Za-z_]\w*)"
)


def _code_spans(text: str, policy: ChunkingPolicy, fallback: str) -> tuple[TextChunkSpan, ...]:
    declarations = tuple(
        match for match in _CODE_BOUNDARY.finditer(text) if not match.group("indent")
    )
    if not declarations:
        return _tagged(_recursive_spans(text, 0, len(text), policy.max_chars), fallback, "code")
    result: list[TextChunkSpan] = []
    prefix = _trim(text, 0, declarations[0].start())
    if prefix is not None:
        result.extend(_tagged(_recursive_spans(text, *prefix, policy.max_chars), "module", "code"))
    for index, declaration in enumerate(declarations):
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
        spans = _recursive_spans(text, declaration.start(), end, policy.max_chars)
        result.extend(_tagged(spans, declaration.group("decl"), "code"))
    return tuple(result)


_SPEAKER = re.compile(r"(?m)^(?P<speaker>[A-Za-z][A-Za-z0-9 _.-]{0,48}):[ \t]*")


def _conversation_spans(
    text: str, policy: ChunkingPolicy, fallback: str
) -> tuple[TextChunkSpan, ...]:
    turns = tuple(_SPEAKER.finditer(text))
    if not turns:
        return _tagged(
            _recursive_spans(text, 0, len(text), policy.max_chars), fallback, "conversation"
        )
    result: list[TextChunkSpan] = []
    prefix = _trim(text, 0, turns[0].start())
    if prefix is not None:
        result.extend(
            _tagged(_recursive_spans(text, *prefix, policy.max_chars), fallback, "conversation")
        )
    for index, turn in enumerate(turns):
        end = turns[index + 1].start() if index + 1 < len(turns) else len(text)
        spans = _recursive_spans(text, turn.start(), end, policy.max_chars)
        result.extend(_tagged(spans, turn.group("speaker"), "conversation"))
    return tuple(result)
