"""Dependency-light textual classification, extraction, projection, and chunking."""

from __future__ import annotations

import re
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

from ragkit.domain import (
    Chunk,
    ChunkId,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionProvenance,
    IntegrityError,
    LimitExceededError,
    SourceId,
    TextContent,
    TextSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    Chunker,
    ChunkingRequest,
    DocumentExtractor,
    DocumentFamily,
    DocumentProjector,
    ExtractionRequest,
    FamilyClassifier,
    ProjectionRequest,
)

_CLASSIFIER = ComponentFingerprint.create("classifier", "stdlib_text", {"version": 1})
_EXTRACTOR = ComponentFingerprint.create("extractor", "stdlib_text", {"version": 1})
_TEXT_KINDS = {
    "text/plain": "plain",
    "text/markdown": "markdown",
    "text/html": "html",
    "message/rfc822": "email",
    "text/x-python": "code",
    "text/javascript": "code",
    "text/typescript": "code",
    "text/x-java-source": "code",
    "text/x-go": "code",
    "text/x-rust": "code",
    "text/x-c": "code",
    "text/x-c++": "code",
    "application/json": "code",
    "application/yaml": "code",
    "application/toml": "code",
}


class TextFamilyClassifier(FamilyClassifier):
    """Recognize the supported native textual formats without content guessing."""

    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        classified: list[AssetClassification] = []
        for asset in assets:
            if asset.reference.media_type not in _TEXT_KINDS:
                raise UnsupportedCapabilityError(
                    f"unsupported textual media type: {asset.reference.media_type}",
                    capability="text_media_type",
                )
            classified.append(
                AssetClassification(asset.reference.asset_id, DocumentFamily.TEXT, 1.0, _CLASSIFIER)
            )
        return tuple(classified)


class TextDocumentExtractor(DocumentExtractor):
    """Extract exact UTF-8 source slices from the five native textual subfamilies."""

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if len(request.assets) > request.max_documents:
            raise LimitExceededError(
                f"document count {len(request.assets)} exceeds limit {request.max_documents}"
            )
        documents: list[Document] = []
        for asset, classification in zip(request.assets, request.classifications, strict=True):
            if classification.family is not DocumentFamily.TEXT:
                raise UnsupportedCapabilityError(
                    f"unsupported extraction family: {classification.family.value}",
                    capability=classification.family.value,
                )
            try:
                decoded = asset.content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise IntegrityError("text asset is not valid UTF-8", cause=error) from error
            kind = _TEXT_KINDS.get(asset.reference.media_type)
            if kind is None:
                raise UnsupportedCapabilityError(
                    f"unsupported textual media type: {asset.reference.media_type}",
                    capability="text_media_type",
                )
            spans = _extract_spans(decoded, asset.content, kind)
            if not spans:
                raise IntegrityError("text asset contains no searchable text")
            source_id = SourceId.from_locator(
                "filesystem", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            parts = tuple(
                TextContent(
                    f"text-{index}",
                    decoded[start:end],
                    ExtractionProvenance(
                        asset.reference,
                        TextSpanLocator(start, end),
                        _EXTRACTOR,
                        1.0,
                    ),
                )
                for index, (start, end) in enumerate(spans)
            )
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    parts,
                    {
                        "text_kind": kind,
                        "source_uri": asset.reference.uri,
                        "file_name": Path(asset.reference.uri or "").name,
                    },
                )
            )
        return tuple(documents)


class NoOpDocumentProjector(DocumentProjector):
    """Return textual documents unchanged after enforcing the declared part limit."""

    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        oversized = next(
            (
                item
                for item in request.documents
                if len(item.parts) > request.max_parts_per_document
            ),
            None,
        )
        if oversized is not None:
            raise LimitExceededError(
                f"document part limit {request.max_parts_per_document} would truncate evidence"
            )
        return request.documents


class StructureAwareChunker(Chunker):
    """Split text at paragraphs/whitespace while retaining exact adjusted spans."""

    def __init__(self, max_chars: int = 800) -> None:
        if max_chars <= 0:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._fingerprint = ComponentFingerprint.create(
            "chunker", "structure_aware_text", {"version": 1, "max_chars": max_chars}
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for document in request.documents:
            ordinal = 0
            for part in document.parts:
                if not isinstance(part, TextContent) or not isinstance(
                    part.provenance.locator, TextSpanLocator
                ):
                    raise UnsupportedCapabilityError(
                        "structure-aware offline chunker supports exact textual parts only",
                        capability="text_chunking",
                    )
                for start, end in _bounded_spans(part.text, self._max_chars):
                    locator = TextSpanLocator(
                        part.provenance.locator.start + start,
                        part.provenance.locator.start + end,
                    )
                    text = part.text[start:end]
                    provenance = replace(part.provenance, locator=locator)
                    metadata = dict(document.metadata)
                    metadata["structural_path"] = _structural_path(part.text, start, part.part_id)
                    chunks.append(
                        Chunk(
                            ChunkId.from_content(
                                document.document_id,
                                self.fingerprint,
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
        if len(chunks) > request.max_chunks:
            raise LimitExceededError(
                f"chunk limit {request.max_chunks} would truncate {len(chunks)} chunks"
            )
        return tuple(chunks)


def _extract_spans(decoded: str, raw: bytes, kind: str) -> tuple[tuple[int, int], ...]:
    if kind in {"plain", "markdown", "code"}:
        return _nonblank_span(decoded)
    if kind == "email":
        message = BytesParser(policy=policy.default).parsebytes(raw)
        if message.is_multipart():
            raise UnsupportedCapabilityError(
                "multipart email requires a MIME projection adapter", capability="multipart_email"
            )
        encoding = (message.get("Content-Transfer-Encoding") or "7bit").casefold()
        if encoding not in {"7bit", "8bit", "binary"}:
            raise UnsupportedCapabilityError(
                "encoded email bodies are unsupported by exact-span extraction",
                capability="encoded_email_body",
            )
        separator = re.search(r"\r?\n\r?\n", decoded)
        if separator is None:
            raise IntegrityError("email has no header/body separator")
        body_start = separator.end()
        return tuple(
            (body_start + start, body_start + end)
            for start, end in _nonblank_span(decoded[body_start:])
        )

    parser = _ExactHTMLSpanParser(decoded)
    parser.feed(decoded)
    parser.close()
    return parser.spans


def _nonblank_span(text: str) -> tuple[tuple[int, int], ...]:
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return ((start, end),) if end > start else ()


class _ExactHTMLSpanParser(HTMLParser):
    """Collect exact source offsets using the standard parser's positions."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self._source = source
        self._line_offsets: list[int] = []
        offset = 0
        for line in source.splitlines(keepends=True):
            self._line_offsets.append(offset)
            offset += len(line)
        if not self._line_offsets:
            self._line_offsets.append(0)
        self._suppressed_depth = 0
        self._spans: list[tuple[int, int]] = []

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._spans)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() in {"script", "style"}:
            self._suppressed_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self._append_exact(data)

    def handle_entityref(self, name: str) -> None:
        if not self._suppressed_depth:
            self._append_exact(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self._suppressed_depth:
            self._append_exact(f"&#{name};")

    def _append_exact(self, value: str) -> None:
        line, column = self.getpos()
        start = self._line_offsets[line - 1] + column
        end = start + len(value)
        if self._source[start:end] != value:
            raise IntegrityError("HTML parser position did not resolve to exact source text")
        self._spans.extend(_trimmed_region(self._source, start, end))


def _trimmed_region(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    region = text[start:end]
    left = len(region) - len(region.lstrip())
    right = len(region.rstrip())
    return ((start + left, start + right),) if right > left else ()


def _bounded_spans(text: str, limit: int) -> tuple[tuple[int, int], ...]:
    candidates = [
        (match.start(), match.end())
        for match in re.finditer(r"(?s)\S(?:.*?\S)?(?=\n\s*\n|\Z)", text)
    ]
    spans: list[tuple[int, int]] = []
    for region_start, region_end in candidates:
        cursor = region_start
        while region_end - cursor > limit:
            boundary = text.rfind(" ", cursor, cursor + limit + 1)
            newline = text.rfind("\n", cursor, cursor + limit + 1)
            boundary = max(boundary, newline)
            if boundary <= cursor:
                boundary = cursor + limit
            end = boundary
            while end > cursor and text[end - 1].isspace():
                end -= 1
            if end > cursor:
                spans.append((cursor, end))
            cursor = boundary
            while cursor < region_end and text[cursor].isspace():
                cursor += 1
        if cursor < region_end:
            spans.append((cursor, region_end))
    return tuple(spans)


def _structural_path(text: str, offset: int, fallback: str) -> str:
    headings: dict[int, str] = {}
    for line in text[:offset].splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            headings[level] = match.group(2)
            headings = {key: value for key, value in headings.items() if key <= level}
    return " / ".join(headings[key] for key in sorted(headings)) or fallback
