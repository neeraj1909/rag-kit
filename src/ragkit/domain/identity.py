"""Canonical serialization and stable versioned identities."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from hashlib import sha256
from typing import ClassVar, Self

from .errors import InvalidDomainValueError


def _normalize(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        data = asdict(value)
        kind = getattr(value, "kind", None)
        if isinstance(kind, str):
            data["kind"] = kind
        return _normalize(data)
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidDomainValueError("canonical numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise InvalidDomainValueError("canonical object keys must be strings")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    raise InvalidDomainValueError(f"value of type {type(value).__name__} is not canonicalizable")


def canonical_json(value: object) -> str:
    """Return deterministic typed JSON after conservative text normalization."""

    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(namespace: str, payload: object) -> str:
    envelope = {"namespace": namespace, "payload": payload}
    return sha256(canonical_json(envelope).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StableId:
    """A full SHA-256 identifier whose prefix declares its type and scheme."""

    value: str
    prefix: ClassVar[str] = "id"
    namespace: ClassVar[str] = "ragkit:id:v1"

    def __post_init__(self) -> None:
        pattern = rf"{re.escape(self.prefix)}_v1_[0-9a-f]{{64}}"
        if re.fullmatch(pattern, self.value) is None:
            raise InvalidDomainValueError(f"invalid {type(self).__name__}")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        return cls(f"{cls.prefix}_v1_{_digest(cls.namespace, payload)}")


@dataclass(frozen=True, slots=True)
class SourceId(StableId):
    prefix: ClassVar[str] = "src"
    namespace: ClassVar[str] = "ragkit:source:v1"

    @classmethod
    def from_locator(cls, connector: str, locator: Mapping[str, object]) -> Self:
        if not connector:
            raise InvalidDomainValueError("connector must not be empty")
        return cls.from_payload({"connector": connector, "locator": locator})


@dataclass(frozen=True, slots=True)
class DocumentId(StableId):
    prefix: ClassVar[str] = "doc"
    namespace: ClassVar[str] = "ragkit:document:v1"

    @classmethod
    def from_assets(
        cls, source_id: SourceId, asset_digests: Sequence[str], *, boundary: str = ""
    ) -> Self:
        if not asset_digests:
            raise InvalidDomainValueError("a document requires at least one asset digest")
        if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in asset_digests):
            raise InvalidDomainValueError("asset digests must be full lowercase SHA-256 values")
        return cls.from_payload(
            {
                "source_id": str(source_id),
                "asset_digests": list(asset_digests),
                "boundary": boundary,
            }
        )


@dataclass(frozen=True, slots=True)
class ComponentFingerprint(StableId):
    prefix: ClassVar[str] = "cmp"
    namespace: ClassVar[str] = "ragkit:component:v1"

    @classmethod
    def create(cls, kind: str, implementation: str, configuration: Mapping[str, object]) -> Self:
        if not kind or not implementation:
            raise InvalidDomainValueError("component kind and implementation must not be empty")
        return cls.from_payload(
            {"kind": kind, "implementation": implementation, "configuration": configuration}
        )


@dataclass(frozen=True, slots=True)
class ChunkId(StableId):
    prefix: ClassVar[str] = "chk"
    namespace: ClassVar[str] = "ragkit:chunk:v1"

    @classmethod
    def from_content(
        cls,
        document_id: DocumentId,
        chunker: ComponentFingerprint,
        parts: Sequence[tuple[str, object]],
        representation: str,
    ) -> Self:
        if not parts or not representation:
            raise InvalidDomainValueError("chunk identity requires parts and a representation")
        return cls.from_payload(
            {
                "document_id": str(document_id),
                "chunker": str(chunker),
                "parts": [{"part_id": part_id, "locator": locator} for part_id, locator in parts],
                "representation": representation,
            }
        )
