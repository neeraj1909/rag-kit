"""Behavior fingerprints and immutable index compatibility manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from .errors import IndexCompatibilityError, InvalidDomainValueError
from .identity import ComponentFingerprint


class NormalizationMode(StrEnum):
    NONE = "none"
    L2 = "l2"


@dataclass(frozen=True, slots=True)
class ComponentManifest:
    kind: str
    implementation: str
    version: str
    configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.kind or not self.implementation or not self.version:
            raise InvalidDomainValueError("component identity fields must not be empty")
        object.__setattr__(self, "configuration", MappingProxyType(dict(self.configuration)))

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return ComponentFingerprint.create(
            self.kind,
            self.implementation,
            {"version": self.version, "configuration": self.configuration},
        )


@dataclass(frozen=True, slots=True)
class IndexManifest:
    schema_version: int
    corpus_fingerprint: ComponentFingerprint
    chunker_fingerprint: ComponentFingerprint
    embedder_fingerprint: ComponentFingerprint
    embedding_dimension: int
    normalization: NormalizationMode
    domain_schema_fingerprint: ComponentFingerprint

    def __post_init__(self) -> None:
        if self.schema_version < 1 or self.embedding_dimension < 1:
            raise InvalidDomainValueError("manifest schema and dimension must be positive")

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return ComponentFingerprint.create("index_manifest", "ragkit", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_fingerprint": str(self.corpus_fingerprint),
            "chunker_fingerprint": str(self.chunker_fingerprint),
            "embedder_fingerprint": str(self.embedder_fingerprint),
            "embedding_dimension": self.embedding_dimension,
            "normalization": self.normalization.value,
            "domain_schema_fingerprint": str(self.domain_schema_fingerprint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> IndexManifest:
        return cls(
            cast(int, value["schema_version"]),
            ComponentFingerprint(cast(str, value["corpus_fingerprint"])),
            ComponentFingerprint(cast(str, value["chunker_fingerprint"])),
            ComponentFingerprint(cast(str, value["embedder_fingerprint"])),
            cast(int, value["embedding_dimension"]),
            NormalizationMode(cast(str, value["normalization"])),
            ComponentFingerprint(cast(str, value["domain_schema_fingerprint"])),
        )

    def require_compatible(self, actual: IndexManifest) -> None:
        differences = compare_manifests(self, actual)
        if differences:
            raise IndexCompatibilityError(differences)


def compare_manifests(
    expected: IndexManifest, actual: IndexManifest
) -> dict[str, tuple[object, object]]:
    """Return non-secret field-level differences between two manifests."""

    differences: dict[str, tuple[object, object]] = {}
    for field in fields(IndexManifest):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        if expected_value != actual_value:
            differences[field.name] = (expected_value, actual_value)
    return differences
