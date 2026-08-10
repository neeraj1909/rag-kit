"""Strict standard-library TOML configuration for runnable profiles."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar, cast

from ragkit.domain import ComponentFingerprint, InvalidDomainValueError
from ragkit.ports import DocumentFamily


@dataclass(frozen=True, slots=True)
class ComponentSelections:
    connector: str
    classifier: str
    extractor: str
    projector: str
    chunker: str
    embedder: str
    vector_store: str
    reranker: str
    prompt_builder: str
    generator: str
    evaluator: str
    telemetry: str

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise InvalidDomainValueError("component selections must not be blank")


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_assets: int
    max_bytes_per_asset: int
    max_documents: int
    max_parts_per_document: int
    max_chunks: int
    chunk_chars: int
    embedding_dimension: int
    top_k: int
    max_context_chars: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(type(value) is not int or value <= 0 for value in values):
            raise InvalidDomainValueError("runtime limits must be positive")


@dataclass(frozen=True, slots=True)
class OfflineProfile:
    name: str
    family: DocumentFamily
    source: str
    components: ComponentSelections
    limits: RuntimeLimits

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.source, str)
            or not self.source.strip()
            or not isinstance(self.family, DocumentFamily)
        ):
            raise InvalidDomainValueError("profile name, source, and family must be valid")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": {"name": self.name, "family": self.family.value, "source": self.source},
            "components": asdict(self.components),
            "limits": asdict(self.limits),
        }

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return ComponentFingerprint.create("profile", "ragkit.toml", self.to_dict())


_T = TypeVar("_T")


def _typed_section(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InvalidDomainValueError(f"{label} must be a TOML table")
    return cast(dict[str, object], value)


def _construct_exact(
    model: type[_T], values: Mapping[str, object], expected: set[str], label: str
) -> _T:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise InvalidDomainValueError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    try:
        return model(**values)
    except (TypeError, ValueError) as error:
        raise InvalidDomainValueError(f"invalid {label}: {error}", cause=error) from error


def load_config(path: str | Path) -> OfflineProfile:
    """Load one exact, secret-free profile without importing optional adapters."""

    config_path = Path(path)
    try:
        decoded = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise InvalidDomainValueError(
            f"cannot load config {config_path}: {error}", cause=error
        ) from error
    if set(decoded) != {"profile", "components", "limits"}:
        raise InvalidDomainValueError("config must contain only profile, components, and limits")

    profile = _typed_section(decoded["profile"], "profile")
    if set(profile) != {"name", "family", "source"}:
        raise InvalidDomainValueError("profile fields must be name, family, and source")
    try:
        family = DocumentFamily(cast(str, profile["family"]))
    except (TypeError, ValueError) as error:
        raise InvalidDomainValueError(f"invalid document family: {profile['family']}") from error

    components = _construct_exact(
        ComponentSelections,
        _typed_section(decoded["components"], "components"),
        set(ComponentSelections.__dataclass_fields__),
        "components",
    )
    limits = _construct_exact(
        RuntimeLimits,
        _typed_section(decoded["limits"], "limits"),
        set(RuntimeLimits.__dataclass_fields__),
        "limits",
    )
    try:
        return OfflineProfile(
            name=cast(str, profile["name"]),
            family=family,
            source=cast(str, profile["source"]),
            components=components,
            limits=limits,
        )
    except (TypeError, ValueError) as error:
        raise InvalidDomainValueError(f"invalid profile: {error}", cause=error) from error
