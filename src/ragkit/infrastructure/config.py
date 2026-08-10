"""Strict standard-library TOML configuration for runnable profiles."""

from __future__ import annotations

import math
import re
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
    retriever: str
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
class AdapterSettings:
    """Explicit optional-adapter settings; credential values are never stored here."""

    ocr_language: str = "eng"
    ocr_max_pages: int = 25
    ocr_max_pixels: int = 20_000_000
    ocr_timeout_seconds: float = 30.0
    layout_max_pages: int = 100
    layout_max_slides: int = 100
    layout_max_sheets: int = 20
    layout_max_cells: int = 100_000
    layout_max_archive_bytes: int = 256 * 1024 * 1024
    layout_max_compression_ratio: float = 100.0
    persistence_path: str = ".ragkit/chroma"
    collection_name: str = "ragkit"
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    device: str = "cpu"
    batch_size: int = 8
    max_length: int = 256
    pooling: str = "mean"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_token_pattern: str = r"[^\W_]+"
    bm25_lowercase: bool = True
    hybrid_rrf_k: int = 60
    hybrid_candidate_multiplier: int = 4
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_max_top_k: int = 100
    reranker_max_candidates: int = 1_000
    vision_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    vision_revision: str = "7e3e67edbbed1bf9888184d9df282b700a323964"
    vision_max_new_tokens: int = 8
    vision_image_longest_edge: int = 128
    vision_max_pixels: int = 4_194_304
    vision_max_dimension: int = 2048
    vision_max_regions: int = 20
    vision_timeout_seconds: float = 60.0
    media_model_id: str = "Systran/faster-whisper-tiny.en"
    media_revision: str = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
    media_max_duration_ms: int = 30 * 60 * 1000
    media_max_segments: int = 200
    media_max_scenes: int = 100
    media_timeout_seconds: float = 2 * 60 * 60
    hosted_model: str = "gpt-5-mini"
    credential_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        phase4_integers = (
            self.hybrid_rrf_k,
            self.hybrid_candidate_multiplier,
            self.reranker_batch_size,
            self.reranker_max_length,
            self.reranker_max_top_k,
            self.reranker_max_candidates,
        )
        if any(type(value) is not int for value in phase4_integers):
            raise InvalidDomainValueError("Phase 4 count and limit settings must be integers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (self.bm25_k1, self.bm25_b)
        ):
            raise InvalidDomainValueError("BM25 numeric settings must be finite numbers")
        strings = (
            self.ocr_language,
            self.persistence_path,
            self.collection_name,
            self.embedder_model_id,
            self.vision_model_id,
            self.media_model_id,
            self.reranker_model_id,
            self.bm25_token_pattern,
            self.hosted_model,
            self.credential_env,
        )
        if any(not value.strip() for value in strings):
            raise InvalidDomainValueError("adapter string settings must not be blank")
        if type(self.bm25_lowercase) is not bool:
            raise InvalidDomainValueError("bm25_lowercase must be a boolean")
        try:
            re.compile(self.bm25_token_pattern)
        except re.error as error:
            raise InvalidDomainValueError("bm25_token_pattern must be a valid regex") from error
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.credential_env) is None:
            raise InvalidDomainValueError("credential_env must be an environment-variable name")
        revisions = (
            self.embedder_revision,
            self.vision_revision,
            self.media_revision,
            self.reranker_revision,
        )
        if any(
            len(value) != 40 or any(char not in "0123456789abcdef" for char in value)
            for value in revisions
        ):
            raise InvalidDomainValueError("model revisions must be immutable 40-character SHAs")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise InvalidDomainValueError("device must be cpu, cuda, or mps")
        if self.pooling not in {"mean", "cls"}:
            raise InvalidDomainValueError("pooling must be mean or cls")
        if (
            min(
                self.batch_size,
                self.max_length,
                self.vision_max_new_tokens,
                self.vision_image_longest_edge,
                self.ocr_max_pages,
                self.ocr_max_pixels,
                self.layout_max_pages,
                self.layout_max_slides,
                self.layout_max_sheets,
                self.layout_max_cells,
                self.layout_max_archive_bytes,
                self.vision_max_pixels,
                self.vision_max_dimension,
                self.vision_max_regions,
                self.media_max_duration_ms,
                self.media_max_segments,
                self.media_max_scenes,
                self.hybrid_rrf_k,
                self.hybrid_candidate_multiplier,
                self.reranker_batch_size,
                self.reranker_max_length,
                self.reranker_max_top_k,
                self.reranker_max_candidates,
            )
            <= 0
            or min(
                self.timeout_seconds,
                self.ocr_timeout_seconds,
                self.vision_timeout_seconds,
                self.media_timeout_seconds,
                self.layout_max_compression_ratio,
                self.bm25_k1,
                self.bm25_b,
            )
            <= 0
        ):
            raise InvalidDomainValueError("adapter numeric limits must be positive")
        if self.bm25_b > 1.0:
            raise InvalidDomainValueError("bm25_b must not exceed 1")
        if self.reranker_max_top_k > self.reranker_max_candidates:
            raise InvalidDomainValueError(
                "reranker_max_top_k must not exceed reranker_max_candidates"
            )
        if self.reranker_batch_size > 128 or self.reranker_max_length > 512:
            raise InvalidDomainValueError(
                "reranker batch size and sequence length exceed supported bounds"
            )
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise InvalidDomainValueError("max_retries must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OfflineProfile:
    name: str
    family: DocumentFamily
    source: str
    components: ComponentSelections
    limits: RuntimeLimits
    settings: AdapterSettings = AdapterSettings()

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
            "settings": asdict(self.settings),
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
    allowed_sections = {"profile", "components", "limits", "settings"}
    if not {"profile", "components", "limits"}.issubset(decoded) or not set(decoded).issubset(
        allowed_sections
    ):
        raise InvalidDomainValueError(
            "config requires profile, components, and limits; settings is optional"
        )

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
    raw_settings = _typed_section(decoded.get("settings", {}), "settings")
    unknown_settings = set(raw_settings) - set(AdapterSettings.__dataclass_fields__)
    if unknown_settings:
        raise InvalidDomainValueError(
            f"settings fields mismatch; missing=[], unknown={sorted(unknown_settings)}"
        )
    try:
        defaults = AdapterSettings()
        settings = AdapterSettings(
            ocr_language=cast(str, raw_settings.get("ocr_language", defaults.ocr_language)),
            ocr_max_pages=cast(int, raw_settings.get("ocr_max_pages", defaults.ocr_max_pages)),
            ocr_max_pixels=cast(int, raw_settings.get("ocr_max_pixels", defaults.ocr_max_pixels)),
            ocr_timeout_seconds=cast(
                float,
                raw_settings.get("ocr_timeout_seconds", defaults.ocr_timeout_seconds),
            ),
            layout_max_pages=cast(
                int, raw_settings.get("layout_max_pages", defaults.layout_max_pages)
            ),
            layout_max_slides=cast(
                int, raw_settings.get("layout_max_slides", defaults.layout_max_slides)
            ),
            layout_max_sheets=cast(
                int, raw_settings.get("layout_max_sheets", defaults.layout_max_sheets)
            ),
            layout_max_cells=cast(
                int, raw_settings.get("layout_max_cells", defaults.layout_max_cells)
            ),
            layout_max_archive_bytes=cast(
                int,
                raw_settings.get("layout_max_archive_bytes", defaults.layout_max_archive_bytes),
            ),
            layout_max_compression_ratio=cast(
                float,
                raw_settings.get(
                    "layout_max_compression_ratio", defaults.layout_max_compression_ratio
                ),
            ),
            persistence_path=cast(
                str, raw_settings.get("persistence_path", defaults.persistence_path)
            ),
            collection_name=cast(
                str, raw_settings.get("collection_name", defaults.collection_name)
            ),
            embedder_model_id=cast(
                str, raw_settings.get("embedder_model_id", defaults.embedder_model_id)
            ),
            embedder_revision=cast(
                str, raw_settings.get("embedder_revision", defaults.embedder_revision)
            ),
            device=cast(str, raw_settings.get("device", defaults.device)),
            batch_size=cast(int, raw_settings.get("batch_size", defaults.batch_size)),
            max_length=cast(int, raw_settings.get("max_length", defaults.max_length)),
            pooling=cast(str, raw_settings.get("pooling", defaults.pooling)),
            bm25_k1=cast(float, raw_settings.get("bm25_k1", defaults.bm25_k1)),
            bm25_b=cast(float, raw_settings.get("bm25_b", defaults.bm25_b)),
            bm25_token_pattern=cast(
                str, raw_settings.get("bm25_token_pattern", defaults.bm25_token_pattern)
            ),
            bm25_lowercase=cast(bool, raw_settings.get("bm25_lowercase", defaults.bm25_lowercase)),
            hybrid_rrf_k=cast(int, raw_settings.get("hybrid_rrf_k", defaults.hybrid_rrf_k)),
            hybrid_candidate_multiplier=cast(
                int,
                raw_settings.get(
                    "hybrid_candidate_multiplier", defaults.hybrid_candidate_multiplier
                ),
            ),
            reranker_model_id=cast(
                str, raw_settings.get("reranker_model_id", defaults.reranker_model_id)
            ),
            reranker_revision=cast(
                str, raw_settings.get("reranker_revision", defaults.reranker_revision)
            ),
            reranker_batch_size=cast(
                int, raw_settings.get("reranker_batch_size", defaults.reranker_batch_size)
            ),
            reranker_max_length=cast(
                int, raw_settings.get("reranker_max_length", defaults.reranker_max_length)
            ),
            reranker_max_top_k=cast(
                int, raw_settings.get("reranker_max_top_k", defaults.reranker_max_top_k)
            ),
            reranker_max_candidates=cast(
                int,
                raw_settings.get("reranker_max_candidates", defaults.reranker_max_candidates),
            ),
            vision_model_id=cast(
                str, raw_settings.get("vision_model_id", defaults.vision_model_id)
            ),
            vision_revision=cast(
                str, raw_settings.get("vision_revision", defaults.vision_revision)
            ),
            vision_max_new_tokens=cast(
                int,
                raw_settings.get("vision_max_new_tokens", defaults.vision_max_new_tokens),
            ),
            vision_image_longest_edge=cast(
                int,
                raw_settings.get("vision_image_longest_edge", defaults.vision_image_longest_edge),
            ),
            vision_max_pixels=cast(
                int, raw_settings.get("vision_max_pixels", defaults.vision_max_pixels)
            ),
            vision_max_dimension=cast(
                int, raw_settings.get("vision_max_dimension", defaults.vision_max_dimension)
            ),
            vision_max_regions=cast(
                int, raw_settings.get("vision_max_regions", defaults.vision_max_regions)
            ),
            vision_timeout_seconds=cast(
                float,
                raw_settings.get("vision_timeout_seconds", defaults.vision_timeout_seconds),
            ),
            media_model_id=cast(str, raw_settings.get("media_model_id", defaults.media_model_id)),
            media_revision=cast(str, raw_settings.get("media_revision", defaults.media_revision)),
            media_max_duration_ms=cast(
                int,
                raw_settings.get("media_max_duration_ms", defaults.media_max_duration_ms),
            ),
            media_max_segments=cast(
                int, raw_settings.get("media_max_segments", defaults.media_max_segments)
            ),
            media_max_scenes=cast(
                int, raw_settings.get("media_max_scenes", defaults.media_max_scenes)
            ),
            media_timeout_seconds=cast(
                float,
                raw_settings.get("media_timeout_seconds", defaults.media_timeout_seconds),
            ),
            hosted_model=cast(str, raw_settings.get("hosted_model", defaults.hosted_model)),
            credential_env=cast(str, raw_settings.get("credential_env", defaults.credential_env)),
            timeout_seconds=cast(
                float, raw_settings.get("timeout_seconds", defaults.timeout_seconds)
            ),
            max_retries=cast(int, raw_settings.get("max_retries", defaults.max_retries)),
        )
    except (TypeError, ValueError) as error:
        raise InvalidDomainValueError(f"invalid settings: {error}", cause=error) from error
    try:
        return OfflineProfile(
            name=cast(str, profile["name"]),
            family=family,
            source=cast(str, profile["source"]),
            components=components,
            limits=limits,
            settings=settings,
        )
    except (TypeError, ValueError) as error:
        raise InvalidDomainValueError(f"invalid profile: {error}", cause=error) from error
