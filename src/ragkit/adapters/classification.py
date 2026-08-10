"""Deterministic media-type classification for explicitly selected families."""

from __future__ import annotations

from ragkit.domain import ComponentFingerprint, UnsupportedCapabilityError
from ragkit.ports import AcquiredAsset, AssetClassification, DocumentFamily, FamilyClassifier

_MEDIA_TYPES: dict[DocumentFamily, frozenset[str]] = {
    DocumentFamily.OCR: frozenset(
        {"image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp", "application/pdf"}
    ),
    DocumentFamily.LAYOUT: frozenset(
        {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel.sheet.macroEnabled.12",
        }
    ),
    DocumentFamily.VISION: frozenset({"image/png", "image/jpeg", "image/webp"}),
    DocumentFamily.MEDIA: frozenset(
        {"audio/wav", "audio/flac", "audio/mpeg", "audio/mp4", "video/mp4", "video/webm"}
    ),
}


class DeclaredFamilyClassifier(FamilyClassifier):
    """Validate media types against one profile-selected non-text family.

    This classifier is deterministic, side-effect free, and thread-safe. It never
    guesses a family from bytes and never silently reroutes ambiguous PDFs.
    """

    def __init__(self, family: DocumentFamily) -> None:
        if family is DocumentFamily.TEXT:
            raise UnsupportedCapabilityError(
                "use TextFamilyClassifier for native text", capability="classifier:text"
            )
        self._family = family
        self._media_types = _MEDIA_TYPES[family]
        self._fingerprint = ComponentFingerprint.create(
            "classifier",
            "declared_family_media_type",
            {"version": 1, "family": family.value, "media_types": sorted(self._media_types)},
        )

    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        classifications: list[AssetClassification] = []
        for asset in assets:
            if asset.reference.media_type not in self._media_types:
                raise UnsupportedCapabilityError(
                    f"{self._family.value} profile does not accept {asset.reference.media_type}",
                    capability=f"{self._family.value}_media_type",
                )
            classifications.append(
                AssetClassification(
                    asset.reference.asset_id,
                    self._family,
                    1.0,
                    self._fingerprint,
                )
            )
        return tuple(classifications)
