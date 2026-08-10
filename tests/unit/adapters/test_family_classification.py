from __future__ import annotations

from hashlib import sha256

import pytest

from ragkit.adapters.classification import DeclaredFamilyClassifier
from ragkit.domain import AssetRef, UnsupportedCapabilityError
from ragkit.ports import AcquiredAsset, DocumentFamily

pytestmark = pytest.mark.unit


def _asset(media_type: str) -> AcquiredAsset:
    content = b"fixture"
    return AcquiredAsset(
        AssetRef("asset", media_type, sha256(content).hexdigest(), None, len(content)), content
    )


@pytest.mark.parametrize(
    ("family", "media_type"),
    [
        (DocumentFamily.OCR, "image/png"),
        (DocumentFamily.LAYOUT, "application/pdf"),
        (DocumentFamily.VISION, "image/jpeg"),
        (DocumentFamily.MEDIA, "audio/wav"),
    ],
)
def test_declared_family_classifier_accepts_only_profile_media_types(
    family: DocumentFamily, media_type: str
) -> None:
    result = DeclaredFamilyClassifier(family).classify((_asset(media_type),))

    assert result[0].family is family
    assert result[0].confidence == 1.0


def test_declared_family_classifier_rejects_cross_family_input() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="does not accept"):
        DeclaredFamilyClassifier(DocumentFamily.MEDIA).classify((_asset("image/png"),))
