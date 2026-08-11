from __future__ import annotations

import math
import os
from hashlib import sha256
from pathlib import Path

import pytest

from ragkit.adapters import (
    LocalFasterWhisperTranscriber,
    LocalSmolVLMBackend,
    MediaDocumentExtractor,
    PySceneDetectBackend,
    TorchTextEmbedder,
    VisionDocumentExtractor,
)
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    ComponentFingerprint,
    ImageContent,
    KeyframeLocator,
    RelationKind,
    TimeSpanLocator,
)
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    DocumentFamily,
    EmbeddingRequest,
    ExtractionRequest,
)


@pytest.mark.integration
@pytest.mark.modality_integration
def test_phase3_vision_and_media_fixtures_are_explicitly_synthetic() -> None:
    for fixture in (Path("tests/fixtures/vision"), Path("tests/fixtures/media")):
        notice = (fixture / "README.md").read_text()
        assert "synthetic" in notice.casefold()
        assert "redistributable" in notice.casefold()


@pytest.mark.integration
@pytest.mark.modality_integration
def test_reviewed_cached_cpu_transformer_is_repeatable() -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip(
            "Set RAGKIT_RUN_MODEL_INTEGRATION=1 after explicitly provisioning the reviewed "
            "revision documented in docs/recipes/equipment-vision.md."
        )
    embedder = TorchTextEmbedder(device="cpu", batch_size=2, max_length=64)

    first = embedder.embed_documents(EmbeddingRequest(("revenue increased", "revenue declined")))
    second = embedder.embed_documents(EmbeddingRequest(("revenue increased", "revenue declined")))

    assert embedder.dimension == 384
    assert first.embedder == second.embedder == embedder.fingerprint
    for left, right in zip(first.embeddings, second.embeddings, strict=True):
        assert left.values == pytest.approx(right.values, abs=1e-7)
        assert math.sqrt(sum(value * value for value in left.values)) == pytest.approx(1.0)


@pytest.mark.integration
@pytest.mark.modality_integration
def test_reviewed_vision_backend_generates_region_linked_evidence() -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("explicit reviewed model provisioning is required")
    path = Path("tests/fixtures/vision/equipment.png").resolve()
    content = path.read_bytes()
    reference = AssetRef(
        "equipment",
        "image/png",
        sha256(content).hexdigest(),
        path.as_uri(),
        len(content),
    )
    classification = AssetClassification(
        reference.asset_id,
        DocumentFamily.VISION,
        1.0,
        ComponentFingerprint.create("classifier", "fixture", {"version": 1}),
    )
    document = VisionDocumentExtractor(
        LocalSmolVLMBackend(image_longest_edge=64),
        max_new_tokens=4,
        timeout_seconds=60.0,
    ).extract(ExtractionRequest((AcquiredAsset(reference, content),), (classification,), 1))[0]

    assert len(document.parts) == 1
    part = document.parts[0]
    assert isinstance(part, ImageContent)
    assert part.description.strip()
    assert part.provenance.locator == BoxLocator(0, 0.0, 0.0, 1.0, 1.0)
    assert {notice.code for notice in part.provenance.notices} == {
        "confidence_unavailable",
        "model_derived",
        "untrusted_description",
    }


@pytest.mark.integration
@pytest.mark.modality_integration
def test_reviewed_media_backends_decode_transcript_scenes_and_keyframes() -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("explicit reviewed model provisioning is required")
    path = Path("tests/fixtures/media/training.mp4").resolve()
    content = path.read_bytes()
    reference = AssetRef(
        "training",
        "video/mp4",
        sha256(content).hexdigest(),
        path.as_uri(),
        len(content),
    )
    classification = AssetClassification(
        reference.asset_id,
        DocumentFamily.MEDIA,
        1.0,
        ComponentFingerprint.create("classifier", "fixture", {"version": 1}),
    )
    document = MediaDocumentExtractor(
        transcriber=LocalFasterWhisperTranscriber(),
        scene_detector=PySceneDetectBackend(),
    ).extract(ExtractionRequest((AcquiredAsset(reference, content),), (classification,), 1))[0]

    assert any(isinstance(part.provenance.locator, TimeSpanLocator) for part in document.parts)
    assert any(isinstance(part.provenance.locator, KeyframeLocator) for part in document.parts)
    assert any(
        relation.kind is RelationKind.KEYFRAME_OF
        for part in document.parts
        for relation in part.relations
    )
