from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import pytest

from ragkit.adapters.media import (
    LocalFasterWhisperTranscriber,
    MediaDocumentExtractor,
    PySceneDetectBackend,
    RawScene,
    RawTranscriptSegment,
)
from ragkit.adapters.torch_embedder import TorchTextEmbedder
from ragkit.adapters.vision import LocalSmolVLMBackend, VisionDescription, VisionDocumentExtractor
from ragkit.domain import (
    AssetRef,
    BoxLocator,
    ComponentFingerprint,
    ImageContent,
    IntegrityError,
    MediaContent,
    MissingDependencyError,
    OperationTimeoutError,
    PartialExtractionError,
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


def _asset(path: Path, media_type: str) -> AcquiredAsset:
    from hashlib import sha256

    content = path.read_bytes()
    return AcquiredAsset(
        AssetRef(
            path.stem,
            media_type,
            sha256(content).hexdigest(),
            path.resolve().as_uri(),
            len(content),
        ),
        content,
    )


def _request(asset: AcquiredAsset, family: DocumentFamily) -> ExtractionRequest:
    classifier = ComponentFingerprint.create("classifier", "fixture", {"version": 1})
    return ExtractionRequest(
        (asset,),
        (AssetClassification(asset.reference.asset_id, family, None, classifier),),
        1,
    )


@dataclass
class FakeVisionBackend:
    descriptions: tuple[VisionDescription, ...]
    calls: list[tuple[str, int]] = field(default_factory=list)
    fingerprint: ComponentFingerprint = field(
        default_factory=lambda: ComponentFingerprint.create(
            "vision_model",
            "fixture_vision",
            {"model": "fixture", "revision": "a" * 40},
        )
    )

    def describe(
        self, content: bytes, *, media_type: str, prompt: str, max_new_tokens: int
    ) -> tuple[VisionDescription, ...]:
        del content, media_type
        self.calls.append((prompt, max_new_tokens))
        return self.descriptions


@dataclass(frozen=True)
class FakeImageInspector:
    dimensions: tuple[int, int] = (640, 360)

    def inspect(self, content: bytes) -> tuple[int, int]:
        del content
        return self.dimensions


@pytest.mark.unit
def test_vision_keeps_original_region_and_marks_description_untrusted() -> None:
    path = Path("tests/fixtures/vision/equipment.png")
    asset = _asset(path, "image/png")
    backend = FakeVisionBackend(
        (VisionDescription("red isolation valve beside pressure gauge", (0.1, 0.2, 0.8, 0.9)),)
    )

    document = VisionDocumentExtractor(
        backend,
        prompt="Describe visible maintenance evidence.",
        max_new_tokens=64,
        inspector=FakeImageInspector(),
    ).extract(_request(asset, DocumentFamily.VISION))[0]

    part = document.parts[0]
    assert isinstance(part, ImageContent)
    assert part.provenance.asset == asset.reference
    assert part.provenance.locator == BoxLocator(0, 0.1, 0.2, 0.8, 0.9)
    assert part.provenance.confidence is None
    assert {notice.code for notice in part.provenance.notices} == {
        "model_derived",
        "confidence_unavailable",
        "untrusted_description",
    }
    assert document.metadata["vision_prompt"] == "Describe visible maintenance evidence."
    assert backend.calls == [("Describe visible maintenance evidence.", 64)]


@pytest.mark.unit
def test_vision_fails_when_capability_or_description_is_missing() -> None:
    asset = _asset(Path("tests/fixtures/vision/equipment.png"), "image/png")
    with pytest.raises(MissingDependencyError, match="cached revision-pinned vision backend"):
        VisionDocumentExtractor(None, inspector=FakeImageInspector()).extract(
            _request(asset, DocumentFamily.VISION)
        )
    with pytest.raises(IntegrityError, match="no visual descriptions"):
        VisionDocumentExtractor(FakeVisionBackend(()), inspector=FakeImageInspector()).extract(
            _request(asset, DocumentFamily.VISION)
        )


@pytest.mark.unit
def test_vision_deadline_interrupts_slow_backend() -> None:
    class SlowVisionBackend(FakeVisionBackend):
        def describe(
            self, content: bytes, *, media_type: str, prompt: str, max_new_tokens: int
        ) -> tuple[VisionDescription, ...]:
            sleep(0.1)
            return super().describe(
                content, media_type=media_type, prompt=prompt, max_new_tokens=max_new_tokens
            )

    asset = _asset(Path("tests/fixtures/vision/equipment.png"), "image/png")
    backend = SlowVisionBackend((VisionDescription("late", (0.0, 0.0, 1.0, 1.0)),))

    with pytest.raises(OperationTimeoutError, match="vision inference"):
        VisionDocumentExtractor(
            backend,
            timeout_seconds=0.01,
            inspector=FakeImageInspector(),
        ).extract(_request(asset, DocumentFamily.VISION))


@pytest.mark.unit
def test_concrete_media_and_vision_backends_require_pinned_revisions() -> None:
    with pytest.raises(ValueError, match="immutable 40-character"):
        LocalSmolVLMBackend(revision="main")
    with pytest.raises(ValueError, match="immutable 40-character"):
        LocalFasterWhisperTranscriber(revision="main")
    with pytest.raises(ValueError, match="multiple of 64"):
        LocalSmolVLMBackend(image_longest_edge=65)
    assert LocalSmolVLMBackend().fingerprint == ComponentFingerprint.create(
        "vision_model",
        "transformers_smolvlm",
        {
            "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
            "device": "cpu",
            "local_files_only": True,
            "trust_remote_code": False,
            "image_longest_edge": 256,
        },
    )
    assert LocalFasterWhisperTranscriber().fingerprint == ComponentFingerprint.create(
        "transcriber",
        "faster_whisper",
        {
            "model_id": "Systran/faster-whisper-tiny.en",
            "revision": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
            "device": "cpu",
            "compute_type": "int8",
            "local_files_only": True,
            "diarization": False,
        },
    )
    assert PySceneDetectBackend().fingerprint == ComponentFingerprint.create(
        "scene_detector",
        "pyscenedetect_content",
        {"version": 1, "threshold": 27.0, "keyframe": "midpoint"},
    )


@dataclass
class FakeTranscriber:
    segments: tuple[RawTranscriptSegment, ...]
    duration_ms: int = 2000
    fingerprint: ComponentFingerprint = field(
        default_factory=lambda: ComponentFingerprint.create(
            "transcriber", "fixture_asr", {"revision": "b" * 40}
        )
    )

    def transcribe(
        self, content: bytes, *, media_type: str
    ) -> tuple[int, tuple[RawTranscriptSegment, ...]]:
        del content, media_type
        return self.duration_ms, self.segments


@pytest.mark.unit
def test_media_deadline_interrupts_slow_transcriber() -> None:
    class SlowTranscriber(FakeTranscriber):
        def transcribe(
            self, content: bytes, *, media_type: str
        ) -> tuple[int, tuple[RawTranscriptSegment, ...]]:
            sleep(0.1)
            return super().transcribe(content, media_type=media_type)

    asset = _asset(Path("tests/fixtures/media/support.wav"), "audio/wav")
    transcriber = SlowTranscriber((RawTranscriptSegment(0.0, 1.0, "late"),))

    with pytest.raises(OperationTimeoutError, match="media transcription"):
        MediaDocumentExtractor(
            transcriber=transcriber,
            timeout_seconds=0.01,
            probe=FakeMediaProbe(),
        ).extract(_request(asset, DocumentFamily.MEDIA))


@dataclass
class FakeSceneDetector:
    duration_ms: int
    scenes: tuple[RawScene, ...]
    fingerprint: ComponentFingerprint = field(
        default_factory=lambda: ComponentFingerprint.create(
            "scene_detector", "fixture_scenes", {"version": 1}
        )
    )

    def detect(self, content: bytes, *, media_type: str) -> tuple[int, tuple[RawScene, ...]]:
        del content, media_type
        return self.duration_ms, self.scenes


@dataclass(frozen=True)
class FakeMediaProbe:
    duration: int = 2000

    def duration_ms(self, content: bytes, *, media_type: str) -> int:
        del content, media_type
        return self.duration


@pytest.mark.unit
def test_audio_normalizes_seconds_to_monotonic_half_open_milliseconds() -> None:
    asset = _asset(Path("tests/fixtures/media/support.wav"), "audio/wav")
    transcriber = FakeTranscriber(
        (
            RawTranscriptSegment(0.0, 0.751, "Pump alarm reported", language="en"),
            RawTranscriptSegment(0.751, 1.5, "Reset the isolation valve", language="en"),
        )
    )

    document = MediaDocumentExtractor(transcriber=transcriber, probe=FakeMediaProbe()).extract(
        _request(asset, DocumentFamily.MEDIA)
    )[0]

    parts = document.parts
    assert all(isinstance(part, MediaContent) for part in parts)
    assert [part.provenance.locator for part in parts] == [
        TimeSpanLocator(0, 751),
        TimeSpanLocator(751, 1500),
    ]
    assert all(part.provenance.confidence is None for part in parts)
    assert all("speaker_unknown" in {n.code for n in part.provenance.notices} for part in parts)
    assert "speaker" not in document.metadata


@pytest.mark.unit
def test_media_rejects_non_monotonic_transcript_and_partial_video() -> None:
    audio = _asset(Path("tests/fixtures/media/support.wav"), "audio/wav")
    bad = FakeTranscriber(
        (
            RawTranscriptSegment(1.0, 2.0, "later"),
            RawTranscriptSegment(0.0, 0.5, "earlier"),
        )
    )
    with pytest.raises(IntegrityError, match="monotonic"):
        MediaDocumentExtractor(transcriber=bad, probe=FakeMediaProbe()).extract(
            _request(audio, DocumentFamily.MEDIA)
        )

    beyond_duration = FakeTranscriber(
        (RawTranscriptSegment(0.0, 2.1, "too long"),), duration_ms=2000
    )
    with pytest.raises(IntegrityError, match="bounded by media duration"):
        MediaDocumentExtractor(transcriber=beyond_duration, probe=FakeMediaProbe()).extract(
            _request(audio, DocumentFamily.MEDIA)
        )

    video = _asset(Path("tests/fixtures/media/training.mp4"), "video/mp4")
    with pytest.raises(PartialExtractionError, match="scene/keyframe"):
        MediaDocumentExtractor(transcriber=FakeTranscriber(()), probe=FakeMediaProbe()).extract(
            _request(video, DocumentFamily.MEDIA)
        )


@pytest.mark.unit
def test_video_links_bounded_keyframes_to_scenes_without_speaker_identity() -> None:
    asset = _asset(Path("tests/fixtures/media/training.mp4"), "video/mp4")
    transcriber = FakeTranscriber((RawTranscriptSegment(0.0, 2.0, "Close the valve"),))
    detector = FakeSceneDetector(2000, (RawScene(0, 2000, 900, 27),))

    document = MediaDocumentExtractor(
        transcriber=transcriber, scene_detector=detector, probe=FakeMediaProbe()
    ).extract(_request(asset, DocumentFamily.MEDIA))[0]

    scene = next(part for part in document.parts if part.part_id.startswith("scene-"))
    keyframe = next(part for part in document.parts if part.part_id.startswith("keyframe-"))
    assert scene.provenance.locator == TimeSpanLocator(0, 2000)
    assert keyframe.relations[0].kind is RelationKind.KEYFRAME_OF
    assert keyframe.relations[0].target_part_id == scene.part_id
    assert keyframe.provenance.asset == asset.reference


@dataclass
class FakeTorchBackend:
    dimension: int = 3
    eval_calls: int = 0
    inference_entries: int = 0
    calls: list[tuple[tuple[str, ...], str, int, bool, str]] = field(default_factory=list)

    def eval(self) -> None:
        self.eval_calls += 1

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        self.inference_entries += 1
        yield

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        device: str,
        max_length: int,
        truncation: bool,
        pooling: str,
    ) -> Sequence[Sequence[float]]:
        self.calls.append((tuple(texts), device, max_length, truncation, pooling))
        return tuple((3.0, 4.0, 0.0) for _ in texts)


@pytest.mark.unit
def test_torch_embedder_fingerprints_runtime_and_uses_eval_inference_batches() -> None:
    backend = FakeTorchBackend()
    embedder = TorchTextEmbedder(
        model_id="fixture/tiny-encoder",
        revision="c" * 40,
        device="cpu",
        batch_size=2,
        max_length=32,
        pooling="mean",
        normalize=True,
        backend=backend,
    )

    batch = embedder.embed_documents(EmbeddingRequest(("alpha", "beta", "gamma")))

    assert backend.eval_calls == 1
    assert backend.inference_entries == 2
    assert [call[0] for call in backend.calls] == [("alpha", "beta"), ("gamma",)]
    assert all(call[1:] == ("cpu", 32, True, "mean") for call in backend.calls)
    assert all(embedding.values == pytest.approx((0.6, 0.8, 0.0)) for embedding in batch.embeddings)
    assert all(embedding.normalized for embedding in batch.embeddings)
    assert embedder.dimension == 3
    assert embedder.fingerprint == ComponentFingerprint.create(
        "embedder",
        "torch_transformers",
        {
            "model_id": "fixture/tiny-encoder",
            "revision": "c" * 40,
            "device": "cpu",
            "batch_size": 2,
            "max_length": 32,
            "truncation": True,
            "pooling": "mean",
            "normalization": "l2",
            "dimension": 3,
        },
    )


@pytest.mark.unit
def test_torch_embedder_requires_immutable_revision_and_cached_model() -> None:
    with pytest.raises(ValueError, match="immutable 40-character"):
        TorchTextEmbedder(model_id="fixture/model", revision="main", backend=FakeTorchBackend())
    embedder = TorchTextEmbedder(model_id="fixture/model", revision="d" * 40)
    with pytest.raises(MissingDependencyError, match="local_files_only"):
        embedder.embed_query("alpha")
