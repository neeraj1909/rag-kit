"""Time-based media normalization behind deterministic injected engines."""

from __future__ import annotations

import math
import os
import re
import tempfile
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Protocol, cast

from ragkit.domain import (
    AssetRef,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionNotice,
    ExtractionProvenance,
    IntegrityError,
    KeyframeLocator,
    LimitExceededError,
    MediaContent,
    MissingDependencyError,
    PartialExtractionError,
    PartRelation,
    RelationKind,
    SourceId,
    TimeSpanLocator,
    UnsupportedCapabilityError,
)
from ragkit.ports import DocumentExtractor, DocumentFamily, ExtractionRequest

from ._deadline import run_with_deadline


@dataclass(frozen=True, slots=True)
class RawTranscriptSegment:
    """Provider transcript interval in seconds; speaker identity is intentionally absent."""

    start_seconds: float
    end_seconds: float
    text: str
    language: str | None = None


@dataclass(frozen=True, slots=True)
class RawScene:
    """Detected scene and representative keyframe positions in milliseconds."""

    start_ms: int
    end_ms: int
    keyframe_ms: int
    frame_number: int


class Transcriber(Protocol):
    @property
    def fingerprint(self) -> ComponentFingerprint: ...

    def transcribe(
        self, content: bytes, *, media_type: str
    ) -> tuple[int, Sequence[RawTranscriptSegment]]: ...


class SceneDetector(Protocol):
    @property
    def fingerprint(self) -> ComponentFingerprint: ...

    def detect(self, content: bytes, *, media_type: str) -> tuple[int, Sequence[RawScene]]: ...


class MediaProbe(Protocol):
    def duration_ms(self, content: bytes, *, media_type: str) -> int: ...


class _ContainerMediaProbe:
    def duration_ms(self, content: bytes, *, media_type: str) -> int:
        return _probe_duration_ms(content, media_type)


class _WhisperSegment(Protocol):
    start: float
    end: float
    text: str


class _WhisperInfo(Protocol):
    duration: float
    language: str


class _WhisperRuntime(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        beam_size: int,
        vad_filter: bool,
        word_timestamps: bool,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]: ...


WHISPER_MODEL_ID = "Systran/faster-whisper-tiny.en"
WHISPER_MODEL_REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"


class LocalFasterWhisperTranscriber:
    """CPU/int8 ASR using only a locally cached immutable model revision."""

    def __init__(
        self,
        *,
        model_id: str = WHISPER_MODEL_ID,
        revision: str = WHISPER_MODEL_REVISION,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("ASR revision must be an immutable 40-character SHA")
        self._model_id = model_id
        self._revision = revision
        self._model: _WhisperRuntime | None = None
        self._fingerprint = ComponentFingerprint.create(
            "transcriber",
            "faster_whisper",
            {
                "model_id": model_id,
                "revision": revision,
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
                "diarization": False,
            },
        )

    def _ensure_model(self) -> _WhisperRuntime:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel(
                self._model_id,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
                revision=self._revision,
            )
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "local ASR requires the 'media' extra with faster-whisper"
            ) from error
        except (OSError, RuntimeError) as error:
            raise MissingDependencyError(
                f"ASR model {self._model_id}@{self._revision} is not cached; provision "
                "and review that revision before constructing the transcriber"
            ) from error
        self._model = cast(_WhisperRuntime, model)
        return self._model

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def transcribe(
        self, content: bytes, *, media_type: str
    ) -> tuple[int, tuple[RawTranscriptSegment, ...]]:
        model = self._ensure_model()
        path = _write_private_media(content, _media_suffix(media_type))
        try:
            segments, info = model.transcribe(
                path,
                beam_size=1,
                vad_filter=False,
                word_timestamps=False,
            )
            normalized = tuple(
                RawTranscriptSegment(
                    segment.start,
                    segment.end,
                    segment.text,
                    language=info.language,
                )
                for segment in segments
            )
            return round(info.duration * 1000), normalized
        except (OSError, RuntimeError, ValueError) as error:
            raise IntegrityError(
                "media could not be decoded or transcribed", cause=error
            ) from error
        finally:
            os.unlink(path)


class PySceneDetectBackend:
    """Detect bounded hard-cut scenes and select deterministic midpoint keyframes."""

    def __init__(self, *, threshold: float = 27.0) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("scene threshold must be finite and positive")
        self._threshold = threshold
        self._fingerprint = ComponentFingerprint.create(
            "scene_detector",
            "pyscenedetect_content",
            {"version": 1, "threshold": threshold, "keyframe": "midpoint"},
        )

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def detect(self, content: bytes, *, media_type: str) -> tuple[int, tuple[RawScene, ...]]:
        if media_type not in {"video/mp4", "video/webm"}:
            raise UnsupportedCapabilityError(
                f"scene detection does not support {media_type}", capability="video_scene"
            )
        try:
            from scenedetect import (  # type: ignore
                ContentDetector,
                SceneManager,
                VideoOpenFailure,
                open_video,
            )
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "video scenes require the 'media' extra with scenedetect[opencv]"
            ) from error
        path = _write_private_media(content, _media_suffix(media_type))
        try:
            video = open_video(path)
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=self._threshold))
            manager.detect_scenes(video, show_progress=False)
            duration_ms = round(video.duration.get_seconds() * 1000)
            scenes = tuple(
                RawScene(
                    round(start.get_seconds() * 1000),
                    round(end.get_seconds() * 1000),
                    round(((start.get_seconds() + end.get_seconds()) / 2) * 1000),
                    (start.get_frames() + end.get_frames()) // 2,
                )
                for start, end in manager.get_scene_list(start_in_scene=True)
            )
            return duration_ms, scenes
        except (OSError, RuntimeError, ValueError, VideoOpenFailure) as error:
            raise IntegrityError(
                "video could not be decoded for scene detection", cause=error
            ) from error
        finally:
            os.unlink(path)


def _media_suffix(media_type: str) -> str:
    suffixes = {
        "audio/wav": ".wav",
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }
    try:
        return suffixes[media_type]
    except KeyError as error:
        raise UnsupportedCapabilityError(
            f"unsupported media type: {media_type}", capability="media_type"
        ) from error


def _write_private_media(content: bytes, suffix: str) -> str:
    descriptor, path = tempfile.mkstemp(prefix="ragkit-media-", suffix=suffix)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except BaseException:
        os.unlink(path)
        raise
    return path


def _probe_duration_ms(content: bytes, media_type: str) -> int:
    """Read container metadata before invoking expensive ASR or scene inference."""

    if media_type == "audio/wav":
        try:
            with wave.open(BytesIO(content), "rb") as source:
                duration = source.getnframes() / source.getframerate()
        except (EOFError, wave.Error, ZeroDivisionError) as error:
            raise IntegrityError("WAV duration could not be decoded", cause=error) from error
        return round(duration * 1000)

    suffix = _media_suffix(media_type)
    path = _write_private_media(content, suffix)
    try:
        try:
            import av  # type: ignore[import-not-found]
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "media duration preflight requires the media extra"
            ) from error
        try:
            with av.open(path) as container:
                if container.duration is None:
                    raise IntegrityError("media container does not declare a duration")
                duration = cast(int, container.duration)
                return round(duration / 1000)
        except (av.error.FFmpegError, OSError) as error:
            raise IntegrityError("media duration could not be decoded", cause=error) from error
    finally:
        os.unlink(path)


class MediaDocumentExtractor(DocumentExtractor):
    """Normalize ASR and scene engines without implying diarization or partial success."""

    _AUDIO_TYPES = frozenset({"audio/wav", "audio/flac", "audio/mpeg", "audio/mp4"})
    _VIDEO_TYPES = frozenset({"video/mp4", "video/webm"})

    def __init__(
        self,
        *,
        transcriber: Transcriber | None,
        scene_detector: SceneDetector | None = None,
        max_duration_ms: int = 30 * 60 * 1000,
        max_segments: int = 200,
        max_scenes: int = 100,
        timeout_seconds: float = 2 * 60 * 60,
        probe: MediaProbe | None = None,
    ) -> None:
        if min(max_duration_ms, max_segments, max_scenes) <= 0 or timeout_seconds <= 0:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("media duration limit must be positive")
        self._transcriber = transcriber
        self._scene_detector = scene_detector
        self._max_duration_ms = max_duration_ms
        self._max_segments = max_segments
        self._max_scenes = max_scenes
        self._timeout_seconds = timeout_seconds
        self._probe = probe or _ContainerMediaProbe()

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if len(request.assets) > request.max_documents:
            raise LimitExceededError("media document count exceeds the requested limit")
        if self._transcriber is None:
            raise MissingDependencyError("media extraction requires a provisioned transcriber")
        documents: list[Document] = []
        for asset, classification in zip(request.assets, request.classifications, strict=True):
            if classification.family is not DocumentFamily.MEDIA:
                raise UnsupportedCapabilityError(
                    f"media extractor cannot process {classification.family.value}",
                    capability=classification.family.value,
                )
            media_type = asset.reference.media_type
            is_video = media_type in self._VIDEO_TYPES
            if not is_video and media_type not in self._AUDIO_TYPES:
                raise UnsupportedCapabilityError(
                    f"unsupported media type: {media_type}", capability="media_type"
                )
            if is_video and self._scene_detector is None:
                raise PartialExtractionError(
                    "video transcript without scene/keyframe processing is incomplete"
                )
            probed_duration_ms = self._probe.duration_ms(asset.content, media_type=media_type)
            if probed_duration_ms > self._max_duration_ms:
                raise LimitExceededError("media duration exceeds configured limit")
            duration_ms, raw_segments = run_with_deadline(
                partial(
                    self._transcriber.transcribe,
                    asset.content,
                    media_type=media_type,
                ),
                self._timeout_seconds,
                "media transcription",
            )
            if duration_ms <= 0 or duration_ms > self._max_duration_ms:
                raise LimitExceededError("media duration must be within the configured limit")
            if abs(duration_ms - probed_duration_ms) > 1000:
                raise IntegrityError("transcriber and container duration disagree")
            transcript_parts = self._transcript_parts(
                asset.reference,
                probed_duration_ms,
                tuple(raw_segments),
                self._transcriber.fingerprint,
            )
            scene_parts: tuple[MediaContent, ...] = ()
            scene_fingerprint: str | None = None
            if is_video:
                if self._scene_detector is None:  # narrowed above; retained for type checkers
                    raise PartialExtractionError("video scene detector is unavailable")
                scene_duration_ms, scenes = run_with_deadline(
                    partial(
                        self._scene_detector.detect,
                        asset.content,
                        media_type=media_type,
                    ),
                    self._timeout_seconds,
                    "media scene detection",
                )
                if abs(scene_duration_ms - probed_duration_ms) > 1000:
                    raise IntegrityError("scene and container duration disagree")
                scene_parts = self._scene_parts(asset.reference, probed_duration_ms, tuple(scenes))
                scene_fingerprint = str(self._scene_detector.fingerprint)
            if not transcript_parts and not scene_parts:
                raise IntegrityError("media engines returned no transcript or visual evidence")
            source_id = SourceId.from_locator(
                "media_asset", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    transcript_parts + scene_parts,
                    {
                        "transcriber_fingerprint": str(self._transcriber.fingerprint),
                        "scene_detector_fingerprint": scene_fingerprint,
                        "duration_ms": probed_duration_ms,
                        "speaker_identity": "unknown",
                    },
                )
            )
        return tuple(documents)

    def _transcript_parts(
        self,
        asset: AssetRef,
        duration_ms: int,
        segments: tuple[RawTranscriptSegment, ...],
        transcriber_fingerprint: ComponentFingerprint,
    ) -> tuple[MediaContent, ...]:
        if len(segments) > self._max_segments:
            raise LimitExceededError("transcript segment count exceeds configured limit")
        parts: list[MediaContent] = []
        previous_end = 0
        for index, segment in enumerate(segments):
            values = (segment.start_seconds, segment.end_seconds)
            if not all(math.isfinite(value) for value in values):
                raise IntegrityError("transcript interval must be finite")
            start_ms = round(segment.start_seconds * 1000)
            end_ms = round(segment.end_seconds * 1000)
            if start_ms < previous_end:
                raise IntegrityError("transcript intervals must be monotonic and non-overlapping")
            if start_ms < 0 or end_ms > duration_ms:
                raise IntegrityError("transcript intervals must be bounded by media duration")
            if not segment.text.strip():
                raise IntegrityError("transcript segment must not be blank")
            locator = TimeSpanLocator(start_ms, end_ms)
            notices = [
                ExtractionNotice(
                    "speaker_unknown", "Speaker diarization and identity are not available."
                ),
                ExtractionNotice(
                    "asr_signal_unavailable",
                    "No calibrated transcript correctness confidence was provided.",
                ),
            ]
            if segment.language is None:
                notices.append(ExtractionNotice("language_unknown", "ASR language is unknown."))
            parts.append(
                MediaContent(
                    f"transcript-{index}",
                    segment.text,
                    ExtractionProvenance(
                        asset,
                        locator,
                        transcriber_fingerprint,
                        None,
                        tuple(notices),
                    ),
                )
            )
            previous_end = end_ms
        return tuple(parts)

    def _scene_parts(
        self, asset: AssetRef, duration_ms: int, scenes: tuple[RawScene, ...]
    ) -> tuple[MediaContent, ...]:
        if self._scene_detector is None:
            raise IntegrityError("invalid scene normalization state")
        if duration_ms <= 0:
            raise IntegrityError("video duration must be positive")
        if len(scenes) > self._max_scenes:
            raise LimitExceededError("scene count exceeds configured limit")
        parts: list[MediaContent] = []
        previous_end = 0
        for index, scene in enumerate(scenes):
            if scene.start_ms < previous_end or scene.end_ms > duration_ms:
                raise IntegrityError("scene intervals must be monotonic and bounded by duration")
            if not scene.start_ms <= scene.keyframe_ms < scene.end_ms:
                raise IntegrityError("keyframe must fall inside its scene interval")
            scene_id = f"scene-{index}"
            parts.append(
                MediaContent(
                    scene_id,
                    "[visual scene]",
                    ExtractionProvenance(
                        asset,
                        TimeSpanLocator(scene.start_ms, scene.end_ms),
                        self._scene_detector.fingerprint,
                        None,
                        (ExtractionNotice("scene_derived", "Scene boundary is detector-derived."),),
                    ),
                )
            )
            parts.append(
                MediaContent(
                    f"keyframe-{index}",
                    "[representative keyframe]",
                    ExtractionProvenance(
                        asset,
                        KeyframeLocator(scene.keyframe_ms, scene.frame_number),
                        self._scene_detector.fingerprint,
                        None,
                        (
                            ExtractionNotice(
                                "keyframe_derived", "Keyframe selection is detector-derived."
                            ),
                        ),
                    ),
                    (PartRelation(f"keyframe-{index}", scene_id, RelationKind.KEYFRAME_OF),),
                )
            )
            previous_end = scene.end_ms
        return tuple(parts)
