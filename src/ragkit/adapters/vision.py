"""Fail-closed normalization of injected vision-model descriptions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from typing import Protocol, cast

from ragkit.domain import (
    BoxLocator,
    ComponentFingerprint,
    Document,
    DocumentId,
    ExtractionNotice,
    ExtractionProvenance,
    ImageContent,
    IntegrityError,
    LimitExceededError,
    MissingDependencyError,
    SourceId,
    UnsupportedCapabilityError,
)
from ragkit.ports import DocumentExtractor, DocumentFamily, ExtractionRequest

from ._deadline import run_with_deadline


@dataclass(frozen=True, slots=True)
class VisionDescription:
    """One uncalibrated model description linked to a normalized image region."""

    text: str
    region: tuple[float, float, float, float]


class VisionBackend(Protocol):
    """Explicit, provisioned vision capability used by the normalizing adapter."""

    @property
    def fingerprint(self) -> ComponentFingerprint: ...

    def describe(
        self,
        content: bytes,
        *,
        media_type: str,
        prompt: str,
        max_new_tokens: int,
    ) -> Sequence[VisionDescription]: ...


class ImageInspector(Protocol):
    """Cheap image-header validation seam used before model loading."""

    def inspect(self, content: bytes) -> tuple[int, int]: ...


class _PillowImageInspector:
    def inspect(self, content: bytes) -> tuple[int, int]:
        try:
            from PIL import Image  # type: ignore[import-not-found]

            with Image.open(BytesIO(content)) as image:
                dimensions = cast(tuple[int, int], image.size)
                image.verify()
                return dimensions
        except ModuleNotFoundError as error:
            raise MissingDependencyError("vision extraction requires Pillow") from error
        except OSError as error:
            raise IntegrityError("vision asset could not be decoded", cause=error) from error


SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOLVLM_REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"


class LocalSmolVLMBackend:
    """CPU-only SmolVLM backend that reads an already-provisioned immutable revision."""

    def __init__(
        self,
        *,
        model_id: str = SMOLVLM_MODEL_ID,
        revision: str = SMOLVLM_REVISION,
        image_longest_edge: int = 256,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("vision revision must be an immutable 40-character SHA")
        if image_longest_edge <= 0 or image_longest_edge > 2048:
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError("vision inference image edge must be in [1, 2048]")
        self._model_id = model_id
        self._revision = revision
        self._image_longest_edge = image_longest_edge
        self._loaded = False
        self._fingerprint = ComponentFingerprint.create(
            "vision_model",
            "transformers_smolvlm",
            {
                "model_id": model_id,
                "revision": revision,
                "device": "cpu",
                "local_files_only": True,
                "trust_remote_code": False,
                "image_longest_edge": image_longest_edge,
            },
        )

    def _ensure_runtime(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForMultimodalLM,
                AutoProcessor,
            )

            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                revision=self._revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self._model_id,
                revision=self._revision,
                local_files_only=True,
                trust_remote_code=False,
            ).to("cpu")
        except ModuleNotFoundError as error:
            raise MissingDependencyError(
                "local SmolVLM requires the 'vision' extra (Pillow, PyTorch, Transformers)"
            ) from error
        except OSError as error:
            raise MissingDependencyError(
                f"SmolVLM {self._model_id}@{self._revision} is not cached; provision that reviewed "
                "revision explicitly before constructing the backend"
            ) from error
        self._model.eval()
        self._loaded = True

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return self._fingerprint

    def describe(
        self,
        content: bytes,
        *,
        media_type: str,
        prompt: str,
        max_new_tokens: int,
    ) -> tuple[VisionDescription, ...]:
        self._ensure_runtime()
        if media_type not in VisionDocumentExtractor._MEDIA_TYPES:
            raise UnsupportedCapabilityError(
                f"unsupported vision media type: {media_type}", capability="vision_media_type"
            )
        try:
            from PIL import Image

            image = Image.open(BytesIO(content)).convert("RGB")
        except (ModuleNotFoundError, OSError) as error:
            raise IntegrityError("vision asset could not be decoded", cause=error) from error
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        rendered = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=rendered,
            images=[image],
            return_tensors="pt",
            do_image_splitting=False,
            size={"longest_edge": self._image_longest_edge},
        ).to("cpu")
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        prefix_length = inputs["input_ids"].shape[-1]
        description = self._processor.batch_decode(
            generated[:, prefix_length:], skip_special_tokens=True
        )[0].strip()
        if not description:
            raise IntegrityError("SmolVLM generated a blank description")
        return (VisionDescription(description, (0.0, 0.0, 1.0, 1.0)),)


class VisionDocumentExtractor(DocumentExtractor):
    """Create provenance-linked, explicitly untrusted visual descriptions.

    Model provisioning is deliberately outside this adapter. Callers must inject a
    revision-pinned backend; absence never falls back to OCR, filenames, or empty text.
    """

    _MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

    def __init__(
        self,
        backend: VisionBackend | None,
        *,
        prompt: str = "Describe only visible evidence relevant to retrieval.",
        max_new_tokens: int = 256,
        max_pixels: int = 4_194_304,
        max_dimension: int = 2048,
        max_regions: int = 20,
        timeout_seconds: float = 60.0,
        inspector: ImageInspector | None = None,
    ) -> None:
        if (
            not prompt.strip()
            or min(max_new_tokens, max_pixels, max_dimension, max_regions) <= 0
            or timeout_seconds <= 0
        ):
            from ragkit.domain import InvalidDomainValueError

            raise InvalidDomainValueError(
                "vision prompt must be non-blank and token limit positive"
            )
        self._backend = backend
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._max_pixels = max_pixels
        self._max_dimension = max_dimension
        self._max_regions = max_regions
        self._timeout_seconds = timeout_seconds
        self._inspector = inspector or _PillowImageInspector()

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if len(request.assets) > request.max_documents:
            raise LimitExceededError("vision document count exceeds the requested limit")
        if self._backend is None:
            raise MissingDependencyError(
                "vision extraction requires an explicitly provisioned cached revision-pinned "
                "vision backend; no fallback is permitted"
            )
        documents: list[Document] = []
        for asset, classification in zip(request.assets, request.classifications, strict=True):
            if classification.family is not DocumentFamily.VISION:
                raise UnsupportedCapabilityError(
                    f"vision extractor cannot process {classification.family.value}",
                    capability=classification.family.value,
                )
            if asset.reference.media_type not in self._MEDIA_TYPES:
                raise UnsupportedCapabilityError(
                    f"unsupported vision media type: {asset.reference.media_type}",
                    capability="vision_media_type",
                )
            width, height = self._inspector.inspect(asset.content)
            if max(width, height) > self._max_dimension or width * height > self._max_pixels:
                raise LimitExceededError("vision image exceeds configured dimension or pixel limit")
            descriptions = tuple(
                run_with_deadline(
                    partial(
                        self._backend.describe,
                        asset.content,
                        media_type=asset.reference.media_type,
                        prompt=self._prompt,
                        max_new_tokens=self._max_new_tokens,
                    ),
                    self._timeout_seconds,
                    "vision inference",
                )
            )
            if not descriptions:
                raise IntegrityError("vision backend returned no visual descriptions")
            if len(descriptions) > self._max_regions:
                raise LimitExceededError("vision region count exceeds configured limit")
            source_id = SourceId.from_locator(
                "vision_asset", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            parts: list[ImageContent] = []
            for index, description in enumerate(descriptions):
                if not description.text.strip():
                    raise IntegrityError("vision backend returned a blank description")
                locator = BoxLocator(0, *description.region)
                parts.append(
                    ImageContent(
                        f"vision-{index}",
                        description.text,
                        ExtractionProvenance(
                            asset.reference,
                            locator,
                            self._backend.fingerprint,
                            None,
                            (
                                ExtractionNotice(
                                    "model_derived", "Description was generated from image pixels."
                                ),
                                ExtractionNotice(
                                    "confidence_unavailable",
                                    "The vision model exposes no calibrated factual confidence.",
                                ),
                                ExtractionNotice(
                                    "untrusted_description",
                                    "Treat description as a retrieval lead; cite the original "
                                    "region.",
                                ),
                            ),
                        ),
                    )
                )
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    tuple(parts),
                    {
                        "vision_prompt": self._prompt,
                        "vision_max_new_tokens": self._max_new_tokens,
                        "vision_model_fingerprint": str(self._backend.fingerprint),
                    },
                )
            )
        return tuple(documents)
