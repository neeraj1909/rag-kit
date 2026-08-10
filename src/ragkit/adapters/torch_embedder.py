"""Revision-pinned local PyTorch/Transformers text embeddings."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, cast

from ragkit.domain import (
    ComponentFingerprint,
    Embedding,
    IntegrityError,
    InvalidDomainValueError,
    MissingDependencyError,
    NormalizationMode,
)
from ragkit.ports import Embedder, EmbeddingBatch, EmbeddingRequest


class TorchEmbeddingBackend(Protocol):
    """Small seam around tensor/model APIs for deterministic contract tests."""

    @property
    def dimension(self) -> int: ...

    def eval(self) -> None: ...

    def inference_mode(self) -> AbstractContextManager[None]: ...

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        device: str,
        max_length: int,
        truncation: bool,
        pooling: str,
    ) -> Sequence[Sequence[float]]: ...


DEFAULT_TEXT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class TorchTextEmbedder(Embedder):
    """Batch a local encoder under explicit, fingerprinted inference semantics."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_TEXT_MODEL_ID,
        revision: str = DEFAULT_TEXT_MODEL_REVISION,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        pooling: str = "mean",
        normalize: bool = True,
        backend: TorchEmbeddingBackend | None = None,
    ) -> None:
        if not model_id.strip():
            raise InvalidDomainValueError("model ID must not be blank")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise InvalidDomainValueError("model revision must be an immutable 40-character SHA")
        if device not in {"cpu", "cuda", "mps"}:
            raise InvalidDomainValueError("device must be cpu, cuda, or mps")
        if batch_size <= 0 or max_length <= 0:
            raise InvalidDomainValueError("batch size and maximum length must be positive")
        if pooling not in {"mean", "cls"}:
            raise InvalidDomainValueError("pooling must be mean or cls")
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._pooling = pooling
        self._normalize = normalize
        self._backend = backend
        if backend is not None and backend.dimension <= 0:
            raise InvalidDomainValueError("embedding backend dimension must be positive")

    @property
    def dimension(self) -> int:
        return self._ensure_backend().dimension

    @property
    def normalization(self) -> NormalizationMode:
        return NormalizationMode.L2 if self._normalize else NormalizationMode.NONE

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return ComponentFingerprint.create(
            "embedder",
            "torch_transformers",
            {
                "model_id": self._model_id,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch_size,
                "max_length": self._max_length,
                "truncation": True,
                "pooling": self._pooling,
                "normalization": self.normalization.value,
                "dimension": self.dimension,
            },
        )

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        backend = self._ensure_backend()
        backend.eval()
        embeddings: list[Embedding] = []
        for offset in range(0, len(request.texts), self._batch_size):
            texts = request.texts[offset : offset + self._batch_size]
            with backend.inference_mode():
                rows = backend.encode_batch(
                    texts,
                    device=self._device,
                    max_length=self._max_length,
                    truncation=True,
                    pooling=self._pooling,
                )
            if len(rows) != len(texts):
                raise IntegrityError("embedding backend output count does not match input count")
            embeddings.extend(self._embedding(row) for row in rows)
        return EmbeddingBatch(tuple(embeddings), self.fingerprint)

    def embed_query(self, text: str) -> Embedding:
        if not text.strip():
            raise InvalidDomainValueError("query must not be blank")
        return self.embed_documents(EmbeddingRequest((text,))).embeddings[0]

    def _embedding(self, row: Sequence[float]) -> Embedding:
        if len(row) != self.dimension:
            raise IntegrityError("embedding backend output dimension changed")
        values = tuple(float(value) for value in row)
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or norm == 0:
                raise IntegrityError("cannot L2-normalize a zero or non-finite embedding")
            values = tuple(value / norm for value in values)
        return Embedding(values, self.dimension, self._normalize)

    def _ensure_backend(self) -> TorchEmbeddingBackend:
        if self._backend is None:
            self._backend = _load_local_transformers_backend(
                self._model_id, self._revision, self._device
            )
        return self._backend


def _load_local_transformers_backend(
    model_id: str, revision: str, device: str
) -> TorchEmbeddingBackend:
    """Load only a locally cached model; ordinary embedding never downloads weights."""

    try:
        return _TransformersTorchBackend(model_id, revision, device)
    except ModuleNotFoundError as error:
        raise MissingDependencyError(
            "local Torch embeddings require the 'vision' extra and a reviewed cached model; "
            "loading always uses local_files_only=True"
        ) from error
    except OSError as error:
        raise MissingDependencyError(
            f"model {model_id}@{revision} is not available locally; provision the reviewed "
            "artifact first (local_files_only=True prevents implicit download)"
        ) from error


class _TransformersTorchBackend:
    """Thin optional adapter over the pinned Transformers/PyTorch APIs."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)
        self._dimension = int(self._model.config.hidden_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    def eval(self) -> None:
        self._model.eval()

    def inference_mode(self) -> AbstractContextManager[None]:
        return cast(AbstractContextManager[None], self._torch.inference_mode())

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        device: str,
        max_length: int,
        truncation: bool,
        pooling: str,
    ) -> Sequence[Sequence[float]]:
        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = self._model(**inputs).last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return cast(list[list[float]], pooled.detach().cpu().tolist())
