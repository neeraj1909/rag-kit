"""Revision-pinned, local-only CPU cross-encoder reranking."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, cast

from ragkit.domain import (
    ComponentFingerprint,
    IntegrityError,
    InvalidDomainValueError,
    MissingDependencyError,
    RetrievalScore,
    ScoredChunk,
    ScoreKind,
    ScoreProvenance,
)
from ragkit.ports import Reranker, RerankRequest

DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MAX_BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = 512


class CrossEncoderBackend(Protocol):
    """Minimal model seam used by deterministic core tests."""

    def eval(self) -> None: ...

    def inference_mode(self) -> AbstractContextManager[None]: ...

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        device: str,
        max_length: int,
        truncation: bool,
    ) -> Sequence[float]: ...


class LocalCrossEncoderReranker(Reranker):
    """Score existing candidates with a bounded, locally cached cross-encoder."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RERANKER_MODEL_ID,
        revision: str = DEFAULT_RERANKER_MODEL_REVISION,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        max_top_k: int = 100,
        max_candidates: int = 1_000,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        if not model_id.strip():
            raise InvalidDomainValueError("model ID must not be blank")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise InvalidDomainValueError("model revision must be an immutable 40-character SHA")
        if device != "cpu":
            raise InvalidDomainValueError("local cross-encoder reranking requires a CPU device")
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise InvalidDomainValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
        if not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
            raise InvalidDomainValueError(
                f"maximum length must be between 1 and {MAX_SEQUENCE_LENGTH}"
            )
        if max_top_k <= 0:
            raise InvalidDomainValueError("maximum top-k must be positive")
        if max_candidates <= 0:
            raise InvalidDomainValueError("maximum candidate count must be positive")
        if max_top_k > max_candidates:
            raise InvalidDomainValueError("maximum top-k cannot exceed maximum candidate count")
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_top_k = max_top_k
        self._max_candidates = max_candidates
        self._backend = backend

    @property
    def fingerprint(self) -> ComponentFingerprint:
        return ComponentFingerprint.create(
            "reranker",
            "torch_transformers_cross_encoder",
            {
                "model_id": self._model_id,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch_size,
                "max_length": self._max_length,
                "truncation": True,
                "max_top_k": self._max_top_k,
                "max_candidates": self._max_candidates,
                "score_kind": ScoreKind.LOGIT.value,
                "conversion": "identity:v1",
            },
        )

    def rerank(self, request: RerankRequest) -> tuple[ScoredChunk, ...]:
        if request.top_k > self._max_top_k:
            raise InvalidDomainValueError(
                f"rerank top_k exceeds configured maximum {self._max_top_k}"
            )
        if len(request.candidates) > self._max_candidates:
            raise InvalidDomainValueError(
                f"rerank candidate count exceeds configured maximum {self._max_candidates}"
            )
        identifiers = tuple(candidate.chunk.chunk_id for candidate in request.candidates)
        if len(set(identifiers)) != len(identifiers):
            raise InvalidDomainValueError("rerank candidates contain duplicate chunk IDs")
        if not request.candidates:
            return ()

        backend = self._ensure_backend()
        backend.eval()
        raw_scores: list[float] = []
        for offset in range(0, len(request.candidates), self._batch_size):
            batch = request.candidates[offset : offset + self._batch_size]
            pairs = tuple((request.query, candidate.chunk.text) for candidate in batch)
            with backend.inference_mode():
                values = backend.score_pairs(
                    pairs,
                    device=self._device,
                    max_length=self._max_length,
                    truncation=True,
                )
            if len(values) != len(batch):
                raise IntegrityError(
                    "cross-encoder backend output count does not match input count"
                )
            scores = [float(value) for value in values]
            if not all(math.isfinite(value) for value in scores):
                raise IntegrityError("cross-encoder backend scores must be finite")
            raw_scores.extend(scores)

        provenance = ScoreProvenance(
            self.fingerprint,
            "reranking",
            ScoreKind.LOGIT,
            "cross_encoder_logit",
            "identity:v1",
        )
        ranked = sorted(
            zip(request.candidates, raw_scores, strict=True),
            key=lambda item: (-item[1], str(item[0].chunk.chunk_id)),
        )[: request.top_k]
        return tuple(
            ScoredChunk(
                candidate.chunk,
                RetrievalScore.from_raw(raw_score, provenance),
                rank,
                prior_scores=(candidate.score, *candidate.prior_scores),
            )
            for rank, (candidate, raw_score) in enumerate(ranked, start=1)
        )

    def _ensure_backend(self) -> CrossEncoderBackend:
        if self._backend is None:
            self._backend = _load_local_cross_encoder_backend(
                self._model_id, self._revision, self._device
            )
        return self._backend


def _load_local_cross_encoder_backend(
    model_id: str, revision: str, device: str
) -> CrossEncoderBackend:
    """Load an exact cached revision without provider code or network fallback."""

    try:
        return _TransformersCrossEncoderBackend(model_id, revision, device)
    except ModuleNotFoundError as error:
        raise MissingDependencyError(
            "local cross-encoder reranking requires the 'reranking' extra and a reviewed "
            "cached model; loading always uses local_files_only=True"
        ) from error
    except OSError as error:
        raise MissingDependencyError(
            f"model {model_id}@{revision} is not available locally; provision the reviewed "
            "artifact first (local_files_only=True prevents implicit download)"
        ) from error


class _TransformersCrossEncoderBackend:
    """Thin optional boundary around Transformers sequence classification."""

    def __init__(self, model_id: str, revision: str, device: str) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)

    def eval(self) -> None:
        self._model.eval()

    def inference_mode(self) -> AbstractContextManager[None]:
        return cast(AbstractContextManager[None], self._torch.inference_mode())

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        device: str,
        max_length: int,
        truncation: bool,
    ) -> Sequence[float]:
        inputs = self._tokenizer(
            [query for query, _ in pairs],
            [text for _, text in pairs],
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        logits = self._model(**inputs).logits.reshape(-1)
        return cast(list[float], logits.detach().cpu().tolist())
