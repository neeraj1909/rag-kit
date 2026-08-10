from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

import pytest

from conftest import ContractCorpus
from ragkit.adapters import cross_encoder_reranker
from ragkit.adapters.cross_encoder_reranker import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_MODEL_REVISION,
    LocalCrossEncoderReranker,
)
from ragkit.domain import (
    IntegrityError,
    InvalidDomainValueError,
    MissingDependencyError,
    ScoreKind,
)
from ragkit.ports import RerankRequest


class FakeCrossEncoderBackend:
    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)
        self.eval_calls = 0
        self.inference_calls = 0
        self.batches: list[tuple[tuple[tuple[str, str], ...], str, int, bool]] = []

    def eval(self) -> None:
        self.eval_calls += 1

    def inference_mode(self) -> AbstractContextManager[None]:
        self.inference_calls += 1
        return nullcontext()

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        device: str,
        max_length: int,
        truncation: bool,
    ) -> Sequence[float]:
        self.batches.append((tuple(pairs), device, max_length, truncation))
        offset = sum(len(batch[0]) for batch in self.batches[:-1])
        return self._scores[offset : offset + len(pairs)]


@pytest.mark.unit
def test_cross_encoder_reranks_with_stable_ties_and_preserves_candidates(
    contract_corpus: ContractCorpus,
) -> None:
    candidates = tuple(reversed(contract_corpus.scored[:5]))
    backend = FakeCrossEncoderBackend([0.5] * len(candidates))
    reranker = LocalCrossEncoderReranker(
        backend=backend,
        batch_size=2,
        max_length=64,
        max_top_k=5,
        max_candidates=5,
    )

    result = reranker.rerank(RerankRequest("find evidence", candidates, 5))

    assert [str(item.chunk.chunk_id) for item in result] == sorted(
        str(item.chunk.chunk_id) for item in candidates
    )
    original_by_id = {item.chunk.chunk_id: item for item in candidates}
    assert all(item.chunk is original_by_id[item.chunk.chunk_id].chunk for item in result)
    assert all(
        item.prior_scores
        == (
            original_by_id[item.chunk.chunk_id].score,
            *original_by_id[item.chunk.chunk_id].prior_scores,
        )
        for item in result
    )
    assert [item.rank for item in result] == [1, 2, 3, 4, 5]
    assert all(item.score.raw_score == 0.5 for item in result)
    assert all(item.score.relevance == 0.5 for item in result)
    assert all(item.score.provenance.kind is ScoreKind.LOGIT for item in result)
    assert all(item.score.provenance.stage == "reranking" for item in result)
    assert backend.eval_calls == 1
    assert backend.inference_calls == 3
    assert [len(batch[0]) for batch in backend.batches] == [2, 2, 1]
    assert all(batch[1:] == ("cpu", 64, True) for batch in backend.batches)


@pytest.mark.unit
def test_cross_encoder_bounds_top_k_and_candidate_work(
    contract_corpus: ContractCorpus,
) -> None:
    candidates = contract_corpus.scored[:3]
    reranker = LocalCrossEncoderReranker(
        backend=FakeCrossEncoderBackend([0.1, 0.2, 0.3]),
        max_top_k=2,
        max_candidates=2,
    )

    with pytest.raises(InvalidDomainValueError, match="top_k exceeds"):
        reranker.rerank(RerankRequest("query", candidates[:2], 3))
    with pytest.raises(InvalidDomainValueError, match="candidate count exceeds"):
        reranker.rerank(RerankRequest("query", candidates, 2))


@pytest.mark.unit
def test_cross_encoder_defensively_rejects_duplicate_candidates(
    contract_corpus: ContractCorpus,
) -> None:
    request = RerankRequest("query", contract_corpus.scored[:1], 1)
    object.__setattr__(request, "candidates", (contract_corpus.scored[0],) * 2)

    with pytest.raises(InvalidDomainValueError, match="duplicate chunk IDs"):
        LocalCrossEncoderReranker(backend=FakeCrossEncoderBackend([0.1, 0.2])).rerank(request)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revision": "main"}, "immutable 40-character SHA"),
        ({"device": "cuda"}, "CPU"),
        ({"batch_size": 0}, "batch size"),
        ({"batch_size": 129}, "batch size"),
        ({"max_length": 0}, "maximum length"),
        ({"max_length": 513}, "maximum length"),
        ({"max_top_k": 0}, "top-k"),
        ({"max_candidates": 0}, "candidate"),
        ({"max_top_k": 2, "max_candidates": 1}, "top-k cannot exceed"),
    ],
)
def test_cross_encoder_rejects_unbounded_or_mutable_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidDomainValueError, match=message):
        LocalCrossEncoderReranker(backend=FakeCrossEncoderBackend([]), **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("scores", [[0.1], [math.inf, 0.2], [math.nan, 0.2]])
def test_cross_encoder_rejects_invalid_backend_output(
    contract_corpus: ContractCorpus, scores: list[float]
) -> None:
    candidates = contract_corpus.scored[:2]
    reranker = LocalCrossEncoderReranker(backend=FakeCrossEncoderBackend(scores))

    with pytest.raises(IntegrityError, match="cross-encoder backend"):
        reranker.rerank(RerankRequest("query", candidates, 2))


@pytest.mark.unit
def test_cross_encoder_fingerprint_records_exact_model_and_execution_policy() -> None:
    reranker = LocalCrossEncoderReranker(backend=FakeCrossEncoderBackend([]))

    assert DEFAULT_RERANKER_MODEL_ID == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert DEFAULT_RERANKER_MODEL_REVISION == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert (
        reranker.fingerprint
        != LocalCrossEncoderReranker(backend=FakeCrossEncoderBackend([]), max_length=64).fingerprint
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (ModuleNotFoundError("transformers"), "reranking.*local_files_only=True"),
        (OSError("cache miss"), "not available locally.*local_files_only=True"),
    ],
)
def test_cross_encoder_loader_reports_actionable_offline_failures(
    monkeypatch: pytest.MonkeyPatch,
    contract_corpus: ContractCorpus,
    failure: Exception,
    message: str,
) -> None:
    def fail(*args: object) -> object:
        raise failure

    monkeypatch.setattr(cross_encoder_reranker, "_TransformersCrossEncoderBackend", fail)

    with pytest.raises(MissingDependencyError, match=message):
        LocalCrossEncoderReranker().rerank(RerankRequest("query", contract_corpus.scored[:1], 1))
