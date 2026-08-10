from __future__ import annotations

import math
import os

import pytest

from conftest import ContractCorpus
from ragkit.adapters.cross_encoder_reranker import LocalCrossEncoderReranker
from ragkit.ports import RerankRequest


@pytest.mark.integration
def test_reviewed_cached_cpu_cross_encoder_is_repeatable(
    contract_corpus: ContractCorpus,
) -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip(
            "Set RAGKIT_RUN_MODEL_INTEGRATION=1 after explicitly provisioning the exact "
            "revision documented in docs/recipes/cross-encoder-reranking.md."
        )
    candidates = contract_corpus.scored[:3]
    reranker = LocalCrossEncoderReranker(batch_size=2, max_length=64)
    request = RerankRequest("which evidence mentions an invoice?", candidates, 3)

    first = reranker.rerank(request)
    second = reranker.rerank(request)

    assert first == second
    assert {item.chunk.chunk_id for item in first} == {item.chunk.chunk_id for item in candidates}
    assert all(math.isfinite(item.score.relevance) for item in first)
    assert all(item.score.raw_score == item.score.relevance for item in first)
