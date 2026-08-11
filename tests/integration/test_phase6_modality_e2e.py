from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ragkit.application import AnsweringRequest, IndexingRequest
from ragkit.domain import KeyframeLocator, TimeSpanLocator
from ragkit.infrastructure import bootstrap, load_config

pytestmark = [pytest.mark.integration, pytest.mark.modality_integration]


def test_reviewed_video_profile_indexes_and_cites_exact_media_evidence() -> None:
    if os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("explicit reviewed model provisioning is required")

    profile = load_config("configs/media-video.toml")
    runtime = bootstrap(profile)
    source = str(Path(profile.source).resolve())
    manifest = runtime.manifest_for(source)

    indexed = runtime.pipeline.index(
        IndexingRequest(
            source,
            manifest,
            max_assets=profile.limits.max_assets,
            max_bytes_per_asset=profile.limits.max_bytes_per_asset,
            max_documents=profile.limits.max_documents,
            max_parts_per_document=profile.limits.max_parts_per_document,
            max_chunks=profile.limits.max_chunks,
        )
    )
    answered = runtime.pipeline.ask(
        AnsweringRequest(
            "Which training step requires closing the isolation valve?",
            manifest,
            retrieval_top_k=profile.limits.top_k,
            rerank_top_k=profile.limits.top_k,
            max_context_chars=profile.limits.max_context_chars,
            max_output_tokens=profile.limits.max_output_tokens,
        )
    )

    assert indexed.asset_count == indexed.document_count == 1
    assert indexed.chunk_count == 5
    assert answered.generation is not None
    assert "close the isolation valve" in answered.generation.answer.casefold()
    assert len(answered.citations) == 1
    assert any(
        isinstance(provenance.locator, TimeSpanLocator)
        for provenance in answered.citations[0].provenance
    )

    scene_answer = runtime.pipeline.ask(
        AnsweringRequest(
            "Where can a reviewer find a representative frame for a scene?",
            manifest,
            retrieval_top_k=3,
            rerank_top_k=3,
            max_context_chars=profile.limits.max_context_chars,
            max_output_tokens=profile.limits.max_output_tokens,
        )
    )
    assert scene_answer.generation is not None
    assert len(scene_answer.citations) == 1
    scene_citation = scene_answer.citations[0]
    assert scene_citation.rank == 1
    assert [type(item.locator) for item in scene_citation.provenance] == [
        KeyframeLocator,
        TimeSpanLocator,
    ]

    cited_keyframe = next(
        candidate.chunk
        for candidate in scene_answer.context
        if candidate.chunk.chunk_id == scene_citation.chunk_id
    )
    cited_source_parts = set(cited_keyframe.source_part_ids)
    assert any(
        candidate.rank > scene_citation.rank
        and cited_source_parts.isdisjoint(candidate.chunk.source_part_ids)
        for candidate in scene_answer.context
    )

    related_keyframes = [
        candidate.chunk
        for candidate in scene_answer.context
        if "source_relations_json" in candidate.chunk.metadata
    ]
    assert len(related_keyframes) == 2
    for chunk in related_keyframes:
        relation_value = chunk.metadata["source_relations_json"]
        assert isinstance(relation_value, str)
        relation = json.loads(relation_value)
        assert relation == [
            {
                "kind": "keyframe_of",
                "source_part_id": chunk.source_part_ids[0],
                "target_part_id": chunk.source_part_ids[1],
            }
        ]
        assert isinstance(chunk.provenance[0].locator, KeyframeLocator)
        assert isinstance(chunk.provenance[1].locator, TimeSpanLocator)
        assert chunk.provenance[0].asset == chunk.provenance[1].asset
