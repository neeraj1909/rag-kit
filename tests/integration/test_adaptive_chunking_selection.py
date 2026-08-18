from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ragkit.application import IndexingRequest, IndexingResult
from ragkit.cli.main import main
from ragkit.domain import InvalidDomainValueError
from ragkit.infrastructure import bootstrap, load_config
from ragkit.infrastructure.bootstrap import OfflineRuntime
from ragkit.ports import ChunkingPolicy, ChunkingStrategy, DocumentFamily

pytestmark = pytest.mark.integration


def _index(source: Path, strategy: ChunkingStrategy) -> tuple[OfflineRuntime, IndexingResult]:
    base = load_config("configs/offline.toml")
    profile = replace(
        base,
        source=str(source),
        settings=replace(
            base.settings,
            chunking_strategy=strategy,
            chunk_overlap_chars=4,
            chunk_min_chars=1,
        ),
        limits=replace(base.limits, chunk_chars=24),
    )
    runtime = bootstrap(profile)
    limits = profile.limits
    result = runtime.pipeline.index(
        IndexingRequest(
            source_uri=str(source),
            manifest=runtime.manifest_for(str(source)),
            max_assets=limits.max_assets,
            max_bytes_per_asset=limits.max_bytes_per_asset,
            max_documents=limits.max_documents,
            max_parts_per_document=limits.max_parts_per_document,
            max_chunks=limits.max_chunks,
            chunking_policy=runtime.chunking_policy,
        )
    )
    return runtime, result


def test_selected_strategy_changes_manifest_and_chunk_identity(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "report.txt").write_text(
        "Section one has facts.\n\nSection two has other facts.", encoding="utf-8"
    )

    fixed_runtime, fixed = _index(source, ChunkingStrategy.FIXED)
    paragraph_runtime, paragraph = _index(source, ChunkingStrategy.PARAGRAPH)

    assert fixed_runtime.chunking_policy.strategy is ChunkingStrategy.FIXED
    assert paragraph_runtime.chunking_policy.strategy is ChunkingStrategy.PARAGRAPH
    assert fixed.manifest.chunker_fingerprint != paragraph.manifest.chunker_fingerprint
    assert fixed.indexed_chunk_ids != paragraph.indexed_chunk_ids


def test_auto_is_resolved_before_manifest_creation(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "note.txt").write_text("A paragraph with evidence.", encoding="utf-8")

    runtime, result = _index(source, ChunkingStrategy.AUTO)

    assert runtime.chunking_policy.strategy is ChunkingStrategy.RECURSIVE
    assert result.manifest.chunker_fingerprint == runtime.chunker_fingerprint


def test_request_policy_must_match_the_manifest_bound_chunker(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "note.txt").write_text("One paragraph. Another sentence.", encoding="utf-8")
    runtime, _ = _index(source, ChunkingStrategy.PARAGRAPH)

    with pytest.raises(InvalidDomainValueError, match="policy"):
        runtime.pipeline.index(
            IndexingRequest(
                str(source),
                runtime.manifest_for(str(source)),
                chunking_policy=ChunkingPolicy(
                    strategy=ChunkingStrategy.FIXED,
                    max_chars=24,
                    overlap_chars=4,
                    min_chunk_chars=1,
                ),
            )
        )


def test_unsupported_family_strategy_pair_fails_during_composition() -> None:
    base = load_config("configs/vision.toml")
    invalid = replace(
        base,
        family=DocumentFamily.VISION,
        settings=replace(base.settings, chunking_strategy=ChunkingStrategy.TABLE),
    )

    with pytest.raises(InvalidDomainValueError, match=r"table.*vision"):
        bootstrap(invalid)


def test_cli_strategy_parameter_selects_the_indexing_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "note.txt").write_text("alpha beta gamma delta", encoding="utf-8")

    exit_code = main(
        [
            "index",
            "--config",
            "configs/offline.toml",
            "--source",
            str(source),
            "--chunking-strategy",
            "fixed",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["chunking_strategy"] == "fixed"
