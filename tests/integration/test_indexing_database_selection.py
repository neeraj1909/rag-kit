from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ragkit.adapters import DisabledEmbedder
from ragkit.cli.main import main
from ragkit.domain import InvalidDomainValueError, UnsupportedCapabilityError
from ragkit.infrastructure import bootstrap, load_config
from ragkit.ports import (
    IndexingPolicy,
    IndexingStrategy,
    PhysicalIndexStrategy,
    VectorDatabase,
)

pytestmark = pytest.mark.integration


def test_profile_resolves_typed_indexing_and_database_policy() -> None:
    profile = load_config("configs/offline.toml")

    assert profile.indexing_policy.strategy is IndexingStrategy.DENSE
    assert profile.indexing_policy.vector_database is VectorDatabase.MEMORY
    assert profile.indexing_policy.physical_index is PhysicalIndexStrategy.EXACT


def test_profile_rejects_retriever_and_indexing_strategy_drift() -> None:
    profile = load_config("configs/offline.toml")
    invalid = replace(
        profile,
        components=replace(profile.components, vector_store="none"),
        settings=replace(profile.settings, indexing_strategy=IndexingStrategy.SPARSE),
    )

    with pytest.raises(InvalidDomainValueError, match=r"retriever.*indexing"):
        bootstrap(invalid)


def test_unknown_database_selection_fails_without_fallback() -> None:
    profile = load_config("configs/offline.toml")
    invalid = replace(
        profile,
        components=replace(profile.components, vector_store="not-a-database"),
    )

    with pytest.raises(UnsupportedCapabilityError, match="vector database"):
        bootstrap(invalid)


def test_cli_parameters_select_sparse_without_dense_index_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "answer.txt").write_text("The project codename is Indigo Harbor.", encoding="utf-8")

    assert (
        main(
            [
                "index",
                "--config",
                "configs/offline.toml",
                "--source",
                str(source),
                "--indexing-strategy",
                "sparse",
                "--vector-database",
                "none",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["indexing_strategy"] == "sparse"
    assert payload["vector_database"] == "none"
    assert "embed" not in payload["timings_ms"]
    assert "upsert" not in payload["timings_ms"]
    assert "sparse_upsert" in payload["timings_ms"]


def test_cli_auto_keeps_profile_strategy_and_sparse_rejects_explicit_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["index", "--config", "configs/offline.toml", "--indexing-strategy", "auto"]) == 0
    assert json.loads(capsys.readouterr().out)["indexing_strategy"] == "dense"

    assert (
        main(
            [
                "index",
                "--config",
                "configs/offline.toml",
                "--indexing-strategy",
                "sparse",
                "--vector-database",
                "qdrant",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["message"] == "sparse indexing requires --vector-database none"


@pytest.mark.parametrize(
    ("database", "physical"),
    [
        (VectorDatabase.MEMORY, PhysicalIndexStrategy.HNSW),
        (VectorDatabase.SQLITE, PhysicalIndexStrategy.IVF_FLAT),
        (VectorDatabase.PINECONE, PhysicalIndexStrategy.EXACT),
    ],
)
def test_database_rejects_unsupported_physical_index_before_composition(
    database: VectorDatabase, physical: PhysicalIndexStrategy
) -> None:
    profile = load_config("configs/offline.toml")
    invalid = replace(
        profile,
        components=replace(profile.components, vector_store=database.value),
        settings=replace(profile.settings, physical_index_strategy=physical),
    )

    with pytest.raises(InvalidDomainValueError, match="physical index"):
        bootstrap(invalid)


@pytest.mark.parametrize(
    ("path", "database", "physical"),
    (
        ("configs/pgvector.toml", VectorDatabase.PGVECTOR, PhysicalIndexStrategy.EXACT),
        ("configs/qdrant.toml", VectorDatabase.QDRANT, PhysicalIndexStrategy.HNSW),
        ("configs/pinecone.toml", VectorDatabase.PINECONE, PhysicalIndexStrategy.MANAGED),
        ("configs/opensearch.toml", VectorDatabase.OPENSEARCH, PhysicalIndexStrategy.HNSW),
    ),
)
def test_service_profiles_resolve_without_loading_provider_sdks(
    path: str, database: VectorDatabase, physical: PhysicalIndexStrategy
) -> None:
    profile = load_config(path)

    assert profile.indexing_policy == IndexingPolicy(IndexingStrategy.DENSE, database, physical)


def test_sparse_composition_does_not_construct_the_selected_dense_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = load_config("configs/sparse.toml")
    profile = replace(profile, components=replace(profile.components, embedder="torch"))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sparse composition loaded a dense model")

    bootstrap_module = importlib.import_module("ragkit.infrastructure.bootstrap")
    monkeypatch.setattr(bootstrap_module, "TorchTextEmbedder", forbidden)
    runtime = bootstrap(profile)

    assert runtime.embedder_fingerprint == DisabledEmbedder().fingerprint
    assert runtime.embedding_dimension == 1
