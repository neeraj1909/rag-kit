from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ragkit.domain import ComponentFingerprint, InvalidDomainValueError
from ragkit.ports import (
    DocumentFamily,
    IndexingPolicy,
    IndexingStrategy,
    PhysicalIndexStrategy,
    VectorDatabase,
    derive_indexing_fingerprint,
    is_indexing_strategy_supported,
    resolve_indexing_policy,
    supported_indexing_strategies,
    validate_indexing_strategy,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", tuple(DocumentFamily))
def test_every_document_family_supports_each_logical_indexing_strategy(
    family: DocumentFamily,
) -> None:
    expected = frozenset({IndexingStrategy.DENSE, IndexingStrategy.SPARSE, IndexingStrategy.HYBRID})
    assert supported_indexing_strategies(family) == expected
    for strategy in expected:
        assert is_indexing_strategy_supported(family, strategy)
        validate_indexing_strategy(family, strategy)


def test_indexing_policy_is_strict_immutable_and_fingerprints_all_inputs() -> None:
    policy = IndexingPolicy(
        IndexingStrategy.HYBRID,
        VectorDatabase.QDRANT,
        PhysicalIndexStrategy.HNSW,
    )
    assert policy.fingerprint_inputs() == {
        "schema": "indexing-policy-v1",
        "strategy": "hybrid",
        "vector_database": "qdrant",
        "physical_index": "hnsw",
    }
    assert (
        policy.fingerprint
        == IndexingPolicy(
            IndexingStrategy.HYBRID,
            VectorDatabase.QDRANT,
            PhysicalIndexStrategy.HNSW,
        ).fingerprint
    )
    assert (
        replace(policy, physical_index=PhysicalIndexStrategy.EXACT).fingerprint
        != policy.fingerprint
    )
    with pytest.raises(FrozenInstanceError):
        policy.strategy = IndexingStrategy.DENSE  # type: ignore[misc]
    with pytest.raises(InvalidDomainValueError, match="IndexingStrategy"):
        IndexingPolicy("dense")  # type: ignore[arg-type]
    with pytest.raises(InvalidDomainValueError, match="VectorDatabase"):
        IndexingPolicy(vector_database="memory")  # type: ignore[arg-type]
    with pytest.raises(InvalidDomainValueError, match="PhysicalIndexStrategy"):
        IndexingPolicy(physical_index="exact")  # type: ignore[arg-type]


@pytest.mark.parametrize("family", tuple(DocumentFamily))
def test_auto_resolves_to_dense_memory_exact_for_each_family(family: DocumentFamily) -> None:
    assert resolve_indexing_policy(family, IndexingPolicy()) == IndexingPolicy(
        IndexingStrategy.DENSE,
        VectorDatabase.MEMORY,
        PhysicalIndexStrategy.EXACT,
    )


@pytest.mark.parametrize("family", tuple(DocumentFamily))
def test_sparse_requires_explicitly_absent_vector_database_and_physical_index(
    family: DocumentFamily,
) -> None:
    assert resolve_indexing_policy(
        family,
        IndexingPolicy(
            IndexingStrategy.SPARSE,
            VectorDatabase.NONE,
            PhysicalIndexStrategy.AUTO,
        ),
    ) == IndexingPolicy(
        IndexingStrategy.SPARSE,
        VectorDatabase.NONE,
        PhysicalIndexStrategy.NONE,
    )
    with pytest.raises(InvalidDomainValueError, match="sparse indexing policy"):
        resolve_indexing_policy(
            family,
            IndexingPolicy(
                IndexingStrategy.SPARSE,
                VectorDatabase.OPENSEARCH,
                PhysicalIndexStrategy.HNSW,
            ),
        )


@pytest.mark.parametrize(
    ("database", "expected"),
    [
        (VectorDatabase.MEMORY, PhysicalIndexStrategy.EXACT),
        (VectorDatabase.SQLITE, PhysicalIndexStrategy.EXACT),
        (VectorDatabase.PGVECTOR, PhysicalIndexStrategy.HNSW),
        (VectorDatabase.QDRANT, PhysicalIndexStrategy.HNSW),
        (VectorDatabase.PINECONE, PhysicalIndexStrategy.MANAGED),
        (VectorDatabase.OPENSEARCH, PhysicalIndexStrategy.HNSW),
    ],
)
def test_physical_auto_resolves_to_a_concrete_database_strategy(
    database: VectorDatabase, expected: PhysicalIndexStrategy
) -> None:
    resolved = resolve_indexing_policy(
        DocumentFamily.TEXT,
        IndexingPolicy(IndexingStrategy.DENSE, database, PhysicalIndexStrategy.AUTO),
    )
    assert resolved.physical_index is expected


@pytest.mark.parametrize(
    "policy",
    [
        IndexingPolicy(IndexingStrategy.DENSE, VectorDatabase.NONE, PhysicalIndexStrategy.NONE),
        IndexingPolicy(IndexingStrategy.HYBRID, VectorDatabase.MEMORY, PhysicalIndexStrategy.HNSW),
        IndexingPolicy(
            IndexingStrategy.DENSE, VectorDatabase.PINECONE, PhysicalIndexStrategy.EXACT
        ),
        IndexingPolicy(
            IndexingStrategy.DENSE, VectorDatabase.QDRANT, PhysicalIndexStrategy.IVF_FLAT
        ),
        IndexingPolicy(
            IndexingStrategy.DENSE, VectorDatabase.OPENSEARCH, PhysicalIndexStrategy.EXACT
        ),
    ],
)
def test_incompatible_database_and_physical_strategy_is_rejected(
    policy: IndexingPolicy,
) -> None:
    with pytest.raises(InvalidDomainValueError, match="indexing policy"):
        resolve_indexing_policy(DocumentFamily.TEXT, policy)


def test_public_compatibility_helpers_reject_untyped_values() -> None:
    with pytest.raises(InvalidDomainValueError, match="DocumentFamily"):
        supported_indexing_strategies("text")  # type: ignore[arg-type]
    with pytest.raises(InvalidDomainValueError, match="IndexingStrategy"):
        is_indexing_strategy_supported(DocumentFamily.TEXT, "dense")  # type: ignore[arg-type]


def test_indexing_fingerprint_binds_concrete_store_and_sparse_codec() -> None:
    policy = resolve_indexing_policy(DocumentFamily.TEXT, IndexingPolicy())
    first = ComponentFingerprint.create("vector_store", "first", {"codec": 1})
    second = ComponentFingerprint.create("vector_store", "second", {"codec": 1})

    assert derive_indexing_fingerprint(policy, first, None) != derive_indexing_fingerprint(
        policy, second, None
    )
