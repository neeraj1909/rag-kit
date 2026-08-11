from __future__ import annotations

import json
from dataclasses import replace

import pytest

from conftest import ContractCorpus
from ragkit.domain import LimitExceededError, OcrContent, PartRelation, RelationKind, TextContent
from ragkit.ports import ChunkingRequest

pytestmark = pytest.mark.unit


def test_evidence_chunker_preserves_every_family_part_and_locator(
    contract_corpus: ContractCorpus,
) -> None:
    from ragkit.adapters.multimodal import EvidenceChunker

    chunker = EvidenceChunker(max_chars=12)
    chunks = chunker.chunk(ChunkingRequest((contract_corpus.document,), max_chunks=100))

    assert chunks == chunker.chunk(ChunkingRequest((contract_corpus.document,), max_chunks=100))
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert {part_id for chunk in chunks for part_id in chunk.source_part_ids} == {
        part.part_id for part in contract_corpus.document.parts
    }
    parts = {part.part_id: part for part in contract_corpus.document.parts}
    for chunk in chunks:
        part = parts[chunk.source_part_ids[0]]
        assert chunk.provenance[0] == part.provenance
        assert chunk.document_id == contract_corpus.document.document_id
        assert len(chunk.text) <= 12
        assert chunk.metadata["content_family"] == part.family


def test_evidence_chunker_rejects_silent_truncation_and_foreign_parts(
    contract_corpus: ContractCorpus,
) -> None:
    from ragkit.adapters.multimodal import EvidenceChunker

    chunker = EvidenceChunker(max_chars=12)
    with pytest.raises(LimitExceededError, match="chunk limit"):
        chunker.chunk(ChunkingRequest((contract_corpus.document,), max_chunks=1))

    empty = replace(contract_corpus.document, parts=())
    assert chunker.chunk(ChunkingRequest((empty,), max_chunks=1)) == ()


def test_evidence_chunker_projects_source_relationships_for_persistence(
    contract_corpus: ContractCorpus,
) -> None:
    from ragkit.adapters.multimodal import EvidenceChunker

    original = contract_corpus.document.parts[1]
    related = replace(
        original,
        relations=(
            PartRelation(
                original.part_id,
                contract_corpus.document.parts[0].part_id,
                RelationKind.CONTINUES,
            ),
        ),
    )
    document = replace(
        contract_corpus.document,
        parts=(contract_corpus.document.parts[0], related, *contract_corpus.document.parts[2:]),
    )
    chunks = EvidenceChunker().chunk(ChunkingRequest((document,), max_chunks=100))
    chunk = next(item for item in chunks if related.part_id in item.source_part_ids)

    assert json.loads(str(chunk.metadata["source_relations_json"])) == [
        {
            "kind": relation.kind.value,
            "source_part_id": relation.source_part_id,
            "target_part_id": relation.target_part_id,
        }
        for relation in related.relations
    ]


def test_evidence_chunker_indexes_direct_relation_context_with_all_provenance(
    contract_corpus: ContractCorpus,
) -> None:
    from ragkit.adapters.multimodal import EvidenceChunker

    label = contract_corpus.document.parts[0]
    value_id = contract_corpus.document.parts[1].part_id
    value = replace(
        contract_corpus.document.parts[1],
        relations=(PartRelation(value_id, label.part_id, RelationKind.LABELED_BY),),
    )
    assert isinstance(label, TextContent)
    assert isinstance(value, OcrContent)
    document = replace(contract_corpus.document, parts=(label, value))

    chunks = EvidenceChunker(max_chars=1_000).chunk(ChunkingRequest((document,), max_chunks=10))
    value_chunk = next(item for item in chunks if item.source_part_ids[0] == value_id)

    assert value_chunk.text == f"{label.text} {value.text}"
    assert value_chunk.source_part_ids == (value.part_id, label.part_id)
    assert value_chunk.provenance == (value.provenance, label.provenance)
    assert not any(item.source_part_ids == (label.part_id,) for item in chunks)
