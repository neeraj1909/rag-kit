"""Index the synthetic corpus and print one cited answer without network access."""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.application import AnsweringRequest, IndexingRequest
from ragkit.domain import locator_to_dict
from ragkit.infrastructure import bootstrap, load_config

ROOT = Path(__file__).parents[1]


def main() -> None:
    profile = load_config(ROOT / "configs" / "offline.toml")
    source = str(ROOT / profile.source)
    runtime = bootstrap(profile)
    manifest = runtime.manifest_for(source)
    limits = profile.limits
    indexed = runtime.pipeline.index(
        IndexingRequest(
            source,
            manifest,
            limits.max_assets,
            limits.max_bytes_per_asset,
            limits.max_documents,
            limits.max_parts_per_document,
            limits.max_chunks,
        )
    )
    result = runtime.pipeline.ask(
        AnsweringRequest(
            "What is the fixture answer?",
            manifest,
            limits.top_k,
            limits.top_k,
            limits.max_context_chars,
            0.0,
            limits.max_output_tokens,
        )
    )
    if result.generation is None:
        raise RuntimeError("the fixture query returned no answer")
    citations = [
        {
            "chunk_id": str(citation.chunk_id),
            "document_id": str(citation.document_id),
            "rank": citation.rank,
            "evidence": [
                {
                    "source_uri": evidence.asset.uri,
                    "locator": locator_to_dict(evidence.locator),
                }
                for evidence in citation.provenance
            ],
        }
        for citation in result.citations
    ]
    print(
        json.dumps(
            {
                "answer": result.generation.answer,
                "citations": citations,
                "chunks": indexed.chunk_count,
                "config_fingerprint": str(profile.fingerprint),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
