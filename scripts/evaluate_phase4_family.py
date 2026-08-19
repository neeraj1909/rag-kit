#!/usr/bin/env python3
"""Execute one Phase 4 family profile in a failure-isolated process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ragkit.application import AnsweringRequest, IndexingRequest
from ragkit.domain import locator_to_dict
from ragkit.infrastructure import bootstrap, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--query", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    profile = load_config(args.profile)
    runtime = bootstrap(profile)
    manifest = runtime.manifest_for(profile.source)
    limits = profile.limits
    indexed = runtime.pipeline.index(
        IndexingRequest(
            profile.source,
            manifest,
            limits.max_assets,
            limits.max_bytes_per_asset,
            limits.max_documents,
            limits.max_parts_per_document,
            limits.max_chunks,
            chunking_policy=runtime.chunking_policy,
            indexing_policy=runtime.indexing_policy,
        )
    )
    answer = runtime.pipeline.ask(
        AnsweringRequest(
            query=args.query,
            expected_manifest=manifest,
            retrieval_top_k=limits.top_k,
            rerank_top_k=limits.top_k,
            max_context_chars=limits.max_context_chars,
            max_output_tokens=limits.max_output_tokens,
        )
    )

    def provenance_rows(values: Any) -> list[dict[str, object]]:
        return [
            {
                "asset_sha256": item.asset.sha256,
                "locator": locator_to_dict(item.locator),
                "confidence": item.confidence,
            }
            for item in values
        ]

    payload = {
        "indexed_evidence": [
            {
                "chunk_id": str(item.chunk_id),
                "provenance": provenance_rows(item.provenance),
            }
            for item in indexed.indexed_evidence
        ],
        "context": [
            {
                "chunk_id": str(item.chunk.chunk_id),
                "provenance": provenance_rows(item.chunk.provenance),
            }
            for item in answer.context
        ],
        "citations": [str(item.chunk_id) for item in answer.citations],
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
