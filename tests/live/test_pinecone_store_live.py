"""Explicitly authorized, read-only Pinecone manifest reachability check."""

from __future__ import annotations

import json
import os

import pytest

from ragkit.adapters.pinecone_store import PineconeVectorStore
from ragkit.domain import IndexManifest


@pytest.mark.live
def test_preprovisioned_pinecone_manifest_is_reachable() -> None:
    if os.environ.get("RAGKIT_RUN_LIVE") != "1":
        pytest.skip("set RAGKIT_RUN_LIVE=1 to authorize the Pinecone reachability check")
    required = {
        name: os.environ.get(name)
        for name in (
            "PINECONE_API_KEY",
            "RAGKIT_PINECONE_INDEX_HOST",
            "RAGKIT_PINECONE_NAMESPACE",
            "RAGKIT_PINECONE_MANIFEST_JSON",
        )
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        pytest.skip(f"pre-provisioned Pinecone smoke requires: {', '.join(missing)}")
    decoded = json.loads(required["RAGKIT_PINECONE_MANIFEST_JSON"] or "")
    assert isinstance(decoded, dict)
    manifest = IndexManifest.from_dict(decoded)
    store = PineconeVectorStore(
        index_host=required["RAGKIT_PINECONE_INDEX_HOST"] or "",
        namespace=required["RAGKIT_PINECONE_NAMESPACE"] or "",
        api_key=required["PINECONE_API_KEY"] or "",
        timeout_seconds=30.0,
        max_retries=0,
    )

    store.require_compatible(manifest)

    assert "pinecone_cosine" in str(store.fingerprint)
