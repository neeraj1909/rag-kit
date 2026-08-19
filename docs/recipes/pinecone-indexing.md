# Pinecone indexing

## Business use case

Use a managed vector service for any supported document family when an assignment
needs remote persistence and metadata-filtered dense retrieval without operating a
database. Extraction and chunking remain modality-specific before this boundary.

## Contract

`PineconeVectorStore` implements `VectorStore`. Every operation fetches and validates
a pre-provisioned namespace manifest before data-plane mutation or search. Stable
chunk IDs, complete canonical chunks, provenance, finite native scores, and exact
supported filters cross the boundary without provider types.

## Config schema

Select `vector_store = "pinecone"`. Configure an index host, namespace, API-key
environment-variable name, positive timeout, non-negative retries, bounded batch
size, and metadata byte bound. Composition resolves the credential; values never
enter profiles, fingerprints, logs, or errors.

## Registry and bootstrap

The bootstrap registry constructs one adapter for every family after chunks and
embeddings have been normalized. It targets the data plane by host and does not call
Pinecone control-plane create, configure, or delete operations.

## Tests

Injected-client unit tests prove manifest-first rejection, stable idempotent writes,
typed filter compilation, exact chunk decoding, deletion, score provenance, metadata
bounds, and malformed-response handling without a socket. The live test is a
read-only, double-opt-in manifest reachability check for a pre-provisioned namespace.

## Optional extra

Install `rag-kit[pinecone]`. Core import and the other stores do not import the SDK.

## Limits

The index must already use cosine and the exact embedding dimension. Application
calls never create or overwrite the sentinel. Metadata is capped at 40,000 encoded
bytes and writes are split into bounded batches. Integrated embeddings, sparse
vectors, native hybrid fusion, index lifecycle, and migration are out of scope.

## Determinism

Same-ID writes replace the same record. Returned hits are deduplicated and ordered by
descending native cosine similarity then stable chunk ID. Pinecone is eventually
consistent, so immediate read-after-write visibility and global ANN tie membership
are not promised.

## Confidence and fallback

Native cosine similarity is ranking evidence, not probability or confidence. A
missing SDK, credential, index, sentinel, filter capability, or compatible manifest
has no memory-store fallback.

## Failure modes

Provision a cosine index and insert this reserved record before using the adapter:

```python
from pinecone import Pinecone
from ragkit.adapters.pinecone_store import PineconeVectorStore
from ragkit.domain import canonical_json

metadata = {
    "_rk_kind": "manifest",
    "_rk_schema": "pinecone-vector-store-v1",
    "_rk_manifest": canonical_json(manifest.to_dict()),
}
vector = [1.0] + [0.0] * (manifest.embedding_dimension - 1)
Pinecone(api_key=api_key, timeout=30.0).Index(host=index_host).upsert(
    namespace=namespace,
    vectors=[{"id": PineconeVectorStore.MANIFEST_ID, "values": vector, "metadata": metadata}],
)
```

A missing/malformed/incompatible sentinel, invalid unit vector or stable ID,
oversized metadata, duplicate/malformed hit, timeout, rate limit, or provider failure
fails explicitly. Separate provisioning avoids an incompatible-first-writer race.
