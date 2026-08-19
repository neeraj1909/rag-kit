# Qdrant indexing

`QdrantVectorStore` is a family-neutral remote dense-vector adapter. Every
modality stores the full serialized `Chunk`, including its original asset and
locator provenance, beside one named `ragkit_dense` cosine vector.

```python
from ragkit.adapters.qdrant_store import QdrantVectorStore

store = QdrantVectorStore.from_url(
    qdrant_url,
    "assignment-corpus",
    api_key=qdrant_api_key,
    timeout_seconds=30.0,
    max_batch_size=10_000,
    max_top_k=1_000,
)
```

Resolve the URL and API key only at composition. Neither credential is stored,
logged, or fingerprinted. Importing the module does not import `qdrant-client`;
`from_url` raises an actionable `MissingDependencyError` when the optional extra
is absent.

The adapter creates a named cosine collection on the first upsert and writes a
reserved manifest sentinel. Existing collections must contain exactly that
compatible sentinel; a missing or malformed sentinel fails closed. Chunk IDs
are mapped to deterministic UUIDv5 point IDs because Qdrant does not accept the
ragkit stable-ID string as a native point ID. Every read verifies the original
chunk ID, UUID mapping, serialized chunk, and derived stable identity.

Supported filters are the exact Qdrant subset: scalar non-null equality,
homogeneous non-null `IN`, numeric ranges, `AND`, and `OR`. Null/missing,
inequality, negation, string ordering, and field paths that cannot be addressed
without reinterpretation raise `UnsupportedCapabilityError` before provider
work. Native cosine similarity is retained as both raw score and canonical
higher-is-better relevance; returned candidates are re-sorted by score and
stable chunk ID.

Two limitations are architectural, not hidden implementation details:

- Qdrant cannot atomically combine collection creation, sentinel insertion, and
  chunk insertion. A crash can leave a collection without a sentinel; retries
  fail closed and require explicit operator cleanup.
- Qdrant does not document a stable-ID secondary order for equal scores at the
  top-k cutoff. Local sorting stabilizes only the candidates Qdrant returned.

The injected backend unit tests prove translation, validation, identity, and
failure ordering without network access. An SDK integration uses Qdrant's local
in-memory client to exercise real request/response models, named vectors,
manifest retrieval, filtering, and scoring when the optional extra is installed.
That does not prove remote persistence or service operations. The separate
opt-in integration test requires `RAGKIT_RUN_VECTOR_SERVICES=1` and
`RAGKIT_QDRANT_URL`, exercises a real service, reopens the collection, filters,
deletes idempotently, and removes its unique test collection. Routine CI skips
it, so no live-service claim exists unless that evidence is recorded.

## Business use case
Serve multimodal support evidence from Qdrant while retaining timestamps and keyframes.

## Contract
Implements `VectorStore` with a named cosine vector, full chunks, and manifest sentinel.

## Config schema
Select `qdrant`, HNSW, URL, collection, timeout, and optional API-key environment name.

## Optional extra
Install `rag-kit[qdrant]`; core imports remain SDK-free.

## Registry and bootstrap
`bootstrap()` calls `QdrantVectorStore.from_url` only when selected.

## Limits
Batch and `top_k` are locally bounded; service capacity and HNSW tuning are external.

## Determinism
Stable IDs and local ties cover returned candidates, not Qdrant's ANN cutoff.

## Confidence and fallback
Native cosine similarity is retained; no provider or score fallback occurs.

## Failure modes
Missing SDK/service, malformed sentinel, unsupported filters, and corrupt IDs fail explicitly.

## Tests
Injected-backend tests are default; the disposable real-service test is opt-in.
