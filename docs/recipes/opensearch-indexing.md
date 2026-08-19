# OpenSearch indexing

## Business use case

Use an existing search platform for any supported document family when an assignment
needs remote dense retrieval, exact metadata filters, and operationally inspectable
documents. Modality-specific provenance is complete before storage selection.

## Contract

`OpenSearchVectorStore` implements `VectorStore`. A read-only compatibility probe
leaves an absent index absent. First upsert may atomically create the manifest-bound
mapping; every existing index is validated before bulk, search, or delete work.

## Config schema

Select `vector_store = "opensearch"`. Configure an absolute endpoint, lowercase index
name, optional paired credential environment-variable names, CA bundle, positive
timeout, non-negative retries, and bounded batch size. HTTPS certificate verification
is mandatory; plain HTTP is accepted only on loopback.

## Registry and bootstrap

Bootstrap injects the selected OpenSearch adapter only for dense or hybrid indexing.
The fixed mapping uses Lucene HNSW `cosinesimil`, a non-indexed exact chunk object,
and nested typed metadata. Provider selection never enters the domain or application
pipeline.

## Tests

Injected-client tests cover atomic mapping creation, absent and incompatible
preflights, bulk shape, filter placement/rejection, source decoding, stable IDs,
deletion, score provenance, and malformed hits. A separately enabled integration
test creates and removes an owned `ragkit-live-*` index against a real service.

## Optional extra

Install `rag-kit[opensearch]`. The dependency-free core and other store profiles do
not import `opensearchpy`.

## Limits

Inputs and unit vectors are validated before the first bulk request. Batches are
bounded and mutations use `refresh=wait_for`. The adapter fixes Lucene HNSW cosine;
other engines, metrics, authentication plugins, native hybrid queries, migration,
and production scale are not claimed.

## Determinism

Stable `_id` values make replayed indexing idempotent. Results are unique and sorted
by descending provider score then stable ID. HNSW candidate membership is approximate,
so stable ordering applies to returned hits rather than unseen equal-score neighbors.

## Confidence and fallback

`opensearch_lucene_cosinesimil_score` is OpenSearch's transformed native score, not raw
cosine, probability, or confidence. Missing dependencies, credentials, mappings, or
filter support never cause a memory-store or client-side post-filter fallback.

## Failure modes

An existing index with a missing, malformed, or incompatible manifest/mapping is never
retrofitted, recreated, or migrated. A losing create racer validates the winner before
writing. Bulk operations are non-transactional: item failures are reported with a
sanitized count and stable-ID replay is safe. Delete 404 is idempotent; corrupt chunks,
duplicate hits, non-finite scores, timeouts, and provider failures fail explicitly.
