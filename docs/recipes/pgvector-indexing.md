# pgvector indexing

`PgVectorStore` is the PostgreSQL service adapter for L2-normalized dense
embeddings. It is family-neutral: text, OCR, layout, image-derived, and media
chunks all use the same `Chunk` and exact-provenance contract.

## Provisioning contract

Provision PostgreSQL, the `vector` extension, and these tables outside an index
request. The adapter deliberately performs no implicit extension installation or
schema migration:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE ragkit_manifests (
    collection_name text PRIMARY KEY,
    manifest_json jsonb NOT NULL
);
CREATE TABLE ragkit_entries (
    collection_name text NOT NULL,
    chunk_id text NOT NULL,
    chunk_json jsonb NOT NULL,
    metadata jsonb NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (collection_name, chunk_id),
    FOREIGN KEY (collection_name)
        REFERENCES ragkit_manifests(collection_name) ON DELETE CASCADE
);
```

Composition resolves the secret DSN and passes a connection factory. The DSN is
not retained in the component fingerprint:

```python
from ragkit.adapters.pgvector_store import PgVectorStore

store = PgVectorStore.from_dsn(
    postgres_dsn,
    "assignment-corpus",
    physical_index_policy="exact",
    max_batch_size=10_000,
    max_top_k=1_000,
)
```

The physical policy records whether the deployment pre-provisioned an exact,
HNSW, or IVFFlat path. The adapter always uses canonical cosine-distance SQL:
distance ascending, then stable chunk ID ascending. `exact` is the only policy
with a complete deterministic cutoff claim. PostgreSQL may use approximate
candidates for HNSW or IVFFlat, so those policies do not prove cutoff
completeness.

Because the shared table admits collections with different dimensions, ANN
policies query through a dimensioned cast. Provision a matching partial
expression index for each collection. For a 1,536-dimensional collection:

```sql
CREATE INDEX assignment_corpus_hnsw
ON ragkit_entries
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE collection_name = 'assignment-corpus';

-- Choose this instead when the profile selects ivf_flat.
CREATE INDEX assignment_corpus_ivf_flat
ON ragkit_entries
USING ivfflat ((embedding::vector(1536)) vector_cosine_ops)
WITH (lists = 100)
WHERE collection_name = 'assignment-corpus';
```

The operator owns index naming, IVFFlat list/probe tuning, HNSW build/search
parameters, and verification with `EXPLAIN`. The adapter does not claim that a
profile label proves PostgreSQL used an ANN plan.

Upsert and delete serialize manifest access in a transaction, compare the
immutable manifest before entry mutation, and remain idempotent by
`(collection_name, chunk_id)`. Search preserves native cosine distance as
`raw_score` and exposes `-distance` as higher-is-better relevance. Filters are
parameterized JSONB expressions; unsupported ordered boolean/null comparisons
fail with `UnsupportedCapabilityError` before opening a connection.

Unit tests use an injected DB-API-shaped connection and prove ordering,
preflight, stable identity, decoding, and error behavior without a service.
Those tests do not prove extension provisioning, driver compatibility,
concurrent transactions, TLS/authentication, restart persistence, query plans,
or live ANN behavior. A Docker-backed PostgreSQL/pgvector run is required before
claiming the adapter as proven.

## Business use case
Persist a mixed claims corpus in operator-owned PostgreSQL while retaining exact OCR and table locators.

## Contract
Implements `VectorStore`; manifests precede mutation/search and complete chunks round-trip.

## Config schema
Select `pgvector`, exact/HNSW/IVF-flat, collection, limits, and a DSN environment-variable name.

## Optional extra
Install `rag-kit[pgvector]`; PostgreSQL and its `vector` extension are separately provisioned.

## Registry and bootstrap
`bootstrap()` resolves the DSN only when constructing `PgVectorStore.from_dsn`.

## Limits
Batch and `top_k` bounds are local; ANN tuning and database resources are operator-owned.

## Determinism
Exact search orders by cosine distance then chunk ID; ANN only stabilizes returned candidates.

## Confidence and fallback
Native distance provenance is retained. There is no database or score fallback.

## Failure modes
Missing driver, DSN, schema, extension, or manifest fails explicitly with sanitized errors.

## Tests
Injected DB-API tests cover SQL/invariants; restart and ANN behavior require service evidence.
