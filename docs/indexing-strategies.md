# Indexing and vector-database strategies

Indexing has three logical strategies. They apply after every document family has
become provenance-complete chunks, so text, OCR, layout, vision, and media share
the same policy instead of receiving provider-labelled wrappers.

| Strategy | Materialized indexes | Retrieval |
|---|---|---|
| `dense` | embeddings plus the selected `VectorStore` | vector similarity |
| `sparse` | BM25 only; no embedding or vector-store mutation | lexical BM25 |
| `hybrid` | both, after both stores pass read-only manifest preflight | deterministic reciprocal-rank fusion |

`auto` resolves from the profile retriever during composition. A request must
carry the exact resolved policy. Strategy, vector database, and physical index
are fingerprinted into the immutable index manifest, so changing any of them
requires a new compatible index rather than silently reusing old state.

## Vector database matrix

| Selector | Physical index | Evidence level | Operational boundary |
|---|---|---|---|
| `memory` | exact | Proven by shared contracts | process-local |
| `sqlite` | exact | Proven by reopen/integration tests | local persistent file |
| `pgvector` | exact, HNSW, IVF-flat | Adapter/SQL tests; service not proven | operator provisions PostgreSQL, extension, schema, and ANN index |
| `qdrant` | HNSW | Adapter plus local SDK test; remote service opt-in | first upsert creates collection/sentinel; interrupted initialization fails closed |
| `pinecone` | managed | Injected-client tests; live smoke opt-in | operator pre-provisions index and manifest sentinel; eventual consistency |
| `opensearch` | HNSW | Injected-client tests; service opt-in | first upsert may atomically create mapping/manifest |

The database parameter is composition input, not application branching. Provider
credentials are read only by `bootstrap()` and never enter policy fingerprints,
manifests, telemetry, or error text. Unsupported database/physical-index pairs
fail before connector or provider work; there is no fallback to memory.

## Selection

Profiles set `components.retriever`, `components.vector_store`,
`settings.indexing_strategy`, and `settings.physical_index_strategy`. The CLI can
override them before composition:

```bash
uv run ragkit index --config configs/offline.toml \
  --indexing-strategy hybrid --vector-database sqlite
```

HTTP callers may echo `indexing_strategy` and `vector_database`, but only the
already composed profile values are accepted. Requests cannot instantiate a new
provider or escape deployment policy.

## Deliberate limits

- HNSW, IVF-flat, and managed indexes do not promise exact top-k membership.
  Stable-ID ordering applies to candidates the provider returns.
- Native scores retain provider name, kind, metric, and conversion. Scores are
  not calibrated across databases.
- Pinecone provisioning and hosted/service calls are never run by default.
- Mocked clients prove translation and failure ordering, not service durability,
  authentication, concurrency, latency, or production capacity.

Provider-specific setup and evidence are in the four indexing recipes under
`docs/recipes/`.
