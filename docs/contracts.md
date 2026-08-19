# Core Port Contracts

The ports are synchronous capability boundaries. Application services compose
them; adapters implement them. Port values are immutable and provider-neutral,
and importing `ragkit.ports` requires no optional SDK, credential, network,
model download, database, event loop, or GPU.

## Shared rules

- Calls return a complete result or raise a typed `ragkit.domain` error. They do
  not return awaitables, start hidden background work, or expose unbounded
  streams.
- Invalid values use `InvalidDomainValueError`; declared bounds use
  `LimitExceededError`; unavailable behavior uses
  `UnsupportedCapabilityError`; index mismatch uses
  `IndexCompatibilityError`; provider failures are translated while retaining
  their original cause.
- Sequence order is preserved unless a contract explicitly ranks results.
  Ranked chunks use descending canonical relevance and ascending stable chunk ID
  for ties. Results are unique and capped by positive `top_k`.
- Relevance is a finite higher-is-better ordering value, not a probability or
  confidence value. Native scores and conversion provenance remain attached.
  Extraction/classification confidence is either absent or finite in `[0, 1]`.
- Every adapter documents determinism, thread safety, external effects, and hard
  limits. A capability gap fails explicitly; it never silently drops a modality,
  filter, provenance locator, or requested control.

## Boundary matrix

| Port | Input → output | Ordering and effects | Required failure behavior |
|---|---|---|---|
| `SourceConnector` | `SourceRequest` → acquired assets | Source order; bounded acquisition I/O | Unsupported scheme, size/count limit, partial read, provider I/O |
| `FamilyClassifier` | acquired assets → classifications | Same length/order; read-only | Invalid bytes, unsupported/unclassifiable family, provider failure |
| `DocumentExtractor` | classified assets → documents | Asset order; model/subprocess effects allowed in adapter | Misalignment, unsupported family, rejected partial extraction |
| `DocumentProjector` | documents → projected documents | Document order; original evidence remains | Unsupported representation, missing locator, part limit |
| `Chunker` | documents → chunks | Document/source order; pure | Invalid provenance, unsupported family, chunk limit |
| `Embedder` | texts/query → embeddings | Batch alignment; model/cache effects adapter-specific | Blank text, dimension mismatch, unsupported mode, provider failure |
| `VectorStore` | upsert/search/delete requests | Writes are explicit; search is ranked | Manifest/embedder/dimension/normalization/filter mismatch before adapter work |
| `Retriever` | query/filter/top-k → scored chunks | Ranked, unique, bounded; read-only | Blank query, invalid/unsupported filter, provider failure |
| `Reranker` | query/candidates/top-k → scored subset | New score stage; never adds candidates | Duplicate/invalid candidates, unsupported mode, provider failure |
| `PromptBuilder` | query/context/budget → prompt | Context order; pure and deterministic | Invalid budget or unsupported content representation |
| `Generator` | exact prompt/context/controls → result | Citation order retained; model/network effects possible | Unsupported controls, limit/rate/provider failures |
| `Evaluator` | observed cases → report | Case evidence order; pure | Empty/invalid cases or unsupported metrics |
| `Telemetry` | sanitized event → none | Ordered external recording effect | Invalid timings/attributes or unavailable sink; no silent loss |

## Capability and limit policy

Limits are explicit request fields, not provider keyword bags. Adapters may
enforce stricter documented limits but cannot silently truncate or reinterpret
the request. Empty batch behavior is contract-specific: classification,
extraction, projection, chunking, and document embedding may return an empty
tuple/batch; single-query operations reject blank input. Unsupported filters or
modalities raise `UnsupportedCapabilityError` instead of producing a weaker
result.

`ChunkingRequest.policy` is an immutable indexing-time contract. Composition
resolves `auto`, validates family compatibility, and binds the complete policy to
the chunker fingerprint before manifest construction. An explicit request policy
that differs from the bound policy fails before index mutation; no adapter may
silently substitute a strategy. See [Chunking strategies](chunking-strategies.md).

`IndexingRequest.indexing_policy` selects dense, sparse, or hybrid
materialization. Composition resolves it, validates the chosen database and
physical index, and binds the policy fingerprint into `IndexManifest`. A
mismatch fails before acquisition. Provider selection remains behind
`VectorStore`; see [Indexing strategies](indexing-strategies.md).

An upsert batch and query vector carry the embedder fingerprint that produced
them. Their fingerprint, dimension, and normalization must match the index
manifest before the vector-store adapter is called. Rerank requests reject
duplicate chunk IDs, generated prompt citations must be a subset of the supplied
context, and evaluation requires at least one observed case.

## Determinism and concurrency

Pure implementations (`Chunker`, `PromptBuilder`, and `Evaluator`) are expected
to be deterministic and thread-safe. I/O/model adapters declare their guarantees
for a fixed component fingerprint and unchanged external state. V1 callers must
not assume concurrent use is safe unless the concrete adapter says so. Async
delivery can place blocking application calls in workers; it must not change the
port signatures.

## Adapter compliance

Each adapter runs the reusable contract suite in addition to its provider tests.
Compliance includes exact provenance survival, aligned embedding dimensions,
manifest validation before store work, idempotent upsert/delete, canonical score
ordering, bounded outputs, typed capability failure, and absence of provider
objects from public values.

## Extending a boundary

An implementation inherits every guarantee in its port docstring. Its adapter
docstring and recipe must narrow the remaining uncertainty: concrete formats and
limits, external effects, state and thread safety, determinism conditions, the exact
meaning of confidence or scores, whether fallback exists, and typed failure modes.
Configuration is a validated schema in `infrastructure/config.py`; selection is an
explicit factory in `infrastructure/bootstrap.py`. Provider option dictionaries,
runtime dynamic imports from configuration, and undocumented fallbacks are not public
extension mechanisms.

The executable procedure and a contract-tested example live in
[`extension-guide.md`](extension-guide.md). Contract helpers are deliberately shared
between fakes and concrete adapters so a new implementation is judged by observable
behavior rather than inheritance alone.
