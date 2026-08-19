# ADR 0007: Indexing policy and vector-store portability

- Status: Accepted
- Date: 2026-08-19

## Context

Take-home assignments may require lexical, dense, or hybrid retrieval and may
name a local, service, or hosted vector database. Treating each database as a
separate modality implementation would duplicate orchestration and weaken the
existing inward dependency law. Treating HNSW or IVF-flat as logical RAG
strategies would also conflate materialized evidence with provider-specific
candidate selection.

## Decision

Introduce a typed, immutable `IndexingPolicy` with orthogonal logical strategy,
vector database, and physical-index selections. All five document families
support dense, sparse, and hybrid indexing after chunk normalization.

Application orchestration branches only on dense/sparse/hybrid. Provider
selection remains in configuration and `bootstrap()`, behind `VectorStore`.
Dense performs embedding and vector upsert; sparse performs BM25 upsert without
decorative vector work; hybrid preflights both stores before either mutation.

The resolved policy fingerprint is part of `IndexManifest`. Every store exposes
a read-only compatibility preflight. Missing fresh state may be initialized only
where the provider can do so without overwriting existing ownership: Pinecone
therefore requires explicit sentinel provisioning, while OpenSearch can use an
atomic create operation.

## Invariants

- Unknown selectors and incompatible physical indexes fail before data access.
- Provider SDKs, endpoints, and credentials never cross into application/domain.
- Credential values are composition-only and absent from fingerprints/errors.
- Stored chunks retain stable identity and complete provenance on every backend.
- Relevance is finite and higher-is-better with native score provenance.
- Hybrid compatibility failure cannot mutate only one of its two indexes.
- No adapter silently falls back to a different database or client-side filter.

## Consequences

Adding a database requires one real adapter plus the shared contract, explicit
configuration, optional dependency, recipe, and honest service evidence. It does
not require modality wrappers. Indexes created under another policy are
intentionally incompatible. ANN providers can guarantee stable ordering only
over returned candidates unless their service proves exact cutoff membership.

## Alternatives considered

- One adapter per modality/provider pair: rejected as duplication without a new
  boundary behavior.
- A provider name inside `IndexingRequest`: rejected because application would
  become a service locator.
- Automatic fallback to memory: rejected because it conceals deployment errors
  and changes persistence semantics.
- Calling HNSW/IVF logical indexing strategies: rejected because both are
  physical retrieval implementations beneath the vector-store port.

## Validation implications

Policy tests cover every family and logical strategy. Application tests assert
stage call counts and hybrid preflight ordering. Each provider runs injected
failure/codec contracts; local SDK or service tests are separately gated and
claims remain at their demonstrated evidence level.
