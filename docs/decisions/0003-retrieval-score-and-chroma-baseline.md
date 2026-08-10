# ADR 0003: Retrieval score and Chroma baseline

- Status: Accepted
- Date: 2026-08-09

## Context

Vector stores and retrievers expose different notions of score: some return a
similarity where larger is better, others a distance where smaller is better,
and rerankers may emit uncalibrated logits. Treating those numbers as
interchangeable makes ordering incorrect and evaluation misleading.

The kit also needs one restart-safe persistent-store implementation for the
first production-oriented vertical slice. The planning capsule recommends
Chroma for low setup friction while requiring storage to remain swappable.

## Decision

### Core retrieval score

Every core retrieval/reranking result contains:

- `relevance`: a finite numeric ordering value whose only universal semantic is
  **higher is better**;
- `raw_score`: the exact finite native numeric score when the provider exposes
  one, otherwise absent;
- score provenance sufficient to diagnose conversion: producing component,
  stage, declared score kind (`similarity`, `distance`, `logit`, or a future
  typed kind), metric, and conversion/version identity; and
- one stable chunk ID plus complete source provenance.

Adapters convert native scores monotonically into higher-is-better relevance.
The default conversion for a higher-is-better similarity/logit is identity; the
default for a lower-is-better distance is negation. A metric-specific bounded
mapping is allowed only when its range/formula is documented, versioned, and
contract-tested. The raw value and conversion metadata are retained either way.
Non-finite raw or canonical values are rejected with a typed adapter/data error.

Relevance is not required to be in `[0, 1]`, is not a probability, and is not
comparable across different retrievers, metrics, models, stages, or conversion
versions without an explicit calibration/fusion policy. Fusion and reranking
must therefore operate under their own declared score policy rather than mixing
raw values accidentally.

Results are unique by chunk ID, limited to positive `top_k`, and ordered by
descending relevance. Equal relevance is resolved by ascending full stable
chunk ID. If a reranker assigns new relevance, it records a new score stage and
preserves prior score provenance for diagnostics.

### Persistent-store baseline

Choose Chroma as the initial persistent vector-store baseline, implemented
behind the provider-neutral vector-store port. Chroma is an optional adapter,
not a core dependency. Domain records, application services, manifests,
filters, and public results must not import or expose Chroma classes.

The adapter must:

- create/open a persistent collection without network for the local baseline;
- persist and validate the index manifest defined by
  [ADR 0002](0002-stable-identity-and-index-compatibility.md) before upsert or
  query;
- map stable chunk IDs and provenance to storage without losing types;
- make repeated same-ID/same-manifest upserts idempotent;
- translate supported provider filters exactly and reject unsupported
  expressions rather than weakening them;
- explicitly declare the configured metric and native score kind, retain raw
  scores, and convert them to canonical relevance;
- provide reopen, query, delete, and collection/index-generation behavior
  required by the common port; and
- translate provider failures into typed project errors while preserving the
  original cause.

Chroma's exact package version, client construction, persistence calls,
collection metadata constraints, filter grammar, distance metric configuration,
result shape, and concurrency guarantees are SDK-version-dependent. They are
intentionally deferred until implementation pins a supported version and checks
its official documentation. No remembered SDK behavior may override the core
semantics in this ADR.

## Invariants

1. Larger `relevance` always ranks ahead of smaller `relevance` in the same
   declared scoring stage.
2. Native distance, similarity, and reranker values are never treated as
   interchangeable or silently presented as probabilities.
3. Score conversion is monotonic, explicit, versioned, and covered by tests;
   the exact native score remains available when supplied.
4. Ties resolve by full stable chunk ID, independent of provider return order.
5. Chroma is referenced only by its adapter, configuration/composition, tests,
   and documentation—not by domain, ports, or application code.
6. Manifest mismatch prevents Chroma query/upsert; it never triggers implicit
   collection recreation or re-indexing.
7. Replacing Chroma requires configuration/composition and a conforming adapter,
   not changes to core pipeline logic.

## Consequences

Positive:

- All core consumers have one unambiguous sort direction.
- Raw provider evidence makes conversion defects diagnosable.
- Chroma enables a quick local persistent baseline while contract tests preserve
  a path to another store.
- Provider score and filter quirks stay outside the core.

Costs:

- Canonical relevance alone cannot be compared across heterogeneous retrievers;
  hybrid fusion needs a separate explicit algorithm.
- Adapter authors must understand and test the provider's actual score kind and
  metric for every supported SDK version.
- Chroma adds an optional dependency and persistence integration matrix.
- A provider feature outside the common port needs an explicit capability or
  extension rather than leaking native objects.

## Alternatives considered

- **Expose native scores only:** preserves provider fidelity but gives callers no
  reliable ordering contract.
- **Force every score into `[0, 1]`:** suggests false probability/calibration and
  may require undocumented metric assumptions.
- **Use `1 - distance` universally:** not valid for every distance range and can
  mislead consumers; negation safely preserves order without claiming a scale.
- **Make Chroma the domain repository model:** reduces adapter translation but
  couples the entire system to one SDK.
- **Qdrant as the first baseline:** stronger client/server deployment path, but
  more operational setup than required for the initial local assignment path;
  it remains a valid future conforming adapter.
- **In-memory store only:** useful for deterministic tests, but does not prove
  restart-safe persistence.

## Validation implications

- Unit/property tests cover distance and similarity conversion, ordering,
  non-finite rejection, deterministic ties, uniqueness, and `top_k` bounds.
- Contract tests require score provenance and run unchanged against the
  in-memory and Chroma adapters.
- Chroma integration tests pin/report the SDK version and cover create, upsert,
  reopen, query, exact supported filters, explicit unsupported-filter failure,
  delete, and persistence across a fresh client/process.
- Known vectors/queries must prove native score direction and conversion for
  every configured metric; tests must inspect raw and canonical scores.
- A mismatch spy/assertion must prove manifest rejection occurs before provider
  search or mutation.
- Upgrading Chroma requires rerunning this matrix and reviewing official SDK
  documentation for changed score, filter, persistence, and concurrency
  semantics.

## References

- [Architecture overview](../../ARCHITECTURE.md)
- [ADR 0001: Functional core and synchronous ports](0001-functional-core-and-sync-first-ports.md)
- [ADR 0002: Stable identity and index compatibility](0002-stable-identity-and-index-compatibility.md)
- [Cosmic Python: A Brief Interlude: On Coupling and Abstractions][abstractions]

[abstractions]: https://www.cosmicpython.com/book/chapter_03_abstractions.html
