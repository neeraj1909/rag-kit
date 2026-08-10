# ragkit Architecture

This document is the stable map of the intended system. The detailed decisions
are recorded in the linked architecture decision records (ADRs); implementation
must preserve their invariants unless a superseding ADR is accepted.

## Design goals

`ragkit` is a modular retrieval-augmented generation kit supporting textual,
OCR, layout-aware, vision-language, and time-based media documents. It preserves
the original evidence behind every searchable projection and allows I/O-heavy
components to be replaced without changing domain or application logic.

The architecture follows the dependency-inversion and service-layer guidance in
[Cosmic Python's repository chapter][cosmic-repository],
[abstractions chapter][cosmic-abstractions], and
[service-layer chapter][cosmic-service]. The project planning capsule's target
state and evidence ledger apply those sources to this codebase. The ADRs below
turn that planning guidance into repository-level decisions.

## Dependency direction

```text
delivery / configuration / bootstrap
                  |
                  v
              adapters
                  |
                  v
             application
                  |
                  v
                ports
                  |
                  v
               domain
```

Dependencies point inward only:

- `domain` uses the Python standard library and contains immutable values,
  invariants, deterministic transformations, and typed errors.
- `ports` may import `domain` and defines narrow synchronous capabilities.
- `application` may import `ports` and `domain`; it orchestrates use cases but
  performs no provider I/O directly.
- `adapters` implement ports and translate file, model, database, and provider
  behavior into core contracts.
- delivery, configuration, and the bootstrap composition root select and wire
  adapters. Provider selection must not appear as conditionals in the domain or
  application pipeline.

This is a functional-core/imperative-shell design. Ranking rules, identity,
manifest comparison, provenance validation, and other deterministic policies
belong in the core. Network, filesystem, model, subprocess, and database work
belongs at adapter boundaries. See
[ADR 0001: Functional core and synchronous ports](docs/decisions/0001-functional-core-and-sync-first-ports.md).

## Runtime flow

```text
Source -> Connector -> Classifier/Extractor -> Document + ContentParts
                                                    |
                                                    v
                                      Chunker / Projector -> Chunks
                                                    |
                                                    v
                                            Embedder -> Store

Query -> Embedder -> Store/Retriever -> Reranker -> PromptBuilder -> Generator
                             |                                      |
                             +------- provenance-bearing results ---+-> Answer
```

Application services own this sequencing and receive ports through constructor
injection. V1 ports are synchronous. Concurrency, background jobs, and async
delivery may wrap the synchronous use cases at the outer edge; async variants
will be introduced only when measured workload needs justify a separate
contract.

## Identity and provenance

Three identifiers have distinct meanings:

- a source ID identifies a normalized logical acquisition address;
- a document ID identifies an immutable acquired rendition of that source;
- a chunk ID identifies a derived searchable unit at exact source locations
  under one chunking algorithm/configuration.

All are deterministic, versioned SHA-256 identifiers over canonical typed data.
They are never generated from database row order or random UUIDs. Exact
canonicalization, composition rules, collision handling, and evolution policy
are fixed by [ADR 0002: Stable identity and index compatibility](docs/decisions/0002-stable-identity-and-index-compatibility.md).

Every derived OCR string, table projection, image description, transcript, or
keyframe must retain the document/asset identity and its span, page/box/cell, or
time locator. A searchable representation is not a replacement for its source
evidence.

## Index compatibility

Each persisted index has an immutable manifest containing the corpus, chunker,
embedder, vector dimension, normalization, and domain-schema fingerprints. A
fingerprint is a versioned SHA-256 digest of canonical component identity and
behavior-affecting configuration; it is not merely a display name.

Upsert and query must compare the expected manifest with the persisted
manifest. Any incompatible field produces a typed compatibility error before
mutation or search. The implementation must never silently recreate, migrate,
or re-index a collection on mismatch. Operational code may offer an explicit
new-index or migration workflow outside that failing operation.

## Retrieval scores

Core results expose a finite `relevance` value with one ordering rule: larger is
better. Results are sorted by descending relevance, then ascending stable chunk
ID for deterministic ties. Adapters retain the native `raw_score` and its
declared kind/metric, and apply an explicit monotonic conversion: similarities
normally keep their direction; distances reverse it (for example `-distance`).
Canonical relevance is an ordering value, not a promise of probability or
cross-retriever calibration.

Chroma is the first persistent vector-store baseline, but it is reachable only
through the vector-store port and adapter. Its exact client calls, supported
filters, persistence API, and returned score shape are version-dependent SDK
details to lock when the adapter is implemented. They do not leak into the core
contract. See [ADR 0003: Retrieval score and persistent-store semantics](docs/decisions/0003-retrieval-score-and-chroma-baseline.md).

## Architectural invariants

1. No inward layer imports an adapter, provider SDK, delivery framework, or
   configuration implementation.
2. Domain policy is deterministic and testable without network, database,
   model download, GPU, or heavyweight optional dependency.
3. Ports are small capability contracts; application services, not adapters,
   orchestrate workflows.
4. Source, document, and chunk IDs are stable for identical canonical inputs
   and change when an identity-defining input changes.
5. Index operations reject incompatible manifests before observable store work.
6. Retrieval output is unique by chunk ID, bounded by `top_k`, provenance
   complete, and ordered by descending canonical relevance with deterministic
   tie-breaking.
7. Native scores remain diagnosable and are never treated as interchangeable
   across metrics or retrieval stages.
8. Persistent-store choice is replaceable through composition; no core type or
   use case names Chroma.

## Validation implications

- An import-boundary check must mechanically enforce the dependency graph and
  include a negative fixture proving a forbidden import is detected.
- Pure unit/property tests must cover canonicalization, stable IDs, fingerprint
  changes, manifest comparison, score ordering, tie-breaking, and invalid
  non-finite scores.
- Reusable port contract tests must run against in-memory fakes and each real
  adapter.
- Persistent-store integration tests must cover create, idempotent upsert,
  process restart/reopen, query, filtering, delete, native-score conversion,
  and manifest mismatch rejection without mutation/search.
- SDK-specific tests must pin and report the tested dependency version. A later
  SDK upgrade that changes filters, persistence, metric behavior, or result
  shapes requires adapter changes and regression evidence, not a weakened core
  contract.

## Decision index

- [ADR 0001: Functional core and synchronous ports](docs/decisions/0001-functional-core-and-sync-first-ports.md)
- [ADR 0002: Stable identity and index compatibility](docs/decisions/0002-stable-identity-and-index-compatibility.md)
- [ADR 0003: Retrieval score and Chroma baseline](docs/decisions/0003-retrieval-score-and-chroma-baseline.md)
- [ADR 0004: Modality adapter baselines](docs/decisions/0004-modality-adapter-baselines.md)
- [ADR 0005: Python packaging and tooling baseline](docs/decisions/0005-python-packaging-and-tooling.md)

[cosmic-repository]: https://www.cosmicpython.com/book/chapter_02_repository.html
[cosmic-abstractions]: https://www.cosmicpython.com/book/chapter_03_abstractions.html
[cosmic-service]: https://www.cosmicpython.com/book/chapter_04_service_layer.html
