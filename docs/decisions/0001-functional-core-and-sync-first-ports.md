# ADR 0001: Functional core and synchronous ports

- Status: Accepted
- Date: 2026-08-09

## Context

The kit must support five document families and multiple providers without
turning the core pipeline into provider-specific branching. It also needs a
zero-network reference path and reusable behavioral tests. The planning
capsule's target dependency flow is:

```text
delivery/config/bootstrap -> adapters -> application -> ports -> domain
```

[Cosmic Python][cosmic] recommends dependency inversion around repositories,
small abstractions, a service layer for orchestration, and one explicit
bootstrap/composition boundary. That structure fits a RAG pipeline whose I/O
choices vary while provenance and ranking policies must remain stable.

The primary v1 workflows are batch-like indexing and request/response answering.
Making every port async at inception would spread event-loop and cancellation
semantics into interfaces before there is measured concurrent-I/O demand.

## Decision

Adopt a functional core with an imperative shell and enforce inward-only
dependencies:

- `domain` contains immutable value records, invariants, typed errors, and pure
  policies, and imports only the standard library.
- `ports` imports `domain` and defines narrow nominal capability contracts.
- `application` imports `domain` and `ports`, owns use-case sequencing and
  transaction/workflow decisions, and receives port implementations by
  constructor injection.
- `adapters` implement one or more ports and own all external I/O, optional SDK
  imports, native-to-core translation, retries that are safe for that provider,
  and exception translation.
- delivery/configuration/bootstrap may depend on all layers and is the only
  place that selects concrete adapters.

V1 ports are synchronous. A port method returns its result or raises a typed
error before returning. It must not return an awaitable, secretly schedule a
background task, require an active event loop, or hide an unbounded stream.
Delivery layers may run synchronous use cases in an appropriate worker. An
async port family requires a later ADR based on an observed need; it must not be
added piecemeal to existing synchronous interfaces.

Ports describe domain-relevant behavior, not an SDK surface. They use explicit
typed request/response objects rather than `Any`, provider dictionaries, or
arbitrary `**kwargs`. A new port needs either two plausible implementations or
a clear test seam at an external boundary.

## Invariants

1. No `domain`, `ports`, or `application` module imports from `adapters`,
   infrastructure configuration, CLI/API delivery, Chroma, Torch, or another
   provider SDK.
2. `ports` depends only on `domain`; `application` depends only on `ports` and
   `domain` among inner layers.
3. Application services orchestrate adapters. An adapter must not select or
   call a sibling adapter to complete a use case.
4. Core import and pure tests require no network, model download, database,
   GPU, or heavyweight optional extra.
5. Provider capability gaps fail explicitly; they are not silently weakened to
   a lowest-common-denominator behavior.
6. V1 port method signatures and results are synchronous.

## Consequences

Positive:

- Domain and application behavior is fast and deterministic to test with fakes.
- Provider SDK churn is contained in adapters.
- The same use cases serve CLI, API, and background-job delivery.
- Optional modality/provider dependencies do not burden a core install.

Costs:

- Adapters must translate provider types, errors, scores, and capabilities.
- Some provider-specific features require explicit extension contracts rather
  than passthrough keyword arguments.
- Synchronous provider calls need worker isolation in highly concurrent async
  servers and do not provide cancellation semantics across the port boundary.

## Alternatives considered

- **Provider SDK objects throughout the pipeline:** less translation initially,
  but couples core logic and tests to SDK versions and defeats swappability.
- **One generic `run(**kwargs)` component interface:** superficially uniform,
  but discards static guarantees and behavioral meaning.
- **Async-first ports:** useful for proven concurrent remote I/O, but premature
  for the deterministic offline baseline and contagious across all callers and
  fakes.
- **Adapters orchestrate complete workflows:** reduces application code, but
  duplicates sequencing and makes cross-provider composition difficult.

## Validation implications

- Configure a mechanical import-boundary rule and prove it with a temporary
  forbidden-import negative probe.
- Unit-test application sequencing with synchronous fakes, including error and
  empty-result paths.
- Run the same behavioral contract suite against every adapter implementation.
- Test that importing the public core does not import or initialize optional
  SDKs and opens no socket.
- Document blocking behavior and thread-safety for each adapter. Version-specific
  SDK concurrency, cancellation, and retry details remain adapter implementation
  decisions and must be verified against the pinned SDK when implemented.

## References

- [Architecture overview](../../ARCHITECTURE.md)
- [Cosmic Python: Repository Pattern and Dependency Inversion][cosmic]
- [Cosmic Python: A Brief Interlude: On Coupling and Abstractions][abstractions]
- [Cosmic Python: Our First Use Case: Flask API and Service Layer][service]
- [Cosmic Python: Dependency Injection (and Bootstrapping)][di]

[cosmic]: https://www.cosmicpython.com/book/chapter_02_repository.html
[abstractions]: https://www.cosmicpython.com/book/chapter_03_abstractions.html
[service]: https://www.cosmicpython.com/book/chapter_04_service_layer.html
[di]: https://www.cosmicpython.com/book/chapter_13_dependency_injection.html
