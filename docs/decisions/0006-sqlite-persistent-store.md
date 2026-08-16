# ADR 0006: Replace Chroma with SQLite persistence

- Status: Accepted
- Date: 2026-08-15
- Supersedes: [ADR 0003](0003-retrieval-score-and-chroma-baseline.md) only for the persistent-store selection; ADR 0003's canonical score semantics remain accepted.

## Context

The initial persistent adapter used `chromadb`. GitHub's reviewed
[GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c)
affects every available Chroma 1.x release through 1.5.9 and lists no patched
release. The vulnerable unauthenticated server endpoint was not exposed by
rag-kit, but retaining the package left an avoidable critical dependency in the
supported wheel and container.

The toolkit needs local restart-safe persistence, exact manifest rejection,
metadata filters, deterministic cosine ranking, and provenance round trips. It
does not need a separate vector-database server for the bounded assignment
profiles.

## Decision

Use Python's standard-library `sqlite3` module for the supported persistent
`VectorStore` adapter. One database file may contain multiple explicitly named
collections. Each collection stores one immutable canonical manifest and rows
containing the stable chunk ID, complete canonical `Chunk` JSON, and complete
embedding JSON.

Upsert and delete use `BEGIN IMMEDIATE` transactions. Every operation validates
L2/cosine semantics; search and delete require an existing manifest; incompatible
manifests fail before row query or mutation. Search decodes and validates every
stored row, applies the provider-neutral metadata-filter semantics, computes raw
cosine similarity, and sorts by descending relevance then stable chunk ID.

The `persistent` extra remains as a stable install/profile name but has no
third-party dependency. The supported source tree, wheel, lock, and container no
longer contain or import `chromadb`. Existing Chroma directories are not migrated
implicitly; operators create a new `.sqlite3` index explicitly.

## Invariants

1. Domain, ports, and application code remain storage-provider neutral.
2. A missing or incompatible manifest fails before search/delete or row mutation.
3. Upserting identical stable IDs under the same manifest is idempotent and atomic.
4. Stored chunk identity, provenance, metadata, and vector dimensions are
   validated on every read; malformed rows never become silent empty results.
5. Raw and canonical score provenance declares cosine similarity with identity
   conversion; higher remains better.
6. SQLite paths and collection names come only from trusted configuration, never
   from HTTP request fields.

## Consequences

- The critical Chroma dependency advisory is eliminated from supported artifacts.
- Core and persistent installs remain dependency-free and work on Python 3.11/3.12.
- Exact search scans the bounded local collection rather than using an ANN index;
  this is suitable for the existing assignment limits, not a claim of large-scale
  vector-database performance.
- Existing Chroma state needs an explicit future migration tool if preservation is
  required; no silent conversion is attempted.

## Validation implications

- Run the reusable vector-store contract against SQLite.
- Prove idempotent upsert/delete, metadata filters, exact row round-trip,
  incompatible-manifest rejection, malformed-row rejection, and fresh-process
  reopen.
- Build wheel/sdist and verify `chromadb` is absent from metadata and the lock.
- Rebuild the container and repeat cited-answer persistence/restart evidence.
