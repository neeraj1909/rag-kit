# ADR 0002: Stable identity and index compatibility

- Status: Accepted
- Date: 2026-08-09

## Context

Idempotent indexing, exact citations, restart-safe persistence, and reproducible
evaluation require identities that do not depend on process state or storage row
order. At the same time, changing the chunker, embedding model, normalization,
or schema can make stored vectors semantically incompatible even if a database
accepts their shape.

The planning requirements therefore call for stable source/document/chunk IDs
and an index manifest that rejects incompatible query and upsert operations.
This decision makes those terms precise without choosing concrete Python types
or a provider SDK API; those version-dependent details belong to Phase 1 and
the relevant adapter.

## Decision

### Canonical digest primitive

All stable IDs and fingerprints use SHA-256 over a canonical, typed payload:

1. Include a distinct namespace and scheme version (for example
   `ragkit:chunk:v1`) so values from different domains cannot alias.
2. Encode values as UTF-8 canonical JSON: object keys sorted, no insignificant
   whitespace, arrays kept in meaningful order, explicit JSON types, and no
   floats where an exact integer/string representation is available.
3. Reject non-finite numbers and ambiguous/unserializable values.
4. Render the public identifier as its type prefix plus the full lowercase hex
   digest, for example `chk_v1_<64 hex characters>`.

Text normalization used for identity is deliberately conservative: normalize
line endings to LF and Unicode to NFC, but do not strip, case-fold, collapse
whitespace, or otherwise erase meaningful content unless a versioned component
contract explicitly declares that behavior.

### Identifier composition

- **Source ID** identifies a logical acquisition address. Hash the connector
  namespace plus its canonical source locator. Locator normalization removes
  only equivalences the connector can prove (for example, resolving an input
  filesystem path to its documented canonical URI). It must not casually drop
  URL query/fragment data, case-fold case-sensitive paths, or incorporate
  credentials. The same logical address produces the same source ID; moving it
  produces a new source ID.
- **Document ID** identifies one immutable acquired rendition. Hash the source
  ID, the SHA-256 digest(s) of the ordered original asset bytes, and an explicit
  document-boundary discriminator when one acquisition contains multiple
  documents. Re-extracting identical acquired evidence retains the document ID;
  changed bytes or boundaries produce a new one.
- **Chunk ID** identifies one derived searchable unit. Hash the document ID,
  chunker fingerprint, ordered source part IDs and canonical locators, and the
  conservatively normalized chunk representation. Do not use database row ID,
  random UUID, wall-clock time, or ordinal alone. Identical text at two source
  locations remains distinct; a meaningful boundary/content/configuration
  change produces a new chunk ID.

Algorithm upgrades that change canonicalization use a new scheme version and
therefore new IDs. Existing IDs are immutable. Truncated digests may be shown in
logs/UI, but are never persisted as authoritative identity. A full-digest
collision with unequal canonical payloads is a fatal integrity error, not an
upsert.

### Component and index fingerprints

A component fingerprint uses the same canonical digest primitive over:

- component kind and implementation identity;
- behavior-affecting configuration;
- algorithm/model/tokenizer identity and immutable revision where applicable;
- output schema/semantic version; and
- environment facts only when they can change the promised output semantics.

Secrets, filesystem installation paths, timestamps, and performance-only
settings do not belong in fingerprints. Token-based chunkers include tokenizer
identity/revision. Embedders include document/query projection mode, model
revision, pooling, truncation, normalization, and output dimension.

An `IndexManifest` contains at least:

- manifest schema version;
- corpus fingerprint (a digest over the logical corpus namespace plus its
  versioned source-selection and normalization policy);
- chunker fingerprint;
- embedder fingerprint;
- embedding dimension and normalization mode; and
- persisted chunk/domain schema fingerprint.

The manifest itself has a fingerprint over all of those fields. The corpus
fingerprint describes which logical corpus and ingestion policy an index serves;
it does not normally hash the mutable inventory of document IDs. Consequently,
new renditions selected by the same corpus policy can be upserted under the same
manifest. A reproducible frozen dataset may opt into an inventory digest as a
versioned corpus-policy field. Stores track indexed document/chunk IDs separately
from the compatibility manifest.

### Compatibility behavior

The persisted manifest is immutable for an index generation. Upsert requires an
exact manifest-fingerprint match. Query requires equality of every
query-relevant expected field: manifest/schema version, corpus/index identity,
embedder fingerprint, vector dimension, normalization, and persisted schema.
The application may supply an expected full manifest or a query-compatibility
projection derived from it; the adapter may not infer compatibility from vector
dimension alone.

Missing, unreadable, or mismatched manifests raise a typed compatibility or
integrity error **before** any record mutation or provider search. The error
reports non-secret field-level differences. The operation must not silently
create, clear, migrate, re-embed, or query the index. Creation of a new index
generation and migrations are explicit outer workflows.

## Invariants

1. Identical canonical inputs under the same scheme produce identical IDs and
   fingerprints across processes and supported platforms.
2. Identity-defining changes produce a different identifier or fingerprint.
3. A chunk always resolves to one document and ordered exact provenance; equal
   text at different locations cannot alias.
4. Repeating an upsert of the same chunks under the same manifest is idempotent.
5. No upsert or query reaches provider mutation/search after a compatibility
   check fails.
6. Dimension equality alone never establishes compatibility.
7. Secrets and transient runtime state never enter identifiers, manifests, or
   mismatch diagnostics.

## Consequences

Positive:

- Re-indexing the same evidence is idempotent and citations remain stable.
- Stale or semantically incompatible vectors fail loudly.
- Fingerprints make evaluation and operational diagnostics reproducible.
- The rules work with in-memory and persistent stores alike.

Costs:

- Canonical payload schemas must be maintained and versioned carefully.
- Content changes intentionally create new document/chunk identities and need
  explicit stale-record cleanup at an outer workflow.
- Index creation must persist a sidecar/metadata manifest atomically enough that
  records are never treated as compatible without it.
- Some model/provider revisions are mutable aliases; production profiles must
  resolve immutable revisions when the provider supports them and otherwise
  record the limitation.

## Alternatives considered

- **Random UUIDs:** easy, but prevent repeat-run identity and idempotent upsert.
- **Source URI as document ID:** conflates a logical source with changing
  renditions and can leak credentials.
- **Text-only chunk hashes:** collapse identical passages at different locations
  and sever exact provenance.
- **Dimension-only compatibility:** accepts embeddings with incompatible models,
  normalization, or schema.
- **Automatic re-index on mismatch:** surprising, potentially destructive and
  expensive; it also hides operational mistakes.

## Validation implications

- Golden-vector tests must assert exact IDs/fingerprints for canonical payloads,
  including Unicode, line endings, duplicate text at different locators, and
  equivalent source-selection policies expressed in different discovery order.
- Property tests must prove determinism, sensitivity to every identity-defining
  field, and rejection of non-finite/ambiguous payloads.
- Contract tests must prove repeat-upsert idempotency and preflight rejection of
  missing/malformed/mismatched manifests.
- Persistent adapter tests must reopen an index in a new process/client and
  repeat both compatible and incompatible operations.
- Test spies must prove mismatch paths execute no provider upsert or query.
- Exact SDK storage and atomicity mechanisms are version-dependent adapter
  details deferred until the persistent adapter is pinned and implemented.

## References

- [Architecture overview](../../ARCHITECTURE.md)
- [ADR 0001: Functional core and synchronous ports](0001-functional-core-and-sync-first-ports.md)
- [Cosmic Python: A Brief Interlude: On Coupling and Abstractions][abstractions]

[abstractions]: https://www.cosmicpython.com/book/chapter_03_abstractions.html
