# Repository Agent Router

Start with [the agent map](docs/agent-map.md). For a code change, read in this
order: `ARCHITECTURE.md`, the relevant contract in
`src/ragkit/ports/interfaces.py` and value in `src/ragkit/ports/models.py`, its
adapter, its reusable contract tests, then `src/ragkit/infrastructure/config.py`
and `src/ragkit/infrastructure/bootstrap.py`.

## Dependency law

Dependencies point inward: delivery/configuration/composition → adapters →
application → ports → domain. Domain and application never import adapters or
provider SDKs. `bootstrap(profile)` is the composition root; provider selection
does not belong in domain or application code. See `ARCHITECTURE.md`.

## Stable invariants

- Core import requires no optional extra, network, credential, model download,
  database, or GPU.
- IDs and component fingerprints are deterministic; incompatible index
  manifests fail before mutation or search.
- Retrieval relevance is finite and higher-is-better. Preserve native score
  provenance and stable-ID tie-breaking.
- Derived text, tables, images, transcripts, and keyframes retain original
  asset identity and exact span, page/region/cell, timestamp, or keyframe.
- Unsupported capabilities and degraded evidence are explicit. Never silently
  drop a modality, locator, filter, or confidence limitation.
- `memory` is process-local; `sqlite` is persistent. Local model adapters use
  provisioned, revision-pinned files and never download implicitly.
- Hosted credentials are resolved only during composition and are never stored,
  fingerprinted, logged, or included in errors.

## Change and validation contract

Use the task-to-file map and path-specific instructions linked from
`docs/agent-map.md`. Run the authoritative commands in `CONTRIBUTING.md`.
Do not weaken typing, tests, import boundaries, bounds, provenance, score
semantics, or confidence reporting to make a check pass. A change is done only
when its contract and failure behavior are documented, focused tests pass, and
the full applicable validation loop is green.
