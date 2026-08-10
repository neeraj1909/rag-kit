# Repository Agent Router

Read `ARCHITECTURE.md`, the relevant port, its default adapter, its contract
tests, and the bootstrap/configuration path in that order.

The Phase 3 profile matrix is composed in
`src/ragkit/infrastructure/bootstrap.py`; configuration lives in
`src/ragkit/infrastructure/config.py`, orchestration in
`src/ragkit/application/`, dependency-free adapters in
`src/ragkit/adapters/`, and delivery in `src/ragkit/cli/main.py`.

## Invariants

- Dependencies point inward: delivery/config/bootstrap → adapters →
  application → ports → domain.
- Domain and application code never import provider SDKs or adapters.
- Core import must not require optional extras, network access, model downloads,
  credentials, or a GPU.
- Canonical relevance is higher-is-better; adapters retain raw provider scores.
- Derived content always retains original-asset and exact source-locator
  provenance. Unsupported modality handling is explicit.
- The `memory` vector store is process-local; the `chroma` selection is the
  explicit persistent option. Never describe one as the other.
- Model adapters are local-files-only. Model provisioning is an explicit,
  revision-pinned operator step; tests must not download weights implicitly.
- Hosted credentials are resolved only at composition and never serialized,
  fingerprinted, logged, or included in exception text.

## Validation

Run the commands in `CONTRIBUTING.md`. Do not weaken typing, tests, import
boundaries, confidence reporting, or provenance to make a check pass.
