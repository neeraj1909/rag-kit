# Repository Agent Router

Read `ARCHITECTURE.md`, the relevant port, its default adapter, its contract
tests, and the bootstrap/configuration path in that order.

The Phase 2 offline path is composed in
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
- The offline vector store is process-local. Delivery may rebuild explicitly,
  but must not describe it as persistent storage.

## Validation

Run the commands in `CONTRIBUTING.md`. Do not weaken typing, tests, import
boundaries, confidence reporting, or provenance to make a check pass.
