# Repository Agent Router

Read `ARCHITECTURE.md`, the relevant port, its default adapter, its contract
tests, and the bootstrap/configuration path in that order.

## Invariants

- Dependencies point inward: delivery/config/bootstrap → adapters →
  application → ports → domain.
- Domain and application code never import provider SDKs or adapters.
- Core import must not require optional extras, network access, model downloads,
  credentials, or a GPU.
- Canonical relevance is higher-is-better; adapters retain raw provider scores.
- Derived content always retains original-asset and exact source-locator
  provenance. Unsupported modality handling is explicit.

## Validation

Run the commands in `CONTRIBUTING.md`. Do not weaken typing, tests, import
boundaries, confidence reporting, or provenance to make a check pass.
