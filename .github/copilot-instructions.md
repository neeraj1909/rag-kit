# ragkit repository guidance

Read `AGENTS.md`, then `docs/agent-map.md`; both route to authoritative source
and deeper documentation. Follow the matching `.github/instructions/` file for
the path being edited.

Keep dependencies inward and make provider choices only in
`src/ragkit/infrastructure/bootstrap.py`. Preserve deterministic identity,
manifest preflight, higher-is-better score semantics, raw-score history, exact
source provenance, explicit capability failures, and dependency-free core
imports. Validate with `CONTRIBUTING.md`; never relax a contract or quality gate
to obtain a passing result.
