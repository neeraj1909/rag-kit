# Coding-agent map

This page answers the first navigation questions for a cold contributor. It is
a router, not a second architecture specification: follow each link to the
authoritative contract, behavior, or command.

## Read order

1. Read [`ARCHITECTURE.md`](../ARCHITECTURE.md) for dependency direction and
   invariants.
2. Read the relevant ABC in
   [`src/ragkit/ports/interfaces.py`](../src/ragkit/ports/interfaces.py) and its
   request/result values in
   [`src/ragkit/ports/models.py`](../src/ragkit/ports/models.py).
3. Read the concrete implementation in
   [`src/ragkit/adapters/`](../src/ragkit/adapters/) and the matching behavior
   guide in [`docs/offline-adapters.md`](offline-adapters.md) or
   [`docs/production-adapters.md`](production-adapters.md).
4. Read its reusable tests under
   [`tests/contract/`](../tests/contract/) and focused unit/integration tests.
5. Read strict profile loading in
   [`src/ragkit/infrastructure/config.py`](../src/ragkit/infrastructure/config.py)
   and wiring in
   [`src/ragkit/infrastructure/bootstrap.py`](../src/ragkit/infrastructure/bootstrap.py).

## Five-question cold-agent drill

### Where is the contract?

Port behavior is authoritative in
[`src/ragkit/ports/interfaces.py`](../src/ragkit/ports/interfaces.py), with
provider-neutral values in
[`src/ragkit/ports/models.py`](../src/ragkit/ports/models.py) and expanded
semantics in [`docs/contracts.md`](contracts.md).

### Where is the adapter?

Concrete implementations are in
[`src/ragkit/adapters/`](../src/ragkit/adapters/). Start from the port name, find
its subclass, then use the adapter guides and matching contract test; do not
infer behavior from a class name alone.

### Where is composition?

[`bootstrap(profile)`](../src/ragkit/infrastructure/bootstrap.py) is the
composition root. Profile types and exact TOML loading live in
[`src/ragkit/infrastructure/config.py`](../src/ragkit/infrastructure/config.py);
the CLI in [`src/ragkit/cli/main.py`](../src/ragkit/cli/main.py) is delivery, not
an alternate composition root.

### How do I validate?

[`CONTRIBUTING.md`](../CONTRIBUTING.md) owns the complete validation loop. The
core checks include:

```bash
uv run ruff format --check .
uv run mypy src tests
uv run python scripts/check_imports.py
timeout 60 uv run pytest -m contract --no-cov
```

Run focused tests while developing, then every applicable command in that file.

### What must not be changed?

Do not weaken the invariants in [`ARCHITECTURE.md`](../ARCHITECTURE.md), the
port semantics in [`docs/contracts.md`](contracts.md), or validation gates in
[`CONTRIBUTING.md`](../CONTRIBUTING.md) merely to make an implementation pass.
In particular, do not introduce outward core imports, hidden downloads or
fallbacks, silent index recreation, score-direction ambiguity, lost provenance,
or fabricated confidence/quality claims. Import direction is mechanically
enforced by [`scripts/check_imports.py`](../scripts/check_imports.py) and its
negative contract cases in
[`tests/contract/test_import_boundaries.py`](../tests/contract/test_import_boundaries.py).

## Task-to-file map

| Task | Contract/source first | Implementation and proof |
|---|---|---|
| Change a domain value, ID, manifest, locator, or score | `ARCHITECTURE.md`; `src/ragkit/domain/` | Domain unit tests; import-boundary check |
| Add or change a capability | `src/ragkit/ports/interfaces.py`; `src/ragkit/ports/models.py`; `docs/contracts.md` | Adapter plus reusable contract tests |
| Add an offline implementation | Relevant port; `docs/offline-adapters.md` | `src/ragkit/adapters/`; unit and contract tests |
| Add an SDK/model/store implementation | Relevant port; `docs/production-adapters.md` | Adapter; optional-extra guard; integration tests |
| Change indexing or answering flow | Ports and domain records | `src/ragkit/application/`; orchestration tests |
| Add a selectable component/profile field | Port and adapter first | `config.py`, then `bootstrap.py`, profile tests, CLI inspection |
| Change CLI behavior | Application use case and profile contract | `src/ragkit/cli/main.py`; e2e tests |
| Change retrieval or evaluation | Score/provenance contracts; `docs/evaluation.md` | Retrieval adapter/evaluation modules; fixed-corpus tests |

## Definition of done

A change is complete when the public contract and failures are explicit; layer
direction, deterministic identity, manifest safety, bounds, score history,
confidence, and exact provenance still hold; every shipped adapter satisfies its
reusable contract; optional capabilities fail actionably without hidden network
work; focused tests and the applicable `CONTRIBUTING.md` loop pass; and docs do
not claim quality that the recorded evidence does not prove.
