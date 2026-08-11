# SM-08 cold-agent drill v1

Date: 2026-08-11
Input boundary: a fresh agent received only `AGENTS.md`, `README.md`, and
`docs/agent-map.md`; it did not scan the repository before answering.
Assessment rule: a navigation answer passes only when its cited repository path
exists and source inspection or a mechanical checker confirms the claimed role.

## Verification history

The first response located all five architectural areas, but repeated four
incorrect README implementation links: `HashingEmbedder`,
`InMemoryVectorStore`, and `NoOpReranker` were attributed to `textual.py`, and
`DeterministicEvaluator` to `observability.py`. No question was awarded a pass
at that point. Source verification found the definitions in `retrieval.py` and
`generation.py`; the README was corrected outside this evidence lane.

After correction, `uv run python scripts/check_readme.py --root . --execute`
completed successfully: two copied commands ran, local links passed, and the
offline answer returned a rank-1 `text_span` citation to `answer.txt`. Its
AST-backed symbol-link gate verifies that every named implementation is defined
in the linked Python file. The focused agent-guidance contract additionally
verifies contract/adapter inheritance, the `bootstrap` function, validation
commands, and import-boundary enforcement. Scores below were assigned only
after those checks were green.

## Scorecard

| ID | Question | Result | Source-grounded answer |
|---|---|---|---|
| Q1 | Where is the contract? | PASS | `src/ragkit/ports/interfaces.py`, `src/ragkit/ports/models.py`, and `docs/contracts.md` |
| Q2 | Where is the adapter? | PASS | Concrete implementations are under `src/ragkit/adapters/`; match subclasses to their port and reusable contract tests. |
| Q3 | Where is composition? | PASS | `src/ragkit/infrastructure/bootstrap.py` owns bootstrap(profile); `src/ragkit/infrastructure/config.py` owns the strict profile schema and loader. |
| Q4 | How do I validate? | PASS | `CONTRIBUTING.md` owns the complete validation loop and distinguishes core checks from the all-extras coverage prerequisite. |
| Q5 | What must not be changed? | PASS | Preserve `ARCHITECTURE.md` invariants; `scripts/check_imports.py` and `tests/contract/test_import_boundaries.py` mechanically protect inward dependencies. |

Final score: 5/5. No substantive contradiction was found among the three input
documents. Their command lists differ in granularity but consistently defer to
`CONTRIBUTING.md` as the authoritative complete gate.
