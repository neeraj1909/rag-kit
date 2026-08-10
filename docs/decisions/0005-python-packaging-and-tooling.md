# ADR 0005: Python packaging and tooling baseline

- Status: Accepted
- Date: 2026-08-09

## Context

The kit needs a small importable core, reproducible developer environments, and
separately installable modality/provider families. The Python range must work
for the local ML and document ecosystem without requiring the newest
interpreter during take-home assignments.

## Decision

- Use distribution name `rag-kit` and import package `ragkit`.
- Support CPython `>=3.11,<3.13`; use Python 3.12 as the local default and test
  both 3.11 and 3.12 in CI.
- Use PEP 621 metadata in `pyproject.toml`, Hatchling as the build backend, and
  MIT as the repository license.
- Keep `project.dependencies` empty in Phase 0. Publish independently selectable
  extras named `text`, `ocr`, `layout`, `vision`, `media`, `persistent`, and
  `hosted`; keep test/lint/type/build tooling in the local `dev` dependency
  group.
- Use uv 0.9.22 for the initial resolution workflow and commit the universal
  `uv.lock`. The lockfile, not version prose, owns exact Python package
  resolution. Updating it requires the relevant clean-install and contract
  matrix.
- Use Ruff formatting/linting, strict mypy, strict pytest markers, coverage,
  and the repository import-boundary checker as matching local/CI gates.
- Package `py.typed` and build both wheel and sdist. Publication is outside
  Phase 0 and requires separate authorization.

Official uv documentation distinguishes published runtime dependencies,
published extras, and local dependency groups, and describes the lockfile as
the exact reproducible resolution:

- <https://docs.astral.sh/uv/concepts/projects/dependencies/>
- <https://docs.astral.sh/uv/concepts/projects/sync/>
- <https://docs.astral.sh/uv/concepts/projects/layout/>
- <https://docs.astral.sh/uv/guides/integration/github/>

## Invariants

1. `import ragkit` succeeds after a core-only install without heavy extras,
   model downloads, network access, native tools, credentials, or GPU.
2. An optional dependency is imported only inside its outward adapter.
3. Local and CI quality commands remain identical and non-mutating when used as
   evidence.
4. Python 3.11 compatibility constrains syntax and typing even when development
   runs on 3.12.
5. Exact model revisions, native binaries, language data, codecs, and fixture
   rights are deployment inventory, not implied by the Python lockfile.

## Consequences

The core remains fast to install and provider-neutral, while modality profiles
pay only for the dependencies they need. The universal lock is larger because
it resolves every extra and platform marker, including accelerator variants;
ordinary `uv sync --group dev` does not install those extras. Supporting two
Python minors adds CI work but catches compatibility drift before adapters ship.

## Alternatives considered

- **Python 3.13-only:** rejected because it unnecessarily narrows ML/document
  dependency compatibility and assignment environments.
- **Setuptools:** viable, but Hatchling provides the small explicit src-layout
  build surface selected here.
- **Put all adapters in core dependencies:** rejected because it violates the
  lightweight-core invariant and makes offline assignments install unused ML,
  OCR, storage, and media stacks.
- **Pin every direct requirement with `==` in `pyproject.toml`:** rejected;
  compatible published ranges plus the committed lock separate library API
  intent from one exact development resolution.

## Validation implications

- Run core-only sync/import, locked dev sync, both Python minors in CI, and
  clean installs of each extra before release.
- Run `uv lock --check`, Ruff, mypy, compile, import-boundary, unit, contract,
  coverage, and build gates using the commands in `CONTRIBUTING.md`.
- Inspect wheel/sdist contents for `ragkit`, `py.typed`, README, and license.
- An extra/version-range change requires a new lock and the affected adapter
  contract/integration evidence.

## References

- [Architecture overview](../../ARCHITECTURE.md)
- [ADR 0004: modality adapter baselines](0004-modality-adapter-baselines.md)
- [Modality support matrix](../modality-support.md)
