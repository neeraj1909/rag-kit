# Contributing

Use the locked `uv` development environment and keep the dependency direction
inward: `domain` uses the standard library, `ports` may also use `domain`, and
`application` may also use `domain` and `ports`. Provider SDKs and adapters stay
outside those layers.

## Set up

```bash
uv sync --frozen --group dev
```

## Authoritative quality checks

Run the core, non-mutating checks used by CI:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run python -m compileall -q src tests
uv run python -c "import ragkit"
uv run python scripts/check_imports.py
uv run python scripts/check_readme.py
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m contract --no-cov
timeout 60 uv run pytest -m integration --no-cov
timeout 60 uv run pytest -m e2e --no-cov
```

The comprehensive 80% coverage gate requires Tesseract 5.x with English
language data in addition to all locked Python extras. Its first run may update
the project environment with those Python extras:

```bash
tesseract --version
tesseract --list-langs | rg '^eng$'
uv run --frozen --all-extras pytest --cov=ragkit --cov-report=term-missing
```

Model downloads and paid provider calls remain disabled.

For the Phase 5 user-facing evidence, execute the copied README quickstart and
validate the assignment templates without writing a destination:

```bash
uv run python scripts/check_readme.py --execute
uv run python scripts/bootstrap_assignment.py --template local-offline \
  --destination /tmp/ragkit-assignment-preview --dry-run
```

During implementation, apply automatic formatting and safe lint fixes with:

```bash
uv run ruff format .
uv run ruff check --fix .
```

Then rerun the non-mutating checks above. Tests that require network access,
model downloads, provider credentials, or a GPU must never use the `unit` or
`contract` markers.
