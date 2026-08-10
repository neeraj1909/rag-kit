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

Run the same non-mutating commands used by CI:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run python -m compileall -q src tests
uv run python -c "import ragkit"
uv run python scripts/check_imports.py
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m contract --no-cov
uv run pytest -m "unit or contract" --cov=ragkit --cov-report=term-missing
```

During implementation, apply automatic formatting and safe lint fixes with:

```bash
uv run ruff format .
uv run ruff check --fix .
```

Then rerun the non-mutating checks above. Tests that require network access,
model downloads, provider credentials, or a GPU must never use the `unit` or
`contract` markers.
