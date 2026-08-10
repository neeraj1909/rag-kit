# rag-kit

`rag-kit` is a modular Python foundation for provenance-complete retrieval over
textual, OCR, layout-aware, vision-language, and time-based documents.

The repository now includes the Phase 2 offline reference pipeline: immutable
multimodal contracts plus a deterministic, dependency-free vertical slice for
text, Markdown, HTML, email, and code-like files. The configured CLI can
inspect a profile, build a process-local index, answer with exact citations,
and evaluate a JSONL dataset without network access.

## Offline quickstart

```bash
uv sync --frozen --group dev
uv run ragkit inspect-config --config configs/offline.toml
uv run ragkit index --config configs/offline.toml --source tests/fixtures/corpus
uv run ragkit ask --config configs/offline.toml --source tests/fixtures/corpus \
  "What is the fixture answer?"
uv run ragkit evaluate --config configs/offline.toml \
  --source tests/fixtures/corpus --dataset tests/fixtures/eval.jsonl
```

The reference vector store is deliberately process-local. `ask` and
`evaluate` rebuild the configured source in the same process; the CLI reports
this behavior rather than implying durable persistence.

## Development baseline

- Python: 3.12 locally; supported range is recorded in `pyproject.toml`.
- Package manager and lockfile: `uv` and `uv.lock`.
- Core runtime: dependency-free and importable without modality or provider
  extras.
- Architecture: see `ARCHITECTURE.md` and `docs/decisions/`.
- Domain and port semantics: see `docs/contracts.md`.
- Offline adapter behavior: see `docs/offline-adapters.md`.
- Modality decisions: see `docs/modality-support.md`.

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m contract --no-cov
timeout 60 uv run pytest -m integration --no-cov
timeout 60 uv run pytest -m e2e --no-cov
```

Optional dependency groups are installation boundaries, not implicit
fallbacks. Install only the family required by a profile; configuration must
fail explicitly when a requested capability is unavailable.

| Extra | Phase 0 baseline |
|---|---|
| `text` | Standard-library text, HTML, email, and code-like input |
| `ocr` | Pillow, pytesseract, and pypdfium2; requires Tesseract/tessdata |
| `layout` | pdfplumber, python-pptx, and openpyxl |
| `vision` | Pillow, PyTorch, Transformers, and a revision-pinned SmolVLM profile |
| `media` | faster-whisper plus PySceneDetect/OpenCV |
| `persistent` | Chroma behind the vector-store port |
| `hosted` | OpenAI SDK behind optional provider adapters |

The lockfile records resolved Python package versions. System binaries, model
weights, codecs, language data, and fixture licenses remain separately
inventoried deployment artifacts.
