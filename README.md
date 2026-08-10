# rag-kit

`rag-kit` is a modular Python foundation for provenance-complete retrieval over
textual, OCR, layout-aware, vision-language, and time-based documents.

The repository is currently at the Phase 0 foundation boundary. It contains the
package, dependency and quality configuration plus architectural decisions; the
domain contracts and runnable RAG pipeline begin in Phase 1 and later slices.

## Development baseline

- Python: 3.12 locally; supported range is recorded in `pyproject.toml`.
- Package manager and lockfile: `uv` and `uv.lock`.
- Core runtime: dependency-free and importable without modality or provider
  extras.
- Architecture: see `ARCHITECTURE.md` and `docs/decisions/`.
- Modality decisions: see `docs/modality-support.md`.

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m contract --no-cov
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
