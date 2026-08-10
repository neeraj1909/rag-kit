# rag-kit

`rag-kit` is a modular Python foundation for provenance-complete retrieval over
textual, OCR, layout-aware, vision-language, and time-based documents.

The repository includes five provenance-preserving document families plus a
dependency-free textual reference pipeline. Phase 3 adds printed OCR,
layout-aware PDF/PPTX/XLSX extraction, local-only vision/ASR/model seams, a
pinned CPU Torch embedder, persistent Chroma, and an optional hosted generator.
The CLI inspects every profile without importing optional SDKs or reading secret
values.

Phase 4 adds manifest-bound BM25, rank-only reciprocal-rank fusion, a pinned
local cross-encoder reranker, versioned five-family evaluation metrics, and an
honest benchmark harness. Retrieval and reranking strategies change through
profile configuration; native scores from different stages are never treated
as calibrated values.

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

## Phase 3 profiles

```bash
uv sync --frozen --extra ocr --extra layout --group dev
uv run ragkit inspect-config --config configs/ocr.toml
uv run ragkit ask --config configs/ocr.toml "What is the claim ID?"
uv run ragkit ask --config configs/layout.toml "What is the standard price?"

# Inspecting model/provider profiles is safe before provisioning.
uv run ragkit inspect-config --config configs/vision.toml
uv run ragkit inspect-config --config configs/media.toml
uv run ragkit inspect-config --config configs/torch-local.toml
uv run ragkit inspect-config --config configs/persistent.toml
uv run ragkit inspect-config --config configs/hosted.toml
```

Model adapters never download weights during construction or inference. Run the
exact `hf download ... --revision <sha>` action reported by `inspect-config`,
review the artifact, and then opt into model integration with
`RAGKIT_RUN_MODEL_INTEGRATION=1`. Hosted calls are opt-in and require the named
credential environment variable. See `docs/production-adapters.md` and the
business recipes under `docs/recipes/`.

## Phase 4 retrieval profiles

```bash
uv run ragkit ask --config configs/sparse.toml "What is the fixture answer?"
uv run ragkit ask --config configs/hybrid.toml "What is the fixture answer?"

# This profile requires the exact cached model revision described by inspection.
uv run ragkit inspect-config --config configs/reranked.toml
uv sync --frozen --extra reranking --group dev
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run ragkit ask --config configs/reranked.toml "What is the fixture answer?"
```

BM25 scores, dense similarities, reciprocal-rank fusion values, and
cross-encoder logits retain separate provenance. See `docs/evaluation.md` for
the fixed-dataset quality report and benchmark commands.

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

| Extra | Capability |
|---|---|
| `text` | Standard-library text, HTML, email, and code-like input |
| `ocr` | Pillow, pytesseract, and pypdfium2; requires Tesseract/tessdata |
| `layout` | pdfplumber, python-pptx, and openpyxl |
| `vision` | Pillow, PyTorch, Transformers, and a revision-pinned SmolVLM profile |
| `media` | faster-whisper plus PySceneDetect/OpenCV |
| `persistent` | Chroma behind the vector-store port |
| `hosted` | OpenAI SDK behind optional provider adapters |
| `reranking` | PyTorch and Transformers for the revision-pinned local cross-encoder |

The lockfile records resolved Python package versions. System binaries, model
weights, codecs, language data, and fixture licenses remain separately
inventoried deployment artifacts.
