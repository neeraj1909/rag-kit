# rag-kit

## Purpose and non-goals

`rag-kit` is a typed Python foundation for building retrieval-augmented systems
over text, OCR, layout-aware files, images, and time-based media while keeping
every derived result linked to its original evidence. Its dependency-free
reference profile is small enough to run locally and its ports let an assignment
replace one capability without moving provider types into the application core.

This is a pre-alpha toolkit, not a hosted service, an autonomous ingestion
platform, or a promise that every modality has equal retrieval quality. It does
not download model weights implicitly, treat OCR/model output as certain, hide
unsupported input behind a text fallback, or make the process-local `memory`
store durable. The `chroma` profile is the explicit persistent option. Hosted
generation is optional and no paid live-provider result is claimed.

## Offline quickstart

Prerequisites are Python 3.11 or 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). From a clone,
copy these commands exactly:

<!-- readme-quickstart:start -->
```bash
uv sync --frozen --group dev
uv run ragkit ask --config configs/offline.toml --source tests/fixtures/corpus \
  "What is the fixture answer?"
```
<!-- readme-quickstart:end -->

The second command emits one JSON object. Its answer contains `cobalt
observatory`; its first citation points to `answer.txt` with a `text_span`
locator and rank `1`. The command indexes and queries in one process because the
profile uses the in-memory store. It needs no credentials, model weights, GPU,
Docker, or network access at runtime. The initial `uv sync` may access a package
registry when the locked packages are not already cached.

To use your own text corpus, replace `tests/fixtures/corpus` with a file or
directory. The configured byte, document, part, chunk, context, and output
limits still apply.

## Data flow

```text
source -> connect -> classify -> extract/project -> chunk -> embed -> index
                                                               |
query  -> embed/search or BM25 -> optional fusion -> rerank -> context -> generate
                                                               |
                                      answer <- citations <- original locators
```

The application services own the sequence; adapters own filesystem, model,
database, subprocess, and provider I/O. Derived OCR text, cells, image
descriptions, transcript spans, and keyframes remain attached to the original
asset and an exact typed locator.

## Choose a profile

Every profile is strict TOML. `ragkit inspect-config --config <path>` validates
the selection and reports missing extras, binaries, cached models, credentials,
limits, degraded modes, and fingerprints without reading secret values.

| Profile | Choose it for | Requirements and runtime behavior | Current evidence boundary |
|---|---|---|---|
| [`offline`](configs/offline.toml) | Local text/Markdown/HTML/email/code baseline | Core only; hashing vectors, dense retrieval, extractive answer, process-local memory | Offline cited-answer e2e passes |
| [`sparse`](configs/sparse.toml) | Exact-term text retrieval | Core only; manifest-bound BM25 in process-local memory | Fixed text quality thresholds pass |
| [`hybrid`](configs/hybrid.toml) | Dense plus lexical text retrieval | Core only; bounded rank-only reciprocal-rank fusion | Fixed text quality thresholds pass; fusion scores are not probabilities |
| [`reranked`](configs/reranked.toml) | Local cross-encoder reranking | `reranking` extra and exact cached revision; CPU reference path, no implicit download | Pinned offline integration passes |
| [`ocr`](configs/ocr.toml) | Printed scans and bounded scanned PDFs | `ocr` extra plus Tesseract and language data; process-local memory | Extraction/retrieval/box locator pass; citation coverage currently fails |
| [`layout`](configs/layout.toml) | PDF, PPTX, XLSX, and XLSM structure | `layout` extra; parser selected by suffix; process-local memory | Extraction passes; business-query retrieval, cell locator, and citation currently fail |
| [`vision`](configs/vision.toml) | Image-derived descriptions with region evidence | `vision` extra and exact cached SmolVLM revision; CPU/local-files-only | Bounded quality runner is ineligible on its resource policy; no quality pass is claimed |
| [`mixed-image`](configs/mixed-image.toml) | OCR and vision over the same raster page | `ocr` and `vision` extras, Tesseract, cached SmolVLM | Both extractors are required; neither silently substitutes for the other |
| [`media`](configs/media.toml) | Timestamped audio/video evidence | `media` extra and exact cached faster-whisper revision | Audio timestamp case passes; video scene/keyframe quality remains partial |
| [`torch-local`](configs/torch-local.toml) | Pinned local text embeddings | `vision` extra supplies Torch/Transformers; exact cached MiniLM revision | CPU repeatability integration passes |
| [`persistent`](configs/persistent.toml) | A store that survives process exit | `persistent` extra; Chroma collection at the configured path | Reopen/query/delete and incompatible-manifest rejection pass |
| [`hosted`](configs/hosted.toml) | Optional OpenAI generation | `hosted` extra and `OPENAI_API_KEY`; network call is explicit | Mocked secret/error behavior passes; no paid live smoke is claimed |

The family boundary is narrower than the set of file extensions a third-party
library might accept:

| Family | Accepted reference inputs; profile/extra | Searchable evidence and locator | Confidence/fallback policy | Fixture, business path, and proof |
|---|---|---|---|---|
| Text | TXT, Markdown, HTML, email, and code-like UTF text; [`offline`](configs/offline.toml) / core (`text` adds no dependency) | Decoded text with `TextSpanLocator` and structural path | Confidence does not apply; ambiguous encoding and malformed MIME fail or emit a notice rather than guessing | [Corpus](tests/fixtures/corpus), [knowledge-base recipe](docs/recipes/knowledge-base-text.md), `unit`/`contract`/`e2e` |
| OCR | PNG, JPEG, TIFF, BMP, WebP, bounded scanned PDF; [`ocr`](configs/ocr.toml) / `ocr` | OCR words/lines with page plus `BoxLocator` | Raw Tesseract confidence stays visible; handwriting is best-effort, missing engines/languages fail, and there is no native-text fallback | [OCR fixtures](tests/fixtures/ocr), [claims recipe](docs/recipes/claims-ocr.md), `modality_integration` |
| Layout | Machine-generated PDF, PPTX, XLSX/XLSM; [`layout`](configs/layout.toml) / `layout` | Ordered regions, cells, headers, merged-cell/formula relations with page, box, slide/sheet, or `CellLocator` | Ambiguous reading order emits notices; scanned pages and embedded images require explicit OCR/vision routing | [Layout fixtures](tests/fixtures/layout), [financial recipe](docs/recipes/financial-layout.md), `modality_integration` |
| Vision | PNG, JPEG, WebP; [`vision`](configs/vision.toml) / `vision` | Model-derived descriptions retain the original asset and image region | Factual confidence is unavailable; output is untrusted derived evidence and never falls back to OCR or filename text | [Vision fixture](tests/fixtures/vision), [equipment recipe](docs/recipes/equipment-vision.md), opt-in model `integration` |
| Media | WAV, FLAC, MP3, M4A/MP4 audio, short MP4/WebM video; [`media`](configs/media.toml) / `media` | Transcript spans use `[start_ms,end_ms)`; scenes retain `KeyframeLocator` links | Speaker identity and ASR confidence are unavailable; an unprocessed video stream cannot be hidden as transcript-only success | [Media fixtures](tests/fixtures/media), [support recipe](docs/recipes/support-media.md), `modality_integration` |

The five-family report is deliberately segmented: a passing text or audio case
cannot mask an OCR, layout, vision, or video gap. See the
[evaluation runner](docs/evaluation-runner.md) and the
[committed requirement evidence](reports/evaluation/requirements-evidence-v1.json)
before making capability claims.

Canonical retrieval relevance is finite and higher-is-better, with stable chunk
ID tie-breaking. Native dense similarity, distance, BM25, fusion, and
cross-encoder values retain their kind and are not cross-stage calibrated.
Index manifests bind corpus, chunker, embedder, vector dimension,
normalization, and schema; mismatch fails before search or mutation.

## Extension map

Start at the contract, implement one adapter, run its reusable contract tests,
then add its configuration selection and composition entry. Do not put provider
selection or SDK types in `domain`, `ports`, or `application`.

| Change | Contract | Existing adapter to read | Composition/configuration | Proof |
|---|---|---|---|---|
| Source acquisition | [`SourceConnector`](src/ragkit/ports/interfaces.py) | [`FilesystemSourceConnector`](src/ragkit/adapters/filesystem.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.connector` | [Adapter contracts](tests/contract/test_phase2_adapter_contracts.py) |
| Classification/extraction | [`FamilyClassifier`](src/ragkit/ports/interfaces.py), [`DocumentExtractor`](src/ragkit/ports/interfaces.py) | [`TextFamilyClassifier`](src/ragkit/adapters/textual.py), [`DeclaredFamilyClassifier`](src/ragkit/adapters/classification.py), [`TextDocumentExtractor`](src/ragkit/adapters/textual.py), [`OcrDocumentExtractor`](src/ragkit/adapters/ocr.py), [`LayoutDocumentExtractor`](src/ragkit/adapters/layout.py), [`VisionDocumentExtractor`](src/ragkit/adapters/vision.py), [`MediaDocumentExtractor`](src/ragkit/adapters/media.py) | [Profile schema](src/ragkit/infrastructure/config.py), `family`, `components.classifier`, `components.extractor` | [Family integration](tests/integration/test_ocr_layout_integration.py), [model/media integration](tests/integration/test_vision_media_ml_integration.py) |
| Projection/chunking | [`DocumentProjector`](src/ragkit/ports/interfaces.py), [`Chunker`](src/ragkit/ports/interfaces.py) | [`NoOpDocumentProjector`](src/ragkit/adapters/textual.py), [`StructureAwareChunker`](src/ragkit/adapters/textual.py), [`EvidenceChunker`](src/ragkit/adapters/multimodal.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.projector`, `components.chunker` | [Offline adapter tests](tests/unit/adapters/test_offline_rag_adapters.py) |
| Embeddings | [`Embedder`](src/ragkit/ports/interfaces.py) | [`HashingEmbedder`](src/ragkit/adapters/retrieval.py), [`TorchTextEmbedder`](src/ragkit/adapters/torch_embedder.py) | [Profile settings](src/ragkit/infrastructure/config.py), `components.embedder` | [Production adapter tests](tests/unit/adapters/test_production_adapters.py) |
| Vector storage | [`VectorStore`](src/ragkit/ports/interfaces.py) | [`InMemoryVectorStore`](src/ragkit/adapters/retrieval.py), [`ChromaVectorStore`](src/ragkit/adapters/chroma_store.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.vector_store` | [Vector-store contract/integration](tests/integration/test_chroma_store.py) |
| Retrieval/fusion | [`Retriever`](src/ragkit/ports/interfaces.py) | [`DenseRetriever`](src/ragkit/adapters/retrieval.py), [`BM25Retriever`](src/ragkit/adapters/retrieval.py), [`HybridRetriever`](src/ragkit/adapters/retrieval.py) | [Profile settings](src/ragkit/infrastructure/config.py), `components.retriever` | [Retrieval contract](tests/contract/test_phase4_retrieval_contract.py) |
| Reranking | [`Reranker`](src/ragkit/ports/interfaces.py) | [`NoOpReranker`](src/ragkit/adapters/retrieval.py), [`LocalCrossEncoderReranker`](src/ragkit/adapters/cross_encoder_reranker.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.reranker` | [Reranker integration](tests/integration/test_cross_encoder_reranker_integration.py) |
| Prompt/generation | [`PromptBuilder`](src/ragkit/ports/interfaces.py), [`Generator`](src/ragkit/ports/interfaces.py) | [`TemplatePromptBuilder`](src/ragkit/adapters/generation.py), [`ExtractiveGenerator`](src/ragkit/adapters/generation.py), [`OpenAIHostedGenerator`](src/ragkit/adapters/hosted.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.prompt_builder`, `components.generator` | [Application orchestration](tests/unit/application/test_orchestration.py) |
| Evaluation | [`Evaluator`](src/ragkit/ports/interfaces.py) | [`DeterministicEvaluator`](src/ragkit/adapters/generation.py) | [Evaluation package](src/ragkit/evaluation), `components.evaluator` | [Hand-calculated metrics](tests/unit/evaluation/test_evaluation.py) |
| Telemetry | [`Telemetry`](src/ragkit/ports/interfaces.py) | [`InMemoryTelemetry`](src/ragkit/adapters/observability.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.telemetry` | [Application orchestration](tests/unit/application/test_orchestration.py) |

For a new assignment, inspect a reviewed template without writing files:

```bash
uv run python scripts/bootstrap_assignment.py --template local-offline \
  --destination ./my-rag-assignment --dry-run
```

Then remove `--dry-run` to copy the
[local/offline](examples/assignment_profiles/local-offline) or
[hosted/persistent](examples/assignment_profiles/hosted-persistent) template.
The helper refuses collisions by default; `--overwrite` is a separate explicit
choice and replaces only changed managed files after preflight. Customize the
generated profile before changing toolkit internals.

The stable IDs, immutable records, locators, error taxonomy, ordering,
determinism, side effects, and optional-dependency rules are authoritative in
[Core Port Contracts](docs/contracts.md). Changing one of those semantics is an
architecture decision, not an adapter-local convenience.

## Validation

Run the core, non-mutating baseline before handing work to another person or
agent:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run python -m compileall -q src tests
uv run python scripts/check_imports.py
timeout 60 uv run pytest -m unit --no-cov
timeout 60 uv run pytest -m contract --no-cov
timeout 60 uv run pytest -m integration --no-cov
timeout 60 uv run pytest -m e2e --no-cov
uv run python scripts/check_readme.py --execute
```

The comprehensive 80% coverage gate additionally requires Tesseract 5.x with
English language data. Its first run may resolve the locked Python extras into
the project environment:

```bash
tesseract --version
tesseract --list-langs | rg '^eng$'
uv run --frozen --all-extras pytest --cov=ragkit --cov-report=term-missing
```

Optional model/provider tests stay opt-in and must not be moved into unit or
contract markers. The comprehensive command performs no model download or paid
call. The complete authoritative gate, including formatting commands for
active edits, is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Deeper documentation

- [Architecture and dependency direction](ARCHITECTURE.md)
- [Port contracts and canonical semantics](docs/contracts.md)
- [Adapter extension guide](docs/extension-guide.md)
- [Cold-agent comprehension evidence](reports/agent-guidance/cold-agent-drill-v1.md)
- [Offline adapter behavior](docs/offline-adapters.md)
- [Five-family support, limits, confidence, and fallback policy](docs/modality-support.md)
- [Production adapters and provisioning](docs/production-adapters.md)
- [Evaluation schema, metrics, and benchmark contract](docs/evaluation.md)
- [Executable Phase 4 evaluation runner](docs/evaluation-runner.md)
- Business paths: [knowledge-base text](docs/recipes/knowledge-base-text.md), [claims OCR](docs/recipes/claims-ocr.md), [financial layout](docs/recipes/financial-layout.md), [equipment vision](docs/recipes/equipment-vision.md), [support media](docs/recipes/support-media.md), and [cross-encoder reranking](docs/recipes/cross-encoder-reranking.md)
- [Architecture decisions](docs/decisions/0001-functional-core-and-sync-first-ports.md)
- [Fixture provenance and limits](tests/fixtures/README.md)
