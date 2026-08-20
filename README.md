# rag-kit

`rag-kit` is a typed, modular foundation for building retrieval-augmented
generation (RAG) systems. It can ingest five document families, choose a
document-aware chunking policy, build dense, sparse, or hybrid indexes, retrieve
and optionally rerank evidence, generate a cited answer, and evaluate the result.

The default path is dependency-free, local, deterministic, and small enough for
a take-home assignment. Optional adapters add OCR, layout parsing, local models,
HTTP delivery, hosted generation, and several vector databases without moving
provider types into the application core.

## Purpose and non-goals

The project is designed for engineers who need a credible RAG starting point but
still want to show their own architecture and product decisions. Its central
contract is **evidence preservation**: every searchable chunk and answer citation
retains the original asset identity and an exact text span, page/box, table cell,
timestamp, or keyframe locator.

What is included:

- text, OCR, layout, vision, and audio/video ingestion;
- 20 concrete chunking strategies with an explicit family compatibility matrix;
- dense, BM25 sparse, and reciprocal-rank-fused hybrid indexing/retrieval;
- exact, HNSW, IVF-flat, and provider-managed physical index selections where
  the selected database supports them;
- memory, SQLite, pgvector, Qdrant, Pinecone, and OpenSearch vector stores;
- optional local cross-encoder reranking and optional OpenAI generation;
- Python library, CLI, dependency-light ASGI, and hardened Docker Compose paths;
- deterministic IDs, immutable index manifests, typed errors, evaluation
  reports, benchmarks, and redacted request-correlated telemetry.

This remains a pre-alpha toolkit, not a managed ingestion service or a claim of
production-scale quality. It does not download model weights implicitly, hide an
unsupported modality behind a text fallback, treat OCR/model output as certain,
or make the process-local `memory` store durable. Provider mocks do not prove a
remote service, and no paid live-provider result is claimed.

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
database, subprocess, and provider I/O. These choices are independent:

| Decision | Examples | Meaning |
|---|---|---|
| Document family | text, OCR, layout, vision, media | What evidence is extracted and how it is cited |
| Chunking strategy | paragraph, legal, table, image region, scene | Where evidence boundaries are placed |
| Logical indexing | dense, sparse, hybrid | Which searchable projections are materialized |
| Physical index | exact, HNSW, IVF-flat, managed | How a vector backend searches candidates |
| Vector database | memory, SQLite, pgvector, Qdrant, Pinecone, OpenSearch | Where vectors and manifests live |
| Retrieval | dense, BM25, hybrid RRF | How candidates are selected |
| Reranking | no-op, local cross-encoder | How retrieved candidates are rescored |
| Generation | extractive, hosted OpenAI | How authorized evidence becomes an answer |

A vector database is not a chunking or indexing strategy, and a cross-encoder is
a post-retrieval reranker rather than another retrieval system.

## Document families and provenance

All five families normalize into the same immutable `Document`, `Chunk`, and
`ScoredChunk` contracts. That is why any family can use dense, sparse, or hybrid
indexing and any configured vector store.

| Family | Reference inputs | Searchable evidence | Preserved locator and confidence |
|---|---|---|---|
| Text | TXT, Markdown, HTML, email, code-like UTF text | decoded text and structural paths | exact `TextSpanLocator` |
| OCR | PNG, JPEG, TIFF, BMP, WebP, scanned PDF | printed/form/handwriting-best-effort OCR | page/`BoxLocator`, token confidence, degradation notices |
| Layout | native PDF, PPTX, XLSX/XLSM | ordered regions, tables, headers, merged cells, formulas | page/box/slide/sheet/`CellLocator` and relations |
| Vision | PNG, JPEG, WebP, mixed image+OCR pages | model-derived descriptions and image regions | original asset plus region/page locator; no invented confidence |
| Media | WAV, FLAC, MP3, M4A/MP4 audio, MP4/WebM video | transcript segments, scenes, midpoint keyframes | `[start_ms,end_ms)`, `KeyframeLocator`, and scene links |

The committed [five-family report](reports/evaluation/five-family-execution-v1.json)
contains one bounded passing extraction, retrieval, citation, and locator case per
family. This is representative fixture evidence—not a claim of general OCR,
vision, ASR, or retrieval accuracy. Detailed limits and fallback behavior are in
[modality support](docs/modality-support.md).

## Chunking strategies

Choose a strategy with `settings.chunking_strategy` in TOML or
`--chunking-strategy` on `index`, `ask`, or `evaluate`. `auto` resolves to
`recursive` for text and conservative atomic `evidence` chunks for other
families. The resolved policy and its bounds are fingerprinted into the index
manifest, so changing them requires a compatible re-index.

| Group | Strategies | Good fit |
|---|---|---|
| General text | `fixed`, `sliding_window`, `recursive`, `sentence`, `paragraph` | prose, logs, articles |
| Structured prose | `section`, `hierarchical`, `semantic`, `proposition` | manuals, topic-oriented reports, factual statements |
| Domain-aware | `book`, `legal`, `medical`, `code`, `conversation` | chapters, clauses, clinical sections, declarations, speaker turns |
| Structured/multimodal | `table`, `layout_region`, `image_region`, `transcript_segment`, `scene`, `evidence` | rows/cells, page regions, image regions, timed media, atomic evidence |

The `semantic` strategy is deterministic lexical segmentation, not an embedding
model. Domain-aware strategies are rule-based. Non-character-addressable cells,
regions, or media evidence stay atomic when splitting would invent provenance,
even if that means exceeding a character target. Unsupported family/strategy
pairs fail explicitly; the complete matrix is in
[Chunking strategies](docs/chunking-strategies.md).

Example:

```bash
uv run ragkit index --config configs/offline.toml \
  --source tests/fixtures/corpus --chunking-strategy paragraph
```

## Indexing, retrieval, and reranking

| Logical strategy | Index work | Retrieval behavior |
|---|---|---|
| `dense` | embed chunks and write the selected vector store | query embedding plus vector similarity/distance |
| `sparse` | build BM25 only; no embedder/model or vector-store work | lexical BM25 |
| `hybrid` | preflight and build both dense and sparse indexes | deterministic reciprocal-rank fusion over ranks, not mixed raw scores |

All three strategies work with text, OCR, layout, vision, and media after those
families become provenance-complete chunks. Retrieval returns finite,
higher-is-better relevance with stable chunk-ID tie-breaking while retaining the
provider's raw score kind, metric, and conversion.

The optional [`LocalCrossEncoderReranker`](src/ragkit/adapters/cross_encoder_reranker.py)
can rescore candidates from any retrieval mode. It uses a revision-pinned local
model, bounded candidates/batches/sequence length, CPU inference, and no implicit
download. The default `noop` reranker preserves the original ranking.

Select the logical and physical path independently:

```bash
uv run ragkit ask --config configs/offline.toml \
  --indexing-strategy hybrid --vector-database sqlite \
  "What is the fixture answer?"
```

The request policy must match the composed policy and immutable manifest before
source acquisition or index mutation. Full semantics are in
[Indexing and vector-database strategies](docs/indexing-strategies.md).

## Vector databases

Every database implements the same modality-neutral `VectorStore` port. A
database selector changes composition; it does not create a separate OCR,
vision, or media implementation.

| Selector | Physical index | Persistence/evidence boundary | Setup |
|---|---|---|---|
| `memory` | exact | process-local; shared contract tests | core, no setup |
| `sqlite` | exact | local persistent file; reopen/query/delete integration | core, configure path/collection |
| `pgvector` | exact, HNSW, IVF-flat | adapter and SQL behavior tested; service/restart not claimed | `pgvector` extra; operator-provisioned PostgreSQL/extension/schema |
| `qdrant` | HNSW | adapter plus real local-SDK test; remote service opt-in | `qdrant` extra; URL and optional API-key env |
| `pinecone` | managed | injected-client tests; paid live smoke opt-in | `pinecone` extra; pre-provisioned index/manifest sentinel |
| `opensearch` | HNSW | injected-client tests; service test opt-in | `opensearch` extra; local/hosted endpoint and index |

Stores reject incompatible manifests before search or mutation, preserve full
chunk/provenance round trips, make stable-ID upserts idempotent, translate only
filters they can represent exactly, and keep native score provenance. ANN
backends do not promise exact top-k membership. Provider-specific installation,
provisioning, consistency, and test boundaries are in the linked profiles and
recipes below.

## Ways to use rag-kit

| Surface | What it provides | Starting point |
|---|---|---|
| Python library | compose an `OfflineRuntime` or inject ports into `IndexingService`, `AnsweringService`, and `RagPipeline` | [`examples/minimal_offline.py`](examples/minimal_offline.py) |
| CLI | `inspect-config`, `index`, `ask`, `evaluate`; strategy/database overrides are resolved before composition | `uv run ragkit --help` |
| HTTP/ASGI | `/healthz`, `/readyz`, `/v1/index`, `/v1/ask`; exact JSON schemas, 16 KiB body bound, request IDs, configured-source confinement | [HTTP and observability](docs/observability.md) |
| Docker Compose | non-root, read-only-root, loopback-bound HTTP service with SQLite named-volume persistence | [Container deployment](docs/deployment.md) |

`ask` intentionally indexes and queries in one process, which is convenient for
the default memory profile. The HTTP API separates `/v1/index` and `/v1/ask`.
HTTP requests may echo only the strategy/database already selected by the
profile; callers cannot instantiate arbitrary providers or read arbitrary host
paths.

## Configuration and optional capabilities

Profiles are strict TOML with four readable groups:

- `[profile]`: name, document family, and source;
- `[components]`: connector through telemetry adapter selections;
- `[limits]`: asset, byte, document, part, chunk, context, output, and ranking
  bounds;
- `[settings]`: chunking/indexing policy, model revisions, store settings,
  provider timeouts, and credential environment-variable names.

Run this before doing expensive work:

```bash
uv run ragkit inspect-config --config configs/offline.toml
```

Inspection reports resolved strategies, fingerprints, installed extras,
binaries, cached models, and credential **presence** without reading secret
values or contacting a provider. Core import has no optional dependency,
network, credential, model, database, or GPU requirement.

| Extra | Adds |
|---|---|
| `text`, `persistent` | named zero-dependency capabilities; text baseline and SQLite persistence |
| `ocr` | Pillow, PDF rasterization, pytesseract; Tesseract/tessdata remain system requirements |
| `layout` | PDF, PowerPoint, and Excel parsers |
| `vision` | Pillow, Torch/Torchvision, Transformers for pinned local vision/text models |
| `media` | faster-whisper and SceneDetect |
| `reranking` | Torch/Transformers cross-encoder |
| `hosted` | OpenAI generation adapter |
| `http` | Uvicorn server launcher |
| `pgvector`, `qdrant`, `pinecone`, `opensearch` | one optional vector-store SDK boundary each |

Local model adapters use provisioned, exact revisions and `local_files_only`;
normal tests deny sockets and never download weights. Hosted credential values
are resolved only during composition and never enter manifests, fingerprints,
telemetry, serialization, or error chains.

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
| [`ocr`](configs/ocr.toml) | Printed scans and bounded scanned PDFs | `ocr` extra plus Tesseract and language data; process-local memory | Bounded extraction/retrieval/citation/box-locator case passes |
| [`layout`](configs/layout.toml) | PDF, PPTX, XLSX, and XLSM structure | `layout` extra; parser selected by suffix; process-local memory | Bounded extraction/retrieval/citation/cell-linkage case passes |
| [`vision`](configs/vision.toml) | Image-derived descriptions with region evidence | `vision` extra and exact cached SmolVLM revision; CPU/local-files-only | Bounded region-linked model case passes; general visual accuracy is not claimed |
| [`mixed-image`](configs/mixed-image.toml) | OCR and vision over the same raster page | `ocr` and `vision` extras, Tesseract, cached SmolVLM | Both extractors are required; neither silently substitutes for the other |
| [`media`](configs/media.toml) | Timestamped audio/video evidence | `media` extra and exact cached faster-whisper revision | Bounded transcript/timestamp case passes; linked video keyframe citation has separate e2e proof |
| [`torch-local`](configs/torch-local.toml) | Pinned local text embeddings | `vision` extra supplies Torch/Transformers; exact cached MiniLM revision | CPU repeatability integration passes |
| [`persistent`](configs/persistent.toml) | A store that survives process exit | Standard-library SQLite database at the configured path | Reopen/query/delete and incompatible-manifest rejection pass |
| [`hosted`](configs/hosted.toml) | Optional OpenAI generation | `hosted` extra and `OPENAI_API_KEY`; network call is explicit | Mocked secret/error behavior passes; no paid live smoke is claimed |
| [`pgvector`](configs/pgvector.toml) | PostgreSQL vector storage | `pgvector` extra and DSN env; exact/HNSW/IVF-flat selection | Adapter/SQL proof only; operator service evidence remains required |
| [`qdrant`](configs/qdrant.toml) | Qdrant vector service | `qdrant` extra; HNSW | Real local-SDK contract passes; remote persistence is opt-in |
| [`pinecone`](configs/pinecone.toml) | Managed Pinecone vector index | `pinecone` extra, host/key env, pre-provisioned sentinel | Injected-client proof; paid live test is opt-in |
| [`opensearch`](configs/opensearch.toml) | OpenSearch vector search | `opensearch` extra; HNSW | Injected-client proof; service test is opt-in |

Each profile is a reviewed example, not the only valid combination. Copy one,
then change family, chunking, logical indexing, physical index, vector database,
reranker, generator, and limits as independent choices. Invalid combinations
fail during profile loading or composition rather than silently falling back.

## Evaluation, benchmarks, and observability

The evaluation package uses versioned, strict JSON schemas and records dataset,
corpus, configuration, component, software, and model provenance. It reports
per-case and per-family retrieval recall/hit/reciprocal rank, citation precision
and coverage, extraction coverage, and locator validity. Missing or ineligible
evidence stays explicit and cannot become a decorative pass.

`scripts/benchmark.py` adds warmups, repetitions, p50/p95 latency, throughput,
and a clearly named cumulative child-process memory scope. Latency is
informational unless an explicit gate is configured. Committed reports live in
[`reports/evaluation`](reports/evaluation) and [`reports/benchmarks`](reports/benchmarks).

`InMemoryTelemetry` supports tests and embedding applications;
`JsonLinesTelemetry` supports operations. HTTP mode correlates every nested
index/answer stage with the request ID and emits only bounded scalar metadata:
duration, outcome, component fingerprint, count, and stable error category. It
does not emit queries, source URIs, document text, prompts, answers, exception
messages, credentials, or token values.

## Start a take-home assignment

Two reviewed templates provide a small starting boundary without copying the
whole toolkit. Preview exactly what would be created:

```bash
uv run python scripts/bootstrap_assignment.py --template local-offline \
  --destination ./my-rag-assignment --dry-run
```

Then remove `--dry-run` to copy either the
[local/offline](examples/assignment_profiles/local-offline) or
[hosted/persistent](examples/assignment_profiles/hosted-persistent) template.
The helper preflights both managed files, refuses collisions by default, rejects
links and non-regular targets, and uses staged in-process rollback for failures.
`--overwrite` is an explicit boundary and does not make abrupt process or
filesystem failure crash-atomic. Customize the generated profile before
changing toolkit internals.

## Extension map

Start at the contract, implement one adapter, run its reusable contract tests,
then add its configuration selection and composition entry. Do not put provider
selection or SDK types in `domain`, `ports`, or `application`.

| Change | Contract | Existing adapter to read | Composition/configuration | Proof |
|---|---|---|---|---|
| Source acquisition | [`SourceConnector`](src/ragkit/ports/interfaces.py) | [`FilesystemSourceConnector`](src/ragkit/adapters/filesystem.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.connector` | [Adapter contracts](tests/contract/test_phase2_adapter_contracts.py) |
| Classification/extraction | [`FamilyClassifier`](src/ragkit/ports/interfaces.py), [`DocumentExtractor`](src/ragkit/ports/interfaces.py) | [`TextFamilyClassifier`](src/ragkit/adapters/textual.py), [`DeclaredFamilyClassifier`](src/ragkit/adapters/classification.py), [`TextDocumentExtractor`](src/ragkit/adapters/textual.py), [`OcrDocumentExtractor`](src/ragkit/adapters/ocr.py), [`LayoutDocumentExtractor`](src/ragkit/adapters/layout.py), [`VisionDocumentExtractor`](src/ragkit/adapters/vision.py), [`MediaDocumentExtractor`](src/ragkit/adapters/media.py) | [Profile schema](src/ragkit/infrastructure/config.py), `family`, `components.classifier`, `components.extractor` | [Family integration](tests/integration/test_ocr_layout_integration.py), [model/media integration](tests/integration/test_vision_media_ml_integration.py) |
| Projection/chunking | [`DocumentProjector`](src/ragkit/ports/interfaces.py), [`Chunker`](src/ragkit/ports/interfaces.py) | [`NoOpDocumentProjector`](src/ragkit/adapters/textual.py), [`AdaptiveChunker`](src/ragkit/adapters/adaptive_chunking.py), [`TextStrategyChunker`](src/ragkit/adapters/text_chunking.py), [`ModalityChunker`](src/ragkit/adapters/modality_chunking.py), [`StructureAwareChunker`](src/ragkit/adapters/textual.py), [`EvidenceChunker`](src/ragkit/adapters/multimodal.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.projector`, `components.chunker`, `settings.chunking_strategy` | [Chunking strategy tests](tests/unit/adapters/test_text_chunking_strategies.py) |
| Embeddings | [`Embedder`](src/ragkit/ports/interfaces.py) | [`HashingEmbedder`](src/ragkit/adapters/retrieval.py), [`TorchTextEmbedder`](src/ragkit/adapters/torch_embedder.py) | [Profile settings](src/ragkit/infrastructure/config.py), `components.embedder` | [Production adapter tests](tests/unit/adapters/test_production_adapters.py) |
| Vector storage | [`VectorStore`](src/ragkit/ports/interfaces.py) | [`InMemoryVectorStore`](src/ragkit/adapters/retrieval.py), [`SQLiteVectorStore`](src/ragkit/adapters/sqlite_store.py), [`PgVectorStore`](src/ragkit/adapters/pgvector_store.py), [`QdrantVectorStore`](src/ragkit/adapters/qdrant_store.py), [`PineconeVectorStore`](src/ragkit/adapters/pinecone_store.py), [`OpenSearchVectorStore`](src/ragkit/adapters/opensearch_store.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.vector_store`, `settings.indexing_strategy` | [Indexing policy tests](tests/unit/ports/test_indexing_policy.py) and provider contracts |
| Retrieval/fusion | [`Retriever`](src/ragkit/ports/interfaces.py) | [`DenseRetriever`](src/ragkit/adapters/retrieval.py), [`BM25Retriever`](src/ragkit/adapters/retrieval.py), [`HybridRetriever`](src/ragkit/adapters/retrieval.py) | [Profile settings](src/ragkit/infrastructure/config.py), `components.retriever` | [Retrieval contract](tests/contract/test_phase4_retrieval_contract.py) |
| Reranking | [`Reranker`](src/ragkit/ports/interfaces.py) | [`NoOpReranker`](src/ragkit/adapters/retrieval.py), [`LocalCrossEncoderReranker`](src/ragkit/adapters/cross_encoder_reranker.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.reranker` | [Reranker integration](tests/integration/test_cross_encoder_reranker_integration.py) |
| Prompt/generation | [`PromptBuilder`](src/ragkit/ports/interfaces.py), [`Generator`](src/ragkit/ports/interfaces.py) | [`TemplatePromptBuilder`](src/ragkit/adapters/generation.py), [`ExtractiveGenerator`](src/ragkit/adapters/generation.py), [`OpenAIHostedGenerator`](src/ragkit/adapters/hosted.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.prompt_builder`, `components.generator` | [Application orchestration](tests/unit/application/test_orchestration.py) |
| Evaluation | [`Evaluator`](src/ragkit/ports/interfaces.py) | [`DeterministicEvaluator`](src/ragkit/adapters/generation.py) | [Evaluation package](src/ragkit/evaluation), `components.evaluator` | [Hand-calculated metrics](tests/unit/evaluation/test_evaluation.py) |
| Telemetry | [`Telemetry`](src/ragkit/ports/interfaces.py) | [`InMemoryTelemetry`](src/ragkit/adapters/observability.py), [`JsonLinesTelemetry`](src/ragkit/adapters/observability.py) | [Bootstrap](src/ragkit/infrastructure/bootstrap.py), `components.telemetry` | [Observability tests](tests/unit/adapters/test_observability.py) |

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
- [Chunking strategy catalog and compatibility](docs/chunking-strategies.md)
- [Indexing strategies and vector-database matrix](docs/indexing-strategies.md)
- [Adapter extension guide](docs/extension-guide.md)
- [Cold-agent comprehension evidence](reports/agent-guidance/cold-agent-drill-v1.md)
- [Offline adapter behavior](docs/offline-adapters.md)
- [Five-family support, limits, confidence, and fallback policy](docs/modality-support.md)
- [Production adapters and provisioning](docs/production-adapters.md)
- [HTTP schemas, correlation, and redaction](docs/observability.md)
- [Container deployment and persistence](docs/deployment.md)
- [Packaging and optional-extra evidence](docs/release-packaging.md)
- [Evaluation schema, metrics, and benchmark contract](docs/evaluation.md)
- [Executable Phase 4 evaluation runner](docs/evaluation-runner.md)
- Business paths: [knowledge-base text](docs/recipes/knowledge-base-text.md), [claims OCR](docs/recipes/claims-ocr.md), [financial layout](docs/recipes/financial-layout.md), [equipment vision](docs/recipes/equipment-vision.md), [support media](docs/recipes/support-media.md), and [cross-encoder reranking](docs/recipes/cross-encoder-reranking.md)
- Vector-store setup: [pgvector](docs/recipes/pgvector-indexing.md), [Qdrant](docs/recipes/qdrant-indexing.md), [Pinecone](docs/recipes/pinecone-indexing.md), and [OpenSearch](docs/recipes/opensearch-indexing.md)
- [Architecture decisions](docs/decisions/0001-functional-core-and-sync-first-ports.md)
- [Fixture provenance and limits](tests/fixtures/README.md)
