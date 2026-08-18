# Phase 2 Offline Adapter Behavior

The offline profile is synchronous, deterministic for unchanged inputs and
configuration, and performs no network access. Calls block the invoking thread.
No adapter starts background work.

| Adapter | Effects and determinism | Limits and concurrency |
|---|---|---|
| `FilesystemSourceConnector` | Reads local regular files in stable path order. Directory selection recursively excludes generated `__pycache__` trees and rejects symbolic links. Asset identity uses the canonical file URI and bytes. | Enforces request asset-count and per-asset byte limits before returning. Filesystem mutation during a read raises an integrity/provider error. Instances hold no mutable state and may be called concurrently, subject to filesystem consistency. |
| `TextFamilyClassifier` | Pure media-type classification for text, Markdown, HTML, email, and supported code suffixes. | No internal limit or mutable state; thread-safe. Unsupported types fail explicitly. |
| `TextDocumentExtractor` | Deterministic UTF-8 and standard-library extraction with exact source spans. HTML markup, scripts, and styles are excluded without rewriting visible source text. | Enforces the document limit. Multipart/encoded email and invalid or empty searchable input fail explicitly. Stateless and thread-safe. |
| `NoOpDocumentProjector` | Pure identity projection. | Rejects, rather than truncates, documents above the part limit. Stateless and thread-safe. |
| `StructureAwareChunker` | Pure deterministic paragraph/whitespace splitting; retains exact locators and Markdown heading paths when present. | Enforces configured character and request chunk limits. Stateless after construction and thread-safe. |
| `AdaptiveChunker` | Pure deterministic dispatch across a resolved family-compatible `ChunkingPolicy`; retains exact text spans or atomic non-text provenance. | Enforces request chunk limits; atomic cells, regions, and timed evidence may exceed the character target rather than fabricate finer locators. Stateless after construction and thread-safe. |
| `HashingEmbedder` | Pure SHA-256 feature hashing with fixed dimension and L2 normalization; no model or cache I/O. | Rejects non-positive dimensions and blank queries. Stateless after construction and thread-safe. |
| `InMemoryVectorStore` | Process-local mutation and deterministic dense cosine ranking. A manifest must be established by upsert before search/delete; every operation validates manifest, fingerprint, dimension, normalization, and vector norm before store work. | Atomic under an internal re-entrant lock and safe for concurrent calls in one process. Memory is bounded only by explicit outer workflow limits; it is not persistent. |
| `DenseRetriever` | Embeds one query and delegates manifest-aware search to the configured vector store. | Read-only and bounded by request `top_k`; provider effects and concurrency follow the injected embedder and store. |
| `BM25Retriever` | Maintains a process-local manifest-bound lexical index and emits native BM25 scores with explicit provenance. Tokenization, IDF, filtering, and parameters are fingerprinted. | Upsert/delete are lock-protected and idempotent; retrieval is bounded by `top_k`. It rejects use before initialization and incompatible manifests. |
| `HybridRetriever` | Queries named children and fuses ranks only through deterministic reciprocal-rank fusion. Child raw scores remain separate history and are never numerically mixed. | Read-only; candidate expansion is explicitly bounded and final ties use stable chunk IDs. Effects and thread safety follow the child retrievers. |
| `NoOpReranker` | Pure deterministic canonical reordering of the supplied subset. | Bounded by request `top_k`; stateless and thread-safe. |
| `LocalCrossEncoderReranker` | Loads one reviewed cached revision with network access disabled, runs evaluation/inference mode, and emits finite logits as a new score stage while preserving exact chunks and prior scores. | CPU-only and bounded by configured batches, sequence length, candidate count, and top-k. Model loading/inference blocks the caller; concurrent safety follows PyTorch/Transformers. |
| `TemplatePromptBuilder` | Pure deterministic prompt construction. Evidence is JSON-quoted so delimiter-like content cannot become prompt structure. | Includes only complete chunks within the character budget; stateless and thread-safe. |
| `ExtractiveGenerator` | Pure deterministic first-evidence excerpt at temperature zero. It reports no tokenizer usage because this dependency-free implementation has no tokenizer. | Bounded by `max_output_tokens`, interpreted as whitespace-separated output words; nonzero temperature fails explicitly. Stateless and thread-safe. |
| `DeterministicEvaluator` | Pure deterministic retrieval-hit and expected-answer containment metrics. Retrieval hit rate is emitted only when relevance labels exist. | Requires at least one case through the port model; stateless and thread-safe. |
| `InMemoryTelemetry` | Records sanitized events in call order with no external I/O. | Enforces configured event, attribute, and value bounds. Mutable and lock-protected for concurrent calls; records live only in the process. |

The filesystem selection-policy identifier is included in the corpus manifest
fingerprint. Changing exclusions or canonicalization therefore creates an
incompatible index generation instead of silently reusing prior vectors.
