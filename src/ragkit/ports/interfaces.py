"""Synchronous capability boundaries implemented by adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.domain import Chunk, ComponentFingerprint, Document, Embedding, ScoredChunk

from .models import (
    AcquiredAsset,
    AssetClassification,
    ChunkingRequest,
    DeleteRequest,
    EmbeddingBatch,
    EmbeddingRequest,
    EvaluationReport,
    EvaluationRequest,
    ExtractionRequest,
    GenerationRequest,
    GenerationResult,
    ProjectionRequest,
    Prompt,
    PromptRequest,
    RerankRequest,
    RetrievalRequest,
    SourceRequest,
    TelemetryEvent,
    UpsertRequest,
    VectorSearchRequest,
)


class SourceConnector(ABC):
    """Acquire original bytes without interpreting document modality.

    Implementations are blocking and need not be thread-safe unless their adapter
    documentation says otherwise. They must enforce caller limits before returning,
    translate I/O failures to typed project errors, and raise
    ``UnsupportedCapabilityError`` for unsupported URI schemes. A successful call
    returns assets in source order, rejects invalid requests, and has no effects beyond
    acquisition documented by the adapter. It is deterministic when the source is
    unchanged. This boundary produces no confidence or retrieval score.
    """

    @abstractmethod
    def fetch(self, request: SourceRequest) -> tuple[AcquiredAsset, ...]:
        """Return at most ``max_assets`` complete assets in source order.

        Blank or malformed locations are invalid. Size/count excess and partial
        reads raise typed limit or integrity errors rather than returning truncated
        bytes. Repeated calls may observe a changed external source.
        """


class FamilyClassifier(ABC):
    """Classify acquired assets into supported document families.

    The blocking operation preserves input order and count and performs no mutation.
    Confidence, when available, is a calibrated or adapter-documented value in
    ``[0, 1]``; absence means unknown, never certainty. Invalid assets and provider
    failures raise typed errors, while an unclassifiable family raises
    ``UnsupportedCapabilityError``. The operation's only effects are adapter-documented
    model reads. Results are deterministic only when the adapter
    fingerprint and input bytes are unchanged. Thread safety is adapter-specific.
    Classification has no retrieval-score semantics; caller acquisition limits still
    bound inputs.
    """

    @abstractmethod
    def classify(self, assets: tuple[AcquiredAsset, ...]) -> tuple[AssetClassification, ...]:
        """Return exactly one classification per asset in the same order."""


class DocumentExtractor(ABC):
    """Extract provenance-bearing documents from classified original assets.

    Output follows input asset order and is capped by the ``max_documents`` limit. Invalid
    classification alignment, unsupported families, provider failures, and rejected
    partial extraction use typed errors. The adapter may read models or subprocesses
    but must not mutate inputs; other effects and thread safety are adapter-specific.
    Confidence describes extraction evidence, not correctness or retrieval score.
    Identical bytes, request, and fingerprint must produce deterministic ordering;
    stochastic implementations must document their weaker repeatability.
    """

    @abstractmethod
    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        """Return complete documents whose parts retain exact asset locators."""


class DocumentProjector(ABC):
    """Add searchable representations without replacing original evidence.

    The blocking operation preserves document order, enforces the part limit, and
    returns new immutable documents without mutating input. Unsupported family or
    projection capability and invalid provenance raise typed errors. Confidence is
    attached to derived evidence and is not a retrieval score. Determinism and thread
    safety follow the adapter fingerprint and documentation; external model effects
    may occur only in the implementation.
    """

    @abstractmethod
    def project(self, request: ProjectionRequest) -> tuple[Document, ...]:
        """Return documents with provenance-linked searchable projections."""


class Chunker(ABC):
    """Create bounded ordered searchable units from immutable documents.

    Chunking is side-effect free and deterministic for a fingerprint and input.
    Output preserves document order, then the adapter's declared within-document
    order, and never exceeds the ``max_chunks`` limit. Invalid or provenance-free parts raise
    typed errors; unsupported families raise ``UnsupportedCapabilityError``.
    Extraction confidence is preserved but does not become a score. Pure chunkers
    should be thread-safe; an exception must be documented by the adapter.
    """

    @property
    @abstractmethod
    def fingerprint(self) -> ComponentFingerprint:
        """Identify every behavior-affecting algorithm and limit parameter."""

    @abstractmethod
    def chunk(self, request: ChunkingRequest) -> tuple[Chunk, ...]:
        """Return provenance-complete chunks in stable source order."""


class Embedder(ABC):
    """Map text to fixed-width vectors under one component fingerprint.

    Document embedding preserves input order and count; empty batches return an empty
    batch. Invalid blank individual text and limit violations raise typed errors. Unsupported
    query/document modes raise ``UnsupportedCapabilityError``. Calls have no intended
    mutation effects; model/cache effects and thread safety are adapter-specific.
    Determinism is declared by the adapter. Embeddings carry neither confidence nor a
    retrieval score; later adapters define score conversion.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the positive, invariant vector width; invalid setup raises an error."""

    @property
    @abstractmethod
    def fingerprint(self) -> ComponentFingerprint:
        """Identify model revision, pooling, normalization, dimension, and limits."""

    @abstractmethod
    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingBatch:
        """Return one aligned embedding for each input text."""

    @abstractmethod
    def embed_query(self, text: str) -> Embedding:
        """Embed one non-blank query using declared query-specific behavior."""


class VectorStore(ABC):
    """Persist and search chunks behind manifest-compatible vector semantics.

    Operations are blocking; thread safety is adapter-specific. Manifest mismatch,
    invalid dimensions/filters/limits, unsupported filters, and provider failures are
    typed errors. Upsert and delete mutate persistent state; search is read-only.
    Same-ID/same-manifest upsert is idempotent and deterministic. Search returns at
    most ``top_k`` unique chunks ordered by descending canonical relevance, then
    stable chunk ID. Raw score provenance is retained; relevance is higher-is-better,
    not confidence or probability. Compatibility is checked before any effect.
    """

    @abstractmethod
    def upsert(self, request: UpsertRequest) -> None:
        """Idempotently store aligned chunks and embeddings after manifest validation."""

    @abstractmethod
    def search(self, request: VectorSearchRequest) -> tuple[ScoredChunk, ...]:
        """Return bounded, unique, deterministically ordered scored chunks."""

    @abstractmethod
    def delete(self, request: DeleteRequest) -> None:
        """Delete the named chunks idempotently after manifest validation."""


class Retriever(ABC):
    """Retrieve provenance-complete candidates for one query.

    The operation is blocking and read-only, so its effect does not mutate the index. Its
    positive ``top_k`` limit caps results, which contain
    unique chunks ordered by descending higher-is-better relevance with stable-ID tie
    breaks. Relevance is an ordering score, not confidence or probability; raw score
    provenance remains available. Blank queries, invalid filters, unsupported search
    capability, and provider errors are typed. Determinism and thread safety depend
    on the declared implementation and unchanged index.
    """

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> tuple[ScoredChunk, ...]:
        """Return at most ``top_k`` candidates under the declared score policy."""


class Reranker(ABC):
    """Assign a new explicit scoring stage to an existing candidate set.

    The blocking, read-only operation returns at most positive ``top_k`` candidates,
    never introduces a chunk, and orders higher relevance first with stable-ID ties.
    Prior score provenance and extraction confidence are preserved. Invalid duplicate
    candidates and limits or unsupported reranking raise typed errors. Determinism,
    provider effects, and thread safety are adapter-specific and fingerprinted.
    """

    @abstractmethod
    def rerank(self, request: RerankRequest) -> tuple[ScoredChunk, ...]:
        """Return a bounded subset with reranker score provenance and stable order."""


class PromptBuilder(ABC):
    """Build a bounded prompt from a query and ordered retrieval evidence.

    This operation is deterministic, side-effect free, and thread-safe. It preserves
    context order and citation identity while enforcing ``max_context_chars``.
    Invalid text or limits raise typed errors; an unsupported content representation
    raises ``UnsupportedCapabilityError``. Confidence and retrieval scores may guide
    explicit context selection but must not be rewritten or presented as probability.
    """

    @abstractmethod
    def build(self, request: PromptRequest) -> Prompt:
        """Return an injection-aware prompt and the exact included chunk IDs."""


class Generator(ABC):
    """Generate an answer from the exact supplied prompt and retrieval context.

    The blocking operation preserves cited-chunk order, enforces output token limits,
    and does not mutate retrieval scores or extraction confidence. Invalid requests,
    unsupported generation controls, and provider failures raise typed errors.
    Network/model effects, determinism, and thread safety are adapter-specific; the
    result identifies the fingerprint needed to interpret those guarantees.
    """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return generated text, actual citations, model identity, and optional usage."""


class Evaluator(ABC):
    """Compute reproducible metrics from already-observed pipeline cases.

    Evaluation is side-effect free, preserves case order in its evidence, and applies
    no implicit case limit. Empty or invalid cases and unsupported metrics raise typed
    errors. Metric values are scores under named definitions, not confidence or
    cross-evaluator probabilities. For fixed inputs and fingerprint results are
    deterministic and thread-safe; this port performs no provider orchestration.
    """

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationReport:
        """Return finite named metrics plus the ordered IDs of evaluated cases."""


class Telemetry(ABC):
    """Record bounded operational metadata without raw document or prompt content.

    Calls are blocking and preserve call order; invalid timings or attributes raise
    typed errors and unsupported sinks fail explicitly. Recording is an intentional
    external effect and implementations document durability and thread safety.
    Delivery may be nondeterministic, but event values are not changed. Attributes do
    not encode extraction confidence or retrieval scores unless explicitly named and
    safe; adapter limits prevent unbounded cardinality.
    """

    @abstractmethod
    def record(self, event: TelemetryEvent) -> None:
        """Record one sanitized event or raise a typed error; never silently discard it."""
