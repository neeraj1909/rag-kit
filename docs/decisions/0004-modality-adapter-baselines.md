# ADR 0004: CPU/offline-first modality adapter baselines

- Status: Accepted
- Date: 2026-08-09
- Decision owners: ingestion/adapters maintainers
- Validation owners: per-family adapter contract and integration suites
- Related requirements: F-13 through F-18

## Context

`rag-kit` must support textual, OCR, layout-aware, vision-language, and
time-based media documents while preserving exact provenance. Core installation
must stay dependency-free, and a CPU/offline path must be credible for a timed ML
assignment. A file suffix cannot establish capability: PDFs, presentations, and
videos commonly contain subparts that require more than one adapter.

The live TechDocs command surface was checked, but its configured index directory
was absent. Official project documentation was reviewed instead; the source and
version-gap record is in `docs/modality-support.md`. No exact dependency version is
asserted by this ADR. The resolved `uv.lock`, immutable model revision, and adapter
contract evidence are the version authority once populated.

## Decision

Adopt five independently installable extras and a single normalized extractor
boundary:

| Extra | Baseline | Explicit external/model dependency |
|---|---|---|
| `text` | Python standard-library decoding, MIME/email, and HTML event parsing | None |
| `ocr` | Pillow + `pytesseract` + `pypdfium2` | Tesseract executable and reviewed `tessdata` language files |
| `layout` | `pdfplumber` + `python-pptx` + `openpyxl` | None in the baseline; encrypted inputs require caller credentials |
| `vision` | PyTorch/Transformers image-to-text | Cached, revision-pinned `HuggingFaceTB/SmolVLM-256M-Instruct` weights |
| `media` | `faster-whisper` CPU/int8 ASR + `scenedetect` scenes/keyframes | Cached, inventoried converted Whisper model; no system `ffmpeg` executable for the baseline path |

Every adapter returns normalized typed parts, asset identity/digest, exact locators,
extractor/model identity, raw confidence signals where the engine has them, and
structured degradation notices. The orchestrator enumerates and routes subparts.
There is no implicit cross-family fallback. In particular:

- an image-only PDF is not considered successfully processed by empty PDF text;
- layout extraction does not silently OCR scanned pages;
- a vision model does not silently become OCR or filename-based captioning;
- a video is not complete when only its audio track was processed; and
- missing optional packages, binaries, language data, weights, codecs, or passwords
  produce typed actionable errors.

Partial output exists only through an explicit caller policy and always records
which assets/regions/tracks were unprocessed. The default policy is fail-closed for
required capabilities.

### Confidence and degradation

- Deterministic textual extraction reports confidence as not applicable.
- Tesseract word confidence is retained raw and aggregated only with a documented
  method; low-confidence words/fields remain visible for review.
- Layout reading order and table detection are heuristic; ambiguity is a notice,
  not a made-up probability.
- SmolVLM descriptions are generated, uncalibrated evidence. Their confidence is
  unavailable and the original image/region remains the authoritative citation.
- Faster-whisper engine signals are retained as raw ASR metadata, not presented as
  a universal probability. Diarization and inferred speaker identity are excluded.

Printed/scanned OCR is the acceptance baseline. Handwriting is explicitly
best-effort and degraded until a separately reviewed handwriting model passes a
representative contract suite. This limitation is preferable to implying that
Tesseract provides dependable handwriting recognition.

### Bounded execution

Adopt the default source/page/pixel/cell/image/duration/scene/time envelopes in
`docs/modality-support.md`. Limit breaches fail before unbounded work. Model downloads
are provisioning operations, never implicit during ordinary ingestion or tests.
Offline tests deny network access and use deterministic fakes except for marked,
bounded local integration tests.

## Representative acceptance set

Each family owns a redistributable synthetic/public-domain fixture and one end-to-end
business recipe:

1. textual policy/email/code bundle -> internal knowledge search;
2. scanned printed claim form with a blurred field -> claims intake and review;
3. report PDF, pricing deck, and formula workbook -> financial/pricing analysis;
4. equipment photo, chart, diagram, and mixed page -> maintenance triage; and
5. short support audio and three-scene training video -> timestamped support/training search.

Fixtures assert provenance, relationships, degradation and missing-capability
behavior. Generative outputs use semantic/schema assertions rather than brittle
exact prose. Fixture provenance and licenses are recorded alongside the assets; no
real customer PII is permitted.

## Consequences

### Positive

- Core remains small while every mandatory family has an explicit installation and
  validation owner.
- CPU/offline operation is possible without hiding model provisioning or native
  runtime costs.
- Separate adapters can be replaced later without changing domain/application code.
- Provenance and negative-path requirements prevent impressive-looking but
  incomplete multimodal demos.

### Trade-offs and risks

- Tesseract is not a dependable handwriting engine; handwriting-specific support
  remains an explicit future adapter.
- `pdfplumber` reading order/table extraction, PPTX shape ordering, and spreadsheet
  cached formula values need document-specific policies and cannot guarantee visual
  semantics.
- A 256M VLM is resource-friendly but weak on fine print, charts, diagrams, and
  specialist imagery. It is a replaceable baseline, not a quality ceiling.
- CPU ASR/VLM latency can exceed interactive budgets, so duration, image, and timeout
  limits are required.
- Python package licenses alone are insufficient: native libraries, codec/patent
  constraints, model weights/training terms, and fixture rights need release review.

## Rejected alternatives

- **One heavyweight universal document framework:** rejected for core dependency
  size, opaque fallback/routing, difficult provenance mapping, and slower assignment
  customization.
- **Hosted OCR/VLM/ASR as the default:** rejected because credentials, cost, privacy,
  and network availability would violate the offline reference profile.
- **PyMuPDF as the PDF baseline:** technically attractive, but its AGPL/commercial
  licensing boundary is a poor default for a reusable permissively licensed kit.
- **OpenAI Whisper Python package as the ASR runtime:** valid, but faster-whisper's
  CPU/int8 CTranslate2 path and PyAV decoding are a more pragmatic local baseline.
- **Treat all extracted/generated text equally:** rejected because confidence,
  provenance, and evidence authority differ fundamentally by family.
- **Automatic silent fallback:** rejected because it makes completeness and business
  risk impossible to audit.

## Validation gates

- Core import succeeds with no modality extras installed and no network/model access.
- Each extra installs independently and missing runtime/model errors name the exact
  extra and provisioning action.
- Per-family contract tests verify normalized parts, locators, extractor metadata,
  limits, confidence semantics, and corrupt/unsupported input behavior.
- Cross-family tests prove scanned PDF OCR routing, embedded-image vision routing,
  and video transcript/keyframe linkage without silent loss.
- Release validation inventories dependency, native binary, model, and fixture
  licenses and records the immutable model revisions used.

## Authoritative references

See the direct official URLs in `docs/modality-support.md`, especially the
pytesseract/Tesseract output contracts, pdfplumber object coordinates and tables,
Transformers image-text-to-text task, SmolVLM model card, faster-whisper runtime
requirements, Whisper license, and PySceneDetect scene/timecode APIs.
