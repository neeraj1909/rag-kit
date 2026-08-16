# Phase 3 production adapters

Phase 3 keeps provider details in adapters and explicit configuration. Core
imports remain dependency-free; each optional capability either runs as
selected or raises a typed, actionable error without a fallback.

| Selection | Determinism and effects | Hard bounds / thread safety | Honest limitation |
|---|---|---|---|
| `ocr` | Local Tesseract subprocess; fixed bytes/settings retain boxes and confidence | Page, pixel, time, document, and part limits; PDFium calls are serialized | Handwriting/forms are best-effort and low confidence is explicit |
| `layout` | Local PDF/PPTX/XLSX reads with deterministic container order | Page/slide/sheet/cell/archive limits; no shared mutable state | Embedded images fail closed for OCR/vision routing |
| `vision` | Cached immutable SmolVLM revision, CPU eval/inference only | Pixel/dimension/output/region limits plus a 60-second deadline; model instance is not declared thread-safe | Descriptions are model-derived, untrusted, and uncalibrated |
| `media` | Cached immutable faster-whisper revision and deterministic scene midpoint | 30-minute, segment, scene, and two-hour operation limits; engines are not declared thread-safe | No diarization; speaker identity and ASR confidence remain unknown |
| `torch` | Cached immutable encoder revision, eval/inference, explicit batching/pooling/L2 | Batch/max-length/device settings; model instance is not declared thread-safe | No implicit downloads or GPU fallback |
| `sqlite` | Standard-library local database with transactional manifest preflight and exact JSON row decoding | Request `top_k`; SQLite serializes writers while reads use independent connections | Exact cosine search scans the bounded profile index; no implicit migration/reset |
| `openai` | Explicit network call with bounded SDK timeout/retry settings | Request output limit; SDK client concurrency semantics | Mocked by default; one double-opt-in paid smoke exists, but no live-quality or availability claim |

`ragkit inspect-config` reports the five supported families, selected component
fingerprints, device, limits, degraded modes, optional extras/binaries,
credential presence, and exact missing-model provisioning commands. It checks
credential presence only and never emits the value.

The synchronous reference CLI enforces local model deadlines with POSIX process
timers on the main thread. Alternate worker/thread runtimes fail closed and must
provide a process-level deadline boundary instead of silently disabling timeouts.

The repository fixtures are synthetic and redistributable. OCR/layout fixtures
exercise real local engines; generated PNG/WAV/MP4 fixtures also exercise the
pinned VLM, ASR, and scene adapters in an opt-in offline run. These runs prove
execution and provenance, not VLM/ASR quality.
The pinned MiniLM integration is opt-in after explicit provisioning:

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RAGKIT_RUN_MODEL_INTEGRATION=1 \
  uv run pytest -m modality_integration --no-cov
```

## Hosted live smoke

Mocked tests own request bounds, citation sanitization, timeout/retry wiring, and
redacted error translation. A real provider reachability check is deliberately
separate because it reads a credential, opens a network connection, and may incur
cost. It is collected but skipped unless both gates are present:

```bash
RAGKIT_RUN_LIVE=1 OPENAI_API_KEY="$OPENAI_API_KEY" \
  uv run --frozen --extra hosted pytest -m live \
  tests/live/test_openai_hosted_live.py --no-cov -q
```

The smoke makes one Responses API request with zero retries, an eight-token output
cap, and a 30-second timeout. Passing proves bounded adapter reachability for the
selected model at that moment; it does not prove quality, uptime, or future model
compatibility. Routine CI never enables it.
