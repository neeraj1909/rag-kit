# Modality support baseline

This document defines what the five required document families mean in `rag-kit`.
Support is capability-based, not extension-based: a container may contain several
families, and every discovered family must be processed or reported as unsupported.

## Version and source surface

Research was refreshed on 2026-08-09. The local TechDocs CLI was present, but its
configured index (`/home/neeraj/Code/docs-mcp-server/mcp-data`) did not exist, so
`search-all`, `find`, `list`, and `describe` could not return tenants. The official
project documentation linked below is therefore the source surface for these
decisions. The first Phase 0 `uv.lock` is now the authoritative Python-package
resolution for CPython 3.11–3.12. Direct resolved baselines are Pillow 12.3.0,
`pytesseract` 0.3.13, `pypdfium2` 4.30.0, `pdfplumber` 0.11.9,
`python-pptx` 1.0.2, `openpyxl` 3.1.5, PyTorch 2.13.0, Transformers 5.14.1,
`faster-whisper` 1.2.1, PySceneDetect 0.6.7.1, Chroma 1.5.9, and OpenAI 2.53.0.
These versions do not lock external Tesseract/tessdata, model weights, native
codec/runtime policy, or fixture licenses. Do not infer an exact API from an
unpinned URL, and do not advance the lock without rerunning adapter contracts
and reviewing dependency and model licenses.

## Common adapter contract

Each family adapter sits behind the same extraction port. It receives a bounded,
identified source asset plus explicit configuration and returns:

- typed content parts rather than an undifferentiated string;
- the original asset identifier and digest;
- an exact locator appropriate to the family (character span, page and box,
  slide/sheet/cell, image region, or half-open millisecond interval);
- the extractor name/version/config fingerprint and any model revision;
- raw engine confidence signals when available, clearly marked when absent or
  uncalibrated;
- relationships between derived parts and their source asset/region; and
- structured notices for degradation, skipped subparts, limit breaches, or
  unsupported content.

The orchestrator classifies every discovered subpart before extraction. Success is
not allowed when a table, image, audio track, page, or attachment was silently
discarded. A missing extra, missing executable/model, encrypted file, corrupt input,
limit breach, or unsupported codec produces a typed actionable failure. A caller may
explicitly opt into `partial` results; those results carry notices and may not be
reported as complete.

Locators use zero-based indices internally. Boxes are `(x0, y0, x1, y1)` in a
documented top-left coordinate space and include the page/image dimensions needed to
interpret them. Time ranges are `[start_ms, end_ms)`. Adapters never fabricate a
numeric confidence merely to satisfy a schema.

## Capability matrix

| Family / extra | Accepted baseline inputs | CPU/offline-first adapter baseline | Evidence and degraded behavior | Representative fixture | Business recipe | Validation owner |
|---|---|---|---|---|---|---|
| Textual / `text` | UTF-8/UTF-16 `.txt`, `.md`, source/config text, `.html`/`.htm`, `.eml` | Python `pathlib`, codecs, `html.parser`, and `email`; no third-party runtime dependency | Exact decoded character spans and structural paths. Invalid/ambiguous encoding, malformed MIME, excluded attachments, and lossy HTML structure are notices or failures; never guess an encoding silently. Confidence is not applicable. | Small policy bundle with Markdown headings, HTML sections, an email thread/attachment declaration, and code-like text | Internal policy and knowledge search with span-level citations | Core/text contract tests; no-network unit suite |
| OCR / `ocr` | PNG, JPEG, TIFF, BMP, WebP, and bounded scanned-PDF pages | Pillow + `pytesseract`; `pypdfium2` rasterizes PDF pages; the Tesseract engine and requested `tessdata` languages are explicit system prerequisites | Word/line text, page, box, Tesseract confidence, language, orientation, and OCR config. Low confidence remains visible. Handwriting is best-effort/degraded in this baseline; do not label it verified handwriting support. Missing Tesseract/language data is an actionable failure. | Synthetic two-page scanned printed claim form with checkboxes, a deliberately blurred field, and expected boxes/confidence bands | Claims intake that routes low-confidence fields to human review | OCR adapter integration and fixture e2e |
| Layout-aware / `layout` | Machine-generated PDF, PPTX, XLSX/XLSM (read-only), plus CSV/TSV through the textual boundary | `pdfplumber` for PDF text/objects/tables, `python-pptx` for slide shapes/tables, and `openpyxl` for workbook structure/cells | Page/slide/sheet, ordered region, cell/header/merged-cell relationships, formula plus cached/display-value policy, and embedded-asset declarations. Reading order/table recognition are heuristic and emit notices when ambiguous. Scanned pages route explicitly to `ocr`; embedded images route to `vision`. | Synthetic annual-report PDF, pricing PPTX, and workbook with headings, merged cells, formulas, and two sheets | Financial and supplier-pricing analysis without flattening tables | Layout adapter integration per container plus cross-family routing test |
| Vision-language / `vision` | PNG, JPEG, WebP and extracted image regions from mixed pages/slides | Pillow + PyTorch/Transformers image-to-text using configured `HuggingFaceTB/SmolVLM-256M-Instruct` revision; CPU is the reference device | Original asset/region plus prompt, model ID/revision, generated description, and generation settings. The model does not provide calibrated factual confidence, so confidence is `unavailable`; descriptions are untrusted derived evidence. Charts, diagrams, small text, and domain imagery can be wrong and are marked model-derived. Model absence never falls back to OCR or filename text. | Redistributable equipment photo, simple chart, simple process diagram, and a mixed page with expected source links (semantic assertions, not exact prose) | Maintenance triage over manuals, diagrams, and equipment photographs | Vision adapter integration on CPU; deterministic fake in unit tests |
| Time-based / `media` | WAV, FLAC, MP3, M4A/MP4 audio and short MP4/WebM video supported by the installed decoder | `faster-whisper` on CPU (`int8`) for ASR; its PyAV path decodes audio; PySceneDetect `ContentDetector` selects scene intervals/keyframes through its OpenCV extra | Transcript segments/words with `[start_ms,end_ms)`, language and raw ASR signals; scene boundaries and linked keyframe asset/time. Speaker identity/diarization is not in the baseline. Silence, overlap, uncertain language, no-scene video, and decoder limitations produce notices. No transcript-only success may hide an unprocessed video stream. | Synthetic narrated support call and a short narrated training video with hard cuts and redistributable frames | Search support calls and jump from an answer to the cited training-video moment | Media ASR and scene integration tests; cross-link e2e |

## Default resource envelope

These are defensive reference-profile defaults, not claims about library limits.
Changing them requires explicit configuration and must remain bounded.

| Family | Default ingestion envelope |
|---|---|
| Textual | 10 MiB source, 2,000,000 decoded characters, 100 MIME parts |
| OCR | 25 pages, 20 megapixels per raster, 300 DPI target, 30 seconds per page |
| Layout-aware | 100 PDF pages or slides; 20 workbook sheets and 100,000 populated cells; macros are never executed |
| Vision-language | 20 regions per image, 2,048-pixel input limit, 128-pixel inference resize, 8 generated tokens, and 60 seconds per image |
| Time-based | 250 MiB and 30 minutes, 200 transcript segments, 100 detected scenes, one keyframe per scene, 2-hour wall-clock job timeout |

Global archive expansion, nesting-depth, and decompression-ratio limits apply before
family routing. Passwords are caller-supplied through a secret-safe channel; the
system does not attempt password discovery. Temporary files are isolated, bounded,
and deleted after processing.

## Dependency and license boundary

- `text` is deliberately empty and must remain importable with core alone.
- `ocr` is the Python dependency bundle; Tesseract and its language/model data are
  separately installed system/data artifacts. `pytesseract` is an Apache-2.0 wrapper,
  while Tesseract is Apache-2.0; language data must be reviewed independently.
- `layout` contains `pdfplumber` (MIT), `python-pptx` (MIT), and `openpyxl` (MIT).
  `pypdfium2` is owned by `ocr` for PDF rasterization; its wrapper and bundled PDFium
  licenses/notices must be retained and rechecked at lock time.
- `vision` contains Pillow, PyTorch, and Transformers. Package licenses do not grant
  rights to arbitrary downloaded weights. The selected SmolVLM model card currently
  identifies Apache-2.0; configuration must pin a reviewed immutable revision.
- `media` contains `faster-whisper` and `scenedetect[opencv]`. Faster-whisper is MIT
  and documents PyAV-bundled FFmpeg libraries, so an `ffmpeg` executable is not a
  baseline requirement for transcription. PySceneDetect's splitting helpers may
  require external FFmpeg, but rag-kit only performs detection/keyframe extraction.
  Whisper code and weights are documented as MIT; pin and inventory the converted
  model artifact actually deployed.

An SBOM/license scan of the resolved lock, binary wheels, system packages, model
weights, and fixture assets is a release gate. A permissive top-level package license
does not erase native-library, model, training-data, codec/patent, or fixture terms.

## Official references

- Python text/email/HTML surfaces: <https://docs.python.org/3/library/email.parser.html>,
  <https://docs.python.org/3/library/html.parser.html>
- pytesseract outputs/prerequisite: <https://pypi.org/project/pytesseract/>
- Tesseract TSV/hOCR: <https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html>
- PDFium rendering: <https://pypdfium2.readthedocs.io/en/stable/python_api.html>
- PDF layout/table extraction: <https://github.com/jsvine/pdfplumber>
- PowerPoint text and tables: <https://python-pptx.readthedocs.io/en/stable/user/text.html>,
  <https://python-pptx.readthedocs.io/en/stable/user/table.html>
- Workbook iteration/cells: <https://openpyxl.readthedocs.io/en/stable/tutorial.html>
- Transformers image-text-to-text: <https://huggingface.co/docs/transformers/main/tasks/image_text_to_text>
- Selected vision model card: <https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct>
- Faster-whisper runtime and decoding: <https://github.com/SYSTRAN/faster-whisper>
- Whisper model license/source: <https://github.com/openai/whisper>
- Scene detection and timecodes: <https://www.scenedetect.com/docs/latest/api/scene_manager.html>
