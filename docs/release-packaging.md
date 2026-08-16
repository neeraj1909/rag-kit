# Release packaging validation

Phase 6 validates the artifacts a consumer installs, not only the editable source
tree. Publication remains out of scope.

## Archive gate

```bash
uv build --no-sources
uv run python scripts/check_package.py \
  --wheel dist/rag_kit-0.1.0-py3-none-any.whl \
  --sdist dist/rag_kit-0.1.0.tar.gz \
  --extra text --extra ocr --extra layout --extra vision \
  --extra media --extra persistent --extra hosted --extra http --extra reranking
```

The checker rejects unsafe archive paths and verifies the distribution name,
Python range, all published extras, both console scripts, Markdown README
metadata, MIT license, and `ragkit/py.typed`. The source archive must also carry
the repository README, license, `pyproject.toml`, and type marker.

## Clean-install matrix

CI installs the wheel's core and every extra independently on Python 3.11 and
3.12. It also installs the source archive's core on both interpreters. These
installs resolve the compatible ranges published in the artifact; `uv.lock`
separately owns the exact development and full-test resolution.

| Install profile | Clean checks |
|---|---|
| Core wheel and source archive | `import ragkit`; optional SDKs absent; every missing-extra action names `rag-kit[extra]`; the HTTP launcher gives the same actionable guidance |
| `text` | Dependency-free textual adapter import |
| `ocr` | Pillow, PDFium, pytesseract, and OCR adapter imports |
| `layout` | openpyxl, pdfplumber, python-pptx, and layout adapter imports |
| `vision` | Pillow, Torch, torchvision, Transformers, and adapter imports |
| `media` | faster-whisper, PySceneDetect/OpenCV, and media adapter imports |
| `persistent` | Standard-library SQLite and persistent adapter imports; no third-party distribution |
| `hosted` | OpenAI SDK and hosted adapter imports, with no request |
| `http` | Uvicorn, ASGI/server module imports, and `ragkit-http --help` |
| `reranking` | Torch, Transformers, and adapter imports |

The model-heavy checks import their SDK boundaries but deliberately do not load
weights or initialize a model.

The Phase 6 dependency audit upgraded PySceneDetect to 0.7.1 and Click beyond
8.3.3, closing PYSEC-2026-2132. Phase 7 removed `chromadb` from supported
artifacts because GHSA-f4j7-r4q5-qw2c still had no patched release; persistence
now uses standard-library SQLite as recorded in [ADR 0006](decisions/0006-sqlite-persistent-store.md).
Every job sets Hugging Face and Transformers offline modes, and the evidence JSON
records zero network attempts, model downloads, and provider calls. Socket
connections are denied while the probes run. This is installation evidence, not
OCR accuracy, retrieval quality, media decoding, model inference, or
provider-live evidence. Those behaviors remain owned by their integration and
opt-in live suites.

Each optional matrix job uploads
`install-<extra>-python-<version>` containing the interpreter, imported modules,
and installed direct-distribution versions.

## CI action runtime and pins

Every external action is pinned to a reviewed immutable commit whose `action.yml`
declares `node24`:

| Action | Reviewed release | Pinned commit |
|---|---|---|
| [`actions/checkout`](https://github.com/actions/checkout/releases/tag/v7.0.1) | 7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| [`actions/upload-artifact`](https://github.com/actions/upload-artifact/releases/tag/v7.0.1) | 7.0.1 | `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` |
| [`actions/download-artifact`](https://github.com/actions/download-artifact/releases/tag/v8.0.1) | 8.0.1 | `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c` |
| [`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv/releases/tag/v10.0.1) | 10.0.1 | `20cfd1bf945f4377ade1205e4dbc17946fc9a30d` |

`tests/contract/test_maintenance_caveats.py` rejects a tag, unknown action, stale
revision, or mismatched release comment. Upgrades require repeating official
release and `action.yml` review rather than changing only the comment.
