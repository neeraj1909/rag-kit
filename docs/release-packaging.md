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
| `persistent` | Chroma and persistent adapter imports |
| `hosted` | OpenAI SDK and hosted adapter imports, with no request |
| `http` | Uvicorn, ASGI/server module imports, and `ragkit-http --help` |
| `reranking` | Torch, Transformers, and adapter imports |

The model-heavy checks import their SDK boundaries but deliberately do not load
weights or initialize a model.

The Phase 6 dependency audit upgraded PySceneDetect to 0.7.1 and Click beyond
8.3.3, closing PYSEC-2026-2132. The remaining Chroma advisory and the deployment
boundary that prevents its vulnerable server endpoint from being exposed are
recorded in [deployment.md](deployment.md#chroma-advisory-boundary).
Every job sets Hugging Face and Transformers offline modes, and the evidence JSON
records zero network attempts, model downloads, and provider calls. Socket
connections are denied while the probes run. This is installation evidence, not
OCR accuracy, retrieval quality, media decoding, model inference, or
provider-live evidence. Those behaviors remain owned by their integration and
opt-in live suites.

Each optional matrix job uploads
`install-<extra>-python-<version>` containing the interpreter, imported modules,
and installed direct-distribution versions.
