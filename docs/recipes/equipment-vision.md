# Equipment-image retrieval

Use the vision adapter when maintenance evidence exists in photographs, charts, or
diagrams rather than native text. Provision a reviewed model artifact outside the
request path, pin its immutable revision, then inject the resulting backend into
`VisionDocumentExtractor`.

The baseline pins `HuggingFaceTB/SmolVLM-256M-Instruct` at revision
`7e3e67edbbed1bf9888184d9df282b700a323964`. Provisioning is an explicit,
networked operator action, never an ingestion side effect:

```bash
uv run --with huggingface-hub hf download \
  HuggingFaceTB/SmolVLM-256M-Instruct \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964
```

After license and artifact review, `LocalSmolVLMBackend` loads that cache with
`local_files_only=True`, `trust_remote_code=False`, CPU placement, `eval()`, and
PyTorch inference mode.

The extractor records the prompt and model fingerprint, retains the original asset
and normalized image region, and leaves confidence unavailable. Generated
descriptions are explicitly marked `model_derived` and `untrusted_description`:
they are retrieval leads, not proof that a valve is damaged or a gauge reading is
correct. Always show the cited image region to the maintainer before action.

There is no filename or empty-text fallback. A missing model, unsupported format,
blank generation, or absent capability fails the operation. The explicit
`mixed_image` profile runs both OCR and vision on the same raster page and fails if
either capability is absent.

The generated equipment PNG is exercised in an opt-in real CPU run. At a
128-pixel inference edge and eight output tokens, the profile keeps CPU work
bounded. Release evidence records measured runtime separately; this proves
execution, not model accuracy.
Do not use this baseline
for safety-critical decisions without a task-specific quality evaluation and human
review.

The separate text embedder defaults to
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; it is not the SmolVLM checkpoint.
Provision it explicitly with the same `hf download <model> --revision <sha>`
pattern before selecting the local-ML profile.
