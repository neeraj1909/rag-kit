# Claims OCR with human review

Family: `ocr`

## Business use case

Extract printed claim forms so an adjuster can search policy numbers and open the
cited page region for review before making a decision.

## Contract

`OcrDocumentExtractor` implements `DocumentExtractor`. Every word retains a zero-based
page and normalized box plus the engine's raw confidence; output stays in deterministic
page/reading order.

## Config schema

Use `configs/ocr.toml`: select declared classification, extractor `"ocr"`, evidence
chunking, and `ocr_language`. `AdapterSettings` owns page, pixel, and timeout values;
outer `RuntimeLimits` bounds assets, parts, and chunks. The extractor's review
threshold is fingerprinted but is not profile-configurable in the baseline.

## Registry and bootstrap

The `"ocr"` extractor factory in `infrastructure/bootstrap.py` passes validated
settings to `OcrDocumentExtractor`. A different engine needs a new selection and
factory plus capability inspection; do not switch engines implicitly.

## Tests

Run OCR unit and integration tests and the shared extractor contract. Cover printed,
low-confidence, empty, corrupt, oversized, timeout, and missing-language-data cases.

## Optional extra

Install `rag-kit[ocr]`. Tesseract and requested language data are separate system
prerequisites; the Python extra does not install them.

## Limits

Enforce asset bytes, page count, decoded pixels, documents/parts/chunks, and subprocess
timeout before returning a partial result.

## Determinism

Fixed bytes, Tesseract/language-data version, configuration, and fingerprint produce
stable ordering. Engine upgrades require a new fingerprint and evaluation.

## Confidence and fallback

Raw OCR confidence is extraction evidence, not correctness. Values below the review
threshold carry `low_ocr_confidence`. Handwriting and form modes are explicitly
best-effort; there is no filename or invented field-value fallback.

## Failure modes

Empty/corrupt input, missing Tesseract or language data, unsupported format, timeout,
and page/pixel limits fail actionably. Checkbox state and form relationships are not
claimed by the printed-text baseline.
