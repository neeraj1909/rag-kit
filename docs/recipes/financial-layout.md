# Financial and pricing layout extraction

Family: `layout`

## Business use case

Search pricing workbooks and annual reports while citing the exact sheet/cell, slide
shape, table region, or PDF page box that supports a financial answer.

## Contract

`LayoutDocumentExtractor` implements `DocumentExtractor`. It retains zero-based PDF
word boxes, slide/shape or table boundaries, workbook cell/merged-range coordinates,
and explicit `continues` relationships in stable source order.

## Config schema

Use `configs/layout.toml`: declared classifier, extractor `"layout"`, and evidence
chunker. `AdapterSettings` validates maximum pages, slides, sheets, cells, archive
bytes, and compression ratio; `RuntimeLimits` supplies outer bounds.

## Registry and bootstrap

The `"layout"` extractor factory in `infrastructure/bootstrap.py` passes every layout
bound explicitly. Add a new format through a distinct adapter/factory and update
capability inspection for its dependency.

## Tests

Run layout unit/integration tests and the shared extractor contract. Fixtures must
assert exact locators, relationships, formula/displayed-value behavior, embedded-image
rejection, malformed archives, and bounds.

## Optional extra

Install `rag-kit[layout]` for PDF, PPTX, and XLSX dependencies.

## Limits

Bound file bytes, archive expansion and ratio, page/slide/sheet/cell counts, and outer
document/part/chunk counts. The adapter must fail rather than return truncated tables.

## Determinism

Fixed bytes, dependency versions, settings, and fingerprint produce stable source
order. PDF reading order remains a documented heuristic.

## Confidence and fallback

The baseline invents no layout confidence. Image-only PDF pages route explicitly to
OCR and slide images to vision; layout extraction never silently substitutes either.

## Failure modes

Encrypted, corrupt, oversized, empty, image-only, or unsupported containers fail with
typed errors. Missing cached formula values produce an explicit notice, not a value.
