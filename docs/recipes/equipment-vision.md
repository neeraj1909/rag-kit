# Equipment-image retrieval

Family: `vision`

## Business use case

Create retrieval leads from equipment photographs and diagrams so a maintainer can
open the cited image region; the description alone is never maintenance evidence.

## Contract

`VisionDocumentExtractor` implements `DocumentExtractor`. It retains the original
asset/normalized region plus prompt, model fingerprint, settings, and a model-derived
description linked to that evidence.

## Config schema

Use `configs/vision.toml`: declared classifier, extractor `"vision"`, evidence
chunking. `AdapterSettings` pins model ID/revision and bounds image dimensions/pixels,
regions, new tokens, inference edge, and timeout. `mixed_image` is a separate explicit
selection when both OCR and vision are required.

## Registry and bootstrap

The `"vision"` factory injects `LocalSmolVLMBackend`; `"mixed_image"` composes OCR and
vision and fails if either is absent. Add a backend by explicit selection/factory and
capability declaration, never by an untyped model name switch.

## Tests

Run vision fake/unit contracts, then the opt-in CPU integration with the exact cached
revision. Assert provenance and derived-evidence notices semantically, not exact model
prose.

## Optional extra

Install `rag-kit[vision]` and provision `HuggingFaceTB/SmolVLM-256M-Instruct` at
`7e3e67edbbed1bf9888184d9df282b700a323964` before offline execution.

```bash
uv run --with huggingface-hub hf download \
  HuggingFaceTB/SmolVLM-256M-Instruct \
  --revision 7e3e67edbbed1bf9888184d9df282b700a323964
```

## Limits

Bound asset bytes, decoded pixels/dimensions, regions, output tokens, inference resize,
timeout, and outer document/part/chunk counts. The release fixture uses a 128-pixel
edge and eight output tokens to bound the smoke run.

## Determinism

The pinned backend runs local-files-only on CPU in evaluation/inference mode. Exact
generated prose is not promised across dependency or hardware changes; provenance and
ordering are deterministic under the fingerprinted setup.

## Confidence and fallback

Model confidence is unavailable. Descriptions carry `model_derived` and
`untrusted_description`; missing capability never falls back to filename or empty text.

## Failure modes

Missing cache/dependency, unsupported/corrupt image, pixel/dimension/region/token/time
limit, or blank generation fails explicitly. Human review and task-specific evaluation
are required for safety-related decisions.
