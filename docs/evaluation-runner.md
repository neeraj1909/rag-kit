# Phase 4 evaluation runner

`scripts/evaluate_phase4.py` is the repeatable outer evidence run for retrieval
profiles. It indexes the configured corpus, asks the real application pipeline,
and compares returned chunk IDs and locator kinds with labels committed before
the run. It does not call hosted services or download model weights.

## Fixed gold set

`tests/fixtures/evaluation/phase4-text-v1.json` contains three independently
labeled text questions. The companion `phase4-text-gold-selectors-v1.json` ties
each stable symbolic evidence label to a fixture-relative source digest and an
exact source locator. The runner maps the checkout-specific chunk ID to that
label only after the indexed provenance matches both fields. It never learns
relevance from the rank produced by the retriever it is scoring. If fixture
content or chunking semantics change, the run must fail until a reviewer
deliberately versions both files.

The thresholds are explicit and profile-local:

- text recall at k must equal `1.0`;
- text hit rate must equal `1.0`;
- citation precision and coverage must equal `1.0`;
- extraction coverage must equal `1.0`;
- text locator validity must equal `1.0`.

The indexing boundary exposes content-free chunk IDs plus exact provenance. The
runner matches those records to the fixture digest and locator in the selector,
then maps the checkout-specific chunk ID to the stable symbolic gold label. This
measures extraction independently of retrieval without exposing source text.

## Repeatable command

Use an isolated environment so another agent or checkout cannot change the
evidence run underneath you:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/ragkit-phase4-eval uv sync --group dev
RAGKIT_RUN_MODEL_INTEGRATION=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
UV_PROJECT_ENVIRONMENT=/tmp/ragkit-phase4-eval uv run python scripts/evaluate_phase4.py \
  --dataset tests/fixtures/evaluation/phase4-text-v1.json \
  --gold-selectors tests/fixtures/evaluation/phase4-text-gold-selectors-v1.json \
  --profile configs/sparse.toml \
  --profile configs/hybrid.toml \
  --family-matrix tests/fixtures/evaluation/phase4-family-matrix-v1.json \
  --output-dir reports/evaluation
```

The sparse and hybrid reports include dataset, configuration, corpus, component
fingerprints, rag-kit/Python versions, and Git build identity. Observation
artifacts retain the human-readable selections separately and report query
latency p50/p95 as explicitly non-gated information. A failed profile threshold
or an executed family failure makes the command exit non-zero.

## Five-family honesty check

The family matrix separately preflights and, when locally eligible, executes
the text, OCR, layout-aware, vision, and media profiles. Missing optional
modules or missing revision-pinned model caches produce `ineligible` with the
exact preflight reason. An unexpected execution error or a result that misses
the fixed relevant ID produces `fail`. Only an actual pipeline result matching
a fixed label produces `pass`. The vision profile uses the reviewed 64-pixel
fixture envelope and four-token output bound so its pinned CPU model executes
inside the runner's unchanged 60-second family timeout.

The committed all-extras evidence records text, OCR, layout, vision, and media
as passing their fixed extraction, retrieval, citation, and locator checks. The
queries do not contain their gold answers. Separate real-model integration tests
prove the vision description path and audio/video decoder path; these bounded
fixtures are not claims of production model accuracy.

This matrix is not a cross-family quality average. Ineligible families stay
visible, and they are never replaced with synthetic observations or claimed as
successful. Provision optional extras and pinned local models explicitly before
requesting evidence for those families.

Model-backed family execution is always opt-in, even when a reviewed revision is
already cached. Without `RAGKIT_RUN_MODEL_INTEGRATION=1`, those rows remain
explicitly ineligible; ordinary unit, integration, and coverage runs never start
model inference merely because a developer machine happens to have a cache.

Each family record includes only sanitized capability fields: extra, module,
distribution, installed version, binary status, pinned model/revision, cache
availability, and a boolean credential-presence signal. Cache paths, credential
names, and credential values are never written.

For repeatability comparisons, compare the canonical quality reports and the
family matrix byte-for-byte. Normalize `query_latency_ns` out of observation
artifacts before comparing them; latency is machine-sensitive evidence and is
never a quality gate.
