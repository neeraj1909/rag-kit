# Evaluation and benchmarking

The `ragkit.evaluation` package evaluates already-observed results. It does not
run extractors or models and therefore does not turn a schema test into a model
quality claim. Integration code maps stable evidence labels from the dataset to
the chunk IDs and locators produced by a configured pipeline.

## Dataset contract

`tests/fixtures/evaluation/five-family-v1.json` is the versioned reference
dataset. Each case names a business use case, an existing redistributable
fixture, relevant evidence labels, expected extraction count, and required
locator kinds. The degraded handwritten OCR and unsupported image-only layout
cases are explicitly ineligible with reasons. An ineligible metric is JSON
`null` plus a reason; it is never silently coerced to zero or omitted.

The dataset fingerprint is SHA-256 over canonical JSON. Reports retain that
fingerprint together with the configuration, corpus, component, and software
versions supplied by the caller. JSON serialization is stable and round-trips
through typed values.

## Metric definitions

- `recall_at_k`: relevant evidence labels retrieved in the first `k`, divided
  by all relevant labels.
- `reciprocal_rank`: reciprocal of the first relevant rank, or zero for a miss.
- `hit_rate`: one when at least one relevant label is in the first `k`.
- `citation_precision`: relevant cited labels divided by all cited labels.
- `citation_coverage`: relevant cited labels divided by relevant labels.
- `extraction_coverage`: observed extractions divided by expected extractions,
  capped at one.
- `locator_validity`: Jaccard overlap between required and observed locator
  kinds. Missing required kinds and unexpected observed kinds both reduce it.

Every result is emitted per case and per required family. A missing observation
is a failure. A family with no eligible result remains `ineligible`, making the
overall report `incomplete`. Cross-family thresholds cannot pass if any required
family value is ineligible. This prevents a high text score from concealing an
untested vision or media lane.

## Integration hook

Composition or CLI code can load `Dataset`, execute each case through its
profile, translate returned chunks to the dataset's stable evidence labels,
create `CaseObservation` values, and call `evaluate`. That outer integration
owns model provisioning and capability failures. This package remains usable in
the dependency-light core environment.

## Benchmark harness

The harness records individual repetitions, p50/p95 latency, throughput, a
named memory measurement scope, warmups, and
hardware/software/config/corpus/workload fingerprints.
Software metadata includes the installed `rag-kit` distribution version and a
Git revision marked `+clean` or `+dirty`; compare dirty runs only when the exact
patch is preserved alongside the report.
The clock and memory sampler are injectable for deterministic tests. Latency is reported,
not universally gated; hosts and optional models are not comparable unless the
operator deliberately supplies a host-specific policy.

Run a fixed end-to-end query and save its report:

```bash
uv run python scripts/benchmark.py \
  --config configs/offline.toml \
  --source tests/fixtures/corpus \
  --query "What is the fixture answer?" \
  --warmups 1 --repetitions 5 \
  --output reports/benchmarks/offline-text.local.json
```

Benchmark artifacts that depend on a particular machine should normally remain
local or be attached to CI. Compare quality only with fixed datasets; interpret
latency only alongside the report metadata.
The CLI benchmark's memory field is cumulative child-process high-water RSS and
includes the nested `uv` launcher and setup subprocesses; it is useful for
same-command regressions, not as isolated `ragkit` process memory.
