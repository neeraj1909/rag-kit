# Evaluation report artifacts

Committed reports must be deterministic, name their dataset/config/corpus and
component provenance, and preserve ineligible cases and threshold failures.
Machine-sensitive benchmark outputs belong in `reports/benchmarks/` or CI
artifacts and must retain their hardware and software metadata.

`five-family-report-contract-v1.json` is a reporting-contract artifact, not a
model-quality result. Its observations are deliberately marked unexecuted so it
demonstrates that every family remains visible and the overall status cannot
pass without evidence.

`requirements-evidence-v1.json` maps F-13 through F-18 and NFR-10 to exact
artifacts and tests. The all-extras Phase 6 run promotes the five family paths
only because fixed gold, real local-model integrations, and negative/degraded
contracts now pass at their named boundaries.
