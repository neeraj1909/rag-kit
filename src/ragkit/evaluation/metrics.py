"""Deterministic, eligibility-aware retrieval and evidence metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from .report import (
    CaseObservation,
    CaseResult,
    EvaluationProvenance,
    FamilyResult,
    MetricThreshold,
    MetricValue,
    Report,
    Status,
    ThresholdResult,
)
from .schema import Dataset, EvaluationCase, Family

METRICS = (
    "recall_at_k",
    "reciprocal_rank",
    "hit_rate",
    "citation_precision",
    "citation_coverage",
    "extraction_coverage",
    "locator_validity",
)


def _not_eligible(reason: str) -> dict[str, MetricValue]:
    return {name: MetricValue(None, reason) for name in METRICS}


def _case_metrics(
    case: EvaluationCase, observed: CaseObservation, top_k: int
) -> dict[str, MetricValue]:
    if case.eligibility == "ineligible":
        return _not_eligible(case.ineligible_reason or "case declared ineligible")
    if observed.ineligible_reason:
        return _not_eligible(observed.ineligible_reason)
    relevant = set(case.relevant_evidence_ids)
    retrieved = observed.retrieved_evidence_ids[:top_k]
    retrieved_relevant = relevant.intersection(retrieved)
    first_rank = next((rank for rank, item in enumerate(retrieved, 1) if item in relevant), None)
    citations = observed.cited_evidence_ids
    if citations is None:
        citation_precision = MetricValue(None, "generation/citations were not observed")
        citation_coverage = MetricValue(None, "generation/citations were not observed")
    elif not citations:
        citation_precision = MetricValue(None, "no citations were emitted")
        citation_coverage = MetricValue(0.0)
    else:
        citation_relevant = relevant.intersection(citations)
        citation_precision = MetricValue(len(citation_relevant) / len(citations))
        citation_coverage = MetricValue(len(citation_relevant) / len(relevant))
    required_locators = set(case.required_locator_kinds)
    observed_locators = set(observed.locator_kinds)
    locator_union = required_locators | observed_locators
    locator = (
        MetricValue(len(required_locators & observed_locators) / len(locator_union))
        if observed_locators
        else MetricValue(None, "no source locators were observed")
    )
    extraction = (
        MetricValue(min(observed.extracted_count / case.expected_extractions, 1.0))
        if observed.extracted_count is not None
        else MetricValue(
            None,
            observed.extraction_ineligible_reason or "extraction was not observed",
        )
    )
    return {
        "recall_at_k": MetricValue(len(retrieved_relevant) / len(relevant)),
        "reciprocal_rank": MetricValue(0.0 if first_rank is None else 1 / first_rank),
        "hit_rate": MetricValue(float(bool(retrieved_relevant))),
        "citation_precision": citation_precision,
        "citation_coverage": citation_coverage,
        "extraction_coverage": extraction,
        "locator_validity": locator,
    }


def _family_result(family: Family, results: list[CaseResult]) -> FamilyResult:
    metrics: dict[str, MetricValue] = {}
    for name in METRICS:
        values = [
            cast(float, result.metrics[name].value)
            for result in results
            if result.metrics[name].value is not None
        ]
        if values:
            metrics[name] = MetricValue(sum(values) / len(values))
        else:
            reasons = sorted(
                {result.metrics[name].ineligible_reason or "ineligible" for result in results}
            )
            metrics[name] = MetricValue(None, "; ".join(reasons))
    failures = tuple(result.failure for result in results if result.failure is not None)
    status: Status
    if failures or any(result.status == "fail" for result in results):
        status = "fail"
    elif all(result.status == "pass" for result in results):
        status = "pass"
    elif all(result.status == "ineligible" for result in results):
        status = "ineligible"
    else:
        status = "incomplete"
    return FamilyResult(
        family, status, tuple(result.case_id for result in results), metrics, failures
    )


def evaluate(
    dataset: Dataset,
    observations: tuple[CaseObservation, ...],
    *,
    top_k: int,
    provenance: EvaluationProvenance,
    thresholds: tuple[MetricThreshold, ...] = (),
    required_families: tuple[Family, ...] = ("text", "ocr", "layout", "vision", "media"),
) -> Report:
    """Evaluate exact observations and refuse reports that omit a required family."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    dataset_families = {case.family for case in dataset.cases}
    missing_families = set(required_families) - dataset_families
    if missing_families:
        raise ValueError(f"required families absent from dataset: {sorted(missing_families)}")
    invalid_threshold_families = {
        threshold.family
        for threshold in thresholds
        if threshold.family is not None and threshold.family not in required_families
    }
    if invalid_threshold_families:
        raise ValueError(
            f"threshold families are not required: {sorted(invalid_threshold_families)}"
        )
    observed_by_id = {item.case_id: item for item in observations}
    if len(observed_by_id) != len(observations):
        raise ValueError("observation case IDs must be unique")
    unknown = set(observed_by_id) - {case.case_id for case in dataset.cases}
    if unknown:
        raise ValueError(f"observations reference unknown cases: {sorted(unknown)}")

    case_results: list[CaseResult] = []
    grouped: dict[Family, list[CaseResult]] = defaultdict(list)
    failures: list[str] = []
    for case in dataset.cases:
        observed = observed_by_id.get(case.case_id)
        if observed is None:
            missing_reason = "case was not observed"
            result = CaseResult(
                case.case_id,
                case.family,
                "fail",
                _not_eligible(missing_reason),
                missing_reason,
            )
            failures.append(f"{case.case_id}: {missing_reason}")
        else:
            metrics = _case_metrics(case, observed, top_k)
            status: Status
            if all(item.value is None for item in metrics.values()):
                status = "ineligible"
            elif any(item.value is None for item in metrics.values()):
                status = "incomplete"
            else:
                status = "pass"
            result = CaseResult(case.case_id, case.family, status, metrics)
        case_results.append(result)
        grouped[case.family].append(result)

    family_results: dict[str, FamilyResult] = {
        family: _family_result(family, grouped[family]) for family in required_families
    }
    threshold_results: list[ThresholdResult] = []
    for threshold in thresholds:
        threshold_reason: str | None
        if threshold.family is None:
            values = [family.metrics[threshold.metric].value for family in family_results.values()]
            if any(value is None for value in values):
                actual = None
                threshold_reason = "one or more required families are ineligible"
            else:
                actual = sum(value for value in values if value is not None) / len(values)
                threshold_reason = None
        else:
            actual = family_results[threshold.family].metrics[threshold.metric].value
            threshold_reason = None if actual is not None else "family metric is ineligible"
        passed = actual is not None and actual >= threshold.minimum
        threshold_results.append(
            ThresholdResult(
                threshold.metric,
                threshold.minimum,
                threshold.family,
                actual,
                passed,
                threshold_reason,
            )
        )
        if not passed and actual is not None:
            failures.append(
                "threshold failed: "
                f"{threshold.family or 'all'} {threshold.metric} >= {threshold.minimum}"
            )

    all_families_pass = all(result.status == "pass" for result in family_results.values())
    all_thresholds_pass = all(result.passed for result in threshold_results)
    explicit_threshold_failure = any(
        not result.passed and result.actual is not None for result in threshold_results
    )
    if failures or any(result.status == "fail" for result in family_results.values()):
        overall: Status = "fail"
    elif explicit_threshold_failure:
        overall = "fail"
    elif all_families_pass and all_thresholds_pass:
        overall = "pass"
    else:
        overall = "incomplete"
    return Report(
        "ragkit-evaluation-report/v1",
        dataset.dataset_id,
        dataset.fingerprint,
        top_k,
        provenance,
        tuple(case_results),
        family_results,
        tuple(threshold_results),
        overall,
        tuple(failures),
    )
