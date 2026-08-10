"""Stable evaluation observations and report serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

from .schema import Family

MetricName = Literal[
    "recall_at_k",
    "reciprocal_rank",
    "hit_rate",
    "citation_precision",
    "citation_coverage",
    "extraction_coverage",
    "locator_validity",
]
Status = Literal["pass", "fail", "ineligible", "incomplete"]
METRIC_NAMES: tuple[MetricName, ...] = (
    "recall_at_k",
    "reciprocal_rank",
    "hit_rate",
    "citation_precision",
    "citation_coverage",
    "extraction_coverage",
    "locator_validity",
)
_STATUSES = {"pass", "fail", "ineligible", "incomplete"}
_FAMILIES = {"text", "ocr", "layout", "vision", "media"}


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has unexpected fields")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class CaseObservation:
    """Observed pipeline evidence; ``None`` citations means generation was not run."""

    case_id: str
    retrieved_evidence_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...] | None
    extracted_count: int | None
    locator_kinds: tuple[str, ...]
    ineligible_reason: str | None = None
    extraction_ineligible_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("observation case ID must not be empty")
        if self.extracted_count is not None and (
            type(self.extracted_count) is not int or self.extracted_count < 0
        ):
            raise ValueError("extracted count must be non-negative")
        if (self.extracted_count is None) == (self.extraction_ineligible_reason is None):
            raise ValueError(
                "extraction observation must have exactly one of count or ineligible reason"
            )
        if len(set(self.retrieved_evidence_ids)) != len(self.retrieved_evidence_ids):
            raise ValueError("retrieved evidence IDs must be unique")
        if self.cited_evidence_ids is not None and len(set(self.cited_evidence_ids)) != len(
            self.cited_evidence_ids
        ):
            raise ValueError("cited evidence IDs must be unique")
        if self.cited_evidence_ids is not None and not set(self.cited_evidence_ids) <= set(
            self.retrieved_evidence_ids
        ):
            raise ValueError("citations must refer to retrieved evidence")

    @classmethod
    def ineligible(cls, case_id: str, reason: str) -> CaseObservation:
        if not reason:
            raise ValueError("ineligible observation requires a reason")
        return cls(case_id, (), None, None, (), reason, reason)


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    config_fingerprint: str
    corpus_fingerprint: str
    components: dict[str, str]
    software: dict[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config_fingerprint, str)
            or not self.config_fingerprint
            or not isinstance(self.corpus_fingerprint, str)
            or not self.corpus_fingerprint
        ):
            raise ValueError("configuration and corpus fingerprints are required")
        if not self.components or not self.software:
            raise ValueError("component and software provenance are required")
        if not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for values in (self.components, self.software)
            for key, value in values.items()
        ):
            raise ValueError("provenance maps require non-empty string keys and values")
        if not self.components or not all(self.components) or not all(self.components.values()):
            raise ValueError("component versions are required")
        if not self.software or not all(self.software) or not all(self.software.values()):
            raise ValueError("software versions are required")


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    ineligible_reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or not 0 <= self.value <= 1
        ):
            raise ValueError("evaluation metric must be finite and in [0, 1]")
        if (self.value is None) == (self.ineligible_reason is None):
            raise ValueError("metric must have exactly one of value or ineligible reason")


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    family: Family
    status: Status
    metrics: dict[str, MetricValue]
    failure: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id or self.family not in _FAMILIES or self.status not in _STATUSES:
            raise ValueError("case result identity, family, and status must be valid")
        if set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("case result must contain the complete metric set")
        if (self.status == "fail") != (self.failure is not None):
            raise ValueError("only failed case results require a failure reason")


@dataclass(frozen=True, slots=True)
class FamilyResult:
    family: Family
    status: Status
    case_ids: tuple[str, ...]
    metrics: dict[str, MetricValue]
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES or self.status not in _STATUSES:
            raise ValueError("family result family and status must be valid")
        if not self.case_ids or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("family result case IDs must be non-empty and unique")
        if set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("family result must contain the complete metric set")
        if (self.status == "fail") != bool(self.failures):
            raise ValueError("only failed family results require failure reasons")


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    metric: MetricName
    minimum: float
    family: Family | None = None

    def __post_init__(self) -> None:
        if self.metric not in METRIC_NAMES or (
            self.family is not None and self.family not in _FAMILIES
        ):
            raise ValueError("threshold metric and family must be valid")
        if (
            isinstance(self.minimum, bool)
            or not isinstance(self.minimum, (int, float))
            or not math.isfinite(self.minimum)
            or not 0 <= self.minimum <= 1
        ):
            raise ValueError("threshold minimum must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    metric: MetricName
    minimum: float
    family: Family | None
    actual: float | None
    passed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.metric not in METRIC_NAMES or (
            self.family is not None and self.family not in _FAMILIES
        ):
            raise ValueError("threshold result metric and family must be valid")
        if (
            isinstance(self.minimum, bool)
            or not isinstance(self.minimum, (int, float))
            or not math.isfinite(self.minimum)
            or not 0 <= self.minimum <= 1
        ):
            raise ValueError("threshold result minimum must be finite and in [0, 1]")
        if self.actual is not None and (
            isinstance(self.actual, bool)
            or not isinstance(self.actual, (int, float))
            or not math.isfinite(self.actual)
            or not 0 <= self.actual <= 1
        ):
            raise ValueError("threshold result actual must be finite and in [0, 1]")
        if type(self.passed) is not bool or self.passed != (
            self.actual is not None and self.actual >= self.minimum
        ):
            raise ValueError("threshold pass flag must agree with the observed value")


@dataclass(frozen=True, slots=True)
class Report:
    schema_version: str
    dataset_id: str
    dataset_fingerprint: str
    top_k: int
    provenance: EvaluationProvenance
    case_results: tuple[CaseResult, ...]
    family_results: dict[str, FamilyResult]
    threshold_results: tuple[ThresholdResult, ...]
    overall_status: Status
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "ragkit-evaluation-report/v1":
            raise ValueError("unsupported evaluation report schema")
        if (
            not isinstance(self.dataset_id, str)
            or not self.dataset_id
            or not isinstance(self.dataset_fingerprint, str)
            or not self.dataset_fingerprint
            or type(self.top_k) is not int
            or self.top_k <= 0
        ):
            raise ValueError("report dataset identity and top_k must be valid")
        case_ids = tuple(item.case_id for item in self.case_results)
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("report case IDs must be non-empty and unique")
        if set(self.family_results) != {item.family for item in self.case_results}:
            raise ValueError("family results must align with case families")
        for family, result in self.family_results.items():
            expected_case_ids = tuple(
                item.case_id for item in self.case_results if item.family == family
            )
            if result.family != family or result.case_ids != expected_case_ids:
                raise ValueError("family result identity and case partition must align")
            case_statuses = tuple(
                item.status for item in self.case_results if item.family == family
            )
            if any(status == "fail" for status in case_statuses):
                expected_family_status: Status = "fail"
            elif all(status == "pass" for status in case_statuses):
                expected_family_status = "pass"
            elif all(status == "ineligible" for status in case_statuses):
                expected_family_status = "ineligible"
            else:
                expected_family_status = "incomplete"
            if result.status != expected_family_status:
                raise ValueError("family status must agree with its case results")
        if self.overall_status not in _STATUSES:
            raise ValueError("report overall status must be valid")
        has_explicit_failure = any(
            result.status == "fail" for result in self.family_results.values()
        ) or any(
            not result.passed and result.actual is not None for result in self.threshold_results
        )
        if has_explicit_failure:
            expected_overall: Status = "fail"
        elif all(result.status == "pass" for result in self.family_results.values()) and all(
            result.passed for result in self.threshold_results
        ):
            expected_overall = "pass"
        else:
            expected_overall = "incomplete"
        if self.overall_status != expected_overall:
            raise ValueError("overall status must agree with family and threshold results")
        if (self.overall_status == "fail") != bool(self.failures):
            raise ValueError("only failed reports require failure reasons")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Report:
        value = _exact_dict(
            json.loads(payload),
            {
                "schema_version",
                "dataset_id",
                "dataset_fingerprint",
                "top_k",
                "provenance",
                "case_results",
                "family_results",
                "threshold_results",
                "overall_status",
                "failures",
            },
            "evaluation report",
        )
        raw_provenance = _exact_dict(
            value["provenance"],
            {"config_fingerprint", "corpus_fingerprint", "components", "software"},
            "evaluation provenance",
        )
        provenance = EvaluationProvenance(
            cast(str, raw_provenance["config_fingerprint"]),
            cast(str, raw_provenance["corpus_fingerprint"]),
            cast(dict[str, str], raw_provenance["components"]),
            cast(dict[str, str], raw_provenance["software"]),
        )

        def metric_map(raw: object) -> dict[str, MetricValue]:
            if not isinstance(raw, dict) or set(raw) != set(METRIC_NAMES):
                raise ValueError("metric map must contain the complete metric set")
            result: dict[str, MetricValue] = {}
            for name, value_item in raw.items():
                metric = _exact_dict(value_item, {"value", "ineligible_reason"}, "metric value")
                result[cast(str, name)] = MetricValue(
                    cast(float | None, metric["value"]),
                    cast(str | None, metric["ineligible_reason"]),
                )
            return result

        raw_cases = value["case_results"]
        if not isinstance(raw_cases, list):
            raise ValueError("case results must be a list")
        cases: list[CaseResult] = []
        for value_item in raw_cases:
            row = _exact_dict(
                value_item,
                {"case_id", "family", "status", "metrics", "failure"},
                "case result",
            )
            cases.append(
                CaseResult(
                    cast(str, row["case_id"]),
                    cast(Family, row["family"]),
                    cast(Status, row["status"]),
                    metric_map(row["metrics"]),
                    cast(str | None, row["failure"]),
                )
            )
        raw_families = value["family_results"]
        if not isinstance(raw_families, dict):
            raise ValueError("family results must be an object")
        families: dict[str, FamilyResult] = {}
        for name, value_item in raw_families.items():
            row = _exact_dict(
                value_item,
                {"family", "status", "case_ids", "metrics", "failures"},
                "family result",
            )
            families[cast(str, name)] = FamilyResult(
                cast(Family, row["family"]),
                cast(Status, row["status"]),
                tuple(cast(list[str], row["case_ids"])),
                metric_map(row["metrics"]),
                tuple(cast(list[str], row["failures"])),
            )
        raw_thresholds = value["threshold_results"]
        if not isinstance(raw_thresholds, list):
            raise ValueError("threshold results must be a list")
        thresholds: list[ThresholdResult] = []
        for value_item in raw_thresholds:
            row = _exact_dict(
                value_item,
                {"metric", "minimum", "family", "actual", "passed", "reason"},
                "threshold result",
            )
            thresholds.append(
                ThresholdResult(
                    cast(MetricName, row["metric"]),
                    cast(float, row["minimum"]),
                    cast(Family | None, row["family"]),
                    cast(float | None, row["actual"]),
                    cast(bool, row["passed"]),
                    cast(str | None, row["reason"]),
                )
            )
        return cls(
            schema_version=cast(str, value["schema_version"]),
            dataset_id=cast(str, value["dataset_id"]),
            dataset_fingerprint=cast(str, value["dataset_fingerprint"]),
            top_k=cast(int, value["top_k"]),
            provenance=provenance,
            case_results=tuple(cases),
            family_results=families,
            threshold_results=tuple(thresholds),
            overall_status=cast(Status, value["overall_status"]),
            failures=tuple(cast(list[str], value["failures"])),
        )
