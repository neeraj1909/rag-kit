from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ragkit.evaluation import (
    CaseObservation,
    Dataset,
    EvaluationCase,
    EvaluationProvenance,
    MetricThreshold,
    Report,
    evaluate,
)

FIXTURE = Path("tests/fixtures/evaluation/five-family-v1.json")
REPORT_FIXTURE = Path("reports/evaluation/five-family-report-contract-v1.json")


@pytest.mark.unit
def test_five_family_dataset_has_stable_round_trip_and_fingerprint() -> None:
    dataset = Dataset.load(FIXTURE)

    assert dataset.schema_version == "ragkit-evaluation-dataset/v1"
    assert {case.family for case in dataset.cases} == {
        "text",
        "ocr",
        "layout",
        "vision",
        "media",
    }
    assert Dataset.from_json(dataset.to_json()) == dataset
    assert Dataset.from_json(dataset.to_json()).fingerprint == dataset.fingerprint
    assert all(Path(case.source_uri).is_file() for case in dataset.cases)


@pytest.mark.unit
def test_committed_report_contract_covers_every_case_without_quality_claims() -> None:
    dataset = Dataset.load(FIXTURE)
    report = Report.from_json(REPORT_FIXTURE.read_text(encoding="utf-8"))

    assert report.dataset_fingerprint == dataset.fingerprint
    assert {result.case_id for result in report.case_results} == {
        case.case_id for case in dataset.cases
    }
    assert set(report.family_results) == {"text", "ocr", "layout", "vision", "media"}
    assert report.overall_status == "incomplete"
    assert all(result.status == "ineligible" for result in report.family_results.values())


@pytest.mark.unit
def test_dataset_loader_rejects_wrong_runtime_types() -> None:
    payload = (
        Dataset.load(FIXTURE)
        .to_json()
        .replace('"expected_extractions":1', '"expected_extractions":true', 1)
    )

    with pytest.raises(ValueError, match="expected extractions must be an integer"):
        Dataset.from_json(payload)


@pytest.mark.unit
def test_metrics_are_hand_calculated_and_segmented_without_hiding_family_failures() -> None:
    dataset = Dataset(
        schema_version="ragkit-evaluation-dataset/v1",
        dataset_id="hand-calculated",
        cases=(
            EvaluationCase(
                case_id="text-ok",
                family="text",
                business_use_case="policy lookup",
                source_uri="fixture.txt",
                query="q",
                relevant_evidence_ids=("a", "b"),
                expected_extractions=2,
                required_locator_kinds=("text_span",),
            ),
            EvaluationCase(
                case_id="ocr-unsupported",
                family="ocr",
                business_use_case="claim intake",
                source_uri="scan.png",
                query="q",
                relevant_evidence_ids=("c",),
                expected_extractions=1,
                required_locator_kinds=("box",),
                eligibility="ineligible",
                ineligible_reason="OCR executable unavailable",
            ),
        ),
    )
    observations = (
        CaseObservation(
            case_id="text-ok",
            retrieved_evidence_ids=("x", "b", "a"),
            cited_evidence_ids=("b", "x"),
            extracted_count=1,
            locator_kinds=("text_span", "page"),
        ),
        CaseObservation.ineligible("ocr-unsupported", "OCR executable unavailable"),
    )

    report = evaluate(
        dataset,
        observations,
        top_k=2,
        provenance=EvaluationProvenance(
            config_fingerprint="cfg",
            corpus_fingerprint="corpus",
            components={"retriever": "hashing:v1"},
            software={"rag-kit": "0.1.0"},
        ),
        thresholds=(MetricThreshold("recall_at_k", 0.5, "text"),),
        required_families=("text", "ocr"),
    )

    text = report.case_results[0]
    assert text.metrics["recall_at_k"].value == 0.5
    assert text.metrics["reciprocal_rank"].value == 0.5
    assert text.metrics["hit_rate"].value == 1.0
    assert text.metrics["citation_precision"].value == 0.5
    assert text.metrics["citation_coverage"].value == 0.5
    assert text.metrics["extraction_coverage"].value == 0.5
    assert text.metrics["locator_validity"].value == 0.5
    assert report.family_results["ocr"].status == "ineligible"
    assert report.overall_status == "incomplete"
    assert report.threshold_results[0].passed is True
    assert Report.from_json(report.to_json()) == report
    assert json.loads(report.to_json())["dataset_fingerprint"] == dataset.fingerprint


@pytest.mark.unit
def test_empty_citations_are_explicitly_ineligible_not_zero() -> None:
    case = EvaluationCase(
        case_id="no-generation",
        family="vision",
        business_use_case="equipment triage",
        source_uri="image.png",
        query="q",
        relevant_evidence_ids=("evidence",),
        expected_extractions=1,
        required_locator_kinds=("box",),
    )
    report = evaluate(
        Dataset("ragkit-evaluation-dataset/v1", "one", (case,)),
        (
            CaseObservation(
                case_id="no-generation",
                retrieved_evidence_ids=("evidence",),
                cited_evidence_ids=None,
                extracted_count=1,
                locator_kinds=("box",),
            ),
        ),
        top_k=1,
        provenance=EvaluationProvenance("cfg", "corpus", {"evaluator": "v1"}, {"rag-kit": "0.1.0"}),
        required_families=("vision",),
    )

    metric = report.case_results[0].metrics["citation_precision"]
    assert metric.value is None
    assert metric.ineligible_reason == "generation/citations were not observed"
    assert report.case_results[0].status == "incomplete"
    assert report.overall_status == "incomplete"


@pytest.mark.unit
def test_missing_required_family_is_a_report_failure() -> None:
    case = EvaluationCase("text", "text", "search", "a.txt", "q", ("a",), 1, ("text_span",))
    with pytest.raises(ValueError, match="required families absent"):
        evaluate(
            Dataset("ragkit-evaluation-dataset/v1", "missing", (case,)),
            (CaseObservation("text", ("a",), ("a",), 1, ("text_span",)),),
            top_k=1,
            provenance=EvaluationProvenance(
                "cfg", "corpus", {"evaluator": "v1"}, {"rag-kit": "0.1.0"}
            ),
            required_families=("text", "ocr"),
        )


@pytest.mark.unit
def test_ineligible_threshold_stays_incomplete_and_unknown_family_is_rejected() -> None:
    case = EvaluationCase(
        "vision",
        "vision",
        "inspection",
        "image.png",
        "q",
        ("a",),
        1,
        ("box",),
        eligibility="ineligible",
        ineligible_reason="model unavailable",
    )
    dataset = Dataset("ragkit-evaluation-dataset/v1", "ineligible-threshold", (case,))
    provenance = EvaluationProvenance("cfg", "corpus", {"retriever": "cmp"}, {"rag-kit": "0.1.0"})

    report = evaluate(
        dataset,
        (CaseObservation.ineligible("vision", "model unavailable"),),
        top_k=1,
        provenance=provenance,
        thresholds=(MetricThreshold("recall_at_k", 0.5, "vision"),),
        required_families=("vision",),
    )
    assert report.overall_status == "incomplete"
    assert report.failures == ()

    with pytest.raises(ValueError, match="threshold families"):
        evaluate(
            dataset,
            (CaseObservation.ineligible("vision", "model unavailable"),),
            top_k=1,
            provenance=provenance,
            thresholds=(MetricThreshold("recall_at_k", 0.5, "ocr"),),
            required_families=("vision",),
        )


@pytest.mark.unit
def test_missing_observation_is_visible_as_family_failure() -> None:
    case = EvaluationCase("text", "text", "search", "a.txt", "q", ("a",), 1, ("text_span",))
    report = evaluate(
        Dataset("ragkit-evaluation-dataset/v1", "unobserved", (case,)),
        (),
        top_k=1,
        provenance=EvaluationProvenance("cfg", "corpus", {"evaluator": "v1"}, {"rag-kit": "0.1.0"}),
        required_families=("text",),
    )

    assert report.family_results["text"].status == "fail"
    assert report.failures == ("text: case was not observed",)
    assert report.overall_status == "fail"


@pytest.mark.unit
def test_locator_validity_requires_every_declared_kind_and_rejects_extras() -> None:
    case = EvaluationCase(
        "mixed-locators",
        "layout",
        "table evidence",
        "table.xlsx",
        "q",
        ("cell",),
        1,
        ("cell", "page"),
    )
    report = evaluate(
        Dataset("ragkit-evaluation-dataset/v1", "locators", (case,)),
        (CaseObservation("mixed-locators", ("cell",), ("cell",), 1, ("cell", "box")),),
        top_k=1,
        provenance=EvaluationProvenance("cfg", "corpus", {"evaluator": "v1"}, {"rag-kit": "0.1.0"}),
        required_families=("layout",),
    )

    assert report.case_results[0].metrics["locator_validity"].value == pytest.approx(1 / 3)


@pytest.mark.unit
def test_observation_rejects_a_citation_that_was_not_retrieved() -> None:
    with pytest.raises(ValueError, match="citations must refer to retrieved evidence"):
        CaseObservation("invented", ("retrieved",), ("not-retrieved",), 1, ("text_span",))


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: {**value, "schema_version": "future/v9"},
        lambda value: {**value, "unexpected": True},
        lambda value: {
            **value,
            "case_results": [{**value["case_results"][0], "status": "excellent"}],
        },
        lambda value: {
            **value,
            "case_results": [
                {
                    **value["case_results"][0],
                    "metrics": {
                        key: item
                        for key, item in value["case_results"][0]["metrics"].items()
                        if key != "locator_validity"
                    },
                }
            ],
        },
        lambda value: {
            **value,
            "family_results": {"text": {**value["family_results"]["text"], "family": "ocr"}},
        },
        lambda value: {
            **value,
            "family_results": {
                "text": {**value["family_results"]["text"], "case_ids": ["made-up"]}
            },
        },
        lambda value: {**value, "overall_status": "ineligible", "failures": []},
        lambda value: {**value, "top_k": True},
        lambda value: {**value, "failures": ["fabricated"]},
        lambda value: {
            **value,
            "family_results": {
                "text": {**value["family_results"]["text"], "failures": ["fabricated"]}
            },
        },
        lambda value: {
            **value,
            "case_results": [
                {
                    **value["case_results"][0],
                    "metrics": {
                        **value["case_results"][0]["metrics"],
                        "recall_at_k": {"value": True, "ineligible_reason": None},
                    },
                }
            ],
        },
        lambda value: {
            **value,
            "threshold_results": [{**value["threshold_results"][0], "passed": 1}],
        },
    ),
)
def test_report_loader_rejects_untyped_or_future_shapes(
    mutation: Callable[[dict[str, object]], dict[str, object]],
) -> None:
    case = EvaluationCase("text", "text", "search", "a.txt", "q", ("a",), 1, ("text_span",))
    report = evaluate(
        Dataset("ragkit-evaluation-dataset/v1", "strict-report", (case,)),
        (CaseObservation("text", ("a",), ("a",), 1, ("text_span",)),),
        top_k=1,
        provenance=EvaluationProvenance(
            "cfg", "corpus", {"retriever": "cmp"}, {"rag-kit": "0.1.0"}
        ),
        thresholds=(MetricThreshold("recall_at_k", 1.0, "text"),),
        required_families=("text",),
    )
    value = json.loads(report.to_json())

    with pytest.raises(ValueError):
        Report.from_json(json.dumps(mutation(value)))


@pytest.mark.unit
def test_unobserved_extraction_is_explicit_and_makes_case_incomplete() -> None:
    case = EvaluationCase("text", "text", "search", "a.txt", "q", ("a",), 1, ("text_span",))
    report = evaluate(
        Dataset("ragkit-evaluation-dataset/v1", "extraction-unobserved", (case,)),
        (
            CaseObservation(
                "text",
                ("a",),
                ("a",),
                None,
                ("text_span",),
                extraction_ineligible_reason="extractor output was not instrumented",
            ),
        ),
        top_k=1,
        provenance=EvaluationProvenance("cfg", "corpus", {"evaluator": "v1"}, {"rag-kit": "0.1.0"}),
        required_families=("text",),
    )

    extraction = report.case_results[0].metrics["extraction_coverage"]
    assert extraction.value is None
    assert extraction.ineligible_reason == "extractor output was not instrumented"
    assert report.case_results[0].status == "incomplete"
    assert report.overall_status == "incomplete"
