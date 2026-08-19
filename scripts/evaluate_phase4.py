#!/usr/bin/env python3
"""Run the Phase 4 retrieval evidence suite without network or model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from ragkit.application import AnsweringRequest, IndexingRequest
from ragkit.domain import locator_to_dict
from ragkit.evaluation import (
    CaseObservation,
    Dataset,
    EvaluationProvenance,
    MetricThreshold,
    evaluate,
)
from ragkit.evaluation.benchmark import content_fingerprint
from ragkit.infrastructure import bootstrap, inspect_profile, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--gold-selectors", required=True, type=Path)
    parser.add_argument("--profile", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--family-matrix", type=Path)
    return parser


def _content_fingerprint(path: Path) -> str:
    return content_fingerprint(path.resolve())


def _software() -> dict[str, str]:
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain"), check=True, capture_output=True, text=True
    ).stdout
    return {
        "python": platform.python_version(),
        "rag-kit": version("rag-kit"),
        "git": f"{revision}+{'dirty' if dirty else 'clean'}",
    }


def _evidence_label(actual: object, gold_ids: set[str], label: str) -> str:
    value = str(actual)
    return label if value in gold_ids else f"actual:{value}"


def _percentile(values: tuple[int, ...], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _query_latency(answers: list[object]) -> dict[str, object]:
    samples = tuple(
        sum(item.duration_ns for item in cast(Any, answer).timings) for answer in answers
    )
    return {
        "sample_count": len(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "gated": False,
    }


def _distribution_name(module: str) -> str:
    return {
        "PIL": "pillow",
        "pptx": "python-pptx",
        "faster_whisper": "faster-whisper",
        "cv2": "opencv-python",
    }.get(module, module)


def _sanitized_requirements(
    requirements: list[dict[str, object]],
) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for item in requirements:
        credential = item.get("credential")
        credential_present = None if credential == "not-required" else credential == "configured"
        module = cast(str, item["module"])
        sanitized.append(
            {
                "extra": item["extra"],
                "module": module,
                "distribution": _distribution_name(module),
                "version": item.get("version"),
                "binary": item.get("binary"),
                "model": item.get("model"),
                "model_cached": item.get("model_cached"),
                "credential_present": credential_present,
            }
        )
    return sanitized


def _index_and_ask(profile_path: Path, queries: tuple[str, ...]) -> tuple[object, ...]:
    profile = load_config(profile_path)
    runtime = bootstrap(profile)
    manifest = runtime.manifest_for(profile.source)
    limits = profile.limits
    indexed = runtime.pipeline.index(
        IndexingRequest(
            source_uri=profile.source,
            manifest=manifest,
            max_assets=limits.max_assets,
            max_bytes_per_asset=limits.max_bytes_per_asset,
            max_documents=limits.max_documents,
            max_parts_per_document=limits.max_parts_per_document,
            max_chunks=limits.max_chunks,
            chunking_policy=runtime.chunking_policy,
            indexing_policy=runtime.indexing_policy,
        )
    )
    answers = tuple(
        runtime.pipeline.ask(
            AnsweringRequest(
                query=query,
                expected_manifest=manifest,
                retrieval_top_k=limits.top_k,
                rerank_top_k=limits.top_k,
                max_context_chars=limits.max_context_chars,
                max_output_tokens=limits.max_output_tokens,
            )
        )
        for query in queries
    )
    return profile, indexed, *answers


def _load_gold_selectors(path: Path, dataset: Dataset) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ragkit-independent-gold-selectors/v1":
        raise ValueError("unsupported gold selector schema")
    raw = payload.get("cases")
    if not isinstance(raw, list):
        raise ValueError("gold selector cases must be a list")
    selectors = {cast(str, row["case_id"]): cast(dict[str, Any], row) for row in raw}
    if set(selectors) != {case.case_id for case in dataset.cases}:
        raise ValueError("gold selectors must align exactly with dataset cases")
    for case in dataset.cases:
        selector = selectors[case.case_id]
        if tuple(case.relevant_evidence_ids) != (selector["relevant_evidence_id"],):
            raise ValueError(f"gold selector ID mismatch: {case.case_id}")
        source = Path(cast(str, selector["source_uri"]))
        if hashlib.sha256(source.read_bytes()).hexdigest() != selector["source_sha256"]:
            raise ValueError(f"gold source digest does not match fixture: {case.case_id}")
    return selectors


def _run_text_profile(
    dataset: Dataset,
    selectors: dict[str, dict[str, Any]],
    profile_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    execution = _index_and_ask(profile_path, tuple(case.query for case in dataset.cases))
    profile, indexed, *answers = execution
    profile = cast(Any, profile)
    indexed = cast(Any, indexed)
    observations = []
    evidence_rows: list[dict[str, object]] = []
    for case, answer in zip(dataset.cases, answers, strict=True):
        answer = cast(Any, answer)
        selector = selectors[case.case_id]
        gold_indexed = [
            item
            for item in indexed.indexed_evidence
            if any(
                provenance.asset.sha256 == selector["source_sha256"]
                and locator_to_dict(provenance.locator) == selector["locator"]
                for provenance in item.provenance
            )
        ]
        actual_gold_ids = {str(item.chunk_id) for item in gold_indexed}
        relevant_label = cast(str, selector["relevant_evidence_id"])

        retrieved_ids = tuple(
            _evidence_label(item.chunk.chunk_id, actual_gold_ids, relevant_label)
            for item in answer.context
        )
        cited_ids = tuple(
            _evidence_label(item.chunk_id, actual_gold_ids, relevant_label)
            for item in answer.citations
        )
        relevant_chunks = [
            item for item in answer.context if str(item.chunk.chunk_id) in actual_gold_ids
        ]
        observations.append(
            CaseObservation(
                case_id=case.case_id,
                retrieved_evidence_ids=retrieved_ids,
                cited_evidence_ids=cited_ids,
                extracted_count=len(gold_indexed),
                locator_kinds=tuple(
                    provenance.locator.kind
                    for item in relevant_chunks
                    for provenance in item.chunk.provenance
                ),
            )
        )
        evidence_rows.append(
            {
                "case_id": case.case_id,
                "source_selector": selector,
                "retrieved_evidence_ids": list(retrieved_ids),
                "actual_retrieved_chunk_ids": [str(item.chunk.chunk_id) for item in answer.context],
                "cited_evidence_ids": list(cited_ids),
                "actual_indexed_gold_chunk_ids": sorted(actual_gold_ids),
                "gold_chunk_locators": [
                    {
                        "chunk_id": str(item.chunk.chunk_id),
                        "locators": [
                            locator_to_dict(provenance.locator)
                            for provenance in item.chunk.provenance
                        ],
                    }
                    for item in relevant_chunks
                ],
            }
        )
    provenance = EvaluationProvenance(
        config_fingerprint=str(profile.fingerprint),
        corpus_fingerprint=_content_fingerprint(Path(profile.source)),
        components=cast(dict[str, str], inspect_profile(profile)["selection_fingerprints"]),
        software=_software(),
    )
    thresholds = (
        MetricThreshold("recall_at_k", 1.0, "text"),
        MetricThreshold("hit_rate", 1.0, "text"),
        MetricThreshold("citation_precision", 1.0, "text"),
        MetricThreshold("citation_coverage", 1.0, "text"),
        MetricThreshold("extraction_coverage", 1.0, "text"),
        MetricThreshold("locator_validity", 1.0, "text"),
    )
    report = evaluate(
        dataset,
        tuple(observations),
        top_k=profile.limits.top_k,
        provenance=provenance,
        thresholds=thresholds,
        required_families=("text",),
    )
    destination = output_dir / f"{profile.name}-text-evaluation-v1.json"
    destination.write_text(report.to_json() + "\n", encoding="utf-8")
    evidence_destination = output_dir / f"{profile.name}-text-observations-v1.json"
    query_latency = _query_latency(answers)
    evidence_destination.write_text(
        json.dumps(
            {
                "schema_version": "ragkit-evaluation-observations/v1",
                "dataset_fingerprint": dataset.fingerprint,
                "config_fingerprint": str(profile.fingerprint),
                "corpus_fingerprint": _content_fingerprint(Path(profile.source)),
                "component_selections": asdict(profile.components),
                "query_latency_ns": query_latency,
                "cases": evidence_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "profile": profile.name,
        "status": report.overall_status,
        "report": destination.name,
        "observations": evidence_destination.name,
        "query_latency_ns": query_latency,
    }


def _unmet_requirement(requirements: list[dict[str, object]]) -> str | None:
    missing = [str(item["module"]) for item in requirements if not item["installed"]]
    uncached = [
        str(item["model"])
        for item in requirements
        if item.get("model") and item.get("model_cached") is False
    ]
    if missing:
        return "optional modules unavailable: " + ", ".join(missing)
    if uncached:
        return "revision-pinned local model unavailable: " + ", ".join(uncached)
    selected_models = [str(item["model"]) for item in requirements if item.get("model")]
    if selected_models and os.environ.get("RAGKIT_RUN_MODEL_INTEGRATION") != "1":
        return "local model execution is opt-in: set RAGKIT_RUN_MODEL_INTEGRATION=1"
    return None


def _unavailable_family_metrics(reason: str) -> dict[str, object]:
    return {
        "retrieval_recall": {"value": None, "reason": reason},
        "citation_coverage": {"value": None, "reason": reason},
        "extraction_coverage": {"value": None, "reason": reason},
        "locator_validity": {"value": None, "reason": reason},
    }


def _family_specific(family: str, gold_chunks: list[dict[str, Any]]) -> dict[str, object]:
    provenance = [item for chunk in gold_chunks for item in chunk["provenance"]]
    if family == "ocr":
        confidence = [item["confidence"] for item in provenance]
        return {"ocr_confidence": confidence or None}
    linkage_name = {
        "text": "source_span_linkage",
        "layout": "structure_cell_linkage",
        "vision": "image_region_linkage",
        "media": "timestamp_linkage",
    }[family]
    return {linkage_name: bool(provenance) if gold_chunks else None}


def _execute_family(profile_path: Path, query: str) -> dict[str, Any]:
    completed = subprocess.run(
        (
            sys.executable,
            "scripts/evaluate_phase4_family.py",
            "--profile",
            str(profile_path),
            "--query",
            query,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(f"isolated family process exited {completed.returncode}")
    return cast(dict[str, Any], json.loads(completed.stdout))


def _family_matrix(path: Path, output_dir: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ragkit-family-execution-matrix/v1":
        raise ValueError("unsupported family matrix schema")
    rows = payload.get("families")
    if not isinstance(rows, dict) or set(rows) != {"text", "ocr", "layout", "vision", "media"}:
        raise ValueError("family matrix must define text, ocr, layout, vision, and media")
    results: dict[str, object] = {}
    for family in ("text", "ocr", "layout", "vision", "media"):
        row = cast(dict[str, object], rows[family])
        profile_path = Path(cast(str, row["profile"]))
        profile = load_config(profile_path)
        requirements = cast(list[dict[str, object]], inspect_profile(profile)["requirements"])
        unmet = _unmet_requirement(requirements)
        base: dict[str, object] = {
            "profile": profile.name,
            "config_fingerprint": str(profile.fingerprint),
            "corpus_fingerprint": _content_fingerprint(Path(profile.source)),
            "components": cast(dict[str, str], inspect_profile(profile)["selection_fingerprints"]),
            "component_selections": asdict(profile.components),
            "requirements": _sanitized_requirements(requirements),
            "software": _software(),
        }
        if row.get("execution_policy") == "preflight_only":
            reason = cast(str, row["ineligible_reason"])
            results[family] = {
                **base,
                "status": "ineligible",
                "evidence": reason,
                "retrieved_evidence_ids": [],
                "retrieved_locators": [],
                "metrics": _unavailable_family_metrics(reason),
                "family_specific": _family_specific(family, []),
            }
            continue
        if unmet is not None:
            results[family] = {
                **base,
                "status": "ineligible",
                "evidence": unmet,
                "retrieved_evidence_ids": [],
                "retrieved_locators": [],
                "metrics": _unavailable_family_metrics(unmet),
                "family_specific": _family_specific(family, []),
            }
            continue
        try:
            execution = _execute_family(profile_path, cast(str, row["query"]))
            label = cast(str | None, row["relevant_evidence_id"])
            locator = cast(dict[str, object] | None, row["locator"])
            gold_indexed = (
                [
                    item
                    for item in execution["indexed_evidence"]
                    if any(
                        provenance["asset_sha256"] == row["source_sha256"]
                        and provenance["locator"] == locator
                        for provenance in item["provenance"]
                    )
                ]
                if label is not None and locator is not None
                else []
            )
            gold_ids = {cast(str, item["chunk_id"]) for item in gold_indexed}
            gold_chunks = [item for item in execution["context"] if item["chunk_id"] in gold_ids]
            retrieved = tuple(
                _evidence_label(item["chunk_id"], gold_ids, label or "ineligible")
                for item in execution["context"]
            )
            cited_ids = set(cast(list[str], execution["citations"]))
            extracted = bool(gold_indexed)
            retrieved_gold = bool(gold_chunks)
            cited_gold = bool(gold_ids & cited_ids)
            locator_valid = retrieved_gold and any(
                provenance["locator"] == locator
                for item in gold_chunks
                for provenance in item["provenance"]
            )
            if label is None or locator is None:
                status = "ineligible"
                evidence = "pipeline ran; no independently fixed semantic gold selector"
                metrics = _unavailable_family_metrics(evidence)
            elif not extracted:
                status = "fail"
                evidence = "indexing ran but no evidence matched the fixed source selector"
                metrics = {
                    "retrieval_recall": {"value": 0.0, "reason": None},
                    "citation_coverage": {"value": 0.0, "reason": None},
                    "extraction_coverage": {"value": 0.0, "reason": None},
                    "locator_validity": {"value": None, "reason": "gold chunk absent"},
                }
            else:
                values = (
                    float(retrieved_gold),
                    float(cited_gold),
                    1.0,
                    float(locator_valid),
                )
                status = "pass" if all(value == 1.0 for value in values) else "fail"
                evidence = "fixed source digest and locator matched indexed evidence"
                metrics = {
                    "retrieval_recall": {"value": values[0], "reason": None},
                    "citation_coverage": {"value": values[1], "reason": None},
                    "extraction_coverage": {"value": values[2], "reason": None},
                    "locator_validity": {"value": values[3], "reason": None},
                }
            results[family] = {
                **base,
                "status": status,
                "evidence": evidence,
                "retrieved_evidence_ids": list(retrieved),
                "retrieved_locators": [
                    {
                        "chunk_id": item["chunk_id"],
                        "locators": [value["locator"] for value in item["provenance"]],
                    }
                    for item in execution["context"]
                ],
                "metrics": metrics,
                "family_specific": _family_specific(family, gold_chunks),
            }
        except Exception as error:  # Boundary report: failures must remain visible.
            results[family] = {
                **base,
                "status": "fail",
                "evidence": f"pipeline execution failed: {type(error).__name__}",
                "retrieved_evidence_ids": [],
                "retrieved_locators": [],
                "metrics": _unavailable_family_metrics(
                    f"pipeline execution failed: {type(error).__name__}"
                ),
                "family_specific": _family_specific(family, []),
            }
    report = {
        "schema_version": "ragkit-family-execution-report/v1",
        "matrix_fingerprint": _content_fingerprint(path),
        "families": results,
    }
    (output_dir / "five-family-execution-v1.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return not any(
        cast(dict[str, object], result)["status"] == "fail" for result in results.values()
    )


def main() -> int:
    args = _parser().parse_args()
    dataset_path = cast(Path, args.dataset)
    dataset = Dataset.load(dataset_path)
    if {case.family for case in dataset.cases} != {"text"}:
        raise ValueError("the retrieval runner accepts a text-only gold dataset")
    output_dir = cast(Path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selectors = _load_gold_selectors(cast(Path, args.gold_selectors), dataset)
    summaries = [
        _run_text_profile(dataset, selectors, path, output_dir)
        for path in cast(list[Path], args.profile)
    ]
    family_matrix = cast(Path | None, args.family_matrix)
    matrix_has_no_failures = True
    if family_matrix is not None:
        matrix_has_no_failures = _family_matrix(family_matrix, output_dir)
    summary = {
        "schema_version": "ragkit-phase4-evaluation-run/v1",
        "dataset_fingerprint": dataset.fingerprint,
        "profiles": summaries,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return (
        0 if all(item["status"] == "pass" for item in summaries) and matrix_has_no_failures else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
