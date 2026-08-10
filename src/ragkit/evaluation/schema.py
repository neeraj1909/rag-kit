"""Versioned, dependency-free evaluation dataset schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

Family = Literal["text", "ocr", "layout", "vision", "media"]
Eligibility = Literal["eligible", "ineligible"]
SCHEMA_VERSION = "ragkit-evaluation-dataset/v1"


def _require(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One fixed business question and its evidence/locator gold labels."""

    case_id: str
    family: Family
    business_use_case: str
    source_uri: str
    query: str
    relevant_evidence_ids: tuple[str, ...]
    expected_extractions: int
    required_locator_kinds: tuple[str, ...]
    eligibility: Eligibility = "eligible"
    ineligible_reason: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (
                self.case_id,
                self.family,
                self.business_use_case,
                self.source_uri,
                self.query,
            )
        ):
            raise ValueError("evaluation case text fields must be strings")
        for value, label in (
            (self.case_id, "case ID"),
            (self.business_use_case, "business use case"),
            (self.source_uri, "source URI"),
            (self.query, "query"),
        ):
            _require(value, label)
        if self.family not in ("text", "ocr", "layout", "vision", "media"):
            raise ValueError(f"unsupported document family: {self.family}")
        if not self.relevant_evidence_ids or not all(
            isinstance(item, str) and item for item in self.relevant_evidence_ids
        ):
            raise ValueError("relevant evidence IDs must not be empty")
        if len(set(self.relevant_evidence_ids)) != len(self.relevant_evidence_ids):
            raise ValueError("relevant evidence IDs must be unique")
        if isinstance(self.expected_extractions, bool) or not isinstance(
            self.expected_extractions, int
        ):
            raise ValueError("expected extractions must be an integer")
        if self.expected_extractions <= 0:
            raise ValueError("expected extractions must be positive")
        if not self.required_locator_kinds or not all(
            isinstance(item, str) and item for item in self.required_locator_kinds
        ):
            raise ValueError("required locator kinds must not be empty")
        if self.eligibility not in ("eligible", "ineligible"):
            raise ValueError("eligibility must be eligible or ineligible")
        if self.ineligible_reason is not None and not isinstance(self.ineligible_reason, str):
            raise ValueError("ineligible reason must be a string or null")
        if self.eligibility == "ineligible" and not self.ineligible_reason:
            raise ValueError("ineligible cases require a reason")
        if self.eligibility == "eligible" and self.ineligible_reason is not None:
            raise ValueError("eligible cases cannot have an ineligible reason")


@dataclass(frozen=True, slots=True)
class Dataset:
    """A stable collection of evaluation cases with canonical fingerprinting."""

    schema_version: str
    dataset_id: str
    cases: tuple[EvaluationCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not isinstance(self.dataset_id, str):
            raise ValueError("dataset schema version and ID must be strings")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported evaluation schema: {self.schema_version}")
        _require(self.dataset_id, "dataset ID")
        if not self.cases:
            raise ValueError("evaluation dataset requires cases")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")

    @property
    def fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> Dataset:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "dataset_id",
            "cases",
        }:
            raise ValueError("evaluation dataset has unexpected fields")
        rows = value["cases"]
        if not isinstance(rows, list):
            raise ValueError("evaluation cases must be a list")
        cases: list[EvaluationCase] = []
        expected = {
            "case_id",
            "family",
            "business_use_case",
            "source_uri",
            "query",
            "relevant_evidence_ids",
            "expected_extractions",
            "required_locator_kinds",
            "eligibility",
            "ineligible_reason",
        }
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError("evaluation case has unexpected fields")
            cases.append(
                EvaluationCase(
                    case_id=cast(str, row["case_id"]),
                    family=cast(Family, row["family"]),
                    business_use_case=cast(str, row["business_use_case"]),
                    source_uri=cast(str, row["source_uri"]),
                    query=cast(str, row["query"]),
                    relevant_evidence_ids=tuple(cast(list[str], row["relevant_evidence_ids"])),
                    expected_extractions=cast(int, row["expected_extractions"]),
                    required_locator_kinds=tuple(cast(list[str], row["required_locator_kinds"])),
                    eligibility=cast(Eligibility, row["eligibility"]),
                    ineligible_reason=cast(str | None, row["ineligible_reason"]),
                )
            )
        return cls(cast(str, value["schema_version"]), cast(str, value["dataset_id"]), tuple(cases))

    @classmethod
    def load(cls, path: Path) -> Dataset:
        return cls.from_json(path.read_text(encoding="utf-8"))
