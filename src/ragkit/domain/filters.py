"""Provider-neutral, serializable metadata-filter expression tree."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeAlias

from .errors import InvalidDomainValueError

Scalar: TypeAlias = str | int | float | bool | None


class ComparisonOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"


class MetadataFilter(ABC):
    """Closed filter AST; adapters reject nodes they cannot represent exactly."""

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Return a lossless provider-neutral representation."""

    @staticmethod
    def from_dict(value: dict[str, object]) -> MetadataFilter:
        kind = value.get("kind")
        if kind == "comparison":
            raw_operator = value.get("operator")
            raw_field = value.get("field")
            if not isinstance(raw_operator, str) or not isinstance(raw_field, str):
                raise InvalidDomainValueError("invalid comparison encoding")
            operator = ComparisonOperator(raw_operator)
            raw_value = value.get("value")
            comparison_value = (
                tuple(raw_value)
                if operator is ComparisonOperator.IN and isinstance(raw_value, list)
                else raw_value
            )
            return Comparison(raw_field, operator, comparison_value)
        if kind in {"and", "or"}:
            raw_children = value.get("children")
            if not isinstance(raw_children, list) or not all(
                isinstance(item, dict) for item in raw_children
            ):
                raise InvalidDomainValueError("invalid composite filter encoding")
            children = tuple(MetadataFilter.from_dict(item) for item in raw_children)
            return And(children) if kind == "and" else Or(children)
        if kind == "not":
            child = value.get("child")
            if not isinstance(child, dict):
                raise InvalidDomainValueError("invalid not-filter encoding")
            return Not(MetadataFilter.from_dict(child))
        raise InvalidDomainValueError("unknown metadata filter kind")


@dataclass(frozen=True, slots=True)
class Comparison(MetadataFilter):
    field: str
    operator: ComparisonOperator
    value: object

    def __post_init__(self) -> None:
        if not self.field:
            raise InvalidDomainValueError("comparison field must not be empty")
        if self.operator is ComparisonOperator.IN:
            if not isinstance(self.value, tuple) or not self.value:
                raise InvalidDomainValueError("in comparison requires a non-empty tuple")
            values = self.value
        else:
            values = (self.value,)
        if any(not isinstance(item, (str, int, float, bool, type(None))) for item in values):
            raise InvalidDomainValueError("metadata values must be scalar")
        if any(isinstance(item, float) and not isfinite(item) for item in values):
            raise InvalidDomainValueError("metadata numbers must be finite")

    def to_dict(self) -> dict[str, object]:
        value: object = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "kind": "comparison",
            "field": self.field,
            "operator": self.operator.value,
            "value": value,
        }


@dataclass(frozen=True, slots=True)
class And(MetadataFilter):
    children: tuple[MetadataFilter, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise InvalidDomainValueError("and filter requires children")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "and", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True, slots=True)
class Or(MetadataFilter):
    children: tuple[MetadataFilter, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise InvalidDomainValueError("or filter requires children")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "or", "children": [child.to_dict() for child in self.children]}


@dataclass(frozen=True, slots=True)
class Not(MetadataFilter):
    child: MetadataFilter

    def to_dict(self) -> dict[str, object]:
        return {"kind": "not", "child": self.child.to_dict()}
