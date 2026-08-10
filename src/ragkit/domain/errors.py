"""Typed, provider-neutral domain failures."""

from __future__ import annotations

from collections.abc import Mapping


class RagkitError(Exception):
    """Base class for failures callers may handle without provider knowledge."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class InvalidDomainValueError(RagkitError, ValueError):
    """A value violates a domain invariant."""


class UnsupportedCapabilityError(RagkitError):
    """A selected component cannot provide a requested capability."""

    def __init__(self, message: str, *, capability: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause=cause)
        self.capability = capability


class MissingDependencyError(RagkitError):
    """An optional component dependency is unavailable."""


class ProviderError(RagkitError):
    """An external provider failed after its error was translated."""


class OperationTimeoutError(RagkitError):
    """A bounded adapter operation exceeded its configured wall-clock deadline."""


class IntegrityError(RagkitError):
    """Persisted or acquired data violates an integrity requirement."""


class LimitExceededError(RagkitError):
    """A declared resource or input limit was exceeded."""


class PartialExtractionError(RagkitError):
    """Extraction produced incomplete evidence that policy does not accept."""


class IndexCompatibilityError(RagkitError):
    """Expected and persisted index semantics are incompatible."""

    def __init__(self, differences: Mapping[str, tuple[object, object]]) -> None:
        self.differences = dict(differences)
        fields = ", ".join(sorted(differences))
        super().__init__(f"index manifest mismatch: {fields}")
