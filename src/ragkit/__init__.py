"""Public package surface for rag-kit.

Phase 0 intentionally keeps this module dependency-free. Domain contracts and
pipeline behavior are introduced by later implementation slices.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rag-kit")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
