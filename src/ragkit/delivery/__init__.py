"""Dependency-light delivery adapters."""

from .http import HttpApp, create_app

__all__ = ["HttpApp", "create_app"]
