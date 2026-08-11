"""Dependency-free inspection of optional runtime capabilities."""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OptionalCapability:
    """Static installation requirements for one optional profile capability."""

    extra: str
    module: str
    credential_env: str | None = None
    binary: str | None = None
    model: str | None = None
    distribution: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityInspection:
    extra: str
    module: str
    installed: bool
    action: str | None
    credential: str
    binary: str
    model: str | None
    model_cached: bool | None
    version: str | None


def _credential_is_present(name: str, environment: Mapping[str, str] | None = None) -> bool:
    """Check membership without retrieving the credential value."""

    return name in (os.environ if environment is None else environment)


def _model_cache_state(model: str | None) -> tuple[bool | None, str | None]:
    if model is None or "@" not in model or "/" not in model:
        return None, None
    model_id, revision = model.rsplit("@", maxsplit=1)
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = (
        cache_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    )
    cached = snapshot.is_dir()
    action = None if cached else f"hf download {model_id} --revision {revision}"
    return cached, action


def inspect_optional_capability(capability: OptionalCapability) -> CapabilityInspection:
    """Inspect availability without importing an optional SDK or reading secret values."""

    installed = importlib.util.find_spec(capability.module) is not None
    try:
        version = (
            metadata.version(capability.distribution or capability.module) if installed else None
        )
    except metadata.PackageNotFoundError:
        version = None
    credential = (
        "not-required"
        if capability.credential_env is None
        else "configured"
        if _credential_is_present(capability.credential_env)
        else f"missing:{capability.credential_env}"
    )
    binary = (
        "not-required"
        if capability.binary is None
        else "available"
        if shutil.which(capability.binary) is not None
        else f"missing:{capability.binary}"
    )
    model_cached, model_action = _model_cache_state(capability.model)
    action = None if installed else f"install rag-kit[{capability.extra}]"
    if action is None and capability.binary is not None and binary.startswith("missing:"):
        action = f"install executable {capability.binary}"
    if action is None:
        action = model_action
    return CapabilityInspection(
        extra=capability.extra,
        module=capability.module,
        installed=installed,
        action=action,
        credential=credential,
        binary=binary,
        model=capability.model,
        model_cached=model_cached,
        version=version,
    )
