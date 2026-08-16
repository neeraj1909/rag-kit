from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from ragkit.domain import InvalidDomainValueError
from ragkit.infrastructure.config import AdapterSettings
from ragkit.infrastructure.optional import (
    OptionalCapability,
    _credential_is_present,
    inspect_optional_capability,
)

pytestmark = pytest.mark.unit


def test_inspection_reports_missing_extra_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: imported.append(name))

    result = inspect_optional_capability(
        OptionalCapability("example-extra", "missing_example_sdk", credential_env=None)
    )

    assert result.installed is False
    assert result.action == "install rag-kit[example-extra]"
    assert result.credential == "not-required"
    assert imported == []


def test_inspection_reports_credential_presence_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    result = inspect_optional_capability(
        OptionalCapability("hosted", "openai", credential_env="OPENAI_API_KEY")
    )

    assert result.installed is True
    assert result.credential == "configured"
    assert "sk-secret-value" not in repr(result)


def test_inspection_checks_credential_presence_without_reading_value() -> None:
    class PresenceOnlyEnvironment(Mapping[str, str]):
        def __contains__(self, key: object) -> bool:
            return key == "OPENAI_API_KEY"

        def __getitem__(self, key: str) -> str:
            raise AssertionError("credential value access is forbidden during inspection")

        def __iter__(self) -> Iterator[str]:
            return iter(("OPENAI_API_KEY",))

        def __len__(self) -> int:
            return 1

    environment = PresenceOnlyEnvironment()

    assert _credential_is_present("OPENAI_API_KEY", environment)
    assert not _credential_is_present("MISSING", environment)


def test_inspection_reports_binary_and_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = inspect_optional_capability(
        OptionalCapability("ocr", "pytesseract", binary="tesseract", model="eng")
    )

    assert result.binary == "missing:tesseract"
    assert result.model == "eng"
    assert result.model_cached is None


def test_inspection_reports_exact_model_provisioning_without_importing_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    result = inspect_optional_capability(
        OptionalCapability(
            "vision",
            "transformers",
            model="owner/model@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        )
    )

    assert result.model_cached is False
    assert result.action == (
        "hf download owner/model --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )


def test_adapter_settings_reject_secret_like_credential_environment_name() -> None:
    secret = "sk-secret-value"

    with pytest.raises(InvalidDomainValueError) as caught:
        AdapterSettings(credential_env=secret)

    assert secret not in str(caught.value)
