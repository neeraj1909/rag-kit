from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ragkit.cli.main import main
from ragkit.domain import MissingDependencyError, UnsupportedCapabilityError
from ragkit.infrastructure import bootstrap, inspect_profile, load_config

pytestmark = pytest.mark.integration

PROFILES = tuple(sorted(Path("configs").glob("*.toml")))


@pytest.mark.parametrize("path", PROFILES, ids=lambda path: path.stem)
def test_every_example_profile_has_secret_free_capability_diagnostics(path: Path) -> None:
    profile = load_config(path)

    diagnostics = inspect_profile(profile)
    rendered = json.dumps(diagnostics, sort_keys=True)

    assert diagnostics["selected_family"] == profile.family.value
    assert diagnostics["supported_families"] == ["text", "ocr", "layout", "vision", "media"]
    fingerprints = cast(Mapping[str, object], diagnostics["selection_fingerprints"])
    assert set(fingerprints) == set(profile.components.__dataclass_fields__)
    assert "sk-" not in rendered
    assert "raw content" not in rendered


@pytest.mark.parametrize("name", ["offline", "ocr", "layout", "media"])
def test_local_profiles_compose_without_hidden_fallback(name: str) -> None:
    profile = load_config(Path("configs") / f"{name}.toml")

    runtime = bootstrap(profile)

    assert runtime.embedding_dimension == profile.limits.embedding_dimension


def test_hosted_profile_requires_named_credential_without_disclosing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(UnsupportedCapabilityError, match="OPENAI_API_KEY") as caught:
        bootstrap(load_config("configs/hosted.toml"))

    assert "sk-" not in str(caught.value)


@pytest.mark.parametrize("name", ["vision", "torch-local"])
def test_model_profiles_fail_with_provisioning_action_when_cache_is_absent(name: str) -> None:
    profile = load_config(Path("configs") / f"{name}.toml")
    try:
        runtime = bootstrap(profile)
    except MissingDependencyError as error:
        assert "provision" in str(error).casefold() or "cached" in str(error).casefold()
        return
    assert runtime.embedding_dimension > 0


def test_inspect_config_emits_phase3_matrix_without_loading_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["inspect-config", "--config", "configs/vision.toml"]) == 0
    result = json.loads(capsys.readouterr().out)

    capabilities = result["capabilities"]
    assert capabilities["selected_family"] == "vision"
    assert capabilities["device"] == "cpu"
    assert capabilities["degraded_modes"] == ["model_descriptions_uncalibrated"]
    assert capabilities["requirements"][0]["extra"] == "vision"
    assert capabilities["limits"]["adapter"]["vision_max_pixels"] == 4_194_304


def test_xlsm_diagnostics_select_openpyxl() -> None:
    profile = replace(load_config("configs/layout.toml"), source="pricing.xlsm")

    requirements = cast(list[dict[str, object]], inspect_profile(profile)["requirements"])

    assert requirements[0]["module"] == "openpyxl"


def test_mixed_profile_wires_configured_ocr_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap_module = importlib.import_module("ragkit.infrastructure.bootstrap")
    original = bootstrap_module.OcrDocumentExtractor
    calls: list[dict[str, object]] = []

    def build_ocr(**kwargs: object) -> object:
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(bootstrap_module, "OcrDocumentExtractor", build_ocr)
    profile = load_config("configs/mixed-image.toml")
    profile = replace(
        profile,
        settings=replace(
            profile.settings,
            ocr_max_pages=3,
            ocr_max_pixels=123_456,
            ocr_timeout_seconds=7.0,
        ),
    )

    bootstrap(profile)

    assert calls == [
        {
            "language": "eng",
            "max_pages": 3,
            "max_pixels": 123_456,
            "timeout_seconds": 7.0,
        }
    ]


def test_cli_reports_persistent_store_truthfully(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("chromadb")
    source = Path("configs/persistent.toml").read_text()
    config = tmp_path / "persistent.toml"
    config.write_text(source.replace(".ragkit/chroma", str(tmp_path / "chroma")))

    profile = load_config(config)
    requirement = cast(list[dict[str, object]], inspect_profile(profile)["requirements"])[0]
    assert requirement["version"] is not None

    assert main(["index", "--config", str(config)]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["storage"] == "persistent"
