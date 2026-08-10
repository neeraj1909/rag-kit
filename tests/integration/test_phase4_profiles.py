from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ragkit.cli.main import main
from ragkit.domain import InvalidDomainValueError, UnsupportedCapabilityError
from ragkit.infrastructure import bootstrap, inspect_profile, load_config

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", ["sparse", "hybrid"])
def test_retrieval_profiles_answer_through_the_selected_strategy(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config = f"configs/{name}.toml"

    assert main(["ask", "--config", config, "What is the fixture answer?"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "cobalt observatory" in payload["answer"].casefold()
    assert payload["citations"]
    assert "retrieve" in payload["timings_ms"]
    assert "search" not in payload["timings_ms"]


def test_phase4_selection_and_model_requirements_are_explicit() -> None:
    hybrid = load_config("configs/hybrid.toml")
    reranked = load_config("configs/reranked.toml")

    assert hybrid.components.retriever == "hybrid"
    assert hybrid.settings.hybrid_candidate_multiplier == 4
    requirements = cast(list[dict[str, object]], inspect_profile(reranked)["requirements"])
    limits = cast(dict[str, object], inspect_profile(hybrid)["limits"])
    adapter_limits = cast(dict[str, object], limits["adapter"])
    assert adapter_limits["hybrid_candidate_multiplier"] == 4
    reranking = [item for item in requirements if item["extra"] == "reranking"]
    assert [item["module"] for item in reranking] == ["torch", "transformers"]
    assert reranking[1]["model"] == (
        "cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a"
    )


def test_unknown_retriever_selection_fails_without_fallback() -> None:
    profile = load_config("configs/offline.toml")
    invalid = replace(
        profile,
        components=replace(profile.components, retriever="not-implemented"),
    )

    with pytest.raises(UnsupportedCapabilityError, match=r"retriever.*not-implemented"):
        bootstrap(invalid)


def test_phase4_settings_reject_invalid_scoring_and_reranker_bounds() -> None:
    settings = load_config("configs/hybrid.toml").settings

    with pytest.raises(InvalidDomainValueError, match="bm25_b"):
        replace(settings, bm25_b=1.1)
    with pytest.raises(InvalidDomainValueError, match="valid regex"):
        replace(settings, bm25_token_pattern="[")
    with pytest.raises(InvalidDomainValueError, match="supported bounds"):
        replace(settings, reranker_batch_size=129)


def test_toml_rejects_fractional_candidate_multiplier(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-hybrid.toml"
    invalid.write_text(
        Path("configs/hybrid.toml")
        .read_text(encoding="utf-8")
        .replace("hybrid_candidate_multiplier = 4", "hybrid_candidate_multiplier = 1.5"),
        encoding="utf-8",
    )

    with pytest.raises(InvalidDomainValueError, match="must be integers"):
        load_config(invalid)
