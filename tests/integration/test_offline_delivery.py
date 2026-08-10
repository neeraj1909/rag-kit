from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkit.cli.main import main
from ragkit.domain import InvalidDomainValueError, UnsupportedCapabilityError
from ragkit.infrastructure.bootstrap import bootstrap
from ragkit.infrastructure.config import load_config

FIXTURES = Path(__file__).parents[1] / "fixtures"
CONFIG = Path(__file__).parents[2] / "configs" / "offline.toml"


@pytest.mark.integration
def test_equivalent_toml_profiles_have_the_same_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    second.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            '[profile]\nname = "offline"\nfamily = "text"',
            '[profile]\nfamily = "text"\nname = "offline"',
        ),
        encoding="utf-8",
    )
    changed = tmp_path / "changed.toml"
    changed.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "embedding_dimension = 128", "embedding_dimension = 64"
        ),
        encoding="utf-8",
    )

    assert load_config(first).fingerprint == load_config(second).fingerprint
    assert load_config(first).fingerprint != load_config(changed).fingerprint


@pytest.mark.integration
def test_equivalent_filesystem_source_paths_share_one_manifest() -> None:
    runtime = bootstrap(load_config(CONFIG))
    relative = Path("tests/fixtures/corpus")
    absolute = relative.resolve()

    assert runtime.manifest_for(str(relative)) == runtime.manifest_for(str(absolute))
    assert runtime.manifest_for(absolute.as_uri()) == runtime.manifest_for(str(absolute))


@pytest.mark.integration
def test_config_rejects_unknown_fields_and_non_positive_limits(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            'name = "offline"', 'name = "offline"\nsecret_token = "must-not-be-accepted"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(InvalidDomainValueError, match="profile fields"):
        load_config(unknown)

    invalid_limit = tmp_path / "invalid-limit.toml"
    invalid_limit.write_text(
        CONFIG.read_text(encoding="utf-8").replace("top_k = 3", "top_k = 0"),
        encoding="utf-8",
    )
    with pytest.raises(InvalidDomainValueError, match="positive"):
        load_config(invalid_limit)

    wrong_type = tmp_path / "wrong-type.toml"
    wrong_type.write_text(
        CONFIG.read_text(encoding="utf-8").replace('name = "offline"', "name = 7"),
        encoding="utf-8",
    )
    with pytest.raises(InvalidDomainValueError, match="profile"):
        load_config(wrong_type)


@pytest.mark.integration
def test_profile_schema_names_later_families_but_bootstrap_rejects_them(
    tmp_path: Path,
) -> None:
    for family in ("ocr", "layout", "vision", "media"):
        profile = tmp_path / f"{family}.toml"
        profile.write_text(
            CONFIG.read_text(encoding="utf-8").replace('family = "text"', f'family = "{family}"'),
            encoding="utf-8",
        )
        parsed = load_config(profile)
        assert parsed.family.value == family
        with pytest.raises(UnsupportedCapabilityError, match=family):
            bootstrap(parsed)

    unavailable = tmp_path / "hosted-embedder.toml"
    unavailable.write_text(
        CONFIG.read_text(encoding="utf-8").replace('embedder = "hashing"', 'embedder = "hosted"'),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedCapabilityError, match=r"embedder.*hosted"):
        bootstrap(load_config(unavailable))


@pytest.mark.integration
def test_inspect_index_ask_evaluate_cli_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect-config", "--config", str(CONFIG)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["profile"] == "offline"
    assert inspected["family"] == "text"
    assert inspected["config_fingerprint"].startswith("cmp_v1_")
    assert inspected["components"]["embedder"] == "hashing"

    assert main(["index", "--config", str(CONFIG), "--source", str(FIXTURES / "corpus")]) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["documents"] == 5
    assert indexed["chunks"] >= 5
    assert indexed["config_fingerprint"] == inspected["config_fingerprint"]
    assert indexed["storage"] == "process_local"
    assert set(indexed["timings_ms"]) >= {"fetch", "extract", "chunk", "embed", "upsert"}

    assert (
        main(
            [
                "ask",
                "--config",
                str(CONFIG),
                "What is the fixture answer?",
            ]
        )
        == 0
    )
    answered = json.loads(capsys.readouterr().out)
    assert "cobalt observatory" in answered["answer"].lower()
    assert answered["citations"]
    citation = answered["citations"][0]
    assert citation["rank"] == 1
    evidence = citation["evidence"][0]
    assert evidence["source_uri"].endswith("answer.txt")
    assert evidence["locator"] == {"end": 80, "kind": "text_span", "start": 0}
    assert answered["config_fingerprint"] == inspected["config_fingerprint"]
    assert answered["index_mode"] == "rebuilt_in_process"
    assert set(answered["timings_ms"]) >= {"index_total", "retrieve", "generate"}

    assert (
        main(
            [
                "evaluate",
                "--config",
                str(CONFIG),
                "--dataset",
                str(FIXTURES / "eval.jsonl"),
            ]
        )
        == 0
    )
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["evaluated_cases"] == 1
    assert evaluated["metrics"]["answer_contains_expected"] == 1.0
    assert evaluated["config_fingerprint"] == inspected["config_fingerprint"]
    assert evaluated["timings_ms"]["evaluate"] >= 0
